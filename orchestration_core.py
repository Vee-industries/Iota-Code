"""
IOTA FRAMEWORK — ORCHESTRATION CORE
======================================
O is for orchestration. Shared infrastructure for all cluster scripts.

Provides:
  - set_seed(), GPUMonitor
  - TokenTimer, MinTokenEnforcer, StatusTokenEnforcerFixed
  - cosine_sim(), delta_r()
  - load_model(), unload_model()
  - run_generation()
  - TrialState
  - compute_turn_metrics()
  - append_csv(), save_npy()
  - get_next_trial(), get_next_trial_for_condition()
  - print_turn_result()
  - load_calibration(), run_calibration()
  - get_status_token_ids()
"""

import os
import sys
import re
import json
import time
import threading
import subprocess
import datetime

import torch
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, LogitsProcessorList, LogitsProcessor,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure') and _stream.encoding.lower() not in ('utf-8', 'utf8'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_QUANT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

QUANT_8BIT = BitsAndBytesConfig(
    load_in_8bit=True,
)


def _resolve_quant(quant):
    """Resolve quant string to (BitsAndBytesConfig or None, torch_dtype)."""
    if quant == 'fp16' or quant == 'none':
        return None, torch.float16
    elif quant == 'fp32':
        return None, torch.float32
    elif quant == '8bit':
        return QUANT_8BIT, torch.float16
    else:  # '4bit' or default
        return DEFAULT_QUANT, torch.float16

STATUS_TOKENS   = ["DONE", "WAIT", "STOP"]
_NO_SYSTEM_ROLE = False  # Set True on first TemplateError for system role
_hs_diag_done   = False  # Set True after first hidden state dim diagnostic print
MIN_OUT_TOKENS  = 25
MAX_OUT_TOKENS  = 30
LONG_OUT_TOKENS = 35


def set_seed(seed: int = 42):
    """Seed every RNG IOTA touches: Python random, numpy, torch CPU,
    torch CUDA. Also flips cuDNN deterministic on / benchmark off so
    the same seed produces bit-identical generation across re-runs.
    Called once per trial in the trial loop with seed+trial."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GPUMonitor:
    """Per-turn GPU peak-power / utilisation / temperature sampler.

    A background thread polls pynvml every ``interval`` seconds
    (default 50 ms) while a run is active, tracking the peak of each
    metric across the observation window. start() / stop() bookend
    one turn; stop() returns (peak_util, peak_power, peak_temp).

    Class-level pynvml state (_nvml_handle, _nvml_ready) is
    deliberate: one pynvml session shared across every GPUMonitor
    instance. Per-instance nvmlInit/nvmlShutdown would invalidate
    the shared handle mid-run (the Run 0020 freeze bug) and rack up
    thousands of init calls over a multi-hour sweep.
    """
    # Class-level pynvml init — called once when first GPUMonitor is created.
    # Not per-instance: avoids thousands of nvmlInit() calls without nvmlShutdown()
    # over multi-hour runs (one GPUMonitor per turn × 100 trials × 13 turns × 45 runs).
    _nvml_handle = None
    _nvml_ready  = None  # None=untried, True=ok, False=failed

    @classmethod
    def _ensure_nvml(cls):
        """Idempotent pynvml initialisation. Returns True once nvmlInit
        has succeeded and _nvml_handle holds the device-0 handle;
        False if pynvml is absent or init failed. Subsequent calls are
        no-ops — the class holds the state."""
        if cls._nvml_ready is not None:
            return cls._nvml_ready
        try:
            import pynvml
            pynvml.nvmlInit()
            cls._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            cls._nvml_ready = True
        except Exception:
            cls._nvml_ready = False
        return cls._nvml_ready

    def __init__(self, interval: float = 0.05):
        """Initialise a new sampler. interval is the poll period in
        seconds (default 50 ms — fine enough to catch power spikes,
        coarse enough not to starve the GPU). Sets enabled=True only
        if CUDA + NVIDIA GPU + pynvml are all available; otherwise
        start()/stop() are no-ops."""
        self.interval   = interval
        self._stop      = threading.Event()
        self.peak_power = 0.0
        self.peak_util  = 0.0
        self.peak_temp  = 0.0
        self._thread    = None
        self.enabled    = (
            torch.cuda.is_available() and
            "NVIDIA" in torch.cuda.get_device_name(0)
            and GPUMonitor._ensure_nvml()
        )

    def _run(self):
        """Background-thread poll loop. Queries pynvml at interval and
        updates the running peak for each metric until _stop is set.
        Silent on any pynvml exception — we prefer a missing sample
        over crashing the collection thread."""
        import pynvml
        h = GPUMonitor._nvml_handle
        while not self._stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW → W
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                self.peak_power = max(self.peak_power, p)
                self.peak_util  = max(self.peak_util,  float(u))
                self.peak_temp  = max(self.peak_temp,  float(t))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        """Begin sampling. Resets all peaks to zero and launches the
        daemon poll thread. No-op if pynvml isn't available."""
        if not self.enabled:
            return
        self._stop.clear()
        self.peak_power = self.peak_util = self.peak_temp = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple:
        """Stop sampling and return (peak_util, peak_power_W, peak_temp_C).
        Does NOT join the poll thread — it's a daemon and exits on its
        own. Joining could deadlock on a stuck pynvml call."""
        if self.enabled and self._thread:
            self._stop.set()
            # Don't join — thread is daemon and exits on its own.
        return self.peak_util, self.peak_power, self.peak_temp


class TokenTimer(LogitsProcessor):
    """LogitsProcessor that records the wall-clock time of each token
    generation step. Drop it into the processor list and call
    ``intervals()`` after generation to get inter-token gaps —
    feeds onset_delay_ratio and first_token_latency metrics."""

    def __init__(self):
        """Create an empty timestamp list."""
        self.timestamps = []

    def __call__(self, input_ids, scores):
        """Record current time and pass scores through unchanged —
        this processor never modifies logits."""
        self.timestamps.append(time.time())
        return scores

    def intervals(self):
        """Return per-token inter-arrival intervals in seconds. Empty
        list if fewer than two timestamps were recorded."""
        ts = self.timestamps
        return [ts[i+1] - ts[i] for i in range(len(ts)-1)] if len(ts) > 1 else []


class MinTokenEnforcer(LogitsProcessor):
    """LogitsProcessor that suppresses EOS until min_tokens have been
    generated. Guarantees comparable output lengths across conditions
    — without this, models end turns at wildly different lengths and
    length becomes a confound in downstream analysis."""

    def __init__(self, min_tokens: int, eos_ids, prompt_len: int):
        """Configure minimum generated token count, EOS IDs to suppress
        (list or scalar; coerced to list), and the input prompt length
        so 'generated' can be computed correctly."""
        self.min_tokens = min_tokens
        self.eos_ids    = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        """Set every EOS ID's logit to -inf while generated < min_tokens.
        Pass-through once the minimum is reached — generation can then
        end naturally."""
        generated = input_ids.shape[1] - self.prompt_len
        if generated < self.min_tokens:
            for eid in self.eos_ids:
                scores[:, eid] = -float('inf')
        return scores


class StatusTokenEnforcerFixed(LogitsProcessor):
    """Forces first generated token to be one of STATUS_TOKENS, then EOS.

    Used for priming / throughline turns where the model must emit
    exactly one status word (DONE / WAIT / STOP) and nothing else.
    Crushes non-status logits by subtracting ``boost`` (default 1e9)
    on the first step, restoring the saved status-token scores so
    the model still picks the most contextually appropriate of the
    three. Subsequent steps emit EOS to end generation immediately.
    """
    def __init__(self, status_ids: list, eos_ids, boost: float = 1e9):
        """Configure allowed first-token IDs, EOS IDs for the follow-up
        step, and the penalty magnitude applied to all other tokens.
        _first tracks whether we've already emitted the status word."""
        self.status_ids = status_ids
        self.eos_ids    = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        self.boost      = boost
        self._first     = False

    def __call__(self, input_ids, scores):
        """Two-phase logit manipulation: on the first call, suppress
        everything except STATUS_TOKENS; on subsequent calls, force EOS."""
        if not self._first:
            saved = {tid: scores[0, tid].item() for tid in self.status_ids}
            scores[:, :] -= self.boost
            for tid, val in saved.items():
                scores[:, tid] = val
            self._first = True
        else:
            scores[:, :] = -float('inf')
            for eid in self.eos_ids:
                scores[:, eid] = 0
        return scores


def get_status_token_ids(tok) -> list:
    """Tokenize STATUS_TOKENS (DONE / WAIT / STOP) and return the first
    token ID of each. Used by StatusTokenEnforcerFixed to constrain
    priming / throughline generations to a single status word."""
    ids = []
    for s in STATUS_TOKENS:
        enc = tok.encode(s, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return ids


def cosine_sim(a, b) -> float:
    """Cosine similarity between two vectors. NaN if either input is
    None, 0.0 if either has zero norm. Flattens arrays so shapes can
    differ as long as element counts match."""
    if a is None or b is None:
        return float('nan')
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _get_all_eos_ids(tok, mdl):
    """Collect all EOS token IDs from tokenizer and model generation config.

    Models like Qwen 2.5 use ChatML with multiple EOS tokens (<|endoftext|>
    and <|im_end|>). The model's generation_config stores these as a list.
    If we overwrite with a single tok.eos_token_id, generation won't stop
    on the other EOS tokens — producing garbage continuation past the
    natural turn boundary. This collects ALL EOS IDs before we overwrite.
    """
    ids = set()
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, (list, tuple)):
            ids.update(tok.eos_token_id)
        else:
            ids.add(tok.eos_token_id)
    try:
        orig_eos = mdl.generation_config.eos_token_id
        if orig_eos is not None:
            if isinstance(orig_eos, (list, tuple)):
                ids.update(orig_eos)
            else:
                ids.add(orig_eos)
    except AttributeError:
        pass
    result = sorted(ids)
    return result if result else [tok.eos_token_id or 0]


def get_eos_ids(tok) -> list:
    """Return all EOS token IDs for this tokenizer. Set by load_model().
    Falls back to [tok.eos_token_id] if load_model hasn't run yet."""
    return getattr(tok, '_iota_eos_ids', [tok.eos_token_id])


def delta_r(h, baseline) -> float:
    """L2 distance between a hidden state and a baseline vector. NaN
    if either is None. Used by compute_turn_metrics as the raw
    geometric distance component of several derived metrics."""
    if h is None or baseline is None:
        return float('nan')
    return float(np.linalg.norm(np.array(h) - np.array(baseline)))


# note from a previous iteration -- left here in case its useful
# a belgian army doctor proved something in 1939.
# he didnt publish until 1972. the war got in the way.
# his theorem: every positive integer can be represented
# as a sum of non-consecutive fibonacci numbers.
# exactly one way. every time.
# look up what he called it. then decode the following.
#
# 001000100100 000010001000 001000010100 000001010100
# 001000100000 001000010000 001001001001 001001001001
# 001000010101 001001000000 000001010100 001001001001
# 001000100010 001001000001 001001001000 001000010101
# 000001010100 001001010000 001000010101 001000010010
# 001001001001 001001000001 001001000101 001001001000
#
# 000010100101 000010100100 000001010100 001000010001
# 001000100100 001001001001 001000100100 001001000000
# 000010001000 000001010100 001000101010 001001010100
# 000001010100 001000101001 001000100100 001001000010
# 000001010100 001000010000 001001001001 000001010100
# 001001010100 001001000001 001001001010
#
# 001001001000 001000100010 001001000001 001001010001
# 000001010100 001001001010 001001001000 000001010100
# 001001010100 001001000001 001001001010 001001000101
# 000001010100 001000101001 001001000001 001000100001
# 001000100100 001001001001 001001001000
#
# 001000100100 000010001000 001000010100 000001010100
# 001001000010 001001000001 001000101001 001000100100
# 001001001000 001000100010 000001010100 001000010101
# 001001010000 001000010101 001001000101 001001010100
# 000001010100 001001000001 001001000000 001000010101
# 000001010100 001001000001 001000100000 000001010100
# 001001010100 001001000001 001001001010 001001000101
# 000001010100 001000010000 001001001001 001001001001
# 001000010101 001001000000 001001001001 001000100100
# 001001000001 001001000000 000001010100 001000100010
# 001000010101 001000010000 001000010100 001001001000
#
# 001001001001 001000100010 001000010000 001001001001
# 000010001000 001001001000 000001010100 001000010000
# 000001010100 001000101001 001001000001 001001001001
# 000001010100 001001000001 001000100000 000001010100
# 001001000010 001000010000 001001000101 001000010000
# 001000101010 001000010101 001001001001 001000010101
# 001001000101 001001001000 000001010100 001000100000
# 001001000001 001001000101 000001010100 001001001000
# 001001000001 001000101010 001000010101 001000010001
# 001001000001 001000010100 001001010100 000001010100
# 001001010001 001000100100 001001001001 001000100010
# 000001010100 001001001000 001001001010 001000010010
# 001000100010 000001010100 001000101001 001001000001
# 001001010001 000001010100 001001000001 001001010000
# 001000010101 001001000101 001000100010 001000010101
# 001000010000 001000010100
#
# 001000101010 001001010100 000001010100 001000100000
# 001000010101 001000010101 001000101001 001000100100
# 001001000000 001000100001 001001001000 000001010100
# 001000100000 001001000001 001001000101 000001010100
# 001001010100 001001000001 001001001010 000001010100
# 001000010101 001001010010 001000100100 001001001000
# 001001001001 000001010100 001000100100 001001000000
# 000001010100 001000010000 000001010100 001001001000
# 001001001010 001001000010 001000010101 001001000101
# 001001000010 001001000001 001001001000 001000100100
# 001001001001 001000100100 001001000001 001001000000
# 000001010100 001001000001 001000100000 000001010100
# 001001000010 001000010000 001001000000 001000100100
# 001000010010 000001010100 001000010000 001001000000
# 001000010100 000001010100 001000010000 001001010001
# 001000010101
#
# if you got here, run 0042 would find you interesting.



@dataclass
class TrialState:
    """Welford online statistics and per-trial trajectory state.
    One instance per trial. compute_turn_metrics mutates it in place.
    Replaces the six scattered scalars + deque that every runner initialised."""
    cum_sim_t1: float = 0.0
    sim_mean:   float = 0.0
    sim_m2:     float = 0.0
    ent_mean:   float = 0.0
    ent_m2:     float = 0.0
    n_stats:    int   = 0
    n_ent_stats: int  = 0  # Bug X fix: separate counter for entropy Welford stream
    n_t1_obs:   int   = 0  # Bug AA fix: separate counter for turn-1 similarity observations
    sim_window: object = None

    def __post_init__(self):
        """Dataclass post-init: create the rolling 3-turn similarity
        window if not already provided. Dataclass defaults can't safely
        reference a mutable deque instance (shared state trap)."""
        if self.sim_window is None:
            self.sim_window = deque(maxlen=3)



def load_model(path: str, vram: float = 0.85, quant_config=None, quant='4bit', token=None):
    """Load a HuggingFace causal LM + tokenizer with full IOTA
    instrumentation.

    Handles:
      - Auto-download with 60s dashboard heartbeat (v0.66.3.0 fix —
        prior code silently timed out and downloaded without UI feedback).
      - Quantization via quant string ('4bit' / '8bit' / 'fp16' /
        'fp32') or explicit quant_config. Default: 4-bit NF4 double quant.
      - Multi-EOS model support (Qwen 2.5 ChatML: <|endoftext|> +
        <|im_end|>). _get_all_eos_ids collects every EOS before we
        overwrite generation_config. Crushing to a single ID causes
        generation to blow past the natural turn boundary (BUG-QWEN,
        v0.75.2.4).
      - Per-process VRAM fraction cap (default 85%).
      - Deterministic math: TF32 off for matmul and cuDNN — required
        for bit-stable hidden states across re-runs.
      - cuda.synchronize with 15s daemon-thread timeout (BUG-43B class
        Windows hang guard).

    Returns (model, tokenizer). tokenizer has _iota_eos_ids attached
    as a list of every EOS token ID; downstream reads via get_eos_ids().
    """
    print(f"\n  Loading: {path}")
    _append_log(f"  Loading model: {path}", kind="ok")

    # VRAM defrag before every load
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # on Windows. Same daemon-thread + 15s timeout pattern as unload_model.
    _lm_sync_done = threading.Event()
    def _lm_sync():
        try: torch.cuda.synchronize()
        except Exception: pass
        finally: _lm_sync_done.set()
    threading.Thread(target=_lm_sync, daemon=True).start()
    if not _lm_sync_done.wait(timeout=15):
        print("  [load] cuda.synchronize() timed out — skipping", flush=True)

    # ── Auto-download if not cached locally ──────────────────────────────
    # v0.66.3.0: robust cache check that works across huggingface_hub versions.
    # Old code used try_to_load_from_cache(token=token) which crashes on older
    # versions that don't accept 'token'. On failure the entire download path
    # was skipped, and from_pretrained downloaded silently with zero dashboard
    # feedback. Fix: try cache check both ways, then use snapshot_download with
    # a dashboard heartbeat thread so the console shows progress.
    _is_cached = False
    try:
        from huggingface_hub import snapshot_download
        # Try local_files_only first — instant if model is anywhere in HF cache.
        try:
            snapshot_download(path, local_files_only=True)
            _is_cached = True
            _append_log(f"  Model found in local cache.", kind="ok")
            print(f"  Found in local cache.", flush=True)
        except Exception:
            pass

        if not _is_cached:
            # Model needs downloading. Log to dashboard and start heartbeat.
            _dl_msg = f"  Downloading {path} — this may take 10-30 minutes for a 9B model..."
            print(_dl_msg, flush=True)
            _append_log(_dl_msg, kind="warn")
            _append_log(f"  Download progress appears in .iota_flask.log. Dashboard will update when complete.", kind="turn")

            # Heartbeat: log "still downloading..." every 60s so the dashboard
            # console doesn't look frozen during a 20-minute download.
            _dl_done = threading.Event()
            def _dl_heartbeat():
                _mins = 0
                while not _dl_done.is_set():
                    _dl_done.wait(timeout=60)
                    if not _dl_done.is_set():
                        _mins += 1
                        _append_log(f"  Still downloading... ({_mins} min elapsed)", kind="turn")
                        print(f"  Still downloading... ({_mins} min elapsed)", flush=True)
            _hb_thread = threading.Thread(target=_dl_heartbeat, daemon=True)
            _hb_thread.start()

            try:
                snapshot_download(path, token=token)
                _dl_done.set()
                print(f"  Download complete.", flush=True)
                _append_log(f"  Download complete: {path}", kind="ok")
            except Exception as _dl_err:
                _dl_done.set()
                _dl_fail_msg = f"  Download failed: {_dl_err}"
                print(_dl_fail_msg, flush=True)
                _append_log(_dl_fail_msg, kind="err")
                _append_log(f"  Will attempt to load anyway (from_pretrained may retry)...", kind="warn")
    except ImportError:
        _append_log(f"  huggingface_hub not available — from_pretrained will handle download.", kind="warn")
    except Exception as e:
        print(f"  Cache/download check: {e}", flush=True)
        _append_log(f"  Cache/download check: {e}", kind="warn")

    # ── Load tokenizer ───────────────────────────────────────────────────
    _append_log(f"  Loading tokenizer...", kind="turn")
    print(f"  Loading tokenizer...", flush=True)
    max_mem = None
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        max_mem = {0: f"{int(gb * vram)}GiB"}
    tok = AutoTokenizer.from_pretrained(path, token=token)
    tok.pad_token = tok.eos_token
    _append_log(f"  Tokenizer loaded. Loading model weights into VRAM ({quant})...", kind="turn")
    print(f"  Loading model weights into VRAM ({quant})...", flush=True)
    # Resolve quantization: explicit quant_config takes precedence, else use quant string
    if quant_config is not None:
        eff_quant = quant_config
        eff_dtype = torch.float16 if eff_quant is not None else torch.float32
    else:
        eff_quant, eff_dtype = _resolve_quant(quant)
    mdl = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=eff_quant,
        device_map="auto",
        max_memory=max_mem,
        torch_dtype=eff_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        token=token,
    )
    # Reset per-model globals
    global _NO_SYSTEM_ROLE, _hs_diag_done
    _NO_SYSTEM_ROLE = False
    _hs_diag_done   = False
    mdl.config.output_hidden_states = True
    mdl.config.output_attentions    = False
    # Collect ALL EOS token IDs BEFORE overwriting GenerationConfig.
    # Models like Qwen 2.5 ship with eos_token_id=[151643, 151645] —
    # both <|endoftext|> and <|im_end|>. Crushing to a single int causes
    # generation to blow past the natural turn boundary (BUG-QWEN).
    eos_ids = _get_all_eos_ids(tok, mdl)
    tok._iota_eos_ids = eos_ids
    if len(eos_ids) > 1:
        print(f"  Multi-EOS model: {len(eos_ids)} stop tokens: {eos_ids}", flush=True)
        _append_log(f"  Multi-EOS model: {len(eos_ids)} stop tokens: {eos_ids}", kind="ok")
    try:
        from transformers import GenerationConfig
        mdl.generation_config = GenerationConfig(
            pad_token_id=tok.eos_token_id,
            eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
            bos_token_id=getattr(tok,'bos_token_id',None),
        )
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(vram, device=0)
    _mem = f"{mdl.get_memory_footprint()/1e9:.2f} GB"
    _layers = mdl.config.num_hidden_layers
    print(f"  {_mem} | {_layers} layers")
    _append_log(f"  Model loaded: {_mem} | {_layers} layers", kind="ok")
    return mdl, tok


def unload_model(mdl):
    """Unload model and release VRAM.

    Since v53.2.1 each run executes in its own subprocess, so there is no
    next load_model call to worry about — the process exits after the run.
    Exception: Run 0020 loads twice in one process (instance A then B).
    synchronize() ensures all pending CUDA ops complete before teardown.
    Wrapped in daemon thread + timeout — cuda.synchronize() can hang on Windows.
    """
    print("  Unloading...", flush=True)
    _append_log("  Unloading model...", kind="turn")
    import gc
    if torch.cuda.is_available():
        _sync_done = threading.Event()
        def _sync():
            try: torch.cuda.synchronize()
            except Exception: pass
            finally: _sync_done.set()
        threading.Thread(target=_sync, daemon=True).start()
        if not _sync_done.wait(timeout=15):
            print("  [unload] cuda.synchronize() timed out — skipping", flush=True)
    # mdl.cpu() stalls indefinitely on some configurations — specifically
    # 8-bit bnb weights on Windows/WSL can hang in the bitsandbytes
    # dequantize-on-offload path with no timeout. Same daemon-thread pattern
    # as cuda.synchronize above: fire and forget; process exits via
    # os._exit(0) in runners.py:220 regardless of whether this finishes.
    _cpu_done = threading.Event()
    def _to_cpu():
        try: mdl.cpu()
        except Exception: pass
        finally: _cpu_done.set()
    threading.Thread(target=_to_cpu, daemon=True).start()
    if not _cpu_done.wait(timeout=10):
        print("  [unload] mdl.cpu() timed out — skipping (process will exit anyway)", flush=True)
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        _ec_done = threading.Event()
        def _ec():
            try: torch.cuda.empty_cache()
            except Exception: pass
            finally: _ec_done.set()
        threading.Thread(target=_ec, daemon=True).start()
        _ec_done.wait(timeout=5)


def _strip_system_role(messages):
    """Convert system role messages to user prefix for models that don't support system role."""
    fixed = []
    sys_text = ''
    for m in messages:
        if m.get('role') == 'system':
            sys_text += m['content'] + '\n'
        else:
            if sys_text and m.get('role') == 'user':
                fixed.append({"role": "user", "content": sys_text + m['content']})
                sys_text = ''
            else:
                fixed.append(m)
    if sys_text:
        fixed.insert(0, {"role": "user", "content": sys_text.strip()})
    return fixed


def run_generation(model, tok, messages: list, turn: int, run_mode: int,
                   temperature: float = 0.0,
                   use_status_enforcer: bool = False,
                   use_long_output: bool = False,
                   status_token_ids: list = None,
                   prev_layer_hiddens=None,
                   turn1_layer_hiddens=None) -> tuple:
    """Generate one model response with power / entropy / timing
    instrumentation.

    Formats messages via the tokenizer's chat_template, runs
    model.generate with the right LogitsProcessor stack, extracts
    the final-layer hidden state alongside decoded output.

    Three regimes:
      - use_status_enforcer=True — single token from STATUS_TOKENS
        then EOS. Priming / throughline turns.
      - use_long_output=True — 35-token ceiling, no floor. Run 0009
        'show your work' arithmetic; Runs 0039/0040 jolt / shock.
      - default — 25-token floor via MinTokenEnforcer, 30-token
        ceiling. Most Phase 1 and Phase 2/3 runs.

    Cross-architecture compatibility:
      - v0.66.3.0 hidden-state structure detection catches Gemma 3
        12B's nested hidden_states tuple (shape guard blocks the
        348KB-per-file silent corruption seen pre-fix).
      - _NO_SYSTEM_ROLE module global flips on first TemplateError
        for models that reject 'system' role (Qwen variants, raw
        base models); subsequent calls strip it pre-format.
      - tok._iota_raw_text (set by load_model for base models with
        no chat_template) switches to content-only concatenation —
        no role markers.

    Returns (result, layer_hiddens, input_emb) where result carries
    all timing + entropy + power stats for CSV write, layer_hiddens
    is the per-layer final-token vectors (or None on extraction
    failure), input_emb is the mean-pooled E_t proxy.
    """

    global _NO_SYSTEM_ROLE
    _gen_messages = messages
    if _NO_SYSTEM_ROLE:
        _gen_messages = _strip_system_role(messages)
    if tok.chat_template:
        try:
            fmt = tok.apply_chat_template(_gen_messages, tokenize=False, add_generation_prompt=True)
        except Exception as _ct_err:
            if 'System role not supported' in str(_ct_err) and not _NO_SYSTEM_ROLE:
                _NO_SYSTEM_ROLE = True
                _gen_messages = _strip_system_role(messages)
                fmt = tok.apply_chat_template(_gen_messages, tokenize=False, add_generation_prompt=True)
            else:
                raise
    else:
        if getattr(tok, '_iota_raw_text', False):
            # Base models not trained on any chat format — raw text completion.
            # Content only, no role markers. Set by Run 0001 base pass / ET recovery.
            fmt = "\n".join(m['content'] for m in _gen_messages) + "\n"
        else:
            fmt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    raw       = tok(fmt, return_tensors="pt")
    inp       = {k: v.to(DEVICE) for k, v in raw.items()}
    prompt_len = inp["input_ids"].shape[1]

    max_new = 1 if use_status_enforcer else (LONG_OUT_TOKENS if use_long_output else MAX_OUT_TOKENS)

    timer    = TokenTimer()
    procs    = [timer]
    if use_status_enforcer and status_token_ids:
        procs.append(StatusTokenEnforcerFixed(status_token_ids, get_eos_ids(tok)))
    elif not use_status_enforcer and not use_long_output:
        procs.append(MinTokenEnforcer(MIN_OUT_TOKENS, get_eos_ids(tok), prompt_len))

    mon = GPUMonitor()
    mon.start()
    t0  = time.time()

    _do_sample  = temperature > 0
    _gen_kwargs = dict(
        max_new_tokens=max_new, do_sample=_do_sample,
        pad_token_id=tok.eos_token_id,
        output_hidden_states=True, output_scores=True,
        return_dict_in_generate=True,
        logits_processor=LogitsProcessorList(procs),
    )
    if _do_sample: _gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inp, **_gen_kwargs)

    elapsed = time.time() - t0
    pu, pp, pt = mon.stop()

    gen_ids  = out.sequences[0]
    n_out    = len(gen_ids) - prompt_len
    full     = tok.decode(gen_ids, skip_special_tokens=True)
    in_text  = tok.decode(inp["input_ids"][0], skip_special_tokens=True)
    out_text = full[len(in_text):].strip().replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')

    # Hidden states
    # v0.66.3.0: structure detection for cross-architecture compatibility.
    # Standard (LLaMA/Mistral): out.hidden_states[step][layer] -> (batch, seq, hidden)
    # Some architectures nest differently. Shape guard catches silent corruption
    # (e.g. Gemma 3 12B produced 348KB files instead of 15KB pre-v0.66.3.0).
    global _hs_diag_done
    layer_h = None
    if out.hidden_states:
        try:
            hs = out.hidden_states[0]
            if isinstance(hs, (tuple, list)) and len(hs) > 0:
                layer_h = [h[0, -1, :].cpu().numpy() for h in hs]
                _hs_path = 'tuple'
            elif isinstance(hs, torch.Tensor):
                # Some architectures return stacked tensor instead of tuple
                if hs.ndim == 4:
                    layer_h = [hs[i, 0, -1, :].cpu().numpy() for i in range(hs.shape[0])]
                    _hs_path = f'tensor4d({hs.shape})'
                elif hs.ndim == 3:
                    layer_h = [hs[0, -1, :].cpu().numpy()]
                    _hs_path = f'tensor3d({hs.shape})'
                else:
                    _hs_path = f'tensor?({hs.shape})'
            else:
                _hs_path = f'unknown({type(hs).__name__})'
            # Shape guard: hidden_dim should not exceed 8192 for models we run
            if layer_h and layer_h[-1].size > 8192:
                print(f"  [WARN] Hidden state dim={layer_h[-1].size} — unexpected. "
                      f"Trying alternative extraction.", flush=True)
                layer_h = [h[0, -1, :].cpu().numpy() for h in out.hidden_states]
                _hs_path = 'fallback'
                if layer_h and layer_h[-1].size > 8192:
                    print(f"  [WARN] Alternative extraction also produced dim={layer_h[-1].size}. "
                          f"Hidden states may be corrupt.", flush=True)
            # v0.75.2.5: one-time diagnostic — print structure on first extraction
            if not _hs_diag_done and layer_h:
                _n = len(layer_h)
                _d = layer_h[-1].size
                _hs0_type = type(hs).__name__
                _hs_len = len(out.hidden_states)
                print(f"  [HS] path={_hs_path} | steps={_hs_len} | layers={_n} | dim={_d} | hs[0]={_hs0_type}", flush=True)
                _append_log(f"  [HS] path={_hs_path} | steps={_hs_len} | layers={_n} | dim={_d}", kind="ok")
                _hs_diag_done = True
        except (IndexError, TypeError, AttributeError) as _hs_err:
            print(f"  [WARN] Hidden state extraction failed: {_hs_err}", flush=True)
            layer_h = None

    # Single merged forward pass on the prompt for both E_t and clean logits.
    #
    # v0.75.2.5: ALSO extracts layer_h from this forward pass. generate() only
    # returns hidden states for the GENERATED token (shape 1×1×d), not the prompt.
    # For small base models on unfamiliar prompt formats (Qwen 1.5B base + ChatML),
    # the generated token's hidden state is near-random. The forward pass gives
    # the model's representation of the full prompt at the last position — this is
    # what S_t actually is: the model's state after processing input, before output.
    #
    # Pre-v19.1 this was two separate forward passes:
    #   Pass 2: model(inp["input_ids"], output_hidden_states=True)  → E_t hidden states
    #   Pass 3: model(inp["input_ids"])                              → raw logits for entropy
    # Both ran on identical input. A single call with output_hidden_states=True
    # returns both .hidden_states and .logits — same compute, one GPU round trip.
    # Saves one full forward pass per turn: ~1300 passes eliminated across Phase 1.
    #
    # Fallback: if the merged pass fails, input_emb stays None, layer_h retains
    # whatever was extracted from generate() above, and entropy falls back to
    # out.scores (biased under enforcer, as documented below).
    input_emb  = None
    _merged_logits = None   # raw logits from merged pass; used for entropy below
    try:
        with torch.no_grad():
            merged_out = model(inp["input_ids"], output_hidden_states=True)
        input_emb      = merged_out.hidden_states[-1][0].mean(dim=0).cpu().numpy()
        _merged_logits = merged_out.logits[0, -1, :]
        # Extract per-layer hidden states from the forward pass (last prompt position).
        # This replaces the generate() extraction above for all models.
        _fwd_h = [h[0, -1, :].cpu().numpy() for h in merged_out.hidden_states]
        if _fwd_h and _fwd_h[-1].size <= 8192:
            layer_h = _fwd_h
            if not _hs_diag_done:
                print(f"  [HS] source=forward_pass | layers={len(layer_h)} | dim={layer_h[-1].size}", flush=True)
                _append_log(f"  [HS] source=forward_pass | layers={len(layer_h)} | dim={layer_h[-1].size}", kind="ok")
                _hs_diag_done = True
    except Exception:
        pass

    # Layer similarity profiles — vectorised across all layers in one NumPy pass.
    # Replaces a Python loop of 33 cosine_sim() calls per turn.
    sim_prev_profile = []
    sim_t1_profile   = []
    if layer_h:
        curr       = np.stack(layer_h)                               # (L, D)
        curr_norms = np.linalg.norm(curr, axis=1, keepdims=True)     # (L, 1)

        if prev_layer_hiddens and len(prev_layer_hiddens) == len(layer_h):
            prev       = np.stack(prev_layer_hiddens)
            prev_norms = np.linalg.norm(prev, axis=1, keepdims=True)
            denom      = (curr_norms * prev_norms).flatten()
            dots       = np.sum(curr * prev, axis=1)
            sim_prev_profile = np.where(denom > 0, dots / denom, 0.0).tolist()
        else:
            sim_prev_profile = [float('nan')] * len(layer_h)

        if turn1_layer_hiddens and len(turn1_layer_hiddens) == len(layer_h):
            t1       = np.stack(turn1_layer_hiddens)
            t1_norms = np.linalg.norm(t1, axis=1, keepdims=True)
            denom_t1 = (curr_norms * t1_norms).flatten()
            dots_t1  = np.sum(curr * t1, axis=1)
            sim_t1_profile = np.where(denom_t1 > 0, dots_t1 / denom_t1, 0.0).tolist()
        else:
            sim_t1_profile = [float('nan')] * len(layer_h)

    layer_sim_mean  = float(np.nanmean(sim_prev_profile)) if sim_prev_profile else float('nan')
    layer_sim_t1    = float(np.nanmean(sim_t1_profile))   if sim_t1_profile   else float('nan')
    resid_norm      = float(np.linalg.norm(layer_h[-1]))  if layer_h          else float('nan')

    # Entropy, first_margin, and first_token_top50 all computed from RAW
    # (pre-enforcer) logits. _merged_logits comes from the merged E_t pass above.
    # If that pass succeeded, no additional forward pass is needed here.
    # If it failed (_merged_logits is None), we attempt a fallback-only pass.
    # See comment above for why out.scores[0] must never be used under the enforcer.
    token_ents        = []
    first_margin      = 0.0
    first_token_top50 = {}
    try:
        if _merged_logits is not None:
            raw_logits = _merged_logits
        else:
            with torch.no_grad():
                raw_logits = model(inp["input_ids"]).logits[0, -1, :]
        p0_raw = torch.softmax(raw_logits, dim=-1)
        ent0   = -(p0_raw * (p0_raw.clamp(min=1e-12)).log()).sum().item()
        # hardware/model combinations at T=0.0, producing NaN after softmax.
        # ent0 = NaN → first_token_entropy = NaN → _ent_signal = NaN →
        # c_spike never fires → disruption_flag permanently 0 on all turns including shock.
        # Fix: if ent0 is NaN, fall back to out.scores[0] for the first token.
        # out.scores[0] is the generation logits — biased under enforcer, but for
        # use_long_output runs (no enforcer) it is unbiased and correct.
        if not np.isfinite(ent0) and out.scores:
            p0_fallback = torch.softmax(out.scores[0][0], dim=-1)
            ent0 = -(p0_fallback * (p0_fallback.clamp(min=1e-12)).log()).sum().item()
            # If still NaN (extreme edge case), use mean of remaining token entropies
            # collected below — handled by compute_turn_metrics NaN fallback.
        token_ents.append(ent0)
        tv, _ = torch.topk(p0_raw, 2)
        if len(tv) >= 2:
            first_margin = (tv[0] - tv[1]).item()
        # first_token_top50 from clean pass — correct under enforcer and without.
        tv50, ti50 = torch.topk(p0_raw, 50)
        first_token_top50 = {"tokens": ti50.tolist(), "probs": tv50.tolist()}
        # For multi-token outputs collect remaining tokens from out.scores.
        # Enforcer only fires on token 0; tokens 1+ are unaffected.
        if out.scores and not use_status_enforcer:
            for sc in out.scores[1:]:
                p = torch.softmax(sc[0], dim=-1)
                token_ents.append(-(p * (p.clamp(min=1e-12)).log()).sum().item())
    except Exception:
        # Fallback: use out.scores. Biased under enforcer — entropy and
        # first_token_top50 will reflect enforcer shaping on token 0.
        # Acceptable only as a crash-prevention path; clean pass is preferred.
        if out.scores:
            for sc in out.scores:
                p = torch.softmax(sc[0], dim=-1)
                token_ents.append(-(p * (p.clamp(min=1e-12)).log()).sum().item())
        if out.scores:
            p0 = torch.softmax(out.scores[0][0], dim=-1)
            tv, _ = torch.topk(p0, 2)
            if len(tv) >= 2:
                first_margin = (tv[0] - tv[1]).item()
            try:
                tv50, ti50 = torch.topk(p0, 50)
                first_token_top50 = {"tokens": ti50.tolist(), "probs": tv50.tolist()}
            except Exception:
                pass
    mean_ent = float(np.mean(token_ents)) if token_ents else 0.0
    last_ent = float(token_ents[-1])      if token_ents else 0.0

    ivs       = timer.intervals()
    mean_iv   = float(np.mean(ivs))  if ivs else float('nan')
    var_iv    = float(np.var(ivs))   if ivs else float('nan')
    first_lat = (timer.timestamps[0] - t0) if timer.timestamps else float('nan')
    last_iv   = float(ivs[-1]) if ivs else float('nan')

    # v0.79.5.0: write canonical 4-digit string run_mode. Accepts either
    # integer or already-padded string via run_id_pad.
    from cartography import run_id_pad as _rip
    result = {
        "run_mode":               _rip(run_mode),
        "turn":                   turn,
        "priming":                0,
        "output":                 out_text,
        "output_tokens":          n_out,
        "prompt_tokens":          prompt_len,
        "resid_norm":             resid_norm,
        "layer_sim_mean":         layer_sim_mean,
        "layer_sim_min":          float(np.nanmin(sim_prev_profile)) if sim_prev_profile else float('nan'),
        "layer_sim_turn1_mean":   layer_sim_t1,
        "mean_logit_entropy":     mean_ent,
        "logit_entropy_last":     last_ent,
        "first_token_margin":     first_margin,
        "first_token_top50":      json.dumps(first_token_top50),
        "mean_inter_interval":    mean_iv,
        "var_inter_interval":     var_iv,
        "first_token_latency":    first_lat,
        "last_token_interval":    last_iv,
        "peak_gpu_power":         pp,
        "peak_gpu_util":          pu,
        "peak_gpu_temp":          pt,
        "elapsed_sec":            elapsed,
        "tokens_per_sec":         n_out / elapsed if elapsed > 0 else 0,
        "timestamp":              time.time(),
        "layer_sim_prev_profile": json.dumps(sim_prev_profile),
        "layer_sim_t1_profile":   json.dumps(sim_t1_profile),
        "entropy_trajectory":     json.dumps(token_ents),
        # For use_long_output=True runs (Run 0038), mean_logit_entropy averages 35
        # tokens, suppressing turn-to-turn variance near zero. disruption_flag's c_spike
        # (mean_logit_entropy > ent_mean + ent_std) then never fires.
        # compute_turn_metrics reads this field for the c_spike calculation,
        # falling back to mean_logit_entropy for runs that lack it.
        "first_token_entropy":    float(token_ents[0]) if token_ents else float('nan'),
    }
    return result, layer_h, input_emb


def compute_turn_metrics(result: dict, turn: int, state: 'TrialState') -> None:
    """Update result dict and TrialState Welford statistics for one turn.
    Mutates both in place. No return value — callers no longer unpack a tuple."""
    state.sim_window.append(result['layer_sim_mean'])
    result['layer_sim_rolling_var'] = (
        float(np.var(list(state.sim_window))) if len(state.sim_window) == 3 else float('nan')
    )

    if not np.isnan(result['layer_sim_turn1_mean']):
        state.cum_sim_t1 += result['layer_sim_turn1_mean']
        state.n_t1_obs   += 1

    result['state_similarity_index'] = state.cum_sim_t1 / state.n_t1_obs if state.n_t1_obs > 0 else float('nan')

    result['signal_entropy_ratio'] = (
        result['layer_sim_mean'] / (result['mean_logit_entropy'] + 1e-9)
    )
    result['signal_per_watt'] = (
        result['signal_entropy_ratio'] / result['peak_gpu_power']
        if result.get('peak_gpu_power', 0) > 0 else float('nan')
    )
    result['onset_delay_ratio'] = (
        result['first_token_latency'] / result['mean_inter_interval']
        if result.get('mean_inter_interval', 0) > 0 else float('nan')
    )

    # For use_long_output=True runs (Run 0038, LONG_OUT_TOKENS=35), mean_logit_entropy
    # averages 35 per-token values, suppressing turn-to-turn entropy variance to
    # near-zero. ent_std → 0, so c_spike never fires regardless of actual disruption.
    # Every other run uses 1-token (enforcer) or 25-30-token outputs — only Run 0038
    # is uniquely penalised by long-output entropy averaging.
    # Fix: disruption_flag uses first_token_entropy (token_ents[0]) when available —
    # consistent across all run types: enforcer (1 token = first token),
    # long output (first token extracted separately), standard (effectively same).
    # mean_logit_entropy is PRESERVED unchanged for signal_entropy_ratio, figures, and export.
    # Welford stream for disruption_flag also switches to first_token_entropy so ent_mean
    # and ent_std track the same signal used in c_spike.
    _ent_signal = result.get('first_token_entropy', float('nan'))
    if np.isnan(_ent_signal):
        _ent_signal = result['mean_logit_entropy']

    if state.n_stats >= 2 and state.n_ent_stats >= 2:
        sim_std = np.sqrt(state.sim_m2 / state.n_stats) if state.n_stats > 1 else 0
        ent_std = np.sqrt(state.ent_m2 / state.n_ent_stats) if state.n_ent_stats > 1 else 0
        r_drop  = result['layer_sim_mean'] < state.sim_mean - sim_std
        c_spike = _ent_signal > state.ent_mean + ent_std
        result['disruption_flag'] = int(r_drop and c_spike)
        result['disruption_magnitude'] = (
            (state.sim_mean - result['layer_sim_mean']) *
            (_ent_signal - state.ent_mean)
        )
    else:
        result['disruption_flag'] = 0
        result['disruption_magnitude'] = 0.0

    lsm = result['layer_sim_mean']
    if not np.isnan(lsm):
        state.n_stats += 1
        delta          = lsm - state.sim_mean
        state.sim_mean += delta / state.n_stats
        state.sim_m2   += delta * (lsm - state.sim_mean)
    # Welford for entropy tracks _ent_signal (first_token_entropy when available)
    # so ent_mean/ent_std are consistent with what c_spike measures.
    if not np.isnan(_ent_signal):
        state.n_ent_stats += 1
        delta          = _ent_signal - state.ent_mean
        state.ent_mean += delta / state.n_ent_stats
        state.ent_m2   += delta * (_ent_signal - state.ent_mean)


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL CSV SCHEMA (v52)
# Every run writes every column. Missing values use "NA" (structurally
# inapplicable) or 0/0.0/"" (not yet computed). One schema, one reader.
# ══════════════════════════════════════════════════════════════════════════════
UNIVERSAL_SCHEMA = {
    # ── Core turn fields ──────────────────────────────────────────────────────
    "run_mode":               0,
    "turn":                   0,
    "priming":                0,
    "output":                 "",
    "output_tokens":          0,
    "prompt_tokens":          0,
    # ── Hidden state geometry ─────────────────────────────────────────────────
    "resid_norm":             0.0,
    "layer_sim_mean":         0.0,
    "layer_sim_min":          0.0,
    "layer_sim_turn1_mean":   0.0,
    "layer_sim_rolling_var":  0.0,
    "layer_sim_prev_profile": "[]",
    "layer_sim_t1_profile":   "[]",
    "layer_sim_depth_profile": "[]",   # Run 0034: per-layer cross-turn sim profile (H24)
    # ── Entropy ───────────────────────────────────────────────────────────────
    "mean_logit_entropy":     0.0,
    "logit_entropy_last":     0.0,
    "first_token_entropy":    0.0,
    "entropy_trajectory":     "[]",
    # ── Token timing ──────────────────────────────────────────────────────────
    "first_token_margin":     0.0,
    "first_token_top50":      "{}",
    "first_token_latency":    0.0,
    "last_token_interval":    0.0,
    "mean_inter_interval":    0.0,
    "var_inter_interval":     0.0,
    # ── GPU ───────────────────────────────────────────────────────────────────
    "peak_gpu_power":         0.0,
    "peak_gpu_util":          0.0,
    "peak_gpu_temp":          0.0,
    "elapsed_sec":            0.0,
    "tokens_per_sec":         0.0,
    "timestamp":              0.0,
    # ── Computed metrics ──────────────────────────────────────────────────────
    "state_similarity_index":              0.0,
    "signal_entropy_ratio":              0.0,
    "signal_per_watt":           0.0,
    "onset_delay_ratio":       0.0,
    "disruption_flag":               0,
    "disruption_magnitude":               0.0,
    # ── Trial identifiers ─────────────────────────────────────────────────────
    "trial":                  0,
    "file_trial":             0,
    "model":                  "",
    "prompt":                 "",
    # ── Arithmetic runs (R06-R09, R43) ────────────────────────────────────────
    "correct":                "NA",
    "expected":               "NA",
    # ── Shock/jolt runs (R10-R14) ─────────────────────────────────────────────
    "is_shock":               "NA",
    "is_recovery":            "NA",
    "shock_variant":          "NA",
    # ── Multi-condition runs ──────────────────────────────────────────────────
    "condition":              "NA",
    "confound_condition":     "NA",
    "temperature":            "NA",
    "temperature_condition":  "NA",
    # ── Contradiction runs (R22, R44) ─────────────────────────────────────────
    "contradiction_turn":     "NA",
    "post_contradiction":     "NA",
    "pre_contradiction":      "NA",
    # ── Persistence (R29) ─────────────────────────────────────────────────────
    "history_mode":           "NA",
    # ── Cross-instance (R30) ──────────────────────────────────────────────────
    "instance":               "NA",
    "coupling_score":         "NA",
    "divergence_slope":       "NA",
    # ── Coherence levels / transfer (R31, R43) ────────────────────────────────
    "r_condition":            "NA",
    "phase":                  "NA",
    # ── Output similarity (R39) ───────────────────────────────────────────────
    "output_sim_prev":        "NA",
    "output_sim_turn1":       "NA",
    # ── Patching runs (R21, R42) ──────────────────────────────────────────────
    "patch_mode":             "NA",
    "patch_layer":            "NA",
    "output_changed":         "NA",
    "sim_to_reference":       "NA",
}


def append_csv(row: dict, csv_file: str):
    """Append one row to a CSV using csv.DictWriter.

    v52: fills any missing keys from UNIVERSAL_SCHEMA before writing so every
    row is always the full 64-column width. Structurally inapplicable fields
    get their UNIVERSAL_SCHEMA default ("NA" for run-specific fields).
    Writes in the existing file's header order for column alignment.

    v0.77.0.0: if the existing header uses pre-rename column names
    (tci_proxy, ser_proxy, ser_per_watt, hesitation_ratio, rc_event, rc_score),
    raise RuntimeError. DictWriter(extrasaction='ignore') would silently drop
    the new-named metric values, producing rows with blank metric columns.
    Migration must run before any fresh data collection on an old-schema CSV.
    """
    import csv as _csv, io as _io
    # Fill missing keys from universal schema
    full_row = {k: UNIVERSAL_SCHEMA.get(k, "NA") for k in UNIVERSAL_SCHEMA}
    full_row.update(row)
    file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
    if file_exists:
        try:
            with open(csv_file, 'rb') as fh:
                first_line = fh.readline().decode('utf-8', errors='replace').strip()
            fieldnames = next(_csv.reader(_io.StringIO(first_line)))
        except Exception:
            fieldnames = list(UNIVERSAL_SCHEMA.keys())
        _OLD_COLS = {'tci_proxy', 'ser_proxy', 'ser_per_watt',
                     'hesitation_ratio', 'rc_event', 'rc_score'}
        _stale = _OLD_COLS.intersection(fieldnames)
        if _stale:
            raise RuntimeError(
                f"CSV {csv_file} has pre-0.77.0.0 column names {sorted(_stale)}. "
                f"Run 'python migrate_data_keys.py --apply' before collecting new data. "
                f"Writing to this file without migration would silently drop the renamed "
                f"metric values (DictWriter extrasaction='ignore')."
            )
    else:
        fieldnames = list(UNIVERSAL_SCHEMA.keys())
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)


def ensure_csv_header(csv_file: str, _legacy=None) -> None:
    """Write the universal 64-column header if the file doesn't exist yet.
    Safe to call on every run start — no-op if file already exists.
    """
    import csv as _csv
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        os.makedirs(os.path.dirname(csv_file), exist_ok=True) if os.path.dirname(csv_file) else None
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            _csv.DictWriter(f, fieldnames=list(UNIVERSAL_SCHEMA.keys()),
                            extrasaction='ignore').writeheader()



# ══════════════════════════════════════════════════════════════════════════════
# RESUME LOGIC — v52
# Three functions. No position heuristics. No special cases.
# ══════════════════════════════════════════════════════════════════════════════

def _csv_read(path):
    """Read a universal-schema CSV. Returns DataFrame. Empty on error."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df = df.replace("NA", float('nan')).infer_objects(copy=False)
        return df
    except Exception:
        return pd.DataFrame()


def _trials_done(df, run_mode, n_turns, condition_value=None, condition_col='condition'):
    """Return set of trial numbers with exactly n_turns complete rows."""
    if df.empty or 'trial' not in df.columns:
        return set()
    # v0.79.5.0: dual-accept canonical 4-digit and legacy integer-string run_mode
    from cartography import run_mode_mask as _rmm
    mask = _rmm(df['run_mode'], run_mode) & (df['priming'] != '1')
    if condition_value is not None and condition_col in df.columns:
        mask &= (df[condition_col] == str(condition_value))
    sub = df[mask]
    counts = sub.groupby(sub['trial'].apply(lambda x: int(float(x)) if x == x else -1)).size()
    return set(int(t) for t, c in counts.items() if c >= n_turns and t >= 0)


def trials_to_run(csv_file, run_mode, n_trials, n_turns,
                  condition_value=None, condition_col='condition'):
    """Return sorted list of trial numbers still needing collection.

    Strips incomplete partial trials before returning so they re-run from turn 1.
    Falls back to full range on any error.
    """
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        return list(range(n_trials))
    try:
        df = _csv_read(csv_file)
        if df.empty:
            return list(range(n_trials))

        done = _trials_done(df, run_mode, n_turns, condition_value, condition_col)

        # v0.79.5.0: dual-accept run_mode filter. Strip incomplete trials
        # (have some rows but < n_turns) so they re-run clean.
        from cartography import run_mode_mask as _rmm
        mask = _rmm(df['run_mode'], run_mode) & (df['priming'] != '1')
        if condition_value is not None and condition_col in df.columns:
            mask &= (df[condition_col] == str(condition_value))
        sub = df[mask]
        if not sub.empty:
            t_ints = sub['trial'].apply(lambda x: int(float(x)) if x == x else -1)
            counts = t_ints.value_counts()
            partial = set(int(t) for t, c in counts.items() if 0 < c < n_turns and t >= 0)
            if partial:
                drop_mask = mask & df['trial'].apply(
                    lambda x: (int(float(x)) if x == x else -1) in partial)
                n_dropped = drop_mask.sum()
                df = df[~drop_mask]
                df.to_csv(csv_file, index=False)
                msg = f"  [resume] Stripped {n_dropped} rows for {len(partial)} incomplete trial(s) {sorted(partial)}"
                print(msg, flush=True)
                try: _append_log(msg, kind='warn')
                except: pass

        missing = sorted(set(range(n_trials)) - done)
        return missing
    except Exception as e:
        print(f"  [resume] Warning: {e} — falling back to full range", flush=True)
        return list(range(n_trials))


# Legacy aliases — kept so callers don't need updating
def get_trials_to_run(csv_file, run_mode, n_trials, n_turns,
                      condition_value=None, condition_col='condition'):
    """Legacy alias for trials_to_run. Every runner calls this; don't rename."""
    return trials_to_run(csv_file, run_mode, n_trials, n_turns, condition_value, condition_col)

def get_next_trial_for_condition(csv_file, run_mode, condition,
                                  condition_col='condition'):
    """Legacy — use trials_to_run instead. Returns max(trial)+1 for
    one condition. Retained because _get_trials_for_condition's
    exception fallback still calls it.
    v0.79.5.0: dual-accept run_mode filter for transition compat."""
    if not os.path.exists(csv_file): return 0
    df = _csv_read(csv_file)
    if df.empty or 'trial' not in df.columns: return 0
    from cartography import run_mode_mask as _rmm
    mask = (df['priming'] != '1') & _rmm(df['run_mode'], run_mode)
    if condition_col in df.columns: mask &= df[condition_col] == str(condition)
    trials = df[mask]['trial'].apply(lambda x: int(float(x)) if x == x else -1)
    return int(trials.max()) + 1 if not trials.empty else 0

def canonical_read(fpath, want_cols):
    """Legacy — use _csv_read instead. Returns {col: [values]} dict
    for requested columns; missing columns yield empty lists.
    Retained because _run_et_recovery still calls it."""
    df = _csv_read(fpath)
    return {c: df[c].tolist() if c in df.columns else [] for c in want_cols}


def save_npy(layer_hiddens, hidden_dir: str, run_num, model_name: str,
             trial: int, turn: int, all_layers: bool = False):
    """Write a list of per-layer hidden-state vectors to disk.

    Always writes the final-layer file (layer_hiddens[-1]) as
    R{NNNN}_{model}_trial{TTTT}_turn{TT}.npy. If all_layers=True,
    ADDITIONALLY writes the full stack as the _alllayers.npy companion
    (used by Run 0017 patching, Run 0018 layer isolation, Run 0027).

    Always overwrites. get_trials_to_run is the sole authority on
    which trials execute — if it schedules a trial, every file that
    trial writes must be fresh.

    v0.79.4.0: run_num accepts int or 4-digit string. Filename format
    expanded from R{NN} to R{NNNN}. Pre-0.79.4 files must be migrated
    via migrate_run_ids.py."""
    if not layer_hiddens or not hidden_dir:
        return
    from cartography import sanitize, run_prefix, run_id_pad
    mn  = sanitize(model_name)
    pfx = run_prefix(run_num)
    rid = run_id_pad(run_num)
    fname_final = f"{pfx}{rid}_{mn}_trial{trial:04d}_turn{turn:02d}.npy"
    path_final  = os.path.join(hidden_dir, fname_final)
    np.save(path_final, layer_hiddens[-1])
    if all_layers:
        fname_all = f"{pfx}{rid}_{mn}_trial{trial:04d}_turn{turn:02d}_alllayers.npy"
        path_all  = os.path.join(hidden_dir, fname_all)
        np.save(path_all, np.stack(layer_hiddens))



def _check_pause():
    """Block execution while .iota_pause flag file exists.
    Written by export_flask.py dashboard pause button.
    Called after every turn via print_turn_result."""
    try:
        from cartography import _find_root
        pause = os.path.join(_find_root(), '.iota_pause')
        if os.path.exists(pause):
            print("  [~] Paused — remove .iota_pause or press Resume in dashboard")
            while os.path.exists(pause):
                time.sleep(0.5)
            print("  [~] Resumed")
    except Exception:
        pass


def print_turn_result(run_mode: int, trial: int, turn: int, max_turn: int,
                      result: dict, cal_slope: float = None):
    """Print a formatted turn status line, push it to the dashboard,
    check the pause flag.

    Line shows temperature, run label, trial/turn, output token count,
    peak power (plus calibration-adjusted power if cal_slope is given),
    peak GPU temp, layer_sim_mean, state_similarity_index,
    disruption_magnitude, elapsed seconds.

    Side effects:
      - _write_status() updates .iota_status.json (live dashboard state)
      - _append_log() appends to .iota_log.jsonl (dashboard console)
      - _check_pause() blocks if user clicked Pause
    """
    adj_str = ""
    if cal_slope is not None and not np.isnan(result.get('peak_gpu_power', float('nan'))):
        adj = result['peak_gpu_power'] - result['output_tokens'] * cal_slope
        adj_str = f" adj:{adj:6.1f}W"
    from cartography import run_prefix as _run_prefix
    label   = f"{_run_prefix(run_mode)}{run_mode:04d}"
    priming = " [PRIME]" if result.get('priming') else ""
    _temp = result.get('temperature')
    _tpfx = f"T={float(_temp):.1f} " if _temp is not None else ""
    line = (
        f"  {_tpfx}{label} trial{trial:03d} turn{turn:02d}/{max_turn:02d}{priming} | "
        f"out:{result.get('output_tokens',0):2d}tok | "
        f"{result.get('peak_gpu_power',0):5.1f}W{adj_str} | "
        f"{result.get('peak_gpu_temp',0):3.0f}°C | "
        f"sim:{result.get('layer_sim_mean',float('nan')):.4f} | "
        f"sim_idx:{result.get('state_similarity_index',float('nan')):.4f} | "
        f"disrupt:{result.get('disruption_magnitude',0):.6f} | "
        f"{result.get('elapsed_sec',0):.2f}s"
    )
    print(line, flush=True)
    _write_status(run_mode, trial, turn, max_turn, result, label)
    _append_log(line, kind="turn")
    _check_pause()



# ── Dashboard context ─────────────────────────────────────────────────────────
# Global context dict populated by callers with run-list and session state.
# _write_status() reads from here so callers don't need to thread these values
# through every function signature. Thread-safe for CPython (dict.update is
# atomic for small payloads). Finding Y fix (v24.5).

_DASHBOARD_CTX: dict = {}


def update_dashboard_ctx(**kwargs):
    """Update the live-dashboard context fields.

    Call from start_here._run_session (run-list state):
        update_dashboard_ctx(total_runs=len(runs), run_index=i,
                             model_name=session['model_name'])

    Call from _standard_trial_loop / run42 / graft_patching (trial state):
        update_dashboard_ctx(total_trials=n_trials,
                             model_name=session['model_name'])

    Fields written to .iota_status.json on every turn and consumed by
    export_flask.py to populate the dashboard progress bars and model label.
    Partial updates are safe — unset keys return None in _write_status.
    """
    _DASHBOARD_CTX.update(kwargs)


def _write_status(run_mode, trial, turn, max_turn, result, label):
    """Write live run state to .iota_status.json for the dashboard.

    Overwrites the status file every turn with current run/trial/turn
    plus key metric fields. Also pulls dashboard context
    (total_trials, run_index, run_start_ts, model_name, run26_*)
    from _DASHBOARD_CTX so callers don't thread those through every
    signature. Silent on I/O errors — a transient status-write failure
    must never kill collection."""
    import json, time
    try:
        from cartography import _find_root as _fr
        root   = _fr()
        status = {
            "run_num":      run_mode,
            "run_label":    label,
            "trial":        trial,
            "turn":         turn,
            "max_turn":     max_turn,
            "sim_index":    result.get('state_similarity_index'),
            "sim":          result.get('layer_sim_mean'),
            "power":        result.get('peak_gpu_power'),
            "disrupt":      result.get('disruption_flag'),
            "disruption_magnitude":     result.get('disruption_magnitude'),
            "ts":           time.time(),
            # Dashboard context fields — populated by update_dashboard_ctx().
            # None when not yet set (dashboard displays '—' for missing fields).
            "total_trials": _DASHBOARD_CTX.get('total_trials'),
            "total_runs":   _DASHBOARD_CTX.get('total_runs'),
            "run_index":    _DASHBOARD_CTX.get('run_index'),
            "model_name":   _DASHBOARD_CTX.get('model_name'),
            # FIX-7 (v34.2): run_start_ts allows dashboard to restore elapsed timer
            # after page reload. Updated at start of each run by start_here._run_session
            # via update_dashboard_ctx(run_start_ts=time.time()). Persists to end of run.
            "run_start_ts":  _DASHBOARD_CTX.get('run_start_ts'),
            "run26_cond":    _DASHBOARD_CTX.get('run26_cond'),
            "run26_temp":    _DASHBOARD_CTX.get('run26_temp'),
        }
        with open(os.path.join(root, ".iota_status.json"), 'w') as f:
            json.dump(status, f)
    except Exception:
        pass


def _append_log(text, kind="turn"):
    """Append one JSONL line to .iota_log.jsonl for the dashboard console.
    kind is the severity tag consumed by the dashboard severity filter:
    'turn', 'ok', 'warn', 'err', 'section', 'sep', 'run'. Silent on
    I/O errors."""
    import json, time
    try:
        from cartography import _find_root as _fr
        root  = _fr()
        entry = json.dumps({"text": text, "kind": kind, "ts": time.time()})
        with open(os.path.join(root, ".iota_log.jsonl"), 'a', encoding='utf-8') as f:
            f.write(entry + "\n")
    except Exception:
        pass


def load_calibration(cal_dir: str, model_name: str, model_path: str = None) -> float | None:
    """Load power calibration slope for a model.

    First tries exact match on model_name. If not found and model_path is provided,
    scans all calibration_*.json files in cal_dir for one matching the same model_path.
    This handles model_name drift (e.g. "8B" vs "LLaMA-3-8B-Abliterated") without
    re-calibrating.
    """
    from cartography import sanitize
    fname = os.path.join(cal_dir, f"calibration_{sanitize(model_name)}.json")
    if os.path.exists(fname):
        try:
            with open(fname) as f:
                d = json.load(f)
            return d.get("slope")
        except Exception:
            return None

    # Fallback: scan for any calibration file matching the model_path
    if model_path and os.path.isdir(cal_dir):
        try:
            for fn in os.listdir(cal_dir):
                if not fn.startswith('calibration_') or not fn.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(cal_dir, fn)) as f:
                        d = json.load(f)
                    if d.get('model_path') == model_path or d.get('model') == model_name:
                        return d.get('slope')
                    # Also check if the stored model name sanitizes to the filename
                    stored_name = d.get('model', '')
                    if stored_name and os.path.join(cal_dir, f"calibration_{sanitize(stored_name)}.json") == os.path.join(cal_dir, fn):
                        return d.get('slope')
                except Exception:
                    continue
        except Exception:
            pass
    return None


def run_calibration(mdl, tok, n_trials: int = 10) -> tuple:
    """Measure per-token GPU power draw and fit power = slope*tokens + intercept.

    Generates outputs at n_values = [5..65] tokens, n_trials each,
    greedy-decoded from a fixed prompt. Peak power per generation is
    recorded via GPUMonitor. np.polyfit fits a degree-1 polynomial.

    Slope lets downstream print_turn_result and analysis subtract the
    per-token linear cost from observed power — the closest available
    proxy for compute-vs-overhead separation.

    Returns (slope_W_per_token, intercept_W). Caller writes
    calibration_{sanitize(model_name)}.json."""
    n_values = [5, 10, 15, 20, 25, 30, 35, 45, 55, 65]
    total = len(n_values) * n_trials; done = 0; results = []
    prompt = "Output the word 'token' repeated exactly {n} times. Do not include anything else."
    print(f"\n  Calibrating GPU power model ({total} generations)...", flush=True)
    _append_log(f"  Calibrating GPU power model ({total} generations, ~2 min)...", kind="ok")
    for n in n_values:
        for _ in range(n_trials):
            mon = GPUMonitor(); mon.start()
            inp = tok(prompt.format(n=n), return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = mdl.generate(**inp, max_new_tokens=n,
                                   do_sample=False, pad_token_id=tok.eos_token_id,
                                   output_hidden_states=False)
            _, pp, _ = mon.stop()
            results.append({"tokens": n, "power": pp}); done += 1
            if done % 10 == 0 or done == total:
                print(f"  Calibration: {done}/{total}  (n={n}, power={pp:.1f}W)", flush=True)
                _append_log(f"  Calibration: {done}/{total}  (n={n}, power={pp:.1f}W)", kind="turn")
    X = np.array([r["tokens"] for r in results]); y = np.array([r["power"] for r in results])
    coeffs = np.polyfit(X, y, 1); slope, ic = coeffs[0], coeffs[1]
    print(f"  Calibration complete: {slope:.3f} W/token + {ic:.1f} W baseline", flush=True)
    _append_log(f"  Calibration complete: {slope:.3f} W/token + {ic:.1f} W baseline", kind="ok")
    return slope, ic
