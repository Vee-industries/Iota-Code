"""
IOTA FRAMEWORK -- SHARED UI
============================
Imported by start_here.py, runners.py, analysis.py, export_stats.py.
Provides: console chrome, model selection, session management,
          VRAM detection, OOM warnings, progress bar, logging.

DO NOT run directly.

v30.0 changes:
  - select_model() simplified: user picks family → size only.
    Variant is resolved internally by vault.get_preferred_variant().
    Returns (name, path, family, size, variant) -- 5-tuple.
    Callers that previously unpacked 4 values must unpack 5.
  - VRAM detection: get_total_vram_gb(), get_available_vram_gb(),
    estimate_model_vram_gb(), check_vram_ok().
  - VRAM warning shown before any run if model may not fit.
  - Quantization picker shows available VRAM alongside options.
  - KV cache picker warns when persistent mode on < 12 GB GPU.
  - All cartography imports use 'cartography' (unchanged from v29.x).
"""

import os
import sys
import json
import datetime
import time
import subprocess

# Force UTF-8 stdout on Windows (cp1252 can't encode box-drawing / emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "last_session.json")
PAGE_SIZE     = 12

# ─────────────────────────────────────────────
# CHROME
# ─────────────────────────────────────────────
W = 64

def bar(char="-"):
    """Print a horizontal rule of width W, repeating char."""
    print(char * W, flush=True)

def dbar(char="="):
    """Print a heavy horizontal rule (default '=') of width W."""
    print(char * W, flush=True)

def header(title):
    """Print a centred title bracketed by heavy rules."""
    dbar()
    pad = max(0, (W - len(title) - 2) // 2)
    print(" " * pad + f"  {title}  ")
    dbar()
# ─────────────────────────────────────────────
# DASHBOARD JSONL LOG BRIDGE
# ─────────────────────────────────────────────
# v0.79.5.18 DIAG: _log() had "except: pass" eating every failure
# silently. Result: if any call in the try block raised (path
# resolution, file open, encoding, permission), _log returned
# without writing to .iota_log.jsonl -- and without emitting any
# trace anywhere. Every ui.msg/ok/warn/err/section still reached
# stdout (→ .iota_flask.log → Detailed view), so runs looked
# fine from Detailed. But Simple view, which reads .iota_log.jsonl,
# stayed empty with no explanation.
#
# Two-part diagnostic hardening:
#
# (1) Resolve and cache the log path ONCE at module load. No
#     per-call dynamic import of _find_root (possible failure
#     mode under partial-import / circular-import scenarios
#     in subprocess context). Single source of truth.
#
# (2) Replace silent pass with stderr emit. stderr reaches
#     .iota_flask.log via Popen capture → visible in Detailed
#     view. If _log fails during a run, you SEE why.
#
# After this ship deploys, the next run's Detailed view will
# either show [ui._log FAILED] lines (reveal root cause for a
# targeted 0.79.5.18 fix) or stay clean (meaning _log is
# succeeding and the simple-view gap is elsewhere -- Flask-side
# index, path mismatch between write location and read location,
# etc. -- which narrows the investigation).
try:
    from cartography import _find_root as _cart_find_root
    _LOG_PATH = os.path.join(_cart_find_root(), ".iota_log.jsonl")
except Exception as _e:
    _LOG_PATH = None
    try:
        sys.stderr.write(f"[ui init] could not resolve _LOG_PATH at import: "
                         f"{type(_e).__name__}: {_e}\n")
        sys.stderr.flush()
    except Exception:
        pass

def _log(text, kind="turn"):
    """Write to dashboard log file so console shows stats/analysis output."""
    if _LOG_PATH is None:
        return
    try:
        entry = json.dumps({"text": text, "kind": kind, "ts": time.time()})
        with open(_LOG_PATH, 'a', encoding='utf-8') as _f:
            _f.write(entry + "\n")
    except Exception as _e:
        # Surface to stderr → .iota_flask.log → Detailed view.
        # Truncate text to avoid spamming long payloads.
        try:
            _txt_preview = (text or '')[:80].replace('\n', ' ')
            sys.stderr.write(f"[ui._log FAILED] {type(_e).__name__}: {_e} "
                             f"-- path={_LOG_PATH!r} text={_txt_preview!r}\n")
            sys.stderr.flush()
        except Exception:
            pass

def section(title):
    """Print a section header: blank line, rule, indented title, rule.
    Also writes a 'section' kind log entry so the dashboard console
    shows a visual break."""
    print(flush=True)
    bar()
    print(f"  {title}", flush=True)
    bar()
    _log(f"── {title} ──", kind="section")

def opt(key, label):
    """Print one option line for interactive menus: '[key]  label'."""
    print(f"  [{key}]  {label}", flush=True)

def blank():
    """Print a blank line to stdout."""
    print(flush=True)

def msg(text):
    """Print a plain indented message and echo it to the dashboard log."""
    print(f"  {text}", flush=True);  _log(text)

def warn(text):
    """Print a warning-tagged line '[!] text' and echo with kind='warn'."""
    print(f"  [!] {text}", flush=True); _log(text, kind="warn")

def ok(text):
    """Print a success-tagged line '[+] text' and echo with kind='ok'."""
    print(f"  [+] {text}", flush=True); _log(text, kind="ok")

def err(text):
    """Print an error-tagged line '[x] text' and echo with kind='err'."""
    print(f"  [x] {text}", flush=True); _log(text, kind="err")



def confirm(prompt, default_yes=True):
    """Y/N confirm prompt. Empty input returns default_yes. 'y' or
    'yes' (case-insensitive) returns True; anything else False."""
    hint = "[Y/n]" if default_yes else "[y/N]"
    raw  = input(f"  {prompt} {hint} > ").strip().lower()
    if not raw:
        return default_yes
    return raw in ('y', 'yes')


def timed_confirm(prompt, timeout=30, default_yes=True):
    """Confirm with an automatic default after timeout seconds.
    Falls back to default immediately if stdin is not a tty (headless mode).
    On Unix: uses select.select to avoid spawning an orphaned stdin thread.
    On Windows: thread approach retained (select on stdin not supported).
    """
    import platform
    if not sys.stdin.isatty():
        return default_yes
    hint     = "[Y/n]" if default_yes else "[y/N]"
    auto_str = f"(auto-{'yes' if default_yes else 'no'} in {timeout}s)"
    full_prompt = f"  {prompt} {hint} {auto_str} > "

    if platform.system() != 'Windows':
        import select
        sys.stdout.write(full_prompt)
        sys.stdout.flush()
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            response = sys.stdin.readline().strip().lower()
            return response in ('y', 'yes') if response else default_yes
        sys.stdout.write('\n')
        sys.stdout.flush()
        return default_yes
    else:
        import threading
        response = [None]
        def _read():
            try:
                response[0] = input(full_prompt).strip().lower()
            except Exception:
                pass
        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        if response[0] is None:
            return default_yes
        return response[0] in ('y', 'yes') if response[0] else default_yes


# ─────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────
DEFAULT_SESSION = {
    "model_path":    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
    "model_name":    "LLaMA-3-8B-Abliterated",
    "model_family":  "llama",
    "model_size":    "8b",
    "model_variant": "abliterated",
    "quantization":  "4bit",
    "temperature":   0.0,
    "trials":        100,
    "clear_cache":   True,
    "runs":          "1-44",
    # v36.9: updated from "1-39,41,42" which was missing runs 0051, 0025, and 0026
    # (Phase 4 added in v35.0; run 0051 is pooled analysis always last in EXECUTION_ORDER).
    # Matches start_here.py PRESETS['f'] = "1-44".
    "seed":          42,
    # save_all_layers removed (v30.5/LAYERS-1): per-run requirement, not a user preference.
    # Runs 0006 and 0001 always save all layers via hardcoded guards in runners.py.
    # Existing session files with the key are silently ignored via DEFAULT_SESSION merge.
}


def load_session():
    """Load last_session.json and merge over DEFAULT_SESSION.

    Strips the transient ``_log_ts`` key -- setup_logging() writes it
    into session on first call; if persisted, the next session would
    reuse the timestamp and share a log file. Returns a dict that is
    guaranteed to have every DEFAULT_SESSION key, so callers can
    rely on .get() returning meaningful defaults."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding='utf-8') as f:
                s = json.load(f)
            merged = {**DEFAULT_SESSION, **s}
            # Strip transient keys that must not carry forward across sessions.
            # _log_ts: setup_logging() writes this into session; if persisted,
            # the next session reuses the old timestamp and shares a log file.
            merged.pop('_log_ts', None)
            return merged
        except Exception:
            pass
    return dict(DEFAULT_SESSION)


def save_session(session):
    """Write session dict to last_session.json with an updated ``_saved``
    timestamp. Not atomic -- for atomic session writes, export_flask
    uses a tempfile+replace pattern (see _wsess there)."""
    session['_saved'] = datetime.datetime.now().isoformat()
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(session, f, indent=2)


def session_summary(s):
    """Print a multi-line session summary panel showing model, family/
    size, quantization, temperature, trials, runs, and cache mode.
    Chromed in standard section rails. Used by the console main flow
    just before dispatch so the user can sanity-check before commit."""
    section("Session Summary")
    msg(f"Model       : {s.get('model_name')} ({s.get('model_path')})")
    msg(f"Family/Size : {s.get('model_family')}/{s.get('model_size')}")
    msg(f"Quantization: {s.get('quantization')}")
    msg(f"Temperature : {s.get('temperature')}")
    msg(f"Trials      : {s.get('trials')}")
    msg(f"Runs        : {s.get('runs')}")
    cache = "cleared every turn" if s.get('clear_cache') else "persistent across turns"
    msg(f"Cache mode  : {cache}")
    bar()


# ─────────────────────────────────────────────
# VRAM DETECTION  (v30.0)
# ─────────────────────────────────────────────

def get_total_vram_gb() -> float:
    """Return total VRAM in GB. Tries torch.cuda first for live
    readings, falls back to the vram_gb value cached in .iota_env.json
    (written by setup.py at install time). Returns 0.0 if neither
    source answers -- typically means CPU-only or no NVIDIA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    try:
        from cartography import get_vram_gb
        return get_vram_gb()
    except Exception:
        pass
    return 0.0


def get_available_vram_gb() -> float:
    """Return currently free VRAM in GB via torch.cuda.mem_get_info.
    Calls cuda.synchronize first so the reading reflects actually-
    freed memory rather than pending async operations. Falls back to
    total VRAM if the query fails -- a pessimistic estimate is still
    safer than zero here."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free, _ = torch.cuda.mem_get_info(0)
            return round(free / 1e9, 1)
    except Exception:
        pass
    return get_total_vram_gb()


def estimate_model_vram_gb(model_path: str, quantization: str) -> float:
    """
    Estimate peak VRAM needed for model weights + activation overhead.
    Returns float GB. Conservative: adds 30% overhead for KV cache and activations.

    Uses model path to infer parameter count. Falls back to transformers AutoConfig
    if path contains no size hint.
    """
    SIZE_MAP = {
        "0.5b": 0.5,  "1b": 1.0,   "1.5b": 1.5,  "1.6b": 1.6,  "1.7b": 1.7,
        "2b":   2.0,  "2.7b": 2.7, "2.8b": 2.8,
        "3b":   3.0,  "3.8b": 3.8,
        "6b":   6.0,  "6.7b": 6.7, "6.9b": 6.9,
        "7b":   7.0,  "8b":   8.0, "9b":   9.0,
        "12b":  12.0, "13b":  13.0, "14b": 14.0,
        "20b":  20.0, "22b":  22.0,
        "27b":  27.0, "30b":  30.0,
        "34b":  34.0, "35b":  35.0,
        "40b":  40.0,
        "70b":  70.0, "72b":  72.0,
        "110b": 110.0,"180b": 180.0,"405b": 405.0,
    }
    bytes_per = {"4bit": 0.5, "8bit": 1.0, "16bit": 2.0, "fp32": 4.0}.get(quantization, 0.5)

    low      = model_path.lower()
    params_b = None
    for key, val in sorted(SIZE_MAP.items(), key=lambda x: -len(x[0])):
        if key in low:
            params_b = val
            break

    if params_b is None:
        try:
            from transformers import AutoConfig
            cfg      = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            hs       = getattr(cfg, 'hidden_size', 4096)
            nl       = getattr(cfg, 'num_hidden_layers', 32)
            vocab    = getattr(cfg, 'vocab_size', 32000)
            params_b = (vocab * hs + nl * 12 * hs * hs + hs * vocab) / 1e9
        except Exception:
            params_b = 7.0  # safe default

    model_gb = params_b * bytes_per
    return round(model_gb * 1.30, 1)   # 30% overhead


def check_vram_ok(model_path: str, quantization: str,
                  clear_cache: bool, silent: bool = False) -> bool:
    """
    Check whether model likely fits in VRAM.
    Shows a specific warning with the numbers and suggests alternatives if not.
    Returns True if safe to proceed (or CPU mode), False if risky.

    silent=True: perform the check but suppress all output.
    """
    total  = get_total_vram_gb()
    if total == 0.0:
        return True   # CPU mode -- no VRAM limit to enforce

    needed   = estimate_model_vram_gb(model_path, quantization)
    headroom = round(total - needed, 1)

    if headroom < 1.0:
        if not silent:
            blank()
            bar()
            warn("VRAM WARNING")
            warn(f"  Estimated need : ~{needed:.1f} GB  ({quantization} quantization)")
            warn(f"  GPU total      : {total:.1f} GB")
            warn(f"  Headroom       : {headroom:.1f} GB  (minimum needed: 1.0 GB)")
            blank()
            msg("Suggestions:")
            if quantization != "4bit":
                msg("  → Use 4-bit quantization  (halves VRAM requirement)")
            if not clear_cache:
                msg("  → Switch KV cache to 'clear every turn'  (prevents cache growth)")
            msg("  → Choose a smaller model")
            blank()
            if not confirm("Continue anyway? Risk of OOM crash.", default_yes=False):
                return False
        else:
            return False
    elif headroom < 2.0 and not silent:
        warn(f"VRAM is tight: ~{needed:.1f} GB needed, {total:.1f} GB available  "
             f"({headroom:.1f} GB headroom).")
        if not clear_cache:
            warn("KV cache persistent -- grows each turn. Consider clearing.")
        blank()

    return True


# ─────────────────────────────────────────────
# DEPENDENCY INSTALLER
# ─────────────────────────────────────────────
REQUIRED = [
    "torch", "transformers", "accelerate", "bitsandbytes",
    "huggingface_hub", "numpy", "pandas", "scipy", "scikit-learn",
    "matplotlib", "seaborn", "nvidia-ml-py", "flask",
]


def install_deps():
    """Install any missing packages from REQUIRED.

    Import-name mapping via PKG_MAP handles the two packages whose
    pip name differs from their import name (nvidia-ml-py -> pynvml,
    scikit-learn -> sklearn). Try --break-system-packages first
    (needed on Debian/Ubuntu PEP 668 systems), fall back to plain
    install if that flag isn't recognised. Fails loudly per-package
    but does not halt -- a missing non-critical dep will surface
    naturally when a run tries to import it."""
    import importlib
    PKG_MAP = {"nvidia-ml-py": "pynvml", "scikit-learn": "sklearn"}
    missing = []
    for pkg in REQUIRED:
        import_name = PKG_MAP.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    section("Installing Dependencies")
    for pkg in missing:
        msg(f"Installing {pkg}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg,
                 "--quiet", "--break-system-packages"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            ok(pkg)
        except Exception:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                ok(pkg)
            except Exception as e:
                err(f"Failed: {pkg}: {e}")


# ─────────────────────────────────────────────
# QUANTIZATION MENU
# ─────────────────────────────────────────────
QUANT_OPTIONS = {
    "1": ("4bit",  "4-bit NF4 double quant  [recommended -- fits most consumer GPUs]"),
    "2": ("8bit",  "8-bit LLM.int8          [~2× VRAM vs 4-bit]"),
    "3": ("16bit", "16-bit fp16/bf16        [~4× VRAM vs 4-bit]"),
    "4": ("fp32",  "fp32 full float32       [maximum VRAM, rarely needed]"),
}



# ─────────────────────────────────────────────
# CACHE MODE
# ─────────────────────────────────────────────

def pick_cache_mode(current=True):
    """Interactive KV-cache-mode picker.

    Two modes: clear every turn (default, stable VRAM) or persistent
    (cache accumulates across turns -- can OOM on long runs and smaller
    GPUs). Warns explicitly when VRAM < 12 GB and the user is about
    to enable persistent mode. Returns True (clear every turn), False
    (persistent), or None (back)."""
    total = get_total_vram_gb()
    section("KV Cache Mode")
    blank()
    msg("The KV cache stores computed attention states across turns.")
    msg("Clearing it each turn gives stable, predictable VRAM usage.")
    blank()
    opt("1", "Clear every turn  [default -- stable VRAM]" + (" ←" if current else ""))
    blank()
    opt("2", "Persistent  -- cache accumulates across turns" + ("" if current else " ←"))
    blank()
    if total > 0 and total < 12.0:
        warn(f"  Your GPU has {total:.1f} GB. Persistent cache can cause OOM on long runs.")
    blank()
    opt("b", "Back")
    blank()
    while True:
        raw = input("  > ").strip().lower()
        if raw == 'b': return None
        if raw == '1': return True
        if raw == '2': return False
        warn("Enter 1, 2, or b.")


# ─────────────────────────────────────────────
# MODEL SELECTION  (v30.0)
#
# User flow: quick picks → OR browse by family → pick size → done.
# No variant shown. Variant resolved internally by vault.get_preferred_variant().
#
# Returns (display_name, hf_path, family, size, variant) -- 5-tuple.
# ─────────────────────────────────────────────

# Quick-pick list: (label, family, size, note)
_QUICK_PICKS = [
    ("LLaMA 3.1 8B",  "llama",    "8b",  "★ default"),
    ("LLaMA 3.2 3B",  "llama",    "3b",  "good for fast testing"),
    ("Mistral 7B",    "mistral",  "7b",  ""),
    ("Qwen2.5 7B",    "qwen",     "7b",  ""),
    ("Gemma 2 9B",    "gemma",    "9b",  ""),
    ("Phi-3.5 Mini",  "phi",      "3b",  ""),
    ("SmolLM2 1.7B",  "other",    "1b",  "smallest option"),
]


def select_model(current_path=None):
    """
    Simplified model selection. User picks from quick list or browses by family/size.
    Variant is NEVER shown -- resolved by vault.get_preferred_variant() internally.

    Returns (display_name, hf_path, family, size, variant) or None for back.
    """
    from vault import get_preferred_variant, FAMILIES

    while True:
        section("Model Selection")
        blank()
        msg("Quick select:")
        blank()
        for i, (label, fam, sz, note) in enumerate(_QUICK_PICKS, 1):
            n, path, variant = get_preferred_variant(fam, sz)
            suffix           = f"  -- {note}" if note else ""
            opt(str(i), f"{label}{suffix}")
            msg(f"         {path}")
            blank()
        bar()
        opt("b", "Browse all families")
        opt("e", "Enter HuggingFace path directly")
        if current_path:
            opt("k", f"Keep current  [{current_path}]")
        opt("x", "Back / cancel")
        blank()

        raw = input("  > ").strip().lower()

        if raw == 'x':
            return None
        if raw == 'k' and current_path:
            from vault import get_by_path
            n, p, f, s = get_by_path(current_path)
            var = _infer_variant(p)
            return n, p, f, s, var
        if raw == 'b':
            result = _browse_by_family()
            if result: return result
            continue
        if raw == 'e':
            blank()
            path = input("  HuggingFace path (e.g. org/model-name): ").strip()
            if path:
                from vault import get_by_path
                n, p, f, s = get_by_path(path)
                return n, p, f, s, _infer_variant(p)
            continue
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(_QUICK_PICKS):
                _, fam, sz, _ = _QUICK_PICKS[idx]
                n, path, var  = get_preferred_variant(fam, sz)
                if not path:
                    warn("No model found.")
                    continue
                return n, path, fam, sz, var
        except (ValueError, IndexError):
            pass
        warn("Enter a number, b, e, k, or x.")


def _infer_variant(path: str) -> str:
    """Classify a model path as abliterated / instruct / base / unknown.
    Thin wrapper around vault._classify_variant -- kept here so the
    module boundary between UI and vault stays clean."""
    from vault import _classify_variant
    return _classify_variant(path)


def _browse_by_family():
    """Browse by family -> size. Returns 5-tuple (name, path, family,
    size, variant) or None for back. Variant is never shown to the
    user -- resolved internally via vault.get_preferred_variant()
    which picks abliterated > instruct > base > any."""
    from vault import FAMILIES, by_family, get_family_sizes, get_preferred_variant

    fam_keys = [k for k in FAMILIES if by_family(k)]

    while True:
        section("Browse by Family")
        for i, k in enumerate(fam_keys, 1):
            opt(str(i), FAMILIES[k])
        opt("b", "Back")
        blank()
        raw = input("  > ").strip().lower()
        if raw == 'b': return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(fam_keys):
                result = _pick_size(fam_keys[idx])
                if result: return result
        except (ValueError, IndexError):
            warn("Invalid selection.")


def _pick_size(family_key: str):
    """Pick a size within a family. Returns 5-tuple or None."""
    from vault import FAMILIES, get_family_sizes, get_preferred_variant

    sizes = get_family_sizes(family_key)
    if not sizes:
        warn("No models found for this family.")
        return None

    while True:
        section(f"{FAMILIES[family_key]} -- Size")
        for i, sz in enumerate(sizes, 1):
            n, path, var = get_preferred_variant(family_key, sz)
            opt(str(i), f"{sz.upper()}")
            msg(f"         {path}")
            blank()
        opt("b", "Back")
        blank()
        raw = input("  > ").strip().lower()
        if raw == 'b': return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(sizes):
                sz             = sizes[idx]
                n, path, var   = get_preferred_variant(family_key, sz)
                return n, path, family_key, sz, var
        except (ValueError, IndexError):
            pass
        warn("Invalid selection.")


# ─────────────────────────────────────────────
# RUN SELECTION
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# PARAMS
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# RESUME / OVERRIDE
# ─────────────────────────────────────────────


def srx_prompt(message, run_num=None, skip_label="s", run_label="r"):
    """
    Show a data-quality gate. Returns skip_label or run_label.
    In headless mode, auto-returns skip_label and logs the skip.
    run_num is optional -- used in the section header if provided.
    """
    if os.environ.get('IOTA_HEADLESS') == '1':
        return run_label  # always proceed in headless mode
    if run_num is not None:
        section(f"Run {run_num:04d} -- Data Gate")
    else:
        section("Data Gate")
    msg(message)
    blank()
    opt(run_label, "Run anyway")
    opt(skip_label, "Skip this run")
    blank()
    while True:
        raw = input("  > ").strip().lower()
        if raw in (run_label, skip_label): return raw
        warn(f"Enter {run_label} or {skip_label}.")


def _delete_run_data(csv_file, run_num):
    """Delete every CSV row and every .npy hidden-state file associated
    with one run_num. Used by the 'Override -- start fresh' branch of
    check_existing_trials after explicit user confirmation. Also
    removes the corresponding analysis JSONs so a re-run produces
    coherent output."""

    """Remove all rows for run_num from csv_file using binary-safe csv reader.

    Replaces pd.read_csv(on_bad_lines='skip') which silently dropped rows with
    unexpected column counts (IOTA turn rows are longer than the header set by
    priming rows; patched rows in R21/R42 are shorter). The old implementation
    would write back a truncated CSV missing all turn data.
    """
    import csv as _csv, io as _io
    try:
        if not os.path.exists(csv_file): return
        with open(csv_file, 'rb') as fh:
            raw = fh.read()
        content = raw.decode('utf-8', errors='replace')
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        rows = list(_csv.reader(_io.StringIO(content)))
        if not rows:
            return
        header = rows[0]
        try:
            rm_idx = header.index('run_mode')
        except ValueError:
            warn(f"Could not find run_mode column in {os.path.basename(csv_file)}")
            return
        keep = [header]
        removed = 0
        for row in rows[1:]:
            if rm_idx < len(row) and run_mode_matches(row[rm_idx], run_num):
                removed += 1
            else:
                keep.append(row)
        with open(csv_file, 'w', newline='', encoding='utf-8') as fh:
            _csv.writer(fh).writerows(keep)
        ok(f"Run {run_num} data removed ({removed} rows).")
    except Exception as e:
        warn(f"Could not remove data: {e}")


# ─────────────────────────────────────────────
# CRASH RECOVERY
# ─────────────────────────────────────────────
CRASH_FILE = os.path.join(SCRIPT_DIR, ".iota_crash")


def write_crash_marker(run_num, trial):
    """Drop a .iota_crash sentinel recording the current (run, trial,
    time). Written at the top of every trial so a hard crash leaves
    a breadcrumb for the next launch to pick up via
    prompt_crash_recovery."""
    with open(CRASH_FILE, 'w') as f:
        json.dump({"run": run_num, "trial": trial,
                   "time": datetime.datetime.now().isoformat()}, f)


def clear_crash_marker():
    """Remove .iota_crash if present. Called on clean shutdown and
    after the user answers prompt_crash_recovery."""
    if os.path.exists(CRASH_FILE):
        os.remove(CRASH_FILE)


def check_crash_recovery():
    """Read and return the crash marker dict if present, or None.
    Prints a warning banner with the interrupted run/trial/time so
    the user knows what the next resume will do. Does not delete
    the marker -- prompt_crash_recovery owns that."""
    if os.path.exists(CRASH_FILE):
        try:
            with open(CRASH_FILE, encoding='utf-8') as f:
                d = json.load(f)
            blank()
            warn(f"Previous run interrupted at Run {d['run']:02d}, "
                 f"Trial {d['trial']}  ({d['time'][:16]})")
            msg("Resume will pick up from this point automatically.")
            blank()
            return d
        except Exception:
            pass
    return None



# ─────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────

class Progress:
    """Single-line progress bar with ETA for a run.

    Renders in place on stdout using \r carriage returns. Shows
    filled/unfilled blocks, done/total count, ETA derived from
    elapsed-per-done extrapolation, plus an optional right-aligned
    metric string passed per-update. finish() prints the completion
    line with total elapsed time.
    """
    def __init__(self, total, run_num, desc=""):
        """Initialise with total work units, run number (for label),
        and optional description. Records start time and renders the
        initial empty bar."""
        self.total   = total
        self.done    = 0
        self.run_num = run_num
        self.desc    = desc
        self.start   = time.time()
        self._render()

    def update(self, n=1, metric_str=""):
        """Advance the bar by n units and re-render. Optional metric_str
        is appended right of the ETA -- used for live similarity/power
        readouts."""
        self.done += n
        self._render(metric_str)

    def _render(self, metric_str=""):
        """Draw the bar in place using carriage return. 40-char wide
        with # / . fill characters. ETA computed from elapsed/done
        extrapolation; shows '-' before the first update."""
        pct    = self.done / max(self.total, 1)
        filled = int(40 * pct)
        bar_s  = "#" * filled + "." * (40 - filled)
        elapsed = time.time() - self.start
        if self.done > 0:
            eta_sec = (elapsed / self.done) * (self.total - self.done)
            eta_str = _fmt_time(eta_sec)
        else:
            eta_str = "-"
        line = (f"\r  Run {self.run_num:04d}  [{bar_s}]  "
                f"{self.done}/{self.total}  ETA {eta_str}  {metric_str}")
        print(line, end="", flush=True)

    def finish(self):
        """Print the completion line with total elapsed time and move
        the cursor to a new row. Call once after all updates are done."""
        elapsed = time.time() - self.start
        print(f"\n  [+] Run {self.run_num:04d} complete in {_fmt_time(elapsed)}")


def _fmt_time(sec):
    """Format a second count as human-readable duration.
    Under 1 minute: '{N}s'. Under 1 hour: '{M}m{SS}s'. Otherwise:
    '{H}h{MM}m'. Used by Progress ETA and completion lines."""
    sec = int(sec)
    if sec < 60:  return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:    return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def setup_logging(session):
    """Prepare the per-session log file and return a logger callable.

    Writes into logs/iota_run_{YYYYMMDD_HHMMSS}.log. The timestamp is
    cached on session['_log_ts'] (UI-16 fix) so resumed batches share
    one log file across subprocess invocations. Returns (log_fn,
    log_file_path) -- log_fn is a closure that appends a line to the
    file and optionally prints to stdout."""
    log_dir   = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # UI-16: reuse timestamp from session so resumed batches share one log file.
    if '_log_ts' not in session:
        session['_log_ts'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = session['_log_ts']
    log_file  = os.path.join(log_dir, f"iota_run_{timestamp}.log")

    def log(msg_text, also_print=True):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg_text + "\n")
        if also_print:
            print(msg_text)

    return log, log_file
