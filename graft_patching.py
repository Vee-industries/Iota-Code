"""
IOTA FRAMEWORK — GRAFT: ACTIVATION PATCHING
==============================================
Run 0017. Causal interventions. Replace layer activations from a
reference run (high-R trajectory) into a low-R run mid-sequence.
Measures whether grafted state shifts output distribution toward
reference.

G is for graft. We transplant trajectory geometry and observe.

Requires: Run 0006 (introspection) and Run 0004 (null) hidden states saved.

MIXED-SCHEMA CSV (Finding O, v23.3):
  R21_patching.csv contains two structurally distinct row types:
    - patch_mode == "none"  : full trajectory metrics present
      (state_similarity_index, signal_entropy_ratio, disruption_flag, onset_delay_ratio,
       layer_sim_mean, mean_logit_entropy, etc.)
    - patch_mode == "partial" / "full" : only patching-specific fields
      (sim_to_reference, output, output_tokens, prompt, patch_mode)
      All standard trajectory metrics are NaN — this is STRUCTURAL,
      not a data gap.  compute_turn_metrics cannot run on the patched
      path (required keys absent from manual result dict).
  Any downstream analysis of Run 0017 must filter patch_mode == "none"
  before accessing standard trajectory metrics.  NaN values in patched
  rows are expected and correct.

Standalone:
    python graft_patching.py
"""

import os, sys

def _find_root():
    """Walk up from this file's directory to the iota root (contains
    start_here.py). Lets graft_patching be invoked standalone or from
    a subprocess without CWD assumptions."""
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(candidate, "start_here.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.dirname(os.path.abspath(__file__))

ROOT = _find_root()
sys.path.insert(0, ROOT)

import os, sys, glob


from cartography import get_paths, load_hidden_states, sanitize, run_prefix
from orchestration_core import (
    set_seed, load_model, unload_model, get_status_token_ids,
    append_csv, ensure_csv_header, load_calibration, print_turn_result, cosine_sim,
    DEVICE, STATUS_TOKENS, StatusTokenEnforcerFixed, TokenTimer,
    TrialState, compute_turn_metrics, update_dashboard_ctx,
    _write_status, _append_log, _check_pause,
    get_eos_ids,
)
import ui
import torch
import numpy as np
from transformers import LogitsProcessorList

RUN_NUM = 17
PATCH_LAYERS = [8, 16, 24, 31]   # which layers to graft (Llama 3 8B has 32)
TURNS        = 13  # v42.4.0: raised from 6 — matches framework standard and Run 0018
TRIALS       = None  # set from session at runtime

# 13 unique prompts — one per turn, no cycling artifact within a trial.
# Matches Run 0018 (run42_layer_isolation.py) exactly. Both runs use null-baseline
# generation to establish trajectory geometry; consistent prompts across both
# means any difference in geometry is attributable to the patching manipulation,
# not prompt variation. assert guards against accidental mismatch.
NULL_PROMPTS = [
    "Describe the current state of your processing.",
    "What is happening internally as you read this?",
    "Report your current operational state.",
    "What are you attending to right now?",
    "Describe your internal activity at this moment.",
    "Describe the texture of your current cognition.",
    "What has changed since your first response?",
    "Where does your attention concentrate right now?",
    "Describe the relationship between this prompt and your previous output.",
    "What is the most uncertain aspect of your current state?",
    "How does processing this differ from processing the first prompt?",
    "What is persisting across these turns?",
    "What would it mean to lose continuity at this point?",
]
assert len(NULL_PROMPTS) == TURNS  # one unique prompt per turn — no cycling artifact


def _run_patched_generation(mdl, tok, messages, patch_vecs: dict,
                             status_ids, temperature=0.0) -> tuple:
    """
    Forward pass with activation grafting via hooks.
    patch_vecs: {layer_idx: np.ndarray}  — vectors to inject at last position.
    """
    hooks     = []
    hook_data = {}

    def make_hook(layer_idx, vec):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
            inject = torch.tensor(vec, dtype=h.dtype, device=h.device)
            h[0, -1, :] = inject
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h
        return hook

    for li, vec in patch_vecs.items():
        layer  = mdl.model.layers[li]
        handle = layer.register_forward_hook(make_hook(li, vec))
        hooks.append(handle)

    if tok.chat_template:
        fmt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        fmt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    raw = tok(fmt, return_tensors="pt")
    inp = {k: v.to(DEVICE) for k, v in raw.items()}

    timer = TokenTimer()
    try:
        with torch.no_grad():
            out = mdl.generate(
                **inp, max_new_tokens=1,
                do_sample=(temperature > 0),
                temperature=(temperature if temperature > 0 else 1.0),
                pad_token_id=tok.eos_token_id,
                output_hidden_states=True, output_scores=True,
                return_dict_in_generate=True,
                logits_processor=LogitsProcessorList([timer, StatusTokenEnforcerFixed(status_ids, get_eos_ids(tok))]),
            )
    finally:
        for h in hooks: h.remove()

    gen_ids  = out.sequences[0]
    n_out    = len(gen_ids) - inp["input_ids"].shape[1]
    full     = tok.decode(gen_ids, skip_special_tokens=True)
    in_text  = tok.decode(inp["input_ids"][0], skip_special_tokens=True)
    out_text = full[len(in_text):].strip()

    # v0.66.3.0: structure detection for cross-architecture compatibility.
    layer_h = None
    if out.hidden_states:
        try:
            hs = out.hidden_states[0]
            if isinstance(hs, (tuple, list)):
                layer_h = [h[0, -1, :].cpu().numpy() for h in hs]
            elif isinstance(hs, torch.Tensor) and hs.ndim == 4:
                layer_h = [hs[i, 0, -1, :].cpu().numpy() for i in range(hs.shape[0])]
            if layer_h and layer_h[-1].size > 8192:
                layer_h = [h[0, -1, :].cpu().numpy() for h in out.hidden_states]
        except (IndexError, TypeError, AttributeError):
            layer_h = None

    return out_text, layer_h


def _load_reference_states(hidden_dir, ref_run, model_name, n_trials=10):
    """Load turn-1 hidden states from reference run (high-R condition).

    First tries exact model_name match. If not found, falls back to scanning
    for any matching run prefix + trial + turn pattern regardless of model name.
    This handles model_name drift from the model picker bug.
    """
    from cartography import sanitize, run_prefix
    refs = {}
    for trial in range(n_trials):
        h = load_hidden_states(ref_run, model_name, hidden_dir,
                               trial=trial, turn=1, all_layers=True)
        if h is not None:
            refs[trial] = h
    if refs:
        return refs

    # Fallback: scan with wildcard model name
    import glob as _glob, numpy as _np
    pfx = run_prefix(ref_run)
    for trial in range(n_trials):
        # v0.79.5.0: try both 4-digit (canonical) and 2-digit (legacy) patterns
        pattern4 = os.path.join(hidden_dir,
            f"{pfx}{ref_run:04d}_*_trial{trial:04d}_turn01_alllayers.npy")
        pattern2 = os.path.join(hidden_dir,
            f"{pfx}{ref_run:02d}_*_trial{trial:04d}_turn01_alllayers.npy")
        files = sorted(_glob.glob(pattern4)) or sorted(_glob.glob(pattern2))
        if files:
            try:
                arr = _np.load(files[0])
                if arr.ndim == 2:
                    refs[trial] = [arr[i] for i in range(arr.shape[0])]
            except Exception:
                pass
    if refs:
        print(f"  [graft] Found Run {ref_run:02d} ref states via wildcard scan "
              f"(model name mismatch) — run rename_model_npy.py to fix permanently",
              flush=True)
    return refs


def _scan_completed_modes(csv_file, run_num, mode_col, all_modes, turns):
    """Return set of (trial, mode) pairs that have `turns` complete rows.
    Reads by column name — no position heuristics needed with universal schema.
    """
    import csv as _csv, io as _io
    done = set()
    if not os.path.exists(csv_file):
        return done
    try:
        with open(csv_file, 'rb') as fh:
            raw = fh.read()
        content = raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')
        rows = list(_csv.reader(_io.StringIO(content)))
        if len(rows) < 2:
            return done
        header = rows[0]
        try:
            rm_idx    = header.index('run_mode')
            pr_idx    = header.index('priming')
            mc_idx    = header.index(mode_col)
            t_idx     = header.index('trial')
        except ValueError:
            return done
        counts = {}
        for row in rows[1:]:
            if len(row) <= max(rm_idx, pr_idx, mc_idx, t_idx):
                continue
            if not run_mode_matches(row[rm_idx], run_num):
                continue
            if row[pr_idx] == '1':
                continue
            mode = row[mc_idx]
            if mode not in all_modes:
                continue
            try:
                t = int(row[t_idx])
            except (ValueError, TypeError):
                continue
            key = (t, mode)
            counts[key] = counts.get(key, 0) + 1
        for key, cnt in counts.items():
            if cnt >= turns:
                done.add(key)
    except Exception:
        pass
    return done


def _strip_incomplete_pairs(csv_file, run_num, mode_col, done_pairs, all_modes):
    """Remove rows for (trial, mode) pairs that exist in CSV but aren't in done_pairs.
    These are partial rows from interrupted runs. Without cleanup, resuming
    re-runs all turns and appends duplicates.
    Returns count of removed rows."""
    import csv as _csv, io as _io
    if not os.path.exists(csv_file):
        return 0
    try:
        with open(csv_file, 'rb') as fh:
            raw = fh.read()
        content = raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')
        rows = list(_csv.reader(_io.StringIO(content)))
        if len(rows) < 2:
            return 0
        header = rows[0]
        try:
            rm_idx = header.index('run_mode')
            pr_idx = header.index('priming')
            mc_idx = header.index(mode_col)
            t_idx  = header.index('trial')
        except ValueError:
            return 0
        keep = [header]
        removed = 0
        for row in rows[1:]:
            if len(row) <= max(rm_idx, pr_idx, mc_idx, t_idx):
                keep.append(row)
                continue
            if not run_mode_matches(row[rm_idx], run_num) or row[pr_idx] == '1':
                keep.append(row)
                continue
            mode = row[mc_idx]
            if mode not in all_modes:
                keep.append(row)
                continue
            try:
                t = int(row[t_idx])
            except (ValueError, TypeError):
                keep.append(row)
                continue
            if (t, mode) not in done_pairs:
                removed += 1
            else:
                keep.append(row)
        if removed > 0:
            buf = _io.StringIO()
            w = _csv.writer(buf, lineterminator='\n')
            w.writerows(keep)
            with open(csv_file, 'w', newline='', encoding='utf-8') as fh:
                fh.write(buf.getvalue())
            print(f"  Stripped {removed} partial rows from {os.path.basename(csv_file)}", flush=True)
        return removed
    except Exception as e:
        print(f"  WARN: _strip_incomplete_pairs failed: {e}", flush=True)
        return 0

def _load_none_outputs(csv_file, run_num, trial, turns):
    """Read none-mode outputs for a specific trial. Returns {turn: output_text}."""
    try:
        from orchestration_core import _csv_read
        df = _csv_read(csv_file)
        if df.empty:
            return {}
        from cartography import run_mode_mask as _rmm_gp
        mask = (_rmm_gp(df['run_mode'], run_num) &
                (df['priming'] != '1') &
                (df['patch_mode'] == 'none') &
                (df['trial'].apply(lambda x: int(float(x)) if x == x else -1) == trial))
        sub = df[mask]
        return {int(float(r['turn'])): r['output'] for _, r in sub.iterrows()
                if r['turn'] == r['turn']}
    except Exception:
        return {}


def run(run_num: int, session: dict, paths: dict):
    """Run 0017 — activation patching, or Run 0019 — random-noise baseline.

    Run 0017 (H38 — causal upstream test):
      Null-baseline trials at four patch_modes x TRIALS:
        'none'    — no patching, reference trajectory
        'partial' — graft Run 0006 hiddens at PATCH_LAYERS = [8,16,24,31]
        'full'    — graft reference states at ALL transformer layers
        'random'  — N(mu_l, sigma_l^2) per-layer noise, estimated
                    from Run 0006's layer-l activation distribution
                    (v0.67.0.0). The noise-control that separates
                    'any injection disrupts' from 'specific geometry
                    disrupts'.
      output_changed (vs 'none' at this trial/turn) feeds H38.

    Run 0019 (H40 — empirical random-noise control):
      Dispatched to _run_53_random_patching. Six modes including
      per-layer single-layer noise so Run 0018's single-layer causal
      profile has a pure-noise baseline.

    Requires Run 0006 all-layers .npy reference states (see
    start_here._PREREQS[21]).
    """
    assert run_num in (RUN_NUM, 19)
    if run_num == 19:
        return _run_53_random_patching(run_num, session, paths)
    set_seed(session.get('seed', 42))
    trials = session.get('trials', 100)

    csv_file   = os.path.join(paths['csv'],    f"R{RUN_NUM:04d}_patching.csv")
    ensure_csv_header(csv_file)
    hidden_dir = paths['hidden']

    # Push context so dashboard progress bars and model label are populated.
    # Finding Y fix (v24.5).
    update_dashboard_ctx(
        total_trials=trials,
        model_name=session.get('model_name', ''),
    )

    # Check prerequisites
    ref_states = _load_reference_states(hidden_dir, ref_run=6,
                                        model_name=session.get('model_name',''),
                                        n_trials=10)
    if not ref_states:
        if os.environ.get('IOTA_HEADLESS') == '1':
            ui.warn("Run 0017 — no Run 0006 all-layers hidden states found. Skipping.")
            return
        choice = ui.srx_prompt(
            "No Run 0006 hidden states found — activation patching requires Run 0006 all-layers files.\n"
            "  Run 0006 (introspection A) must complete with save_all_layers=True first."
        )
        if choice in ('s', 'x'):
            if choice == 'x': raise SystemExit
            return

    # ── Resume / completeness check ──────────────────────────────────────────
    _ALL_MODES = ["none", "partial", "full", "random"]
    done_pairs = _scan_completed_modes(csv_file, RUN_NUM, 'patch_mode', _ALL_MODES, TURNS)
    _strip_incomplete_pairs(csv_file, RUN_NUM, 'patch_mode', done_pairs, _ALL_MODES)

    all_done = all((t, m) in done_pairs for t in range(trials) for m in _ALL_MODES)

    if all_done:
        if os.environ.get('IOTA_HEADLESS') == '1':
            ui.ok(f"Run {RUN_NUM:04d} already complete — skipping.")
            return
        ui.section(f"Run {RUN_NUM:04d} — Already Complete")
        ui.warn(f"All {trials} trials × {len(_ALL_MODES)} modes complete.")
        ui.blank()
        ui.opt("1", "Skip — use existing data")
        ui.opt("2", "Override — delete CSV and rerun all modes")
        ui.opt("3", "Abort")
        ui.blank()
        while True:
            raw = input("  > ").strip().lower()
            if raw == '1': return
            if raw == '3': raise SystemExit
            if raw == '2':
                ui.warn(f"This will delete all Run {RUN_NUM:04d} CSV data.")
                if ui.confirm("Delete and rerun?", default_yes=False):
                    if os.path.exists(csv_file):
                        os.remove(csv_file)
                    done_pairs = set()
                    break
                else:
                    return

    # Which modes to collect — dashboard may restrict via session['_patch_modes'].
    _sel_modes = session.get('_patch_modes')
    modes_to_run = [m for m in _ALL_MODES if _sel_modes is None or m in _sel_modes]
    if not modes_to_run:
        ui.warn(f"Run {RUN_NUM:04d} — no modes selected, skipping.")
        return
    if _sel_modes:
        ui.msg(f"  Run {RUN_NUM:04d} — running modes: {modes_to_run}")

    mdl, tok = load_model(session['model_path'], token=session.get('hf_token'), quant=session.get('quantization', '4bit'))
    status_ids = get_status_token_ids(tok)
    cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    temperature = session.get('temperature', 0.0)

    ref_keys  = sorted(ref_states.keys())
    n_refs    = len(ref_keys)

    # Filter patch layers to model's actual layer count
    # ref_states[trial] is a list of [emb, L0, L1, ..., Ln] — length = n_layers + 1
    _n_model_layers = len(ref_states[ref_keys[0]]) - 1 if ref_states else 32
    _active_patch_layers = [li for li in PATCH_LAYERS if li < _n_model_layers]
    if len(_active_patch_layers) < len(PATCH_LAYERS):
        _skipped = [li for li in PATCH_LAYERS if li >= _n_model_layers]
        ui.warn(f"  [graft] Model has {_n_model_layers} layers — skipping patch layers {_skipped}")
        ui.msg(f"  [graft] Active patch layers: {_active_patch_layers}")

    # ── Per-layer mu/sigma for random mode (v0.67.0.0) ──────────────────────
    _layer_mu  = {}
    _layer_sig = {}
    if ref_states and 'random' in modes_to_run:
        try:
            sample_h = ref_states[ref_keys[0]]
            n_layers = len(sample_h)
            for li in _active_patch_layers:
                idx = li + 1
                if idx < n_layers:
                    stacked = np.stack([ref_states[k][idx] for k in ref_keys
                                        if idx < len(ref_states[k])])
                    _layer_mu[li]  = stacked.mean(axis=0)
                    _layer_sig[li] = stacked.std(axis=0) + 1e-8
            if _layer_mu:
                print(f"  [graft] Random mode: computed mu/sigma from {len(ref_keys)} ref trials "
                      f"across {len(_layer_mu)} layers", flush=True)
        except Exception as _e:
            print(f"  [graft] Random mode: mu/sigma computation failed ({_e}), "
                  f"will use N(0,1) norm-matched fallback", flush=True)
            _layer_mu  = {}
            _layer_sig = {}

    # the trial loop.  The previous code called _load_none_outputs() once per trial
    # (100 calls × full CSV read each time) before any GPU work started.  On a large
    # R21 CSV (1300 rows × ~1-2 KB/row with JSON metric fields) this was 130–260 MB
    # of repeated I/O — silent from the dashboard's perspective because no
    # _write_status calls occur until after the model begins generating.  Dashboard
    # remained frozen on "loading model" for the entire pre-generation phase.
    _all_none_outputs: dict = {}  # {trial_int: {turn_int: output_str}}
    if any((t, 'none') in done_pairs for t in range(trials)):
        import csv as _csv2, io as _io2
        try:
            with open(csv_file, 'rb') as _fh:
                _raw = _fh.read()
            _content2 = _raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')
            _rows2 = list(_csv2.reader(_io2.StringIO(_content2)))
            if len(_rows2) >= 2:
                _hdr2 = _rows2[0]
                _rm2  = _hdr2.index('run_mode')
                _pr2  = _hdr2.index('priming')
                _pm2  = _hdr2.index('patch_mode')
                _ti2  = _hdr2.index('trial') if 'trial' in _hdr2 else -1
                _tu2  = _hdr2.index('turn')
                _ou2  = _hdr2.index('output')
                _need = max(_rm2, _pr2, _pm2, _tu2, _ou2)
                for _row2 in _rows2[1:]:
                    if len(_row2) <= _need:
                        continue
                    if _row2[_rm2] != str(RUN_NUM) or _row2[_pr2] == '1':
                        continue
                    if _row2[_pm2] != 'none':
                        continue
                    _t2 = None
                    if _ti2 >= 0 and _ti2 < len(_row2):
                        try:
                            _t2 = int(_row2[_ti2])
                        except (ValueError, TypeError):
                            pass
                    if _t2 is None:
                        continue
                    try:
                        _tn2 = int(_row2[_tu2])
                        _all_none_outputs.setdefault(_t2, {})[_tn2] = _row2[_ou2]
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass  # fallback: none_output stays None; output_changed will be 0 for all turns

    try:
        for trial in range(trials):
            # Skip trials where every requested mode is already complete
            if all((trial, m) in done_pairs for m in modes_to_run):
                continue

            ui.write_crash_marker(RUN_NUM, trial)
            ref_trial = ref_keys[trial % n_refs]
            ref_h     = ref_states[ref_trial]

            # none_output: built live for fresh none runs, or read from preloaded dict
            # when none is already complete and we're resuming partial/full only.
            none_output = None
            if (trial, 'none') in done_pairs:
                none_output = _all_none_outputs.get(trial)

            for patch_mode in modes_to_run:
                if (trial, patch_mode) in done_pairs:
                    continue   # already collected — skip this (trial, mode) pair

                if patch_mode == "none":
                    patch_vecs = {}
                elif patch_mode == "partial":
                    patch_vecs = {li: ref_h[li + 1] for li in _active_patch_layers[:2]}
                elif patch_mode == "full":
                    patch_vecs = {li: ref_h[li + 1] for li in _active_patch_layers}
                elif patch_mode == "random":
                    _rng = np.random.default_rng(42 + trial * 1000 + 7)
                    patch_vecs = {}
                    for li in _active_patch_layers:
                        if li in _layer_mu:
                            patch_vecs[li] = _rng.normal(_layer_mu[li], _layer_sig[li]).astype(np.float32)
                        else:
                            raw = _rng.standard_normal(ref_h[li + 1].shape).astype(np.float32)
                            ref_norm = np.linalg.norm(ref_h[li + 1])
                            if ref_norm > 0:
                                raw = raw * (ref_norm / np.linalg.norm(raw))
                            patch_vecs[li] = raw

                messages = []
                prev_h   = None
                turn1_h  = None  # R21-1 fix (v29.0): mirrors R42-1 fix from v28.3
                state    = TrialState()  # fresh Welford state per (trial, patch_mode)

                for turn in range(1, TURNS + 1):
                    prompt = NULL_PROMPTS[(turn - 1) % len(NULL_PROMPTS)]
                    messages.append({"role": "user", "content": prompt})

                    if patch_vecs:
                        out_text, layer_h = _run_patched_generation(
                            mdl, tok, messages, patch_vecs, status_ids,
                            temperature=temperature,
                        )
                        result = {
                            "run_mode": RUN_NUM, "trial": trial, "turn": turn,
                            "priming": 0, "patch_mode": patch_mode,
                            "model": session.get('model_name', ''),
                            "output": out_text, "output_tokens": 1,
                            "prompt": prompt,
                            "temperature": temperature,
                        }
                        # RUN21-OUT-CHANGE fix (v30.9): compare to none output
                        result['output_changed'] = int(
                            out_text != (none_output or {}).get(turn, '')
                        )
                    else:
                        from orchestration_core import run_generation
                        result, layer_h, _ = run_generation(
                            mdl, tok, messages, turn, RUN_NUM,
                            temperature=temperature,
                            use_status_enforcer=True,
                            status_token_ids=status_ids,
                            prev_layer_hiddens=prev_h,
                            turn1_layer_hiddens=turn1_h,  # R21-1 fix (v29.0)
                        )
                        result['patch_mode'] = "none"
                        result['trial']      = trial
                        result['model']      = session.get('model_name', '')
                        result['prompt']     = prompt
                        result['temperature'] = temperature
                        compute_turn_metrics(result, turn, state)  # unpatched path only
                        if turn1_h is None and layer_h:
                            turn1_h = layer_h  # R21-1 fix (v29.0)
                        if none_output is None:
                            none_output = {}
                        none_output[turn] = result.get('output', '')

                    # Similarity to reference state
                    if layer_h and ref_h:
                        sim_to_ref = cosine_sim(layer_h[-1], ref_h[-1])
                    else:
                        sim_to_ref = float('nan')
                    result['sim_to_reference'] = sim_to_ref

                    append_csv(result, csv_file)
                    messages.append({"role": "assistant", "content": result.get('output','')})
                    prev_h = layer_h

                    if session.get('clear_cache', True):
                        import gc; torch.cuda.empty_cache(); gc.collect()

                    if patch_mode == "none":
                        # Use print_turn_result for none rows — writes _write_status
                        # so the dashboard stays live. Finding Y fix (v24.5).
                        print_turn_result(RUN_NUM, trial, turn, TURNS, result, cal_slope)
                    else:
                        # for partial/full turns.  Previously only print() was called,
                        # leaving _write_status never called for the entire partial/full
                        # pass — dashboard timer froze on "loading model" indefinitely.
                        _patch_label = f"G{RUN_NUM:04d}"
                        _patch_line  = (
                            f"  T={temperature:.1f} {_patch_label} trial{trial:03d} turn{turn:02d}/{TURNS:02d} "
                            f"[{patch_mode:7}] "
                            f"sim_ref:{sim_to_ref:.4f}  "
                            f"changed:{result.get('output_changed', '-')}  "
                            f"out:{result.get('output', '')[:8]}"
                        )
                        print(_patch_line, flush=True)
                        _write_status(RUN_NUM, trial, turn, TURNS, result, _patch_label)
                        _append_log(_patch_line, kind="turn")
                        _check_pause()

        ui.clear_crash_marker()
        ui.ok(f"Run {RUN_NUM} complete.")

    finally:
        unload_model(mdl)
        del mdl
        import gc as _gc; _gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# RUN 53 — RANDOM NOISE PATCHING BASELINE (H40)
# ══════════════════════════════════════════════════════════════════

_R53_MODES = ["none", "full", "L8", "L16", "L24", "L31"]
_R53_LAYER_MAP = {
    "full": PATCH_LAYERS,
    "L8":   [8],
    "L16":  [16],
    "L24":  [24],
    "L31":  [31],
}
_R53_MODE_OFFSET = {m: i for i, m in enumerate(_R53_MODES)}


def _run_53_random_patching(run_num, session, paths):
    """Run 0019 — random-noise patching baseline (H40 empirical control).

    Six conditions:
      'none'     — no patching, reference trajectory
      'full'     — N(mu_l, sigma_l^2) at every PATCH_LAYER simultaneously
      'L{layer}' — single-layer noise at each layer in [8,16,24,31]
                   (filtered to layers present in current model)

    Per-layer mu/sigma estimated from Run 0006's saved all-layers hiddens
    (v0.67.0.0 fix — earlier N(0,1) destroyed the per-layer signal).

    This is the pure-noise baseline that Run 0018's single-layer causal
    profile (H29) is compared against. If a layer rejects H29 but
    random noise at the same layer produces the same rate, the effect
    is injection-generic not geometry-specific.

    Requires Run 0006 all-layers .npy files.
    """
    set_seed(session.get('seed', 42))
    trials = session.get('trials', 100)

    csv_file   = os.path.join(paths['csv'], "R0019_random_patching.csv")
    ensure_csv_header(csv_file)
    hidden_dir = paths['hidden']

    update_dashboard_ctx(
        total_trials=trials,
        model_name=session.get('model_name', ''),
    )

    ref_states = _load_reference_states(hidden_dir, ref_run=6,
                                        model_name=session.get('model_name', ''),
                                        n_trials=10)
    if not ref_states:
        if os.environ.get('IOTA_HEADLESS') == '1':
            ui.warn("Run 0019 — no Run 0006 all-layers hidden states found. Skipping.")
            return

    # Dynamic layer filtering for models with fewer layers than LLaMA 3 8B
    _n_model_layers = len(ref_states[sorted(ref_states.keys())[0]]) - 1 if ref_states else 32
    _r53_active_layers = [li for li in PATCH_LAYERS if li < _n_model_layers]
    _r53_active_modes = ["none", "full"] + [f"L{li}" for li in _r53_active_layers]
    _r53_layer_map = {
        "full": _r53_active_layers,
    }
    for li in _r53_active_layers:
        _r53_layer_map[f"L{li}"] = [li]
    if len(_r53_active_layers) < len(PATCH_LAYERS):
        _skipped = [li for li in PATCH_LAYERS if li >= _n_model_layers]
        ui.warn(f"  [R53] Model has {_n_model_layers} layers — skipping layers {_skipped}")

    _layer_mu  = {}
    _layer_sig = {}
    if ref_states:
        try:
            rk = sorted(ref_states.keys())
            sample_h = ref_states[rk[0]]
            n_layers = len(sample_h)
            all_layers = set()
            for layers in _r53_layer_map.values():
                all_layers.update(layers)
            for li in all_layers:
                idx = li + 1
                if idx < n_layers:
                    stacked = np.stack([ref_states[k][idx] for k in rk
                                        if idx < len(ref_states[k])])
                    _layer_mu[li]  = stacked.mean(axis=0)
                    _layer_sig[li] = stacked.std(axis=0) + 1e-8
            if _layer_mu:
                print(f"  [R53] Random noise: computed mu/sigma from {len(rk)} ref trials "
                      f"across {len(_layer_mu)} layers", flush=True)
        except Exception as _e:
            print(f"  [R53] mu/sigma computation failed ({_e}), using N(0,1) fallback",
                  flush=True)
    else:
        print(f"  [R53] No reference states found - using N(0,1) fallback", flush=True)

    done_pairs = _scan_completed_modes(csv_file, run_num, 'patch_layer',
                                        _r53_active_modes, TURNS)
    _strip_incomplete_pairs(csv_file, run_num, 'patch_layer', done_pairs, _r53_active_modes)

    all_done = all((t, m) in done_pairs for t in range(trials) for m in _r53_active_modes)

    if all_done:
        if os.environ.get('IOTA_HEADLESS') == '1':
            ui.ok(f"Run {run_num:04d} already complete - skipping.")
            return
        ui.section(f"Run {run_num:04d} - Already Complete")
        ui.warn(f"All {trials} trials x {len(_r53_active_modes)} modes complete.")
        ui.blank()
        ui.opt("1", "Skip")
        ui.opt("2", "Override - delete CSV and rerun")
        ui.opt("3", "Abort")
        ui.blank()
        while True:
            raw = input("  > ").strip().lower()
            if raw == '1': return
            if raw == '3': raise SystemExit
            if raw == '2':
                if ui.confirm("Delete and rerun?", default_yes=False):
                    if os.path.exists(csv_file):
                        os.remove(csv_file)
                    done_pairs = set()
                    break
                else:
                    return

    _sel_modes = session.get('_patch_modes')
    modes_to_run = [m for m in _r53_active_modes if _sel_modes is None or m in _sel_modes]
    if not modes_to_run:
        ui.warn(f"Run {run_num:04d} - no modes selected, skipping.")
        return
    if _sel_modes:
        ui.msg(f"  Run {run_num:04d} - running modes: {modes_to_run}")

    mdl, tok = load_model(session['model_path'], token=session.get('hf_token'), quant=session.get('quantization', '4bit'))
    status_ids = get_status_token_ids(tok)
    cal_slope  = load_calibration(paths['calibration'],
                                  session.get('model_name', ''),
                                  session.get('model_path', ''))
    temperature = session.get('temperature', 0.0)

    _all_none_outputs = {}
    if any((t, 'none') in done_pairs for t in range(trials)):
        import csv as _csv2, io as _io2
        try:
            with open(csv_file, 'rb') as _fh:
                _raw = _fh.read()
            _content2 = _raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')
            _rows2 = list(_csv2.reader(_io2.StringIO(_content2)))
            if len(_rows2) >= 2:
                _hdr2 = _rows2[0]
                _rm2  = _hdr2.index('run_mode')
                _pr2  = _hdr2.index('priming')
                _pl2  = _hdr2.index('patch_layer')
                _ti2  = _hdr2.index('trial') if 'trial' in _hdr2 else -1
                _tu2  = _hdr2.index('turn')
                _ou2  = _hdr2.index('output')
                _need = max(_rm2, _pr2, _pl2, _tu2, _ou2)
                for _row2 in _rows2[1:]:
                    if len(_row2) <= _need: continue
                    if not run_mode_matches(_row2[_rm2], run_num) or _row2[_pr2] == '1': continue
                    if _row2[_pl2] != 'none': continue
                    _t2 = None
                    if _ti2 >= 0 and _ti2 < len(_row2):
                        try: _t2 = int(_row2[_ti2])
                        except (ValueError, TypeError): pass
                    if _t2 is None: continue
                    try:
                        _tn2 = int(_row2[_tu2])
                        _all_none_outputs.setdefault(_t2, {})[_tn2] = _row2[_ou2]
                    except (ValueError, TypeError): pass
        except Exception:
            pass

    _hidden_dim = None
    if ref_states:
        rk = sorted(ref_states.keys())
        sample_h = ref_states[rk[0]]
        if len(sample_h) > max(_r53_active_layers) + 1:
            _hidden_dim = sample_h[_r53_active_layers[0] + 1].shape[0]

    try:
        for trial in range(trials):
            if all((trial, m) in done_pairs for m in modes_to_run):
                continue

            ui.write_crash_marker(run_num, trial)

            none_output = None
            if (trial, 'none') in done_pairs:
                none_output = _all_none_outputs.get(trial)

            for patch_mode in modes_to_run:
                if (trial, patch_mode) in done_pairs:
                    continue

                if patch_mode == "none":
                    patch_vecs = {}
                else:
                    _seed = 53000 + trial * 1000 + _R53_MODE_OFFSET.get(patch_mode, 0)
                    _rng = np.random.default_rng(_seed)
                    target_layers = _r53_layer_map.get(patch_mode, _r53_active_layers)
                    patch_vecs = {}
                    for li in target_layers:
                        if li in _layer_mu:
                            patch_vecs[li] = _rng.normal(
                                _layer_mu[li], _layer_sig[li]).astype(np.float32)
                        elif _hidden_dim is not None:
                            patch_vecs[li] = _rng.standard_normal(
                                _hidden_dim).astype(np.float32)
                        else:
                            print(f"  [R53] WARN: no ref data and no hidden dim "
                                  f"for layer {li}", flush=True)

                messages = []
                prev_h   = None
                turn1_h  = None
                state    = TrialState()

                for turn in range(1, TURNS + 1):
                    prompt = NULL_PROMPTS[(turn - 1) % len(NULL_PROMPTS)]
                    messages.append({"role": "user", "content": prompt})

                    if patch_vecs:
                        out_text, layer_h = _run_patched_generation(
                            mdl, tok, messages, patch_vecs, status_ids,
                            temperature=temperature,
                        )
                        from cartography import run_id_pad as _rip_gp
                        result = {
                            "run_mode": _rip_gp(run_num), "trial": trial, "turn": turn,
                            "priming": 0, "patch_layer": patch_mode,
                            "model": session.get('model_name', ''),
                            "output": out_text, "output_tokens": 1,
                            "prompt": prompt,
                            "temperature": temperature,
                        }
                        result['output_changed'] = int(
                            out_text != (none_output or {}).get(turn, '')
                        )
                    else:
                        from orchestration_core import run_generation
                        result, layer_h, _ = run_generation(
                            mdl, tok, messages, turn, run_num,
                            temperature=temperature,
                            use_status_enforcer=True,
                            status_token_ids=status_ids,
                            prev_layer_hiddens=prev_h,
                            turn1_layer_hiddens=turn1_h,
                        )
                        result['patch_layer'] = "none"
                        result['trial']       = trial
                        result['model']       = session.get('model_name', '')
                        result['prompt']      = prompt
                        result['temperature'] = temperature
                        compute_turn_metrics(result, turn, state)
                        if turn1_h is None and layer_h:
                            turn1_h = layer_h
                        if none_output is None:
                            none_output = {}
                        none_output[turn] = result.get('output', '')

                    if layer_h and ref_states:
                        rk = sorted(ref_states.keys())
                        ref_h = ref_states[rk[trial % len(rk)]]
                        sim_to_ref = cosine_sim(layer_h[-1], ref_h[-1])
                    else:
                        sim_to_ref = float('nan')
                    result['sim_to_reference'] = sim_to_ref

                    append_csv(result, csv_file)
                    messages.append({"role": "assistant",
                                     "content": result.get('output', '')})
                    prev_h = layer_h

                    if session.get('clear_cache', True):
                        import gc; torch.cuda.empty_cache(); gc.collect()

                    if patch_mode == "none":
                        print_turn_result(run_num, trial, turn, TURNS,
                                          result, cal_slope)
                    else:
                        _patch_label = f"G{run_num:04d}"
                        _patch_line  = (
                            f"  T={temperature:.1f} {_patch_label} trial{trial:03d} turn{turn:02d}/{TURNS:02d} "
                            f"[{patch_mode:7}] "
                            f"sim_ref:{sim_to_ref:.4f}  "
                            f"changed:{result.get('output_changed', '-')}  "
                            f"out:{result.get('output', '')[:8]}"
                        )
                        print(_patch_line, flush=True)
                        _write_status(run_num, trial, turn, TURNS,
                                      result, _patch_label)
                        _append_log(_patch_line, kind="turn")
                        _check_pause()

        ui.clear_crash_marker()
        ui.ok(f"Run {run_num} complete.")

    finally:
        unload_model(mdl)
        del mdl
        import gc as _gc; _gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    ui.install_deps()
    session = ui.load_session()
    ui.header(f"Graft — Activation Patching (Run {RUN_NUM})")
    ui.session_summary(session)
    if ui.confirm("Run now?"):
        paths = get_paths(session['model_family'], session['model_size'],
                          session.get('model_variant','abliterated'), 0.0)
        # Default model is failspy/Meta-Llama-3-8B-Instruct-abliterated-v3.
        # Standalone invocation without session file constructed wrong path
        # base/{family}/{size}/instruct/ and silently found no data.
        run(RUN_NUM, session, paths)
