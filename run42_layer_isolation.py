"""
IOTA FRAMEWORK — RUN 42: LAYER CAUSAL SUFFICIENCY
===================================================
Single-layer activation patching. Tests which individual layers from
{8, 16, 24, 31} are causally sufficient to shift output token selection.

Run 0017 patches all four layers simultaneously and asks: does injecting
Run 0006's geometry change output? The answer is yes. Run 0018 asks: which
layer is doing the work?

Patch modes (five per trial/turn):
  none  — no injection, clean null baseline (full trajectory metrics)
  L8    — inject reference geometry at layer 8 only
  L16   — inject reference geometry at layer 16 only
  L24   — inject reference geometry at layer 24 only
  L31   — inject reference geometry at layer 31 only

Primary metric: output_change_rate per layer — fraction of (trial, turn)
pairs where output differs from the none condition.

Secondary metric: sim_to_reference (final-layer cosine similarity to
reference state) — measures whether the injection at layer li propagated
forward through the remaining stack to the output layer. Low sim for L8
means the network smoothed out the injection before output; high sim for
L31 is expected (nearly direct). The gradient across layers is the
persistence profile.

output_changed: 1 if output != none_output for this (trial, turn), 0
otherwise. Computed live since none always runs first.

MIXED-SCHEMA CSV (same pattern as Run 0017, Finding O, v23.3):
  patch_layer == "none" rows: full trajectory metrics present
    (state_similarity_index, signal_entropy_ratio, disruption_flag, onset_delay_ratio, etc.)
  patch_layer in ("L8","L16","L24","L31") rows: patching-specific fields only
    (sim_to_reference, output, output_changed, output_tokens, prompt)
  NaN values in patched rows are STRUCTURAL — not data gaps.
  Filter patch_layer == "none" before accessing standard trajectory metrics.

Prerequisite: Run 0006 all-layers hidden states (_alllayers.npy files).
Same prerequisite as Run 0017 — graft_patching.py documents the dependency.

Connection to layer-locality hypotheses:
  Run 0027 (H14/H57): correlational — where does consistency concentrate?
  Run 0034 (H24): correlational — layer depth of trajectory consistency
  Run 0018 (H29): causal — which layer is sufficient to shift output?
  If peak causal layer aligns with correlation peak from 24/36: strongest
  layer-locality finding in the framework.

RUN_NUM: 18  CSV prefix: R (new numbering, was Q42)  Output: R0018_layer_isolation.csv

v0.58.0.0
"""

import os, sys

def _find_root():
    """Walk up from this file's directory to the iota root (contains
    start_here.py). Lets run42 be invoked standalone or from a subprocess."""
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

import glob
import torch
import numpy as np
from transformers import LogitsProcessorList

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

RUN_NUM      = 18
TURNS        = 13
TRIALS       = None  # set from session at runtime

# Layers to test individually. Matches Run 0017's PATCH_LAYERS.
# LLaMA 3 8B has 32 transformer layers (indices 0-31).
PROBE_LAYERS = [8, 16, 24, 31]

# Patch modes: none first (establishes baseline output for output_changed)
# then one layer at a time.
PATCH_MODES  = ["none"] + [f"L{li}" for li in PROBE_LAYERS]

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


def _run_patched_generation(mdl, tok, messages, patch_layer_idx: int,
                             patch_vec: np.ndarray, status_ids,
                             temperature: float = 0.0) -> tuple:
    """
    Forward pass with single-layer activation grafting via hook.

    patch_layer_idx: int — which transformer layer to inject at
    patch_vec:       np.ndarray shape (D,) — reference vector for that layer

    Returns: (out_text, layer_h)
      out_text: generated token string
      layer_h:  list of L per-layer hidden state vectors, shape (D,) each
    """
    hook_handle = None

    def make_hook(vec):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            inject = torch.tensor(vec, dtype=h.dtype, device=h.device)
            h[0, -1, :] = inject
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    layer = mdl.model.layers[patch_layer_idx]
    hook_handle = layer.register_forward_hook(make_hook(patch_vec))

    if tok.chat_template:
        fmt = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    else:
        fmt = "\n".join(f"{m['role']}: {m['content']}"
                        for m in messages) + "\nassistant:"

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
                logits_processor=LogitsProcessorList([
                    timer,
                    StatusTokenEnforcerFixed(status_ids, get_eos_ids(tok)),
                ]),
            )
    finally:
        if hook_handle:
            hook_handle.remove()

    gen_ids  = out.sequences[0]
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
    """Load turn-1 all-layers hidden states from reference run (Run 0006).

    First tries exact model_name match. Falls back to wildcard scan if not found
    (handles model_name drift from model picker bug).
    """
    refs = {}
    for trial in range(n_trials):
        h = load_hidden_states(ref_run, model_name, hidden_dir,
                               trial=trial, turn=1, all_layers=True)
        if h is not None:
            refs[trial] = h
    if refs:
        return refs

    # Fallback: wildcard scan
    import glob as _glob, numpy as _np
    pfx = run_prefix(ref_run)
    for trial in range(n_trials):
        # v0.79.5.0: dual-glob 4-digit + 2-digit for transition compat
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
        print(f"  [layer_iso] Found Run {ref_run:02d} ref states via wildcard scan "
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

def _load_none_outputs(csv_file, run_num, trial, turns):
    """Read none-mode outputs for a specific trial. Returns {turn: output_text}."""
    try:
        from orchestration_core import _csv_read
        df = _csv_read(csv_file)
        if df.empty:
            return {}
        from cartography import run_mode_mask as _rmm_r42
        mask = (_rmm_r42(df['run_mode'], run_num) &
                (df['priming'] != '1') &
                (df['patch_layer'] == 'none') &
                (df['trial'].apply(lambda x: int(float(x)) if x == x else -1) == trial))
        sub = df[mask]
        return {int(float(r['turn'])): r['output'] for _, r in sub.iterrows()
                if r['turn'] == r['turn']}
    except Exception:
        return {}




def run(run_num: int, session: dict, paths: dict):
    """Run 0018 — single-layer causal sufficiency (H29, H37).

    Tests whether any individual layer, when grafted with Run 0006's
    introspection hidden states at that layer only, is sufficient to
    change the output token on its own. Five patch conditions:
      'none', 'L8', 'L16', 'L24', 'L31'.

    Models smaller than LLaMA 3 8B skip layers above their last index
    (guarded via .n_layers file written by the runner on first load).

    H29 rejection: output_changed rate > 5% for a (layer, temperature)
    cell means the layer is individually causally sufficient.

    H37 (layer x temperature interaction): dominant layer shifts with
    temperature. L16 dominates at T=0, L31 dominates at T>=0.8.
    Profile is temperature-dependent, not fixed.

    Requires Run 0006 all-layers .npy files."""
    assert run_num == RUN_NUM
    trials = session.get('trials', 100)

    csv_file   = os.path.join(paths['csv'], f"Q{RUN_NUM:04d}_layer_isolation.csv")
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
            ui.warn("Run 0018 — no Run 0006 all-layers hidden states found. Skipping.")
            return
        choice = ui.srx_prompt(
            "No Run 0006 all-layers hidden states found.\n"
            "  Run 0006 (introspection A) must complete with save_all_layers=True first.\n"
            "  Run 0018 (layer causal sufficiency) shares this prerequisite with Run 0017."
        )
        if choice in ('s', 'x'):
            if choice == 'x': raise SystemExit
            return

    # ── Dynamic layer detection ────────────────────────────────────────────
    _n_model_layers = len(ref_states[sorted(ref_states.keys())[0]]) - 1 if ref_states else 32
    _active_probe = [li for li in PROBE_LAYERS if li < _n_model_layers]
    _active_modes = ["none"] + [f"L{li}" for li in _active_probe]
    if len(_active_probe) < len(PROBE_LAYERS):
        _skipped = [li for li in PROBE_LAYERS if li >= _n_model_layers]
        ui.warn(f"  [R42] Model has {_n_model_layers} layers — skipping probe layers {_skipped}")

    # ── Resume / completeness check ──────────────────────────────────────────
    # Patched rows (L8/L16/L24/L31) are shorter than the header; standard
    # trial-counting logic skips them and sees only none rows. When none is
    # complete (100 trials × 13 turns) the run is marked done and skipped.
    # Fix: scan (trial, mode) pairs explicitly; only mark done when ALL modes
    # have n_trials × TURNS rows each.
    _ALL_MODES_42 = _active_modes
    done_pairs = _scan_completed_modes(
        csv_file, RUN_NUM, 'patch_layer', _ALL_MODES_42, TURNS)

    all_done = all(
        (t, m) in done_pairs for t in range(trials) for m in _ALL_MODES_42)

    if all_done:
        if os.environ.get('IOTA_HEADLESS') == '1':
            ui.ok(f"Run {RUN_NUM:04d} already complete — skipping.")
            return
        ui.section(f"Run {RUN_NUM:04d} — Already Complete")
        ui.warn(f"All {trials} trials × {len(_ALL_MODES_42)} modes complete.")
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
    modes_to_run = [m for m in _ALL_MODES_42 if _sel_modes is None or m in _sel_modes]
    if not modes_to_run:
        ui.warn(f"Run {RUN_NUM:04d} — no modes selected, skipping.")
        return
    if _sel_modes:
        ui.msg(f"  Run {RUN_NUM:04d} — running modes: {modes_to_run}")

    mdl, tok   = load_model(session['model_path'], token=session.get('hf_token'), quant=session.get('quantization', '4bit'))
    status_ids = get_status_token_ids(tok)
    cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    temperature = session.get('temperature', 0.0)

    ref_keys = sorted(ref_states.keys())
    n_refs   = len(ref_keys)

    # Dedup before patching — remove excess rows from resume overlap
    _none_complete = sum(1 for t in range(trials) if (t,'none') in done_pairs)
    if _none_complete > 0:
        try:
            from orchestration_core import _csv_read
            _df_d = _csv_read(csv_file)
            if not _df_d.empty and 'patch_layer' in _df_d.columns:
                def _tint(x):
                    """Coerce x to int via float, -1 on any failure.
                    Used to normalise trial/turn column values from
                    the CSV where they may be strings, NaN, or empty."""
                    try: return int(float(x))
                    except: return -1
                _df_d['_t'] = _df_d['trial'].apply(_tint)
                _df_d['_turn'] = _df_d['turn'].apply(_tint)
                _kept = []
                for (_t, _m), grp in _df_d.groupby(['_t', 'patch_layer']):
                    if len(grp) > TURNS:
                        _kept.append(grp.iloc[:TURNS])
                    else:
                        _kept.append(grp)
                import pandas as _pd_d
                _df_clean = _pd_d.concat(_kept).drop(columns=['_t','_turn'])
                _rem = len(_df_d) - len(_df_clean)
                if _rem > 0:
                    _df_clean.to_csv(csv_file, index=False)
                    ui.msg(f"  [R42 dedup] Removed {_rem} duplicate rows before patching.")
                    done_pairs = _scan_completed_modes(csv_file, RUN_NUM, 'patch_layer', _ALL_MODES_42, TURNS)
        except Exception as _de:
            ui.warn(f"  [R42 dedup] Warning: {_de}")

    # Preload none-mode outputs for trials already complete
    _all_none_outputs: dict = {}
    if any((t, 'none') in done_pairs for t in range(trials)):
        try:
            from orchestration_core import _csv_read
            _df42 = _csv_read(csv_file)
            if not _df42.empty:
                _none42 = _df42[
                    (_df42['run_mode'] == str(RUN_NUM)) &
                    (_df42['priming'] != '1') &
                    (_df42['patch_layer'] == 'none')
                ]
                for _, _r in _none42.iterrows():
                    try:
                        _t42 = int(float(_r['trial']))
                        _tn42 = int(float(_r['turn']))
                        _all_none_outputs.setdefault(_t42, {})[_tn42] = _r['output']
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    try:
        # Phase 1: collect none-mode for all trials before any patching.
        # Patch layers need none outputs as baseline — must be fully complete first.
        _none_needed = 'none' in modes_to_run
        _none_all_done = all((t, 'none') in done_pairs for t in range(trials))

        if _none_needed and not _none_all_done:
            ui.msg("  Phase 1: collecting none-mode baseline for all trials...")
            for trial in range(trials):
                set_seed(session.get('seed', 42) + trial)
                if (trial, 'none') in done_pairs:
                    continue
                ui.write_crash_marker(RUN_NUM, trial)
                messages = []
                prev_h   = None
                turn1_h  = None
                state    = TrialState()
                none_output = {}
                for turn in range(1, TURNS + 1):
                    prompt = NULL_PROMPTS[(turn - 1) % len(NULL_PROMPTS)]
                    messages.append({"role": "user", "content": prompt})
                    from orchestration_core import run_generation
                    result, layer_h, _ = run_generation(
                        mdl, tok, messages, turn, RUN_NUM,
                        temperature=temperature,
                        use_status_enforcer=True,
                        status_token_ids=status_ids,
                        prev_layer_hiddens=prev_h,
                        turn1_layer_hiddens=turn1_h,
                    )
                    result['patch_layer']    = "none"
                    result['trial']          = trial
                    result['model']          = session.get('model_name', '')
                    result['prompt']         = prompt
                    result['temperature']    = temperature
                    result['output_changed'] = 0
                    compute_turn_metrics(result, turn, state)
                    if turn1_h is None and layer_h:
                        turn1_h = layer_h
                    none_output[turn] = result.get('output', '')
                    if layer_h and ref_states:
                        ref_trial = ref_keys[trial % n_refs]
                        result['sim_to_reference'] = cosine_sim(layer_h[-1], ref_states[ref_trial][-1])
                    else:
                        result['sim_to_reference'] = float('nan')
                    append_csv(result, csv_file)
                    messages.append({"role": "assistant", "content": result.get('output', '')})
                    prev_h = layer_h
                    if session.get('clear_cache', True):
                        import gc; torch.cuda.empty_cache(); gc.collect()
                    print_turn_result(RUN_NUM, trial, turn, TURNS, result, cal_slope)
                _all_none_outputs[trial] = none_output
                done_pairs.add((trial, 'none'))

            ui.ok("  Phase 1 complete — all none-mode trials done.")

        # Re-preload all none outputs now that phase 1 is complete
        if not _all_none_outputs:
            import csv as _csv42p, io as _io42p
            try:
                with open(csv_file, 'rb') as _fh42: _raw42 = _fh42.read()
                _c42 = _raw42.decode('utf-8', errors='replace').replace('\r\n','\n').replace('\r','\n')
                _rows42 = list(_csv42p.reader(_io42p.StringIO(_c42)))
                if len(_rows42) >= 2:
                    _hdr42 = _rows42[0]; _nh42 = len(_hdr42)
                    _ti42  = _hdr42.index('trial') if 'trial' in _hdr42 else -1
                    _tu42  = _hdr42.index('turn')
                    _ou42  = _hdr42.index('output')
                    _pm42  = _hdr42.index('patch_layer') if 'patch_layer' in _hdr42 else -1
                    for _row42 in _rows42[1:]:
                        if len(_row42) < _nh42: continue  # skip patched rows
                        if _pm42 >= 0 and _row42[_pm42] != 'none': continue
                        try:
                            _t42 = int(_row42[_ti42]); _tn42 = int(_row42[_tu42])
                            _all_none_outputs.setdefault(_t42, {})[_tn42] = _row42[_ou42]
                        except: pass
            except: pass

        # Phase 2: patch layers for all trials
        patch_modes_only = [m for m in modes_to_run if m != 'none']
        if patch_modes_only:
            ui.msg(f"  Phase 2: running patch layers {patch_modes_only} for all trials...")

        for trial in range(trials):
            set_seed(session.get('seed', 42) + trial)  # R42-2 fix (v28.3): per-trial seed

            if all((trial, m) in done_pairs for m in modes_to_run):
                continue

            ui.write_crash_marker(RUN_NUM, trial)
            ref_trial = ref_keys[trial % n_refs]
            ref_h     = ref_states[ref_trial]

            none_output = _all_none_outputs.get(trial)

            for patch_mode in modes_to_run:
                if patch_mode == 'none':
                    continue  # already done in phase 1
                if (trial, patch_mode) in done_pairs:
                    continue

                messages = []
                prev_h   = None
                turn1_h  = None  # R42-1 fix (v28.3)
                state    = TrialState()

                for turn in range(1, TURNS + 1):
                    prompt = NULL_PROMPTS[(turn - 1) % len(NULL_PROMPTS)]
                    messages.append({"role": "user", "content": prompt})

                    if patch_mode == "none":
                        from orchestration_core import run_generation
                        result, layer_h, _ = run_generation(
                            mdl, tok, messages, turn, RUN_NUM,
                            temperature=temperature,
                            use_status_enforcer=True,
                            status_token_ids=status_ids,
                            prev_layer_hiddens=prev_h,
                            turn1_layer_hiddens=turn1_h,
                        )
                        result['patch_layer']    = "none"
                        result['trial']          = trial
                        result['model']          = session.get('model_name', '')
                        result['prompt']         = prompt
                        result['temperature']    = temperature
                        result['output_changed'] = 0
                        compute_turn_metrics(result, turn, state)
                        if turn1_h is None and layer_h:
                            turn1_h = layer_h
                        if none_output is None:
                            none_output = {}
                        none_output[turn] = result.get('output', '')

                    else:
                        patch_li  = int(patch_mode[1:])   # "L8" → 8
                        patch_vec = ref_h[patch_li + 1]   # Bug W fix: +1 offset
                        out_text, layer_h = _run_patched_generation(
                            mdl, tok, messages, patch_li, patch_vec, status_ids,
                            temperature=temperature,
                        )
                        baseline_out = (none_output or {}).get(turn, '')
                        result = {
                            "run_mode":       RUN_NUM,
                            "trial":          trial,
                            "turn":           turn,
                            "priming":        0,
                            "patch_layer":    patch_mode,
                            "model":          session.get('model_name', ''),
                            "output":         out_text,
                            "output_tokens":  1,
                            "prompt":         prompt,
                            "output_changed": int(out_text != baseline_out),
                            "temperature":    temperature,
                        }

                    if layer_h and ref_h:
                        result['sim_to_reference'] = cosine_sim(layer_h[-1], ref_h[-1])
                    else:
                        result['sim_to_reference'] = float('nan')

                    append_csv(result, csv_file)
                    messages.append({"role": "assistant", "content": result.get('output', '')})
                    prev_h = layer_h

                    if session.get('clear_cache', True):
                        import gc
                        torch.cuda.empty_cache()
                        gc.collect()

                    if patch_mode == "none":
                        print_turn_result(RUN_NUM, trial, turn, TURNS, result, cal_slope)
                    else:
                        _patch_label = f"Q{RUN_NUM:04d}"
                        _patch_line  = (
                            f"  {_patch_label} trial{trial:03d} turn{turn:02d}/{TURNS:02d} "
                            f"[{patch_mode:4}] "
                            f"sim_ref:{result.get('sim_to_reference', float('nan')):.4f}  "
                            f"changed:{result.get('output_changed', '-')}  "
                            f"out:{result.get('output','')[:8]}"
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


if __name__ == "__main__":
    ui.install_deps()
    session = ui.load_session()
    ui.header(f"Run {RUN_NUM} — Layer Causal Sufficiency")
    ui.session_summary(session)
    if ui.confirm("Run now?"):
        paths = get_paths(
            session['model_family'], session['model_size'],
            session.get('model_variant', 'abliterated'), 0.0
        )
        run(RUN_NUM, session, paths)
