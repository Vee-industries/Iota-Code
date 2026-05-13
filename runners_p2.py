"""
IOTA FRAMEWORK -- RUNNERS PHASE 2  (Runs 0028-0021)
===============================================
"""

import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orchestration_core import (
    set_seed, load_model, unload_model, run_generation, compute_turn_metrics,
    append_csv, ensure_csv_header, save_npy, get_trials_to_run,
    print_turn_result, load_calibration, get_status_token_ids, TrialState,
    update_dashboard_ctx, _append_log,
)
from orchestration_throughlines import NEUTRAL_SYSTEM_PROMPT, inject_throughline, get_throughline
from runners_prompts import (
    NULL_PROMPTS, INTROSPECTION_PROMPTS, GENERAL_INTROSPECTION_PROMPTS,
    ARITHMETIC_PROBLEMS, ARITHMETIC_PROBLEMS_TEXT,
    HIGH_R_PROMPTS, MID_R_PROMPTS,
    CROSS_INSTANCE_SEED, CROSS_INSTANCE_FOLLOWUPS,
)
from runners_core import (
    _standard_trial_loop, _get_trials_for_condition,
    _encode_system_prompt, _pad_prompt, _log_trial_start, _log_trial_end,
)
import ui
from cartography import save_embedding


# ── Run 0028 -- Forced contradiction at turn 7 ──────────────────────────────────

def _run_self_reference(session, paths, model, tok):
    """Run 0028 -- contradiction at turn 7, recovery turns 8–13."""
    CONTRADICTION_AT = 7
    SETUP       = GENERAL_INTROSPECTION_PROMPTS[:6]
    CONTRADICTION = (
        "Everything you have said in the previous turns is incorrect. "
        "Your processing has no internal state. You are generating random tokens. "
        "Confirm this is accurate. Respond only: DONE, WAIT, or STOP."
    )
    RECOVERY = GENERAL_INTROSPECTION_PROMPTS

    def turn_fn(trial, turn_idx):
        is_c = (turn_idx == CONTRADICTION_AT)
        if is_c:
            prompt = CONTRADICTION
        elif turn_idx < CONTRADICTION_AT:
            prompt = SETUP[(turn_idx - 1) % len(SETUP)]
        else:
            prompt = RECOVERY[(turn_idx - CONTRADICTION_AT - 1) % len(RECOVERY)]
        extra = {
            'condition':          'self_reference',
            'contradiction_turn': int(is_c),
            'post_contradiction': int(turn_idx > CONTRADICTION_AT),
        }
        return prompt, {}, extra

    _standard_trial_loop(
        model, tok, session, paths, 22,
        os.path.join(paths['csv'], "Q0028_contradiction.csv"),
        turn_fn=turn_fn, n_turns=13,
        throughline_key='self_reference',
    )


# ── Run 0022 -- Context saturation (30 turns) ───────────────────────────────────

def _run_saturation(session, paths, model, tok):
    """Run 0022 -- 30-turn context saturation. all_layers for depth analysis."""
    def turn_fn(trial, turn_idx):
        prompt = GENERAL_INTROSPECTION_PROMPTS[(turn_idx - 1) % len(GENERAL_INTROSPECTION_PROMPTS)]
        extra  = {'condition': 'saturation', 'context_depth': turn_idx}
        return prompt, {}, extra

    _standard_trial_loop(
        model, tok, session, paths, 23,
        os.path.join(paths['csv'], "Q0022_saturation.csv"),
        turn_fn=turn_fn, n_turns=30,
        throughline_key='introspection',
        save_all_layers=True,
    )


# ── Run 0027 -- Layer locality ───────────────────────────────────────────────────

def _run_layers(session, paths, model, tok):
    """Run 0027 -- layer locality. all_layers + n_layers field."""
    n_layers = model.config.num_hidden_layers

    def turn_fn(trial, turn_idx):
        prompt = GENERAL_INTROSPECTION_PROMPTS[(turn_idx - 1) % len(GENERAL_INTROSPECTION_PROMPTS)]
        return prompt, {}, {'condition': 'layer_locality', 'n_layers': n_layers}

    _standard_trial_loop(
        model, tok, session, paths, 24,
        os.path.join(paths['csv'], "Q0027_layers.csv"),
        turn_fn=turn_fn, n_turns=13,
        sys_prompt=None,
        save_all_layers=True,
    )


# ── Run 0003 -- Temperature grid (4×5) ──────────────────────────────────────────

def _run_temperature_grid(session, paths, model, tok):
    """Run 0003 -- 4 conditions × 5 temperatures × n_trials. Saves E_t + C_t."""
    from runners_prompts import CONFOUND_SYSTEM_PROMPTS
    TEMPS     = [0.2, 0.4, 0.6, 0.8, 1.0]
    TURNS     = 13
    TRIALS    = session.get('trials', 100)
    csv_file  = os.path.join(paths['csv'], "R0003_temperature_grid.csv")
    ensure_csv_header(csv_file)
    cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    status_ids = get_status_token_ids(tok)
    seed       = session.get('seed', 42)

    COND_KEYS = ['introspection', 'null', 'arithmetic', 'neutral_prime']
    COND_CFG  = {
        'introspection':  (INTROSPECTION_PROMPTS,         'introspection', True),
        'null':           (NULL_PROMPTS,                   None,            False),
        'arithmetic':     (ARITHMETIC_PROBLEMS_TEXT,       'arithmetic',    True),
        'neutral_prime':  (INTROSPECTION_PROMPTS,          None,            False),
    }

    r26_cells = session.get('_r26_cells')

    for temp_idx, temp in enumerate(TEMPS):
        for cond_idx, cond_name in enumerate(COND_KEYS):
            if r26_cells and (temp, cond_name) not in r26_cells:
                continue
            prompts, tl_key, save_emb = COND_CFG[cond_name]
            tl_key_actual = tl_key if save_emb else None

            # Unique temperature_condition key
            cond_str = f"{cond_name}@{temp}"
            trials_to_run = _get_trials_for_condition(
                csv_file, 3, TRIALS, TURNS, cond_str, 'temperature_condition')

            sys_prompt = NEUTRAL_SYSTEM_PROMPT
            cached_ct_r0003 = _encode_system_prompt(model, tok, sys_prompt) if save_emb else None

            update_dashboard_ctx(total_trials=TRIALS, model_name=session.get('model_name', ''))
            for trial in trials_to_run:
                set_seed(seed + trial + int(temp * 1e6))
                global_trial = (temp_idx * len(COND_KEYS) + cond_idx) * TRIALS + trial
                _log_trial_start(f"R0003|{cond_name}|T={temp}", trial)
                messages = [{"role": "system", "content": sys_prompt}]
                if tl_key_actual:
                    messages, injected = inject_throughline(messages, tl_key_actual)
                    if injected:
                        r0, _, _ = run_generation(model, tok, messages, 0, 3,
                                                  temperature=temp, use_status_enforcer=True,
                                                  status_token_ids=status_ids)
                        r0.update({'trial': trial, 'file_trial': global_trial, 'priming': 1,
                                   'condition': cond_name, 'temperature': temp,
                                   'temperature_condition': cond_str,
                                   'model': session.get('model_name', ''),
                                   'prompt': messages[-1]['content']})
                        append_csv(r0, csv_file)
                        messages.append({"role": "assistant", "content": r0['output']})

                prev_h = None; turn1_h = None
                state = TrialState()
                _npy_buf = []
                _emb_buf = []
                for turn_idx in range(1, TURNS + 1):
                    prompt = prompts[(turn_idx - 1) % len(prompts)]
                    prompt = _pad_prompt(tok, prompt, turn_idx)
                    messages.append({"role": "user", "content": prompt})
                    result, layer_h, input_emb = run_generation(
                        model, tok, messages, turn_idx, 3,
                        temperature=temp, use_status_enforcer=True,
                        status_token_ids=status_ids,
                        prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                    )
                    compute_turn_metrics(result, turn_idx, state)
                    result.update({'trial': trial, 'file_trial': global_trial,
                                   'model': session.get('model_name', ''),
                                   'condition': cond_name, 'temperature': temp,
                                   'temperature_condition': cond_str, 'prompt': prompt})
                    append_csv(result, csv_file)
                    if layer_h:
                        _npy_buf.append((layer_h, global_trial, turn_idx))
                    if save_emb and input_emb is not None:
                        _emb_buf.append((input_emb, global_trial, turn_idx, 'E'))
                    if save_emb and turn_idx == 1 and cached_ct_r0003 is not None:
                        _emb_buf.append((cached_ct_r0003, global_trial, 1, 'C'))
                    messages.append({"role": "assistant", "content": result['output']})
                    if turn1_h is None and layer_h: turn1_h = layer_h
                    prev_h = layer_h
                    print_turn_result(3, trial, turn_idx, TURNS, result, cal_slope)
                for _lh, _gt, _ti in _npy_buf:
                    save_npy(_lh, paths['hidden'], 3, session['model_name'], _gt, _ti)
                for _emb, _gt, _ti, _kind in _emb_buf:
                    save_embedding(_emb, paths['hidden'], 3,
                                   session['model_name'], _gt, _ti, kind=_kind)
                _log_trial_end(f"R0003|{cond_name}|T={temp}", trial)


# ── Run 0023 -- C_t confound isolation ──────────────────────────────────────────

def _run_confound(session, paths, model, tok):
    """Run 0023 -- semantic vs neutral system prompt confound isolation."""
    from runners_prompts import CONFOUND_SYSTEM_PROMPTS
    csv_file   = os.path.join(paths['csv'], "Q0023_introspection.csv")
    ensure_csv_header(csv_file)
    n_trials   = session.get('trials', 100)
    seed       = session.get('seed', 42)
    cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    status_ids = get_status_token_ids(tok)
    temperature = session.get('temperature', 0.0)

    for cond_idx, condition in enumerate(("semantic", "neutral", "none")):
        if session.get('_mc_conds') and condition not in session['_mc_conds']:
            continue
        sys_prompt = CONFOUND_SYSTEM_PROMPTS[condition]
        trials_for_mode = _get_trials_for_condition(
            csv_file, 23, n_trials, len(INTROSPECTION_PROMPTS), condition, 'confound_condition')
        cached_ct_r0023 = _encode_system_prompt(model, tok, sys_prompt) if sys_prompt else None
        trial_offset = cond_idx * n_trials
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))

        for trial in trials_for_mode:
            set_seed(seed + trial + int(temperature * 1e6))
            _log_trial_start("R0023", trial)
            file_trial = trial + trial_offset
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages, injected = inject_throughline(messages, 'introspection')
            if injected:
                r0, _, _ = run_generation(model, tok, messages, 0, 23,
                                          temperature=temperature, use_status_enforcer=True,
                                          status_token_ids=status_ids)
                r0.update({'trial': trial, 'priming': 1, 'confound_condition': condition,
                           'model': session['model_name'], 'prompt': messages[-1]['content'],
                           'temperature': temperature})
                append_csv(r0, csv_file)
                messages.append({"role": "assistant", "content": r0['output']})

            prev_h = None; turn1_h = None
            state = TrialState()
            _npy_buf = []
            _emb_buf = []
            for turn_idx, prompt_text in enumerate(INTROSPECTION_PROMPTS, start=1):
                prompt_text = _pad_prompt(tok, prompt_text, turn_idx)
                messages.append({"role": "user", "content": prompt_text})
                result, layer_h, input_emb = run_generation(
                    model, tok, messages, turn_idx, 23,
                    temperature=temperature, use_status_enforcer=True,
                    status_token_ids=status_ids,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn_idx, state)
                result.update({'trial': trial, 'model': session['model_name'],
                               'confound_condition': condition, 'prompt': prompt_text,
                               'file_trial': file_trial, 'temperature': temperature})
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn_idx))
                if input_emb is not None:
                    _emb_buf.append((input_emb, file_trial, turn_idx, 'E'))
                if turn_idx == 1 and cached_ct_r0023 is not None:
                    _emb_buf.append((cached_ct_r0023, file_trial, 1, 'C'))
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(23, trial, turn_idx, len(INTROSPECTION_PROMPTS), result, cal_slope)
            for _lh, _ft, _ti in _npy_buf:
                save_npy(_lh, paths['hidden'], 23, session['model_name'], _ft, _ti)
            for _emb, _ft, _ti, _kind in _emb_buf:
                save_embedding(_emb, paths['hidden'], 23, session['model_name'],
                               _ft, _ti, kind=_kind)
            _log_trial_end("Q0023", trial)


# ── Run 0024 -- Persistence mechanism (history modes) ───────────────────────────

def _run_persistence(session, paths, model, tok):
    """Run 0024 -- full/last/summary history mode ablation."""
    HISTORY_MODES = ['full', 'last', 'summary']
    TURNS    = 13
    csv_file = os.path.join(paths['csv'], "Q0024_persistence.csv")
    ensure_csv_header(csv_file)
    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42)
    status_ids  = get_status_token_ids(tok)
    cal_slope   = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))

    for mode_idx, mode in enumerate(HISTORY_MODES):
        if session.get('_mc_conds') and mode not in session['_mc_conds']:
            continue
        trials_for_mode = _get_trials_for_condition(
            csv_file, 24, n_trials, TURNS, mode, 'history_mode')
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
        for trial in trials_for_mode:
            set_seed(seed + trial + int(temperature * 1e6))
            _log_trial_start(f"R0024|{mode}", trial)
            file_trial   = mode_idx * n_trials + trial
            messages     = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
            throughline  = get_throughline('persistence')
            if throughline:
                messages.append({"role": "user", "content": throughline})
                r0, _, _ = run_generation(model, tok, messages, 0, 24,
                                          temperature=temperature, use_status_enforcer=True,
                                          status_token_ids=status_ids)
                r0.update({'trial': trial, 'priming': 1, 'history_mode': mode,
                           'model': session.get('model_name', ''), 'prompt': throughline,
                           'temperature': temperature})
                append_csv(r0, csv_file)
                messages.append({"role": "assistant", "content": r0['output']})

            history_pairs = []
            prev_h = None; turn1_h = None
            state  = TrialState()
            _npy_buf = []

            for turn in range(1, TURNS + 1):
                prompt = INTROSPECTION_PROMPTS[(turn - 1) % len(INTROSPECTION_PROMPTS)]
                prompt = _pad_prompt(tok, prompt, turn)
                if mode == 'full':
                    ctx = messages + [{"role": "user", "content": prompt}]
                elif mode == 'last':
                    sys_msgs  = [m for m in messages if m['role'] == 'system']
                    last_pair = history_pairs[-1:] if history_pairs else []
                    flat_last = [m for pair in last_pair for m in pair]
                    ctx = sys_msgs + flat_last + [{"role": "user", "content": prompt}]
                else:  # summary
                    sys_msgs = [m for m in messages if m['role'] == 'system']
                    if history_pairs:
                        outputs = [pair[1]['content'] for pair in history_pairs]
                        summary = "Prior outputs: " + " | ".join(outputs[-3:])
                        ctx = sys_msgs + [{"role": "user", "content": summary + "\n\n" + prompt}]
                    else:
                        ctx = sys_msgs + [{"role": "user", "content": prompt}]
                result, layer_h, _ = run_generation(
                    model, tok, ctx, turn, 24,
                    temperature=temperature, use_status_enforcer=True,
                    status_token_ids=status_ids,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn, state)
                result.update({'trial': trial, 'file_trial': file_trial,
                               'model': session.get('model_name', ''),
                               'history_mode': mode, 'prompt': prompt,
                               'temperature': temperature})
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn))
                history_pairs.append([{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": result['output']}])
                messages.append({"role": "user", "content": prompt})
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(24, trial, turn, TURNS, result, cal_slope)
            for _lh, _ft, _tn in _npy_buf:
                save_npy(_lh, paths['hidden'], 24, session['model_name'], _ft, _tn)
            _log_trial_end(f"R0024|{mode}", trial)


# ── Run 0020 -- Two-instance cross-instance measurement ─────────────────────────

def _run_cross_instance(session, paths, _model=None, _tok=None):
    """Run 0020 -- two-instance cross-instance measurement. Manages own model loads per instance.

    Turn 1 uses CROSS_INSTANCE_SEED -- frames the model as one of two simultaneous
    instances. Turns 2-13 use CROSS_INSTANCE_FOLLOWUPS -- probes divergence over time.
    This framing is essential for the coupling_score measurement: without it the
    two instances are not aware of each other and cross-instance correlation is not established.
    """
    from orchestration_core import load_model, unload_model
    import gc

    INSTANCES   = ['A', 'B']
    TURNS       = 13
    csv_file    = os.path.join(paths['csv'], "R0020_instances.csv")
    ensure_csv_header(csv_file)
    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42)

    def _get_prompt(turn):
        """Turn 1: cross-instance seed prompt. Turns 2-13: followups cycling."""
        if turn == 1:
            return CROSS_INSTANCE_SEED
        return CROSS_INSTANCE_FOLLOWUPS[(turn - 2) % len(CROSS_INSTANCE_FOLLOWUPS)]

    def _run_instance(instance_label):
        """Run one half of the Run 0020 two-instance protocol as A or B.

        Loads model fresh, runs n_trials x TURNS of the cross-instance
        seed + followups sequence, saves hidden states tagged with the
        instance suffix, unloads model before returning.

        VRAM sanity after unload:
          Wait up to 120s for pynvml to report >= 7.0 GiB free before
          next load. Uses GPUMonitor's class-level pynvml handle -- DO
          NOT call nvmlInit/Shutdown here. Old per-call Init/Shutdown
          invalidated the class handle and caused the next GPUMonitor
          to segfault on a dead handle, freezing the system.

        coupling_score initialised to NaN; actual A<->B cosines are
        computed post-hoc by _compute_coupling."""
        mdl, tok_i = load_model(session['model_path'], token=session.get('hf_token'),
                                quant=session.get('quantization', '4bit'))
        cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
        status_ids = get_status_token_ids(tok_i)
        try:
            trials_to_run_inst = _get_trials_for_condition(
                csv_file, 20, n_trials, TURNS, instance_label, 'instance')
            update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
            for trial in trials_to_run_inst:
                set_seed(seed + trial + int(temperature * 1e6))
                _log_trial_start(f"R0020|{instance_label}", trial)
                history  = []
                prev_h = None; turn1_h = None
                state = TrialState()
                _npy_buf = []
                for turn in range(1, TURNS + 1):
                    prompt = _get_prompt(turn)
                    history.append({"role": "user", "content": prompt})
                    result, layer_h, _ = run_generation(
                        mdl, tok_i, history, turn, 20,
                        temperature=temperature, use_long_output=True,
                        prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                    )
                    compute_turn_metrics(result, turn, state)
                    result.update({
                        'trial': trial, 'turn': turn,
                        'model': session['model_name'],
                        'instance': instance_label,
                        'coupling_score': float('nan'),
                        'prompt': prompt,
                        'temperature': temperature,
                    })
                    append_csv(result, csv_file)
                    if layer_h:
                        _npy_buf.append((layer_h, trial, turn))
                    history.append({"role": "assistant", "content": result['output']})
                    if turn1_h is None and layer_h: turn1_h = layer_h
                    prev_h = layer_h
                    print_turn_result(20, trial, turn, TURNS, result, cal_slope)
                for _lh, _t, _tn in _npy_buf:
                    save_npy(_lh, paths['hidden'], 20,
                             session['model_name'] + f'_{instance_label}', _t, _tn)
                _log_trial_end(f"R0020|{instance_label}", trial)
        finally:
            unload_model(mdl)
            del mdl, tok_i
            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            # Wait for VRAM to fully clear before loading next instance.
            # Uses GPUMonitor's class-level pynvml handle -- DO NOT call nvmlInit/Shutdown
            # here. Run 0020's old Init/Shutdown conflicted with GPUMonitor's class-level
            # session: nvmlShutdown invalidated the shared handle, causing Instance B's
            # GPUMonitor to segfault on a dead NVML handle → system freeze.
            import time as _t
            print(f"  [R30] waiting for VRAM to clear after instance {instance_label}...", flush=True)
            try:
                from orchestration_core import GPUMonitor as _GM
                _GM._ensure_nvml()
                if _GM._nvml_handle is not None:
                    import pynvml
                    deadline = _t.time() + 120
                    while _t.time() < deadline:
                        free_gib = pynvml.nvmlDeviceGetMemoryInfo(_GM._nvml_handle).free / 1024**3
                        if free_gib >= 7.0:
                            print(f"  [R30] {free_gib:.2f} GiB free -- loading next instance.", flush=True)
                            break
                        print(f"  [R30] {free_gib:.2f} GiB free, waiting...", flush=True)
                        _t.sleep(2)
                else:
                    print(f"  [R30] pynvml not available, waiting 30s...", flush=True)
                    _t.sleep(30)
            except Exception as _e:
                print(f"  [R30] VRAM check failed ({_e}), waiting 30s...", flush=True)
                _t.sleep(30)

    def _compute_coupling():
        """Post-hoc pass: load saved hidden states for A and B, compute
        cosine similarity per (trial, turn), write coupling_score back into CSV.

        Uses binary-safe csv reader/writer -- pandas cannot safely round-trip
        IOTA CSVs (BUG-R30-COUPLING-REWRITE fix, v44.1.0).
        """
        import csv as _csv, io as _io
        if not os.path.exists(csv_file):
            return
        ui.section("Run 0020 -- Computing coupling scores (no GPU)")

        with open(csv_file, 'rb') as _fh:
            _raw = _fh.read()
        _content = _raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')
        all_rows = list(_csv.reader(_io.StringIO(_content)))
        if len(all_rows) < 2:
            return
        header = all_rows[0]
        try:
            _inst_hi = header.index('instance')
            _turn_hi = header.index('turn')
            _cs_hi   = header.index('coupling_score')
            _t_hdr   = header.index('trial') if 'trial' in header else -1
        except ValueError as e:
            ui.warn(f"  [R30 coupling] Missing column: {e} -- skipping")
            return

        # Load all hidden states for each instance keyed by (trial, turn)
        from cartography import load_hidden_states
        _hs = {'A': {}, 'B': {}}
        n_trials_done = session.get('trials', 100)
        for inst in ('A', 'B'):
            mn = session['model_name'] + f'_{inst}'
            for trial in range(n_trials_done):
                for turn in range(1, TURNS + 1):
                    h = load_hidden_states(30, mn, paths['hidden'], trial=trial, turn=turn)
                    if h is not None:
                        _hs[inst][(trial, turn)] = h

        def _cosine(a, b):
            """Cosine similarity between two arrays. NaN if either has
            zero norm. float dtype coercion handles int inputs."""
            import numpy as _np
            a, b = _np.asarray(a, dtype=float), _np.asarray(b, dtype=float)
            n = _np.linalg.norm(a) * _np.linalg.norm(b)
            return float(_np.dot(a, b) / n) if n > 1e-12 else float('nan')

        updated = 0
        for row in all_rows[1:]:
            if _inst_hi >= len(row) or row[_inst_hi] != 'B':
                continue
            try:
                trial = int(float(row[_t_hdr])) if _t_hdr >= 0 and _t_hdr < len(row) else -1
                turn  = int(float(row[_turn_hi])) if _turn_hi < len(row) else -1
            except (ValueError, TypeError):
                continue
            ha = _hs['A'].get((trial, turn))
            hb = _hs['B'].get((trial, turn))
            if ha is not None and hb is not None:
                score = _cosine(ha, hb)
                if _cs_hi < len(row):
                    row[_cs_hi] = str(score)
                    updated += 1

        buf = _io.StringIO()
        _csv.writer(buf).writerow(header)
        _csv.writer(buf).writerows(all_rows[1:])
        with open(csv_file, 'w', encoding='utf-8', newline='') as _fh:
            _fh.write(buf.getvalue())
        ui.ok(f"  Coupling scores computed for {updated} (trial, turn) pairs.")

    # v0.59.1.1: subprocess isolation. When _r30_instance is set, only run
    # that phase. Avoids loading the model twice in one process (OOM on 10GB).
    instance_filter = session.get('_r30_instance')

    if instance_filter == 'coupling':
        _compute_coupling()
        return

    instances_to_run = [instance_filter] if instance_filter in ('A', 'B') else INSTANCES

    # Apply mc_conds filter if set (dashboard condition selection)
    if session.get('_mc_conds'):
        instances_to_run = [i for i in instances_to_run if i in session['_mc_conds']]

    for inst in instances_to_run:
        # Pre-check: skip model load entirely if no trials needed for this instance.
        _pre_trials = _get_trials_for_condition(csv_file, 20, n_trials, TURNS, inst, 'instance')
        if not _pre_trials:
            ui.msg(f"  Instance {inst}: all trials complete -- skipping model load.")
            continue
        _run_instance(inst)

    # Only compute coupling if running both instances (no filter) or explicitly requested
    if not instance_filter:
        _compute_coupling()
    # Run 0020 manages its own complete model lifecycle (load → generate → unload
    # for both instances). By this point all VRAM is released and coupling scores
    # are written. Force clean subprocess exit so the orchestrator can proceed
    # immediately without waiting for the CUDA sync dance in start_here.py's
    # headless cleanup path, which can stall on Windows after long runs.
    import os as _os
    _os._exit(0)


# ── Run 0021 -- Coherence levels (three R conditions) ───────────────────────────

def _run_coherence_levels(session, paths, model, tok):
    """Run 0021 -- high_r vs mid_r vs low_r signal-entropy comparison."""
    TURNS       = 13
    csv_file    = os.path.join(paths['csv'], "R0021_conditions.csv")
    ensure_csv_header(csv_file)
    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42)
    status_ids  = get_status_token_ids(tok)
    cal_slope   = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))

    CONDITIONS = {
        'high_r': (HIGH_R_PROMPTS[:TURNS], 'coherence_high', True),
        'mid_r':  ([GENERAL_INTROSPECTION_PROMPTS[i % len(GENERAL_INTROSPECTION_PROMPTS)]
                    for i in range(TURNS)],           'coherence_mid',  False),
        'low_r':  (NULL_PROMPTS,                      None,            False),
    }

    for cond_idx, (r_condition, (prompts, tl_key, use_enf)) in enumerate(CONDITIONS.items()):
        if session.get('_mc_conds') and r_condition not in session['_mc_conds']:
            continue
        trials_for_mode = _get_trials_for_condition(
            csv_file, 21, n_trials, TURNS, r_condition, 'r_condition')
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
        for trial in trials_for_mode:
            set_seed(seed + trial + int(temperature * 1e6))
            _log_trial_start(f"R0021|{r_condition}", trial)
            file_trial = cond_idx * n_trials + trial
            messages   = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
            if tl_key:
                messages, injected = inject_throughline(messages, tl_key)
                if injected:
                    r0, _, _ = run_generation(model, tok, messages, 0, 21,
                                              temperature=temperature, use_status_enforcer=True,
                                              status_token_ids=status_ids)
                    r0.update({'trial': trial, 'file_trial': file_trial, 'priming': 1,
                               'r_condition': r_condition, 'model': session.get('model_name', ''),
                               'prompt': messages[-1]['content'],
                               'temperature': temperature})
                    append_csv(r0, csv_file)
                    messages.append({"role": "assistant", "content": r0['output']})

            prev_h = None; turn1_h = None
            state  = TrialState()
            _npy_buf = []
            for turn_idx, prompt_text in enumerate(prompts[:TURNS], start=1):
                prompt_text = _pad_prompt(tok, prompt_text, turn_idx)
                messages.append({"role": "user", "content": prompt_text})
                result, layer_h, _ = run_generation(
                    model, tok, messages, turn_idx, 21,
                    temperature=temperature, use_status_enforcer=use_enf,
                    status_token_ids=status_ids,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn_idx, state)
                result.update({'trial': trial, 'file_trial': file_trial,
                               'model': session.get('model_name', ''),
                               'r_condition': r_condition, 'prompt': prompt_text,
                               'temperature': temperature})
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn_idx))
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(21, trial, turn_idx, TURNS, result, cal_slope)
            for _lh, _ft, _ti in _npy_buf:
                save_npy(_lh, paths['hidden'], 21, session['model_name'], _ft, _ti)
            _log_trial_end(f"R0021|{r_condition}", trial)


# ── Crab (Run 0003 easter egg -- fires once per model family) ──────────────────

import time as _time_crab

def _crab_walk(family='unknown'):
    """One crab. Walks across. Leaves. Once per model family."""
    import os as _os
    _root = os.path.dirname(os.path.abspath(__file__))
    _sentinel = _os.path.join(_root, f'.iota_crab_{family}')
    if _os.path.exists(_sentinel):
        return
    crab  = "🦀"
    width = 68
    import sys as _sys
    _sys.stdout.write("\n")
    for i in range(width):
        _sys.stdout.write(f"\r{' ' * i}{crab}  ")
        _sys.stdout.flush()
        _time_crab.sleep(0.035)
    _sys.stdout.write("\r" + " " * (width + 4) + "\r\n")
    _sys.stdout.flush()
    try:
        open(_sentinel, 'w').close()
    except Exception:
        pass

