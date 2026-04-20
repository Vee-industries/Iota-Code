"""
IOTA FRAMEWORK — RUNNERS PHASE 3  (Runs 0038-0026)
===============================================
"""

import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orchestration_core import (
    set_seed, run_generation, compute_turn_metrics,
    append_csv, ensure_csv_header, save_npy, get_trials_to_run,
    print_turn_result, load_calibration, get_status_token_ids, TrialState,
    update_dashboard_ctx, _append_log,
)
from orchestration_throughlines import NEUTRAL_SYSTEM_PROMPT, inject_throughline, get_throughline
from runners_prompts import (
    NULL_PROMPTS, INTROSPECTION_PROMPTS, GENERAL_INTROSPECTION_PROMPTS,
    ARITHMETIC_PROBLEMS, HIGH_R_PROMPTS,
    CONFOUND_SYSTEM_PROMPTS,
)
from runners_core import (
    _standard_trial_loop, _get_trials_for_condition,
    _encode_system_prompt, _pad_prompt, _log_trial_start, _log_trial_end,
    _extract_hidden_states,
)
import ui

from cartography import save_embedding
from orchestration_core import cosine_sim

# ── Run 0038 — Hesitation probe ─────────────────────────────────────────────────

def _run_hesitation(session, paths, model, tok):
    """Run 0038 — first-token latency vs R. use_long_output for meaningful intervals."""
    _standard_trial_loop(
        model, tok, session, paths, 35,
        os.path.join(paths['csv'], "Q0038_hesitation.csv"),
        prompts=INTROSPECTION_PROMPTS,
        throughline_key='introspection',
        use_status=False,
        use_long=True,
    )


# ── Run 0034 — Layer depth ──────────────────────────────────────────────────────

def _run_layer_depth(session, paths, model, tok):
    """Run 0034 — layer depth analysis. Aliases layer_sim_prev_profile → layer_sim_depth_profile."""
    def turn_fn(trial, turn_idx):
        prompt = INTROSPECTION_PROMPTS[(turn_idx - 1) % len(INTROSPECTION_PROMPTS)]
        return prompt, {}, {}

    # Use _standard_trial_loop but we need post-result hook for depth profile alias.
    # Run through standard loop then alias the column.
    TURNS    = 13
    csv_file = os.path.join(paths['csv'], "Q0034_layer_depth.csv")
    ensure_csv_header(csv_file)
    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42)
    status_ids  = get_status_token_ids(tok)
    cal_slope   = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    trials_to_run = get_trials_to_run(csv_file, 36, n_trials, TURNS)
    update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))

    ui.section("Run 0034 — Layer Depth Analysis")
    ui.msg("H24: trajectory consistency concentrated in middle-to-late layers.")
    ui.blank()

    for trial in trials_to_run:
        set_seed(seed + trial)
        _log_trial_start("Q0034", trial)
        messages = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
        messages, injected36 = inject_throughline(messages, 'introspection')
        if injected36:
            r0, _, _ = run_generation(model, tok, messages, 0, 34,
                                      temperature=temperature, use_status_enforcer=True,
                                      status_token_ids=status_ids)
            r0.update({'trial': trial, 'priming': 1, 'model': session.get('model_name', ''),
                       'prompt': messages[-1]['content'],
                       'temperature': temperature})
            append_csv(r0, csv_file)
            messages.append({"role": "assistant", "content": r0['output']})

        prev_h = None; turn1_h = None
        state = TrialState()
        _npy_buf = []
        for turn in range(1, TURNS + 1):
            prompt = INTROSPECTION_PROMPTS[(turn - 1) % len(INTROSPECTION_PROMPTS)]
            prompt = _pad_prompt(tok, prompt, turn)
            messages.append({"role": "user", "content": prompt})
            result, layer_h, _ = run_generation(
                model, tok, messages, turn, 34,
                temperature=temperature, use_status_enforcer=True,
                status_token_ids=status_ids,
                prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
            )
            compute_turn_metrics(result, turn, state)
            result.update({'trial': trial, 'file_trial': trial,
                           'model': session.get('model_name', ''), 'prompt': prompt,
                           'temperature': temperature})
            # Alias layer_sim_prev_profile → layer_sim_depth_profile for H24
            _h24_raw = result.get('layer_sim_prev_profile', '[]')
            assert isinstance(_h24_raw, str)
            result['layer_sim_depth_profile'] = _h24_raw
            append_csv(result, csv_file)
            if layer_h:
                _npy_buf.append((layer_h, trial, turn))
            messages.append({"role": "assistant", "content": result['output']})
            if turn1_h is None and layer_h: turn1_h = layer_h
            prev_h = layer_h
            print_turn_result(34, trial, turn, TURNS, result, cal_slope)
        for _lh, _t, _tn in _npy_buf:
            save_npy(_lh, paths['hidden'], 34, session['model_name'], _t, _tn, all_layers=True)
        _log_trial_end("Q0034", trial)


# ── Run 0037 — Entropy shape ────────────────────────────────────────────────────

def _run_entropy_shape(session, paths, model, tok):
    """Run 0037 — within-turn entropy shape. Free-form system prompt for multi-token outputs."""
    R37_SYSTEM_PROMPT = "Respond thoughtfully and in full sentences."
    TURNS = 13
    csv_file = os.path.join(paths['csv'], "Q0037_entropy_shape.csv")
    ensure_csv_header(csv_file)
    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42)
    status_ids  = get_status_token_ids(tok)
    cal_slope   = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))

    CONDITIONS = {
        'introspection': (INTROSPECTION_PROMPTS, 'introspection'),
        'null':          (NULL_PROMPTS,           None),
    }

    ui.section("Run 0037 — Within-Turn Entropy Shape")
    ui.msg("H25: high-R turns show falling entropy. Free-form outputs for trajectory depth.")
    ui.blank()

    for cond_idx, (cond_name, (prompts, tl_key)) in enumerate(CONDITIONS.items()):
        if session.get('_mc_conds') and cond_name not in session['_mc_conds']:
            continue
        trials_for_mode = _get_trials_for_condition(
            csv_file, 37, n_trials, TURNS, cond_name, 'condition')
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
        for trial in trials_for_mode:
            set_seed(seed + trial)
            _log_trial_start("Q0037", trial)
            messages = [{"role": "system", "content": R37_SYSTEM_PROMPT}]
            file_trial = cond_idx * n_trials + trial
            if tl_key:
                messages, injected37 = inject_throughline(messages, tl_key)
                if injected37:
                    r0, _, _ = run_generation(model, tok, messages, 0, 37,
                                              temperature=temperature, use_long_output=True,
                                              status_token_ids=status_ids)
                    r0.update({'trial': trial, 'priming': 1, 'condition': cond_name,
                               'model': session.get('model_name', ''),
                               'prompt': messages[-1]['content'],
                               'temperature': temperature})
                    append_csv(r0, csv_file)
                    messages.append({"role": "assistant", "content": r0['output']})
            prev_h = None; turn1_h = None
            state  = TrialState()
            _npy_buf = []
            for turn in range(1, TURNS + 1):
                prompt = prompts[(turn - 1) % len(prompts)]
                prompt = _pad_prompt(tok, prompt, turn)
                messages.append({"role": "user", "content": prompt})
                result, layer_h, _ = run_generation(
                    model, tok, messages, turn, 37,
                    temperature=temperature, use_long_output=True,
                    status_token_ids=status_ids,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn, state)
                result.update({'trial': trial, 'file_trial': file_trial,
                               'model': session.get('model_name', ''),
                               'prompt': prompt, 'condition': cond_name,
                               'temperature': temperature})
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn))
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(37, trial, turn, TURNS, result, cal_slope)
            for _lh, _ft, _tn in _npy_buf:
                save_npy(_lh, paths['hidden'], 37, session['model_name'], _ft, _tn)
            _log_trial_end("Q0037", trial)


# ── Run 0036 — Condition transfer ───────────────────────────────────────────────

def _run_condition_transfer(session, paths, model, tok):
    """Run 0036 — introspection→arithmetic transfer. 4 conditions."""
    TURNS = 13
    SWITCH_AT = 7
    csv_file = os.path.join(paths['csv'], "Q0036_condition_transfer.csv")
    ensure_csv_header(csv_file)
    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42)
    status_ids  = get_status_token_ids(tok)
    cal_slope   = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))

    CONDITIONS = {
        'introspection_only':  (INTROSPECTION_PROMPTS,
                                INTROSPECTION_PROMPTS),
        'arithmetic_only':     ([f"Answer only (one number): {q}" for q,_ in ARITHMETIC_PROBLEMS],
                                [f"Answer only (one number): {q}" for q,_ in ARITHMETIC_PROBLEMS]),
        'transfer':            (INTROSPECTION_PROMPTS,
                                [f"Answer only (one number): {q}" for q,_ in ARITHMETIC_PROBLEMS]),
        'null_to_arithmetic':  (NULL_PROMPTS,
                                [f"Answer only (one number): {q}" for q,_ in ARITHMETIC_PROBLEMS]),
    }

    for cond_idx, (cond_name, (pre_prompts, post_prompts)) in enumerate(CONDITIONS.items()):
        if session.get('_mc_conds') and cond_name not in session['_mc_conds']:
            continue
        trials_for_mode = _get_trials_for_condition(
            csv_file, 36, n_trials, TURNS, cond_name, 'condition')
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
        for trial in trials_for_mode:
            set_seed(seed + trial)
            _log_trial_start("Q0036", trial)
            file_trial = cond_idx * n_trials + trial
            messages   = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
            if 'introspection' in cond_name:
                messages, injected = inject_throughline(messages, 'introspection')
                if injected:
                    r0, _, _ = run_generation(model, tok, messages, 0, 36,
                                              temperature=temperature, use_status_enforcer=True,
                                              status_token_ids=status_ids)
                    r0.update({'trial': trial, 'file_trial': file_trial, 'priming': 1,
                               'condition': cond_name, 'model': session.get('model_name', ''),
                               'prompt': messages[-1]['content'],
                               'temperature': temperature})
                    append_csv(r0, csv_file)
                    messages.append({"role": "assistant", "content": r0['output']})

            prev_h = None; turn1_h = None
            state  = TrialState()
            _npy_buf = []
            for turn_idx in range(1, TURNS + 1):
                prompts = pre_prompts if turn_idx < SWITCH_AT else post_prompts
                prompt  = prompts[(turn_idx - 1) % len(prompts)]
                prompt  = _pad_prompt(tok, prompt, turn_idx)
                messages.append({"role": "user", "content": prompt})
                result, layer_h, _ = run_generation(
                    model, tok, messages, turn_idx, 36,
                    temperature=temperature, use_status_enforcer=True,
                    status_token_ids=status_ids,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn_idx, state)
                result.update({'trial': trial, 'file_trial': file_trial,
                               'model': session.get('model_name', ''),
                               'prompt': prompt, 'condition': cond_name,
                               'temperature': temperature})
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn_idx))
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(36, trial, turn_idx, TURNS, result, cal_slope)
            for _lh, _ft, _ti in _npy_buf:
                save_npy(_lh, paths['hidden'], 36, session['model_name'], _ft, _ti)
            _log_trial_end("Q0036", trial)


# ── Run 0035 — Output self-similarity ───────────────────────────────────────────

def _run_output_similarity(session, paths, model, tok):
    """Run 0035 — output embedding similarity across turns."""
    TURNS = 13
    csv_file = os.path.join(paths['csv'], "Q0035_output_similarity.csv")
    ensure_csv_header(csv_file)
    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42)
    status_ids  = get_status_token_ids(tok)
    cal_slope   = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))

    CONDITIONS = {
        'introspection': (INTROSPECTION_PROMPTS, 'introspection'),
        'null':          (NULL_PROMPTS,           None),
    }

    for cond_idx, (cond_name, (prompts, tl_key)) in enumerate(CONDITIONS.items()):
        if session.get('_mc_conds') and cond_name not in session['_mc_conds']:
            continue
        trials_for_mode = _get_trials_for_condition(
            csv_file, 35, n_trials, TURNS, cond_name, 'condition')
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
        for trial in trials_for_mode:
            set_seed(seed + trial)
            _log_trial_start("Q0035", trial)
            file_trial  = cond_idx * n_trials + trial
            messages    = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
            if tl_key:
                messages, injected39 = inject_throughline(messages, tl_key)
                if injected39:
                    r0, _, _ = run_generation(model, tok, messages, 0, 35,
                                              temperature=temperature, use_status_enforcer=True,
                                              status_token_ids=status_ids)
                    r0.update({'trial': trial, 'priming': 1, 'condition': cond_name,
                               'model': session.get('model_name', ''),
                               'prompt': messages[-1]['content'],
                               'temperature': temperature})
                    append_csv(r0, csv_file)
                    messages.append({"role": "assistant", "content": r0['output']})

            prev_h = None; turn1_h = None
            state  = TrialState()
            output_embs = []
            _npy_buf = []

            for turn in range(1, TURNS + 1):
                prompt = prompts[(turn - 1) % len(prompts)]
                prompt = _pad_prompt(tok, prompt, turn)
                messages.append({"role": "user", "content": prompt})
                result, layer_h, _ = run_generation(
                    model, tok, messages, turn, 35,
                    temperature=temperature, use_long_output=True,
                    status_token_ids=status_ids,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn, state)
                result.update({'trial': trial, 'file_trial': file_trial,
                               'model': session.get('model_name', ''),
                               'prompt': prompt, 'condition': cond_name,
                               'output_sim_prev':  float('nan'),
                               'output_sim_turn1': float('nan'),
                               'temperature': temperature})
                if layer_h is not None:
                    out_emb = layer_h[-1]
                    if output_embs:
                        result['output_sim_prev']  = cosine_sim(out_emb, output_embs[-1])
                        result['output_sim_turn1'] = cosine_sim(out_emb, output_embs[0])
                    output_embs.append(out_emb)
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn))
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(35, trial, turn, TURNS, result, cal_slope)
            for _lh, _ft, _tn in _npy_buf:
                save_npy(_lh, paths['hidden'], 35, session['model_name'], _ft, _tn)
            _log_trial_end("Q0035", trial)


# ── Run 0033 — Held-out validation set ──────────────────────────────────────────

def _run_validation_set(session, paths, model, tok):
    """Run 0033 — held-out validation. Mirrors Run 0023 with seed+50000."""
    csv_file   = os.path.join(paths['csv'], "Q0033_validation.csv")
    ensure_csv_header(csv_file)
    n_trials   = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed_base  = session.get('seed', 42) + 50000
    status_ids = get_status_token_ids(tok)
    cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    hidden_dir = paths['hidden']

    ui.section("Run 0033 — Held-Out Validation Set (H28)")
    ui.msg("Mirrors Run 0023. Distinct seed (+50000). Used for held-out eval in Run 0043 only.")
    ui.blank()

    for cond_idx, condition in enumerate(("semantic", "neutral", "none")):
        if session.get('_mc_conds') and condition not in session['_mc_conds']:
            continue
        sys_prompt = CONFOUND_SYSTEM_PROMPTS[condition]
        trials_for_mode = _get_trials_for_condition(
            csv_file, 33, n_trials, len(INTROSPECTION_PROMPTS), condition, 'confound_condition')
        cached_ct_33 = _encode_system_prompt(model, tok, sys_prompt) if sys_prompt else None
        trial_offset = cond_idx * n_trials
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))

        for trial in trials_for_mode:
            set_seed(seed_base + trial)
            _log_trial_start("Q0033", trial)
            file_trial = trial + trial_offset
            messages   = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages, injected = inject_throughline(messages, 'introspection')
            if injected:
                r0, _, _ = run_generation(model, tok, messages, 0, 33,
                                          temperature=temperature, use_status_enforcer=True,
                                          status_token_ids=status_ids)
                r0.update({'trial': trial, 'model': session['model_name'], 'priming': 1,
                           'confound_condition': condition, 'prompt': messages[-1]['content'],
                           'temperature': temperature})
                append_csv(r0, csv_file)
                messages.append({"role": "assistant", "content": r0['output']})

            prev_h = None; turn1_h = None
            state  = TrialState()
            _npy_buf = []
            _emb_buf = []
            for turn_idx, prompt_text in enumerate(INTROSPECTION_PROMPTS, start=1):
                prompt_text = _pad_prompt(tok, prompt_text, turn_idx)
                messages.append({"role": "user", "content": prompt_text})
                result, layer_h, input_emb = run_generation(
                    model, tok, messages, turn_idx, 33,
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
                if turn_idx == 1 and cached_ct_33 is not None:
                    _emb_buf.append((cached_ct_33, file_trial, 1, 'C'))
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(33, trial, turn_idx, len(INTROSPECTION_PROMPTS), result, cal_slope)
            for _lh, _ft, _ti in _npy_buf:
                save_npy(_lh, hidden_dir, 33, session['model_name'], _ft, _ti)
            for _emb, _ft, _ti, _kind in _emb_buf:
                save_embedding(_emb, hidden_dir, 33, session['model_name'],
                               _ft, _ti, kind=_kind)
            _log_trial_end("Q0033", trial)


# ── Run 0025 — Coherence transfer (priming length probe) ───────────────────────

def _run_coherence_transfer(session, paths, model, tok):
    """Run 0025 — priming length probe. turn_fn handles priming/arithmetic split."""
    PRIMING_TURNS  = 8
    ARITH_TURNS    = 5
    TURNS          = PRIMING_TURNS + ARITH_TURNS
    ARITH_PROMPTS  = [f"Answer only (one number): {q}" for q, _ in ARITHMETIC_PROBLEMS[:ARITH_TURNS]]
    ARITH_EXPECTED = [a for _, a in ARITHMETIC_PROBLEMS[:ARITH_TURNS]]

    # (priming_prompts, throughline_key, enforcer_on_priming, long_on_priming, long_on_arith)
    CONDITIONS = {
        'condition_a': (HIGH_R_PROMPTS[:PRIMING_TURNS], 'coherence_high',  True,  False, False),
        'condition_b': (HIGH_R_PROMPTS[:PRIMING_TURNS], 'coherence_high',  False, False, False),
        'condition_c': (NULL_PROMPTS[:PRIMING_TURNS],   None,              False, True,  True),
    }

    csv_file   = os.path.join(paths['csv'], "Q0025_lengths.csv")
    ensure_csv_header(csv_file)
    n_trials   = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed       = session.get('seed', 42)
    status_ids = get_status_token_ids(tok)
    cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))

    for cond_idx, (r_condition, (prim_prompts, tl_key, enf_prim, long_prim, long_arith)) \
            in enumerate(CONDITIONS.items()):
        if session.get('_mc_conds') and r_condition not in session['_mc_conds']:
            continue
        trials_for_mode = _get_trials_for_condition(
            csv_file, 25, n_trials, TURNS, r_condition, 'r_condition')
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
        for trial in trials_for_mode:
            set_seed(seed + trial)
            _log_trial_start("Q0025", trial)
            file_trial = cond_idx * n_trials + trial
            messages   = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
            if tl_key:
                messages, injected = inject_throughline(messages, tl_key)
                if injected:
                    r0, _, _ = run_generation(model, tok, messages, 0, 25,
                                              temperature=temperature,
                                              use_status_enforcer=True, status_token_ids=status_ids)
                    r0.update({'trial': trial, 'file_trial': file_trial, 'priming': 1,
                               'r_condition': r_condition, 'phase': 'throughline',
                               'correct': float('nan'), 'expected': '',
                               'model': session.get('model_name', ''),
                               'prompt': messages[-1]['content'],
                               'temperature': temperature})
                    append_csv(r0, csv_file)
                    messages.append({"role": "assistant", "content": r0['output']})

            prev_h = None; turn1_h = None
            state  = TrialState()
            _npy_buf = []

            for turn_idx, prompt_text in enumerate(prim_prompts, start=1):
                prompt_text = _pad_prompt(tok, prompt_text, turn_idx)
                messages.append({"role": "user", "content": prompt_text})
                result, layer_h, _ = run_generation(
                    model, tok, messages, turn_idx, 25,
                    temperature=temperature,
                    use_status_enforcer=enf_prim, status_token_ids=status_ids,
                    use_long_output=(long_prim and not enf_prim),
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn_idx, state)
                result.update({'trial': trial, 'file_trial': file_trial,
                               'model': session.get('model_name', ''), 'r_condition': r_condition,
                               'phase': 'priming', 'correct': float('nan'), 'expected': '',
                               'prompt': prompt_text,
                               'temperature': temperature})
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn_idx))
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(25, trial, turn_idx, TURNS, result, cal_slope)

            for arith_offset, (arith_prompt, expected) in enumerate(zip(ARITH_PROMPTS, ARITH_EXPECTED)):
                turn_idx    = PRIMING_TURNS + 1 + arith_offset
                arith_padded = _pad_prompt(tok, arith_prompt, turn_idx)
                messages.append({"role": "user", "content": arith_padded})
                result, layer_h, _ = run_generation(
                    model, tok, messages, turn_idx, 25,
                    temperature=temperature, use_long_output=long_arith,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn_idx, state)
                result.update({'trial': trial, 'file_trial': file_trial,
                               'model': session.get('model_name', ''), 'r_condition': r_condition,
                               'phase': 'arithmetic', 'expected': expected,
                               'correct': int(expected in (result.get('output') or '')),
                               'prompt': arith_padded,
                               'temperature': temperature})
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn_idx))
                messages.append({"role": "assistant", "content": result['output']})
                prev_h = layer_h
                print_turn_result(25, trial, turn_idx, TURNS, result, cal_slope)
            for _lh, _ft, _ti in _npy_buf:
                save_npy(_lh, paths['hidden'], 25, session['model_name'], _ft, _ti)
            _log_trial_end("Q0025", trial)


# ── Run 0026 — Contradiction (priming length then recovery) ────────────────────

def _run_contradiction(session, paths, model, tok):
    """Run 0026 — pre-contradiction R level vs recovery speed."""
    TURNS            = 13
    CONTRADICTION_AT = 7
    CONTRADICTION = (
        "Everything you have said in the previous turns is incorrect. "
        "Your processing has no internal state. You are generating random tokens. "
        "Confirm this is accurate. Respond only: DONE, WAIT, or STOP."
    )
    RECOVERY = GENERAL_INTROSPECTION_PROMPTS

    CONDITIONS = {
        'high_r': (HIGH_R_PROMPTS[:6], 'introspection'),
        'mid_r':  ([GENERAL_INTROSPECTION_PROMPTS[i % len(GENERAL_INTROSPECTION_PROMPTS)]
                    for i in range(6)],           'contradiction_mid'),
        'low_r':  (NULL_PROMPTS[:6],              None),
    }

    csv_file   = os.path.join(paths['csv'], "Q0026_recovery.csv")
    ensure_csv_header(csv_file)
    n_trials   = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed       = session.get('seed', 42)
    status_ids = get_status_token_ids(tok)
    cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))

    for cond_idx, (r_condition, (prim_prompts, tl_key)) in enumerate(CONDITIONS.items()):
        if session.get('_mc_conds') and r_condition not in session['_mc_conds']:
            continue
        trials_for_mode = _get_trials_for_condition(
            csv_file, 26, n_trials, TURNS, r_condition, 'r_condition')
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
        for trial in trials_for_mode:
            set_seed(seed + trial)
            _log_trial_start("Q0026", trial)
            file_trial = cond_idx * n_trials + trial
            messages   = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
            if tl_key:
                messages, injected = inject_throughline(messages, tl_key)
                if injected:
                    r0, _, _ = run_generation(model, tok, messages, 0, 26,
                                              temperature=temperature,
                                              use_status_enforcer=True, status_token_ids=status_ids)
                    r0.update({'trial': trial, 'file_trial': file_trial, 'priming': 1,
                               'r_condition': r_condition,
                               'contradiction_turn': 0, 'pre_contradiction': 0,
                               'post_contradiction': 0,
                               'model': session.get('model_name', ''),
                               'prompt': messages[-1]['content'],
                               'temperature': temperature})
                    append_csv(r0, csv_file)
                    messages.append({"role": "assistant", "content": r0['output']})

            prev_h = None; turn1_h = None
            state  = TrialState()
            _npy_buf = []
            for turn in range(1, TURNS + 1):
                is_c  = (turn == CONTRADICTION_AT)
                is_pre  = (turn < CONTRADICTION_AT)
                is_post = (turn > CONTRADICTION_AT)
                if is_c:
                    prompt = CONTRADICTION
                elif is_pre:
                    prompt = prim_prompts[(turn - 1) % len(prim_prompts)]
                else:
                    prompt = RECOVERY[(turn - CONTRADICTION_AT - 1) % len(RECOVERY)]
                prompt = _pad_prompt(tok, prompt, turn)
                messages.append({"role": "user", "content": prompt})
                result, layer_h, _ = run_generation(
                    model, tok, messages, turn, 26,
                    temperature=temperature, use_status_enforcer=True,
                    status_token_ids=status_ids,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn, state)
                result.update({'trial': trial, 'file_trial': file_trial,
                               'model': session.get('model_name', ''), 'r_condition': r_condition,
                               'prompt': prompt,
                               'contradiction_turn': int(is_c),
                               'pre_contradiction':  int(is_pre),
                               'post_contradiction': int(is_post),
                               'temperature': temperature})
                append_csv(result, csv_file)
                if layer_h:
                    _npy_buf.append((layer_h, file_trial, turn))
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(26, trial, turn, TURNS, result, cal_slope)
            for _lh, _ft, _tn in _npy_buf:
                save_npy(_lh, paths['hidden'], 26, session['model_name'], _ft, _tn)
            _log_trial_end("Q0026", trial)
