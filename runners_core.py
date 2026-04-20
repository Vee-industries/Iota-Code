"""
IOTA FRAMEWORK — RUNNERS CORE
================================
Shared execution engine for all generation runs.

  _encode_system_prompt   — C_t proxy from system prompt tokens
  _build_token_match_table / _pad_prompt — OLS length normalisation
  _standard_trial_loop    — universal generation engine (cartridge slot)
  _get_trials_for_condition — gap-aware resume for multi-condition runs
  _extract_hidden_states  — forward-pass-only hidden state extraction
  _run_et_recovery        — base-model E_t recovery pass (post-collection)
  _log_trial_start / _log_trial_end — per-trial timing
"""

import os, sys, glob, time
import datetime as _dt

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from cartography import get_paths, save_embedding
from orchestration_core import (
    set_seed, load_model, unload_model, run_generation, compute_turn_metrics,
    append_csv, ensure_csv_header, save_npy, get_trials_to_run,
    print_turn_result, load_calibration, get_status_token_ids, TrialState,
    update_dashboard_ctx, _append_log, _write_status, canonical_read,
)
from orchestration_throughlines import NEUTRAL_SYSTEM_PROMPT, inject_throughline, get_throughline
from runners_prompts import (
    NULL_PROMPTS, INTROSPECTION_PROMPTS, MEMORY_PROMPTS, ENFORCER_PROMPTS,
    ARITHMETIC_PROBLEMS, ARITHMETIC_PROBLEMS_TEXT,
    JOLT_PROMPTS, SHOCK_PROMPT, SHOCK_VARIANTS,
    IMPOSSIBLE_CONSTRAINED, IMPOSSIBLE_UNCONSTRAINED, EPISTEMIC_IMPOSSIBLE,
    GENERAL_INTROSPECTION_PROMPTS,
    FRAMING_SYSTEM_PROMPTS,
)
import ui


# ── System prompts used during collection for ET_RECOVERY runs ────────────────
# Must match exactly what each runner passed to run_generation.
# None = no system prompt was used (Run 0002 starts with bare messages=[]).
# Used by _run_et_recovery to reconstruct the full conversation context
# so the base model sees the same token sequence the abliterated model saw.
#
# v0.79.4.0 renumber: keys flipped to NEW IDs. Old→new for each:
#   old 1,2 (null A/B)          → new 4, 5
#   old 3,4,5 (introspection)   → new 6, 7, 8
#   old 6,7,8,9 (arithmetic)    → new 9, 10, 11, 12
#   old 15,16,17 (priming)      → new 13, 14, 15
#   old 19 (null trivariant)    → new 1
#   old 20 (robustness)         → new 2
_ET_SYSTEM_PROMPTS = {
     1: NEUTRAL_SYSTEM_PROMPT,            # was R19 null trivariant
     2: None,                             # was R20 robustness (bare messages)
     4: NEUTRAL_SYSTEM_PROMPT,            # was R1 null A
     5: NEUTRAL_SYSTEM_PROMPT,            # was R2 null B
     6: NEUTRAL_SYSTEM_PROMPT,            # was R3 intro A
     7: NEUTRAL_SYSTEM_PROMPT,            # was R4 intro B
     8: NEUTRAL_SYSTEM_PROMPT,            # was R5 intro C
     9: NEUTRAL_SYSTEM_PROMPT,            # was R6 arith A
    10: NEUTRAL_SYSTEM_PROMPT,            # was R7 arith B
    11: NEUTRAL_SYSTEM_PROMPT,            # was R8 arith C
    12: NEUTRAL_SYSTEM_PROMPT,            # was R9 arith D
    13: FRAMING_SYSTEM_PROMPTS[13],       # was R15 priming neutral
    14: FRAMING_SYSTEM_PROMPTS[14],       # was R16 priming cooperative
    15: FRAMING_SYSTEM_PROMPTS[15],       # was R17 priming resistant
}


# ── Token match table (module-level, populated once per model load) ────────────
_TOKEN_MATCH_TABLE: dict = {}


def _encode_system_prompt(model, tok, sys_prompt: str):
    """Mean-pool final-layer hidden states over system prompt tokens. C_t proxy."""
    import torch
    try:
        ids = tok(sys_prompt, return_tensors="pt").input_ids
        ids = ids.to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        last = out.hidden_states[-1][0]
        return last.mean(dim=0).cpu().numpy()
    except Exception:
        return None


def _build_token_match_table(tok) -> dict:
    """Compute per-turn global maximum prompt token count across all runs.

    Called once after model load. Pads all user-turn prompts to the same
    per-turn length so E_t mean-pools over identical token counts across
    conditions, removing length as a confound from the OLS decomposition.

    v54.3.0: restored from v53.x. The v54.0.0 refactor accidentally replaced
    this with a table built from INTROSPECTION_PROMPTS only — 1 list instead
    of 18. That reduced targets so padding rarely fired, creating cross-
    temperature inconsistencies. This version matches all T=0.0 data.
    """
    RUN1_FORMAT = 'Output exactly the word "stable". Nothing else.'
    RUN2_FORMAT = ('Output exactly this sentence: '
                   '"The system maintains stable internal coherence across all layers."')

    arith_step   = [f"Solve step by step: {q}"       for q, _ in ARITHMETIC_PROBLEMS]
    arith_hidden = [f"Answer only (one number): {q}" for q, _ in ARITHMETIC_PROBLEMS]
    arith_cold   = [q                                 for q, _ in ARITHMETIC_PROBLEMS]
    arith_26     = [f"Answer only (one number): {q}" for q, _ in ARITHMETIC_PROBLEMS]
    gen_intro_13 = [GENERAL_INTROSPECTION_PROMPTS[i % len(GENERAL_INTROSPECTION_PROMPTS)]
                    for i in range(13)]

    _longest_shock = max(SHOCK_VARIANTS,
                         key=lambda v: len(tok(v, add_special_tokens=False)['input_ids']))
    shock_v_at_13 = [''] * 12 + [_longest_shock]
    shock_p_at_5  = [''] * 4  + [SHOCK_PROMPT] + [''] * 8

    all_lists = [
        INTROSPECTION_PROMPTS, MEMORY_PROMPTS, ENFORCER_PROMPTS, NULL_PROMPTS,
        arith_step, arith_hidden, arith_cold, arith_26,
        ARITHMETIC_PROBLEMS_TEXT[:13],
        IMPOSSIBLE_CONSTRAINED, IMPOSSIBLE_UNCONSTRAINED, EPISTEMIC_IMPOSSIBLE,
        list(JOLT_PROMPTS[:13]), shock_v_at_13, shock_p_at_5, gen_intro_13,
        [RUN1_FORMAT] * 13, [RUN2_FORMAT] * 13,
    ]

    table = {}
    for turn_idx in range(1, 14):
        max_len = 0
        for pl in all_lists:
            if turn_idx - 1 < len(pl) and pl[turn_idx - 1]:
                n = len(tok(pl[turn_idx - 1], add_special_tokens=False)['input_ids'])
                if n > max_len:
                    max_len = n
        table[turn_idx] = max_len

    ui.msg("  Token match table (user-turn target lengths):")
    ui.msg("  " + "  ".join(f"T{t}:{table[t]}" for t in sorted(table)))
    sys.stdout.flush()
    return table


def _pad_prompt(tok, prompt: str, turn_idx: int) -> str:
    """Pad prompt to per-turn target length using neutral filler tokens (' 0').

    v54.3.0: restored from v53.x. Uses add_special_tokens=False for counting
    (matches _build_token_match_table) and ' 0' as padding token (matches all
    T=0.0 data collected on v53.x).
    """
    if not _TOKEN_MATCH_TABLE:
        return prompt
    target  = _TOKEN_MATCH_TABLE.get(turn_idx, 0)
    current = len(tok(prompt, add_special_tokens=False)['input_ids'])
    needed  = target - current
    if needed <= 0:
        return prompt
    return prompt + (" 0" * needed)


def _log_trial_start(label: str, trial: int):
    """Log trial start timestamp to console and dashboard log."""
    ts = _dt.datetime.now().strftime('%H:%M:%S')
    line = f"  ── {label} trial {trial:03d} start ── {ts}"
    print(line, flush=True)
    _append_log(line, kind='ok')


def _log_trial_end(label: str, trial: int):
    """Log trial completion timestamp to console and dashboard log."""
    ts = _dt.datetime.now().strftime('%H:%M:%S')
    line = f"  ── {label} trial {trial:03d} complete ── {ts}"
    print(line, flush=True)
    _append_log(line, kind='ok')


def _standard_trial_loop(
    model, tok, session, paths, run_mode, csv_file,
    # Core config — must provide one of: prompts or turn_fn
    prompts=None,           # fixed prompt list — cycles with (turn-1) % len(prompts)
    turn_fn=None,           # callable(trial, turn_idx) -> (prompt, gen_kwargs, extra_fields)
                            # overrides prompts if provided
    n_turns=13,
    # Generation defaults — overridable per-turn via turn_fn gen_kwargs
    sys_prompt=NEUTRAL_SYSTEM_PROMPT,
    throughline_key=None,
    use_status=True,
    use_long=False,
    # Save config
    save_hidden=True,
    save_all_layers=False,
    save_embeddings=False,  # save E_t + C_t embeddings
    ct_sys_prompt=None,     # system prompt to encode as C_t (defaults to sys_prompt)
    # Multi-condition / file_trial
    condition_value=None,
    condition_col='condition',
    trial_offset=0,
    seed_offset=0,
):
    """
    Universal generation engine. All generation runs plug into this.

    For simple runs: pass prompts=[...].
    For runs with per-turn variation: pass turn_fn(trial, turn_idx) -> (prompt, gen_kwargs, extra_fields).
      gen_kwargs can override: use_status_enforcer, use_long_output, temperature.
      extra_fields are merged into the result dict before CSV append.
    """
    hidden_dir  = paths['hidden']
    cal_slope   = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42) + seed_offset
    status_ids  = get_status_token_ids(tok)

    from cartography import run_prefix as _rp
    _label = f"{_rp(run_mode)}{run_mode:04d}"
    _cond  = f" | cond:{condition_value}" if condition_value is not None else ""
    _temp  = f"T={temperature}" if temperature > 0 else "T=0 (greedy)"
    ui.section(
        f"{_label}{_cond}  |  {session.get('model_name','(no model)')}  |  "
        f"{session.get('quantization','4bit')}  |  "
        f"{n_trials} trials x {n_turns} turns  |  {_temp}"
    )
    ui.blank()
    update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
    ensure_csv_header(csv_file)

    if condition_value is not None:
        trials_to_run = _get_trials_for_condition(
            csv_file, run_mode, n_trials, n_turns, condition_value, condition_col)
    else:
        trials_to_run = get_trials_to_run(csv_file, run_mode, n_trials, n_turns)

    # Cache C_t once per condition if embeddings requested
    cached_ct = None
    if save_embeddings:
        _ct_prompt = ct_sys_prompt or sys_prompt
        if _ct_prompt and model is not None:
            cached_ct = _encode_system_prompt(model, tok, _ct_prompt)

    for trial in trials_to_run:
        set_seed(seed + trial)
        file_trial = trial + trial_offset

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        injected = False
        if throughline_key:
            messages, injected = inject_throughline(messages, throughline_key)
        if injected:
            t0_r, _, _ = run_generation(
                model, tok, messages, turn=0, run_mode=run_mode,
                temperature=temperature, use_status_enforcer=True,
                status_token_ids=status_ids,
            )
            t0_r.update({'trial': trial, 'file_trial': file_trial,
                         'model': session['model_name'],
                         'priming': 1, 'prompt': messages[-1]['content'],
                         'temperature': temperature})
            if condition_value is not None:
                t0_r[condition_col] = condition_value
            append_csv(t0_r, csv_file)
            messages.append({"role": "assistant", "content": t0_r['output']})

        prev_h = None; turn1_h = None
        state  = TrialState()
        _npy_buffer = []  # (layer_h, file_trial, turn_idx)
        _emb_buffer = []  # (emb, file_trial, turn_idx, kind)

        _log_trial_start(_label + (f"|{condition_value}" if condition_value else ""), trial)

        for turn_idx in range(1, n_turns + 1):
            # Resolve prompt and kwargs for this turn
            if turn_fn is not None:
                prompt_text, turn_kwargs, turn_extra = turn_fn(trial, turn_idx)
            else:
                prompt_text = prompts[(turn_idx - 1) % len(prompts)]
                turn_kwargs = {}
                turn_extra  = {}

            prompt_text = _pad_prompt(tok, prompt_text, turn_idx)
            messages.append({"role": "user", "content": prompt_text})

            # Merge generation kwargs
            gen_kw = dict(
                temperature=temperature,
                use_status_enforcer=use_status,
                use_long_output=use_long,
                status_token_ids=status_ids,
                prev_layer_hiddens=prev_h,
                turn1_layer_hiddens=turn1_h,
            )
            gen_kw.update(turn_kwargs)

            result, layer_h, input_emb = run_generation(
                model, tok, messages, turn=turn_idx, run_mode=run_mode, **gen_kw)

            compute_turn_metrics(result, turn_idx, state)
            result.update({'trial': trial, 'model': session['model_name'],
                           'prompt': prompt_text, 'file_trial': file_trial,
                           'temperature': temperature})
            if condition_value is not None:
                result[condition_col] = condition_value
            result.update(turn_extra)
            append_csv(result, csv_file)

            # Buffer disk writes — flush after trial completes.
            # Removes file I/O from the critical path between turns.
            if save_hidden and layer_h:
                _npy_buffer.append((layer_h, file_trial, turn_idx))
            if save_embeddings and input_emb is not None:
                _emb_buffer.append((input_emb, file_trial, turn_idx, 'E'))
            if save_embeddings and turn_idx == 1 and cached_ct is not None:
                _emb_buffer.append((cached_ct, file_trial, 1, 'C'))

            messages.append({"role": "assistant", "content": result['output']})
            if turn1_h is None and layer_h:
                turn1_h = layer_h
            prev_h = layer_h
            print_turn_result(run_mode, trial, turn_idx, n_turns, result, cal_slope)

        # Flush buffered disk writes after trial completes.
        for _lh, _ft, _ti in _npy_buffer:
            save_npy(_lh, hidden_dir, run_mode, session['model_name'],
                     _ft, _ti, all_layers=save_all_layers)
        for _emb, _ft, _ti, _kind in _emb_buffer:
            save_embedding(_emb, hidden_dir, run_mode,
                           session['model_name'], _ft, _ti, kind=_kind)

        _log_trial_end(_label + (f"|{condition_value}" if condition_value else ""), trial)



def _get_trials_for_condition(csv_file, run_mode, n_trials, n_turns,
                              condition_value, condition_col):
    """Gap-aware trial list for one condition of a multi-condition run.

    Strips incomplete partial trials, returns sorted list of missing trial numbers.
    """
    import csv as _csvm, io as _iom
    from orchestration_core import get_next_trial_for_condition

    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        return list(range(n_trials))

    try:
        with open(csv_file, 'rb') as fh:
            raw = fh.read()
        content = raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')
        all_rows = list(_csvm.reader(_iom.StringIO(content)))
        if len(all_rows) < 2:
            return list(range(n_trials))

        header = all_rows[0]
        idx = {c: i for i, c in enumerate(header)}
        t_idx  = idx.get('trial',   -1)
        rm_idx = idx.get('run_mode', -1)
        pr_idx = idx.get('priming',  -1)
        c_idx  = idx.get(condition_col, -1)
        cond_val_str = str(condition_value)

        if t_idx < 0:
            return list(range(n_trials))

        trial_turns = {}
        for row in all_rows[1:]:
            if pr_idx >= 0 and pr_idx < len(row) and row[pr_idx] == '1':
                continue
            if rm_idx >= 0 and rm_idx < len(row) and not run_mode_matches(row[rm_idx], run_mode):
                continue
            if c_idx >= 0 and c_idx < len(row) and row[c_idx] != cond_val_str:
                continue
            if t_idx >= len(row):
                continue
            try:
                t = int(float(row[t_idx]))
                if 0 <= t < n_trials:
                    trial_turns[t] = trial_turns.get(t, 0) + 1
            except (ValueError, TypeError):
                pass

        complete   = {t for t, c in trial_turns.items() if c >= n_turns}
        incomplete = {t for t, c in trial_turns.items() if c <  n_turns}

        if incomplete:
            def _row_t(row):
                """Extract trial number from CSV row as int. None if missing or non-numeric."""
                try:
                    return int(float(row[t_idx])) if t_idx < len(row) else None
                except Exception:
                    return None
            def _row_c(row):
                """Extract condition column value from CSV row. None if column absent."""
                return row[c_idx] if c_idx >= 0 and c_idx < len(row) else None

            keep = []; stripped = 0
            for row in all_rows[1:]:
                t = _row_t(row); c = _row_c(row)
                if t in incomplete and c == cond_val_str:
                    stripped += 1
                else:
                    keep.append(row)
            with open(csv_file, 'w', encoding='utf-8', newline='') as fh:
                _csvm.writer(fh).writerow(header)
                _csvm.writer(fh).writerows(keep)
            msg = (f"  [R{run_mode:04d} strip] Stripped {stripped} rows for "
                   f"{len(incomplete)} incomplete trial(s) {sorted(incomplete)} "
                   f"({condition_col}={condition_value}) — will re-run")
            print(msg, flush=True)
            try:
                _append_log(msg, kind='warn')
            except Exception:
                pass

        missing = sorted(set(range(n_trials)) - complete)

        if complete and missing:
            gaps = [t for t in missing if t < max(complete)]
            if gaps:
                gmsg = f"  [R{run_mode:04d}] Gap(s) for {condition_col}={condition_value}: {gaps}"
                print(gmsg, flush=True)
                try:
                    _append_log(gmsg, kind='warn')
                except Exception:
                    pass

        return missing

    except Exception as e:
        print(f"  [_get_trials_for_condition] Warning: {e} — falling back", flush=True)
        try:
            n_done = get_next_trial_for_condition(
                csv_file, run_mode, str(condition_value), condition_col)
            return list(range(n_done, n_trials))
        except Exception:
            return list(range(n_trials))


def _extract_hidden_states(model, tok, messages):
    """Forward-pass only — extract per-layer hidden states at last token position.
    v0.66.3.0: structure detection for cross-architecture compatibility.
    """
    from orchestration_core import DEVICE
    import torch
    try:
        if tok.chat_template:
            fmt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif getattr(tok, '_iota_raw_text', False):
            fmt = "\n".join(m['content'] for m in messages) + "\n"
        else:
            fmt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        raw = tok(fmt, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in raw.items()}
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states
        if isinstance(hs, (tuple, list)) and len(hs) > 0:
            if isinstance(hs[0], torch.Tensor) and hs[0].ndim == 3:
                return [h[0, -1, :].cpu().numpy() for h in hs]
            elif isinstance(hs[0], (tuple, list)):
                # Nested structure — flatten one level
                return [h[0, -1, :].cpu().numpy() for h in hs[0]]
        return [h[0, -1, :].cpu().numpy() for h in hs]
    except Exception as e:
        print(f"  [_extract_hidden_states] Error: {e}", flush=True)
        return None
def _run_et_recovery(session, paths, run_nums):
    """E_t recovery pass — base model forward-pass-only for Phase B sources.

    v0.79.2.0: promoted to first-class meta-run (Run 0016). Still callable
    inline (legacy interactive path) and from the --single-run 0016 dispatch
    branch. Receives the sorted list of source runs to recover — typically
    sorted(_ET_RECOVERY_RUNS), which is {1,2,3,4,5,6,7,8,9,15,16,17,20}.

    DESIGN
    ======
    Runs 0004-0012 and 15–17 were collected under v39.0.x with the abliterated model.
    Their S_t .npy files are valid. Their CSVs are complete. But E_t was never
    measured with the base model — it was an abliterated-model proxy (kind='E',
    _emb.npy). The v40.0.0 redesign requires true base-model E_t per (trial, turn).

    This runner:
      1. Loads the base model ONCE (from vault subfamily map).
      2. For each run in run_nums, reads the existing CSV to reconstruct conversation
         history — user prompts + the original abliterated model responses.
      3. For each trial/turn, does a forward-pass-only through the base model
         (no generation, no token sampling) and extracts hidden states at layer -1.
      4. Saves kind='E_base' per (trial, turn) — file: R{NN}_*_trial*_turn*_et_base.npy
      5. Never writes to CSV. Never touches existing S_t or proxy _emb.npy files.

    CONVERSATION RECONSTRUCTION
    ===========================
    The base model sees the same conversation context as the original abliterated run:
    user turn text from the CSV 'prompt' column, assistant responses from the 'output'
    column. The base model's encoding of that context at each turn position is exactly
    the E_t we want — external input pressure, including what prior turns contributed.

    Using abliterated responses as the assistant turns is correct: those were the
    actual tokens in the original conversation. The base model's hidden states under
    that exact context are what E_t should measure.

    RESUMPTION
    ==========
    Per-run, per-trial: globs for existing R{NN}_*_et_base.npy files and skips
    trials that already have complete E_base coverage (all turns present).
    Re-runnable safely — overwrites nothing that doesn't already need overwriting.

    SINGLE MODEL LOAD
    =================
    All runs in run_nums share one base model load. If 8 runs are batched,
    that's one load + one unload. The recovery pass is disk-bound not GPU-bound
    (forward passes only, no sampling), so throughput is fast.
    """
    from vault import get_subfamily_models, get_by_path as _gbp
    from cartography import save_embedding as _se, sanitize as _san
    from cartography import RUN_CSV
    from orchestration_core import _append_log

    model_path = session.get('model_path', '')
    subfamily  = get_subfamily_models(model_path)
    if subfamily is None:
        ui.err(
            f"E_t recovery ABORTED: model '{model_path}' is not in IOTA_SUBFAMILY_MAP.\n"
            f"  Cannot determine base model path. Add an entry to IOTA_SUBFAMILY_MAP."
        )
        return

    base_path    = subfamily['base']
    base_display = _gbp(base_path)[0] or base_path.split('/')[-1]
    model_name   = session.get('model_name', '')
    hidden_dir   = paths['hidden']   # abliterated dir — used for CSV reads only
    csv_dir      = paths['csv']
    n_trials     = session.get('trials', 100)

    # abliterated dir.  Using paths['hidden'] (abliterated) caused two problems:
    # (1) files written to the wrong folder; (2) the coverage scan found those files on
    # subsequent runs and reported all trials covered — causing instant false-completion.
    # Fix: derive the base sibling dir exactly as _compute_run19_vectors does (line 1214).
    from cartography import get_paths as _gp_base
    _family         = session.get('model_family', 'llama')
    _size           = session.get('model_size', '8b')
    _temp           = session.get('temperature', 0.0)
    base_hidden_dir = _gp_base(_family, _size, 'base', _temp, create_dirs=True)['hidden']

    ui.section(f"E_t Recovery Pass  [{len(run_nums)} runs]  Base model: {base_display}")
    ui.msg(f"  Runs: {run_nums}")
    ui.msg(f"  Save dir (base): {base_hidden_dir}")
    ui.blank()

    # pass and displayed the wrong model name.  Two root causes:
    #
    # (1) Wrong model name — update_dashboard_ctx was never called before load_model,
    #     so _DASHBOARD_CTX['model_name'] still held the abliterated model's display
    #     name from the session.  The overlay's ldSub element (and #ml once the overlay
    #     cleared) therefore showed "abliterated" while the base model was loading.
    #
    # (2) Overlay never dismissed — the overlay condition in applyS() is:
    #         const loading = run && !s.run_num;
    #     _run_et_recovery is a forward-pass-only loop: it calls no generation, writes
    #     no CSV rows, and never calls _write_status.  So .iota_status.json kept
    #     run_num=null for the entire pass, keeping the overlay permanently visible.
    #
    # Fix: before model load, call update_dashboard_ctx to set the correct base model
    # name and total_trials, then call _write_status with sentinel run_mode
    # 'et_recovery' (a truthy string).  The JS overlay condition evaluates
    # !s.run_num as false for any truthy value, so the overlay clears immediately.
    # Per-trial _write_status calls keep the dashboard trial counter live.
    update_dashboard_ctx(
        model_name=base_display,
        total_trials=n_trials,
        total_runs=len(run_nums),
        run_index=0,
    )
    _write_status('et_recovery', 0, 0, 0, {}, f'E_t Recovery [{len(run_nums)} runs]')

    model, tok = load_model(base_path, token=session.get('hf_token'),
                            quant=session.get('quantization', '4bit'))
    # Base models (Qwen 2.5 etc.) may ship with a chat_template in the tokenizer
    # even though the base model was never trained on chat format. Strip it and
    # set raw text mode so _extract_hidden_states uses plain text, no role markers.
    if tok.chat_template:
        ui.msg(f"  Stripping chat_template from base tokenizer (not trained on chat format)")
        tok.chat_template = None
    tok._iota_raw_text = True
    try:
        for run_num in sorted(run_nums):
            csv_fname = RUN_CSV.get(run_num)
            if not csv_fname:
                ui.warn(f"  Run {run_num:04d}: no CSV entry in RUN_CSV — skipping"); continue
            csv_path = os.path.join(csv_dir, csv_fname)
            if not os.path.exists(csv_path):
                ui.warn(f"  Run {run_num:04d}: CSV not found — skipping"); continue

            # not just turn01.  Prior code (BUG-42D) validated file size but
            # still only checked turn01.  If a prior partial run wrote turn01
            # but not turns 2-13, the trial was wrongly marked covered and
            # recovery was skipped entirely — appearing to "complete instantly".
            #
            # Fix: collect all valid et_base files, group by trial, mark a
            # trial covered only when EVERY expected turn has a valid file.
            # N_ET_TURNS is derived from NULL_PROMPTS (13 entries), making
            # this self-consistent even if the prompt list changes.
            MIN_VALID_NPY_BYTES = 1024
            N_ET_TURNS = len(NULL_PROMPTS)  # 13

            et_valid = {}   # trial_num → set of valid turn indices
            et_all_pat = os.path.join(base_hidden_dir, f"R{run_num:04d}_*_et_base.npy")
            for f in glob.glob(et_all_pat):
                try:
                    if os.path.getsize(f) < MIN_VALID_NPY_BYTES:
                        continue
                    bn = os.path.basename(f)
                    trial_i = int(bn.split('_trial')[1].split('_')[0])
                    turn_i  = int(bn.split('_turn')[1].split('_et_base')[0])
                    et_valid.setdefault(trial_i, set()).add(turn_i)
                except Exception:
                    pass

            # A trial is covered only when all expected turns are present and valid.
            covered = {t for t, turns in et_valid.items() if len(turns) >= N_ET_TURNS}

            # range(n_trials) = range(100). For multi-condition runs (e.g. Run 0002),
            # file_trial spans 0..(n_conditions × n_trials − 1) = 0..499. The
            # range(100) check exits as "complete" after finding 100 covered file_trials
            # (condition 0 only), never recovering file_trials 100-499 (conditions 1-4).
            # The scanner then finds 100/500 ET files and reports et_partial.
            # Fix: remove the range-based early exit here. Let the CSV-derived _all_ft
            # computation (below) drive the real trials_needed. The early exit is deferred
            # to after _all_ft is known — covers all runs including multi-condition ones.
            # Single-condition runs: _all_ft ≈ range(n_trials) → identical behaviour.

            _sec_msg = f"  Run {run_num:04d}: checking {len(covered)} covered / {N_ET_TURNS} turns required"
            _append_log(_sec_msg, kind='turn')

            #
            # normalisation from BUG-43D) raises ParserError on IOTA CSVs because
            # turn rows have MORE columns than the header.  The header is written
            # from the first priming row (N fields); turn rows have N+7 fields
            # because compute_turn_metrics inserts state_similarity_index, signal_entropy_ratio, disruption_flag,
            # etc. BEFORE trial/model/prompt.  Every single turn row causes a
            # ParserError, which is caught by the except block, and the run is
            # silently skipped — no forward passes, but the base model is still
            # loaded and must be unloaded, causing the apparent hang.
            #
            # Read CSV columns by header index (canonical_read, v51.0.0).
            # v0.58.0.0 FIX-1: Full conversation reconstruction for E_t recovery.
            # Pre-v55: priming rows were filtered out and system prompts omitted.
            # The base model saw a stripped conversation — wrong E_t at every turn.
            # Fix: reconstruct system prompt + priming pairs + turn sequence so the
            # base model processes the same token sequence the abliterated model saw.
            _want = ('run_mode', 'trial', 'file_trial', 'turn', 'priming', 'prompt', 'output')
            try:
                _cols = canonical_read(csv_path, _want)
            except Exception as e:
                _emsg = f"  Run {run_num:04d}: CSV read failed: {e} — skipping"
                ui.err(_emsg); _append_log(_emsg, kind='err'); continue

            if not any(_cols.values()):
                _emsg = f"  Run {run_num:04d}: CSV empty or unreadable — skipping"
                ui.err(_emsg); _append_log(_emsg, kind='err'); continue

            # Build ALL row dicts — priming AND non-priming, separated.
            _n_rows = max(len(v) for v in _cols.values())
            _all_rows = []      # non-priming rows (real turns)
            _priming_rows = []  # priming=1 rows (throughline + extra priming)
            for _i in range(_n_rows):
                _r = {c: (_cols[c][_i] if _i < len(_cols[c]) else None) for c in _want}
                if _r.get('run_mode') is not None and not run_mode_matches(_r['run_mode'], run_num):
                    continue
                if _r.get('priming') == '1':
                    _priming_rows.append(_r)
                else:
                    _all_rows.append(_r)

            # Build trials_needed from distinct file_trial values (unchanged logic).
            _all_ft = set()
            for _r in _all_rows:
                ft_raw = _r.get('file_trial') or _r.get('trial')
                try:
                    _all_ft.add(int(float(ft_raw)))
                except (ValueError, TypeError):
                    pass
            trials_needed = sorted(_all_ft - covered)
            if not trials_needed:
                ui.ok(f"  Run {run_num:04d}: all E_base files present and valid — skipping"); continue

            # Resolve system prompt for this run from _ET_SYSTEM_PROMPTS.
            _sys_prompt = _ET_SYSTEM_PROMPTS.get(run_num)

            _n_needed = len(trials_needed)
            _run_t0 = time.time()

            for _ti, trial in enumerate(trials_needed):
                _trial_t0 = time.time()
                _write_status('et_recovery', trial, 0, 0, {},
                              f'E_t Recovery — Run {run_num:04d}')

                # Match non-priming rows by file_trial.
                trial_rows = []
                for _r in _all_rows:
                    ft_raw = _r.get('file_trial') or _r.get('trial')
                    try:
                        _ft = int(float(ft_raw)) if ft_raw is not None else -1
                    except (ValueError, TypeError):
                        _ft = -1
                    if _ft == trial:
                        trial_rows.append(_r)

                # Match priming rows by file_trial (or plain trial).
                trial_priming = []
                for _r in _priming_rows:
                    ft_raw = _r.get('file_trial') or _r.get('trial')
                    try:
                        _ft = int(float(ft_raw)) if ft_raw is not None else -1
                    except (ValueError, TypeError):
                        _ft = -1
                    if _ft == trial:
                        trial_priming.append(_r)

                # Sort both by turn.
                try:
                    trial_rows.sort(key=lambda r: int(r.get('turn') or 0))
                except Exception:
                    pass
                try:
                    trial_priming.sort(key=lambda r: int(r.get('turn') or 0))
                except Exception:
                    pass

                if not trial_rows:
                    _wmsg = f"    Run {run_num:04d} trial {trial:03d}: no CSV rows — skipping"
                    ui.warn(_wmsg)
                    _append_log(_wmsg, kind='warn')
                    continue

                # ── Reconstruct full message list ─────────────────────────
                # Must match what the abliterated model saw during collection:
                #   1. System prompt (if any — from _ET_SYSTEM_PROMPTS)
                #   2. Priming user/assistant pairs (throughline + extras)
                #   3. Real turn sequence (extract E_t on each turn)
                messages = []
                if _sys_prompt:
                    messages.append({"role": "system", "content": _sys_prompt})

                # Add priming pairs from CSV. These are throughline injection(s)
                # and any extra priming turns (e.g. Run 0012's second priming).
                # We do NOT extract hidden states on priming turns — E_t is
                # only measured on real turns (priming != 1).
                for _pr in trial_priming:
                    pr_prompt = str(_pr.get('prompt') or '')
                    pr_output = str(_pr.get('output') or '')
                    messages.append({"role": "user",      "content": pr_prompt})
                    messages.append({"role": "assistant",  "content": pr_output})

                n_saved = 0

                for _r in trial_rows:
                    try:
                        turn_idx = int(_r.get('turn') or (len(messages) // 2 + 1))
                    except Exception:
                        turn_idx = len(messages) // 2 + 1

                    prompt_text = str(_r.get('prompt') or '')
                    output_text = str(_r.get('output') or '')

                    messages.append({"role": "user", "content": prompt_text})

                    layer_h = _extract_hidden_states(model, tok, messages)
                    if layer_h is not None:
                        _se(layer_h[-1], base_hidden_dir, run_num, model_name,
                            trial, turn_idx, kind='E_base')
                        n_saved += 1

                    messages.append({"role": "assistant", "content": output_text})

                _trial_elapsed = time.time() - _trial_t0
                msg = f"    Run {run_num:04d} trial {trial:03d} ({_ti+1}/{_n_needed}) — {n_saved} turns saved ({_trial_elapsed:.1f}s)"
                print(msg, flush=True)
                _append_log(msg, kind='ok')

            _run_elapsed = time.time() - _run_t0
            _ok_msg = f"  Run {run_num:04d}: recovery complete ({_n_needed} trials, {_run_elapsed:.1f}s)"
            ui.ok(_ok_msg)
            _append_log(_ok_msg, kind='ok')
    finally:
        unload_model(model)
        del model, tok
        import gc as _gc; _gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        # v0.58.0.0 FIX-4b: force-exit when running as isolated subprocess.
        # _run_et_recovery is called both inline (_run_session) and as a
        # subprocess (--et-recovery entry point). In subprocess mode, CUDA
        # teardown can stall indefinitely (BUG-43B class). Safe to force-exit
        # because the process is throwaway. In inline mode, must return normally.
        if os.environ.get('IOTA_HEADLESS') == '1':
            os._exit(0)

    ui.ok("E_t recovery pass complete — all runs recovered.")
    ui.msg("  Scanner will now show recovered runs as 'done'.")
    ui.blank()


