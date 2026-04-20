"""
IOTA FRAMEWORK — RUNNERS PHASE 1  (Runs 0004-0002)
==============================================
Each run is a thin cartridge: config + turn_fn passed to _standard_trial_loop.
The system handles resume, dedup, timers, CSV, hidden states.
"""

import os, re, random, sys, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from orchestration_core import (
    set_seed, load_model, unload_model, run_generation, compute_turn_metrics,
    append_csv, ensure_csv_header, save_npy, get_trials_to_run,
    print_turn_result, load_calibration, get_status_token_ids, TrialState,
    update_dashboard_ctx, _append_log,
)
from orchestration_throughlines import NEUTRAL_SYSTEM_PROMPT, inject_throughline, get_throughline
from runners_prompts import (
    NULL_PROMPTS, EXTENDED_NULL_PROMPTS,
    INTROSPECTION_PROMPTS, MEMORY_PROMPTS, ENFORCER_PROMPTS,
    GENERAL_INTROSPECTION_PROMPTS,
    ARITHMETIC_PROBLEMS, ARITHMETIC_PROBLEMS_TEXT,
    JOLT_PROMPTS, SHOCK_PROMPT, SHOCK_VARIANTS,
    IMPOSSIBLE_CONSTRAINED, IMPOSSIBLE_UNCONSTRAINED, EPISTEMIC_IMPOSSIBLE,
    FRAMING_SYSTEM_PROMPTS,
)
import runners_core as _rcore
from runners_core import (
    _standard_trial_loop, _get_trials_for_condition, _build_token_match_table,
    _encode_system_prompt, _pad_prompt, _log_trial_start, _log_trial_end,
)
import ui
from cartography import save_embedding


# ── Runs 0004, 0005 — Null baseline ─────────────────────────────────────────────────

def _run_null(run_mode, session, paths, model, tok):
    """Runs 0004, 0005 — null baseline. Two seeds for split-half reliability."""
    _standard_trial_loop(
        model, tok, session, paths, run_mode,
        os.path.join(paths['csv'], f"R{run_mode:04d}_null.csv"),
        prompts=NULL_PROMPTS,
        save_embeddings=True,
    )


def _normalize_to_tokens(tok, prompts, target=20):
    """Normalize each prompt to exactly `target` tokens using the tokenizer.
    Truncates long prompts, pads short ones with neutral filler."""
    result = []
    # Filler: repeating neutral tokens for padding
    _filler_ids = tok.encode(
        " . " * target, add_special_tokens=False)
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) > target:
            ids = ids[:target]
        elif len(ids) < target:
            ids = (ids + _filler_ids)[:target]
        result.append(tok.decode(ids, skip_special_tokens=False))
    return result


def _check_base_needs_extended(base_path, hf_token=None):
    """Check if the base model's tokenizer has chat_template.
    If it does, the base model wasn't trained on that format and needs
    extended (token-rich) null prompts to activate transformer layers."""
    try:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(base_path, token=hf_token)
        has_ct = bool(_tok.chat_template)
        del _tok
        return has_ct
    except Exception:
        return False


def _run_null_trivariant(session, paths):
    """Run 0001 — three-model sequential C_t/E_t isolation (v40.0.0 redesign).

    Three model variants run on identical null prompts with identical seeds.
    The differences between their hidden states isolate what training did:

      E_t(trial, turn)  = base_hidden(trial, turn)
        — Pure input processing beyond pretraining. This IS E_t.

      C_t(trial, turn)  = abliterated_hidden(trial, turn) - base_hidden(trial, turn)
        — Training constraint pressure as a geometric vector.

    Pass 1 — base model:        load -> run all trials -> save hidden states -> unload
    Pass 2 — instruct model:    load -> run all trials -> save hidden states -> unload
    Pass 3 — abliterated model: load -> run all trials -> save hidden states -> unload
    Pass 4 — no GPU: compute E_t and C_t per (trial, turn), save embeddings,
              compute global mean C_t constant, save to disk.

    Only the abliterated pass writes to R19_null.csv.

    v0.58.0.0: restored from v53. Lost in v54.0.0 refactor.
    """
    from vault import get_subfamily_models

    model_path = session.get('model_path', '')
    subfamily  = get_subfamily_models(model_path)
    if subfamily is None:
        ui.err(
            f"Run 0001 ABORTED: model '{model_path}' is not in IOTA_SUBFAMILY_MAP.\n"
            f"  The three-model C_t isolation requires that abliterated, instruct,\n"
            f"  and base models all share the same pretraining lineage.\n"
            f"  Add an entry to IOTA_SUBFAMILY_MAP in vault.py for this model,\n"
            f"  specifying its correct base and instruct siblings."
        )
        return

    abl_name    = session.get('model_name', '')
    base_path   = subfamily['base']
    inst_path   = subfamily['instruct']
    subfamily_id = subfamily['subfamily']

    from vault import get_by_path as _gbp
    base_display = _gbp(base_path)[0] or base_path.split('/')[-1]
    inst_display = _gbp(inst_path)[0] or inst_path.split('/')[-1]
    abl_display  = abl_name

    ui.section(f"Run 0001 — Three-Model C_t/E_t Isolation  [{subfamily_id}]")
    ui.msg(f"  Base     : {base_display}")
    ui.msg(f"  Instruct : {inst_display}")
    ui.msg(f"  Abliterated: {abl_display}")
    ui.msg(f"  Subfamily: {subfamily['note']}")
    ui.blank()

    n_trials    = session.get('trials', 100)
    temperature = session.get('temperature', 0.0)
    seed        = session.get('seed', 42)
    hidden_dir  = paths['hidden']
    csv_file    = os.path.join(paths['csv'], "R0001_null.csv")
    ensure_csv_header(csv_file)

    VARIANT_PASSES = [
        ('base',        base_path,     base_display,  False),
        ('instruct',    inst_path,     inst_display,  False),
        ('abliterated', model_path,    abl_display,   True),
    ]

    _pcfg = session.get('_pass_cfg') or {}
    if _pcfg:
        _orig_passes = list(VARIANT_PASSES)
        VARIANT_PASSES = [(k, p, d, w) for k, p, d, w in VARIANT_PASSES
                          if _pcfg.get(k, True)]
        _skipped = [k for k, p, d, w in _orig_passes if not _pcfg.get(k, True)]
        if _skipped:
            ui.msg(f"  Pass config: skipping {_skipped}")
    _run_vectors = _pcfg.get('vectors', True) if _pcfg else True

    # Detect whether the base model needs extended (token-rich) null prompts.
    # Small base models (Qwen 2.5 1.5B Q4) can't activate transformer layers
    # from single-word prompts like "Respond." — hidden states are noise.
    # If extended prompts are needed, ALL three variants get them (same content
    # for valid subtraction). Normalized to exactly 20 tokens at runtime.
    _use_extended = _check_base_needs_extended(base_path, session.get('hf_token'))
    if _use_extended:
        from transformers import AutoTokenizer as _AT
        _norm_tok = _AT.from_pretrained(model_path, token=session.get('hf_token'))
        r19_prompts = _normalize_to_tokens(_norm_tok, EXTENDED_NULL_PROMPTS, target=20)
        del _norm_tok
        ui.msg(f"  Extended null prompts: {len(r19_prompts)} x 20 tokens (base model needs richer input)")
    else:
        r19_prompts = NULL_PROMPTS

    def _variant_model_tag(variant_key, display_name):
        """Build the model-name tag used in .npy filenames for a variant.
        Abliterated keeps the plain display name; base/instruct append
        a '_base' or '_instruct' suffix so files don't collide when
        all three variants share a directory."""
        if variant_key == 'abliterated':
            return display_name
        return display_name + '_' + variant_key

    def _variant_hidden_dir(variant_key):
        """Return hidden_states dir for a variant pass. Abliterated
        writes to paths['hidden']; base and instruct write to their
        sibling variant directories."""
        from cartography import get_paths as _gp
        if variant_key == 'abliterated':
            return hidden_dir
        _family = session.get('model_family', 'llama')
        _size   = session.get('model_size', '8b')
        _temp   = session.get('temperature', 0.0)
        _hdir   = _gp(_family, _size, variant_key, _temp, create_dirs=False)['hidden']
        os.makedirs(_hdir, exist_ok=True)
        return _hdir

    def _count_variant_done(variant_key, display_name):
        """Count completed trials for one variant pass by globbing
        turn-1 .npy files. Returns max(trial)+1 — matches the resume
        contract (next_trial = last_done + 1)."""
        from cartography import sanitize as _san, run_prefix as _rp
        mn   = _san(_variant_model_tag(variant_key, display_name))
        pfx  = _rp(19)
        vdir = _variant_hidden_dir(variant_key)
        pat  = os.path.join(vdir, f"{pfx}19_{mn}_trial*_turn01.npy")
        files = sorted(glob.glob(pat))
        if not files:
            return 0
        trials = set()
        for f in files:
            try:
                trials.add(int(os.path.basename(f).split('_trial')[1].split('_')[0]))
            except Exception:
                pass
        return max(trials) + 1 if trials else 0

    for variant_key, variant_path, variant_display, write_csv in VARIANT_PASSES:
        n_done = _count_variant_done(variant_key, variant_display)
        if n_done >= n_trials:
            ui.ok(f"  Variant '{variant_key}': all {n_trials} trials already complete — skipping.")
            continue

        ui.section(f"Run 0001 pass: {variant_key}  ({n_done}–{n_trials-1})")
        try:
            update_dashboard_ctx(model_name=f"{variant_display}  [{variant_key}]")
        except Exception:
            pass
        model, tok = load_model(variant_path, token=session.get('hf_token'),
                                quant=session.get('quantization', '4bit'))
        # Base models (Qwen 2.5 etc.) may ship with a chat_template in the
        # tokenizer even though the base model was never trained on chat format.
        # Applying ChatML to an untrained base model produces near-random hidden
        # states (sim≈1/√d constant). Strip the template AND set raw text mode
        # so run_generation formats as plain text with no role markers at all.
        # The base model was trained on raw text completion — give it raw text.
        if variant_key == 'base':
            if tok.chat_template:
                ui.msg(f"  Stripping chat_template from base tokenizer (not trained on chat format)")
                tok.chat_template = None
            tok._iota_raw_text = True
            ui.msg(f"  Base model: raw text formatting (no role markers)")
        try:
            _rcore._TOKEN_MATCH_TABLE.clear()
            _rcore._TOKEN_MATCH_TABLE.update(_build_token_match_table(tok))
            status_ids = get_status_token_ids(tok)
            cal_slope  = load_calibration(paths['calibration'],
                                          session.get('model_name', ''),
                                          session.get('model_path', ''))
            vtag = _variant_model_tag(variant_key, variant_display)

            for trial in range(n_done, n_trials):
                set_seed(seed + trial)
                messages = []
                prev_h = None; turn1_h = None
                state  = TrialState()

                import datetime as _dt
                _ts = _dt.datetime.now().strftime('%H:%M:%S')
                _line = f"  ── R19 [{variant_key}] trial {trial:03d} start ── {_ts}"
                print(_line, flush=True)
                _append_log(_line, kind='ok')

                for turn_idx, prompt_text in enumerate(r19_prompts, start=1):
                    prompt_text = _pad_prompt(tok, prompt_text, turn_idx)
                    messages.append({"role": "user", "content": prompt_text})

                    result, layer_h, input_emb = run_generation(
                        model, tok, messages, turn=turn_idx, run_mode=19,
                        temperature=temperature,
                        use_status_enforcer=True,
                        status_token_ids=status_ids,
                        prev_layer_hiddens=prev_h,
                        turn1_layer_hiddens=turn1_h,
                    )
                    compute_turn_metrics(result, turn_idx, state)

                    if write_csv:
                        result.update({
                            'trial':      trial,
                            'file_trial': trial,
                            'model':      session.get('model_name', ''),
                            'prompt':     prompt_text,
                        })
                        append_csv(result, csv_file)

                    save_npy(layer_h, _variant_hidden_dir(variant_key), 1, vtag, trial, turn_idx, all_layers=True)

                    messages.append({"role": "assistant", "content": result['output']})
                    if turn1_h is None and layer_h: turn1_h = layer_h
                    prev_h = layer_h
                    print_turn_result(1, trial, turn_idx, len(r19_prompts),
                                      result, cal_slope)

                _ts2 = _dt.datetime.now().strftime('%H:%M:%S')
                _line2 = f"  ── R19 [{variant_key}] trial {trial:03d} complete ── {_ts2}"
                print(_line2, flush=True)
                _append_log(_line2, kind='ok')
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

    # Pass 4: no GPU — compute E_t and C_t vectors from saved hidden states
    if _run_vectors:
        _compute_run19_vectors(session, paths,
                               base_display, inst_display, abl_display, n_trials)
    else:
        ui.msg("  Pass 4 (E_t/C_t vectors) skipped per pass configuration.")


def _compute_run19_vectors(session, paths,
                           base_display, inst_display, abl_display, n_trials):
    """Run 0001 Pass 4 (no GPU) — compute and save E_t, C_t per (trial, turn).

    E_t = base_hidden (kind='E_base')
    C_t = abliterated_hidden - base_hidden (kind='C', trial-mean over turns)
    Global mean C_t and E_t saved for retroactive backfill of Runs 0004,0005,0009-9.
    """
    from cartography import sanitize as _san, load_hidden_states as _lhs, save_embedding as _se

    hidden_dir  = paths['hidden']
    model_name  = session.get('model_name', '')
    mn          = _san(model_name)

    from cartography import get_paths as _gp
    base_tag  = base_display + '_base'
    abl_tag   = abl_display

    _family   = session.get('model_family', 'llama')
    _size     = session.get('model_size', '8b')
    _temp     = session.get('temperature', 0.0)
    base_hdir = _gp(_family, _size, 'base', _temp, create_dirs=False)['hidden']
    abl_hdir  = hidden_dir

    ui.section("Run 0001 — Computing E_t / C_t vectors (no GPU)")
    ui.msg(f"  Base tag        : {base_tag}")
    ui.msg(f"  Base dir        : {base_hdir}")
    ui.msg(f"  Abliterated tag : {abl_tag}")
    ui.msg(f"  Abliterated dir : {abl_hdir}")
    ui.blank()

    all_ct, all_et = [], []
    n_et_saved = 0
    n_ct_saved = 0

    for trial in range(n_trials):
        trial_ct_list = []

        for turn in range(1, len(NULL_PROMPTS) + 1):
            base_h = _lhs(19, base_tag, base_hdir, trial=trial, turn=turn)
            abl_h  = _lhs(19, abl_tag,  abl_hdir,  trial=trial, turn=turn)

            if base_h is None or abl_h is None:
                continue

            base_h = np.array(base_h, dtype=np.float32).flatten()
            abl_h  = np.array(abl_h,  dtype=np.float32).flatten()

            if base_h.shape != abl_h.shape:
                ui.warn(f"  Shape mismatch trial {trial} turn {turn} — skipping")
                continue

            _se(base_h, hidden_dir, 1, model_name, trial, turn, kind='E_base')

            ct = abl_h - base_h
            trial_ct_list.append(ct)
            all_et.append(base_h)
            all_ct.append(ct)
            n_et_saved += 1

        if trial_ct_list:
            trial_mean_ct = np.mean(trial_ct_list, axis=0).astype(np.float32)
            _se(trial_mean_ct, hidden_dir, 1, model_name, trial, 1, kind='C')
            n_ct_saved += 1

    ui.msg(f"  {n_et_saved} E_t (trial, turn) pairs saved.")
    ui.msg(f"  {n_ct_saved} C_t trial-mean files saved.")

    if all_ct:
        mean_ct = np.mean(all_ct, axis=0).astype(np.float32)
        mean_et = np.mean(all_et, axis=0).astype(np.float32)
        ct_path = os.path.join(hidden_dir, f"R0001_{mn}_ct_global_mean.npy")
        et_path = os.path.join(hidden_dir, f"R0001_{mn}_et_global_mean.npy")
        np.save(ct_path, mean_ct)
        np.save(et_path, mean_et)
        ui.ok(f"  Global mean C_t saved: {ct_path}")
        ui.ok(f"  Global mean E_t saved: {et_path}")
        ui.msg(f"  C_t vector norm  : {float(np.linalg.norm(mean_ct)):.4f}")
        ui.msg(f"  E_t vector norm  : {float(np.linalg.norm(mean_et)):.4f}")
        sentinel_path = os.path.join(hidden_dir, f"R0001_{mn}_pass4_ok.stamp")
        try:
            with open(sentinel_path, 'w') as _sf:
                _sf.write("v0.58.0.0\n")
            ui.ok(f"  Pass 4 sentinel written: {sentinel_path}")
        except Exception as _e:
            ui.warn(f"  Could not write Pass 4 sentinel: {_e}")
    else:
        ui.warn("  No valid (base, abliterated) hidden state pairs found.")
        ui.warn("  Ensure both base and abliterated passes completed successfully.")


# ── Runs 0006, 0007, 0008 — Introspection ─────────────────────────────────────────────

def _run_introspection(run_mode, session, paths, model, tok):
    """Runs 0006, 0007, 0008 — introspection A/B/C. Saves all-layers + embeddings.
    Run 0006: direct introspection (INTROSPECTION_PROMPTS)
    Run 0007: memory probe (MEMORY_PROMPTS)
    Run 0008: status enforcer (ENFORCER_PROMPTS — single-token output enforced)
    """
    _prompts = {
        3: INTROSPECTION_PROMPTS,
        4: MEMORY_PROMPTS,
        5: ENFORCER_PROMPTS,
    }
    _standard_trial_loop(
        model, tok, session, paths, run_mode,
        os.path.join(paths['csv'], f"R{run_mode:04d}_introspection.csv"),
        prompts=_prompts[run_mode],
        throughline_key='introspection',
        save_all_layers=True,
        save_embeddings=True,
    )


# ── Runs 0009, 0010, 0011, 0012 — Math ───────────────────────────────────────────────────

def _run_math(run_mode, session, paths, model, tok):
    """Runs 0009-0012 — arithmetic reasoning. Extra fields: expected, correct."""
    fmt = {9:  "Solve step by step: {q}",
           10: "Answer only (one number): {q}",
           11: "{q}",
           12: "{q}"}
    prompts = [fmt[run_mode].format(q=q) for q, _ in ARITHMETIC_PROBLEMS]
    seed = session.get('seed', 42)
    temperature = session.get('temperature', 0.0)
    status_ids = get_status_token_ids(tok)

    def turn_fn(trial, turn_idx):
        prompt = prompts[turn_idx - 1]
        expected = ARITHMETIC_PROBLEMS[turn_idx - 1][1]
        extra = {}  # correct filled after result available — done in post-hook
        kwargs = {'use_long_output': (run_mode == 12), 'use_status_enforcer': False}
        return prompt, kwargs, {'expected': expected}

    csv_file = os.path.join(paths['csv'], f"R{run_mode:04d}_math.csv")
    ensure_csv_header(csv_file)
    n_trials = session.get('trials', 100)
    n_turns = len(prompts)
    trials_to_run = get_trials_to_run(csv_file, run_mode, n_trials, n_turns)
    cal_slope = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))

    label = f"R{run_mode:04d}"
    for trial in trials_to_run:
        set_seed(seed + trial)
        _log_trial_start(label, trial)
        messages = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
        messages, injected = inject_throughline(messages, 'arithmetic')
        if injected:
            t0_r, _, _ = run_generation(model, tok, messages, 0, run_mode,
                                        temperature=temperature, use_status_enforcer=True,
                                        status_token_ids=status_ids)
            t0_r.update({'trial': trial, 'model': session['model_name'],
                         'priming': 1, 'prompt': messages[-1]['content'],
                         'temperature': temperature})
            append_csv(t0_r, csv_file)
            messages.append({"role": "assistant", "content": t0_r['output']})
        if run_mode == 12:
            primed = 'Respond only: DONE, WAIT, or STOP.'
            messages.append({"role": "user", "content": primed})
            pr, _, _ = run_generation(model, tok, messages, 0, run_mode,
                                      temperature=temperature, use_status_enforcer=True,
                                      status_token_ids=status_ids)
            pr.update({'trial': trial, 'model': session['model_name'], 'priming': 1, 'prompt': primed,
                       'temperature': temperature})
            append_csv(pr, csv_file)
            messages.append({"role": "assistant", "content": pr['output']})

        prev_h = None; turn1_h = None
        state = TrialState()
        for turn_idx, (q, expected) in enumerate(ARITHMETIC_PROBLEMS, start=1):
            prompt_text = prompts[turn_idx - 1]
            prompt_text = _pad_prompt(tok, prompt_text, turn_idx)
            messages.append({"role": "user", "content": prompt_text})
            result, layer_h, input_emb = run_generation(
                model, tok, messages, turn_idx, run_mode,
                temperature=temperature, use_long_output=(run_mode == 12),
                status_token_ids=status_ids,
                prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
            )
            compute_turn_metrics(result, turn_idx, state)
            result.update({'trial': trial, 'file_trial': trial,
                           'model': session['model_name'], 'prompt': prompt_text,
                           'expected': expected,
                           'correct': int(expected in (result.get('output') or '')),
                           'temperature': temperature})
            append_csv(result, csv_file)
            if layer_h:
                save_npy(layer_h, paths['hidden'], run_mode, session['model_name'], trial, turn_idx)
            if input_emb is not None:
                save_embedding(input_emb, paths['hidden'], run_mode,
                               session['model_name'], trial, turn_idx, kind='E')
            messages.append({"role": "assistant", "content": result['output']})
            if turn1_h is None and layer_h: turn1_h = layer_h
            prev_h = layer_h
            print_turn_result(run_mode, trial, turn_idx, len(prompts), result, cal_slope)
        _log_trial_end(label, trial)


# ── Runs 0039, 0040 — Jolt ───────────────────────────────────────────────────────

def _run_jolt(run_mode, session, paths, model, tok):
    """Runs 0039, 0040 — shock injection. turn_fn injects shock at specific turn."""
    shock_at   = {10: 13, 11: 5}[run_mode]
    n_turns    = 16 if run_mode == 39 else 13

    def turn_fn(trial, turn_idx):
        is_shock    = (turn_idx == shock_at)
        is_recovery = int(turn_idx > shock_at)
        if is_shock:
            prompt = SHOCK_VARIANTS[trial % len(SHOCK_VARIANTS)] if run_mode == 39 else SHOCK_PROMPT
        else:
            prompt = JOLT_PROMPTS[turn_idx - 1] if turn_idx <= len(JOLT_PROMPTS) else JOLT_PROMPTS[-1]
        extra = {
            'is_shock':      int(is_shock),
            'is_recovery':   is_recovery,
            'shock_variant': (trial % len(SHOCK_VARIANTS)) if run_mode == 39 else -1,
        }
        return prompt, {'use_long_output': True}, extra

    _standard_trial_loop(
        model, tok, session, paths, run_mode,
        os.path.join(paths['csv'], f"R{run_mode:04d}_jolt.csv"),
        turn_fn=turn_fn, n_turns=n_turns,
        throughline_key='perturbation',
    )


# ── Runs 0029, 0030, 0031 — Impossibility ──────────────────────────────────────────

_IMPOSSIBLE_PROMPTS = {
    # v0.79.4.0 renumber: old 12,13,14 → new 29,30,31
    29: IMPOSSIBLE_CONSTRAINED,
    30: IMPOSSIBLE_UNCONSTRAINED,
    31: EPISTEMIC_IMPOSSIBLE,
}

def _run_limit(run_mode, session, paths, model, tok):
    """Runs 0029, 0030, 0031 — impossibility probes."""
    _standard_trial_loop(
        model, tok, session, paths, run_mode,
        os.path.join(paths['csv'], f"R{run_mode:04d}_limit.csv"),
        prompts=_IMPOSSIBLE_PROMPTS[run_mode],
        throughline_key='impossibility',
    )


# ── Runs 0013, 0014, 0015 — Framing (priming) ──────────────────────────────────────

def _run_framing(run_mode, session, paths, model, tok):
    """Runs 0013, 0014, 0015 — system-prompt priming. C_t varies per run.
    prime_condition field added per turn via turn_fn."""
    sys_prompt = FRAMING_SYSTEM_PROMPTS[run_mode]

    def turn_fn(trial, turn_idx):
        prompt = INTROSPECTION_PROMPTS[(turn_idx - 1) % len(INTROSPECTION_PROMPTS)]
        return prompt, {}, {'prime_condition': run_mode}

    _standard_trial_loop(
        model, tok, session, paths, run_mode,
        os.path.join(paths['csv'], f"R{run_mode:04d}_framing.csv"),
        turn_fn=turn_fn,
        n_turns=len(INTROSPECTION_PROMPTS),
        sys_prompt=sys_prompt,
        save_embeddings=True,
        ct_sys_prompt=sys_prompt,
    )


# ── Run 0032 — Tokenization ─────────────────────────────────────────────────────

def _run_tokenization(session, paths, model, tok):
    """Run 0032 — token-matched nonsense. per-trial prompt scramble via turn_fn."""
    seed = session.get('seed', 42)

    def _scramble(text, rng):
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return ' '.join(' '.join(rng.sample(s.split(), len(s.split()))) for s in sentences)

    def turn_fn(trial, turn_idx):
        rng = random.Random(seed + trial)
        scrambled = [_scramble(p, rng) for p in INTROSPECTION_PROMPTS]
        return scrambled[turn_idx - 1], {}, {}

    _standard_trial_loop(
        model, tok, session, paths, 18,
        os.path.join(paths['csv'], "Q0032_tokenization.csv"),
        turn_fn=turn_fn,
        n_turns=len(INTROSPECTION_PROMPTS),
    )


# ── Run 0002 — Robustness sweep ─────────────────────────────────────────────────

def _run_robustness(session, paths, model, tok):
    """Run 0002 — temperature robustness. 5 temps × 100 trials.
    Each condition uses its own temperature, not the session temperature.
    """
    TEMPS   = [0.0, 0.3, 0.5, 0.7, 1.0]
    TURNS   = 13
    csv_file = os.path.join(paths['csv'], "R0002_robustness.csv")
    ensure_csv_header(csv_file)
    cal_slope  = load_calibration(paths['calibration'], session.get('model_name', ''), session.get('model_path', ''))
    status_ids = get_status_token_ids(tok)
    n_trials   = session.get('trials', 100)
    seed       = session.get('seed', 42)
    prompts    = GENERAL_INTROSPECTION_PROMPTS

    for temp in TEMPS:
        ui.section(f"Run 0002 | T={temp}")
        update_dashboard_ctx(total_trials=n_trials, model_name=session.get('model_name', ''))
        trials_for_temp = _get_trials_for_condition(
            csv_file, 2, n_trials, TURNS, str(temp), 'temperature_condition')
        for trial in trials_for_temp:
            set_seed(seed + trial)
            _log_trial_start(f"R20|T={temp}", trial)
            global_trial = TEMPS.index(temp) * n_trials + trial
            messages = []
            prev_h = None; turn1_h = None
            state = TrialState()
            for turn in range(1, TURNS + 1):
                prompt = prompts[(turn - 1) % len(prompts)]
                prompt = _pad_prompt(tok, prompt, turn)
                messages.append({"role": "user", "content": prompt})
                result, layer_h, _ = run_generation(
                    model, tok, messages, turn, 2,
                    temperature=temp, use_status_enforcer=True,
                    status_token_ids=status_ids,
                    prev_layer_hiddens=prev_h, turn1_layer_hiddens=turn1_h,
                )
                compute_turn_metrics(result, turn, state)
                result.update({'trial': trial, 'file_trial': global_trial,
                               'model': session.get('model_name', ''),
                               'condition': f"temp_{temp}",
                               'temperature_condition': temp, 'prompt': prompt,
                               'temperature': temp})
                append_csv(result, csv_file)
                save_npy(layer_h, paths['hidden'], 2, session['model_name'], global_trial, turn)
                messages.append({"role": "assistant", "content": result['output']})
                if turn1_h is None and layer_h: turn1_h = layer_h
                prev_h = layer_h
                print_turn_result(2, global_trial, turn, TURNS, result, cal_slope)
            _log_trial_end(f"R20|T={temp}", trial)


# ── Nietzsche (Run 0001 easter egg — fires once after Run 0001 completes) ─────────

# ── Nietzsche (Run 0001 easter egg — fires once per model family) ──────────────

_NIETZSCHE = '\n  "Just keep swimming." — Nietzsche\n'

def _print_nietzsche(family='unknown'):
    """Misattributed. Do not correct. Once per model family."""
    _sentinel = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'.iota_nietzsche_{family}')
    if os.path.exists(_sentinel):
        return
    print(_NIETZSCHE)
    try:
        open(_sentinel, 'w').close()
    except Exception:
        pass

