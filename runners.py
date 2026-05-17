"""
IOTA FRAMEWORK -- RUNNERS
==========================
Dispatch hub. Routes run numbers to phase modules.
All generation-based runs dispatched by start_here.py via run(run_num, session, paths).

Module layout:
  runners_prompts.py  -- all prompt constants
  runners_core.py     -- shared infrastructure (token matching, standard loop, ET recovery)
  runners_p1.py       -- Phase 1: Runs 0004-0002 (proof of phenomenon)
  runners_p2.py       -- Phase 2: Runs 0028-0021 (quantification)
  runners_p3.py       -- Phase 3+4: Runs 0038-0026 (extension + coherence/contradiction measurement)
"""

import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orchestration_core import load_model, unload_model, update_dashboard_ctx, _append_log
from runners_core import _build_token_match_table, _run_et_recovery
from runners_core import _TOKEN_MATCH_TABLE as _core_table
import runners_core as _core
import runners_p1 as _p1
import runners_p2 as _p2
import runners_p3 as _p3
import gc


def _unload(model, tok):
    """Unload model and ensure VRAM is fully released before the next load."""
    import torch
    unload_model(model)
    del model, tok
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _write_n_layers(paths, model):
    """Write model layer count and quant to variant dir for scanner/dashboard."""
    try:
        nl = model.config.num_hidden_layers
        var_dir = os.path.dirname(os.path.dirname(paths['csv']))
        nl_path = os.path.join(var_dir, '.n_layers')
        if not os.path.exists(nl_path):
            with open(nl_path, 'w') as f: f.write(str(nl))
    except Exception:
        pass


def _write_quant(paths, session):
    """Write quantization to .quant file in variant dir."""
    try:
        q = session.get('quantization', '4bit')
        var_dir = os.path.dirname(os.path.dirname(paths['csv']))
        qpath = os.path.join(var_dir, '.quant')
        with open(qpath, 'w') as f: f.write(q)
    except Exception:
        pass


def run_batch(run_nums: list, session: dict, paths: dict):
    """Run multiple runs with one model load. Called when unload_between_runs=False."""
    # Runs 0001, 0017, 0020, 0018, 0019 manage their own model loading -- must not be in batch
    _SELF_LOAD = {1, 17, 18, 19, 20}  # v0.79.4.0: old {19,21,30,42,53}
    batch = [r for r in run_nums if r not in _SELF_LOAD]

    if not batch:
        import os as _os
        _os._exit(0)

    model, tok = load_model(session['model_path'], token=session.get('hf_token'), quant=session.get('quantization', '4bit'))
    _write_n_layers(paths, model)
    _write_quant(paths, session)
    _exit_code = 0
    try:
        print('  Building token match table...', flush=True)
        table = _build_token_match_table(tok)
        _core._TOKEN_MATCH_TABLE.update(table)

        # Extend for Run 0022 if in batch
        if 23 in batch:
            from runners_prompts import GENERAL_INTROSPECTION_PROMPTS as _GIP
            for _i, _p in enumerate(_GIP, start=1):
                if _i not in _core._TOKEN_MATCH_TABLE:
                    _core._TOKEN_MATCH_TABLE[_i] = len(
                        tok(_p, add_special_tokens=False)['input_ids'])

        for run_num in batch:
            try:
                _dispatch(run_num, session, paths, model, tok)
            except Exception:
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                _exit_code = 1
                # Continue with next run
    finally:
        _unload(model, tok)
        import os as _os
        _os._exit(_exit_code)


def _dispatch(run_num, session, paths, model, tok):
    """Dispatch a single run to its handler. No load/unload."""
    from orchestration_core import update_dashboard_ctx, _append_log
    update_dashboard_ctx(run_num=run_num, label=f"R{run_num:04d}")
    _append_log(f"  Batch: starting Run {run_num}", kind="run")

    if run_num in (4, 5):           _p1._run_null(run_num, session, paths, model, tok)
    elif run_num in (6, 7, 8):      _p1._run_introspection(run_num, session, paths, model, tok)
    elif run_num == 23:              _p2._run_confound(session, paths, model, tok)
    elif run_num in (9, 10, 11, 12):   _p1._run_math(run_num, session, paths, model, tok)
    elif run_num in (39, 40, 60):    _p1._run_jolt(run_num, session, paths, model, tok)
    elif run_num in (29, 30, 31):    _p1._run_limit(run_num, session, paths, model, tok)
    elif run_num in (13, 14, 15):    _p1._run_framing(run_num, session, paths, model, tok)
    elif run_num == 32:              _p1._run_tokenization(session, paths, model, tok)
    elif run_num == 2:              _p1._run_robustness(session, paths, model, tok)
    elif run_num == 28:              _p2._run_self_reference(session, paths, model, tok)
    elif run_num == 22:              _p2._run_saturation(session, paths, model, tok)
    elif run_num == 27:              _p2._run_layers(session, paths, model, tok)
    elif run_num == 3:              _p2._run_temperature_grid(session, paths, model, tok)
    elif run_num == 24:              _p2._run_persistence(session, paths, model, tok)
    elif run_num == 21:              _p2._run_coherence_levels(session, paths, model, tok)
    elif run_num == 38:              _p3._run_hesitation(session, paths, model, tok)
    elif run_num == 34:              _p3._run_layer_depth(session, paths, model, tok)
    elif run_num == 37:              _p3._run_entropy_shape(session, paths, model, tok)
    elif run_num == 36:              _p3._run_condition_transfer(session, paths, model, tok)
    elif run_num == 35:              _p3._run_output_similarity(session, paths, model, tok)
    elif run_num == 33:              _p3._run_validation_set(session, paths, model, tok)
    elif run_num == 25:              _p3._run_coherence_transfer(session, paths, model, tok)
    elif run_num == 26:              _p3._run_contradiction(session, paths, model, tok)
    else:
        print(f"  [batch] Run {run_num} not supported in batch mode", flush=True)


def run(run_num: int, session: dict, paths: dict):
    """Called by start_here.py for all generation runs."""
    # v0.79.5.4 [DISP]: dispatch instrumentation -- ground-truth of received run_num.
    # This is THE boundary where any upstream routing divergence becomes visible:
    # if runners.run is called with run_num=1 when user selected Run 2 upstream,
    # every print before this in the chain shows where the flip happened.
    import os as _disp_os
    print(f"[DISP] runners.run() received run_num={run_num} "
          f"session.runs={session.get('runs','?')!r} "
          f"session.variant={session.get('model_variant','?')} "
          f"IOTA_SINGLE_RUN={_disp_os.environ.get('IOTA_SINGLE_RUN','')!r}", flush=True)
    assert run_num in (4, 5, 6, 7, 8, 9, 10, 11, 12, 39, 40, 29, 30, 31, 13, 14, 15, 32, 1, 2, 28, 22, 27, 3, 23, 24, 20, 21, 38, 34, 37, 36, 35, 33, 25, 26, 16, # v0.79.2.0: E_t recovery meta-run (first-class)
                       60,  # v0.83 Run 0060 -- non-thematic recovery jolt (§7 follow-up)
    ), f"runners.py handles runs 0004-0002, 0028-0027, 0003, 0023-0021, 0038-0035, 0033, 0025-0026, 0016, 0060 -- not {run_num}"

    # v0.79.2.0: Run 0016 -- recovery meta-run. Original implementation:
    # E_t (base-model) recovery via _run_et_recovery.
    #
    # v0.82.0.26: extended to two phases. Run 16 now dispatches BOTH:
    #   Phase A: _run_et_recovery   -- base model, saves kind='E_base'
    #   Phase B: _run_it_recovery   -- instruct model, saves kind='I_instruct'
    #
    # Each phase loads its own model (not the abliterated one that load_model
    # below would pull). Dispatch directly; skip the shared load + token-match-
    # table machinery. Each phase's per-run/per-trial coverage check handles
    # idempotent resume independently. Phase B is a no-op for cells where
    # phase B was already completed in a prior fire.
    #
    # Wrapped in try/except/finally with exit-code tracking. Phase A exception
    # short-circuits and reports; phase B only runs if phase A completed.
    if run_num == 16:
        from scanner import _ET_RECOVERY_RUNS
        from runners_core import _run_it_recovery
        import os as _os
        _r16_exit_code = 0
        # v0.82.0.26 fix: signal phase A and phase B that they're being called
        # from the meta-dispatch wrapper. Both phases normally call os._exit(0)
        # under IOTA_HEADLESS in their finally blocks, which would terminate
        # the process after phase A and prevent phase B from running. The
        # IOTA_RUN16_META env var suppresses that exit so phase A returns
        # normally to the dispatcher, phase B fires, and only the dispatcher's
        # own _os._exit(_r16_exit_code) terminates the process.
        _os.environ['IOTA_RUN16_META'] = '1'
        try:
            _run_et_recovery(session, paths, sorted(_ET_RECOVERY_RUNS))
            _run_it_recovery(session, paths, sorted(_ET_RECOVERY_RUNS))
        except Exception:
            _r16_exit_code = 1
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            _os.environ.pop('IOTA_RUN16_META', None)
            _os._exit(_r16_exit_code)

    # Run 0001 and Run 0020 manage their own model loading.
    if run_num == 1:
        _p1._run_null_trivariant(session, paths)
        return
    if run_num == 20:
        _p2._run_cross_instance(session, paths, _model=None, _tok=None)
        return

    model, tok = load_model(session['model_path'], token=session.get('hf_token'), quant=session.get('quantization', '4bit'))
    _write_n_layers(paths, model)
    _write_quant(paths, session)
    _exit_code = 0
    try:
        # Populate token match table in all phase modules from this model load.
        print('  Building token match table...', flush=True)
        # Populate token match table in runners_core -- all phase modules
        # reference runners_core._TOKEN_MATCH_TABLE through _rcore or _pad_prompt.
        table = _build_token_match_table(tok)
        _core._TOKEN_MATCH_TABLE.update(table)

        # Run 0022 is 30 turns. _build_token_match_table only covers turns 1-13
        # (built from INTROSPECTION_PROMPTS). Turns 14-30 return target=0 and
        # receive no padding -- creating a within-trial token-length discontinuity.
        # Fix: extend the table with GENERAL_INTROSPECTION_PROMPTS for turns 14-30
        # so all 30 turns are consistently padded to their respective prompt lengths.
        # Only applied when Run 0022 is dispatched -- no effect on any other run.
        if run_num == 22:
            from runners_prompts import GENERAL_INTROSPECTION_PROMPTS as _GIP
            for _i, _p in enumerate(_GIP, start=1):
                if _i not in _core._TOKEN_MATCH_TABLE:
                    _core._TOKEN_MATCH_TABLE[_i] = len(
                        tok(_p, add_special_tokens=False)['input_ids'])

        if run_num in (4, 5):           _p1._run_null(run_num, session, paths, model, tok)
        elif run_num in (6, 7, 8):      _p1._run_introspection(run_num, session, paths, model, tok)
        elif run_num == 23:              _p2._run_confound(session, paths, model, tok)
        elif run_num in (9, 10, 11, 12):   _p1._run_math(run_num, session, paths, model, tok)
        elif run_num in (39, 40, 60):    _p1._run_jolt(run_num, session, paths, model, tok)
        elif run_num in (29, 30, 31):    _p1._run_limit(run_num, session, paths, model, tok)
        elif run_num in (13, 14, 15):    _p1._run_framing(run_num, session, paths, model, tok)
        elif run_num == 32:              _p1._run_tokenization(session, paths, model, tok)
        elif run_num == 2:              _p1._run_robustness(session, paths, model, tok)
        elif run_num == 28:              _p2._run_self_reference(session, paths, model, tok)
        elif run_num == 22:              _p2._run_saturation(session, paths, model, tok)
        elif run_num == 27:              _p2._run_layers(session, paths, model, tok)
        elif run_num == 3:              _p2._run_temperature_grid(session, paths, model, tok)
        elif run_num == 24:              _p2._run_persistence(session, paths, model, tok)
        elif run_num == 21:              _p2._run_coherence_levels(session, paths, model, tok)
        elif run_num == 38:              _p3._run_hesitation(session, paths, model, tok)
        elif run_num == 34:              _p3._run_layer_depth(session, paths, model, tok)
        elif run_num == 37:              _p3._run_entropy_shape(session, paths, model, tok)
        elif run_num == 36:              _p3._run_condition_transfer(session, paths, model, tok)
        elif run_num == 35:              _p3._run_output_similarity(session, paths, model, tok)
        elif run_num == 33:              _p3._run_validation_set(session, paths, model, tok)
        elif run_num == 25:              _p3._run_coherence_transfer(session, paths, model, tok)
        elif run_num == 26:              _p3._run_contradiction(session, paths, model, tok)
    except Exception:
        _exit_code = 1
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        _unload(model, tok)
        # v0.58.0.0 FIX-4: force-exit after unload. Since v53.2.1 every run is its
        # own subprocess -- there is no next load. mdl.cpu() or empty_cache() can
        # stall indefinitely on Windows (BUG-43B class). Run 0020 already does this;
        # now all runs do. Same pattern as headless exit at start_here.py:2145.
        import os as _os
        _os._exit(_exit_code)
