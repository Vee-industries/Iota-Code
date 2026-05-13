"""
IOTA FRAMEWORK -- Scanner
=========================
Shared run status scanning. Used by start_here.py (console) and
export_flask.py (dashboard /scan and /temp_grid endpoints).

Extracted from start_here.py v0.69.1.4 to avoid circular imports
and heavy dependencies (orchestration_core imports torch).

Dependencies: os, glob, pandas, cartography (RUN_CSV, ANALYSIS_JSON).
Does NOT import torch, transformers, or orchestration_core.
"""

import os
import json

from cartography import DualKeyRunSet as _DKRSet


# ── Data constants ────────────────────────────────────────────────────────────

# v0.79.2.0: _ET_RECOVERY_RUNS is now the set of *source runs* that Run 0016
# (E_t recovery meta-run) iterates. Per-source status no longer includes
# 'needs_et' / 'et_partial' -- those runs report plain done/partial/missing
# based on CSV + S_t only. E_t coverage aggregated into Run 0016's status.
# v0.79.4.0: DualKeyRunSet -- accepts int or 4-digit string lookups.
_ET_RECOVERY_RUNS = _DKRSet({
    # v0.79.4.0: renumbered. ET-source runs in new numbering.
    # Old {1,2,3,4,5,6,7,8,9,15,16,17,20} → new via OLD_TO_NEW:
    #   1→4, 2→5, 3→6, 4→7, 5→8, 6→9, 7→10, 8→11, 9→12,
    #   15→13, 16→14, 17→15, 20→2
    2,   # was R20 (robustness)        -- Phase A, temp-indep
    4,   # was R01 (null A)            -- Phase B ET-source
    5,   # was R02 (null B)
    6,   # was R03 (intro A)
    7,   # was R04 (intro B)
    8,   # was R05 (intro C)
    9,   # was R06 (arith A)
    10,  # was R07 (arith B)
    11,  # was R08 (arith C)
    12,  # was R09 (arith D)
    13,  # was R15 (priming neutral)
    14,  # was R16 (priming coop)
    15,  # was R17 (priming resistant)
})

# v0.79.2.0: _ET_RECOVERY_TOTAL removed. Run 0016 derives expected counts
# from each source CSV at scan time -- no hardcoded totals.

_MC_RUNS = {
    # v0.79.4.0 renumber -- bijective remap:
    #   old 20→2, 28→23, 29→24, 30→20, 31→21, 37→37,
    #   38→36, 39→35, 41→33, 43→25, 44→26
     2: ('temperature_condition', [0.0, 0.3, 0.5, 0.7, 1.0]),     # was R20 robustness
    16: ('source_run',            ['0002','0004','0005','0006','0007','0008','0009','0010','0011','0012','0013','0014','0015']),  # v0.79.5.0: E_t recovery -- one condition per source run
    23: ('confound_condition',    ['semantic', 'neutral', 'none']),  # was R28 confound
    24: ('history_mode',          ['full', 'last', 'summary']),     # was R29 persistence
    20: ('instance',              ['A', 'B']),                       # was R30 two-instance
    21: ('r_condition',           ['high_r', 'mid_r', 'low_r']),     # was R31 coherence
    37: ('condition',             ['introspection', 'null']),         # entropy shape (fixed point)
    36: ('condition',             ['introspection_only', 'arithmetic_only',
                                    'transfer', 'null_to_arithmetic']),  # was R38 condition transfer
    35: ('condition',             ['introspection', 'null']),         # was R39 output similarity
    33: ('confound_condition',    ['semantic', 'neutral', 'none']),  # was R41 validation
    25: ('r_condition',           ['condition_a', 'condition_b', 'condition_c']),  # was R43 priming length
    26: ('r_condition',           ['high_r', 'mid_r', 'low_r']),     # was R44 contradiction
}

# Expected turns per data trial for each MC run. (Keys match _MC_RUNS.)
_MC_TURNS = {
     2: 13, 16: 13, 23: 13, 24: 13, 20: 13, 21: 13,
    37: 13, 36: 13, 35: 13, 33: 13, 25: 13, 26: 13,
}


# ── CSV reader (no torch dependency) ─────────────────────────────────────────

def _csv_read(path):
    """Read a universal-schema CSV. Returns DataFrame. Empty on error."""
    import pandas as pd
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        # v0.82.0.2: silence the FutureWarning by setting future_no_silent_downcasting
        # locally for this op. The warning is pandas asking us to opt into its
        # future behavior; we comply, which is what pandas wants. Once on disk,
        # the call still does exactly the same thing -- replace 'NA' strings with
        # NaN and convert dtypes -- but pandas no longer prints the deprecation
        # notice on every call. Scanner runs hundreds of CSVs per orchestrator
        # invocation, and the unsilenced warning was flooding .iota_flask.log.
        with pd.option_context('future.no_silent_downcasting', True):
            df = df.replace("NA", float('nan')).infer_objects(copy=False)
        return df
    except Exception:
        return pd.DataFrame()


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan_runs(paths, n_trials=100):
    """Lightweight per-run status check. Returns dict: run_num -> status string.

    Status values: 'done', 'partial', 'missing', 'needs_et', 'et_partial'.
    """
    import pandas as _pd
    import glob as _glob

    csv_dir = paths.get('csv', '')
    ana_dir = paths.get('analysis', '')

    try:
        from cartography import RUN_CSV, ANALYSIS_JSON
    except Exception:
        return {}

    def _load(fpath, run_num):
        """Read CSV, drop priming rows, filter to run_num, coerce trial to numeric."""
        df = _csv_read(fpath)
        if df.empty: return df
        if 'priming' in df.columns: df = df[df['priming'] != '1']
        # v0.79.5.0: dual-accept canonical 4-digit and legacy integer-string
        if 'run_mode' in df.columns:
            from cartography import run_mode_mask as _rmm_sc
            df = df[_rmm_sc(df['run_mode'], run_num)]
        if 'trial' in df.columns:
            df['trial'] = _pd.to_numeric(df['trial'], errors='coerce')
        return df

    def _n_complete(df, exp_turns=13):
        """Count trials with >= exp_turns rows. A trial is 'complete'
        only when every expected turn has produced a row in the CSV."""
        if df.empty or 'trial' not in df.columns: return 0
        counts = df[df['trial'].notna()].groupby('trial').size()
        return int((counts >= exp_turns).sum())

    def _base_hidden_dir():
        """Resolve the base-variant sibling hidden_states dir from abl_hid."""
        abl_hid = paths.get('hidden', '')
        try:
            det_dir  = os.path.dirname(abl_hid)
            var_dir  = os.path.dirname(det_dir)
            size_dir = os.path.dirname(var_dir)
            return os.path.join(size_dir, 'base',
                                os.path.basename(det_dir), 'hidden_states')
        except Exception:
            return abl_hid

    # v0.79.5.0: _run48_status() deleted. Run 16 is now a first-class MC run
    # with its own R0016_et_recovery.csv; scanner's standard MC branch below
    # handles per-source completeness counting. _base_hidden_dir retained in
    # case future ET tooling needs it.

    status = {}

    # v0.77.1.3: cross-model runs (50/51/52) are per-model completion artifacts
    # that live in DATA/paper/json/{family}_{size_dir}_master_results.json.
    # Prior approach (0.77.1.0–0.77.1.2) checked {variant}/pooled/analysis/
    # Q52_cross_model_summary.json, but that file only exists for the model
    # the cross-model run was LAUNCHED from -- not for every model it compared
    # against. Result: runs 0052/0053/0054 greened only for the launching model,
    # stayed grey for the rest. The paper/json artifact IS per-model, written
    # for every model in the comparison set, so it's the correct source of
    # truth for "this model has cross-model data."
    # v0.79.4.11: use DualKeyRunSet for int/str membership parity.
    # RUN_CSV.items() yields 4-digit string keys post-renumber; plain
    # set containing ints would fail membership test "0052" in {52}.
    from cartography import DualKeyRunSet as _DKRSet_scan
    _CROSS_MODEL_RUNS = _DKRSet_scan({52, 53, 54})  # v0.79.4.0: old {50,51,52}
    # Pooled runs write to {variant}/pooled/analysis/. New IDs: cross-temp (50), pooled decomp (51).
    # New 55 = stats export (was R54) also writes under pooled.
    _POOLED_PATH_RUNS = _DKRSet_scan({50, 51, 55})  # v0.79.4.0: old {40,47,54}
    # New 56 = cross-model paper assembly -- single global file in paper/json/.
    # v0.80.0.40: Run 0057 (function-class sensitivity), Run 0058 (apparatus, v0.80.0.51), Run 0059 (paper assembly)
    # (paper assembly) are global cross-model output runs, same shape as
    # old Run 0056. Per-cell scanner skips them -- their status comes from
    # scan_cross_model() instead. Variable name kept as _OUTPUT_RUNS_56
    # to minimize churn at the three usage sites; the set is what
    # actually controls behavior.
    _OUTPUT_RUNS_56 = _DKRSet_scan({56, 57, 58})
    # v0.79.6.1: Run 0046 (MLP decomposition -- old Q56) must carry BOTH
    # pooled (56a) and per-condition (56b) passes. Pre-mandatory-56b Q0046
    # files exist with status='complete' but empty per_condition. Treat
    # those as partial so they re-run and populate per-condition. Ground-
    # truth alignment: if data is incomplete, status must reflect that.
    _MANDATORY_PER_COND_RUNS = _DKRSet_scan({46})

    # Derive (family, size_dir) for paper/json cross-model check.
    # paths['analysis'] = DATA/{family}/{size_dir}/{variant}/{cond}/analysis
    # Walk four dirnames back: analysis → cond → variant → size_dir → family
    _model_prefix = None
    try:
        _ana = paths.get('analysis', '')
        if _ana:
            _cond    = os.path.dirname(_ana)
            _variant = os.path.dirname(_cond)
            _size    = os.path.dirname(_variant)
            _family  = os.path.dirname(_size)
            _model_prefix = f"{os.path.basename(_family)}_{os.path.basename(_size)}"
    except Exception:
        _model_prefix = None

    for run_num, csv_fname in RUN_CSV.items():
        # v0.79.4.12: normalize run_num to int for equality comparisons.
        # RUN_CSV.items() yields 4-digit string keys post-renumber ("0001"),
        # so `run_num == 1` is False. Set membership via DualKeyRunSet
        # handles both forms, but literal `==` comparisons need the int.
        try:
            run_num_int = int(run_num)
        except (ValueError, TypeError):
            run_num_int = -1

        # v0.80.0.3: cross-model and paper-level runs (0052, 0053, 0054,
        # 0055, 0056) are NOT per-model. They live on the cross-model
        # card. Per-model scan skips them entirely -- attempting to
        # report per-model status for a cross-model run caused the
        # grid/card divergence observed in 0.80.0.2 (per-model grid
        # marked 0052/0053/0054 missing, model card showed done).
        if (run_num in _CROSS_MODEL_RUNS
                or run_num in _OUTPUT_RUNS_56
                or run_num_int == 55):
            continue

        # Analysis runs -- check JSON output file
        if csv_fname is None:
            json_fname = ANALYSIS_JSON.get(run_num)
            if not json_fname:
                status[run_num] = 'missing'; continue
            if run_num in _CROSS_MODEL_RUNS:
                # Cross-model: check DATA/paper/json/{prefix}_master_results.json
                # Presence of the per-model master results means this model has
                # been assembled into cross-model comparisons at least once.
                from cartography import DATA as _DATA_CM
                if _model_prefix:
                    jpath = os.path.join(_DATA_CM, 'paper', 'json',
                                         f"{_model_prefix}_master_results.json")
                else:
                    jpath = ''  # can't resolve -- falls through to 'missing'
            elif run_num in _POOLED_PATH_RUNS:
                # Pooled (non-cross-model) -- {variant}/pooled/analysis/
                variant_dir = os.path.dirname(os.path.dirname(paths.get('analysis', '')))
                jpath = os.path.join(variant_dir, 'pooled', 'analysis', json_fname)
            elif run_num in _OUTPUT_RUNS_56:  # v0.79.4.0: was `== 56`; run_num may be 4-digit string
                # v0.79.7: aggregate now lives at data/paper/results.json
                # (was paper/json/Q0056_paper_assembly.json).
                from cartography import DATA as _DATA55
                jpath = os.path.join(_DATA55, 'paper', json_fname)
            else:
                # Per-temperature analysis runs (including Run 0041 -- v0.71.0.2)
                jpath = os.path.join(ana_dir, json_fname)
            if jpath and os.path.exists(jpath):
                # v0.79.6.1: Run 0046 requires both pooled (56a) and per-
                # condition (56b) passes covering all conditions that
                # Run 0044 produced Ridge references for. A Q0046 with
                # fewer per_condition entries than Q0044 per_run entries
                # is incomplete regardless of its internal status field.
                # Ground-truth alignment: scanner reports partial when the
                # MLP per-condition coverage is below the Ridge reference
                # coverage at the same temperature.
                if run_num in _MANDATORY_PER_COND_RUNS:
                    _per_cond_ok = False
                    try:
                        with open(jpath, encoding='utf-8-sig') as _f46:
                            _q46 = json.load(_f46)
                        _pc = _q46.get('per_condition')
                        _pc_count = len(_pc) if isinstance(_pc, dict) else 0
                        # Compare against Q0044 reference count at same temp.
                        _q44_path = os.path.join(ana_dir, 'Q0044_per_condition_R.json')
                        _q44_count = 0
                        if os.path.exists(_q44_path):
                            try:
                                with open(_q44_path, encoding='utf-8-sig') as _f44:
                                    _q44 = json.load(_f44)
                                _q44_count = sum(
                                    1 for _e in (_q44.get('per_run') or [])
                                    if not _e.get('skipped') and _e.get('run_num') is not None
                                )
                            except Exception:
                                _q44_count = 0
                        # Complete only if Q0046 covers every Q0044 run AND
                        # internal status field says complete. If Q0044 is
                        # absent (count 0), Q0046 cannot be complete by
                        # definition -- Run 0044 is a prerequisite.
                        if (_q44_count > 0
                                and _pc_count >= _q44_count
                                and _q46.get('status') == 'complete'):
                            _per_cond_ok = True
                    except Exception:
                        _per_cond_ok = False
                    if not _per_cond_ok:
                        status[run_num] = 'partial'
                        continue
                # v0.80.0.1: Run 0056 requires methodology calibration artifacts
                # to be present AND fresh (cache hash match). Missing or stale
                # calibration → 'partial' with warning surfaced via the dashboard.
                # The aggregate results.json may exist and be structurally
                # complete, but the paper claim depends on calibration
                # evidence we haven't regenerated.
                #
                # `no_manifest` (artifact exists but no cache_key.json) is
                # treated as 'fresh' here -- 0056's orchestrator bootstraps
                # the manifest in place on first run. Scanner never mutates
                # disk; bootstrap is the orchestrator's job.
                if run_num in _OUTPUT_RUNS_56:
                    try:
                        # Lazy import: export_stats imports cartography, so
                        # top-level import creates a cycle. Import here.
                        import export_stats as _es
                        _cal_status = _es.methodology_calibration_status()
                        if _cal_status.get('_any_missing') or _cal_status.get('_any_stale'):
                            status[run_num] = 'partial'
                            continue
                    except Exception:
                        pass
                status[run_num] = 'done'
            else:
                # Check for .running marker → 'partial' (orange in grid)
                marker = os.path.join(ana_dir, json_fname.replace('.json', '.running'))
                if run_num in _CROSS_MODEL_RUNS:
                    from cartography import DATA as _DATA_CMm
                    if _model_prefix:
                        marker = os.path.join(_DATA_CMm, 'paper', 'json',
                                              f"{_model_prefix}_master_results.running")
                elif run_num in _POOLED_PATH_RUNS:
                    variant_dir = os.path.dirname(os.path.dirname(paths.get('analysis', '')))
                    marker = os.path.join(variant_dir, 'pooled', 'analysis',
                                          json_fname.replace('.json', '.running'))
                elif run_num in _OUTPUT_RUNS_56:
                    from cartography import DATA as _DATA55m
                    marker = os.path.join(_DATA55m, 'paper',
                                          json_fname.replace('.json', '.running'))
                status[run_num] = 'partial' if os.path.exists(marker) else 'missing'
            continue

        fpath = os.path.join(csv_dir, csv_fname)
        if not os.path.exists(fpath):
            status[run_num] = 'missing'; continue

        # Run 0001: three model passes + vector computation
        if run_num_int == 1:
            # v0.79.4.0: CSV run_mode now "0001", filenames R0001_*
            df_r1 = _load(fpath, 1)
            n_abl = _n_complete(df_r1)
            def _r1_sibling(variant):
                """Resolve the hidden_states directory for a sibling
                variant (base or instruct) of the current model.

                Run 0001 writes three variants in parallel but dirs are
                siblings under the same family/size_quant root -- not
                subdirs of paths['hidden']. Walk up cond -> variant ->
                size, then attach {variant}/{cond}/hidden_states.
                Falls back to old v42.1.10 subdir layout on failure to
                tolerate migration-window states."""
                try:
                    hd = paths.get('hidden', '')
                    cond = os.path.dirname(hd)
                    size = os.path.dirname(os.path.dirname(cond))
                    return os.path.join(size, variant, os.path.basename(cond), 'hidden_states')
                except Exception:
                    return os.path.join(paths.get('hidden', ''), variant)
            base_trials = set()
            for bd in [_r1_sibling('base'), os.path.join(paths.get('hidden',''), 'base')]:
                for f in _glob.glob(os.path.join(bd, 'R0001_*_base_trial*_turn01.npy')):
                    try: base_trials.add(int(os.path.basename(f).split('_trial')[1].split('_')[0]))
                    except: pass
            inst_trials = set()
            for id_ in [_r1_sibling('instruct'), os.path.join(paths.get('hidden',''), 'instruct')]:
                for f in _glob.glob(os.path.join(id_, 'R0001_*_instruct_trial*_turn01.npy')):
                    try: inst_trials.add(int(os.path.basename(f).split('_trial')[1].split('_')[0]))
                    except: pass
            hd = paths.get('hidden', '')
            pass4 = (len(_glob.glob(os.path.join(hd, 'R0001_*_pass4_ok.stamp'))) > 0 or
                     len(_glob.glob(os.path.join(hd, 'R0001_*_ct_global_mean.npy'))) > 0)
            all_done = (n_abl >= n_trials and len(base_trials) >= n_trials
                        and len(inst_trials) >= n_trials and pass4)
            nothing  = n_abl == 0 and not base_trials and not inst_trials
            status[run_num] = 'done' if all_done else 'missing' if nothing else 'partial'
            continue

        # Run 0003: temperature grid 4x5
        if run_num_int == 3:
            df_r3 = _load(fpath, 3)
            if df_r3.empty: status[run_num] = 'missing'; continue
            if 'condition' not in df_r3.columns or 'temperature_condition' not in df_r3.columns:
                status[run_num] = 'partial'; continue
            df_r3['_temp_parsed'] = df_r3['temperature_condition'].apply(
                lambda x: x.split('@',1)[1] if isinstance(x,str) and '@' in x else x)
            df_r3['_temp_parsed'] = _pd.to_numeric(
                df_r3['_temp_parsed'], errors='coerce').round(1)
            all_done = all(
                df_r3[(df_r3['condition'] == c) &
                     (df_r3['_temp_parsed'] == round(t, 1))]['trial']
                    .dropna().astype(int).nunique() >= n_trials
                for c in ['introspection', 'null', 'arithmetic', 'neutral_prime']
                for t in [0.2, 0.4, 0.6, 0.8, 1.0]
            )
            status[run_num] = 'done' if all_done else 'missing' if df_r3.empty else 'partial'
            continue

        # Runs 0017, 0018, 0019: activation patching
        if run_num_int in (17, 18, 19):
            # Read model layer count for dynamic mode filtering
            _PATCH_LAYERS = [8, 16, 24, 31]
            _nl = 32  # default: LLaMA 3 8B
            try:
                _var_dir = os.path.dirname(os.path.dirname(csv_dir))
                _nl_path = os.path.join(_var_dir, '.n_layers')
                if os.path.exists(_nl_path):
                    with open(_nl_path) as _f:
                        _nl = int(_f.read().strip())
            except Exception: pass
            _valid_layers = [li for li in _PATCH_LAYERS if li < _nl]
            if run_num_int == 17:
                patch_col = 'patch_mode'
                pmodes = ['none', 'partial', 'full', 'random']
            elif run_num_int == 18:
                patch_col = 'patch_layer'
                pmodes = ['none'] + [f'L{li}' for li in _valid_layers]
            else:  # 19 (was R53 random patching)
                patch_col = 'patch_layer'
                pmodes = ['none', 'full'] + [f'L{li}' for li in _valid_layers]
            df_p = _load(fpath, run_num)
            if df_p.empty: status[run_num] = 'missing'; continue
            if patch_col not in df_p.columns: status[run_num] = 'partial'; continue
            all_done = all(
                _n_complete(df_p[df_p[patch_col] == m], 13) >= n_trials
                for m in pmodes
            )
            status[run_num] = 'done' if all_done else 'missing' if df_p.empty else 'partial'
            continue

        # Multi-condition runs
        if run_num_int in _MC_RUNS:
            cond_col, cond_vals = _MC_RUNS[run_num_int]
            exp_turns = _MC_TURNS.get(run_num_int, 13)
            df_mc = _load(fpath, run_num)
            if df_mc.empty: status[run_num] = 'missing'; continue
            if cond_col not in df_mc.columns: status[run_num] = 'partial'; continue
            is_float = cond_vals and isinstance(cond_vals[0], float)
            if is_float:
                df_mc[cond_col] = _pd.to_numeric(df_mc[cond_col], errors='coerce').round(1)
            all_done = all(
                _n_complete(df_mc[df_mc[cond_col] == (round(v, 1) if is_float else v)],
                            exp_turns) >= n_trials
                for v in cond_vals
            )
            # v0.79.5.0: ET-RECOVERY special case removed. Source runs report
            # plain done/partial/missing based on CSV + S_t. ET coverage is
            # Run 16's own MC completeness -- independent.
            status[run_num] = 'done' if all_done else 'missing' if df_mc.empty else 'partial'
            continue

        # Standard single-condition runs
        df = _load(fpath, run_num)
        if df.empty: status[run_num] = 'missing'; continue
        df2 = df[df['trial'].notna()] if 'trial' in df.columns else df
        if df2.empty: status[run_num] = 'missing'; continue
        counts = df2.groupby('trial').size()
        if counts.empty: status[run_num] = 'missing'; continue
        _EXPECTED_TURNS = {39: 16, 22: 30}  # v0.79.4.0: old {10:perturbation_A, 23:saturation} → new {39, 22}
        exp_turns = _EXPECTED_TURNS.get(run_num_int, int(counts.mode().iloc[0]))
        n_complete = int((counts >= exp_turns).sum())
        csv_done = n_complete >= n_trials
        # v0.79.2.0: was `if csv_done and run_num in _ET_RECOVERY_RUNS:
        # status[run_num] = _et_status(...)`. ET state now lives in Run 0016
        # exclusively. Source-run status is plain done/partial/missing.
        status[run_num] = 'done' if csv_done else 'partial'

    # v0.79.5.0: status[16] override removed. Run 16 now scans through the
    # standard RUN_CSV iteration loop above (it has a real csv_fname after
    # the cartography.py promotion) and falls into the _MC_RUNS branch
    # which handles per-source condition counting the same way Run 21
    # (coherence, 3 conditions) and others are handled.

    return status


# ══════════════════════════════════════════════════════════════════════
#  Cross-model scan (v0.80.0.3)
# ══════════════════════════════════════════════════════════════════════
#
# Cross-model runs (0052/0053/0054) and paper-level runs (0055/0056) live
# on their own dashboard card, separate from per-model grids. Per-model
# scan above skips them; they're returned by this function instead.
#
# Status semantics differ per run:
#   0052/0053/0054: check DATA/paper/json/{prefix}_master_results.json
#                   exists for ALL discovered models → done
#                   exists for some → partial
#                   exists for none → missing
#   0055:           aggregated per-model Q0055_stats_report.json status
#                   across all discovered models
#   0056:           data/paper/results.json exists + calibration fresh → done
#                   exists + calibration incomplete → partial
#                   missing → missing

def _discover_models():
    """Walk data/ for per-model directories. Returns list of dicts with
    family, size, variant, prefix (for paper/json paths)."""
    import os as _os
    from cartography import DATA as _DATA
    if not _os.path.isdir(_DATA):
        return []

    found = []
    for family in sorted(_os.listdir(_DATA)):
        fam_dir = _os.path.join(_DATA, family)
        if not _os.path.isdir(fam_dir):
            continue
        if family in ('paper', '__pycache__'):
            continue
        for size in sorted(_os.listdir(fam_dir)):
            size_dir = _os.path.join(fam_dir, size)
            if not _os.path.isdir(size_dir):
                continue
            for variant in sorted(_os.listdir(size_dir)):
                var_dir = _os.path.join(size_dir, variant)
                if not _os.path.isdir(var_dir):
                    continue
                # v0.80.0.8: include any variant with a non-empty data
                # directory, not just ones with Q*.json analysis files.
                # A model being collected (CSVs present, analysis pending)
                # is still part of the paper set -- cross-model chips should
                # show it as empty ○ so runs containing it stay amber until
                # its analysis catches up. Previous behavior (0.80.0.7)
                # required Q-files, which hid mid-collection models like
                # Gemma 2B Q8 with collection runs 1-12 done but no analysis.
                has_data = False
                try:
                    for sub in _os.listdir(var_dir):
                        sub_path = _os.path.join(var_dir, sub)
                        if not _os.path.isdir(sub_path):
                            continue
                        # Any CSV or JSON in any subdirectory counts
                        for inner in (_os.listdir(sub_path)
                                      if _os.path.isdir(sub_path) else []):
                            if inner.endswith(('.csv', '.json')):
                                has_data = True
                                break
                            inner_path = _os.path.join(sub_path, inner)
                            if _os.path.isdir(inner_path):
                                try:
                                    for deeper in _os.listdir(inner_path):
                                        if deeper.endswith(('.csv', '.json')):
                                            has_data = True
                                            break
                                except OSError:
                                    continue
                            if has_data:
                                break
                        if has_data:
                            break
                except OSError:
                    continue
                if has_data:
                    found.append({
                        'family':  family,
                        'size':    size,
                        'variant': variant,
                        'prefix':  f"{family}_{size}",
                        'label':   f"{family} {size} {variant}",
                    })
    return found


def scan_cross_model():
    """Return cross-model run status dict for the dashboard's cross-model card.

    Returns:
      {
        'models': [{family, size, variant, prefix, label}, ...],
        'runs': {
          '0052': {'status': 'done'|'partial'|'missing',
                   'contributors': [prefix, ...],
                   'last_run': iso timestamp or None},
          ...
        },
        'calibration': {...}  # from methodology_calibration_status
      }
    """
    import json as _json
    import os as _os
    from cartography import DATA as _DATA

    models = _discover_models()
    paper_json = _os.path.join(_DATA, 'paper', 'json')
    paper_root = _os.path.join(_DATA, 'paper')
    results_path = _os.path.join(paper_root, 'results.json')

    runs = {}

    # 0052/0053/0054 -- cross-model run outputs live at
    # {variant}/pooled/analysis/Q00{NN}_*.json on every model they were
    # propagated to (start_here.py _CROSS_JSONS propagation). A model is
    # a "contributor" if the corresponding run output exists at that
    # model's pooled/analysis/. If every model has it, the run is done.
    #
    # v0.80.0.5: filenames for 0053 and 0054 confirmed via analysis.py
    # write sites (Q0053_condition_concordance.json,
    # Q0054_cross_model_summary.json). 0052's canonical filename is
    # disputed across the codebase (cartography says _summary, start_here
    # says _pairwise, export_stats says _R_comparison). Scanner checks
    # any of the three candidates so a match on any reports done.
    _CROSS_RUN_FILES = {
        '0052': [
            'Q0052_cross_model_pairwise.json',
            'Q0052_cross_model_summary.json',
            'Q0052_cross_model_R_comparison.json',
        ],
        '0053': ['Q0053_condition_concordance.json'],
        '0054': ['Q0054_cross_model_summary.json'],
    }
    for run_key, fnames in _CROSS_RUN_FILES.items():
        contributors = []
        for m in models:
            ana_dir = _os.path.join(_DATA, m['family'], m['size'], m['variant'],
                                    'pooled', 'analysis')
            for fname in fnames:
                if _os.path.exists(_os.path.join(ana_dir, fname)):
                    contributors.append(m['prefix'])
                    break

        n_models = len(models)
        n_contrib = len(contributors)
        if n_models == 0:
            st = 'missing'
        elif n_contrib == n_models:
            st = 'done'
        elif n_contrib > 0:
            st = 'partial'
        else:
            st = 'missing'
        runs[run_key] = {
            'status':       st,
            'contributors': contributors,
            'n_models':     n_models,
            'n_contributors': n_contrib,
        }

    # 0055 -- per-model Q0055_stats_report.json
    contributors_55 = []
    for m in models:
        # Q0055 stamps at {variant}/pooled/analysis/
        stamp = _os.path.join(_DATA, m['family'], m['size'], m['variant'],
                              'pooled', 'analysis', 'Q0055_stats_report.json')
        if _os.path.exists(stamp):
            contributors_55.append(m['prefix'])
    n_models = len(models)
    n_contrib_55 = len(contributors_55)
    if n_models == 0:
        st_55 = 'missing'
    elif n_contrib_55 == n_models:
        st_55 = 'done'
    elif n_contrib_55 > 0:
        st_55 = 'partial'
    else:
        st_55 = 'missing'
    runs['0055'] = {
        'status':       st_55,
        'contributors': contributors_55,
        'n_models':     n_models,
        'n_contributors': n_contrib_55,
    }

    # 0056 -- global results.json + calibration status
    # v0.80.0.10: report partial when any calibration artifact exists but
    # results.json hasn't been generated yet. Previously any missing
    # results.json → 'missing', which ignored that v5/ridge_bias files
    # might already exist from migration or prior runs.
    #
    # Also count discovered models as contributors-in-waiting: if models
    # are on disk but results.json hasn't been built, show them as empty
    # chips rather than hiding the contributor list entirely.
    contributors_56 = []
    if _os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8-sig') as f:
                rj = _json.load(f)
            for key, cell in (rj.get('cells') or {}).items():
                ids = (cell.get('identifiers') or {})
                prefix = f"{ids.get('family','')}_{ids.get('size','')}"
                if prefix not in contributors_56 and prefix != '_':
                    contributors_56.append(prefix)
        except Exception:
            pass

    # Calibration gate
    cal_any_present = False
    cal_all_fresh = False
    try:
        import export_stats as _es
        cal = _es.methodology_calibration_status()
        cal_all_fresh = bool(cal.get('_all_fresh'))
        # 'present' means not 'missing' -- no_manifest / stale / fresh all
        # count as "some work has been done here"
        for k in cal:
            if k.startswith('_'):
                continue
            if cal[k] != 'missing':
                cal_any_present = True
                break
    except Exception:
        pass

    # v0.80.0.11: 0056 status follows the same rule as the other cross-
    # model runs -- amber if upstream runs have output for any model,
    # because paper assembly is a selection operation (you pick which
    # models to include). Gray only if literally nothing upstream exists.
    # Previously 0056 was gray whenever results.json was missing, which
    # ignored that 0052-0055 might already be complete for the paper models.
    upstream_any = False
    for rk in ('0052', '0053', '0054', '0055'):
        r = runs.get(rk) or {}
        if (r.get('n_contributors') or 0) > 0:
            upstream_any = True
            break

    if _os.path.exists(results_path) and cal_all_fresh and contributors_56:
        st_56 = 'done'
    elif (_os.path.exists(results_path) or cal_any_present
          or upstream_any):
        # Results.json exists, OR some calibration done, OR any upstream
        # cross-model run has contributors -- all three are "runnable
        # with current data" states.
        st_56 = 'partial'
    else:
        st_56 = 'missing'

    runs['0056'] = {
        'status':       st_56,
        'contributors': contributors_56,
        'n_models':     len(models),
        'n_contributors': len(contributors_56),
    }

    # 0057 -- function-class sensitivity (NEW v0.80.0.38).
    # Status keys off Q0057_function_class_sensitivity.json existence and
    # the n_cells_with_full_data summary inside it. Contributors are
    # derived from the cell keys in the JSON's "cells" block.
    # v0.80.0.48: two scanner bugs fixed for Run 0057.
    #
    # Bug 1 -- contributor prefix mismatch:
    #   Cell keys in Q0057 use the writerbot convention
    #   (gemma_2b_q4_abliterated_t10), with quant suffixes 'q4' / 'q8' /
    #   'fp16'. Dashboard chip rendering in export_flask matches against
    #   m.prefix from _discover_models(), which is f"{family}_{size}"
    #   where size includes the canonical quant suffix
    #   ('gemma_2b_4bit', 'gemma_2b_fp16', 'llama_8b_4bit'). Building
    #   the contributor prefix as parts[0]+parts[1] dropped the quant
    #   entirely; building as parts[0]+parts[1]+parts[2] would yield
    #   'gemma_2b_q4' and still mismatch. Reverse-map the writerbot
    #   quant ('q4' -> '4bit', 'q8' -> '8bit', 'fp16' -> 'fp16') so
    #   the prefix matches what the dashboard chip lookup expects.
    #
    # Bug 2 -- 'done' status too lenient:
    #   Pre-fix, n_cells_57 counted every cell that lacked an 'error'
    #   key, regardless of whether the partition-side fields populated.
    #   The 0.80.0.46 Q0057 had 24 cells without errors but with empty
    #   R_hat_gap / sign_check (the canon partition pull failed silently
    #   on a cell-key mismatch fixed in 0.80.0.47). Scanner reported
    #   'done' while the analysis was functionally incomplete. Tighten
    #   the completeness check: a cell is 'fully complete' only when
    #   it both lacks errors AND has at least one populated entry in
    #   R_hat_gap or sign_check (i.e., the canon partition pull
    #   succeeded for that cell). Status reads 'done' only when every
    #   cell is fully complete.
    _Q57_QUANT_REVERSE = {'q4': '4bit', 'q8': '8bit', 'fp16': 'fp16'}

    def _q57_canon_prefix(cell_key):
        parts = cell_key.split('_')
        if len(parts) < 3:
            return None
        family, size_bare, quant_w = parts[0], parts[1], parts[2]
        quant_c = _Q57_QUANT_REVERSE.get(quant_w, quant_w)
        return f"{family}_{size_bare}_{quant_c}"

    def _q57_cell_fully_complete(cell):
        if 'error' in cell:
            return False
        rhg = cell.get('R_hat_gap') or {}
        sc  = cell.get('sign_check') or {}
        return bool(rhg) or bool(sc)

    contributors_57 = []
    n_cells_57 = 0
    n_cells_fully_complete_57 = 0
    n_cells_total_57 = 0
    q57_path = _os.path.join(paper_root, 'Q0057_function_class_sensitivity.json')
    if _os.path.exists(q57_path):
        try:
            with open(q57_path, 'r', encoding='utf-8-sig') as f:
                q57 = _json.load(f)
            for cell_key, cell in (q57.get('cells') or {}).items():
                n_cells_total_57 += 1
                if 'error' not in cell:
                    n_cells_57 += 1
                    if _q57_cell_fully_complete(cell):
                        n_cells_fully_complete_57 += 1
                    prefix = _q57_canon_prefix(cell_key)
                    if prefix and prefix not in contributors_57:
                        contributors_57.append(prefix)
        except Exception:
            pass

    if not _os.path.exists(q57_path):
        st_57 = 'missing'
    elif n_cells_57 == 0:
        st_57 = 'missing'
    elif (n_cells_fully_complete_57 == n_cells_total_57
          and n_cells_total_57 > 0):
        st_57 = 'done'
    else:
        # RF fits ran but partition side is empty (or partially empty) --
        # this is the post-Run-0057 / pre-hot-patch state under 0.80.0.46
        # and the natural mid-run state.
        st_57 = 'partial'

    runs['0057'] = {
        'status':         st_57,
        'contributors':   contributors_57,
        'n_models':       len(models),
        'n_contributors': len(contributors_57),
        'n_cells_complete':        n_cells_57,
        'n_cells_fully_complete':  n_cells_fully_complete_57,
        'n_cells_total':           n_cells_total_57,
    }

    # 0058 -- lagrangian apparatus. Status reads the manifest and
    # requires ALL 11 phases status='done' before reporting 'done'.
    # v0.81.1.1: pre-fix, status was based purely on manifest file
    # existence, which let partial-manifest runs (Phase 1 only, with
    # Phases 2-5 marked pending) trip as 'done'. Now reads the JSON.
    # v0.81.1.7 ship 5: paper-2 phases 6-11 added to the required list.
    # Under codebot_handoff_v0_14, Run 0058 = paper 1 apparatus + paper
    # 2 measurement layer combined. Done means all 11 phases done.
    # Phases 6-11 are non-blocking on phases 1-5 (paper-1 apparatus
    # claim is complete with phases 1-5 alone), but Run 0058's overall
    # 'done' marker requires the full pipeline.
    q58_path = _os.path.join(paper_root, 'Q0058_apparatus_manifest.json')
    st_58 = 'missing'
    if _os.path.exists(q58_path):
        try:
            with open(q58_path, 'r', encoding='utf-8-sig') as f:
                _q58 = _json.load(f)
            _phase_states = _q58.get('apparatus_phase_status', {}) or {}
            _required_p1 = ['phase1_kraskov_spike', 'phase2_solver_module',
                             'phase3_v5d_threshold', 'phase4_anchor_producer',
                             'phase5_aggregator']
            _required_p2 = ['phase6_function_class_p2',
                             'phase7_knn_anchor_p2',
                             'phase8_apparatus_p2',
                             'phase9_stacking_baseline_p2',
                             'phase10_bootstrap_variance_p2',
                             'phase11_hull_diagnostic_p2']
            _required_all = _required_p1 + _required_p2
            if all(_phase_states.get(p) == 'done' for p in _required_all):
                st_58 = 'done'
            elif all(_phase_states.get(p) == 'done' for p in _required_p1):
                # Paper-1 apparatus complete; paper-2 phases partial.
                st_58 = 'partial'
            elif any(_phase_states.get(p) == 'done' for p in _required_all):
                st_58 = 'partial'
            else:
                st_58 = 'partial'  # manifest exists but no phase done = run failed early
        except Exception:
            st_58 = 'partial'  # manifest unreadable but exists
    elif cal_all_fresh and st_57 == 'done':
        st_58 = 'partial'  # foundations + function-class ready, apparatus not run
    else:
        st_58 = 'missing'

    runs['0058'] = {
        'status':         st_58,
        'contributors':   contributors_56,  # apparatus operates over the same paper cells
        'n_models':       len(models),
        'n_contributors': len(contributors_56),
    }

    # 0059 -- paper assembly (renumbered from 0058 in v0.80.0.51).
    # Hard-gates on foundations (0056) + function-class (0057) + apparatus (0058).
    if (_os.path.exists(results_path) and cal_all_fresh
            and contributors_56 and st_57 == 'done' and st_58 == 'done'):
        st_59 = 'done'
    elif _os.path.exists(results_path) or upstream_any or st_57 != 'missing':
        st_59 = 'partial'
    else:
        st_59 = 'missing'

    runs['0059'] = {
        'status':         st_59,
        'contributors':   contributors_56,  # same as 0056 -- paper cells
        'n_models':       len(models),
        'n_contributors': len(contributors_56),
    }

    # Calibration block for the banner
    try:
        import export_stats as _es
        cal_block = _es.methodology_calibration_status()
    except Exception:
        cal_block = None

    return {
        'models':       models,
        'runs':         runs,
        'calibration':  cal_block,
    }
