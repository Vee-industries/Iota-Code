"""
IOTA FRAMEWORK — Scanner
=========================
Shared run status scanning. Used by start_here.py (console) and
export_flask.py (dashboard /scan and /temp_grid endpoints).

Extracted from start_here.py v0.69.1.4 to avoid circular imports
and heavy dependencies (orchestration_core imports torch).

Dependencies: os, glob, pandas, cartography (RUN_CSV, ANALYSIS_JSON).
Does NOT import torch, transformers, or orchestration_core.
"""

import os

from cartography import DualKeyRunSet as _DKRSet


# ── Data constants ────────────────────────────────────────────────────────────

# v0.79.2.0: _ET_RECOVERY_RUNS is now the set of *source runs* that Run 0016
# (E_t recovery meta-run) iterates. Per-source status no longer includes
# 'needs_et' / 'et_partial' — those runs report plain done/partial/missing
# based on CSV + S_t only. E_t coverage aggregated into Run 0016's status.
# v0.79.4.0: DualKeyRunSet — accepts int or 4-digit string lookups.
_ET_RECOVERY_RUNS = _DKRSet({
    # v0.79.4.0: renumbered. ET-source runs in new numbering.
    # Old {1,2,3,4,5,6,7,8,9,15,16,17,20} → new:
    1,   # was R20 (robustness) — Phase A, temp-indep
    4,   # was R01 (null A)      — Phase B
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
# from each source CSV at scan time — no hardcoded totals.

_MC_RUNS = {
    # v0.79.4.0 renumber — bijective remap:
    #   old 20→2, 28→23, 29→24, 30→20, 31→21, 37→37,
    #   38→36, 39→35, 41→33, 43→25, 44→26
     2: ('temperature_condition', [0.0, 0.3, 0.5, 0.7, 1.0]),     # was R20 robustness
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
     2: 13, 23: 13, 24: 13, 20: 13, 21: 13,
    37: 13, 36: 13, 35: 13, 33: 13, 25: 13, 26: 13,
}


# ── CSV reader (no torch dependency) ─────────────────────────────────────────

def _csv_read(path):
    """Read a universal-schema CSV. Returns DataFrame. Empty on error."""
    import pandas as pd
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
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

    def _run48_status():
        """v0.79.2.0: Run 0016 — E_t recovery meta-run status aggregated
        across all 13 source runs in _ET_RECOVERY_RUNS.

        For each source with a CSV on disk, count expected (unique non-
        priming (trial, turn) pairs) vs actual *_et_base.npy files in
        the base variant's hidden_states dir.

        Returns:
          'done'    — every source with a CSV has 100% ET coverage
          'partial' — some ET work done but incomplete (any source > 0
                      but < expected, or some sources done and others at 0)
          'missing' — no ET files anywhere (including case: no source CSVs
                      exist yet, so there's nothing to recover)
        """
        base_hid = _base_hidden_dir()
        any_started = False
        any_incomplete = False
        any_source_counted = False

        for src_run in sorted(_ET_RECOVERY_RUNS):
            csv_fname = RUN_CSV.get(src_run)
            if not csv_fname:
                continue
            csv_path = os.path.join(csv_dir, csv_fname)
            if not os.path.exists(csv_path):
                # Source CSV not collected yet — skip entirely. Can't recover
                # ET for data that doesn't exist. Doesn't contribute to status.
                continue
            df = _load(csv_path, src_run)
            if df.empty or 'trial' not in df.columns or 'turn' not in df.columns:
                continue
            # Count expected ET files = unique non-priming (trial, turn) rows.
            if 'priming' in df.columns:
                df_np = df[df['priming'].astype(str) != '1']
            else:
                df_np = df
            df_np = df_np[df_np['trial'].notna() & df_np['turn'].notna()]
            if df_np.empty:
                continue
            n_expected = len(df_np.groupby(['trial', 'turn']).size())
            if n_expected == 0:
                continue

            # v0.79.4.0: glob both 4-digit (post-migration) and 2-digit
            # (legacy) patterns; files on disk may be either format until
            # migrate_run_ids.py has been run.
            from cartography import run_id_pad as _rip
            _rid4 = _rip(src_run)
            n_actual = (
                len(_glob.glob(
                    os.path.join(base_hid, f"R{_rid4}_*_et_base.npy"))) +
                len(_glob.glob(
                    os.path.join(base_hid, f"R{src_run:02d}_*_et_base.npy")))
            )

            any_source_counted = True
            if n_actual > 0:
                any_started = True
            if n_actual < n_expected:
                any_incomplete = True

        if not any_source_counted:
            # No source CSVs exist yet — nothing to recover, show missing.
            return 'missing'
        if not any_incomplete and any_started:
            return 'done'
        if any_started:
            return 'partial'
        return 'missing'

    status = {}

    # v0.77.1.3: cross-model runs (50/51/52) are per-model completion artifacts
    # that live in DATA/paper/json/{family}_{size_dir}_master_results.json.
    # Prior approach (0.77.1.0–0.77.1.2) checked {variant}/pooled/analysis/
    # Q52_cross_model_summary.json, but that file only exists for the model
    # the cross-model run was LAUNCHED from — not for every model it compared
    # against. Result: runs 0052/0053/0054 greened only for the launching model,
    # stayed grey for the rest. The paper/json artifact IS per-model, written
    # for every model in the comparison set, so it's the correct source of
    # truth for "this model has cross-model data."
    _CROSS_MODEL_RUNS = {52, 53, 54}  # v0.79.4.0: old {50,51,52}
    # Pooled runs (40/47/54) still write to {variant}/pooled/analysis/.
    _POOLED_PATH_RUNS = {51, 50, 55}  # v0.79.4.0: old {40,47,54}

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
                    jpath = ''  # can't resolve — falls through to 'missing'
            elif run_num in _POOLED_PATH_RUNS:
                # Pooled (non-cross-model) -- {variant}/pooled/analysis/
                variant_dir = os.path.dirname(os.path.dirname(paths.get('analysis', '')))
                jpath = os.path.join(variant_dir, 'pooled', 'analysis', json_fname)
            elif run_num == 56:
                # Cross-model paper assembly — single global file in paper/json/
                from cartography import DATA as _DATA55
                jpath = os.path.join(_DATA55, 'paper', 'json', json_fname)
            else:
                # Per-temperature analysis runs (including Run 0041 — v0.71.0.2)
                jpath = os.path.join(ana_dir, json_fname)
            if jpath and os.path.exists(jpath):
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
                elif run_num == 56:
                    from cartography import DATA as _DATA55m
                    marker = os.path.join(_DATA55m, 'paper', 'json',
                                          json_fname.replace('.json', '.running'))
                status[run_num] = 'partial' if os.path.exists(marker) else 'missing'
            continue

        fpath = os.path.join(csv_dir, csv_fname)
        if not os.path.exists(fpath):
            status[run_num] = 'missing'; continue

        # Run 0001: three model passes + vector computation
        if run_num == 1:
            df19 = _load(fpath, 19)
            n_abl = _n_complete(df19)
            def _r19_sibling(variant):
                """Resolve the hidden_states directory for a sibling
                variant (base or instruct) of the current model.

                Run 0001 writes three variants in parallel but dirs are
                siblings under the same family/size_quant root — not
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
            for bd in [_r19_sibling('base'), os.path.join(paths.get('hidden',''), 'base')]:
                for f in _glob.glob(os.path.join(bd, 'R19_*_base_trial*_turn01.npy')):
                    try: base_trials.add(int(os.path.basename(f).split('_trial')[1].split('_')[0]))
                    except: pass
            inst_trials = set()
            for id_ in [_r19_sibling('instruct'), os.path.join(paths.get('hidden',''), 'instruct')]:
                for f in _glob.glob(os.path.join(id_, 'R19_*_instruct_trial*_turn01.npy')):
                    try: inst_trials.add(int(os.path.basename(f).split('_trial')[1].split('_')[0]))
                    except: pass
            hd = paths.get('hidden', '')
            pass4 = (len(_glob.glob(os.path.join(hd, 'R19_*_pass4_ok.stamp'))) > 0 or
                     len(_glob.glob(os.path.join(hd, 'R19_*_ct_global_mean.npy'))) > 0)
            all_done = (n_abl >= n_trials and len(base_trials) >= n_trials
                        and len(inst_trials) >= n_trials and pass4)
            nothing  = n_abl == 0 and not base_trials and not inst_trials
            status[run_num] = 'done' if all_done else 'missing' if nothing else 'partial'
            continue

        # Run 0003: temperature grid 4x5
        if run_num == 3:
            df26 = _load(fpath, 26)
            if df26.empty: status[run_num] = 'missing'; continue
            if 'condition' not in df26.columns or 'temperature_condition' not in df26.columns:
                status[run_num] = 'partial'; continue
            df26['_temp_parsed'] = df26['temperature_condition'].apply(
                lambda x: x.split('@',1)[1] if isinstance(x,str) and '@' in x else x)
            df26['_temp_parsed'] = _pd.to_numeric(
                df26['_temp_parsed'], errors='coerce').round(1)
            all_done = all(
                df26[(df26['condition'] == c) &
                     (df26['_temp_parsed'] == round(t, 1))]['trial']
                    .dropna().astype(int).nunique() >= n_trials
                for c in ['introspection', 'null', 'arithmetic', 'neutral_prime']
                for t in [0.2, 0.4, 0.6, 0.8, 1.0]
            )
            status[run_num] = 'done' if all_done else 'missing' if df26.empty else 'partial'
            continue

        # Runs 0017, 0018, 0019: activation patching
        if run_num in (17, 18, 19):
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
            if run_num == 17:
                patch_col = 'patch_mode'
                pmodes = ['none', 'partial', 'full', 'random']
            elif run_num == 18:
                patch_col = 'patch_layer'
                pmodes = ['none'] + [f'L{li}' for li in _valid_layers]
            else:  # 53
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
        if run_num in _MC_RUNS:
            cond_col, cond_vals = _MC_RUNS[run_num]
            exp_turns = _MC_TURNS.get(run_num, 13)
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
            if all_done and run_num in _ET_RECOVERY_RUNS:
                # v0.79.2.0: was `status[run_num] = _et_status(...)`. ET state
                # now lives in Run 0016 exclusively. Source-run status is plain
                # done/partial/missing based on CSV + S_t only.
                status[run_num] = 'done'
            else:
                status[run_num] = 'done' if all_done else 'missing' if df_mc.empty else 'partial'
            continue

        # Standard single-condition runs
        df = _load(fpath, run_num)
        if df.empty: status[run_num] = 'missing'; continue
        df2 = df[df['trial'].notna()] if 'trial' in df.columns else df
        if df2.empty: status[run_num] = 'missing'; continue
        counts = df2.groupby('trial').size()
        if counts.empty: status[run_num] = 'missing'; continue
        _EXPECTED_TURNS = {10: 16, 23: 30}
        exp_turns = _EXPECTED_TURNS.get(run_num, int(counts.mode().iloc[0]))
        n_complete = int((counts >= exp_turns).sum())
        csv_done = n_complete >= n_trials
        # v0.79.2.0: was `if csv_done and run_num in _ET_RECOVERY_RUNS:
        # status[run_num] = _et_status(...)`. ET state now lives in Run 0016
        # exclusively. Source-run status is plain done/partial/missing.
        status[run_num] = 'done' if csv_done else 'partial'

    # v0.79.2.0: Run 0016 — E_t recovery meta-run, first-class status.
    # Aggregates ET coverage across all 13 sources. Not in RUN_CSV because
    # it writes no CSV of its own — its output is *_et_base.npy files in
    # the base-variant hidden_states dir.
    status[48] = _run48_status()

    return status
