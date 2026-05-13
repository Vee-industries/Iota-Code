"""
run_function_class_sensitivity.py -- Run 0057.

Function-class sensitivity analysis for paper 1 §5.4.

Validates the cross-cell asymmetry-vs-gap pattern under a third nonlinear
function class (Random Forest), beyond the canon Ridge-vs-MLP pair.

Per writerbot's handoff:
- Reads canon Ridge + MLP per-channel R² from
  data/paper/calibration/channel_marginal/channel_marginal_nonlinearity.csv
  (produced by Run 0056 calibration phase, activation='tanh' MLP).
- Reads canon Ridge + MLP partition fractions (ehat_*, chat_*, rhat_*)
  from results.json's per-cell measurements (or refits cheap if not
  yet stored; flagged in metadata as 'computed' vs 'pulled').
- Fits RF on the same 80/20 split (random_state=42) using the same
  StandardScaler + same protocol as the canon analysis.
- Computes the sign relationship between asymmetry and Ŕ-gap for both
  Ridge-vs-MLP and Ridge-vs-RF pairs.
- Writes data/paper/Q0057_function_class_sensitivity.json with full
  per-cell breakdown + cross_cell_aggregates summary block.

Activation-protocol note:
  Canon MLP uses activation='tanh', hidden_layer_sizes=(256,),
  early_stopping=True, max_iter=500, random_state=42. Writerbot's
  original handoff erroneously specified ReLU; canon wins. The
  discrepancy is documented in metadata and is being corrected in
  paper v0_13 §4.2 independently of this run's outcome.

RF hyperparameters (from handoff §3):
  n_estimators=100, max_depth=None, min_samples_leaf=5,
  random_state=42, n_jobs=4 (was -1; reduced in v0.80.0.46
  to bound Windows-spawn worker memory copies -- see RF_HYPERS comment).

Output: data/paper/Q0057_function_class_sensitivity.json

Runtime: ~1-3 min per cell CPU, 24 cells in ~30-90 min total.
Hard-depended on by Run 0058 (apparatus) and Run 0059 (paper assembly).

v0.80.0.38: created.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys
import time

# v0.80.0.20 utf8 reconfigure pattern, copied from run_channel_marginal.py
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    import io as _io
    if hasattr(_sys.stdout, 'buffer'):
        _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(_sys.stderr, 'buffer'):
        _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')

# v0.80.0.44: route every progress/status print through ui.* so output
# lands in BOTH .iota_flask.log (raw stdout, Detailed console) AND
# .iota_log.jsonl (curated log, Simple console). Pre-0.80.0.44 we
# used raw print(..., flush=True), which went only to stdout -- the
# Detailed view saw everything but the Simple view was silent. This
# is a recurring class of bug: any new analysis script that uses
# print() instead of ui.* is invisible to Simple. Standard fix: use
# ui.section / ui.msg / ui.ok / ui.warn / ui.err for every line.
sys.path.insert(0, ROOT)
import ui  # noqa: E402

# v0.80.0.47: import canonical cell_key builder so partition lookups
# use the same convention as results.json. Pre-fix, _pull_canon_partition
# used the writerbot-style cell_key (e.g. 'gemma_2b_q4_abliterated_t00')
# which never matches the results.json convention
# (e.g. 'gemma_2b_4bit_abliterated_T0.0'). Every per-cell partition pull
# returned None, leaving R_hat_gap and sign_check empty in Q0057 JSON.
import results_schema as _sch  # noqa: E402

# Output destinations
OUT_PATH = os.path.join(DATA, 'paper', 'Q0057_function_class_sensitivity.json')
CHANNEL_MARGINAL_CSV = os.path.join(
    DATA, 'paper', 'calibration', 'channel_marginal',
    'channel_marginal_nonlinearity.csv'
)
RESULTS_JSON = os.path.join(DATA, 'paper', 'results.json')

# RF hyperparameters (handoff §3, n_jobs adjusted in v0.80.0.46)
# v0.80.0.46: n_jobs reduced from -1 (all cores) to 4. Windows uses
# spawn -- every worker re-pickles the full training matrix. On the
# joint fit (d=3073 features × 17280 train rows × 100 trees) with
# ~12 cores, 12 simultaneous matrix copies overflowed available
# memory and the process exited with 0xC000013A (Windows OOM abort)
# at "[rf] fitting joint" after both univariate fits had completed
# cleanly. n_jobs=4 keeps useful parallelism but bounds the copy
# count to a fixed ceiling. Wall-clock per fit goes up moderately;
# memory ceiling drops by ~3x. Ridge and MLP don't share this
# hazard because they fit in one process.
RF_HYPERS = {
    'n_estimators':     100,
    'max_depth':        None,
    'min_samples_leaf': 5,
    'random_state':     42,
    'n_jobs':           4,
}

# Canon protocol constants (matches run_channel_marginal.py + analysis.py)
N_PERMUTATIONS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHA = 0.01
MLP_HYPERS = {
    'hidden_layer_sizes': (256,),
    'activation':         'tanh',     # canon -- NOT ReLU (writerbot v0_12 erratum)
    'early_stopping':     True,
    'validation_fraction': 0.1,
    'max_iter':           500,
    'random_state':       42,
    'tol':                1e-5,
}

# v0.81.1.4: RKHS kernel ridge regression with characteristic kernel grid.
# Matérn ν ∈ {1/2, 3/2, 5/2}: ν=1/2 is exponential / Laplace (least smooth),
# ν=3/2 is once-differentiable, ν=5/2 is twice-differentiable. Plus RBF
# (Gaussian) at four length scales σ ∈ {0.5, 1, 2, 4} multiplied by the
# median pairwise distance -- adaptive bandwidth on a fixed grid. Total
# 7 kernels per fit; RKHS_median is the median across the grid, range
# reported alongside. ridge alpha = 1e-3 chosen to match analysis.py
# linearity_check kernel-ridge spec (small but nonzero -- kernel matrices
# are nearly singular without it).
# v0.82.0.7: cut from 7 kernels to 3 (matern_nu1.5 as canonical RKHS
# choice + RBF at two length scales) for compute-budget reasons. The
# committee chose option B from the trajectory_plan_v6 followup
# discussion: "RKHS as directional sanity check that nonlinear
# estimators broadly agree with Ridge" rather than "robust median
# across full kernel grid." See CHANGELOG 0.82.0.7. The rkhs_range
# diagnostic still fires, just over a smaller grid.
RKHS_KERNELS = [
    {'kind': 'matern', 'nu': 1.5, 'length_scale_mult': 1.0},
    {'kind': 'rbf',    'length_scale_mult': 1.0},
    {'kind': 'rbf',    'length_scale_mult': 2.0},
]
RKHS_ALPHA = 1e-3
# v0.82.0.7: reduced from 5000 to 2500 to bring per-cell wall time
# under 5 min on a 3080. Kernel matrix is now 2500x2500 = 6.25M
# entries = 50MB at float64. Per-kernel fit ~5-10s; 3 kernels per cell
# × 24 cells = ~75 min total.
RKHS_N_SUBSAMPLE = 2500


# ─────────────────────────────────────────────────────────────────────────
#  Cell key + parsing -- match v0_12 schema
# ─────────────────────────────────────────────────────────────────────────

def _temp_str(temp):
    """Format temperature as canonical T-suffix used in cell keys."""
    if temp is None:
        return 'Tnone'
    return f"T{temp:.1f}"


def _cell_key(family, size, variant, temp):
    """Cell key: e.g. 'llama_8b_q4_t10' (matches handoff schema convention).

    Uses the writerbot-style _t10 / _t08 / _t00 suffix (T*10) for join key.
    """
    # Writerbot's handoff uses keys like 'llama_8b_q4_t10' -- t suffix is
    # T * 10 with no decimal. Map our internal sizes to writerbot's
    # convention.
    _SIZE_MAP = {
        '2b_4bit':  '2b_q4',
        '2b_8bit':  '2b_q8',
        '2b_fp16':  '2b_fp16',
        '9b_4bit':  '9b_q4',
        '8b_4bit':  '8b_q4',
    }
    size_key = _SIZE_MAP.get(size, size)
    if temp is None:
        t_key = 'tnone'
    else:
        t_key = f"t{int(round(temp * 10)):02d}"
    return f"{family}_{size_key}_{variant}_{t_key}"


def _parse_q42_path(q42_path):
    """Extract (family, size, variant, temp) from a Q0042_decomposition.json path."""
    rel = os.path.relpath(q42_path, DATA).replace('\\', '/').split('/')
    if len(rel) < 6:
        return None
    family, size, variant, cond = rel[0], rel[1], rel[2], rel[3]
    if cond == 'deterministic':
        temp = 0.0
    elif cond.startswith('temp_'):
        try:    temp = float(cond.replace('temp_', ''))
        except: temp = None
    else:
        temp = None
    return family, size, variant, cond, temp


# ─────────────────────────────────────────────────────────────────────────
#  Load canon -- channel marginal CSV (Ridge + MLP per-channel R²)
# ─────────────────────────────────────────────────────────────────────────

def _load_channel_marginal_csv():
    """Returns dict: (family, size, variant, condition) -> row dict."""
    if not os.path.exists(CHANNEL_MARGINAL_CSV):
        return {}
    rows = {}
    with open(CHANNEL_MARGINAL_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            k = (r.get('family'), r.get('size'),
                 r.get('variant'), r.get('condition'))
            rows[k] = r
    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Load canon -- results.json partition fractions
# ─────────────────────────────────────────────────────────────────────────

def _load_results_json():
    """Returns dict cell_key -> measurements dict (or {} if not present)."""
    if not os.path.exists(RESULTS_JSON):
        return {}
    try:
        with open(RESULTS_JSON, 'r', encoding='utf-8-sig') as f:
            raw = f.read()
        raw = re.sub(r'\bNaN\b', 'null', raw)
        raw = re.sub(r'\bInfinity\b', 'null', raw)
        raw = re.sub(r'\b-Infinity\b', 'null', raw)
        data = json.loads(raw)
        return data.get('cells', {})
    except Exception as e:
        print(f"  ! results.json load failed: {e}", file=sys.stderr)
        return {}


def _pull_canon_partition(cell_key, results_cells,
                          family=None, size=None, variant=None, temp=None):
    """Pull canon Ridge + MLP partition fractions from results.json.
    Returns dict with 'ridge'/'mlp' subdicts of E/C/R, or None per class
    if the field set is incomplete. Marks each as 'pulled' for metadata.

    v0.80.0.47: results.json keys cells via results_schema.cell_key
    convention ('gemma_2b_4bit_abliterated_T0.0'), not the writerbot-
    style cell_key used as the Q0057 output key
    ('gemma_2b_q4_abliterated_t00'). The two never match. When the
    original tuple args are supplied, build the canon-convention key
    via _sch.cell_key(...) and look up by that. Falls back to the
    writerbot-style cell_key if tuple args are not provided (legacy
    callers). Pre-0.80.0.47, every Q0057 cell came back with
    partition_source='cell_not_in_results_json' and empty R_hat_gap /
    sign_check blocks because of this mismatch.
    """
    canon_key = None
    if family is not None and size is not None and variant is not None:
        canon_key = _sch.cell_key(family, size, variant, temp)
        cell = results_cells.get(canon_key)
    else:
        # Legacy fallback path
        cell = results_cells.get(cell_key)

    if cell is None:
        # Last-ditch fallback: try the writerbot-style key in case
        # results.json was ever generated under that convention
        # somewhere downstream.
        cell = results_cells.get(cell_key)
        if cell is None:
            return None, None, (
                f'cell_not_in_results_json (tried canon_key={canon_key!r}, '
                f'writerbot_key={cell_key!r})'
            )

    m = cell.get('measurements', {})
    ridge = {}
    mlp = {}
    # v0.81.1.3: pull canonical paper §5/§6 fields, not the H1 cross-source
    # quantity. The previous code pulled `*_ridge_heldout` and `*_mlp_reference`
    # which are NOT the paper's headline values:
    #   - rhat_ridge_heldout = Ridge fit on the 9-source pool, applied to
    #     Run 0033 (seed-shifted twin). This is the H1 cross-source quantity,
    #     not the headline Ridge value. Misnamed historically.
    #   - rhat_mlp_reference = single-fit value from the 256-unit architecture
    #     only. Recorded for inspection; not the headline.
    # Headline (matches paper 1 v0_13 §4.2):
    #   - rhat_ridge_insample = Ridge fit on 80% train, permutation drops on
    #     20% test. This is what paper 1 §5/§6 reports as the headline Ridge.
    #   - rhat_mlp_pooled_mean = arithmetic mean across 64 / 256 / 128–64
    #     architectures. This is what paper 1 §5/§6 reports as the headline MLP.
    # Sign-flips on Gemma 9B Q4 and Gemma 2B FP16 between the two pairings;
    # cross-cell aggregates regenerate after this fix.
    for ch, ridge_field, mlp_field in [
        ('E', 'ehat_ridge_insample', 'ehat_mlp_pooled_mean'),
        ('C', 'chat_ridge_insample', 'chat_mlp_pooled_mean'),
        ('R', 'rhat_ridge_insample', 'rhat_mlp_pooled_mean'),
    ]:
        if ridge_field in m:
            ridge[ch] = float(m[ridge_field])
        if mlp_field in m:
            mlp[ch] = float(m[mlp_field])

    ridge_complete = all(k in ridge for k in ('E', 'C', 'R'))
    mlp_complete = all(k in mlp for k in ('E', 'C', 'R'))
    if not ridge_complete:
        ridge = None
    if not mlp_complete:
        mlp = None
    return ridge, mlp, f'pulled_from_results_json (canon_key={canon_key!r})'


# ─────────────────────────────────────────────────────────────────────────
#  RF analysis on a single cell -- mirrors run_channel_marginal protocol
# ─────────────────────────────────────────────────────────────────────────

def _run_rf_on_cell(q42_path):
    """Per-channel and joint RF fits on one cell. Returns dict or {'error': ...}.

    Mirrors run_channel_marginal._compute_cell exactly for data loading,
    POOL_DIM detection, source-runs selection, train/test split, and
    StandardScaler protocol -- only the regressor differs.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # POOL_DIM + source_runs detection -- same logic as run_channel_marginal
    sys.path.insert(0, ROOT)
    import analysis as _ana

    cell_dir = os.path.dirname(os.path.dirname(q42_path))
    hidden_dir = os.path.join(cell_dir, 'hidden_states')

    pool_dim = None
    source_runs_min_max = None
    cache_glob = os.path.join(hidden_dir, '_qcache_ct_d*_r*.npz')
    cache_files = sorted(glob.glob(cache_glob))
    if cache_files:
        m = re.match(r'_qcache_ct_d(\d+)_r(\d+)-(\d+)\.npz$',
                     os.path.basename(cache_files[0]))
        if m:
            pool_dim = int(m.group(1))
            source_runs_min_max = (int(m.group(2)), int(m.group(3)))

    if pool_dim is None:
        pool_dim = _ana.POOL_DIM
        if pool_dim is None:
            return {'error': 'pool_dim unavailable'}

    saved_pool_dim = _ana.POOL_DIM
    _ana.POOL_DIM = int(pool_dim)

    try:
        # Find model name from any hidden state file
        mn_candidate = None
        for pfx in ('R0003_', 'R0001_'):
            for fp in sorted(glob.glob(os.path.join(hidden_dir, f'{pfx}*_trial*_turn01.npy'))):
                base = os.path.basename(fp)
                if '_trial' in base and base.startswith(pfx):
                    head = base.split('_trial')[0]
                    mn_candidate = head[len(pfx):]
                    break
            if mn_candidate:
                break
        if mn_candidate is None:
            return {'error': f'no hidden state files under {hidden_dir}'}

        # Source runs -- match canon
        SOURCE_RUNS_3WAY_LEGACY = [6, 7, 8, 13, 14, 15, 1, 3, 23]
        SOURCE_RUNS_3WAY_RENUM  = [3, 6, 7, 8, 13, 14, 15, 23, 28]
        if source_runs_min_max == (1, 23):
            source_runs = SOURCE_RUNS_3WAY_LEGACY
        elif source_runs_min_max == (3, 28):
            source_runs = SOURCE_RUNS_3WAY_RENUM
        else:
            source_runs = SOURCE_RUNS_3WAY_RENUM

        # v0.80.0.42: bypass _load_quadruplets entirely. Rationale:
        # _load_quadruplets includes a freshness check that invalidates
        # the cache if ANY .npy file in hidden_dir is newer than the
        # cache mtime, OR if any *_et_base.npy in the base sibling dir
        # is newer. After Run 0016 recovery (E_base regeneration), or
        # Run 0001 Pass 4 recompute, all caches across the project
        # would be invalidated en masse, forcing cold reload that
        # takes hours per cell on Windows.
        #
        # For Run 0057 we explicitly DON'T want to follow that
        # invalidation. Canon Ridge+MLP per-cell numbers in
        # channel_marginal_nonlinearity.csv were computed from the
        # data the existing caches reflect. If we force a cold reload
        # under a more recent E_base, RF gets fit on different data
        # than canon Ridge+MLP -- comparison stops being apples-to-
        # apples. Use the cache as-is. If it's truly broken, the
        # downstream stack will fail loud and we'll see it.
        #
        # Direct cache reader matches the load path in
        # analysis._qcache_load (lines 552-586) but skips the
        # freshness check. Schema is documented there.
        cache_path = os.path.join(
            hidden_dir,
            f'_qcache_ct_d{int(pool_dim)}_r{source_runs_min_max[0] if source_runs_min_max else 3}-{source_runs_min_max[1] if source_runs_min_max else 28}.npz'
        ) if source_runs_min_max else None
        if cache_path is None or not os.path.exists(cache_path):
            return {'error': f'no qcache at expected path {cache_path}'}

        import time as _time
        _t_load = _time.time()
        ui.msg(f"[load] reading {os.path.basename(cache_path)} ({os.path.getsize(cache_path)/1024/1024:.1f}MB)...")
        try:
            _d = np.load(cache_path, allow_pickle=False)
            _et = np.array(_d['e_t'])
            _ct = np.array(_d['c_t'])
            _sp = np.array(_d['s_prev'])
            _sn = np.array(_d['s_next'])
            _hr = np.array(_d['has_real_ct'])
            _pt = np.array(_d['prompt_tokens'])
            _d.close()
        except Exception as _le:
            return {'error': f'cache load failed: {type(_le).__name__}: {_le}'}
        _dt_load = _time.time() - _t_load
        ui.msg(f"[load] done in {_dt_load:.1f}s -- {len(_et)} rows, "
               f"{int(_hr.sum())} with real C_t")

        # Filter to rows with real C_t (mirrors analysis.py:1112).
        # Use boolean mask on the loaded arrays directly -- no list
        # comprehension over Python objects.
        _has_ct_mask = _hr.astype(bool)
        if not _has_ct_mask.any():
            return {'error': f'no rows with real C_t (loaded {len(_et)} raw)'}

        Xe = _et[_has_ct_mask].astype(np.float32)
        Xc = _ct[_has_ct_mask].astype(np.float32)
        Xs = _sp[_has_ct_mask].astype(np.float32)
        Xp = _pt[_has_ct_mask].reshape(-1, 1).astype(np.float32)
        # Replace nan prompt_tokens with 0 (matches canon convention)
        Xp[np.isnan(Xp)] = 0.0
        y  = _sn[_has_ct_mask].astype(np.float32)

        n = Xe.shape[0]
        ui.msg(f"[load] feature matrices ready: n={n}, "
               f"d_e={Xe.shape[1]}, d_c={Xc.shape[1]}, d_s={Xs.shape[1]}")

        # Per-channel feature matrices (with prompt_tokens covariate)
        Xe_Xp = np.hstack([Xe, Xp])
        Xs_Xp = np.hstack([Xs, Xp])

        # 80/20 split with random_state=42 -- canon
        Xe_tr, Xe_te, y_tr_raw, y_te_raw = train_test_split(
            Xe_Xp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        Xs_tr, Xs_te, _,        _        = train_test_split(
            Xs_Xp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # v0.80.0.45: target-PCA for RF on cells where pool_dim > 64.
        # Ridge and MLP handle multi-output natively in one fit; sklearn's
        # RandomForestRegressor stores a vector at each leaf and computes
        # variance reduction summed across all output dims at every
        # split. That cost scales linearly in n_outputs. At pool_dim=1024
        # (Gemma 2B cells) it's ~16x the per-split work of pool_dim=64
        # (LLaMA 8B / Gemma 9B), on top of unbounded max_depth and
        # min_samples_leaf=5 producing ~3,400 leaves per tree across
        # n_estimators=100 × 3 RF fits per cell. The 8B/9B cells run
        # fine native; the 2B cells hung 8+ min on the first fit pre-fix.
        #
        # Fix: PCA target to 64 dims for RF only on cells where
        # pool_dim > 64. Ridge and MLP retain canon native target dim
        # (pulled from channel_marginal_nonlinearity.csv, not re-fit)
        # per the pull-don't-recompute architecture. Within-cell
        # Ridge-vs-RF on 2B cells is then between estimators with
        # different output-space dims; honestly disclosed in the
        # per-cell target_dimensionality block, the cross-cell
        # cells_with_target_dim_reduction list, and the metadata's
        # protocol_decisions_during_run audit trail. The §5.4 mechanism
        # prediction (sign of asymmetry vs sign of R-hat gap) is
        # robust to this difference if the PCA reduction preserves
        # the dominant variance modes -- testable per-cell via the
        # recorded explained_variance_ratio.
        from sklearn.decomposition import PCA as _PCA
        native_pool_dim = int(y_tr_raw.shape[1])
        target_pca = None
        target_pca_evr = None
        target_pca_applied = False
        if native_pool_dim > 64:
            target_pca = _PCA(n_components=64,
                              random_state=RANDOM_STATE).fit(y_tr_raw)
            target_pca_evr = float(target_pca.explained_variance_ratio_.sum())
            target_pca_applied = True
            rf_target_dim = 64
            ui.msg(f"[pca] target reduced {native_pool_dim} -> 64 dims, "
                   f"explained variance ratio = {target_pca_evr:.4f}")
        else:
            rf_target_dim = native_pool_dim
            ui.msg(f"[pca] target at native dim {native_pool_dim}, "
                   f"no reduction")
        y_tr = (target_pca.transform(y_tr_raw)
                if target_pca is not None else y_tr_raw)
        y_te = (target_pca.transform(y_te_raw)
                if target_pca is not None else y_te_raw)

        sxE = StandardScaler().fit(Xe_tr)
        sxS = StandardScaler().fit(Xs_tr)
        sy  = StandardScaler().fit(y_tr)
        Xe_tr_s = sxE.transform(Xe_tr); Xe_te_s = sxE.transform(Xe_te)
        Xs_tr_s = sxS.transform(Xs_tr); Xs_te_s = sxS.transform(Xs_te)
        y_tr_s  = sy.transform(y_tr);   y_te_s  = sy.transform(y_te)

        # ─── Per-channel RF univariate fits ───────────────────────────
        import time as _time
        _t_rf_E = _time.time()
        ui.msg(f"[rf] fitting univariate-E (n={Xe_tr_s.shape[0]}, d={Xe_tr_s.shape[1]})...")
        rf_E = RandomForestRegressor(**RF_HYPERS).fit(Xe_tr_s, y_tr_s)
        ui.ok(f"[rf] univariate-E fit done ({_time.time()-_t_rf_E:.1f}s)")
        _t_rf_S = _time.time()
        ui.msg(f"[rf] fitting univariate-S (n={Xs_tr_s.shape[0]}, d={Xs_tr_s.shape[1]})...")
        rf_S = RandomForestRegressor(**RF_HYPERS).fit(Xs_tr_s, y_tr_s)
        ui.ok(f"[rf] univariate-S fit done ({_time.time()-_t_rf_S:.1f}s)")
        r2_rf_E = float(r2_score(y_te_s, rf_E.predict(Xe_te_s),
                                 multioutput='variance_weighted'))
        r2_rf_S = float(r2_score(y_te_s, rf_S.predict(Xs_te_s),
                                 multioutput='variance_weighted'))

        # ─── Joint RF fit (E + C + S + prompt) + permutation drops ────
        full_X = np.hstack([Xe, Xc, Xs, Xp])
        Xj_tr, Xj_te, yj_tr_raw, yj_te_raw = train_test_split(
            full_X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        # v0.80.0.45: apply same target-PCA as univariate fits for
        # within-cell consistency. random_state=42 ensures yj_tr_raw
        # equals y_tr_raw, but applying the transform explicitly is
        # clearer and safer than relying on split determinism across
        # multiple train_test_split calls on the same y.
        yj_tr = (target_pca.transform(yj_tr_raw)
                 if target_pca is not None else yj_tr_raw)
        yj_te = (target_pca.transform(yj_te_raw)
                 if target_pca is not None else yj_te_raw)
        sxJ = StandardScaler().fit(Xj_tr)
        syJ = StandardScaler().fit(yj_tr)
        Xj_tr_s = sxJ.transform(Xj_tr); Xj_te_s = sxJ.transform(Xj_te)
        yj_tr_s = syJ.transform(yj_tr); yj_te_s = syJ.transform(yj_te)

        _t_rf_J = _time.time()
        ui.msg(f"[rf] fitting joint (n={Xj_tr_s.shape[0]}, d={Xj_tr_s.shape[1]})...")
        rf_joint = RandomForestRegressor(**RF_HYPERS).fit(Xj_tr_s, yj_tr_s)
        ui.ok(f"[rf] joint fit done ({_time.time()-_t_rf_J:.1f}s)")
        r2_joint_train = float(r2_score(yj_tr_s, rf_joint.predict(Xj_tr_s),
                                        multioutput='variance_weighted'))
        r2_joint_test = float(r2_score(yj_te_s, rf_joint.predict(Xj_te_s),
                                       multioutput='variance_weighted'))

        # Permutation drops: shuffle each block, recompute test R²
        d_e = Xe.shape[1]; d_c = Xc.shape[1]; d_s = Xs.shape[1]
        slice_E = slice(0,           d_e)
        slice_C = slice(d_e,         d_e + d_c)
        slice_S = slice(d_e + d_c,   d_e + d_c + d_s)

        rng = np.random.RandomState(RANDOM_STATE)
        drops = {'E': [], 'C': [], 'R': []}
        baseline_r2 = r2_joint_test

        _t_perm = _time.time()
        ui.msg(f"[rf] permutation analysis: {N_PERMUTATIONS} shuffles × 3 blocks "
               f"on n={Xj_te_s.shape[0]} test rows...")
        for label, sl in [('E', slice_E), ('C', slice_C), ('R', slice_S)]:
            _t_block = _time.time()
            for _ in range(N_PERMUTATIONS):
                Xj_te_perm = Xj_te_s.copy()
                perm_idx = rng.permutation(Xj_te_perm.shape[0])
                Xj_te_perm[:, sl] = Xj_te_perm[perm_idx][:, sl]
                r2_p = float(r2_score(yj_te_s, rf_joint.predict(Xj_te_perm),
                                      multioutput='variance_weighted'))
                drops[label].append(baseline_r2 - r2_p)
            ui.ok(f"[rf] perm block {label} done ({_time.time()-_t_block:.1f}s, "
                  f"mean drop={np.mean(drops[label]):.4f})")
        ui.ok(f"[rf] permutation analysis total: {_time.time()-_t_perm:.1f}s")

        mean_drops = {k: float(np.mean(v)) for k, v in drops.items()}
        # Renormalize to sum 1 (matches canon partition convention)
        total = sum(max(0.0, v) for v in mean_drops.values())
        if total <= 0:
            partition = {'E': 0.0, 'C': 0.0, 'R': 0.0}
        else:
            partition = {k: max(0.0, v) / total for k, v in mean_drops.items()}

        return {
            'pool_dim':     int(pool_dim),
            'n_rows':       int(n),
            'r2_rf_E':      r2_rf_E,
            'r2_rf_S':      r2_rf_S,
            'rf_joint_R2_train':    r2_joint_train,
            'rf_joint_R2_held_out': r2_joint_test,
            'rf_partition':         partition,
            'rf_partition_raw_drops': mean_drops,
            # v0.80.0.45: target dim metadata
            'native_pool_dim':                        native_pool_dim,
            'rf_target_dim':                          rf_target_dim,
            'rf_target_pca_applied':                  target_pca_applied,
            'rf_target_pca_explained_variance_ratio': target_pca_evr,
        }
    finally:
        _ana.POOL_DIM = saved_pool_dim


# ─────────────────────────────────────────────────────────────────────────
#  RKHS analysis on a single cell -- mirrors _run_rf_on_cell exactly except
#  for the regressor (kernel ridge over the characteristic kernel grid).
#  v0.81.1.4 ship 2: replaces the Ridge-clone placeholder in the apparatus
#  loaders. Reuses ALL the data-loading / source-runs / qcache / target-PCA
#  / permutation logic from _run_rf_on_cell verbatim. Don't reinvent.
# ─────────────────────────────────────────────────────────────────────────

def _run_rkhs_on_cell(q42_path):
    """Per-channel and joint RKHS fits across the characteristic kernel grid.
    Returns dict or {'error': ...}.

    For each kernel in RKHS_KERNELS, fits kernel-ridge regression with
    RKHS_ALPHA regularization and computes test R² + permutation drops on
    the joint (E + C + S + prompt) feature matrix. RKHS_median is the
    per-channel median share across the seven kernels; min/max range is
    reported alongside so the dispersion across the grid is visible.

    n_train is capped at RKHS_N_SUBSAMPLE because RKHS scales O(n^3); the
    cap makes the kernel grid tractable on 24 cells. Subsample is fixed-
    seed, drawn from the in-sample 80% train split (test 20% is full).

    Mirrors _run_rf_on_cell's data-loading and permutation protocol exactly
    -- see that function for cache-loading, has_real_ct masking, target-PCA
    on cells with native pool_dim > 64, and the permutation-drop loop. Only
    the regressor differs.
    """
    import numpy as np
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # POOL_DIM + source_runs detection -- copied from _run_rf_on_cell
    sys.path.insert(0, ROOT)
    import analysis as _ana

    cell_dir = os.path.dirname(os.path.dirname(q42_path))
    hidden_dir = os.path.join(cell_dir, 'hidden_states')

    pool_dim = None
    source_runs_min_max = None
    cache_glob = os.path.join(hidden_dir, '_qcache_ct_d*_r*.npz')
    cache_files = sorted(glob.glob(cache_glob))
    if cache_files:
        m = re.match(r'_qcache_ct_d(\d+)_r(\d+)-(\d+)\.npz$',
                     os.path.basename(cache_files[0]))
        if m:
            pool_dim = int(m.group(1))
            source_runs_min_max = (int(m.group(2)), int(m.group(3)))

    if pool_dim is None:
        pool_dim = _ana.POOL_DIM
        if pool_dim is None:
            return {'error': 'pool_dim unavailable'}

    saved_pool_dim = _ana.POOL_DIM
    _ana.POOL_DIM = int(pool_dim)

    try:
        # Cache load -- copied verbatim from _run_rf_on_cell
        cache_path = os.path.join(
            hidden_dir,
            f'_qcache_ct_d{int(pool_dim)}_r{source_runs_min_max[0] if source_runs_min_max else 3}-{source_runs_min_max[1] if source_runs_min_max else 28}.npz'
        ) if source_runs_min_max else None
        if cache_path is None or not os.path.exists(cache_path):
            return {'error': f'no qcache at expected path {cache_path}'}

        import time as _time
        _t_load = _time.time()
        ui.msg(f"[rkhs load] reading {os.path.basename(cache_path)} "
               f"({os.path.getsize(cache_path)/1024/1024:.1f}MB)...")
        try:
            _d = np.load(cache_path, allow_pickle=False)
            _et = np.array(_d['e_t'])
            _ct = np.array(_d['c_t'])
            _sp = np.array(_d['s_prev'])
            _sn = np.array(_d['s_next'])
            _hr = np.array(_d['has_real_ct'])
            _pt = np.array(_d['prompt_tokens'])
            _d.close()
        except Exception as _le:
            return {'error': f'cache load failed: {type(_le).__name__}: {_le}'}
        _dt_load = _time.time() - _t_load
        ui.msg(f"[rkhs load] done in {_dt_load:.1f}s -- {len(_et)} rows, "
               f"{int(_hr.sum())} with real C_t")

        _has_ct_mask = _hr.astype(bool)
        if not _has_ct_mask.any():
            return {'error': f'no rows with real C_t (loaded {len(_et)} raw)'}

        Xe = _et[_has_ct_mask].astype(np.float32)
        Xc = _ct[_has_ct_mask].astype(np.float32)
        Xs = _sp[_has_ct_mask].astype(np.float32)
        Xp = _pt[_has_ct_mask].reshape(-1, 1).astype(np.float32)
        Xp[np.isnan(Xp)] = 0.0
        y  = _sn[_has_ct_mask].astype(np.float32)

        n = Xe.shape[0]

        # Joint matrix: full feature set, mirrors RF
        full_X = np.hstack([Xe, Xc, Xs, Xp])
        Xj_tr_full, Xj_te, yj_tr_full_raw, yj_te_raw = train_test_split(
            full_X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # Subsample train side -- RKHS only. Test is full.
        rng_sub = np.random.RandomState(RANDOM_STATE)
        n_train_full = Xj_tr_full.shape[0]
        if n_train_full > RKHS_N_SUBSAMPLE:
            sub_idx = rng_sub.choice(n_train_full, size=RKHS_N_SUBSAMPLE,
                                      replace=False)
            Xj_tr     = Xj_tr_full[sub_idx]
            yj_tr_raw = yj_tr_full_raw[sub_idx]
            ui.msg(f"[rkhs] subsampled train {n_train_full} -> "
                   f"{RKHS_N_SUBSAMPLE} for kernel-ridge tractability")
        else:
            Xj_tr     = Xj_tr_full
            yj_tr_raw = yj_tr_full_raw

        # Target-PCA -- same as RF, on cells with native pool_dim > 64
        from sklearn.decomposition import PCA as _PCA
        native_pool_dim = int(yj_tr_raw.shape[1])
        target_pca = None
        target_pca_evr = None
        target_pca_applied = False
        if native_pool_dim > 64:
            target_pca = _PCA(n_components=64,
                              random_state=RANDOM_STATE).fit(yj_tr_raw)
            target_pca_evr = float(target_pca.explained_variance_ratio_.sum())
            target_pca_applied = True
            rkhs_target_dim = 64
        else:
            rkhs_target_dim = native_pool_dim
        yj_tr = (target_pca.transform(yj_tr_raw)
                 if target_pca is not None else yj_tr_raw)
        yj_te = (target_pca.transform(yj_te_raw)
                 if target_pca is not None else yj_te_raw)

        # Standardize joint X and y
        sxJ = StandardScaler().fit(Xj_tr)
        syJ = StandardScaler().fit(yj_tr)
        Xj_tr_s = sxJ.transform(Xj_tr); Xj_te_s = sxJ.transform(Xj_te)
        yj_tr_s = syJ.transform(yj_tr); yj_te_s = syJ.transform(yj_te)

        # v0.82.0.6: subsample TEST side for the permutation loop. The
        # baseline R² is computed on the FULL test set (one predict per
        # kernel), but the permutation predicts that follow run on a
        # smaller subset. Each permutation predict's wall time is
        # dominated by the n_te × n_tr Matérn Gram build (numpy O(n_te ×
        # n_tr × d)). Cutting n_te from ~3500 to 1500 cuts per-predict
        # cost by ~2.3x. The mean-drop estimate is unbiased -- same
        # permutation protocol, smaller test sample. Standard error on
        # the per-channel share grows by ~sqrt(2.3) ≈ 1.5x, easily
        # absorbed by the median-across-7-kernels reduction.
        RKHS_N_TEST_PERM = 1500
        if Xj_te_s.shape[0] > RKHS_N_TEST_PERM:
            rng_te = np.random.RandomState(RANDOM_STATE + 1)
            perm_te_idx = rng_te.choice(Xj_te_s.shape[0],
                                          size=RKHS_N_TEST_PERM,
                                          replace=False)
            Xj_te_perm_base = Xj_te_s[perm_te_idx]
            yj_te_perm_base = yj_te_s[perm_te_idx]
        else:
            Xj_te_perm_base = Xj_te_s
            yj_te_perm_base = yj_te_s

        # Median pairwise distance -- for RBF + Matérn length-scale anchor
        # Compute on a 1000-row subsample of train to keep this O(1M flops)
        _ms_idx = rng_sub.choice(Xj_tr_s.shape[0],
                                  size=min(1000, Xj_tr_s.shape[0]),
                                  replace=False)
        _pairwise_d = euclidean_distances(Xj_tr_s[_ms_idx])
        # Off-diagonal median (excludes the zeros on the diagonal)
        _pairwise_offdiag = _pairwise_d[np.triu_indices_from(_pairwise_d, k=1)]
        median_pairwise = float(np.median(_pairwise_offdiag))
        if median_pairwise <= 0:
            median_pairwise = 1.0  # fallback for degenerate case

        # Permutation slicing -- same as RF
        d_e = Xe.shape[1]; d_c = Xc.shape[1]; d_s = Xs.shape[1]
        slice_E = slice(0,           d_e)
        slice_C = slice(d_e,         d_e + d_c)
        slice_S = slice(d_e + d_c,   d_e + d_c + d_s)

        # ─── Per-kernel fits + permutation drops ─────────────────────────
        per_kernel_results = []
        partitions_per_kernel = {'E': [], 'C': [], 'R': []}

        for kernel_spec in RKHS_KERNELS:
            kind = kernel_spec['kind']
            ls_mult = kernel_spec['length_scale_mult']
            length_scale = ls_mult * median_pairwise

            if kind == 'rbf':
                # KernelRidge with kernel='rbf' string takes the C-optimized
                # path: gamma = 1 / (2 * length_scale^2).
                gamma = 1.0 / (2.0 * length_scale ** 2)
                kr = KernelRidge(alpha=RKHS_ALPHA, kernel='rbf', gamma=gamma)
                kernel_id = f"rbf_ls{ls_mult:.1f}"
                use_precomputed = False
                gram_train = None
                matern_obj = None
            elif kind == 'matern':
                # v0.82.0.5 hot-patch: Matérn-as-callable on KernelRidge
                # was the production hang. Passing a sklearn
                # gaussian_process.kernels.Matern callable to KernelRidge
                # triggers per-predict re-evaluation of the kernel through
                # sklearn's slow Python kernel-evaluation machinery -- each
                # predict() rebuilds the n_pred × n_train Gram matrix from
                # scratch, with no caching. In a permutation loop with 50
                # perms × 3 channels = 150 predicts per kernel, that's the
                # 5+ hour-per-cell hang we observed in the field.
                #
                # Fix: precompute the Gram matrices ONCE per fit, then use
                # KernelRidge with kernel='precomputed' so its fit and
                # predict are pure linear-algebra against an already-built
                # matrix. Predict uses the train-vs-test Gram block, also
                # built once outside the loop. Permutation predicts then
                # reuse the same train Gram and rebuild only the relevant
                # subset of the test Gram -- but for now we just rebuild
                # the test Gram per permutation, which is O(n_te × n_tr × d)
                # per call but in numpy/C, not Python -- vastly faster than
                # the callable path.
                from sklearn.gaussian_process.kernels import Matern as _Matern
                matern_obj = _Matern(length_scale=length_scale,
                                       nu=kernel_spec['nu'])
                kr = KernelRidge(alpha=RKHS_ALPHA, kernel='precomputed')
                kernel_id = f"matern_nu{kernel_spec['nu']}"
                use_precomputed = True
                # Build train Gram once. Matern.__call__(X, Y) returns the
                # Gram matrix at C speed via numpy ops inside the Matern
                # implementation -- fast, the slow part was sklearn calling
                # this from KernelRidge.predict on every single call.
                gram_train = matern_obj(Xj_tr_s, Xj_tr_s)
            else:
                continue

            _t_fit = _time.time()
            ui.msg(f"[rkhs] fitting {kernel_id} (n_train={Xj_tr_s.shape[0]})...")
            try:
                if use_precomputed:
                    kr.fit(gram_train, yj_tr_s)
                    # Test Gram: rows = test points, cols = train points
                    gram_test_train = matern_obj(Xj_te_s, Xj_tr_s)
                    pred_te = kr.predict(gram_test_train)
                    # We don't compute pred_tr/r2_train for matern path --
                    # n_train × n_train Gram already built; predicting on
                    # train requires no extra work, but we don't need
                    # r2_train downstream and skipping it saves a matmul.
                    r2_train = None
                else:
                    kr.fit(Xj_tr_s, yj_tr_s)
                    pred_tr = kr.predict(Xj_tr_s)
                    pred_te = kr.predict(Xj_te_s)
                    r2_train = float(r2_score(yj_tr_s, pred_tr,
                                              multioutput='variance_weighted'))
                r2_test  = float(r2_score(yj_te_s, pred_te,
                                          multioutput='variance_weighted'))
            except Exception as _ke:
                ui.warn(f"[rkhs] {kernel_id} fit failed: "
                        f"{type(_ke).__name__}: {_ke}")
                continue
            ui.ok(f"[rkhs] {kernel_id} fit done ({_time.time()-_t_fit:.1f}s, "
                  f"R²_test={r2_test:.4f})")

            # Permutation drops. v0.82.0.7: reduced to 10 perms per
            # channel (committee option B). Mean-drop estimate at 10
            # perms is unbiased; SE is sqrt(3) higher than at 30 perms.
            # Median-across-3-kernels in the final share absorbs that.
            drops = {'E': [], 'C': [], 'R': []}
            rng = np.random.RandomState(RANDOM_STATE)
            n_perm_rkhs = 10
            n_te_perm = Xj_te_perm_base.shape[0]
            for label, sl in [('E', slice_E), ('C', slice_C), ('R', slice_S)]:
                _t_ch = _time.time()
                for _ in range(n_perm_rkhs):
                    Xj_te_perm = Xj_te_perm_base.copy()
                    perm_idx = rng.permutation(n_te_perm)
                    Xj_te_perm[:, sl] = Xj_te_perm[perm_idx][:, sl]
                    if use_precomputed:
                        # Rebuild test Gram against frozen train. This is
                        # the dominant cost (n_te × n_tr × d numpy op) but
                        # it's a single C-level call, not 50k Python kernel
                        # evaluations.
                        gram_perm = matern_obj(Xj_te_perm, Xj_tr_s)
                        r2_p = float(r2_score(yj_te_perm_base, kr.predict(gram_perm),
                                              multioutput='variance_weighted'))
                    else:
                        r2_p = float(r2_score(yj_te_perm_base, kr.predict(Xj_te_perm),
                                              multioutput='variance_weighted'))
                    drops[label].append(r2_test - r2_p)
                _ch_t = _time.time() - _t_ch
                ui.msg(f"[rkhs]   {kernel_id} channel {label}: "
                       f"{n_perm_rkhs} perms in {_ch_t:.1f}s "
                       f"({_ch_t/n_perm_rkhs:.2f}s/perm)")

            mean_drops = {k: float(np.mean(v)) for k, v in drops.items()}
            total = sum(max(0.0, v) for v in mean_drops.values())
            if total <= 0:
                partition = {'E': 0.0, 'C': 0.0, 'R': 0.0}
            else:
                partition = {k: max(0.0, v) / total
                             for k, v in mean_drops.items()}

            per_kernel_results.append({
                'kernel_id':       kernel_id,
                'kernel_kind':     kind,
                'length_scale':    length_scale,
                'length_scale_mult': ls_mult,
                'nu':              kernel_spec.get('nu'),
                'r2_train':        r2_train,
                'r2_test':         r2_test,
                'partition':       partition,
                'partition_raw_drops': mean_drops,
            })
            for ch in ('E', 'C', 'R'):
                partitions_per_kernel[ch].append(partition[ch])

        if not per_kernel_results:
            return {'error': 'all RKHS kernels failed to fit'}

        # ─── Median + range across the kernel grid ───────────────────────
        rkhs_median = {ch: float(np.median(partitions_per_kernel[ch]))
                       for ch in ('E', 'C', 'R')}
        # Renormalize the median triple back to the simplex (median of three
        # numbers that each sum to 1 across kernels need not itself sum to 1).
        med_tot = sum(rkhs_median.values())
        if med_tot > 0:
            rkhs_median = {ch: v / med_tot for ch, v in rkhs_median.items()}
        rkhs_range = {ch: {
            'min': float(min(partitions_per_kernel[ch])),
            'max': float(max(partitions_per_kernel[ch])),
        } for ch in ('E', 'C', 'R')}

        # Median R² across the kernel grid (informational)
        median_r2_test = float(np.median([r['r2_test']
                                           for r in per_kernel_results]))

        return {
            'pool_dim':               int(pool_dim),
            'n_rows':                 int(n),
            'n_train_subsampled':     int(Xj_tr_s.shape[0]),
            'median_pairwise_distance': median_pairwise,
            'rkhs_median_r2_test':    median_r2_test,
            'rkhs_median_partition':  rkhs_median,
            'rkhs_range_per_channel': rkhs_range,
            'rkhs_per_kernel':        per_kernel_results,
            'native_pool_dim':        native_pool_dim,
            'rkhs_target_dim':        rkhs_target_dim,
            'rkhs_target_pca_applied': target_pca_applied,
            'rkhs_target_pca_explained_variance_ratio': target_pca_evr,
        }
    finally:
        _ana.POOL_DIM = saved_pool_dim


# ─────────────────────────────────────────────────────────────────────────
#  Per-cell aggregation -- combine canon + RF
# ─────────────────────────────────────────────────────────────────────────

def _build_cell_entry(family, size, variant, cond, temp,
                      rf_result, canon_csv_row, ridge_partition,
                      mlp_partition, partition_source,
                      rkhs_result=None):
    """Build the schema-conformant cell dict from RF + RKHS + canon pulls.

    v0.81.1.4: rkhs_result added as optional argument. When present, the
    cell entry gains rkhs_median entries in univariate_R2 / joint_R2 /
    permutation_shares parallel to ridge/mlp/rf, and a per-kernel block
    under 'rkhs_kernel_grid' for the full kernel-by-kernel breakdown.
    Backwards compatible: cells without rkhs_result get rkhs_median=None
    populated in the four-class slots.
    """
    if 'error' in rf_result:
        return {'error': rf_result['error']}

    rkhs_partition = (rkhs_result.get('rkhs_median_partition')
                      if (rkhs_result and 'error' not in rkhs_result)
                      else None)
    rkhs_r2_test = (rkhs_result.get('rkhs_median_r2_test')
                    if (rkhs_result and 'error' not in rkhs_result)
                    else None)

    cell_entry = {
        'pool_dim': rf_result['pool_dim'],
        'n_rows':   rf_result['n_rows'],
        'univariate_R2': {
            'ridge': {
                'E': float(canon_csv_row['r2_ridge_E']) if canon_csv_row else None,
                'S': float(canon_csv_row['r2_ridge_S']) if canon_csv_row else None,
            },
            'mlp': {
                'E': float(canon_csv_row['r2_mlp_E']) if canon_csv_row else None,
                'S': float(canon_csv_row['r2_mlp_S']) if canon_csv_row else None,
            },
            'rf': {
                'E': rf_result['r2_rf_E'],
                'S': rf_result['r2_rf_S'],
            },
            # v0.81.1.4: RKHS univariate not separately fit (kernel grid
            # is joint-only -- separate univariate fits would multiply
            # wall-time by 3x and aren't needed for the four-class
            # apparatus consumer). Recorded null here for schema parity.
            'rkhs_median': {
                'E': None,
                'S': None,
            },
        },
        'joint_R2_held_out': {
            'ridge': None,
            'mlp':   None,
            'rf':    rf_result['rf_joint_R2_held_out'],
            'rkhs_median': rkhs_r2_test,
        },
        'permutation_shares': {
            'ridge': ridge_partition,
            'mlp':   mlp_partition,
            'rf':    rf_result['rf_partition'],
            # v0.81.1.4: real RKHS shares (not Ridge-clone placeholder).
            # Median across the 7-kernel characteristic grid (Matérn ν ∈
            # {0.5, 1.5, 2.5} + RBF at 4 length scales of median pairwise
            # distance). Range across the grid recorded under
            # 'rkhs_kernel_grid' for dispersion inspection.
            'rkhs_median': rkhs_partition,
        },
        'partition_source': partition_source,
        # v0.80.0.45: target dim per-estimator audit trail.
        # v0.81.1.4: rkhs_target_dim added; same PCA reduction rule as RF
        # (target -> 64 dims on cells with native pool_dim > 64).
        'target_dimensionality': {
            'native_pool_dim':       rf_result.get('native_pool_dim'),
            'rf_target_dim':         rf_result.get('rf_target_dim'),
            'ridge_target_dim':      rf_result.get('native_pool_dim'),
            'mlp_target_dim':        rf_result.get('native_pool_dim'),
            'rkhs_target_dim':       (rkhs_result.get('rkhs_target_dim')
                                       if rkhs_result else None),
            'rf_target_pca_applied': rf_result.get('rf_target_pca_applied'),
            'rf_target_pca_explained_variance_ratio':
                rf_result.get('rf_target_pca_explained_variance_ratio'),
            'rkhs_target_pca_applied': (rkhs_result.get('rkhs_target_pca_applied')
                                         if rkhs_result else None),
            'rkhs_target_pca_explained_variance_ratio':
                (rkhs_result.get('rkhs_target_pca_explained_variance_ratio')
                 if rkhs_result else None),
        },
    }

    # v0.81.1.4: full per-kernel RKHS breakdown for inspection / debugging
    if rkhs_result and 'error' not in rkhs_result:
        cell_entry['rkhs_kernel_grid'] = {
            'n_train_subsampled':     rkhs_result.get('n_train_subsampled'),
            'median_pairwise_distance': rkhs_result.get('median_pairwise_distance'),
            'range_per_channel':      rkhs_result.get('rkhs_range_per_channel'),
            'per_kernel':             rkhs_result.get('rkhs_per_kernel'),
        }
    elif rkhs_result and 'error' in rkhs_result:
        cell_entry['rkhs_kernel_grid'] = {'error': rkhs_result['error']}
    else:
        cell_entry['rkhs_kernel_grid'] = None

    # Asymmetry computations (canon vs RF)
    asym = {}
    r_hat_gap = {}
    sign_check = {}

    # MLP vs Ridge (canon)
    if canon_csv_row:
        try:
            gap_E_mlp = float(canon_csv_row['gap_E'])
            gap_S_mlp = float(canon_csv_row['gap_S'])
            asymmetry_mlp = float(canon_csv_row['asymmetry'])
            asym['mlp_vs_ridge'] = {
                'gap_E': gap_E_mlp,
                'gap_S': gap_S_mlp,
                'asymmetry': asymmetry_mlp,
            }
            if (ridge_partition and mlp_partition and
                'R' in ridge_partition and 'R' in mlp_partition):
                r_hat_mlp = mlp_partition['R'] - ridge_partition['R']
                r_hat_gap['mlp_vs_ridge'] = r_hat_mlp
                # Sign check -- opposite signs predicted
                sign_check['mlp_vs_ridge_signs_opposite'] = (
                    (asymmetry_mlp > 0 and r_hat_mlp < 0) or
                    (asymmetry_mlp < 0 and r_hat_mlp > 0)
                )
        except (KeyError, TypeError, ValueError):
            pass

    # RF vs Ridge (new)
    if canon_csv_row:
        try:
            gap_E_rf = rf_result['r2_rf_E'] - float(canon_csv_row['r2_ridge_E'])
            gap_S_rf = rf_result['r2_rf_S'] - float(canon_csv_row['r2_ridge_S'])
            asymmetry_rf = gap_E_rf - gap_S_rf
            asym['rf_vs_ridge'] = {
                'gap_E': gap_E_rf,
                'gap_S': gap_S_rf,
                'asymmetry': asymmetry_rf,
            }
            if (ridge_partition and 'R' in ridge_partition):
                r_hat_rf = rf_result['rf_partition']['R'] - ridge_partition['R']
                r_hat_gap['rf_vs_ridge'] = r_hat_rf
                sign_check['rf_vs_ridge_signs_opposite'] = (
                    (asymmetry_rf > 0 and r_hat_rf < 0) or
                    (asymmetry_rf < 0 and r_hat_rf > 0)
                )
        except (KeyError, TypeError, ValueError):
            pass

    cell_entry['asymmetry'] = asym
    cell_entry['R_hat_gap'] = r_hat_gap
    cell_entry['sign_check'] = sign_check

    return cell_entry


# ─────────────────────────────────────────────────────────────────────────
#  Cross-cell aggregates -- writerbot-spec summary
# ─────────────────────────────────────────────────────────────────────────

def _build_cross_cell_aggregates(cells):
    """Per writerbot's handoff: summary fields for §5.4 prose."""
    valid = {k: v for k, v in cells.items() if 'error' not in v}
    n_total = len(valid)

    # Track per-pair sign-check results
    mlp_pattern_cells = []
    rf_pattern_cells = []
    both_cells = []
    only_mlp_cells = []
    only_rf_cells = []
    neither_cells = []

    for k, v in valid.items():
        sc = v.get('sign_check', {})
        mlp_ok = sc.get('mlp_vs_ridge_signs_opposite')
        rf_ok = sc.get('rf_vs_ridge_signs_opposite')

        if mlp_ok is True:
            mlp_pattern_cells.append(k)
        if rf_ok is True:
            rf_pattern_cells.append(k)

        if mlp_ok is True and rf_ok is True:
            both_cells.append(k)
        elif mlp_ok is True and rf_ok is not True:
            only_mlp_cells.append(k)
        elif rf_ok is True and mlp_ok is not True:
            only_rf_cells.append(k)
        elif mlp_ok is False and rf_ok is False:
            neither_cells.append(k)

    n_with_both_signs = len(both_cells)
    n_with_mlp_pattern = len(mlp_pattern_cells)
    n_with_rf_pattern = len(rf_pattern_cells)

    # v0.80.0.45: target-dim reduction summary
    cells_with_reduction = []
    evr_values = []
    for ck, cv in valid.items():
        td = cv.get('target_dimensionality', {})
        if td.get('rf_target_pca_applied'):
            cells_with_reduction.append(ck)
            _evr = td.get('rf_target_pca_explained_variance_ratio')
            if isinstance(_evr, (int, float)):
                evr_values.append(_evr)

    return {
        'n_cells_evaluated': n_total,
        'rf_reproduces_mlp_sign_on_R_hat_gap':
            f"{len(both_cells) + len(only_rf_cells)}/{n_total}"
            if n_total else "0/0",
        'rf_reproduces_mlp_sign_on_asymmetry':
            f"{n_with_rf_pattern}/{n_total}" if n_total else "0/0",
        'both_classes_show_predicted_opposite_signs':
            f"{n_with_both_signs}/{n_total}" if n_total else "0/0",
        'cells_where_only_mlp_shows_pattern':  sorted(only_mlp_cells),
        'cells_where_only_rf_shows_pattern':   sorted(only_rf_cells),
        'cells_where_neither_shows_pattern':   sorted(neither_cells),
        'cells_where_both_show_pattern':       sorted(both_cells),
        # v0.80.0.45: target dim reduction summary
        'cells_with_target_dim_reduction':     sorted(cells_with_reduction),
        'target_pca_explained_variance_minimum':
            float(min(evr_values)) if evr_values else None,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Resume helpers (v0.80.0.49)
# ─────────────────────────────────────────────────────────────────────────

def _is_cell_rf_complete(cell_entry):
    """True iff the expensive RF work for this cell is already in the
    cell_entry. Used by the resume logic to decide whether to skip
    _run_rf_on_cell on the next pass.

    A cell is RF-complete iff:
      - It is a dict with no 'error' key,
      - univariate_R2.rf has both 'E' and 'S' populated,
      - permutation_shares.rf has 'R' populated.

    Canon-side fields (R_hat_gap, sign_check, permutation_shares
    ridge/mlp) are NOT required to be complete -- those are cheap to
    recompute on resume, and a 0.80.0.46-style cell with broken
    canon pull is correctly recognized as RF-complete here so its
    expensive RF work is preserved while its canon side gets
    rebuilt from a fresh pull on resume.
    """
    if not isinstance(cell_entry, dict):
        return False
    if 'error' in cell_entry:
        return False
    uni = cell_entry.get('univariate_R2', {}).get('rf') or {}
    if 'E' not in uni or 'S' not in uni:
        return False
    perm_rf = cell_entry.get('permutation_shares', {}).get('rf')
    if not isinstance(perm_rf, dict) or 'R' not in perm_rf:
        return False
    return True


def _extract_rf_result_from_cell(cell_entry):
    """Reconstruct the rf_result dict that _build_cell_entry expects
    from an RF-complete cell_entry. Used on resume to feed the
    cell_entry rebuild without re-running RF.

    Two fields that the original rf_result had but the cell_entry
    doesn't store are reconstructed as None: rf_joint_R2_train and
    rf_partition_raw_drops. _build_cell_entry doesn't consume
    either, so None is harmless."""
    td = cell_entry.get('target_dimensionality', {}) or {}
    return {
        'pool_dim':                cell_entry.get('pool_dim'),
        'n_rows':                  cell_entry.get('n_rows'),
        'r2_rf_E':                 cell_entry.get('univariate_R2', {}).get('rf', {}).get('E'),
        'r2_rf_S':                 cell_entry.get('univariate_R2', {}).get('rf', {}).get('S'),
        'rf_joint_R2_train':       None,
        'rf_joint_R2_held_out':    cell_entry.get('joint_R2_held_out', {}).get('rf'),
        'rf_partition':            cell_entry.get('permutation_shares', {}).get('rf'),
        'rf_partition_raw_drops':  None,
        'native_pool_dim':         td.get('native_pool_dim'),
        'rf_target_dim':           td.get('rf_target_dim'),
        'rf_target_pca_applied':   td.get('rf_target_pca_applied'),
        'rf_target_pca_explained_variance_ratio':
            td.get('rf_target_pca_explained_variance_ratio'),
    }


def _extract_rkhs_result_from_cell(cell_entry):
    """v0.81.1.4: reconstruct an rkhs_result dict from a cell_entry that
    already has rkhs_kernel_grid populated. Returns None if the cell was
    written pre-0.81.1.4 (no rkhs_kernel_grid block) -- caller falls back
    to running _run_rkhs_on_cell fresh.

    Mirrors _extract_rf_result_from_cell. Used on resume so the kernel
    grid (~10-30 minutes per cell) doesn't re-fire if already cached."""
    if not isinstance(cell_entry, dict):
        return None
    rkg = cell_entry.get('rkhs_kernel_grid')
    if rkg is None:
        return None
    if isinstance(rkg, dict) and 'error' in rkg:
        # Prior run errored on RKHS -- re-run fresh
        return None
    perm = cell_entry.get('permutation_shares', {}) or {}
    j_r2 = cell_entry.get('joint_R2_held_out', {}) or {}
    td = cell_entry.get('target_dimensionality', {}) or {}
    rkhs_partition = perm.get('rkhs_median')
    if rkhs_partition is None:
        return None
    return {
        'pool_dim':                cell_entry.get('pool_dim'),
        'n_rows':                  cell_entry.get('n_rows'),
        'n_train_subsampled':      rkg.get('n_train_subsampled'),
        'median_pairwise_distance': rkg.get('median_pairwise_distance'),
        'rkhs_median_r2_test':     j_r2.get('rkhs_median'),
        'rkhs_median_partition':   rkhs_partition,
        'rkhs_range_per_channel':  rkg.get('range_per_channel'),
        'rkhs_per_kernel':         rkg.get('per_kernel'),
        'native_pool_dim':         td.get('native_pool_dim'),
        'rkhs_target_dim':         td.get('rkhs_target_dim'),
        'rkhs_target_pca_applied': td.get('rkhs_target_pca_applied'),
        'rkhs_target_pca_explained_variance_ratio':
            td.get('rkhs_target_pca_explained_variance_ratio'),
    }


def _load_existing_q57(out_path):
    """Load existing Q0057 JSON for resume. Returns (cells_dict,
    metadata_dict). Both empty if file is missing, corrupt, or
    structurally unexpected."""
    import json as _json
    if not os.path.exists(out_path):
        return {}, {}
    try:
        with open(out_path, 'r', encoding='utf-8-sig') as f:
            data = _json.load(f)
    except Exception:
        return {}, {}
    cells = data.get('cells')
    metadata = data.get('metadata')
    if not isinstance(cells, dict):
        cells = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return cells, metadata


def _write_q57_incremental(out_path, cells, seed_metadata):
    """Atomically write the current Q0057 state. Cross-cell
    aggregates are recomputed from current cells. Metadata
    inherits from seed_metadata (so any prior protocol_decisions
    from earlier ships or the hot-patch are preserved) with
    timestamp + iota_version + incremental_save flag updated.

    Atomic via tempfile + os.replace so a kill during write
    doesn't leave a half-written JSON."""
    import json as _json
    aggregates = _build_cross_cell_aggregates(cells)
    md = dict(seed_metadata) if seed_metadata else {}
    md['timestamp'] = datetime.datetime.utcnow().isoformat() + 'Z'
    md['iota_version'] = '0.82.0.23'
    md['incremental_save'] = True
    out = {
        'function_classes': ['ridge', 'mlp', 'rf', 'rkhs_median'],
        'cells': cells,
        'cross_cell_aggregates': aggregates,
        'metadata': md,
    }
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        _json.dump(out, f, indent=2, ensure_ascii=False)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp_path, out_path)


# ─────────────────────────────────────────────────────────────────────────
#  Main driver
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--root', default=DATA)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"! data root not found: {args.root}", file=sys.stderr)
        return 1

    ui.section("Run 0057 -- Function-Class Sensitivity Analysis")

    # Discover all Q0042 cells
    pattern = os.path.join(args.root, '**', 'Q0042_decomposition.json')
    q42_files = sorted(glob.glob(pattern, recursive=True))
    if not q42_files:
        ui.err(f"no Q0042 files found under {args.root}")
        return 1
    ui.msg(f"Found {len(q42_files)} Q0042 cell(s).")

    # Load canon sources once
    canon_csv = _load_channel_marginal_csv()
    if not canon_csv:
        ui.err("channel_marginal_nonlinearity.csv not found -- Run 0056 (calibration) first.")
        return 1
    ui.ok(f"Loaded canon Ridge+MLP per-channel R² for {len(canon_csv)} cell(s).")

    results_cells = _load_results_json()
    ui.ok(f"Loaded {len(results_cells)} cells from results.json (for partition pull).")

    # v0.80.0.49: load existing Q0057 for resume support. Cells with
    # RF outputs already in place will skip the expensive RF refit;
    # canon-side fields (R_hat_gap, sign_check, permutation_shares
    # ridge/mlp, asymmetry) get rebuilt from a fresh canon pull
    # regardless of resume status, so a 0.80.0.47-style canon fix
    # takes effect on resume without needing a standalone hot-patch.
    existing_cells, existing_metadata = _load_existing_q57(args.out)
    n_resume_skip = sum(
        1 for k, v in existing_cells.items() if _is_cell_rf_complete(v)
    )
    if n_resume_skip > 0:
        ui.ok(f"[resume] {n_resume_skip} cell(s) have cached RF outputs in "
              f"existing Q0057 -- RF will be skipped for those; canon side "
              f"will be rebuilt fresh.")

    # Process each cell
    cells = dict(existing_cells)
    n_done = 0
    n_failed = 0
    n_resumed = 0
    t0 = time.time()
    for idx, q42 in enumerate(q42_files, 1):
        parsed = _parse_q42_path(q42)
        if parsed is None:
            n_failed += 1
            continue
        family, size, variant, cond, temp = parsed

        cell_key = _cell_key(family, size, variant, temp)
        canon_key = (family, size, variant, cond)
        canon_row = canon_csv.get(canon_key)

        if canon_row is None:
            ui.warn(f"[{idx}/{len(q42_files)}] {cell_key}: no canon CSV row -- skipping")
            cells[cell_key] = {'error': 'no canon Ridge/MLP per-channel CSV row'}
            n_failed += 1
            continue

        # Pull canon partition
        # v0.80.0.47: pass (family, size, variant, temp) so the function
        # can build the canon-convention results.json key. Without these
        # tuple args, every lookup returns None and the partition side
        # of Q0057 stays empty.
        ridge_partition, mlp_partition, partition_source = \
            _pull_canon_partition(cell_key, results_cells,
                                  family=family, size=size,
                                  variant=variant, temp=temp)

        t_cell = time.time()
        # v0.80.0.49: resume check. If RF outputs already cached for
        # this cell in the existing Q0057, skip the expensive RF
        # refit and just rebuild the canon-side fields from the
        # fresh canon pull above.
        existing_for_cell = cells.get(cell_key, {})
        rf_cached = _is_cell_rf_complete(existing_for_cell)
        if rf_cached:
            ui.section(f"[{idx}/{len(q42_files)}] {cell_key} (T={temp}) -- RESUME (RF cached)")
            ui.msg("[resume] reusing stored RF outputs; rebuilding canon-side fields from fresh pull")
            rf_result = _extract_rf_result_from_cell(existing_for_cell)
            # v0.81.1.4: RKHS resume -- extract from existing cell if present.
            # If absent (e.g. the cell was last written pre-0.81.1.4), run
            # RKHS fresh; the rest of the cell is reused.
            rkhs_result = _extract_rkhs_result_from_cell(existing_for_cell)
            if rkhs_result is None:
                ui.msg("[resume] no cached RKHS -- fitting kernel grid")
                try:
                    rkhs_result = _run_rkhs_on_cell(q42)
                except Exception as e:
                    import traceback as _tb_r
                    tb_str = _tb_r.format_exc()
                    rkhs_result = {'error': f'{type(e).__name__}: {e}'}
                    ui.err(f"RKHS TRACEBACK for {cell_key}:")
                    for _ln in tb_str.splitlines():
                        ui.err(_ln)
        else:
            # v0.80.0.41: emit two complete lines per cell instead of one
            # split with end=''. The Flask dashboard log filter parses by
            # line, and unterminated buffered lines disappear from the
            # visible log even when flushed. Two terminated lines always
            # show up.
            ui.section(f"[{idx}/{len(q42_files)}] {cell_key} (T={temp})")
            try:
                rf_result = _run_rf_on_cell(q42)
            except Exception as e:
                import traceback as _tb
                tb_str = _tb.format_exc()
                rf_result = {'error': f'{type(e).__name__}: {e}'}
                # v0.80.0.41: log full traceback to stderr so it survives
                # whatever dashboard filtering is happening on stdout
                ui.err(f"TRACEBACK for {cell_key}:")
                for _ln in tb_str.splitlines():
                    ui.err(_ln)
            # v0.81.1.4: RKHS fit (independent of RF -- failure here is
            # logged as 'rkhs_kernel_grid.error' but doesn't fail the
            # cell). RKHS placeholder swap means the apparatus consumer
            # gets real kernel-ridge shares once this completes.
            try:
                rkhs_result = _run_rkhs_on_cell(q42)
            except Exception as e:
                import traceback as _tb_r
                tb_str = _tb_r.format_exc()
                rkhs_result = {'error': f'{type(e).__name__}: {e}'}
                ui.err(f"RKHS TRACEBACK for {cell_key}:")
                for _ln in tb_str.splitlines():
                    ui.err(_ln)

        if 'error' in rf_result:
            ui.err(f"[{idx}/{len(q42_files)}] {cell_key} FAIL: {rf_result['error']}")
            cells[cell_key] = {'error': rf_result['error']}
            n_failed += 1
            # v0.80.0.49: incremental write so a subsequent kill
            # doesn't lose the error record either.
            try:
                _write_q57_incremental(args.out, cells, existing_metadata)
            except Exception as _e:
                ui.warn(f"  ! incremental save failed (run continues): {_e}")
            continue

        try:
            cell_entry = _build_cell_entry(
                family, size, variant, cond, temp,
                rf_result, canon_row, ridge_partition, mlp_partition,
                partition_source,
                rkhs_result=rkhs_result,
            )
        except Exception as e:
            import traceback as _tb
            tb_str = _tb.format_exc()
            ui.err(f"[{idx}/{len(q42_files)}] {cell_key} BUILD FAIL: "
                   f"{type(e).__name__}: {e}")
            for _ln in tb_str.splitlines():
                ui.err(_ln)
            cells[cell_key] = {'error': f'build_cell_entry: {type(e).__name__}: {e}'}
            n_failed += 1
            try:
                _write_q57_incremental(args.out, cells, existing_metadata)
            except Exception as _e:
                ui.warn(f"  ! incremental save failed (run continues): {_e}")
            continue
        cells[cell_key] = cell_entry

        dt = time.time() - t_cell
        sc = cell_entry.get('sign_check', {})
        mlp_pat = sc.get('mlp_vs_ridge_signs_opposite')
        rf_pat = sc.get('rf_vs_ridge_signs_opposite')
        if rf_cached:
            ui.ok(f"[{idx}/{len(q42_files)}] {cell_key} resumed ({dt:.1f}s)  "
                  f"mlp_pattern={mlp_pat}  rf_pattern={rf_pat}")
            n_resumed += 1
        else:
            ui.ok(f"[{idx}/{len(q42_files)}] {cell_key} done ({dt:.1f}s)  "
                  f"mlp_pattern={mlp_pat}  rf_pattern={rf_pat}")
            n_done += 1

        # v0.80.0.49: incremental save after each cell. Atomic via
        # tempfile + os.replace; a kill during write doesn't corrupt
        # the file. Cost: a few KB write per cell, microseconds.
        try:
            _write_q57_incremental(args.out, cells, existing_metadata)
        except Exception as _e:
            ui.warn(f"  ! incremental save failed (run continues): {_e}")

    # Cross-cell aggregates
    aggregates = _build_cross_cell_aggregates(cells)

    # Try to get git commit
    git_commit = "unknown"
    try:
        import subprocess as _sp
        _proc = _sp.run(['git', 'rev-parse', 'HEAD'],
                        cwd=ROOT, capture_output=True, text=True, timeout=2)
        if _proc.returncode == 0:
            git_commit = _proc.stdout.strip()[:12]
    except Exception:
        pass

    # Assemble output
    out = {
        'function_classes': ['ridge', 'mlp', 'rf', 'rkhs_median'],
        'cells': cells,
        'cross_cell_aggregates': aggregates,
        'metadata': {
            'n_permutations': N_PERMUTATIONS,
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE,
            'rf_hyperparameters': RF_HYPERS,
            'mlp_hyperparameters': dict(MLP_HYPERS, hidden_layer_sizes=list(MLP_HYPERS['hidden_layer_sizes'])),
            'ridge_alpha': RIDGE_ALPHA,
            # v0.81.1.4: RKHS hyperparameter spec (Matérn ν grid + RBF
            # length-scale grid). RKHS_median is the per-channel median
            # share across the grid; range under rkhs_kernel_grid block
            # in each cell shows the per-channel min/max.
            'rkhs_alpha': RKHS_ALPHA,
            'rkhs_kernels': RKHS_KERNELS,
            'rkhs_n_subsample': RKHS_N_SUBSAMPLE,
            'rkhs_n_permutations_per_kernel': 50,  # capped vs N_PERMUTATIONS
            'activation_protocol_note': (
                "MLP activation: tanh (canon protocol). Writerbot's original "
                "function-class sensitivity handoff erroneously specified ReLU "
                "(sklearn default). Canon wins; the v0_12 paper's §4.2 stated "
                "'ReLU' which is being corrected to 'tanh' in v0_13 independently "
                "of this run's outcome. The MLP values pulled from "
                "channel_marginal_nonlinearity.csv are canon-tanh values."
            ),
            'held_out_definition': (
                "joint_R2_held_out is the test-set R² from the canon 80/20 split "
                "with random_state=42. NOT the H28-style cross-source held-out "
                "validation (which is a separate analysis covered in §5.3.3)."
            ),
            'partition_source_note': (
                "Ridge and MLP partition fractions (E/C/R) are pulled from "
                "results.json's per-cell measurements: rhat_ridge_insample "
                "and rhat_mlp_pooled_mean (canonical paper §5/§6 fields, "
                "matching paper 1 v0_13 §4.2). Pre-0.81.1.3 this run "
                "incorrectly pulled rhat_ridge_heldout (the H1 cross-source "
                "quantity) and rhat_mlp_reference (single 256-arch fit), "
                "producing sign-flipped R̂_MLP−R̂_Ridge gaps on Gemma 9B Q4 "
                "and Gemma 2B FP16 vs the paper's published numbers. "
                "Each cell's partition_source field documents pulled-vs-"
                "computed status. RF and RKHS_median partitions are "
                "computed in this run."
            ),
            'rf_joint_R2_train_available_in': "cells.<key>.permutation_shares.rf",
            'canon_joint_R2_status': (
                "joint_R2_held_out for ridge and mlp is null in this run because "
                "Q0057 fits Ridge/MLP for permutation-importance partitioning, "
                "not for joint-fit R² on a held-out test set. As of v0.82.0.23, "
                "the canonical per-estimator joint R² (train + held-out) is "
                "surfaced under cells.<key>.measurements.estimator_joint_R2 "
                "by Phase 12 -- fits Ridge/MLP/RF/RKHS at the canonical "
                "partition-B operating point and emits R² values. See "
                "results_schema.py docstring for the relationship between "
                "estimator_joint_R2 and linearity_max_gap; the two are "
                "distinct cuts of related computation, both independently "
                "meaningful for §5 recoverability-plane positioning."
            ),
            'git_commit': git_commit,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'iota_version': '0.82.0.23',
            # v0.80.0.45: audit trail for protocol decisions made
            # during the run that weren't anticipated in the original
            # handoff or bridge orientation. Future-you reading this
            # JSON in six months should be able to reconstruct what
            # happened without needing the chat transcript.
            'protocol_decisions_during_run': [
                {
                    'issue': (
                        "RF multi-output cost on cells with pool_dim > 64"
                    ),
                    'diagnosed_in_version': '0.80.0.45',
                    'decision': (
                        "PCA target to 64 dims for RF only on cells "
                        "where pool_dim > 64; Ridge and MLP retain "
                        "canon native target dim (pulled from "
                        "channel_marginal_nonlinearity.csv, not re-fit)."
                    ),
                    'rationale': (
                        "Preserves within-cell apples-to-apples for "
                        "the comparison-relevant axis (sign of "
                        "asymmetry, sign of R-hat gap). Ridge/MLP "
                        "target shapes documented in target_dimensionality "
                        "block per cell. Honestly disclosable: RF on "
                        "2B cells operates on PCA-reduced target (64 "
                        "dims) while Ridge/MLP retain native (1024 "
                        "dims). The §5.4 mechanism prediction is "
                        "robust to this difference if the PCA "
                        "reduction preserves the dominant variance "
                        "modes -- testable per-cell via the recorded "
                        "explained_variance_ratio."
                    ),
                    'symptom_before_fix': (
                        "Run 0057 hung 8+ minutes on first cell "
                        "(gemma_2b_q4_abliterated_t00, pool_dim=1024) "
                        "during univariate-E RF fit; LLaMA 8B and "
                        "Gemma 9B cells (pool_dim=64) would have "
                        "completed normally but were never reached."
                    ),
                },
            ],
        },
    }

    # Write with fsync (durability per 0.80.0.37 standard)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    elapsed = time.time() - t0
    ui.blank()
    ui.ok(f"Done. {n_done} fresh, {n_resumed} resumed, {n_failed} failed. "
          f"Elapsed: {elapsed:.0f}s")
    ui.ok(f"Wrote {args.out}")
    ui.section("Cross-cell summary")
    for k, v in aggregates.items():
        ui.msg(f"{k}: {v}")

    # v0.80.0.50: count both fresh and resumed cells. Pre-fix this
    # tripped whenever every cell resumed from cache (the natural
    # post-first-run state) and exited 1 even though Q0057 was
    # fully populated on disk via the per-cell incremental writes.
    if (n_done + n_resumed) == 0:
        ui.err("no cells produced output. Run 0058 (apparatus) and Run 0059 (paper assembly) will refuse to fire.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
