"""
function_class_p2.py -- canonical four-class fit/permute helpers.

v0.82.0.23 (paper-2 measurement layer).

Single source of truth for:

  - Hyperparameters: RIDGE_ALPHA, MLP_HYPERS, RF_HYPERS, RKHS_ALPHA,
    RKHS_KERNEL_NU.
  - Permutation protocol: N_PERM_BOOTSTRAP, N_PERM_RKHS_BOOTSTRAP.
  - Sample-size caps for in-bootstrap loops:
    N_BOOTSTRAP_ROWS_CAP, N_TEST_SUBSAMPLE_BOOTSTRAP,
    RKHS_N_TRAIN_SUBSAMPLE.
  - Per-class fit/permute helpers: _ridge_partition, _mlp_partition,
    _rf_partition, _rkhs_partition_matern.
  - Renormalization: _renorm_drops_to_simplex.
  - Cell-data loader: load_cell_data_from_qcache.

Public entry points:

  - compute_partition_b_shares(cell_data, target_pca_components, seed=42)
    -- single-shot full-data partition-B refit; returns the four-class
    shares dict for canonical Phase 6 use.

  - The four _*_partition helpers are exported so
    bootstrap_variance_p2._one_resample can keep using them inside the
    bootstrap loop without code duplication.

Architectural rationale (v0.82.0.23 ship):

Pre-v0.82.0.23, the fit/permute logic lived inside
bootstrap_variance_p2.py and was only callable in bootstrap-resample
mode (with N_BOOTSTRAP_ROWS_CAP and N_TEST_SUBSAMPLE_BOOTSTRAP applied).
Phase 6 was reading Q0057 full-data shares as a transitional path,
which produced a single-shot vs bootstrap disagreement of up to 60×
on the hull-distance diagnostic for cells in the q4_t00 regime.

The V0 protocol confirmed (max |Δ| = 0.125 on q4_t00, Ridge ΔC =
−0.125 dominant) that the disagreement traced to partition mismatch
between Q0057 (full-data, in-sample) shares and partition-B (held-out,
single-shot) refits. Phase 6 now uses partition-B refit shares as
canonical via compute_partition_b_shares; bootstrap_variance_p2 still
imports the per-class helpers for its in-loop refits. Both phases
share hyperparameters and protocol -- single source of truth.

The MLP partial swing on q4_t00 (ΔC = −0.044, intermediate between
Ridge's 12 and RF's 0) is consistent with the C-coordinate being
identification-fragile under linear-leaning methods on full data,
with held-out evaluation correctly disambiguating. Documented as a
methodological observation in §9; cross-validated Ridge stability is
named as a §10 follow-up.
"""

import os
import re
import glob
import warnings

import numpy as np


# ── Sample-size caps (in-bootstrap-loop only) ───────────────────────

N_PERM_BOOTSTRAP = 30
N_PERM_RKHS_BOOTSTRAP = 10
N_TEST_SUBSAMPLE_BOOTSTRAP = 1500

# v0.82.0.14: cap bootstrap resample size to control per-resample
# wall time on large cells. Profile of gemma_2b_q4_abliterated_t00
# (n_rows=21,600, pool_dim=1024) clocked ~17.7 min per resample at
# the original (no cap) setting. Capping resample size to 5000 cuts
# the dominant fit-and-permute costs roughly 4x linearly while still
# bootstrapping a sample large vs the RF/MLP effective parameter
# counts. When the cell's n_rows is below the cap, no truncation
# occurs (full-resample bootstrap as before).
N_BOOTSTRAP_ROWS_CAP = 5000

# v0.82.0.23: full-data partition-B refits do NOT apply
# N_TEST_SUBSAMPLE_BOOTSTRAP / N_BOOTSTRAP_ROWS_CAP. Headline canonical
# shares use full data per statsbot's headline-precision argument:
# "headline shares cap at 5000" invites a reviewer question we can't
# cleanly answer. The caps remain in place for the in-bootstrap-loop
# refits, where they're an internally-consistent statistical procedure.

# RKHS train subsample applies in BOTH partition-B (canonical Q0057
# protocol) and bootstrap modes -- kernel-matrix size is the gating
# constraint regardless of phase.
RKHS_N_TRAIN_SUBSAMPLE = 2500


# ── Hyperparameters (match canonical Q0057 / Q0046 protocols) ───────

RIDGE_ALPHA = 0.01

MLP_HYPERS = {
    'hidden_layer_sizes':  (256,),
    'activation':          'tanh',
    'early_stopping':      True,
    'validation_fraction': 0.1,
    'max_iter':            500,
    'random_state':        42,
    'tol':                 1e-5,
}

RF_HYPERS = {
    'n_estimators':     100,
    'max_depth':        None,
    'min_samples_leaf': 5,
    'random_state':     42,
    'n_jobs':           4,
}

RKHS_ALPHA = 1e-3
RKHS_KERNEL_NU = 1.5  # matern ν=1.5 (single-kernel inside partition-B
                      # and inside bootstrap loop; outside-loop headline
                      # RKHS_median uses 3-kernel grid in Q0057)

# Canonical 80/20 split seed for partition-B headline refit. Same seed
# across all cells means cell-to-cell comparisons are partition-coherent.
PARTITION_B_SEED = 42


# ── Cell data loader ────────────────────────────────────────────────

def load_cell_data_from_qcache(hidden_dir):
    """Load (Xe, Xc, Xs, Xp, y) from the cell's _qcache_ct npz file.

    Mirrors the cache-load path in
    run_function_class_sensitivity._run_rf_on_cell -- uses direct numpy
    load (bypasses analysis._load_quadruplets freshness check, since
    Phase 6/10 explicitly want to share data with the canon Ridge+MLP
    in results.json regardless of E_base regenerations elsewhere).

    Returns:
      dict with Xe, Xc, Xs, Xp, y, pool_dim, n_rows on success.
      dict with 'error' key on failure.
    """
    cache_glob = os.path.join(hidden_dir, '_qcache_ct_d*_r*.npz')
    cache_files = sorted(glob.glob(cache_glob))
    if not cache_files:
        return {'error': f'no qcache in {hidden_dir}'}

    cache_path = cache_files[0]
    m = re.match(r'_qcache_ct_d(\d+)_r(\d+)-(\d+)\.npz$',
                 os.path.basename(cache_path))
    if not m:
        return {'error': f'qcache filename unparseable: {cache_path}'}
    pool_dim = int(m.group(1))

    try:
        d = np.load(cache_path, allow_pickle=False)
        et = np.array(d['e_t'])
        ct = np.array(d['c_t'])
        sp = np.array(d['s_prev'])
        sn = np.array(d['s_next'])
        hr = np.array(d['has_real_ct'])
        pt = np.array(d['prompt_tokens'])
        d.close()
    except Exception as e:
        return {'error': f'cache load failed: {type(e).__name__}: {e}'}

    mask = hr.astype(bool)
    if not mask.any():
        return {'error': f'no rows with real C_t (raw n={len(et)})'}

    Xe = et[mask].astype(np.float32)
    Xc = ct[mask].astype(np.float32)
    Xs = sp[mask].astype(np.float32)
    Xp = pt[mask].reshape(-1, 1).astype(np.float32)
    Xp[np.isnan(Xp)] = 0.0
    y  = sn[mask].astype(np.float32)

    return {
        'Xe': Xe, 'Xc': Xc, 'Xs': Xs, 'Xp': Xp, 'y': y,
        'pool_dim': pool_dim, 'n_rows': int(Xe.shape[0]),
    }


# ── Renormalization to simplex ──────────────────────────────────────

def _renorm_drops_to_simplex(drops_E, drops_C, drops_R):
    """Convert mean R² drops on (E, C, R) to a simplex point summing to 1.

    Matches the renormalization in canonical Q0057 / analysis.py:
    clamp negatives to zero, divide by sum. Returns (E, C, R) tuple.
    Falls back to (1/3, 1/3, 1/3) if all drops are non-positive.
    """
    e = max(0.0, float(drops_E))
    c = max(0.0, float(drops_C))
    r = max(0.0, float(drops_R))
    s = e + c + r
    if s <= 0:
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return (e / s, c / s, r / s)


# ── Per-class fit/permute helpers ───────────────────────────────────

def _ridge_partition(Xj_tr_s, Xj_te_s, y_tr_s, y_te_s,
                     slice_E, slice_C, slice_R, n_perm, rng):
    """Fit Ridge on train, permute test blocks, return (E, C, R) shares."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    mdl = Ridge(alpha=RIDGE_ALPHA).fit(Xj_tr_s, y_tr_s)
    base_r2 = float(r2_score(y_te_s, mdl.predict(Xj_te_s),
                              multioutput='variance_weighted'))
    drops = {'E': [], 'C': [], 'R': []}
    for label, sl in (('E', slice_E), ('C', slice_C), ('R', slice_R)):
        for _ in range(n_perm):
            X_perm = Xj_te_s.copy()
            idx = rng.permutation(X_perm.shape[0])
            X_perm[:, sl] = X_perm[idx][:, sl]
            r2_p = float(r2_score(y_te_s, mdl.predict(X_perm),
                                    multioutput='variance_weighted'))
            drops[label].append(base_r2 - r2_p)
    return _renorm_drops_to_simplex(np.mean(drops['E']),
                                      np.mean(drops['C']),
                                      np.mean(drops['R']))


def _mlp_partition(Xj_tr_s, Xj_te_s, y_tr_s, y_te_s,
                   slice_E, slice_C, slice_R, n_perm, rng):
    """Fit MLP on train, permute test blocks, return (E, C, R) shares."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mdl = MLPRegressor(**MLP_HYPERS).fit(Xj_tr_s, y_tr_s)
    base_r2 = float(r2_score(y_te_s, mdl.predict(Xj_te_s),
                              multioutput='variance_weighted'))
    drops = {'E': [], 'C': [], 'R': []}
    for label, sl in (('E', slice_E), ('C', slice_C), ('R', slice_R)):
        for _ in range(n_perm):
            X_perm = Xj_te_s.copy()
            idx = rng.permutation(X_perm.shape[0])
            X_perm[:, sl] = X_perm[idx][:, sl]
            r2_p = float(r2_score(y_te_s, mdl.predict(X_perm),
                                    multioutput='variance_weighted'))
            drops[label].append(base_r2 - r2_p)
    return _renorm_drops_to_simplex(np.mean(drops['E']),
                                      np.mean(drops['C']),
                                      np.mean(drops['R']))


def _rf_partition(Xj_tr_s, Xj_te_s, y_tr_s, y_te_s,
                  slice_E, slice_C, slice_R, n_perm, rng):
    """Fit RF on train, permute test blocks, return (E, C, R) shares.

    Matches RF_HYPERS from canonical Q0057. Caller is responsible for
    applying target-PCA upstream when pool_dim > 64 (multi-output RF
    cost scales linearly in n_outputs).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mdl = RandomForestRegressor(**RF_HYPERS).fit(Xj_tr_s, y_tr_s)
    base_r2 = float(r2_score(y_te_s, mdl.predict(Xj_te_s),
                              multioutput='variance_weighted'))
    drops = {'E': [], 'C': [], 'R': []}
    for label, sl in (('E', slice_E), ('C', slice_C), ('R', slice_R)):
        for _ in range(n_perm):
            X_perm = Xj_te_s.copy()
            idx = rng.permutation(X_perm.shape[0])
            X_perm[:, sl] = X_perm[idx][:, sl]
            r2_p = float(r2_score(y_te_s, mdl.predict(X_perm),
                                    multioutput='variance_weighted'))
            drops[label].append(base_r2 - r2_p)
    return _renorm_drops_to_simplex(np.mean(drops['E']),
                                      np.mean(drops['C']),
                                      np.mean(drops['R']))


def _rkhs_partition_matern(Xj_tr_s, Xj_te_s, y_tr_s, y_te_s,
                           slice_E, slice_C, slice_R,
                           median_pairwise, n_perm, rng):
    """Fit single-kernel matern ν=1.5 RKHS, permute, return (E, C, R) shares.

    Mirrors the precomputed-Gram pattern from canonical
    run_function_class_sensitivity._run_rkhs_on_cell -- passing a
    sklearn kernel callable to KernelRidge triggers per-predict Python
    re-evaluation; precomputing once and using kernel='precomputed'
    keeps everything in numpy/C. Test Gram is rebuilt per permutation
    against the frozen train.
    """
    from sklearn.gaussian_process.kernels import Matern as _Matern
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score

    matern = _Matern(length_scale=median_pairwise, nu=RKHS_KERNEL_NU)
    gram_train = matern(Xj_tr_s, Xj_tr_s)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kr = KernelRidge(alpha=RKHS_ALPHA, kernel='precomputed')
        kr.fit(gram_train, y_tr_s)

    gram_test_train = matern(Xj_te_s, Xj_tr_s)
    base_r2 = float(r2_score(y_te_s, kr.predict(gram_test_train),
                              multioutput='variance_weighted'))

    drops = {'E': [], 'C': [], 'R': []}
    for label, sl in (('E', slice_E), ('C', slice_C), ('R', slice_R)):
        for _ in range(n_perm):
            X_perm = Xj_te_s.copy()
            idx = rng.permutation(X_perm.shape[0])
            X_perm[:, sl] = X_perm[idx][:, sl]
            gram_perm = matern(X_perm, Xj_tr_s)
            r2_p = float(r2_score(y_te_s, kr.predict(gram_perm),
                                    multioutput='variance_weighted'))
            drops[label].append(base_r2 - r2_p)
    return _renorm_drops_to_simplex(np.mean(drops['E']),
                                      np.mean(drops['C']),
                                      np.mean(drops['R']))


# ── Single-shot partition-B refit (Phase 6 canonical entry point) ───

def compute_partition_b_shares(cell_data, target_pca_components=None,
                                  seed=PARTITION_B_SEED, ui_msg=None):
    """Single-shot partition-B refit -- full data, canonical 80/20 split.

    Headline canonical shares for Phase 6. NO bootstrap resampling, NO
    N_BOOTSTRAP_ROWS_CAP, NO N_TEST_SUBSAMPLE_BOOTSTRAP. Reviewer-defensible
    precision regime: full data on the headline, caps only on bootstrap
    diagnostics that are statistical-resampling procedures.

    Args:
      cell_data: dict from load_cell_data_from_qcache.
      target_pca_components: int or None. PCA-target the y for RF / RKHS
        when pool_dim > 64 (Q0057 protocol). Ridge / MLP use native y.
      seed: random_state for the 80/20 split. Default PARTITION_B_SEED
        (=42); all cells use the same seed for partition coherence.
      ui_msg: optional callable for progress logging.

    Returns:
      dict with 'ridge', 'mlp', 'rf', 'rkhs' -- each a 3-tuple
      (E, C, R) summing to 1.

    Hyperparameters and permutation count match the in-bootstrap
    helpers -- so partition-B canonical shares and bootstrap median
    shares are computed under exactly the same protocol, just with
    different sample-size regimes (full vs capped).
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xe = cell_data['Xe']; Xc = cell_data['Xc']
    Xs = cell_data['Xs']; Xp = cell_data['Xp']
    y  = cell_data['y']

    full_X = np.hstack([Xe, Xc, Xs, Xp])

    Xj_tr, Xj_te, y_tr_raw, y_te_raw = train_test_split(
        full_X, y, test_size=0.2, random_state=seed)

    if target_pca_components is not None:
        target_pca = PCA(n_components=target_pca_components,
                         random_state=42).fit(y_tr_raw)
        y_tr_pca = target_pca.transform(y_tr_raw)
        y_te_pca = target_pca.transform(y_te_raw)
    else:
        y_tr_pca = y_tr_raw
        y_te_pca = y_te_raw

    sxJ = StandardScaler().fit(Xj_tr)
    Xj_tr_s = sxJ.transform(Xj_tr)
    Xj_te_s = sxJ.transform(Xj_te)

    sy_native = StandardScaler().fit(y_tr_raw)
    y_tr_native_s = sy_native.transform(y_tr_raw)
    y_te_native_s = sy_native.transform(y_te_raw)

    sy_pca = StandardScaler().fit(y_tr_pca)
    y_tr_pca_s = sy_pca.transform(y_tr_pca)
    y_te_pca_s = sy_pca.transform(y_te_pca)

    # Full test set for permutations -- NO N_TEST_SUBSAMPLE_BOOTSTRAP cap
    # in canonical mode.
    Xj_te_perm = Xj_te_s
    y_te_native_perm = y_te_native_s
    y_te_pca_perm = y_te_pca_s

    d_e = Xe.shape[1]
    d_c = Xc.shape[1]
    d_s = Xs.shape[1]
    slice_E = slice(0, d_e)
    slice_C = slice(d_e, d_e + d_c)
    slice_R = slice(d_e + d_c, d_e + d_c + d_s)

    rng = np.random.RandomState(seed)
    if ui_msg: ui_msg("    Ridge fit + permute...")
    ridge_share = _ridge_partition(
        Xj_tr_s, Xj_te_perm, y_tr_native_s, y_te_native_perm,
        slice_E, slice_C, slice_R, N_PERM_BOOTSTRAP,
        np.random.RandomState(seed + 1))

    if ui_msg: ui_msg("    MLP fit + permute...")
    mlp_share = _mlp_partition(
        Xj_tr_s, Xj_te_perm, y_tr_native_s, y_te_native_perm,
        slice_E, slice_C, slice_R, N_PERM_BOOTSTRAP,
        np.random.RandomState(seed + 2))

    if ui_msg: ui_msg("    RF fit + permute...")
    rf_share = _rf_partition(
        Xj_tr_s, Xj_te_perm, y_tr_pca_s, y_te_pca_perm,
        slice_E, slice_C, slice_R, N_PERM_BOOTSTRAP,
        np.random.RandomState(seed + 3))

    n_tr_full = Xj_tr_s.shape[0]
    if n_tr_full > RKHS_N_TRAIN_SUBSAMPLE:
        rkhs_idx = rng.choice(n_tr_full, size=RKHS_N_TRAIN_SUBSAMPLE,
                                replace=False)
        X_rkhs_tr = Xj_tr_s[rkhs_idx]
        y_rkhs_tr = y_tr_pca_s[rkhs_idx]
    else:
        X_rkhs_tr = Xj_tr_s
        y_rkhs_tr = y_tr_pca_s
    ms_idx = rng.choice(X_rkhs_tr.shape[0],
                          size=min(1000, X_rkhs_tr.shape[0]),
                          replace=False)
    pd_mat = euclidean_distances(X_rkhs_tr[ms_idx])
    offdiag = pd_mat[np.triu_indices_from(pd_mat, k=1)]
    median_pw = float(np.median(offdiag))
    if median_pw <= 0:
        median_pw = 1.0
    if ui_msg: ui_msg("    RKHS fit + permute...")
    rkhs_share = _rkhs_partition_matern(
        X_rkhs_tr, Xj_te_perm, y_rkhs_tr, y_te_pca_perm,
        slice_E, slice_C, slice_R, median_pw,
        N_PERM_RKHS_BOOTSTRAP,
        np.random.RandomState(seed + 4))

    return {
        'ridge': ridge_share,
        'mlp':   mlp_share,
        'rf':    rf_share,
        'rkhs':  rkhs_share,
        'rkhs_median_length_scale': float(median_pw),
        'partition_seed':           int(seed),
    }
