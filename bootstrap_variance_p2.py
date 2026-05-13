"""
bootstrap_variance_p2.py -- Phase 10 refit-loop module.

v0.82.0.23 (paper-2 measurement layer).

Per cell: N bootstrap resamples of the cell's full data with replacement
(size n). For each resample:

  1. Paired bootstrap indices and paired 80/20 internal train/test split
     shared across all four function classes (Ridge, MLP, RKHS, RF).
     Independent draws per class would break the variance comparison --
     RNG seed is set per resample so all four classes see identical
     samples.

  2. Fit each class on the bootstrap train set, permute one block at a
     time on the bootstrap test set, recompute R² drop, renormalize
     drops to a (E, C, R) simplex point.

  3. Apply the I-projection apparatus (closed form solver) using the
     four class shares + the cell's fixed kNN anchor + the
     globally-chosen lambda_default. Yields q*.

After all N resamples: per-channel variance for each class and for
anchored, computed across the resamples that completed successfully.

Phase 7 anchor and lambda_default are loaded once per cell and held
fixed across all bootstrap iterations -- only the four function-class
shares vary per resample.

Scope limitations (disclosed in CHANGELOG entry 0.82.0.23):

  - RKHS inside the bootstrap loop uses ONE kernel (Matérn ν=1.5).
    Outside the bootstrap, headline RKHS_median uses all three kernels
    (matern_nu1.5 + RBF×1.0 + RBF×2.0). Bootstrap variance for the
    RKHS column therefore reflects single-kernel sampling variance,
    not kernel-grid variance. Wall-time decision per the codebot
    handoff brief.

  - Permutation count inside the bootstrap loop is 30 per channel for
    Ridge / MLP / RF (vs 200 in canonical Q0057) and 10 for RKHS
    (matches canonical Q0057 RKHS). Test-set permutation rows
    subsampled to 1500 (vs full ~3500 canonical). The reduction is
    a wall-time tradeoff: per-resample share precision drops by
    sqrt(200/30) ≈ 2.6× for Ridge/MLP/RF. At v0.82.0.14's
    n_bootstrap=10 the across-resample SE on the variance estimate
    is large in absolute terms (~47% of the variance itself) -- the
    within-resample permutation noise is no longer negligibly small
    by comparison, but is still subdominant. Reported variances
    are point estimates; bootstrap-of-bootstrap CIs are out of
    scope.

  - Phase 6 partition mismatch: Phase 6 currently surfaces Q0057's
    full-data class shares as a transitional input
    (partition_source='q57_full_data_pending_partition_b_refit').
    Phase 10's bootstrap does its own partition-B refits internally
    on each resample, so the bootstrap is internally consistent.
    Headline q* (Phase 8, full-data shares) and bootstrap variance
    (Phase 10, partition-B refits across resamples) don't share a
    partition convention -- methods-section caveat. Reconciled when
    the Phase 6 partition-B refit ships.
"""

import numpy as np
import time
import warnings


# ── Imports from function_class_p2 (single source of truth) ─────────
#
# v0.82.0.23: hyperparameters, sample-size caps, and per-class
# fit/permute helpers moved to function_class_p2.py so Phase 6
# (canonical partition-B refit) and Phase 10 (this module's bootstrap
# loop) share one implementation. Re-exported here as module-level
# names for backward compatibility -- any caller that imports
# bootstrap_variance_p2.RIDGE_ALPHA, ._ridge_partition, etc. continues
# to work unchanged.

import function_class_p2 as fcp2

# Constants
N_PERM_BOOTSTRAP           = fcp2.N_PERM_BOOTSTRAP
N_PERM_RKHS_BOOTSTRAP      = fcp2.N_PERM_RKHS_BOOTSTRAP
N_TEST_SUBSAMPLE_BOOTSTRAP = fcp2.N_TEST_SUBSAMPLE_BOOTSTRAP
N_BOOTSTRAP_ROWS_CAP       = fcp2.N_BOOTSTRAP_ROWS_CAP
RKHS_N_TRAIN_SUBSAMPLE     = fcp2.RKHS_N_TRAIN_SUBSAMPLE
RIDGE_ALPHA                = fcp2.RIDGE_ALPHA
MLP_HYPERS                 = fcp2.MLP_HYPERS
RF_HYPERS                  = fcp2.RF_HYPERS
RKHS_ALPHA                 = fcp2.RKHS_ALPHA
RKHS_KERNEL_NU             = fcp2.RKHS_KERNEL_NU

# Helpers
load_cell_data_from_qcache = fcp2.load_cell_data_from_qcache
_renorm_drops_to_simplex   = fcp2._renorm_drops_to_simplex
_ridge_partition           = fcp2._ridge_partition
_mlp_partition             = fcp2._mlp_partition
_rf_partition              = fcp2._rf_partition
_rkhs_partition_matern     = fcp2._rkhs_partition_matern


# ── Defaults specific to bootstrap loop ─────────────────────────────

N_BOOTSTRAP_DEFAULT = 200


def _apply_apparatus(class_shares_4x3, p_anchor, lam):
    """Closed-form I-projection. Returns q* as 3-vector (E, C, R)."""
    import bayesian_solver as ls
    q = ls.solve_iprojection_closed_form(class_shares_4x3, p_anchor, lam=lam)
    return np.asarray(q, dtype=float).flatten()


# ── One bootstrap resample (paired across the four classes) ─────────

def _one_resample(cell_data, p_anchor, lam, resample_seed,
                    target_pca_components=None, ui_msg=None):
    """Run one bootstrap resample. Returns dict with five (E, C, R)
    triples (ridge / mlp / rkhs / rf / anchored) plus baseline R²
    diagnostics, or {'error': ...} on failure.

    target_pca_components: int or None. If int, target-PCA is fit on
    the resampled training y and applied to RF and RKHS only (not Ridge
    / MLP -- they handle multi-output natively without n_outputs cost).
    Mirrors the canonical Q0057 target-PCA decision rule.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xe = cell_data['Xe']; Xc = cell_data['Xc']
    Xs = cell_data['Xs']; Xp = cell_data['Xp']
    y  = cell_data['y']

    n = Xe.shape[0]
    rng = np.random.RandomState(resample_seed)

    # ── Bootstrap indices: paired across all four classes ───────────
    # v0.82.0.23: cap resample size at N_BOOTSTRAP_ROWS_CAP to control
    # per-resample wall time on large cells. When n <= cap, this is a
    # standard full-size bootstrap (size n with replacement). When n >
    # cap, draw cap rows with replacement instead -- a subsampled
    # bootstrap.
    boot_size = min(n, N_BOOTSTRAP_ROWS_CAP)
    boot_idx = rng.randint(0, n, size=boot_size)
    Xe_b = Xe[boot_idx]
    Xc_b = Xc[boot_idx]
    Xs_b = Xs[boot_idx]
    Xp_b = Xp[boot_idx]
    y_b  = y[boot_idx]

    # Joint feature matrix [Xe, Xc, Xs, Xp] -- layout must match the
    # permutation slices used by all four classes.
    full_X = np.hstack([Xe_b, Xc_b, Xs_b, Xp_b])

    # ── Internal 80/20 split: paired across all four classes ────────
    # Same random_state across classes. The split is on the resampled
    # data, so each resample sees a fresh 80/20 partition of its own
    # bootstrap draw.
    split_seed = int(rng.randint(0, 2**31 - 1))
    Xj_tr, Xj_te, y_tr_raw, y_te_raw = train_test_split(
        full_X, y_b, test_size=0.2, random_state=split_seed)

    # ── Target-PCA for RF / RKHS (only on cells with native pool > 64) ─
    target_pca = None
    if target_pca_components is not None:
        target_pca = PCA(n_components=target_pca_components,
                         random_state=42).fit(y_tr_raw)
        y_tr_pca = target_pca.transform(y_tr_raw)
        y_te_pca = target_pca.transform(y_te_raw)
    else:
        y_tr_pca = y_tr_raw
        y_te_pca = y_te_raw

    # Standardize X (joint) and y (both native + PCA) -- separate scalers
    # because Ridge/MLP use native y and RF/RKHS use PCA y.
    sxJ = StandardScaler().fit(Xj_tr)
    Xj_tr_s = sxJ.transform(Xj_tr)
    Xj_te_s = sxJ.transform(Xj_te)

    sy_native = StandardScaler().fit(y_tr_raw)
    y_tr_native_s = sy_native.transform(y_tr_raw)
    y_te_native_s = sy_native.transform(y_te_raw)

    sy_pca = StandardScaler().fit(y_tr_pca)
    y_tr_pca_s = sy_pca.transform(y_tr_pca)
    y_te_pca_s = sy_pca.transform(y_te_pca)

    # ── Test-set subsampling for permutation loop (RF/MLP/RKHS dominant) ─
    n_test = Xj_te_s.shape[0]
    if n_test > N_TEST_SUBSAMPLE_BOOTSTRAP:
        sub_idx = rng.choice(n_test, size=N_TEST_SUBSAMPLE_BOOTSTRAP,
                              replace=False)
        Xj_te_perm = Xj_te_s[sub_idx]
        y_te_native_perm = y_te_native_s[sub_idx]
        y_te_pca_perm = y_te_pca_s[sub_idx]
    else:
        Xj_te_perm = Xj_te_s
        y_te_native_perm = y_te_native_s
        y_te_pca_perm = y_te_pca_s

    # Slices into the joint feature matrix [Xe, Xc, Xs, Xp]
    d_e = Xe.shape[1]
    d_c = Xc.shape[1]
    d_s = Xs.shape[1]
    slice_E = slice(0,           d_e)
    slice_C = slice(d_e,         d_e + d_c)
    slice_R = slice(d_e + d_c,   d_e + d_c + d_s)

    try:
        # Each class gets a fresh per-class permutation RNG seeded
        # off the resample seed -- drops are independent within
        # resample (within-class ordering of permutations doesn't
        # affect the mean drop at the precision we care about).
        ridge_share = _ridge_partition(
            Xj_tr_s, Xj_te_perm, y_tr_native_s, y_te_native_perm,
            slice_E, slice_C, slice_R, N_PERM_BOOTSTRAP,
            np.random.RandomState(resample_seed + 1))
        mlp_share = _mlp_partition(
            Xj_tr_s, Xj_te_perm, y_tr_native_s, y_te_native_perm,
            slice_E, slice_C, slice_R, N_PERM_BOOTSTRAP,
            np.random.RandomState(resample_seed + 2))

        # RF / RKHS use PCA-transformed y on cells with target PCA.
        # When no PCA, y_tr_pca_s == y_tr_native_s by construction
        # (same scaler input).
        rf_share = _rf_partition(
            Xj_tr_s, Xj_te_perm, y_tr_pca_s, y_te_pca_perm,
            slice_E, slice_C, slice_R, N_PERM_BOOTSTRAP,
            np.random.RandomState(resample_seed + 3))

        # RKHS: subsample TRAIN (kernel matrix size cap) + use median
        # pairwise distance as the matern length scale.
        n_tr_full = Xj_tr_s.shape[0]
        if n_tr_full > RKHS_N_TRAIN_SUBSAMPLE:
            rkhs_tr_idx = rng.choice(n_tr_full, size=RKHS_N_TRAIN_SUBSAMPLE,
                                       replace=False)
            X_rkhs_tr = Xj_tr_s[rkhs_tr_idx]
            y_rkhs_tr = y_tr_pca_s[rkhs_tr_idx]
        else:
            X_rkhs_tr = Xj_tr_s
            y_rkhs_tr = y_tr_pca_s
        # Median pairwise distance on a 1000-row subsample of train
        ms_idx = rng.choice(X_rkhs_tr.shape[0],
                              size=min(1000, X_rkhs_tr.shape[0]),
                              replace=False)
        pd_mat = euclidean_distances(X_rkhs_tr[ms_idx])
        offdiag = pd_mat[np.triu_indices_from(pd_mat, k=1)]
        median_pw = float(np.median(offdiag))
        if median_pw <= 0:
            median_pw = 1.0
        rkhs_share = _rkhs_partition_matern(
            X_rkhs_tr, Xj_te_perm, y_rkhs_tr, y_te_pca_perm,
            slice_E, slice_C, slice_R, median_pw,
            N_PERM_RKHS_BOOTSTRAP,
            np.random.RandomState(resample_seed + 4))
    except Exception as e:
        return {'error': f'fit/permute failed: {type(e).__name__}: {e}'}

    class_shares = np.array([
        list(ridge_share), list(mlp_share),
        list(rf_share),    list(rkhs_share),
    ])

    try:
        q_anchored = _apply_apparatus(class_shares, p_anchor, lam)
    except Exception as e:
        return {'error': f'apparatus failed: {type(e).__name__}: {e}'}

    return {
        'ridge':    list(ridge_share),
        'mlp':      list(mlp_share),
        'rf':       list(rf_share),
        'rkhs':     list(rkhs_share),
        'anchored': [float(q_anchored[0]), float(q_anchored[1]),
                      float(q_anchored[2])],
    }


# ── Per-cell driver ──────────────────────────────────────────────────

def bootstrap_one_cell(cell_data, p_anchor, lam,
                          n_bootstrap=N_BOOTSTRAP_DEFAULT,
                          seed_base=0,
                          target_pca_components=None,
                          ui_msg=None,
                          progress_every=25):
    """Run N bootstraps on one cell. Returns dict with per-channel
    variance for each class and for anchored.

    Args:
      cell_data: dict from load_cell_data_from_qcache (or test fixture).
        Must have keys Xe, Xc, Xs, Xp, y.
      p_anchor: 3-vector (E, C, R) anchor in simplex coords.
      lam: float -- apparatus λ (lambda_default chosen globally).
      n_bootstrap: number of resamples (default 200).
      seed_base: base seed; resample i uses seed_base + 100*i.
      target_pca_components: int or None -- if int, RF/RKHS use a
        PCA-reduced target (fit per resample on bootstrap train y).
      ui_msg: optional callable for progress logging.
      progress_every: log every Nth resample.

    Returns:
      dict with:
        n_bootstrap_requested, n_completed, n_failed,
        Ridge_var_per_channel, MLP_var_per_channel,
        RKHS_median_var_per_channel, RF_var_per_channel,
        anchored_var_per_channel,
        wall_time_seconds, per_resample_shares (list of dicts).
    """
    t0 = time.time()
    shares_acc = {
        'ridge':    [], 'mlp':  [],
        'rf':       [], 'rkhs': [],
        'anchored': [],
    }
    n_failed = 0
    failures = []

    for i in range(n_bootstrap):
        seed = int(seed_base + 100 * i + 1)
        result = _one_resample(cell_data, p_anchor, lam, seed,
                                  target_pca_components=target_pca_components,
                                  ui_msg=ui_msg)
        if 'error' in result:
            n_failed += 1
            if len(failures) < 5:
                failures.append({'i': i, 'seed': seed, 'error': result['error']})
            continue
        for k in ('ridge', 'mlp', 'rf', 'rkhs', 'anchored'):
            shares_acc[k].append(result[k])
        if ui_msg and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            ui_msg(f"    [bootstrap {i+1}/{n_bootstrap}] "
                    f"elapsed={elapsed:.0f}s, failed={n_failed}")

    wall_time = time.time() - t0

    def _per_channel_var(class_name):
        rows = shares_acc[class_name]
        if len(rows) < 2:
            return {'E': 0.0, 'C': 0.0, 'R': 0.0}
        arr = np.asarray(rows, dtype=float)
        return {
            'E': float(np.var(arr[:, 0], ddof=1)),
            'C': float(np.var(arr[:, 1], ddof=1)),
            'R': float(np.var(arr[:, 2], ddof=1)),
        }

    return {
        'n_bootstrap_requested': int(n_bootstrap),
        'n_completed':            int(n_bootstrap - n_failed),
        'n_failed':               int(n_failed),
        'Ridge_var_per_channel':         _per_channel_var('ridge'),
        'MLP_var_per_channel':           _per_channel_var('mlp'),
        'RKHS_median_var_per_channel':   _per_channel_var('rkhs'),
        'RF_var_per_channel':            _per_channel_var('rf'),
        'anchored_var_per_channel':      _per_channel_var('anchored'),
        'wall_time_seconds':      float(wall_time),
        'failures_sample':        failures,
        'per_resample_shares':    shares_acc,
    }
