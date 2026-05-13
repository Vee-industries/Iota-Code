"""
estimator_joint_R2_p2.py -- v0.82.0.23 ship.

Phase 12 module: per-estimator joint R² (train + held-out) at the
canonical partition-B operating point. Surfaces the recoverability-
plane linearity-axis citation for §5.

Protocol coherence: this module shares constants, splits, scaling,
and PCA-target rule with function_class_p2.compute_partition_b_shares.
Same RIDGE_ALPHA, same MLP_HYPERS (canon tanh activation, NOT ReLU),
same RKHS_KERNEL_NU=1.5 single kernel (matches in-bootstrap protocol;
the 7-kernel grid is a Q0057 in-sample construct), same RF_HYPERS,
same PARTITION_B_SEED=42, same RKHS_N_TRAIN_SUBSAMPLE=2500.

Per-estimator outputs:

  ridge.{train, heldout} -- Ridge α=0.01 R² on standardized native y.
  mlp.{train, heldout}   -- MLP (256, tanh) R² on standardized native y.
  rf.{train, heldout}    -- RF on PCA-target y.
  rkhs_median.{train, heldout} -- Matérn ν=1.5 RKHS at median pairwise
                                  length scale, on PCA-target y.

These are the values §5's recoverability-plane linearity axis cites.
The RELATED but DISTINCT linearity_max_gap field (Q0042) carries an
in-sample architecture sweep with default ReLU MLP at 3 configs (256,
64, 128-64); it is NOT derivable from the values here. See schema
docstring in results_schema.py for the field-relationship explanation.
"""

import os
import json
import datetime
import warnings

import numpy as np


def compute_estimator_joint_R2(cell_data, target_pca_components=None,
                                 seed=None, ui_msg=None):
    """Per-estimator joint R² at canonical partition-B operating point.

    Args:
      cell_data: dict from function_class_p2.load_cell_data_from_qcache.
      target_pca_components: int or None -- PCA-target the y for RF /
        RKHS when pool_dim > 64 (matches partition-B protocol).
      seed: random_state for the 80/20 split. Default
        function_class_p2.PARTITION_B_SEED.
      ui_msg: optional callable for progress logging.

    Returns:
      dict with keys 'ridge', 'mlp', 'rf', 'rkhs_median'; each value a
      dict with 'train' and 'heldout' float R² values. Plus
      'rkhs_median_length_scale' for diagnostic provenance and
      'partition_seed'.
    """
    import function_class_p2 as fc
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    if seed is None:
        seed = fc.PARTITION_B_SEED

    Xe = cell_data['Xe']; Xc = cell_data['Xc']
    Xs = cell_data['Xs']; Xp = cell_data['Xp']
    y = cell_data['y']

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

    out = {}

    if ui_msg: ui_msg("    Ridge joint R²...")
    ridge_mdl = Ridge(alpha=fc.RIDGE_ALPHA).fit(Xj_tr_s, y_tr_native_s)
    out['ridge'] = {
        'train':   float(r2_score(y_tr_native_s, ridge_mdl.predict(Xj_tr_s),
                                    multioutput='variance_weighted')),
        'heldout': float(r2_score(y_te_native_s, ridge_mdl.predict(Xj_te_s),
                                    multioutput='variance_weighted')),
    }

    if ui_msg: ui_msg("    MLP joint R²...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mlp_mdl = MLPRegressor(**fc.MLP_HYPERS).fit(Xj_tr_s, y_tr_native_s)
    out['mlp'] = {
        'train':   float(r2_score(y_tr_native_s, mlp_mdl.predict(Xj_tr_s),
                                    multioutput='variance_weighted')),
        'heldout': float(r2_score(y_te_native_s, mlp_mdl.predict(Xj_te_s),
                                    multioutput='variance_weighted')),
    }

    if ui_msg: ui_msg("    RF joint R²...")
    rf_mdl = RandomForestRegressor(**fc.RF_HYPERS).fit(Xj_tr_s, y_tr_pca_s)
    out['rf'] = {
        'train':   float(r2_score(y_tr_pca_s, rf_mdl.predict(Xj_tr_s),
                                    multioutput='variance_weighted')),
        'heldout': float(r2_score(y_te_pca_s, rf_mdl.predict(Xj_te_s),
                                    multioutput='variance_weighted')),
    }

    # RKHS -- single Matérn ν=1.5 kernel at median pairwise length scale.
    # Train subsample to control kernel-matrix memory (matches
    # function_class_p2 RKHS_N_TRAIN_SUBSAMPLE protocol).
    rng = np.random.RandomState(seed)
    n_tr_full = Xj_tr_s.shape[0]
    if n_tr_full > fc.RKHS_N_TRAIN_SUBSAMPLE:
        rkhs_idx = rng.choice(n_tr_full, size=fc.RKHS_N_TRAIN_SUBSAMPLE,
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

    if ui_msg: ui_msg("    RKHS joint R²...")
    rkhs_train_r2, rkhs_heldout_r2 = _rkhs_joint_r2_matern(
        X_rkhs_tr, Xj_te_s, y_rkhs_tr, y_te_pca_s, median_pw)
    out['rkhs_median'] = {
        'train':   float(rkhs_train_r2),
        'heldout': float(rkhs_heldout_r2),
    }

    out['rkhs_median_length_scale'] = float(median_pw)
    out['partition_seed'] = int(seed)
    return out


def _rkhs_joint_r2_matern(X_tr, X_te, y_tr, y_te, length_scale):
    """Matérn ν=1.5 kernel ridge regression -- train + held-out R²."""
    import function_class_p2 as fc
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score

    kernel = Matern(length_scale=length_scale, nu=fc.RKHS_KERNEL_NU)
    K_tr = kernel(X_tr, X_tr)
    K_te = kernel(X_te, X_tr)

    mdl = KernelRidge(alpha=fc.RKHS_ALPHA, kernel='precomputed')
    mdl.fit(K_tr, y_tr)
    train_r2 = float(r2_score(y_tr, mdl.predict(K_tr),
                                multioutput='variance_weighted'))
    heldout_r2 = float(r2_score(y_te, mdl.predict(K_te),
                                  multioutput='variance_weighted'))
    return train_r2, heldout_r2
