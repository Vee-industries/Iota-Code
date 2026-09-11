"""
ridge_policy.py -- one Ridge regularisation policy for every linear fit in the pipeline (v1.0.2).

Why: the released pipeline fitted Ridge(alpha=0.01) on standardized [E_t, C_t, S_t, prompt_tokens] blocks.
C_t takes a handful of distinct values per cell, so the C block is rank-deficient and E/C are near-collinear.
On the duplicated row sets of v1.0.1 the fit happened to behave; on de-duplicated rows (120 to 3,843 per cell)
alpha=0.01 is degenerate: 5-fold R^2 goes negative, coefficient norms are ~10x the honest fit, and the
permutation effects of E and C blow up in near-perfect cancellation (data/paper/calibration/ridge_partition_probe*.json).

Policy (default): alpha chosen by leave-one-out cross-validation (sklearn RidgeCV's closed-form GCV path,
one decomposition of X per selection) over a log grid 1e-3 .. 1e4 on the training rows, then a single Ridge
refit at that alpha. Within one process the selected alpha is memoised by the design shape (n, p), so the
bootstrap and permutation refits of the same cell reuse the cell's alpha instead of re-selecting it
(each per-cell run is its own subprocess; the global runs never see two cells with the same (n, p)).
Set IOTA_RIDGE_ALPHA=<float> to pin a fixed alpha instead (IOTA_RIDGE_ALPHA=0.01 reproduces the v1.0.1 estimator).
"""
import os
import numpy as np

RIDGE_ALPHA_GRID = np.logspace(-3, 4, 15)
_ALPHA_MEMO = {}


def policy():
    """'cv' (default) or a float alpha from IOTA_RIDGE_ALPHA."""
    v = os.environ.get('IOTA_RIDGE_ALPHA', '').strip()
    if not v or v.lower() == 'cv':
        return 'cv'
    return float(v)


def policy_label():
    p = policy()
    return 'loo_gcv_logspace(-3,4,15)' if p == 'cv' else f'fixed_{p:g}'


def select_alpha(X, y, tag=None):
    """Leave-one-out (GCV) alpha over the grid; memoised per (tag, n, p) within the process.
    tag distinguishes designs of equal shape in one process (e.g. the E-only and S-only fits of Run 0056)."""
    key = (tag, int(X.shape[0]), int(X.shape[1]))
    if key in _ALPHA_MEMO:
        return _ALPHA_MEMO[key]
    from sklearn.linear_model import RidgeCV
    cv = RidgeCV(alphas=RIDGE_ALPHA_GRID).fit(X, y)     # cv=None -> efficient LOO
    alpha = float(cv.alpha_)
    _ALPHA_MEMO[key] = alpha
    return alpha


def fit_ridge(X, y, seed=0, tag=None):
    """Fit and return a sklearn Ridge on (X, y) under the policy. The chosen alpha is on mdl.alpha
    (sklearn's own attribute) and the policy label on mdl.iota_alpha_policy."""
    from sklearn.linear_model import Ridge
    p = policy()
    alpha = select_alpha(X, y, tag) if p == 'cv' else float(p)
    mdl = Ridge(alpha=alpha).fit(X, y)
    mdl.iota_alpha_policy = policy_label()
    return mdl
