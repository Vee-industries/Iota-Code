"""
_run_v5g_mid_dim_bridge.py -- V5g mid-dimensional bridge calibration.

The V5 suite spans state dimensions 1 to 8 (V5a/c/d binary, V5e binary,
V5b 8-d continuous, V5f 1-d in S). Empirical transformer cells have
pool_dim 64 to 1024 after PCA reduction. The §5 bridging argument from
V5 to the empirical fleet rests on the recoverability-plane diagnostic
as the operative axis; whether boundary position drifts with dimension
across two orders of magnitude is not directly tested by V5a-V5f.

V5g closes the gap. The construction is the stable linear-Gaussian
system from Paper 0 §8: S_{t+1} = A S_t + B E_t + ε_t, with closed-form
R derivable from the Lyapunov solution. Pure linear, no synergy; this
places V5g in the recoverable corner of the recoverability plane and
tests dimension-stability of regime-position rather than apparatus
behavior in non-recoverable regimes (which V5d/V5e cover at low
dimension).

Dimensions tested: d ∈ {32, 64, 128}. Three configs per dim spanning
R-truth low to high (E-dominant, balanced, S-dominant), mirroring V5b.

Output: data/paper/calibration/v5/v5g_bridge_calibration.json

Usage: py _run_v5g_mid_dim_bridge.py [--n-steps N] [--lambda L]
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from numpy.linalg import slogdet

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings("ignore")

import v5_synthetic_calibration as v5
import bayesian_solver as bs

OUT_DIR = os.path.join(ROOT, 'data', 'paper', 'calibration', 'v5')
OUT_PATH = os.path.join(OUT_DIR, 'v5g_bridge_calibration.json')


# ─── Closed-form R for the linear-Gaussian system (Paper 0 §8) ───────

def closed_form_R(A, B, Sigma_E, Sigma_eps):
    """R for S_{t+1} = A S_t + B E_t + ε under stationarity.

    Returns R = (log|A Σ_S A^T + Σ_ε| - log|Σ_ε|)
                / (log|Σ_S| - log|Σ_ε|)
    where Σ_S satisfies the discrete Lyapunov equation
    Σ_S = A Σ_S A^T + B Σ_E B^T + Σ_ε.
    """
    Q = B @ Sigma_E @ B.T + Sigma_eps
    Sigma_S = solve_discrete_lyapunov(A, Q)
    M_num = A @ Sigma_S @ A.T + Sigma_eps
    M_den = Sigma_S
    s_num, ld_num = slogdet(M_num)
    s_den, ld_den = slogdet(M_den)
    s_eps, ld_eps = slogdet(Sigma_eps)
    if min(s_num, s_den, s_eps) <= 0:
        return None
    R = (ld_num - ld_eps) / (ld_den - ld_eps)
    return float(R)


# ─── Data generator ──────────────────────────────────────────────────

def generate_v5g(dim, a_scale, b_scale, n_steps=10000,
                  sigma_eps=0.1, seed=42):
    """Stable linear-Gaussian system at given dim and config.

    Returns (X_E, X_C, X_S, y, R_truth) where X_C is a zero-info
    Gaussian dummy channel to match the apparatus's 3-channel input.
    """
    rng = np.random.RandomState(seed)
    # A: random with spectral radius < 1
    A_raw = rng.randn(dim, dim)
    A_raw = A_raw / (np.max(np.abs(np.linalg.eigvals(A_raw))) + 0.05)
    A = a_scale * A_raw  # scaled so |a_scale| < 1
    # B: random with unit Frobenius norm scaled by b_scale
    B_raw = rng.randn(dim, dim)
    B = b_scale * B_raw / np.linalg.norm(B_raw, ord='fro')

    Sigma_E = np.eye(dim)
    Sigma_eps = (sigma_eps ** 2) * np.eye(dim)

    # Closed-form truth
    R_truth = closed_form_R(A, B, Sigma_E, Sigma_eps)

    # Sample n_steps
    S = np.zeros((n_steps + 1, dim))
    E = rng.randn(n_steps, dim)
    S[0] = rng.randn(dim) * 0.1
    eps = sigma_eps * rng.randn(n_steps, dim)
    for t in range(n_steps):
        S[t + 1] = A @ S[t] + B @ E[t] + eps[t]
    X_S = S[:-1]
    X_E = E
    X_C = rng.randn(n_steps, dim)  # zero-info dummy
    y = S[1:]
    return X_E, X_C, X_S, y, R_truth


# ─── Apparatus on one V5g point ──────────────────────────────────────

def _kraskov_mi(X, Y, n_neighbors=5, pca_dim=32):
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.decomposition import PCA
    if Y.shape[1] > pca_dim:
        Y = PCA(n_components=pca_dim, random_state=42).fit_transform(Y)
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=42).fit_transform(X)
    mi_per_input = np.zeros(X.shape[1])
    for j in range(Y.shape[1]):
        mi_per_input += mutual_info_regression(
            X, Y[:, j], n_neighbors=n_neighbors, random_state=42)
    return float(mi_per_input.sum())


def _construct_anchor(mi_E, mi_S, G_tilde_C, eps=1e-6):
    total = mi_E + mi_S
    if total <= 0:
        return None
    e_share = mi_E / total
    r_share = mi_S / total
    if 1.0 - G_tilde_C <= eps:
        c_raw = 0.0
    else:
        c_raw = G_tilde_C / (1.0 - G_tilde_C) * (e_share + r_share)
    p = np.array([e_share, c_raw, r_share])
    p = np.clip(p, eps, 1 - eps)
    return p / p.sum()


def compute_v5g_point(label, X_E, X_C, X_S, y, R_truth,
                       lam_default=1.0, n_permutations=100, seed=42):
    print(f"\n[V5g] {label}, truth_R={R_truth:.4f}", flush=True)
    shares = {}
    for reg in ('ridge', 'mlp', 'rkhs', 'rf'):
        t0 = time.time()
        s, r2 = v5.estimate_R_3channel(
            X_E, X_C, X_S, y,
            n_permutations=n_permutations, seed=seed, regressor=reg)
        dt = time.time() - t0
        print(f"    {reg:>5}: E={s['E']:.3f} C={s['C']:.3f} S={s['S']:.3f} "
              f"R2={r2:.3f} ({dt:.1f}s)", flush=True)
        shares[reg] = s

    p_classes = np.array([
        [shares['ridge']['E'], shares['ridge']['C'], shares['ridge']['S']],
        [shares['mlp']['E'],   shares['mlp']['C'],   shares['mlp']['S']],
        [shares['rkhs']['E'],  shares['rkhs']['C'],  shares['rkhs']['S']],
        [shares['rf']['E'],    shares['rf']['C'],    shares['rf']['S']],
    ])
    G = np.exp(np.mean(np.log(np.clip(p_classes, 1e-6, 1 - 1e-6)), axis=0))
    G_tilde = G / G.sum()

    print(f"  computing kNN-MI anchor...", flush=True)
    t0 = time.time()
    mi_E = _kraskov_mi(X_E, y)
    mi_S = _kraskov_mi(X_S, y)
    dt = time.time() - t0
    print(f"    mi_E={mi_E:.4f} mi_S={mi_S:.4f} ({dt:.1f}s)", flush=True)
    p_anchor = _construct_anchor(mi_E, mi_S, float(G_tilde[1]))
    if p_anchor is None:
        return None

    q = bs.solve_iprojection_closed_form(p_classes, p_anchor, lam_default)
    q_R = float(q[2])

    # Also compute a small lambda sweep for diagnostic
    LAMS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    sweep = []
    for lam in LAMS:
        q_lam = bs.solve_iprojection_closed_form(p_classes, p_anchor, lam)
        sweep.append({
            'lambda': lam,
            'q_R': float(q_lam[2]),
            'abs_error': abs(float(q_lam[2]) - R_truth),
        })

    return {
        'label': label,
        'truth_R': float(R_truth),
        'q_star_R': q_R,
        'q_star_abs_error': abs(q_R - R_truth),
        'lambda_used': float(lam_default),
        'p_classes': {
            'ridge': shares['ridge'],
            'mlp':   shares['mlp'],
            'rkhs':  shares['rkhs'],
            'rf':    shares['rf'],
        },
        'p_anchor': p_anchor.tolist(),
        'G_tilde_C': float(G_tilde[1]),
        'mi_E_knn': mi_E,
        'mi_S_knn': mi_S,
        'lambda_sweep': sweep,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-steps', type=int, default=10000)
    parser.add_argument('--lambda-default', type=float, default=None,
                        help='lambda value to report at headline q*; '
                             'falls back to V5 calibration result if None, '
                             'or 1.0 if no calibration is available')
    parser.add_argument('--dims', type=int, nargs='+',
                        default=[32, 64, 128],
                        help='dimensions to test (default: 32 64 128)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Pick lambda_default
    lam_default = args.lambda_default
    lam_source = 'cli_override' if lam_default is not None else None
    if lam_default is None:
        cal_path = os.path.join(OUT_DIR, 'v5_lambda_calibration.json')
        if os.path.exists(cal_path):
            try:
                with open(cal_path, 'r', encoding='utf-8-sig') as f:
                    cal = json.load(f)
                lam_default = cal.get('lambda_star')
                if lam_default == 'inf':
                    lam_default = float('inf')
                else:
                    lam_default = float(lam_default)
                lam_source = f"v5_lambda_calibration.json (RMSE={cal.get('lambda_star_rmse'):.4f})"
            except Exception:
                pass
    if lam_default is None:
        lam_default = 1.0
        lam_source = 'fallback_1.0'

    print(f"Using lambda_default = {lam_default} (source: {lam_source})")
    print(f"Testing dimensions: {args.dims}")
    print(f"n_steps per system: {args.n_steps}")

    configs = [
        ("E-dominant", 0.3, 0.9),
        ("balanced",   0.6, 0.6),
        ("S-dominant", 0.9, 0.3),
    ]

    points = []
    for d in args.dims:
        for cfg_label, a_s, b_s in configs:
            label = f'v5g_d{d}_{cfg_label}'
            X_E, X_C, X_S, y, R_truth = generate_v5g(
                dim=d, a_scale=a_s, b_scale=b_s,
                n_steps=args.n_steps)
            if R_truth is None or not np.isfinite(R_truth):
                print(f"  ! truth R undefined for {label}; skipping")
                continue
            pt = compute_v5g_point(
                label, X_E, X_C, X_S, y, R_truth,
                lam_default=lam_default)
            if pt is not None:
                pt['dim'] = d
                pt['config'] = cfg_label
                pt['a_scale'] = a_s
                pt['b_scale'] = b_s
                points.append(pt)

    # Aggregate: by-dim, by-config
    by_dim = {}
    for pt in points:
        by_dim.setdefault(pt['dim'], []).append(pt)
    aggregate_by_dim = {}
    for d, pts in by_dim.items():
        errors = [p['q_star_abs_error'] for p in pts]
        aggregate_by_dim[str(d)] = {
            'mean_abs_error': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean(np.array(errors) ** 2))),
            'max_error': float(max(errors)),
            'n_configs': len(pts),
        }

    out = {
        'experiment': 'V5g_mid_dimensional_bridge_calibration',
        'description': (
            'Stable linear-Gaussian systems (Paper 0 §8 closed form) at '
            'mid-dimensional state, testing whether the recoverability-'
            'plane boundary is dimension-stable between V5 (1-8d) and '
            'transformer empirical cells (64-1024d after PCA).'),
        'lambda_default': lam_default if np.isfinite(lam_default) else 'inf',
        'lambda_source': lam_source,
        'dims_tested': args.dims,
        'configs_per_dim': [c[0] for c in configs],
        'n_steps': args.n_steps,
        'aggregate_by_dim': aggregate_by_dim,
        'per_point': points,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"V5g bridge results (lambda = {lam_default})")
    print(f"{'='*60}")
    print(f"\n{'dim':>5} | {'RMSE':>8} | {'MAE':>8} | {'MaxErr':>8} | N")
    print("-" * 50)
    for d, a in sorted(aggregate_by_dim.items(), key=lambda x: int(x[0])):
        print(f"{d:>5} | {a['rmse']:>8.4f} | "
              f"{a['mean_abs_error']:>8.4f} | {a['max_error']:>8.4f} | "
              f"{a['n_configs']:>3}")

    print(f"\nPer-point detail:")
    for pt in points:
        print(f"  {pt['label']:>30}: truth={pt['truth_R']:.4f} "
              f"q*={pt['q_star_R']:.4f} err={pt['q_star_abs_error']:.4f}")

    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
