"""
_run_v5_lambda_calibration.py -- V5-integrated lambda calibration for
the apparatus I-projection aggregator.

Phase 8 (decomposition_p2._run_lagrangian_anchoring_p2) currently runs
with lambda_default = 1.0 tagged 'fallback_pending_v5_integration'.
This script computes lambda* by sweeping LAMBDA_SWEEP_GRID against the
V5 calibration suite and picking the lambda minimizing RMSE between
q*[R] and ground-truth R.

Calibration points:
  - V5a: 9 binary persistence points with closed-form truth R(p)
  - V5b: 4 nonlinear-RNN configs with MLP-reference truth R
  - V5e: 6 synergy-mixture alphas with MLP-reference truth R
  - V5f: 1 doubly-conditional point with analytic-truth R = 0.581

V5a/V5b/V5e are 2-channel (E, S) systems. The I-projection aggregator
operates on the 3-channel (E, C, R) simplex. A zero-information
Gaussian dummy C channel is synthesized for these systems so the
four-class permutation importance and the I-projection have matching
3-channel shape. The apparatus's C share is expected to be near zero
on these systems, with truth R unchanged.

V5d (pure XOR) and V5c (Ridge-invariance-under-violation) are excluded:
V5d is the apparatus's documented failure mode (Paper B §5.4) and is
not a calibration target; V5c calibrates exogeneity robustness, not the
lambda parameter.

Output: data/paper/calibration/v5/v5_lambda_calibration.json
Selected lambda is also propagated to Phase 8's per-cell aggregation in
data/paper/calibration/apparatus_p2/per_cell.json via a separate
re-aggregation step (see _run_v5_lambda_apply.py for fleet update).

Usage: py _run_v5_lambda_calibration.py [--n-steps N] [--resume]
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np

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

LAMBDA_SWEEP_GRID = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]

OUT_DIR = os.path.join(ROOT, 'data', 'paper', 'calibration', 'v5')
OUT_PATH = os.path.join(OUT_DIR, 'v5_lambda_calibration.json')


# ─── kNN-MI anchor (port from run_kraskov_anchor._kraskov_mi) ─────────

def _kraskov_mi(X, Y, n_neighbors=5):
    from sklearn.feature_selection import mutual_info_regression
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    mi_per_input = np.zeros(X.shape[1])
    for j in range(Y.shape[1]):
        mi_per_input += mutual_info_regression(
            X, Y[:, j], n_neighbors=n_neighbors, random_state=42)
    return float(mi_per_input.sum())


def _construct_anchor(mi_E, mi_S, G_tilde_C, eps=1e-6):
    """Port of run_kraskov_anchor._construct_anchor_from_kraskov.

    Returns p_anchor = (e_share, c_raw, r_share) / sum, with c_raw set
    by the four-class consensus G_tilde_C and the renormalised kNN-MI
    shares on E and S filling the remaining two coordinates.
    """
    total = mi_E + mi_S
    if total <= 0:
        return None
    e_share = mi_E / total
    r_share = mi_S / total
    if 1.0 - G_tilde_C <= eps:
        c_raw = 0.0
    else:
        c_raw = G_tilde_C / (1.0 - G_tilde_C) * (e_share + r_share)
    p_anchor = np.array([e_share, c_raw, r_share])
    p_anchor = np.clip(p_anchor, eps, 1 - eps)
    return p_anchor / p_anchor.sum()


# ─── Compute one calibration point ───────────────────────────────────

def compute_point(label, X_E, X_C, X_S, y, truth_R,
                   n_permutations=100, seed=42):
    """Four-class shares + kNN anchor + lambda sweep for one V5 point."""
    print(f"  fitting four classes for {label}...", flush=True)
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

    print(f"  computing kNN-MI anchor for {label}...", flush=True)
    t0 = time.time()
    mi_E = _kraskov_mi(X_E, y)
    mi_S = _kraskov_mi(X_S, y)
    dt = time.time() - t0
    print(f"    mi_E={mi_E:.4f} mi_S={mi_S:.4f} ({dt:.1f}s)", flush=True)
    p_anchor = _construct_anchor(mi_E, mi_S, float(G_tilde[1]))
    if p_anchor is None:
        print(f"  ! anchor undefined for {label}; skipping point", flush=True)
        return None

    sweep = []
    for lam in LAMBDA_SWEEP_GRID:
        q = bs.solve_iprojection_closed_form(p_classes, p_anchor, lam)
        q_R = float(q[2])
        sweep.append({
            'lambda': lam if np.isfinite(lam) else 'inf',
            'q_E': float(q[0]),
            'q_C': float(q[1]),
            'q_R': q_R,
            'abs_error': abs(q_R - truth_R),
        })

    return {
        'label': label,
        'truth_R': float(truth_R),
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


# ─── V5 data generators (channel-padded for 3-channel input) ─────────

def generate_v5a(p, n_steps=20000, seed=42):
    rng = np.random.RandomState(seed)
    S = np.zeros(n_steps + 1, dtype=int)
    E = rng.randint(0, 2, size=n_steps)
    S[0] = rng.randint(0, 2)
    for t in range(n_steps):
        if rng.random() < p:
            S[t + 1] = S[t]
        else:
            S[t + 1] = E[t]
    X_S = S[:-1].reshape(-1, 1).astype(float)
    X_E = E.reshape(-1, 1).astype(float)
    X_C = rng.randn(n_steps, 1)
    y = S[1:].astype(float).reshape(-1, 1)
    return X_E, X_C, X_S, y


def generate_v5b(a_scale, b_scale, n_steps=10000, dim=8,
                  noise_std=0.1, seed=42):
    rng = np.random.RandomState(seed)
    A_raw = rng.randn(dim, dim)
    A_raw = A_raw / np.max(np.abs(np.linalg.eigvals(A_raw)))
    A = a_scale * A_raw
    B_raw = rng.randn(dim, dim)
    B = b_scale * B_raw / np.linalg.norm(B_raw)
    S = np.zeros((n_steps + 1, dim))
    E = rng.randn(n_steps, dim)
    S[0] = rng.randn(dim) * 0.1
    for t in range(n_steps):
        S[t + 1] = np.tanh(A @ S[t] + B @ E[t]) + rng.randn(dim) * noise_std
    X_S = S[:-1]
    X_E = E
    X_C = rng.randn(n_steps, dim)
    y = S[1:]
    return X_E, X_C, X_S, y


def generate_v5e(alpha, n_steps=20000, p_persist=0.5, seed=42):
    rng = np.random.RandomState(seed)
    E = rng.randint(0, 2, size=n_steps)
    S = np.zeros(n_steps + 1, dtype=int)
    S[0] = rng.randint(0, 2)
    for t in range(n_steps):
        if rng.random() < alpha:
            S[t + 1] = S[t] ^ E[t]
        else:
            if rng.random() < p_persist:
                S[t + 1] = S[t]
            else:
                S[t + 1] = E[t]
    X_S = S[:-1].reshape(-1, 1).astype(float)
    X_E = E.reshape(-1, 1).astype(float)
    X_C = rng.randn(n_steps, 1)
    y = S[1:].astype(float).reshape(-1, 1)
    return X_E, X_C, X_S, y


def generate_v5f(n_steps=20000, seed=42):
    sigma_S = 0.78209375
    sigma_Y = 0.547765625
    E, C, S, Y = v5._v5f_generate(n_steps, sigma_S, sigma_Y, seed=seed)
    return E, C, S, Y


# ─── Resume + aggregate ──────────────────────────────────────────────

def _load_existing():
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_partial(points, aggregate_so_far_n):
    out = {
        'experiment': 'V5_integrated_lambda_calibration',
        'description': (
            'Sweep lambda across V5a/V5b/V5e/V5f, pick min-RMSE lambda. '
            'V5a/V5b/V5e use 3-channel input with zero-info Gaussian C. '
            'V5f is natively 3-channel.'),
        'lambda_sweep_grid': [str(x) if not np.isfinite(x) else x
                              for x in LAMBDA_SWEEP_GRID],
        'n_calibration_points_so_far': aggregate_so_far_n,
        'per_point': points,
        'status': 'in_progress',
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)


def _aggregate(points):
    aggregate = {}
    for lam in LAMBDA_SWEEP_GRID:
        lam_key = lam if np.isfinite(lam) else 'inf'
        errors = []
        for pt in points:
            for sw in pt['lambda_sweep']:
                if sw['lambda'] == lam_key:
                    errors.append(sw['abs_error'])
                    break
        if errors:
            aggregate[str(lam_key)] = {
                'mean_abs_error': float(np.mean(errors)),
                'rmse': float(np.sqrt(np.mean(np.array(errors) ** 2))),
                'max_error': float(max(errors)),
                'n_points': len(errors),
            }
    finite_lams = [(k, v['rmse']) for k, v in aggregate.items() if k != 'inf']
    lam_star_key, lam_star_rmse = min(finite_lams, key=lambda x: x[1])
    return aggregate, lam_star_key, lam_star_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-steps', type=int, default=None,
                        help='override default n_steps per system (smaller = faster smoke test)')
    parser.add_argument('--resume', action='store_true',
                        help='reuse per-point results from prior partial run')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    cached_points_by_label = {}
    if args.resume:
        existing = _load_existing()
        if existing:
            for pt in (existing.get('per_point') or []):
                cached_points_by_label[pt['label']] = pt
            print(f"[resume] loaded {len(cached_points_by_label)} cached point(s)")

    points = []

    # V5a: 9 binary persistence points
    v5a_ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    v5a_n = args.n_steps if args.n_steps else 20000
    for p in v5a_ps:
        label = f'v5a_p{p}'
        if label in cached_points_by_label:
            print(f"[skip] {label}")
            points.append(cached_points_by_label[label])
            continue
        truth = v5.true_R_toy(p)
        print(f"\n[V5a] {label}, truth_R={truth:.4f}", flush=True)
        X_E, X_C, X_S, y = generate_v5a(p, n_steps=v5a_n)
        pt = compute_point(label, X_E, X_C, X_S, y, truth)
        if pt is not None:
            points.append(pt)
            _save_partial(points, len(points))

    # V5b: 4 nonlinear-RNN configs
    v5b_configs = [
        ("E-dominant", 0.2,  0.8, 0.043),
        ("balanced",   0.5,  0.5, 0.2812),
        ("S-dominant", 0.8,  0.2, 0.7897),
        ("strong-S",   0.95, 0.1, 0.9678),
    ]
    v5b_n = args.n_steps if args.n_steps else 10000
    for cfg_label, a_s, b_s, R_ref in v5b_configs:
        label = f'v5b_{cfg_label}'
        if label in cached_points_by_label:
            print(f"[skip] {label}")
            points.append(cached_points_by_label[label])
            continue
        print(f"\n[V5b] {label}, R_ref={R_ref:.4f}", flush=True)
        X_E, X_C, X_S, y = generate_v5b(a_s, b_s, n_steps=v5b_n)
        pt = compute_point(label, X_E, X_C, X_S, y, R_ref)
        if pt is not None:
            points.append(pt)
            _save_partial(points, len(points))

    # V5e: 6 synergy-mixture alphas
    v5e_truths = {0.0: 0.5024, 0.2: 0.5096, 0.4: 0.505,
                  0.6: 0.5041, 0.8: 0.5007, 1.0: 0.5003}
    v5e_n = args.n_steps if args.n_steps else 20000
    for alpha, R_ref in v5e_truths.items():
        label = f'v5e_alpha{alpha}'
        if label in cached_points_by_label:
            print(f"[skip] {label}")
            points.append(cached_points_by_label[label])
            continue
        print(f"\n[V5e] {label}, R_ref={R_ref:.4f}", flush=True)
        X_E, X_C, X_S, y = generate_v5e(alpha, n_steps=v5e_n)
        pt = compute_point(label, X_E, X_C, X_S, y, R_ref)
        if pt is not None:
            points.append(pt)
            _save_partial(points, len(points))

    # V5f: 1 doubly-conditional point, analytic truth
    v5f_n = args.n_steps if args.n_steps else 20000
    label = 'v5f'
    if label in cached_points_by_label:
        print(f"[skip] {label}")
        points.append(cached_points_by_label[label])
    else:
        print(f"\n[V5f] truth_R=0.5813", flush=True)
        E, C, S, Y = generate_v5f(n_steps=v5f_n)
        pt = compute_point('v5f', E, C, S, Y, 0.5813)
        if pt is not None:
            points.append(pt)
            _save_partial(points, len(points))

    # Aggregate + pick lambda*
    aggregate, lam_star_key, lam_star_rmse = _aggregate(points)

    out = {
        'experiment': 'V5_integrated_lambda_calibration',
        'description': (
            'Sweep lambda across V5a/V5b/V5e/V5f, pick min-RMSE lambda. '
            'V5a/V5b/V5e use 3-channel input with zero-info Gaussian C. '
            'V5f is natively 3-channel.'),
        'lambda_sweep_grid': [str(x) if not np.isfinite(x) else x
                              for x in LAMBDA_SWEEP_GRID],
        'aggregate_per_lambda': aggregate,
        'lambda_star': float(lam_star_key) if lam_star_key != 'inf' else 'inf',
        'lambda_star_rmse': float(lam_star_rmse),
        'lambda_star_selection': 'argmin RMSE across V5 calibration points',
        'n_calibration_points': len(points),
        'per_point': points,
        'status': 'complete',
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"λ* = {lam_star_key}, RMSE = {lam_star_rmse:.4f}, "
          f"N = {len(points)} calibration points")
    print(f"{'='*60}")
    print(f"\n{'λ':>6} | {'RMSE':>8} | {'MAE':>8} | {'MaxErr':>8} | N")
    print("-" * 50)
    for lam in LAMBDA_SWEEP_GRID:
        lam_key = lam if np.isfinite(lam) else 'inf'
        a = aggregate.get(str(lam_key))
        if a is None:
            continue
        marker = ' *' if str(lam_key) == lam_star_key else ''
        print(f"{str(lam_key):>6} | {a['rmse']:>8.4f} | "
              f"{a['mean_abs_error']:>8.4f} | {a['max_error']:>8.4f} | "
              f"{a['n_points']:>3}{marker}")

    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
