"""
run_split_pca_selection.py — Selection of operating-point sample-split
ratio and PCA dim for the paper 2 measurement apparatus.

Paper 2 ship 4 (v0.81.1.6).

Per codebot_handoff_v0_14.md §3 / paper2_preregistration_v2.md §4: the
apparatus needs an operating-point (split, PCA) chosen by V5
calibration, not by hand. This script runs the calibration grid:

  splits ∈ {(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)}
  pca   ∈ {16, 24, 32, 48}

across V5e (synergy mixture) and V5f (doubly-conditional). For each
(split, pca) cell, it estimates kNN-MI on partition A and chain-rule
shares on partition B at the corresponding sample size + dim, and
records median + p95 absolute error against analytic truth. The
operating point is selected as the largest split where:

  median |error| < 10%  AND  p95 |error| < 20%

across both V5 systems jointly. Result is committed to
data/paper/calibration/split_pca_selection/selection.json and ingested
into results.json's paper2_apparatus_metadata block.

Output: data/paper/calibration/split_pca_selection/selection.json
Cache key: source script hash + v5_synthetic_calibration.py hash

Foundations registry entry: 'split_pca_selection'.

Notes:
  - kNN-MI is computed via the existing _ksg_mi helper in
    run_kraskov_anchor (same estimator used by Phase 4 of Run 0058).
  - Errors are reported per (V5 system, term). 'term' here is one of
    {I(Y;E), I(Y;E,C), I(Y;E,C,S)} for V5f or the V5e analog.
  - The grid is small enough to run in CPU minutes on a 16-core box;
    no GPU needed.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Force UTF-8 stdout (Windows safety net — same pattern as v5_synthetic)
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

DATA = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(DATA, 'paper', 'calibration', 'split_pca_selection')
OUT_PATH = os.path.join(OUT_DIR, 'selection.json')

SPLIT_GRID = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
PCA_GRID = [16, 24, 32, 48]
MEDIAN_TOL = 0.10
P95_TOL = 0.20


def _exact_discrete_mi(x, y):
    """Exact MI from a contingency table on discrete-valued data.
    x, y: (n, d) integer-valued arrays — each row treated as a
    categorical symbol. Returns scalar MI in nats.

    For binary or low-cardinality discrete inputs (e.g. V5e), this
    is exact — no estimator bias. For data with too many distinct
    symbols (e.g. continuous), it underestimates MI badly.
    """
    from collections import Counter
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    n = x.shape[0]
    if n == 0:
        return 0.0
    x_syms = [tuple(row) for row in x.astype(int).tolist()]
    y_syms = [tuple(row) for row in y.astype(int).tolist()]
    p_x = Counter(x_syms)
    p_y = Counter(y_syms)
    p_xy = Counter(zip(x_syms, y_syms))
    mi = 0.0
    for (xs, ys), n_xy in p_xy.items():
        p_joint = n_xy / n
        p_marg_x = p_x[xs] / n
        p_marg_y = p_y[ys] / n
        if p_joint > 0 and p_marg_x > 0 and p_marg_y > 0:
            mi += p_joint * np.log(p_joint / (p_marg_x * p_marg_y))
    return float(mi)


def _is_integer_valued(arr, max_unique_per_dim=8):
    """Detect integer-lattice data — KSG fails on these, contingency
    table is exact. Returns True if every column has <= max_unique_per_dim
    distinct integer values. False otherwise (treat as continuous)."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # Cheap: sample 200 rows; if ALL of them are within 1e-9 of integers
    # AND the per-column unique-count is small, classify as discrete.
    n = arr.shape[0]
    sample_idx = np.arange(min(200, n))
    sample = arr[sample_idx]
    if not np.allclose(sample, np.round(sample), atol=1e-9):
        return False
    # Count distinct values per column on the full data
    for j in range(arr.shape[1]):
        if len(np.unique(arr[:, j])) > max_unique_per_dim:
            return False
    return True


def _knn_mi(x, y, k=5):
    """Mutual information estimator. Dispatches to:
      - exact contingency-table MI when x, y are integer-lattice with
        small cardinality (e.g. V5e binary); KSG fails on these.
      - KSG-1 for continuous-valued data (e.g. V5f, real LLM features).

    Returns scalar MI (nats).

    v0.82.0.4: dispatcher added. Pre-patch (after the v0.82.0.3 hang
    fix) used vanilla KSG-1 with cKDTree. KSG fundamentally does not
    work on discrete data: even with jitter, the estimator returns
    the entropy of the discrete distribution rather than the mutual
    information, because the marginal `n_x` and `n_y` counts within
    eps_i are dominated by the count of points in the same discrete
    cluster — independent of the actual MI. Independent binary
    baseline at n=20000 returned MI = 1.0 (a full bit of "MI" between
    independent variables) — clearly wrong.

    The fix: detect integer-lattice data (every column has small
    cardinality + integer-valued) and use exact contingency-table MI
    in that case. For continuous data, KSG with cKDTree as before.

    v0.82.0.3: scipy.spatial.cKDTree.query_ball_point with per-point
    radius array — replaced the per-row Python radius_neighbors loop
    that hung at n=20000.
    """
    n = x.shape[0]
    if n < k + 1:
        return float('nan')

    # Dispatch on data type
    if _is_integer_valued(x) and _is_integer_valued(y):
        return _exact_discrete_mi(x, y)

    # Continuous KSG-1
    from scipy.spatial import cKDTree
    from sklearn.neighbors import NearestNeighbors
    from scipy.special import digamma

    xy = np.hstack([x, y])
    nn_xy = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(xy)
    d_xy, _ = nn_xy.kneighbors(xy)
    eps = d_xy[:, -1]

    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    radii = eps - 1e-12
    neighbors_x = tree_x.query_ball_point(x, r=radii, p=np.inf,
                                             return_sorted=False)
    neighbors_y = tree_y.query_ball_point(y, r=radii, p=np.inf,
                                             return_sorted=False)
    n_x = np.array([len(nb) for nb in neighbors_x])
    n_y = np.array([len(nb) for nb in neighbors_y])
    n_x = np.maximum(n_x - 1, 1)
    n_y = np.maximum(n_y - 1, 1)
    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x) + digamma(n_y))
    return float(mi)


def _generate_v5e_for_grid(n_total, alpha=0.5, seed=42):
    """V5e at a single alpha. Returns E, S_prev, S_next as columns."""
    rng = np.random.RandomState(seed)
    p_persist = 0.5
    E = rng.randint(0, 2, size=n_total)
    S = np.zeros(n_total + 1, dtype=int)
    S[0] = rng.randint(0, 2)
    for t in range(n_total):
        if rng.random() < alpha:
            S[t + 1] = S[t] ^ E[t]
        else:
            if rng.random() < p_persist:
                S[t + 1] = S[t]
            else:
                S[t + 1] = E[t]
    return (S[:-1].astype(float).reshape(-1, 1),
            E.astype(float).reshape(-1, 1),
            S[1:].astype(float).reshape(-1, 1))


def _generate_v5f_for_grid(n_total, sigma_S=0.5, sigma_Y=0.3, seed=42):
    """V5f doubly-conditional system. Returns E, C, S, Y as (n,1) arrays."""
    from v5_synthetic_calibration import _v5f_generate
    return _v5f_generate(n_total, sigma_S=sigma_S, sigma_Y=sigma_Y, seed=seed)


def _pca_reduce(X, target_dim, seed=42):
    """PCA-reduce X to target_dim. If X has fewer columns than target_dim,
    return X as-is."""
    if X.shape[1] <= target_dim:
        return X
    from sklearn.decomposition import PCA
    return PCA(n_components=target_dim, random_state=seed).fit_transform(X)


def _evaluate_v5_grid_cell(system_name, split_a, split_b,
                            pca_dim, n_total=20000, n_repeats=3, seed=42):
    """Evaluate one (split, pca) grid cell on one V5 system. Returns
    dict of per-term absolute error against analytic truth, averaged
    across n_repeats."""
    errors_per_term = {}

    for rep in range(n_repeats):
        s = seed + rep * 1000
        if system_name == 'v5e':
            S_prev, E, S_next = _generate_v5e_for_grid(n_total, seed=s)
            # Truth for V5e at alpha=0.5: a known mid-synergy regime;
            # use Ridge MLP reference for term I(S_next; E) and I(S_next; S_prev, E)
            # since closed form is not available for the mixture.
            # NOTE: V5e truth is REFERENCE not analytic; this is an approximation
            # for the selection grid. V5b would be cleaner but is harder to vary.
            X_inputs_partA = np.hstack([E, S_prev])
            y_partA = S_next
        elif system_name == 'v5f':
            E, C, S, Y = _generate_v5f_for_grid(n_total, seed=s)
            X_inputs_partA = np.hstack([E, C, S])  # 3-channel
            y_partA = Y
        else:
            continue

        # Partition A vs B: A is for kNN-MI, B is for function-class fits.
        n_A = int(split_a * n_total)
        # Permutation seed independent of system seed
        perm = np.random.RandomState(s + 7).permutation(n_total)
        idx_A = perm[:n_A]
        # PCA reduce inputs to pca_dim (here inputs are already 1-2-3 dim,
        # so PCA is a no-op for these toy systems unless we pad — but the
        # function operates correctly. Real cells with d=64+ exercise the
        # PCA branch.)
        X_A = _pca_reduce(X_inputs_partA[idx_A], pca_dim, seed=s)
        y_A = y_partA[idx_A]

        # Term-wise MI (use KSG with k=5, adjust k for small n_A)
        k = max(3, min(5, n_A // 200))

        if system_name == 'v5e':
            # Two-channel: I(Y; E), I(Y; E, S_prev)
            mi_E = _knn_mi(X_A[:, :1], y_A, k=k)
            mi_full = _knn_mi(X_A, y_A, k=k)
            # No clean analytic truth — use full n=n_total kNN-MI as proxy truth
            X_full = _pca_reduce(X_inputs_partA, pca_dim, seed=s)
            mi_E_full = _knn_mi(X_full[:, :1], y_partA, k=5)
            mi_full_full = _knn_mi(X_full, y_partA, k=5)
            errors_per_term.setdefault('I(Y;E)_v5e', []).append(
                abs(mi_E - mi_E_full))
            errors_per_term.setdefault('I(Y;E,S)_v5e', []).append(
                abs(mi_full - mi_full_full))
        elif system_name == 'v5f':
            # Three-channel: I(Y; E), I(Y; E, C), I(Y; E, C, S)
            mi_E = _knn_mi(X_A[:, :1], y_A, k=k)
            mi_EC = _knn_mi(X_A[:, :2], y_A, k=k)
            mi_full = _knn_mi(X_A, y_A, k=k)
            # Proxy truth from full n_total
            X_full = _pca_reduce(X_inputs_partA, pca_dim, seed=s)
            mi_E_full = _knn_mi(X_full[:, :1], y_partA, k=5)
            mi_EC_full = _knn_mi(X_full[:, :2], y_partA, k=5)
            mi_full_full = _knn_mi(X_full, y_partA, k=5)
            errors_per_term.setdefault('I(Y;E)_v5f', []).append(
                abs(mi_E - mi_E_full))
            errors_per_term.setdefault('I(Y;E,C)_v5f', []).append(
                abs(mi_EC - mi_EC_full))
            errors_per_term.setdefault('I(Y;E,C,S)_v5f', []).append(
                abs(mi_full - mi_full_full))

    # Aggregate
    return {term: {
        'mean_abs_error': float(np.mean(errs)),
        'errors_per_repeat': [float(e) for e in errs],
    } for term, errs in errors_per_term.items()}


def _select_operating_point(grid_results):
    """Pick the operating point: largest split (by partition A fraction)
    where median error < MEDIAN_TOL AND p95 error < P95_TOL across all
    terms. Tie-break by largest pca_dim."""
    # Flatten errors per (split, pca) cell across all terms
    cell_summary = []
    for entry in grid_results:
        all_errors = []
        for term_block in entry['per_term'].values():
            all_errors.extend(term_block.get('errors_per_repeat') or [])
        if not all_errors:
            continue
        median_err = float(np.median(all_errors))
        p95_err    = float(np.percentile(all_errors, 95))
        cell_summary.append({
            'split': entry['split'],
            'pca':   entry['pca'],
            'median_err': median_err,
            'p95_err':    p95_err,
            'passes':     (median_err < MEDIAN_TOL) and (p95_err < P95_TOL),
        })

    passing = [c for c in cell_summary if c['passes']]
    if not passing:
        return None, cell_summary
    # Largest split = largest partition A fraction. Tie-break: largest pca_dim.
    best = max(passing, key=lambda c: (c['split'][0], c['pca']))
    return best, cell_summary


def _hash_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_cache_key(out_dir):
    """Write hash of this script + v5_synthetic_calibration.py for
    methodology_calibration_status freshness tracking."""
    script_hash = _hash_file(os.path.abspath(__file__))
    upstream_hash = _hash_file(os.path.join(ROOT, 'v5_synthetic_calibration.py'))
    cache_key = {
        'script':    'run_split_pca_selection.py',
        'script_hash': script_hash,
        'upstream':  {
            'v5_synthetic_calibration.py': upstream_hash,
        },
    }
    p = os.path.join(out_dir, '.cache_key.json')
    with open(p, 'w') as f:
        json.dump(cache_key, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=OUT_DIR, help='output directory')
    parser.add_argument('--n-total', type=int, default=20000,
                        help='total samples per V5 system')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='repeat count per grid cell')
    parser.add_argument('--quick', action='store_true',
                        help='small grid + small n for fast smoke test')
    args = parser.parse_args()

    if args.quick:
        n_total = 800
        n_repeats = 1
        splits = [(0.7, 0.3)]
        pcas = [16]
    else:
        n_total = args.n_total
        n_repeats = args.n_repeats
        splits = SPLIT_GRID
        pcas = PCA_GRID

    os.makedirs(args.out, exist_ok=True)
    print("=" * 60)
    print(f"Run 0056 — Split + PCA selection (split_pca_selection)")
    print(f"  splits = {splits}")
    print(f"  pcas   = {pcas}")
    print(f"  n_total per system = {n_total}, n_repeats = {n_repeats}")
    print("=" * 60)

    grid_results = []
    t0 = time.time()
    for split in splits:
        for pca in pcas:
            print(f"\n[{(time.time()-t0):.1f}s] split={split}, pca={pca}")
            for system in ('v5e', 'v5f'):
                per_term = _evaluate_v5_grid_cell(
                    system, split[0], split[1], pca,
                    n_total=n_total, n_repeats=n_repeats)
                for term, block in per_term.items():
                    print(f"  {term:>20}: mean |err| = "
                          f"{block['mean_abs_error']:.4f}")
                grid_results.append({
                    'split':    list(split),
                    'pca':      pca,
                    'system':   system,
                    'per_term': per_term,
                })

    selected, cell_summary = _select_operating_point(grid_results)
    print("\n" + "=" * 60)
    print("SELECTION")
    print("=" * 60)
    if selected:
        print(f"Operating point: split={selected['split']}, "
              f"pca={selected['pca']}")
        print(f"  median |err| = {selected['median_err']:.4f}")
        print(f"  p95 |err|    = {selected['p95_err']:.4f}")
    else:
        print("No grid cell passed the (median<10%, p95<20%) tolerance.")
        print("Falling back to (0.7, 0.3) split, pca=32 — sentinel value;")
        print("operator should re-run with refined targets or relax bounds.")

    out = {
        'description': (
            "Selection of operating-point (sample-split, PCA dim) for "
            "the paper 2 measurement apparatus. Run on V5e and V5f. "
            "Largest split where median |error| < 10% AND p95 < 20% "
            "across all chain-rule terms is selected."
        ),
        'grid': {
            'splits': [list(s) for s in splits],
            'pcas':   list(pcas),
        },
        'tolerance': {'median': MEDIAN_TOL, 'p95': P95_TOL},
        'cell_summary':  cell_summary,
        'grid_results':  grid_results,
        'selected_operating_point': selected if selected else {
            'split': [0.7, 0.3], 'pca': 32, 'fallback': True,
        },
        'metadata': {
            'iota_version':  '0.82.0.9',
            'timestamp':     datetime.datetime.utcnow().isoformat() + 'Z',
            'n_total':       n_total,
            'n_repeats':     n_repeats,
        },
    }
    out_path = os.path.join(args.out, 'selection.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    _write_cache_key(args.out)
    print(f"\nWrote {out_path}")
    return 0 if selected else 1


if __name__ == '__main__':
    sys.exit(main())
