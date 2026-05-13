"""
run_knn_mi_reliability.py — kNN-MI estimator reliability check on V5
systems at the operating point selected by run_split_pca_selection.

Paper 2 ship 4 (v0.81.1.6).

Per codebot_handoff_v0_14.md §3.4: the apparatus's anchor is the
kNN-MI estimate; if the anchor is unreliable, the apparatus output
is unreliable. This run measures kNN-MI reliability via repeated
estimation under bootstrap resampling on V5e and V5f, at the operating
point chosen by run_split_pca_selection. Reports per-term median and
95th-percentile relative error against analytic-truth (V5f) or kNN-MI-
at-full-n proxy truth (V5e — no closed-form).

Output: data/paper/calibration/knn_mi_reliability/reliability.json
Cache key: source script + v5_synthetic_calibration.py + selection.json

Foundations registry entry: 'knn_mi_reliability'.
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

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

DATA = os.path.join(ROOT, 'data')
SELECTION_PATH = os.path.join(DATA, 'paper', 'calibration',
                               'split_pca_selection', 'selection.json')
OUT_DIR = os.path.join(DATA, 'paper', 'calibration', 'knn_mi_reliability')
OUT_PATH = os.path.join(OUT_DIR, 'reliability.json')

N_BOOTSTRAP = 30  # bootstrap resamples per (system, term)


def _exact_discrete_mi(x, y):
    """Exact MI from contingency table on discrete-valued data."""
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
    """Detect integer-lattice data with small cardinality."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    sample_idx = np.arange(min(200, n))
    sample = arr[sample_idx]
    if not np.allclose(sample, np.round(sample), atol=1e-9):
        return False
    for j in range(arr.shape[1]):
        if len(np.unique(arr[:, j])) > max_unique_per_dim:
            return False
    return True


def _knn_mi(x, y, k=5):
    """MI estimator. Dispatches to exact contingency-table MI for
    integer-lattice data (KSG fails on discrete inputs), KSG-1 for
    continuous. v0.82.0.4. See run_split_pca_selection._knn_mi for
    full rationale."""
    from scipy.spatial import cKDTree
    from sklearn.neighbors import NearestNeighbors
    from scipy.special import digamma
    n = x.shape[0]
    if n < k + 1:
        return float('nan')

    if _is_integer_valued(x) and _is_integer_valued(y):
        return _exact_discrete_mi(x, y)

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
    return float(digamma(k) + digamma(n) - np.mean(digamma(n_x) + digamma(n_y)))


def _load_operating_point():
    """Load operating point from selection.json. Returns
    (split_a_frac, pca_dim) or fallback (0.7, 32) on missing."""
    if not os.path.exists(SELECTION_PATH):
        print(f"  selection.json not found at {SELECTION_PATH} — "
              f"using fallback (split=0.7, pca=32)")
        return 0.7, 32
    try:
        with open(SELECTION_PATH, 'r', encoding='utf-8-sig') as f:
            sel = json.load(f)
    except Exception as e:
        print(f"  selection.json load failed: {e} — using fallback")
        return 0.7, 32
    sp = sel.get('selected_operating_point') or {}
    split = sp.get('split', [0.7, 0.3])
    pca = sp.get('pca', 32)
    return float(split[0]), int(pca)


def _v5e_terms(n_total=20000, alpha=0.5, seed=42):
    """Generate V5e and return per-term truth (kNN-MI on full n).
    Two terms: I(Y; E), I(Y; E, S_prev)."""
    from run_split_pca_selection import _generate_v5e_for_grid, _pca_reduce
    S_prev, E, S_next = _generate_v5e_for_grid(n_total, alpha=alpha, seed=seed)
    X_full = np.hstack([E, S_prev])
    truth = {
        'I(Y;E)':    _knn_mi(X_full[:, :1], S_next, k=5),
        'I(Y;E,S)':  _knn_mi(X_full,         S_next, k=5),
    }
    return X_full, S_next, truth


def _v5f_terms(n_total=20000, sigma_S=0.5, sigma_Y=0.3, seed=42):
    """Generate V5f and return per-term truth (kNN-MI on full n).
    Three terms: I(Y; E), I(Y; E, C), I(Y; E, C, S)."""
    from run_split_pca_selection import _generate_v5f_for_grid
    E, C, S, Y = _generate_v5f_for_grid(n_total, sigma_S=sigma_S,
                                          sigma_Y=sigma_Y, seed=seed)
    X_full = np.hstack([E, C, S])
    truth = {
        'I(Y;E)':       _knn_mi(X_full[:, :1], Y, k=5),
        'I(Y;E,C)':     _knn_mi(X_full[:, :2], Y, k=5),
        'I(Y;E,C,S)':   _knn_mi(X_full,         Y, k=5),
    }
    return X_full, Y, truth


def _bootstrap_term(X, y, term_label, idx_func_x, n_subsample, n_bootstrap,
                     truth_value, seed=42):
    """Subsample n_subsample rows from (X, y) WITHOUT REPLACEMENT
    n_bootstrap times, compute kNN-MI on each subsample (using
    idx_func_x to slice X to the columns this term depends on), and
    report rel-error vs truth_value.

    v0.82.0.4: changed from rng.choice(n, replace=True) to
    rng.choice(n, replace=False). The pre-patch with-replacement form
    was bootstrapping in the textbook statistics sense, but it created
    duplicate points in the subsample, which breaks KSG: duplicate
    points have zero pairwise distance, so eps in joint space collapses
    and the marginal counts within eps explode. The result was wildly
    biased estimates — V5f I(Y;E) reported 190% relative error in the
    field at n_subsample=16000 with replacement, which was not a real
    estimator-instability finding but a duplicate-point artifact.

    Without-replacement subsampling is what the apparatus actually does
    in practice (it splits its data, not bootstraps it), so this is
    also the more relevant reliability signal.
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if n_subsample > n:
        n_subsample = n
    rel_errors = []
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n_subsample, replace=False)
        Xb = idx_func_x(X[idx])
        yb = y[idx]
        mi_b = _knn_mi(Xb, yb, k=max(3, min(5, n_subsample // 200)))
        if not np.isfinite(mi_b) or not np.isfinite(truth_value) or abs(truth_value) < 1e-6:
            continue
        rel_errors.append(abs(mi_b - truth_value) / abs(truth_value))
    if not rel_errors:
        return {'median_rel_err': float('nan'),
                'p95_rel_err':    float('nan'),
                'n_valid':         0}
    return {
        'median_rel_err': float(np.median(rel_errors)),
        'p95_rel_err':    float(np.percentile(rel_errors, 95)),
        'n_valid':         len(rel_errors),
        'truth':           float(truth_value),
    }


def _hash_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_cache_key(out_dir):
    script_hash   = _hash_file(os.path.abspath(__file__))
    v5_hash       = _hash_file(os.path.join(ROOT, 'v5_synthetic_calibration.py'))
    sel_hash      = _hash_file(SELECTION_PATH)
    cache_key = {
        'script':        'run_knn_mi_reliability.py',
        'script_hash':   script_hash,
        'upstream': {
            'v5_synthetic_calibration.py': v5_hash,
            'split_pca_selection/selection.json': sel_hash,
        },
    }
    p = os.path.join(out_dir, '.cache_key.json')
    with open(p, 'w') as f:
        json.dump(cache_key, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=OUT_DIR, help='output directory')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP,
                        help='bootstrap resamples per term')
    parser.add_argument('--n-total', type=int, default=20000,
                        help='full sample size (truth proxy)')
    parser.add_argument('--quick', action='store_true',
                        help='small bootstrap + small n for fast smoke test')
    args = parser.parse_args()

    if args.quick:
        args.n_bootstrap = 3
        args.n_total = 800

    os.makedirs(args.out, exist_ok=True)
    split_a_frac, pca_dim = _load_operating_point()
    n_subsample = int(split_a_frac * args.n_total)
    print("=" * 60)
    print("Run 0056 — kNN-MI reliability check (knn_mi_reliability)")
    print(f"  operating point: split_A = {split_a_frac:.2f}, pca_dim = {pca_dim}")
    print(f"  n_subsample = {n_subsample} (per bootstrap)")
    print(f"  n_bootstrap = {args.n_bootstrap}")
    print("=" * 60)

    out = {
        'description': (
            "kNN-MI estimator reliability under bootstrap resampling at "
            "the operating-point sample size and PCA dim from "
            "run_split_pca_selection. Per-term median + p95 relative "
            "error vs full-n proxy truth (V5e) or analytic-derived "
            "truth-proxy at full n (V5f)."
        ),
        'operating_point': {
            'split_a_frac': split_a_frac,
            'pca_dim':      pca_dim,
            'n_subsample':  n_subsample,
        },
        'systems': {},
        'metadata': {
            'iota_version':  '0.82.0.9',
            'timestamp':     datetime.datetime.utcnow().isoformat() + 'Z',
            'n_total':       args.n_total,
            'n_bootstrap':   args.n_bootstrap,
        },
    }

    # ── V5e ─────────────────────────────────────────────────────
    print("\n[V5e] computing truth at full n...")
    t0 = time.time()
    X_v5e, y_v5e, truth_v5e = _v5e_terms(n_total=args.n_total, seed=42)
    print(f"  truth: {truth_v5e}")
    print(f"  ({time.time()-t0:.1f}s)")
    print(f"[V5e] bootstrap reliability per term ({args.n_bootstrap} resamples)...")
    v5e_terms_out = {}
    for term, truth in truth_v5e.items():
        if term == 'I(Y;E)':
            idx_func = lambda X: X[:, :1]
        else:
            idx_func = lambda X: X
        block = _bootstrap_term(X_v5e, y_v5e, term, idx_func,
                                  n_subsample=n_subsample,
                                  n_bootstrap=args.n_bootstrap,
                                  truth_value=truth, seed=42)
        v5e_terms_out[term] = block
        print(f"  {term:>15}: median rel err = "
              f"{block['median_rel_err']:.4f}, p95 = {block['p95_rel_err']:.4f}")
    out['systems']['v5e'] = v5e_terms_out

    # ── V5f ─────────────────────────────────────────────────────
    print("\n[V5f] computing truth at full n...")
    t0 = time.time()
    X_v5f, y_v5f, truth_v5f = _v5f_terms(n_total=args.n_total, seed=42)
    print(f"  truth: {truth_v5f}")
    print(f"  ({time.time()-t0:.1f}s)")
    print(f"[V5f] bootstrap reliability per term ({args.n_bootstrap} resamples)...")
    v5f_terms_out = {}
    for term, truth in truth_v5f.items():
        if term == 'I(Y;E)':
            idx_func = lambda X: X[:, :1]
        elif term == 'I(Y;E,C)':
            idx_func = lambda X: X[:, :2]
        else:
            idx_func = lambda X: X
        block = _bootstrap_term(X_v5f, y_v5f, term, idx_func,
                                  n_subsample=n_subsample,
                                  n_bootstrap=args.n_bootstrap,
                                  truth_value=truth, seed=42)
        v5f_terms_out[term] = block
        print(f"  {term:>15}: median rel err = "
              f"{block['median_rel_err']:.4f}, p95 = {block['p95_rel_err']:.4f}")
    out['systems']['v5f'] = v5f_terms_out

    # ── Cross-system summary ─────────────────────────────────────
    all_meds = []
    all_p95s = []
    for sys_block in out['systems'].values():
        for tb in sys_block.values():
            if np.isfinite(tb.get('median_rel_err', float('nan'))):
                all_meds.append(tb['median_rel_err'])
            if np.isfinite(tb.get('p95_rel_err', float('nan'))):
                all_p95s.append(tb['p95_rel_err'])
    out['summary'] = {
        'overall_median_rel_err': float(np.median(all_meds)) if all_meds else float('nan'),
        'overall_p95_rel_err':    float(np.median(all_p95s)) if all_p95s else float('nan'),
        'n_terms':                 len(all_meds),
    }
    print("\n" + "=" * 60)
    print(f"OVERALL median rel err: {out['summary']['overall_median_rel_err']:.4f}")
    print(f"OVERALL median p95:     {out['summary']['overall_p95_rel_err']:.4f}")
    print("=" * 60)

    out_path = os.path.join(args.out, 'reliability.json')
    # v0.83.2: atomic write for knn_mi reliability output
    from results_schema import atomic_json_dump
    atomic_json_dump(out, out_path, indent=2)
    _write_cache_key(args.out)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
