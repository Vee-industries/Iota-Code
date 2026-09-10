"""
_gauss_ii_fleet.py -- Gaussian (log-determinant) interaction information on de-duplicated fleet rows.

Written 2026-09-10 for the trilogy consistency pass. The apparatus's kNN-R^2 interaction-information
statistic (analysis._knn_ii, the source of ii_fraction_knn / H4) and the KSG estimator
(run_knn_mi_reliability._knn_mi) both fail a sign test on V5b, where S_t and E_t are independent
inputs so the redundancy-positive interaction information I(S;Y) + I(E;Y) - I(S,E;Y) is <= 0.
This estimator passes that test, passes synergy/redundancy controls, and is unaffected by row
duplication because it works from the (shrunk) covariance. Blind spot: purely nonlinear synergy
(XOR, products) reads exactly 0, so it detects redundancy against linear synergy only.

Definition: PCA-d on S_t, E_t, S_t+1 separately; standardize; Ledoit-Wolf covariance of the
concatenation; Gaussian MI from log-determinants; fraction = II / I(S,E;Y).

Usage: py _gauss_ii_fleet.py validate   -> prints the V5b / control validation
       py _gauss_ii_fleet.py fleet      -> writes data/paper/calibration/gauss_ii_fleet.json
                                           (distinct (S,E,S') rows per cell, d = 4/8/16/32; never writes qcache files)
"""
import sys, os, time, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

OUT_PATH = os.path.join(ROOT, 'data', 'paper', 'calibration', 'gauss_ii_fleet.json')
LEGACY_RUNS = [6, 7, 8, 13, 14, 15, 1, 3, 23]   # the r1-23 apparatus row set (run 23 absent on disk since 2026-09)


def _mi_gauss(C, a, b):
    ab = a + b
    sa = np.linalg.slogdet(C[np.ix_(a, a)])[1]
    sb = np.linalg.slogdet(C[np.ix_(b, b)])[1]
    sab = np.linalg.slogdet(C[np.ix_(ab, ab)])[1]
    return 0.5 * (sa + sb - sab)


def gauss_ii(S, E, Y, d=8, shrink=True):
    d = int(min(d, S.shape[1], E.shape[1], Y.shape[1], max(2, S.shape[0] // 4)))
    P = lambda X: PCA(n_components=d, random_state=42).fit_transform(X) if X.shape[1] > d else X
    Sp, Ep, Yp = P(S), P(E), P(Y)
    Z = np.hstack([Sp, Ep, Yp]).astype(np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    C = LedoitWolf().fit(Z).covariance_ if shrink else np.cov(Z, rowvar=False)
    ds, de, dy = Sp.shape[1], Ep.shape[1], Yp.shape[1]
    s = list(range(ds)); e = list(range(ds, ds + de)); y = list(range(ds + de, ds + de + dy))
    I_S, I_E, I_SE = _mi_gauss(C, s, y), _mi_gauss(C, e, y), _mi_gauss(C, s + e, y)
    II = I_S + I_E - I_SE
    return dict(I_S=I_S, I_E=I_E, I_SE=I_SE, II=II,
                II_frac=II / I_SE if I_SE > 0 else float('nan'), d=d, n=len(S))


def _v5b(a_s, b_s, dim=8, n_steps=20000, noise=0.1, seed=42):
    """Byte-for-byte the V5b generator in v5_synthetic_calibration.run_v5b."""
    rng = np.random.RandomState(seed)
    A_raw = rng.randn(dim, dim); A_raw = A_raw / np.max(np.abs(np.linalg.eigvals(A_raw))); A = a_s * A_raw
    B_raw = rng.randn(dim, dim); B = b_s * B_raw / np.linalg.norm(B_raw)
    S = np.zeros((n_steps + 1, dim)); E = rng.randn(n_steps, dim); S[0] = rng.randn(dim) * 0.1
    for t in range(n_steps):
        S[t + 1] = np.tanh(A @ S[t] + B @ E[t]) + rng.randn(dim) * noise
    return S[:-1], E, S[1:]


def validate():
    print('V5b: truth II_red <= 0 (independent S, E)')
    for label, a_s, b_s in [('E-dominant', 0.2, 0.8), ('balanced', 0.5, 0.5), ('S-dominant', 0.8, 0.2), ('strong-S', 0.95, 0.1)]:
        S, E, Y = _v5b(a_s, b_s)
        for n in (20000, 1000, 200):
            r = gauss_ii(S[:n], E[:n], Y[:n], 8)
            print(f"  {label:11s} n={n:5d}: II_frac={r['II_frac']:+.3f}")
        idx = np.repeat(np.arange(2000), 9)
        print(f"  {label:11s} 2000 rows x9 duplicated: II_frac={gauss_ii(S[idx], E[idx], Y[idx], 8)['II_frac']:+.3f}")
    rng = np.random.RandomState(7); n = 20000
    S = rng.randn(n, 8); E = rng.randn(n, 8)
    print(f"XOR-like synergy control (blind spot, reads 0): {gauss_ii(S, E, np.sign(S) * np.sign(E) + 0.1 * rng.randn(n, 8), 8)['II_frac']:+.3f}")
    print(f"additive independent control (truth negative): {gauss_ii(S, E, S + E + 0.1 * rng.randn(n, 8), 8)['II_frac']:+.3f}")
    S = rng.randn(n, 8); E = 0.7 * S + 0.7 * rng.randn(n, 8)
    print(f"partial-redundancy control (truth positive): {gauss_ii(S, E, S + 0.3 * rng.randn(n, 8), 8)['II_frac']:+.3f}")
    S = rng.randn(n, 8); E = S + 0.1 * rng.randn(n, 8)
    print(f"redundancy control (truth positive): {gauss_ii(S, E, S + 0.1 * rng.randn(n, 8), 8)['II_frac']:+.3f}")


def fleet():
    import analysis
    analysis._qcache_save = lambda *a, **k: None      # never write cache files into the repo
    analysis.POOL_DIM = 1024
    out = {'estimator': 'gauss_ii: PCA-d per block, Ledoit-Wolf covariance, log-det MI; II_frac = (I_S + I_E - I_SE) / I_SE, redundancy-positive',
           'rows': 'apparatus row set (LEGACY_RUNS) rebuilt with analysis._load_quadruplets at POOL_DIM 1024; distinct = byte-identical (S,E,S) triples',
           'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'cells': {}}
    for cell_dir in sorted(glob.glob('data/*/*/abliterated/*')):
        parts = cell_dir.replace('\\', '/').split('/')
        if len(parts) != 5 or not os.path.isdir(f'{cell_dir}/hidden_states') or parts[2] == '2b_8bit':
            continue
        fam, size, var, tdir = parts[1:]
        key = f"{fam}_{size}_{var}_" + ('T0.0' if tdir == 'deterministic' else 'T' + tdir.split('_')[1])
        rows = analysis._load_quadruplets(f'{cell_dir}/hidden_states', LEGACY_RUNS, '', require_ct=True)
        rows = [r for r in rows if r['has_real_ct'] and r['c_t'] is not None]
        S = np.stack([r['s_prev'] for r in rows]).astype(np.float32)
        E = np.stack([r['e_t'] for r in rows]).astype(np.float32)
        Y = np.stack([r['s_next'] for r in rows]).astype(np.float32)
        _, u = np.unique(np.hstack([S, E, Y]), axis=0, return_index=True); u = np.sort(u)
        rec = {'n_rows': len(rows), 'n_distinct': int(len(u))}
        for d in (4, 8, 16, 32):
            rec[f'd{d}'] = {k: float(v) for k, v in gauss_ii(S[u], E[u], Y[u], d).items()}
            rec[f'd{d}_allrows'] = {'II_frac': float(gauss_ii(S, E, Y, d)['II_frac'])}
        out['cells'][key] = rec
        print(key, {k: round(v['II_frac'], 3) for k, v in rec.items() if isinstance(v, dict)}, flush=True)
    json.dump(out, open(OUT_PATH, 'w'), indent=1)
    print('Wrote', OUT_PATH)


if __name__ == '__main__':
    (validate if (sys.argv[1:] or ['fleet'])[0] == 'validate' else fleet)()
