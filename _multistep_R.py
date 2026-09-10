"""
_multistep_R.py -- multi-step persistence R_k on the fleet (R note §7, made empirical).

For each cell and lag k in {1, 2, 3}:
    R_k = I(S_{t+k}; S_t | E_t, ..., E_{t+k-1}) / I(S_{t+k}; S_t, E_t, ..., E_{t+k-1})
computed with the Gaussian log-determinant estimator of _gauss_ii_fleet.py on de-duplicated
tuples (S_t, E_t..E_{t+k-1}, S_{t+k}) built from consecutive turns of the same (run, trial).
k = 1 is the Gaussian version of the apparatus's R (two channels, no C). The decay of R_k
with k is the trajectory's persistence beyond the intervening inputs.

Also reported: the unconditional share I(S_{t+k}; S_t) / I(S_{t+k}; S_t, E-block), and n distinct.
Pre-registered prediction (recorded 2026-09-10, before any fleet number was computed): Gemma 2B FP16 shows the slowest decay of R_k.

Usage: py _multistep_R.py            -> data/paper/calibration/multistep_R_fleet.json
"""
import sys, os, time, json, glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

OUT_PATH = os.path.join(ROOT, 'data', 'paper', 'calibration', 'multistep_R_fleet.json')
RUNS = [6, 7, 8, 13, 14, 15, 1, 3, 23]
D = 8
KS = (1, 2, 3)


def _slogdet(C, idx):
    return np.linalg.slogdet(C[np.ix_(idx, idx)])[1]


def _mi(C, a, b):
    return 0.5 * (_slogdet(C, a) + _slogdet(C, b) - _slogdet(C, a + b))


def _mi_cond(C, a, b, z):
    """I(A;B|Z) = I(A;B,Z) - I(A;Z)."""
    return _mi(C, a, b + z) - _mi(C, a, z)


def multistep(S0, Eblocks, Sk, d=D):
    """S0: (n, w); Eblocks: list of k arrays (n, w); Sk: (n, w). PCA-d each block separately."""
    d = int(min(d, S0.shape[1], max(2, S0.shape[0] // (4 * (len(Eblocks) + 2)))))
    P = lambda X: PCA(n_components=d, random_state=42).fit_transform(X) if X.shape[1] > d else X
    blocks = [P(S0)] + [P(E) for E in Eblocks] + [P(Sk)]
    Z = np.hstack(blocks).astype(np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    C = LedoitWolf().fit(Z).covariance_
    sizes = [b.shape[1] for b in blocks]
    off = np.cumsum([0] + sizes)
    s0 = list(range(off[0], off[1]))
    e = list(range(off[1], off[-2]))
    sk = list(range(off[-2], off[-1]))
    I_sk_s0_e = _mi(C, sk, s0 + e)
    I_sk_s0_given_e = _mi_cond(C, sk, s0, e)
    I_sk_s0 = _mi(C, sk, s0)
    return dict(R_k=I_sk_s0_given_e / I_sk_s0_e if I_sk_s0_e > 0 else float('nan'),
                unconditional_share=I_sk_s0 / I_sk_s0_e if I_sk_s0_e > 0 else float('nan'),
                I_joint=I_sk_s0_e, I_cond=I_sk_s0_given_e, d=d, n=int(S0.shape[0]))


def build_tuples(rows, k):
    """Consecutive-turn chains within (run, trial): S_t = rows[t].s_prev, E_t..E_{t+k-1}, S_{t+k} = rows[t+k-1].s_next."""
    by = defaultdict(dict)
    for r in rows:
        by[(r['run_num'], r['trial'])][r['turn']] = r
    S0, Es, Sk = [], [[] for _ in range(k)], []
    for key, turns in by.items():
        for t in sorted(turns):
            chain = [turns.get(t + j) for j in range(k)]
            if any(c is None for c in chain):
                continue
            S0.append(chain[0]['s_prev'])
            for j in range(k):
                Es[j].append(chain[j]['e_t'])
            Sk.append(chain[-1]['s_next'])
    if not S0:
        return None
    S0 = np.stack(S0).astype(np.float32); Sk = np.stack(Sk).astype(np.float32)
    Es = [np.stack(E).astype(np.float32) for E in Es]
    # de-duplicate on the full tuple
    _, u = np.unique(np.hstack([S0] + Es + [Sk]), axis=0, return_index=True); u = np.sort(u)
    return S0[u], [E[u] for E in Es], Sk[u], int(len(S0))


def main():
    import analysis
    analysis._qcache_save = lambda *a, **k: None
    analysis.POOL_DIM = 1024
    out = {'definition': 'R_k = I(S_{t+k}; S_t | E_t..E_{t+k-1}) / I(S_{t+k}; S_t, E_t..E_{t+k-1}); Gaussian log-det, PCA-%d per block, Ledoit-Wolf; de-duplicated tuples' % D,
           'runs': RUNS, 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'cells': {}}
    for cell_dir in sorted(glob.glob('data/*/*/abliterated/*')):
        parts = cell_dir.replace('\\', '/').split('/')
        if len(parts) != 5 or not os.path.isdir(f'{cell_dir}/hidden_states') or parts[2] == '2b_8bit':
            continue
        fam, size, var, tdir = parts[1:]
        key = f"{fam}_{size}_{var}_" + ('T0.0' if tdir == 'deterministic' else 'T' + tdir.split('_')[1])
        t0 = time.time()
        rows = analysis._load_quadruplets(f'{cell_dir}/hidden_states', RUNS, '', require_ct=True)
        rows = [r for r in rows if r['has_real_ct'] and r['c_t'] is not None]
        rec = {'n_rows': len(rows), 'runs_present': sorted({r['run_num'] for r in rows})}
        for k in KS:
            built = build_tuples(rows, k)
            if built is None:
                rec[f'k{k}'] = None; continue
            S0, Es, Sk, n_all = built
            r = multistep(S0, Es, Sk)
            r['n_tuples_all'] = n_all
            rec[f'k{k}'] = {kk: (float(v) if isinstance(v, (float, np.floating)) else v) for kk, v in r.items()}
        rec['seconds'] = round(time.time() - t0)
        out['cells'][key] = rec
        print(key, {f'k{k}': (round(rec[f'k{k}']['R_k'], 3), rec[f'k{k}']['n']) for k in KS if rec.get(f'k{k}')}, flush=True)
    json.dump(out, open(OUT_PATH, 'w'), indent=1)
    print('Wrote', OUT_PATH)


if __name__ == '__main__':
    main()
