"""
_ridge_partition_probe.py -- why does the Run 0043 Ridge permutation partition collapse on de-duplicated rows?

On the April r1-23 caches (d=64, 24,000 rows, the row set the released Q0043 numbers were computed on)
compute the in-sample permutation effects with the released estimator (Ridge alpha=0.01 on standardized
[E, C, S, prompt_tokens]) under three conditions:
  (a) all rows as released (duplicates in)
  (b) distinct rows only (one per byte-identical (S, E, S', C))
  (c) distinct rows, alpha chosen by 5-fold cross-validation (RidgeCV over 1e-3 .. 1e4)
and report E/C/R effects, fractions, and the in-sample / 5-fold held-out R^2 of each fit.

Usage: py _ridge_partition_probe.py   (writes data/paper/calibration/ridge_partition_probe.json)
"""
import os, sys, json, glob, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import analysis
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

N_PERM = 25
CELLS = ['data/gemma/2b_4bit/abliterated/deterministic', 'data/gemma/2b_4bit/abliterated/temp_0.4', 'data/gemma/2b_4bit/abliterated/temp_1.0',
         'data/gemma/2b_fp16/abliterated/temp_1.0', 'data/gemma/9b_4bit/abliterated/temp_0.6', 'data/llama/8b_4bit/abliterated/deterministic']


def rows_from_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {k: np.array(z[k]) for k in ('has_real_ct', 'e_t', 's_prev', 'c_t', 's_next', 'prompt_tokens', 'run_nums')}   # NpzFile is lazy: index once
    z.close()
    m = d['has_real_ct'].astype(bool)
    rows = [{'e_t': d['e_t'][i], 's_prev': d['s_prev'][i], 'c_t': d['c_t'][i], 's_next': d['s_next'][i],
             'prompt_tokens': float(d['prompt_tokens'][i]), 'run_num': int(d['run_nums'][i])} for i in np.where(m)[0]]
    return rows


def fit(rows, alpha):
    Xe = np.stack([r['e_t'] for r in rows]); Xs = np.stack([r['s_prev'] for r in rows]); Xc = np.stack([r['c_t'] for r in rows])
    y = np.stack([r['s_next'] for r in rows]); Xp = np.array([r['prompt_tokens'] for r in rows], dtype=np.float32).reshape(-1, 1)
    Xp[np.isnan(Xp)] = float(np.nanmean(Xp))
    X = np.hstack([Xe, Xc, Xs, Xp]); sx = StandardScaler(); sy = StandardScaler()
    Xz = sx.fit_transform(X); yz = sy.fit_transform(y)
    if alpha == 'cv':
        mdl = RidgeCV(alphas=np.logspace(-3, 4, 15), cv=KFold(5, shuffle=True, random_state=0)).fit(Xz, yz); a = float(mdl.alpha_)
        mdl = Ridge(alpha=a).fit(Xz, yz)
    else:
        a = float(alpha); mdl = Ridge(alpha=a).fit(Xz, yz)
    r2_in = float(r2_score(yz, mdl.predict(Xz)))
    r2_cv = []
    for tr, te in KFold(5, shuffle=True, random_state=1).split(Xz):
        m2 = Ridge(alpha=a).fit(Xz[tr], yz[tr]); r2_cv.append(r2_score(yz[te], m2.predict(Xz[te])))
    return mdl, sx, sy, a, r2_in, float(np.mean(r2_cv)), float(np.linalg.norm(mdl.coef_))


def effects(rows, mdl, sx, sy):
    out = {}
    for comp, seed in (('E', 99), ('C', 100), ('S_prev', 101)):
        eff, sd, _ = analysis._permutation_sensitivity_on(rows, comp, mdl, sx, sy=sy, n_perm=N_PERM, seed=seed)
        out[comp] = eff
    tot = sum(out.values())
    return {'effect_E': out['E'], 'effect_C': out['C'], 'effect_R': out['S_prev'],
            'frac_E': out['E'] / tot, 'frac_C': out['C'] / tot, 'frac_R': out['S_prev'] / tot}


def main():
    res = {'n_perm': N_PERM, 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'cells': {}}
    for cell in CELLS:
        stash = sorted(glob.glob(f'{cell}/hidden_states/_qcache_ct_d64_r1-23.npz.stash_*'))
        if not stash:
            print(cell, 'no April stash cache'); continue
        rows = rows_from_npz(stash[0])
        keys = {}
        for r in rows:
            k = (r['s_prev'].tobytes(), r['e_t'].tobytes(), r['s_next'].tobytes(), r['c_t'].tobytes())
            keys.setdefault(k, r)
        drows = list(keys.values())
        rec = {'n_rows': len(rows), 'n_distinct': len(drows), 'n_distinct_C': int(len(np.unique(np.stack([r['c_t'] for r in rows]), axis=0)))}
        for label, rr, alpha in (('a_all_rows_alpha0.01', rows, 0.01), ('b_distinct_alpha0.01', drows, 0.01), ('c_distinct_alphaCV', drows, 'cv')):
            t0 = time.time(); mdl, sx, sy, a, r2_in, r2_cv, wn = fit(rr, alpha)
            e = effects(rr, mdl, sx, sy); e.update({'alpha': a, 'r2_in_sample': r2_in, 'r2_5fold': r2_cv, 'coef_norm': wn, 'n': len(rr), 'secs': round(time.time() - t0, 1)})
            rec[label] = e
            print(f"{cell.split('data/')[1]:40s} {label:22s} n={len(rr):5d} alpha={a:8.3g} r2_in={r2_in:.3f} r2_cv={r2_cv:.3f} |w|={wn:8.1f}  E/C/R = {e['frac_E']:.3f}/{e['frac_C']:.3f}/{e['frac_R']:.3f}  effects {e['effect_E']:.2f}/{e['effect_C']:.2f}/{e['effect_R']:.2f}", flush=True)
        res['cells'][cell] = rec
    os.makedirs('data/paper/calibration', exist_ok=True)
    json.dump(res, open('data/paper/calibration/ridge_partition_probe.json', 'w'), indent=1)
    print('wrote data/paper/calibration/ridge_partition_probe.json')


if __name__ == '__main__':
    main()
