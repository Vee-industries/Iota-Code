"""
_simple_baseline_partial_r2.py -- paper B section 6.7 simple-baseline check, recomputed on the de-duplicated caches.

Per cell: Ridge (alpha by ridge_policy) of S_{t+1} on [E_t, C_t, S_t] vs on [E_t, C_t], partition-B 80/20 split
(random_state 42, as the May file), partial R^2 for S_t = (R2_full - R2_reduced) / (1 - R2_reduced) on the held-out
split. Writes data/paper/v26_response/simple_baseline_partial_r2_dedup.json and prints per-family ranges.
"""
import os, sys, json, glob, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import ridge_policy as rp

FAM = [('llama/8b_4bit', 'LLaMA 8B Q4'), ('gemma/9b_4bit', 'Gemma 9B Q4'), ('gemma/2b_fp16', 'Gemma 2B FP16'), ('gemma/2b_4bit', 'Gemma 2B Q4')]
TDIR = ['deterministic', 'temp_0.2', 'temp_0.4', 'temp_0.6', 'temp_0.8', 'temp_1.0']


def load(cache):
    z = np.load(cache, allow_pickle=False); d = {k: np.array(z[k]) for k in ('has_real_ct', 'e_t', 's_prev', 'c_t', 's_next')}; z.close()
    m = d['has_real_ct'].astype(bool)
    return d['e_t'][m].astype(np.float32), d['c_t'][m].astype(np.float32), d['s_prev'][m].astype(np.float32), d['s_next'][m].astype(np.float32)


out = {'method': 'partition-B 80/20 (random_state 42), Ridge alpha by ridge_policy (' + rp.policy_label() + '), partial R2 = (R2_full - R2_reduced)/(1 - R2_reduced) on held-out',
       'rows': 'de-duplicated nine-run apparatus caches (_qcache_ct_d*_r1-23_dedup.npz)', 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'per_cell': {}}
for fam, lab in FAM:
    vals = []
    for td in TDIR:
        cell = f'data/{fam}/abliterated/{td}'; cache = sorted(glob.glob(f'{cell}/hidden_states/_qcache_ct_d*_r1-23_dedup.npz'))
        key = f"{fam.replace('/', '_')}_abliterated_T{'0.0' if td == 'deterministic' else td.split('_')[1]}"
        if not cache: vals.append(None); continue
        E, Cc, S, Y = load(cache[0])
        idx = np.arange(len(Y)); tr, te = train_test_split(idx, test_size=0.2, random_state=42)
        def fit_r2(X):
            sx = StandardScaler().fit(X[tr]); sy = StandardScaler().fit(Y[tr])
            mdl = rp.fit_ridge(sx.transform(X[tr]), sy.transform(Y[tr]), tag='baseline' + str(X.shape[1]))
            return float(r2_score(sy.transform(Y[te]), mdl.predict(sx.transform(X[te])), multioutput='variance_weighted')), float(mdl.alpha)
        r_full, a_full = fit_r2(np.hstack([E, Cc, S])); r_red, a_red = fit_r2(np.hstack([E, Cc]))
        pr2 = None if r_red >= 1 else (r_full - r_red) / (1 - r_red)
        out['per_cell'][key] = {'n_rows': int(len(Y)), 'r2_full': r_full, 'r2_reduced': r_red, 'partial_r2_S': pr2, 'alpha_full': a_full, 'alpha_reduced': a_red}
        vals.append(pr2)
        print(f"  {key:34s} n={len(Y):5d} r2_full={r_full:.3f} r2_red={r_red:.3f} partial_R2(S)={pr2 if pr2 is None else round(pr2,3)} alpha={a_full:g}", flush=True)
    v = [x for x in vals if x is not None]
    print(f"{lab:14s} partial R2 at T>=0.2: {min(v[1:]):.2f} - {max(v[1:]):.2f}; at T=0.0: {v[0]:.2f}")
os.makedirs('data/paper/v26_response', exist_ok=True)
json.dump(out, open('data/paper/v26_response/simple_baseline_partial_r2_dedup.json', 'w'), indent=1)
print('wrote data/paper/v26_response/simple_baseline_partial_r2_dedup.json')
