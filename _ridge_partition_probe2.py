"""
_ridge_partition_probe2.py -- isolate what drives the Run 0043 Ridge-share collapse on the new de-duplicated caches.

For each listed cell: load the nine-run '_dedup' cache, report per-run distinct E / C / S counts, then compute the
released estimator's in-sample permutation partition (Ridge alpha=0.01) on (i) all distinct rows, (ii) without run 23,
(iii) without run 3, (iv) all rows with a cross-validated alpha. Writes data/paper/calibration/ridge_partition_probe2.json.
"""
import os, sys, json, glob, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
os.environ['IOTA_DEDUP_ROWS'] = '1'
import _ridge_partition_probe as P

CELLS = ['data/llama/8b_4bit/abliterated/temp_1.0', 'data/gemma/2b_4bit/abliterated/temp_1.0', 'data/gemma/9b_4bit/abliterated/temp_0.6', 'data/gemma/2b_fp16/abliterated/temp_0.2']


def main():
    res = {'cells': {}, 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    for cell in CELLS:
        cache = sorted(glob.glob(f'{cell}/hidden_states/_qcache_ct_d*_r1-23_dedup.npz'))
        if not cache: print(cell, 'no dedup cache'); continue
        rows = P.rows_from_npz(cache[0])
        rec = {'cache': os.path.basename(cache[0]), 'n_rows': len(rows), 'per_run': {}}
        for rn in sorted(set(r['run_num'] for r in rows)):
            rr = [r for r in rows if r['run_num'] == rn]
            rec['per_run'][rn] = {'n': len(rr), 'distinct_E': int(len(np.unique(np.stack([r['e_t'] for r in rr]), axis=0))),
                                  'distinct_C': int(len(np.unique(np.stack([r['c_t'] for r in rr]), axis=0))), 'distinct_S': int(len(np.unique(np.stack([r['s_prev'] for r in rr]), axis=0)))}
        print(cell, rec['cache'], 'n=', len(rows)); print('   per run:', rec['per_run'], flush=True)
        for label, rr, alpha in (('i_all_alpha0.01', rows, 0.01), ('ii_no_run23_alpha0.01', [r for r in rows if r['run_num'] != 23], 0.01),
                                 ('iii_no_run3_alpha0.01', [r for r in rows if r['run_num'] != 3], 0.01), ('iv_all_alphaCV', rows, 'cv')):
            mdl, sx, sy, a, r2_in, r2_cv, wn = P.fit(rr, alpha)
            e = P.effects(rr, mdl, sx, sy); e.update({'alpha': a, 'r2_in_sample': r2_in, 'r2_5fold': r2_cv, 'coef_norm': wn, 'n': len(rr)})
            rec[label] = e
            print(f"   {label:24s} n={len(rr):5d} alpha={a:8.3g} r2_in={r2_in:.3f} r2_cv={r2_cv:.3f} |w|={wn:8.1f}  E/C/R = {e['frac_E']:.3f}/{e['frac_C']:.3f}/{e['frac_R']:.3f}  effects {e['effect_E']:.2f}/{e['effect_C']:.2f}/{e['effect_R']:.2f}", flush=True)
        res['cells'][cell] = rec
    json.dump(res, open('data/paper/calibration/ridge_partition_probe2.json', 'w'), indent=1)
    print('wrote data/paper/calibration/ridge_partition_probe2.json')


if __name__ == '__main__':
    main()
