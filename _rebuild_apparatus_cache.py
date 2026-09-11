"""
_rebuild_apparatus_cache.py -- (re)build one cell's apparatus quadruplet cache from the raw
per-(trial, turn) .npy files, at the cell's calibrated pool_dim (Q0041), on the nine-run
apparatus row set [6, 7, 8, 13, 14, 15, 1, 3, 23] (the r1-23 cache family Run 0057 and
Run 0058 read). Honours IOTA_DEDUP_ROWS=1 (writes the '_dedup' cache family).

Usage: py _rebuild_apparatus_cache.py <cell_dir>      e.g. data/gemma/2b_4bit/abliterated/temp_0.4
Prints: rows, pool_dim, cache file name.
"""
import os, sys, glob

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
APPARATUS_RUNS = [6, 7, 8, 13, 14, 15, 1, 3, 23]


def main(cell_dir):
    import analysis
    ana_dir = os.path.join(cell_dir, 'analysis'); hidden_dir = os.path.join(cell_dir, 'hidden_states')
    dim = analysis.load_optimal_pool_dim(ana_dir)
    if dim is None:
        print(f"ERROR: no Q0041 calibration in {ana_dir}"); return 1
    analysis.POOL_DIM = int(dim)
    rows = analysis._load_quadruplets(hidden_dir, APPARATUS_RUNS, '', require_ct=True)
    n_ct = sum(1 for r in rows if r['has_real_ct'] and r['c_t'] is not None)
    caches = sorted(os.path.basename(p) for p in glob.glob(os.path.join(hidden_dir, '_qcache_*.npz')))
    print(f"CACHE rows={len(rows)} rows_with_ct={n_ct} pool_dim={dim} dedup={analysis._dedup_mode()} caches={caches}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
