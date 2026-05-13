"""
run_channel_marginal.py — empirical cross-check on channel-marginal nonlinearity.

For each (model, temperature) cell with a completed Q0042_decomposition.json,
fit four models:
  - ridge_E : Ridge on [Xe, Xp] -> S_next
  - ridge_S : Ridge on [Xs, Xp] -> S_next
  - mlp_E   : MLP on [Xe, Xp] -> S_next
  - mlp_S   : MLP on [Xs, Xp] -> S_next

Xp (prompt_tokens) is included in every fit as a never-permuted control,
matching the paper's Ridge protocol in analysis._run_decomposition.

Per-cell derived metrics:
  gap_E     = r2_mlp_E - r2_ridge_E
  gap_S     = r2_mlp_S - r2_ridge_S
  asymmetry = gap_E - gap_S

Plus free harvest from Q0042.interaction_info.knn.pooled:
  r2_knn_E, r2_knn_S, knn_asymmetry = r2_knn_E - r2_knn_S

Usage:
  python run_channel_marginal.py
  python run_channel_marginal.py --out custom.csv
  python run_channel_marginal.py --force

Output:
  ./data/paper/calibration/channel_marginal/channel_marginal_nonlinearity.csv

Protocol frozen to match analysis.py linearity_check exactly.
Runtime: ~30s-2min per cell CPU. 24 cells in ~30-45 min.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

# v0.80.0.20: force UTF-8 stdout/stderr on Windows. Default Python on
# Windows uses cp1252 for stdout, which crashes on Greek letters,
# arrows, and pictographs (⚠ → σ ρ etc.) that calibration scripts use.
# Reconfigure to UTF-8 with replacement chars as final safety net.
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    import io as _io
    if hasattr(_sys.stdout, 'buffer'):
        _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(_sys.stderr, 'buffer'):
        _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
OUT_DIR  = os.path.join(DATA, 'paper', 'calibration', 'channel_marginal')
OUT_PATH = os.path.join(OUT_DIR, 'channel_marginal_nonlinearity.csv')


def _parse_path(q42_path):
    rel = os.path.relpath(q42_path, DATA).replace('\\', '/').split('/')
    if len(rel) < 6:
        return None
    family, size, variant, cond = rel[0], rel[1], rel[2], rel[3]
    if cond == 'deterministic':
        temp = 0.0
    elif cond.startswith('temp_'):
        try:    temp = float(cond.replace('temp_', ''))
        except: temp = None
    else:
        temp = None
    return family, size, variant, cond, temp


def _load_q42(q42_path):
    with open(q42_path, 'r', encoding='utf-8-sig') as f:
        raw = f.read()
    raw = re.sub(r'\bNaN\b', 'null', raw)
    raw = re.sub(r'\bInfinity\b', 'null', raw)
    raw = re.sub(r'\b-Infinity\b', 'null', raw)
    return json.loads(raw)


def _knn_rowdata(q42_json):
    ii = q42_json.get('interaction_info') or {}
    knn = ii.get('knn') or {}
    pooled = knn.get('pooled') or {}
    if not isinstance(pooled, dict) or 'r2_E' not in pooled or 'r2_S' not in pooled:
        return None, None, None
    r2_e = pooled.get('r2_E'); r2_s = pooled.get('r2_S')
    r2_e = float(r2_e) if r2_e is not None else None
    r2_s = float(r2_s) if r2_s is not None else None
    asym = (r2_e - r2_s) if (r2_e is not None and r2_s is not None) else None
    return r2_e, r2_s, asym


def _compute_cell(q42_path):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    q42 = _load_q42(q42_path)
    # v0.80.0.17: pool_dim is not stored in Q0042 schema. The original
    # script attempted q42.get('pool_dim') and failed every cell with
    # "pool_dim missing from Q0042" because that field has never been
    # written. Source of truth is analysis.POOL_DIM (the global the
    # paper's analyses ran under). Optional Q0042 override retained
    # in case future schemas add the field.
    sys.path.insert(0, ROOT)
    import analysis as _ana

    # v0.80.0.18: detect POOL_DIM and source_runs from the cell's existing
    # qcache file. Channel_marginal previously used hardcoded
    # SOURCE_RUNS_3WAY = [1,3,6,7,8,13,14,15,23] (r1-23) and got POOL_DIM
    # from analysis.POOL_DIM (typically 1024). But each cell's cache was
    # built by Q0042/Q0043 at the cell-specific POOL_DIM (varies 64-1024
    # per cell) and source_runs r3-28 in many cases. Filename mismatch
    # → cache miss → ~2 min cold-load per cell × 24 cells = 48 min wasted.
    #
    # Fix: peek at the cache file already in hidden_dir, parse its
    # filename to recover the POOL_DIM and run range that produced it,
    # use those values when calling _load_quadruplets so it hits cache.
    cell_dir   = os.path.dirname(os.path.dirname(q42_path))
    hidden_dir = os.path.join(cell_dir, 'hidden_states')

    pool_dim = None
    source_runs_min_max = None
    cache_glob = os.path.join(hidden_dir, '_qcache_ct_d*_r*.npz')
    cache_files = sorted(glob.glob(cache_glob))
    if cache_files:
        # Parse `_qcache_ct_d{POOL_DIM}_r{rmin}-{rmax}.npz`
        m = re.match(r'_qcache_ct_d(\d+)_r(\d+)-(\d+)\.npz$',
                     os.path.basename(cache_files[0]))
        if m:
            pool_dim = int(m.group(1))
            source_runs_min_max = (int(m.group(2)), int(m.group(3)))

    if pool_dim is None:
        # Fallback: Q0042 (future schema), then analysis.POOL_DIM global.
        pool_dim = q42.get('pool_dim') or _ana.POOL_DIM
        if pool_dim is None:
            return {'error': 'pool_dim unavailable (no qcache, no Q0042 field, no analysis.POOL_DIM)'}

    r2_knn_E, r2_knn_S, knn_asym = _knn_rowdata(q42)

    saved_pool_dim = _ana.POOL_DIM
    _ana.POOL_DIM = int(pool_dim)

    try:
        mn_candidate = None
        for pfx in ('R0003_', 'R0001_'):
            for fp in sorted(glob.glob(os.path.join(hidden_dir, f'{pfx}*_trial*_turn01.npy'))):
                base = os.path.basename(fp)
                if '_trial' in base and base.startswith(pfx):
                    head = base.split('_trial')[0]
                    mn_candidate = head[len(pfx):]
                    break
            if mn_candidate:
                break
        if mn_candidate is None:
            return {'error': f'no hidden state files under {hidden_dir}'}

        # v0.80.0.18: pick source_runs to match the cache filename. The
        # 3-way decomposition needs runs that were originally collected;
        # cache was built using whichever run set the cell's Q0042 used.
        # Use the cache's run range to pick the matching default set.
        # If cache says r1-23, use the legacy 3WAY set [6,7,8,13,14,15,1,3,23].
        # If cache says r3-28, use the canonical post-renumber set
        # [3,6,7,8,13,14,15,23,28].
        SOURCE_RUNS_3WAY_LEGACY = [6, 7, 8, 13, 14, 15, 1, 3, 23]      # r1-23
        SOURCE_RUNS_3WAY_RENUM  = [3, 6, 7, 8, 13, 14, 15, 23, 28]     # r3-28
        if source_runs_min_max == (1, 23):
            source_runs = SOURCE_RUNS_3WAY_LEGACY
        elif source_runs_min_max == (3, 28):
            source_runs = SOURCE_RUNS_3WAY_RENUM
        else:
            # Unknown range — try renumbered set as best guess
            source_runs = SOURCE_RUNS_3WAY_RENUM

        rows = _ana._load_quadruplets(
            hidden_dir, source_runs, mn_candidate, require_ct=True)
        if not rows:
            return {'error': 'no quadruplets loaded'}

        Xe = np.stack([r['e_t']    for r in rows]).astype(np.float32)
        Xs = np.stack([r['s_prev'] for r in rows]).astype(np.float32)
        Xp = np.array([r.get('prompt_tokens', 0) for r in rows],
                      dtype=np.float32).reshape(-1, 1)
        y  = np.stack([r['s_next'] for r in rows]).astype(np.float32)
        n = Xe.shape[0]

        Xe_Xp = np.hstack([Xe, Xp])
        Xs_Xp = np.hstack([Xs, Xp])

        Xe_tr, Xe_te, y_tr, y_te = train_test_split(
            Xe_Xp, y, test_size=0.2, random_state=42)
        Xs_tr, Xs_te, _,    _    = train_test_split(
            Xs_Xp, y, test_size=0.2, random_state=42)

        sxE = StandardScaler().fit(Xe_tr)
        sxS = StandardScaler().fit(Xs_tr)
        sy  = StandardScaler().fit(y_tr)

        Xe_tr_s = sxE.transform(Xe_tr); Xe_te_s = sxE.transform(Xe_te)
        Xs_tr_s = sxS.transform(Xs_tr); Xs_te_s = sxS.transform(Xs_te)
        y_tr_s  = sy.transform(y_tr);   y_te_s  = sy.transform(y_te)

        ridge_E = Ridge(alpha=0.01).fit(Xe_tr_s, y_tr_s)
        ridge_S = Ridge(alpha=0.01).fit(Xs_tr_s, y_tr_s)
        r2_ridge_E = float(r2_score(y_te_s, ridge_E.predict(Xe_te_s),
                                    multioutput='variance_weighted'))
        r2_ridge_S = float(r2_score(y_te_s, ridge_S.predict(Xs_te_s),
                                    multioutput='variance_weighted'))

        mlp_cfg = dict(hidden_layer_sizes=(256,), activation='tanh',
                       early_stopping=True, validation_fraction=0.1,
                       max_iter=500, random_state=42, tol=1e-5)
        mlp_E = MLPRegressor(**mlp_cfg).fit(Xe_tr_s, y_tr_s)
        mlp_S = MLPRegressor(**mlp_cfg).fit(Xs_tr_s, y_tr_s)
        r2_mlp_E = float(r2_score(y_te_s, mlp_E.predict(Xe_te_s),
                                  multioutput='variance_weighted'))
        r2_mlp_S = float(r2_score(y_te_s, mlp_S.predict(Xs_te_s),
                                  multioutput='variance_weighted'))

        gap_E = r2_mlp_E - r2_ridge_E
        gap_S = r2_mlp_S - r2_ridge_S
        asymmetry = gap_E - gap_S

        return {
            'pool_dim':    int(pool_dim),
            'n_rows':      int(n),
            'r2_ridge_E':  r2_ridge_E,
            'r2_ridge_S':  r2_ridge_S,
            'r2_mlp_E':    r2_mlp_E,
            'r2_mlp_S':    r2_mlp_S,
            'gap_E':       gap_E,
            'gap_S':       gap_S,
            'asymmetry':   asymmetry,
            'r2_knn_E':    r2_knn_E,
            'r2_knn_S':    r2_knn_S,
            'knn_asymmetry': knn_asym,
        }
    finally:
        _ana.POOL_DIM = saved_pool_dim


def _load_existing(out_path):
    if not os.path.exists(out_path):
        return {}
    existing = {}
    with open(out_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row.get('family'), row.get('size'),
                 row.get('variant'), row.get('condition'))
            existing[k] = row
    return existing


def _write_csv(out_path, rows, cols):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--root', default=DATA)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"! data root not found: {args.root}", file=sys.stderr)
        return 1

    pattern = os.path.join(args.root, '**', 'Q0042_decomposition.json')
    q42_files = sorted(glob.glob(pattern, recursive=True))
    if not q42_files:
        print(f"! no Q0042 files found under {args.root}", file=sys.stderr)
        return 1
    print(f"Found {len(q42_files)} Q0042 cell(s).")

    existing = {} if args.force else _load_existing(args.out)
    if existing and not args.force:
        print(f"  Resuming with {len(existing)} pre-computed cell(s).")

    cols = ['family', 'size', 'variant', 'condition', 'temperature',
            'pool_dim', 'n_rows',
            'r2_ridge_E', 'r2_ridge_S', 'r2_mlp_E', 'r2_mlp_S',
            'gap_E', 'gap_S', 'asymmetry',
            'r2_knn_E', 'r2_knn_S', 'knn_asymmetry', 'notes']

    rows_out = list(existing.values())
    t0 = time.time()
    n_new = n_skipped = n_failed = 0
    n_parse_failed = 0   # v0.80.0.17: track parse failures separately

    # v0.80.0.17: write CSV header even before any cells process, so a
    # crash mid-loop or all-skip outcome still leaves an output file.
    # Previous behavior: _write_csv only called on first successful row.
    # If all cells were parse-skipped, no CSV was ever written, manifest
    # got stamped on empty output, downstream consumers saw missing data.
    _write_csv(args.out, rows_out, cols)

    for idx, q42 in enumerate(q42_files, 1):
        parsed = _parse_path(q42)
        if parsed is None:
            n_parse_failed += 1
            n_skipped += 1
            print(f"[{idx}/{len(q42_files)}] PARSE FAIL: {q42}", flush=True)
            continue
        family, size, variant, cond, temp = parsed
        key = (family, size, variant, cond)
        if key in existing:
            n_skipped += 1; continue

        t_cell = time.time()
        print(f"[{idx}/{len(q42_files)}] {family}/{size}/{variant}/{cond} "
              f"(T={temp}) ... ", end='', flush=True)
        try:
            result = _compute_cell(q42)
        except Exception as e:
            result = {'error': f'{type(e).__name__}: {e}'}

        if 'error' in result:
            print(f"FAIL ({result['error']})", flush=True)
            n_failed += 1
            row = {'family': family, 'size': size, 'variant': variant,
                   'condition': cond, 'temperature': temp,
                   'notes': result['error']}
            for c in cols:
                row.setdefault(c, None)
        else:
            dt = time.time() - t_cell
            print(f"done ({dt:.1f}s)  gap_E={result['gap_E']:+.3f}  "
                  f"gap_S={result['gap_S']:+.3f}  "
                  f"asym={result['asymmetry']:+.3f}", flush=True)
            n_new += 1
            row = {'family': family, 'size': size, 'variant': variant,
                   'condition': cond, 'temperature': temp, 'notes': '',
                   **result}
        rows_out.append(row)
        _write_csv(args.out, rows_out, cols)

    elapsed = time.time() - t0
    print(f"\nDone. {n_new} new, {n_skipped} skipped "
          f"({n_parse_failed} parse-failed), {n_failed} failed. "
          f"Elapsed: {elapsed:.0f}s")
    print(f"Wrote {args.out} ({len(rows_out)} total rows)")

    # v0.80.0.17: fail loud if no successful cells. Manifest should not
    # stamp on an output file with zero usable data.
    if n_new == 0 and not existing:
        print(f"\n! ERROR: no cells produced output. "
              f"{n_parse_failed} parse failures, {n_failed} compute failures. "
              f"Manifest will NOT be stamped (orchestrator detects rc=1).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
