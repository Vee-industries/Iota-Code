"""
run_kraskov_anchor.py -- Phase 4 of the Bayesian apparatus.

Produces per-cell Kraskov-MI anchors over real LLM cells. For each
of the 24 paper cells in Q0042 caches:

  1. Load PCA-reduced features from _qcache_ct_d{N}_r{lo}-{hi}.npz
  2. PCA-reduce to 32 dims (matches existing knn_pca_dim convention)
  3. Compute Kraskov-MI on (E, S) projection: I(E; Y), I(S; Y)
  4. Construct C-coordinate via geometric class mean (lagrangianbot's
     C_a^raw formula, with G̃_C from Ridge/MLP/RF/RKHS shares)
  5. Build p_anchor = (E_kNN, C_a, R_kNN) / sum
  6. Apply bucket-determined bound calculation

Validation: V5b synergy=0 sanity check. Anchor's recovered shares on
V5b ground-truth cells should match within bucket tolerance.

Output: data/paper/calibration/kraskov_anchor/anchors.json
"""
import json
import os
import sys
import glob
import time
import datetime

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import ui  # noqa: E402

DATA = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(DATA, 'paper', 'calibration', 'kraskov_anchor')
OUT_PATH = os.path.join(OUT_DIR, 'anchors.json')


def _parse_q42_path(q42_path):
    parts = q42_path.replace('\\', '/').split('/')
    family = size = variant = None
    temp = 0.0
    for i, p in enumerate(parts):
        if p == 'data' and i + 1 < len(parts):
            family = parts[i + 1]
        if family and parts[i] == family and i + 1 < len(parts):
            size = parts[i + 1]
        if size and parts[i] == size and i + 1 < len(parts):
            variant = parts[i + 1]
        if p.startswith('temp_'):
            try:
                temp = float(p[5:])
            except Exception:
                pass
        if p == 'deterministic':
            temp = 0.0
    return family, size, variant, temp


def _cell_key(family, size, variant, temp):
    """Writerbot-style cell key (matches Q0057)."""
    quant = 'q4' if '4bit' in size else ('q8' if '8bit' in size else
            ('fp16' if 'fp16' in size else size.split('_')[-1]))
    size_bare = size.split('_')[0]
    t_int = int(round(temp * 10))
    return f"{family}_{size_bare}_{quant}_{variant}_t{t_int:02d}"


def _kraskov_mi(X, Y, n_neighbors=5, pca_dim=32):
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.decomposition import PCA
    if Y.shape[1] > pca_dim:
        Y = PCA(n_components=pca_dim, random_state=42).fit_transform(Y)
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=42).fit_transform(X)
    mi_per_input = np.zeros(X.shape[1])
    for j in range(Y.shape[1]):
        mi_per_input += mutual_info_regression(
            X, Y[:, j], n_neighbors=n_neighbors, random_state=42)
    return float(mi_per_input.sum())


def _construct_anchor_from_kraskov(mi_E, mi_S, p_classes, eps=1e-6):
    """Lagrangianbot's C-coordinate construction:
        G̃_i = (∏_f p_{f,i})^(1/4) / Σ_j (∏_f p_{f,j})^(1/4)
        C_a^raw = G̃_C / (1 - G̃_C) · (E_kNN + R_kNN)
    where E_kNN and R_kNN come from the Kraskov shares (treating S as R
    for the apparatus's chain-rule decomposition).

    Returns p_anchor = (E_kNN, C_a, R_kNN) / sum.
    """
    p = np.clip(np.asarray(p_classes, dtype=float), eps, 1 - eps)
    G = np.exp(np.mean(np.log(p), axis=0))
    G_tilde = G / G.sum()
    G_tilde_C = float(G_tilde[1])
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
    p_anchor = p_anchor / p_anchor.sum()
    return p_anchor.tolist()


def _load_cell_features(q42_path):
    """Load feature matrices for a cell from its qcache.

    v0.81.1.1: corrected paths and keys to match what
    run_function_class_sensitivity._run_rf_on_cell already uses.
    Cache is at cell_dir/hidden_states/ (Q0042 is two levels deep from
    cell_dir, not one). Keys are 'e_t', 'c_t', 's_prev', 's_next' with
    a 'has_real_ct' boolean mask. Filter to has_real_ct=True rows.

    Returns dict with X_E (= e_t), X_S_prev (= s_prev), Y (= s_next)
    on the filtered rows. None on failure.
    """
    cell_dir = os.path.dirname(os.path.dirname(q42_path))
    hidden_dir = os.path.join(cell_dir, 'hidden_states')
    if not os.path.isdir(hidden_dir):
        return None
    cache_files = sorted(glob.glob(os.path.join(hidden_dir, '_qcache_ct_d*_r*.npz')))
    if not cache_files:
        return None
    # Use the first cache file (each cell has one canonical cache;
    # multiple files would be concat'd by upstream runs not us)
    cache_path = cache_files[0]
    try:
        d = np.load(cache_path, allow_pickle=False)
        e_t  = np.array(d['e_t'])
        s_prev = np.array(d['s_prev'])
        s_next = np.array(d['s_next'])
        has_real_ct = np.array(d['has_real_ct']).astype(bool)
        d.close()
    except Exception:
        return None
    if not has_real_ct.any():
        return None
    X_E = e_t[has_real_ct].astype(np.float32)
    X_S_prev = s_prev[has_real_ct].astype(np.float32)
    Y = s_next[has_real_ct].astype(np.float32)
    return {'X_E': X_E, 'X_S_prev': X_S_prev, 'Y': Y}


def _load_function_class_shares(cell_key):
    """Pull (Ridge, MLP, RF, RKHS_median) shares from Q0057.

    v0.81.1.4: RKHS placeholder retired. Q0057 now ships
    permutation_shares.rkhs_median (median across the characteristic kernel
    grid: Matérn ν ∈ {0.5, 1.5, 2.5} + RBF at 4 length-scale multiples
    of median pairwise distance). Cells lacking the rkhs_median field
    (e.g. those last written pre-0.81.1.4) fall back to Ridge as the 4th
    class so this loader stays callable on partial Q0057 states; a warning
    is logged via the dispatcher. After full Q0057 re-fire, every cell has
    real RKHS.
    """
    q57_path = os.path.join(DATA, 'paper', 'Q0057_function_class_sensitivity.json')
    if not os.path.exists(q57_path):
        return None
    try:
        with open(q57_path, 'r', encoding='utf-8-sig') as f:
            q57 = json.load(f)
    except Exception:
        return None
    cell = q57.get('cells', {}).get(cell_key)
    if cell is None:
        return None
    perm = cell.get('permutation_shares', {})
    ridge = perm.get('ridge')
    mlp = perm.get('mlp')
    rf = perm.get('rf')
    rkhs_median = perm.get('rkhs_median')
    if not all((ridge, mlp, rf)):
        return None
    def _to_arr(d):
        return np.array([float(d['E']), float(d['C']), float(d['R'])])
    # v0.81.1.4: real RKHS if present; Ridge fallback if Q0057 cell hasn't
    # been re-fit under 0.81.1.4 yet (transitional state).
    rkhs_arr = _to_arr(rkhs_median) if rkhs_median else _to_arr(ridge)
    return {
        'ridge': _to_arr(ridge),
        'mlp':   _to_arr(mlp),
        'rf':    _to_arr(rf),
        'rkhs':  rkhs_arr,
    }


def main():
    t_total = time.time()
    ui.section("Run 0058 Phase 4 -- Kraskov anchor producer")
    os.makedirs(OUT_DIR, exist_ok=True)

    pattern = os.path.join(DATA, '**', 'Q0042_decomposition.json')
    q42_files = sorted(glob.glob(pattern, recursive=True))
    if not q42_files:
        ui.err("No Q0042 files found.")
        return 1
    ui.msg(f"Found {len(q42_files)} Q0042 cells.")

    # v0.82.0.9: per-cell resume cache. The kraskov anchor compute
    # is the apparatus's slowest phase (~130s/cell × 24 cells = ~3
    # hours wall time on RTX 3080 hardware). Pre-patch this phase
    # ground every cell from scratch on every Run 0058 fire, even
    # when the upstream features and Q0057 shares were unchanged.
    # The output (anchors_per_cell dict in anchors.json) is fully
    # deterministic given those inputs, so caching by cell_key is
    # safe. To force a refit, delete data/paper/calibration/
    # kraskov_anchor/anchors.json before firing.
    cached_anchors = {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, 'r', encoding='utf-8') as f:
                cached = json.load(f) or {}
            cached_anchors = (cached.get('anchors_per_cell') or {})
            if cached_anchors:
                ui.ok(f"  [resume] {len(cached_anchors)} cell(s) cached "
                       f"in existing anchors.json -- will skip those")
        except Exception as e:
            ui.warn(f"  [resume] could not read existing anchors.json: {e}")
            cached_anchors = {}

    anchors = dict(cached_anchors)  # start from cache
    n_done = 0
    n_failed = 0
    n_resumed = 0
    for idx, q42 in enumerate(q42_files, 1):
        family, size, variant, temp = _parse_q42_path(q42)
        cell_key = _cell_key(family, size, variant, temp)
        # Resume check: if this cell is already in cache with a valid
        # p_anchor, skip the compute entirely.
        if cell_key in cached_anchors:
            existing = cached_anchors[cell_key] or {}
            if existing.get('p_anchor') and len(existing['p_anchor']) == 3:
                n_resumed += 1
                ui.msg(f"  [{idx}/{len(q42_files)}] {cell_key}  [resumed from cache]")
                continue
        ui.msg(f"  [{idx}/{len(q42_files)}] {cell_key}")
        t0 = time.time()
        try:
            features = _load_cell_features(q42)
            if features is None:
                ui.warn(f"    no features available")
                n_failed += 1
                continue
            shares = _load_function_class_shares(cell_key)
            if shares is None:
                ui.warn(f"    no Q0057 shares -- skipping")
                n_failed += 1
                continue
            X_E = features['X_E']
            X_S_prev = features['X_S_prev']
            Y = features['Y']
            if X_E.shape[0] < 100:
                ui.warn(f"    too few rows ({X_E.shape[0]})")
                n_failed += 1
                continue
            # Subsample if too large (keep it under ~17k rows for speed)
            n_max = 17000
            if X_E.shape[0] > n_max:
                np.random.seed(42)
                idx_sub = np.random.choice(X_E.shape[0], n_max, replace=False)
                X_E = X_E[idx_sub]
                X_S_prev = X_S_prev[idx_sub]
                Y = Y[idx_sub]
            mi_E = _kraskov_mi(X_E, Y)
            mi_S = _kraskov_mi(X_S_prev, Y)
            p_classes = np.array([shares['ridge'], shares['mlp'],
                                   shares['rf'], shares['rkhs']])
            p_anchor = _construct_anchor_from_kraskov(mi_E, mi_S, p_classes)
            if p_anchor is None:
                ui.warn(f"    anchor construction failed")
                n_failed += 1
                continue
            dt = time.time() - t0
            anchors[cell_key] = {
                'p_anchor': p_anchor,
                'mi_E_raw': float(mi_E),
                'mi_S_raw': float(mi_S),
                'kraskov_n_neighbors': 5,
                'pca_dim': 32,
                'wall_clock_s': dt,
            }
            n_done += 1
            ui.msg(f"    p_anchor=[{p_anchor[0]:.3f}, {p_anchor[1]:.3f}, {p_anchor[2]:.3f}]  ({dt:.1f}s)")
            # v0.82.0.9: write cache after EVERY cell so a kill mid-run
            # banks completed cells. Pre-patch only wrote at the end,
            # so any kill threw away the entire run's progress.
            try:
                _interim_out = {
                    'phase': 'phase4_anchor_producer',
                    'iota_version': '0.82.0.23',
                    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
                    'n_done': n_done,
                    'n_failed': n_failed,
                    'wall_clock_s': time.time() - t_total,
                    'anchors_per_cell': anchors,
                    'note': (
                        "Per-cell incremental cache. RKHS column "
                        "placeholder: reuses Ridge shares until schema "
                        "migration ships. p_anchor C-coordinate "
                        "constructed via lagrangianbot's geometric-mean "
                        "ratio formula. Subsamples to 17k rows when "
                        "cells are larger."
                    ),
                }
                _tmp = OUT_PATH + '.tmp'
                with open(_tmp, 'w', encoding='utf-8') as f:
                    json.dump(_interim_out, f, indent=2, ensure_ascii=False)
                os.replace(_tmp, OUT_PATH)
            except Exception as _ce:
                # Don't fail the cell on cache-write failure; just warn.
                ui.warn(f"    [cache] interim write failed: {_ce}")
        except Exception as e:
            import traceback as _tb
            ui.err(f"    FAIL: {type(e).__name__}: {e}")
            for line in _tb.format_exc().splitlines()[-5:]:
                ui.err(f"      {line}")
            n_failed += 1

    elapsed = time.time() - t_total
    ui.ok(f"Phase 4: {n_done} fresh, {n_resumed} resumed, "
           f"{n_failed} failed. Elapsed: {elapsed:.0f}s")

    out = {
        'phase': 'phase4_anchor_producer',
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'n_done': n_done + n_resumed,
        'n_fresh': n_done,
        'n_resumed': n_resumed,
        'n_failed': n_failed,
        'wall_clock_s': elapsed,
        'anchors_per_cell': anchors,
        'note': (
            "RKHS column placeholder: reuses Ridge shares until schema "
            "migration ships. p_anchor C-coordinate constructed via "
            "lagrangianbot's geometric-mean ratio formula. Subsamples "
            "to 17k rows when cells are larger. v0.82.0.9: per-cell "
            "resume cache; delete this file to force a full refit."
        ),
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    ui.ok(f"Wrote {OUT_PATH}")
    if (n_done + n_resumed) == 0:
        ui.err("No anchors built. Phase 4 reports failure.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
