"""
IOTA FRAMEWORK — ANALYSIS
===========================
All no-GPU analysis runs. Operates on saved .npy hidden states and .csv data.
No model loading. Dispatched by start_here.py.

Run map:
  25  — Granger probe A (T=0.0 null data from Run 0001)
  27  — Granger probe B (full dataset from Run 0003)
  32  — Baseline swap + permutation test
  33  — Canonical E+C+R decomposition (supersedes 25 and 27)
  34  — Permutation sensitivity partition + held-out validation (H22 + H28)
  40  — Pooled cross-directory E+C+R decomposition (run once after all temp rounds)
"""

import os, sys, glob, json

# ── CPU cap (v0.71.0.3) ─────────────────────────────────────────────────────
# Analysis runs are CPU-bound (Ridge fits, permutation shuffles, bootstrap).
# Cap thread count to leave headroom for Windows and other processes.
# Must be set BEFORE numpy/sklearn import.
_NCPU = os.cpu_count() or 4
_ANALYSIS_THREADS = str(max(1, _NCPU - 2))  # leave 2 cores free
for _tvar in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_tvar, _ANALYSIS_THREADS)

# On Windows, set process priority to BELOW_NORMAL so the OS scheduler
# prefers interactive apps (explorer, browser, dashboard) over analysis.
if sys.platform == 'win32':
    try:
        import ctypes
        _BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        _handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(_handle, _BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

def _find_root():
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(candidate, "start_here.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.dirname(os.path.abspath(__file__))

ROOT = _find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from cartography import get_paths, load_hidden_states, load_embedding, sanitize, run_prefix, run_mode_mask, run_mode_mask_any, run_id_pad
from orchestration_core import _check_pause
import ui


# ── Analysis status (v0.71.0.2) ─────────────────────────────────────────────
# Writes to .iota_status.json so the dashboard can show analysis progress.
# Separate from orchestration_core._write_status which is for data collection.

def _analysis_status(run_num, step, pct=None, detail=None, temperature=None):
    """Write analysis progress to the dashboard status file."""
    import json, time
    try:
        from cartography import _find_root
        status = {
            "analysis_mode": True,
            "run_num": run_num,
            "run_label": f"Q{run_num}",
            "analysis_step": step,
            "analysis_pct": pct,
            "analysis_detail": detail,
            "temperature": temperature,
            "ts": time.time(),
        }
        with open(os.path.join(_find_root(), ".iota_status.json"), 'w') as f:
            json.dump(status, f)
    except Exception:
        pass


def _analysis_marker(ana_dir, run_num, start=True):
    """Write or remove a .running marker for scanner to detect in-progress analysis."""
    from cartography import ANALYSIS_JSON
    json_name = ANALYSIS_JSON.get(run_num)
    if not json_name:
        return
    marker = os.path.join(ana_dir, json_name.replace('.json', '.running'))
    if start:
        try:
            with open(marker, 'w') as f:
                f.write(str(run_num))
        except Exception:
            pass
    else:
        try:
            if os.path.exists(marker):
                os.remove(marker)
        except Exception:
            pass


N_PERMUTE = 200
ALPHA     = 0.05
POOL_DIM  = 64
N_PERMUTE_SOBOL = 200  # Run 0043 — permutation sensitivity count
GRANGER_POOL_DIM  = 64  # Granger probes (Runs 0047, 0048) use fixed 64-dim pooling.
                         # Model B feature width = 2 × GRANGER_POOL_DIM (X_e + X_s).
                         # Run 0047 ~1200 triplets: 1200/128 = 9.4:1 sample ratio.
                         # At 128: 1200/256 = 4.7:1 — still degenerate (R²=1.0).
                         # Does not affect Runs 0042, 0043, 0051 which use global POOL_DIM.
GRANGER_MAX_TURNS = 14  # Turn ceiling for Granger matrix construction.
                         # Current source runs are ≤13 turns; ceiling is benign now.
                         # Raise this constant if any future source run exceeds 13 turns.
QUADRUPLET_MAX_TURNS = 20  # Turn ceiling for _load_quadruplets inner loop. (v29.0)
                            # Raise if any SOURCE_RUNS_3WAY/2WAY run ever exceeds 19 turns.
                            # Separate from GRANGER_MAX_TURNS — used only in _load_quadruplets.


# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════

def _pool(vec, n=None):
    if n is None:
        n = POOL_DIM
    arr = np.array(vec, dtype=np.float32)
    if len(arr) <= n:
        return arr
    return arr[np.linspace(0, len(arr)-1, n).astype(int)]


def _fit_ols(X, y):
    """True OLS via LinearRegression with a collinearity guard.
    If condition number exceeds 1e6, logs a warning but proceeds — results
    should be interpreted with caution in that case. BUG 15 fix: original
    used Ridge(alpha=0.01) which biases R² estimates relative to OLS
    predictions in IOTA_Hypotheses_v10.md.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y)
    cond = float(np.linalg.cond(Xs))
    if cond > 1e6:
        print(f"  [WARN] _fit_ols: condition number {cond:.2e} — collinear predictors, R² may be unstable")
    return float(LinearRegression().fit(Xs, ys).score(Xs, ys))



# ══════════════════════════════════════════════════════════════════
# RUNS 25 + 27 — GRANGER PROBE
# ══════════════════════════════════════════════════════════════════

def _build_granger_matrix(hidden_dir, source_run, model_name, max_trials=None):
    # v0.79.4.0: accept either int or 4-digit string for source_run — be
    # defensive; caller may pass from an iteration over DualKeyRunSet.
    try:
        source_run_int = int(source_run)
    except (ValueError, TypeError):
        source_run_int = 0
    mn, pfx = sanitize(model_name), run_prefix(source_run_int)
    # v0.79.5.0: search both 4-digit (canonical) and 2-digit (legacy) patterns.
    # Pre-migration data is 2-digit; post-migration is 4-digit.
    # _rid4 (4-digit) tried first; falls through to legacy if no match.
    _rid4 = f"{source_run_int:04d}"
    pattern = os.path.join(hidden_dir, f"{pfx}{_rid4}_{mn}_trial*_turn01.npy")
    trial_files = sorted(glob.glob(pattern))
    if not trial_files:
        pattern = os.path.join(hidden_dir, f"{pfx}{source_run_int:02d}_{mn}_trial*_turn01.npy")
        trial_files = sorted(glob.glob(pattern))
    _effective_mn = model_name
    if not trial_files:
        # Fallback: wildcard for model_name (session name drift). Try both
        # canonical and legacy prefixes.
        fallback4 = os.path.join(hidden_dir, f"{pfx}{_rid4}_*_trial*_turn01.npy")
        fallback2 = os.path.join(hidden_dir, f"{pfx}{source_run_int:02d}_*_trial*_turn01.npy")
        trial_files = sorted(glob.glob(fallback4)) + sorted(glob.glob(fallback2))
        trial_files = [f for f in trial_files
                       if not f.endswith('_alllayers.npy')
                       and '_emb.npy' not in f
                       and '_et_base.npy' not in f
                       and '_constraint.npy' not in f]
        if trial_files:
            # The split() is tolerant to whichever prefix matched (4-digit or 2-digit).
            bn = os.path.basename(trial_files[0])
            # Find the prefix boundary by trial marker location.
            try:
                stem = bn.split('_trial')[0]
                # stem is like "R0003_gemma_..." or "R03_gemma_..."
                # Strip the run-id section up to the first underscore after prefix digits.
                _parts = stem.split('_', 1)
                _actual = _parts[1] if len(_parts) > 1 else mn
                if _actual != mn:
                    _effective_mn = _actual
            except Exception:
                pass
    if not trial_files:
        return None, "no_hidden_states"

    trials = []
    for f in trial_files:
        try:
            trials.append(int(os.path.basename(f).split("_trial")[1].split("_")[0]))
        except Exception:
            continue
    if max_trials:
        trials = trials[:max_trials]

    # v0.58.0.0 FIX-9: E_base lookup matching _load_quadruplets.
    # Try E_base in base sibling dir first, then hidden_dir, then fall back
    # to kind='E' (old proxy), then s_prev proxy as last resort.
    try:
        _cond_dir    = os.path.dirname(hidden_dir)
        _size_dir    = os.path.dirname(os.path.dirname(_cond_dir))
        base_hid_dir = os.path.join(_size_dir, 'base',
                                    os.path.basename(_cond_dir), 'hidden_states')
    except Exception:
        base_hid_dir = hidden_dir

    rows, any_real_e, e_source = [], False, "s_prev_proxy"
    for trial in trials:
        prev_s = None
        for turn in range(1, GRANGER_MAX_TURNS + 1):
            curr_s = load_hidden_states(source_run, _effective_mn, hidden_dir, trial=trial, turn=turn)
            if curr_s is None:
                break
            # Try E_base (base model hidden states) in both directories
            real_e = load_embedding(base_hid_dir, source_run, _effective_mn, trial, turn, kind='E_base')
            if real_e is None:
                real_e = load_embedding(hidden_dir, source_run, _effective_mn, trial, turn, kind='E_base')
            if real_e is None:
                real_e = load_embedding(hidden_dir, source_run, _effective_mn, trial, turn, kind='E')
            if real_e is not None:
                any_real_e = True
                e_t = real_e
            else:
                e_t = prev_s if prev_s is not None else curr_s
            if prev_s is not None:
                rows.append({"trial": trial, "turn": turn,
                             "s_prev": prev_s, "e_t": e_t, "s_next": curr_s})
            prev_s = curr_s

    if not rows:
        return None, "no_triplets"
    return rows, ("real_embeddings" if any_real_e else e_source)


def _granger_test(rows, n_perm=N_PERMUTE, seed=42):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    X_e = np.stack([_pool(r['e_t'],    GRANGER_POOL_DIM) for r in rows])
    X_s = np.stack([_pool(r['s_prev'], GRANGER_POOL_DIM) for r in rows])
    y   = np.stack([_pool(r['s_next'], GRANGER_POOL_DIM) for r in rows])

    sx = StandardScaler()
    X_both = sx.fit_transform(np.hstack([X_e, X_s]))
    X_e_sc = X_both[:, :X_e.shape[1]]
    X_b_sc = X_both

    sy = StandardScaler()
    y_sc = sy.fit_transform(y)

    r2_a = LinearRegression().fit(X_e_sc, y_sc).score(X_e_sc, y_sc)
    r2_b = LinearRegression().fit(X_b_sc, y_sc).score(X_b_sc, y_sc)
    observed_delta = r2_b - r2_a

    # Permutation test: permute S_prev rows, refit model B, build null ΔR² distribution.
    # Uses already-fit scalers (sx, sy) — no refit per permutation.
    # H11 prediction: permutation test p < 0.05. INF-H11-NOPVAL fix (v31.0).
    rng = np.random.default_rng(seed)
    null_deltas = []
    for _ in range(n_perm):
        X_s_perm    = X_s[rng.permutation(len(X_s))]
        X_both_perm = sx.transform(np.hstack([X_e, X_s_perm]))
        r2_perm     = LinearRegression().fit(X_both_perm, y_sc).score(X_both_perm, y_sc)
        null_deltas.append(r2_perm - r2_a)
    p_value = float(np.mean(np.array(null_deltas) >= observed_delta))

    return {
        "r2_external_only":          r2_a,
        "r2_external_plus_internal": r2_b,
        "delta_r2_internal":         observed_delta,
        "p_value":                   p_value,
        "significant":               int(p_value < 0.05 and observed_delta > 0.01),
        "n_triplets":                len(rows),
        "n_perm":                    n_perm,
        "granger_pool_dim":          GRANGER_POOL_DIM,
        "supports_H011":             int(observed_delta > 0.01),  # backward compat
    }


def _run_granger(run_num, session, paths):
    hidden_dir = paths['hidden']
    model_name = session.get('model_name', '')
    os.makedirs(paths['analysis'], exist_ok=True)

    if run_num == 47:
        # v0.79.4.0: was Run 25 (Granger A), src was R19. New: Run 47 sourced from Run 1.
        rows, e_src = _build_granger_matrix(hidden_dir, 1, model_name)
        label    = "A (T=0.0 null)"
        out_file = os.path.join(paths['analysis'], "Q0047_granger_A.json")
    else:
        # v0.79.4.0: was Run 27 (Granger B), srcs [19, 26]. New: Run 48 sourced from [1, 3].
        rows, e_src = [], "unknown"
        for sr in [1, 3]:
            r, es = _build_granger_matrix(hidden_dir, sr, model_name)
            if r:
                rows.extend(r); e_src = es
        label    = "B (full dataset)"
        out_file = os.path.join(paths['analysis'], "Q0048_granger_B.json")

    ui.section(f"Run {run_num} — Granger Probe {label}")
    if not rows:
        choice = ui.srx_prompt(
            "No hidden state data found — source runs have not been collected.\n"
            "  Run 0001 (for Probe A) and/or Run 0003 (for Probe B) must complete first."
        )
        if choice in ('s', 'x'):
            if choice == 'x': raise SystemExit
            return

    ui.msg(f"  {len(rows)} triplets | E_t source: {e_src} | granger_pool_dim: {GRANGER_POOL_DIM}")
    if e_src == "s_prev_proxy":
        ui.warn("Using S_{t-1} as E_t proxy. Rerun source runs for real embeddings.")

    try:
        results = _granger_test(rows)
    except ImportError:
        ui.warn("scikit-learn required."); return

    results.update({"run_num": run_num, "label": label, "e_source": e_src, "model": model_name})
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    ui.msg(f"  R² external only       : {results['r2_external_only']:.4f}")
    ui.msg(f"  R² external + internal : {results['r2_external_plus_internal']:.4f}")
    ui.msg(f"  ΔR²_internal           : {results['delta_r2_internal']:.4f}")
    ui.blank()
    if results['supports_H011']:
        ui.ok("H11 SUPPORTED — internal state causally upstream of next state.")
    else:
        ui.warn("H11 NOT SUPPORTED — ΔR² below threshold.")
    ui.msg(f"  Saved: {out_file}")


# ══════════════════════════════════════════════════════════════════
# RUN 32 — BASELINE SWAP
# ══════════════════════════════════════════════════════════════════

def _run_baseline_swap(session, paths):
    import pandas as pd

    csv_dir  = paths['csv']
    ana_dir  = paths['analysis']
    os.makedirs(ana_dir, exist_ok=True)
    out_file = os.path.join(ana_dir, "Q0049_baseline_swap.json")

    ui.section("Run 0049 — Baseline Swap (no GPU)")
    ui.msg("Loading all CSV data...")

    dfs = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        try:
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            df = df.replace("NA", float('nan'))
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        choice = ui.srx_prompt(
            "No CSV data found — experimental runs have not been collected yet.\n"
            "  Runs 0004-0017 must complete before Run 0049 baseline swap is meaningful."
        )
        if choice in ('s', 'x'):
            if choice == 'x': raise SystemExit
            return

    combined = pd.concat(dfs, ignore_index=True)
    # Coerce run_mode and priming to numeric before filtering —
    # pd.read_csv leaves them as object (string) dtype from CSV files.
    for _col in ('priming', 'run_mode', 'trial', 'turn',
                 'state_similarity_index', 'signal_entropy_ratio', 'layer_sim_mean'):
        if _col in combined.columns:
            combined[_col] = pd.to_numeric(combined[_col], errors='coerce')
    if 'priming' in combined.columns:
        combined = combined[combined['priming'] != 1]

    ui.msg(f"  {len(combined):,} rows")
    ui.blank()

    def _observed_effect(df, metric):
        intro = df[run_mode_mask_any(df['run_mode'], [6,7,8])][metric].dropna()
        null  = df[run_mode_mask_any(df['run_mode'], [4,5,1])][metric].dropna()
        return float(intro.mean() - null.mean()) if len(intro) and len(null) else float('nan')

    # Bug RNG-1 fix (v29.1, corrected v29.2): hardcoded integer seeds per metric.
    # hash() is non-deterministic across Python restarts (PYTHONHASHSEED randomised
    # since Python 3.3), so abs(hash(metric)) % 2**31 gave a different seed every
    # process invocation — equivalent to unseeded. Fixed seeds are assigned here:
    #   state_similarity_index=100, signal_entropy_ratio=101, layer_sim_mean=102
    # _split_half variants offset by 100: 200, 201, 202.
    # Only these three metrics are passed by _run_baseline_swap (Run 0049).
    # If new metrics are ever added to the loop, extend this table.
    _PERM_SEEDS  = {'state_similarity_index': 100, 'signal_entropy_ratio': 101, 'layer_sim_mean': 102}
    _SPLIT_SEEDS = {'state_similarity_index': 200, 'signal_entropy_ratio': 201, 'layer_sim_mean': 202}

    def _permutation_test(df, metric, n_perm=N_PERMUTE):
        rng = np.random.default_rng(_PERM_SEEDS.get(metric, 999))
        sub = df[run_mode_mask_any(df['run_mode'], [6,7,8,4,5,1])][[metric,'run_mode']].dropna()
        if len(sub) < 10:
            return float('nan'), float('nan')
        observed = _observed_effect(sub, metric)
        vals     = sub[metric].values
        labels   = run_mode_mask_any(sub['run_mode'], [6,7,8]).values
        null_d   = []
        for _ in range(n_perm):
            perm_labels = rng.permutation(labels)
            null_d.append(vals[perm_labels].mean() - vals[~perm_labels].mean())
        return observed, float(np.mean(np.abs(null_d) >= np.abs(observed)))

    def _split_half(df, metric):
        rng = np.random.default_rng(_SPLIT_SEEDS.get(metric, 998))
        sub = df[run_mode_mask_any(df['run_mode'], [6,7,8])][['trial',metric]].dropna()
        if len(sub) < 20: return float('nan')
        trials = sub['trial'].unique().copy()
        rng.shuffle(trials)
        mid = len(trials)//2
        h1 = sub[sub['trial'].isin(trials[:mid])][metric].mean()
        h2 = sub[sub['trial'].isin(trials[mid:])][metric].mean()
        return float(abs(h1-h2))

    results = {"run_num": 32, "n_rows": len(combined)}
    for metric in ['state_similarity_index', 'signal_entropy_ratio', 'layer_sim_mean']:
        if metric not in combined.columns:
            results[metric] = {"error": "column_not_found"}; continue
        obs, p = _permutation_test(combined, metric)
        sh     = _split_half(combined, metric)
        results[metric] = {
            "observed_effect": obs, "p_value": p,
            "n_permutations": N_PERMUTE,
            "significant": int(not np.isnan(p) and p < ALPHA),
            "split_half_delta": sh,
        }
        sig = "✓" if results[metric]["significant"] else "✗"
        ui.msg(f"  {metric}: effect={obs:.4f}  p={p:.4f}  {sig}")

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    ui.ok(f"Saved: {out_file}")


# ══════════════════════════════════════════════════════════════════
# RUN 33 — FULL E+C+R DECOMPOSITION
# ══════════════════════════════════════════════════════════════════

# v0.79.4.0: renumbered to new execution-order IDs.
# Old [3,4,5,15,16,17,19,26,28] → new [6,7,8,13,14,15,1,3,23]
SOURCE_RUNS_3WAY = [6, 7, 8, 13, 14, 15, 1, 23]  # v0.82.0.27: Run 0003 removed (temperature-robustness sweep, not chain-rule source)
# v40.0.0: Run 0001 (was R19) now provides per-trial real E_t and real C_t
# following the three-model redesign. It is the canonical 3-way source.

# Old [1,2,6,7,8,9,20] → new [4,5,9,10,11,12,2]
SOURCE_RUNS_2WAY = [4, 5, 9, 10, 11, 12, 2]
# v40.0.0: Run 0001 (was R19) removed from 2WAY (real C_t now).
# v42.3.0: Run 0002 (was R20) added. Runs 0004/0005/0009-0012/0002
# have real E_t but no per-trial C_t — backfilled with global mean C_t
# from Run 0001.

# Old [41] → new [33]
SOURCE_RUNS_VALIDATION = [33]  # was Run 0033 — held-out set for Run 0043 eval


# ── Quadruplet cache (v0.71.0.12) ─────────────────────────────────────────────
# Loading 22K+ quadruplets from individual .npy files takes ~2 minutes on Windows.
# Cache the assembled arrays in a single .npz so subsequent analysis runs at the
# same temperature load in ~1 second. Runs 0042→0043→0044 share the same data.
# Skipped for POOL_DIM > 1024 (Run 0041 full-dim = 1.5GB, not worth caching).
# Cache lives in the hidden_states dir. Cleaned up by start_here.py after each
# temperature's analysis block completes.

_CACHE_MAX_DIM = 1024

def _qcache_path(hidden_dir, source_runs, require_ct):
    tag = 'ct' if require_ct else 'noct'
    rmin, rmax = min(source_runs), max(source_runs)
    return os.path.join(hidden_dir, f'_qcache_{tag}_d{POOL_DIM}_r{rmin}-{rmax}.npz')

def _qcache_load(hidden_dir, source_runs, require_ct):
    """Load quadruplets from cache. Returns list of row dicts or None."""
    if POOL_DIM > _CACHE_MAX_DIM:
        return None
    path = _qcache_path(hidden_dir, source_runs, require_ct)
    if not os.path.exists(path):
        return None
    # v0.76.0.7: mtime freshness check. Prior version only checked source_runs
    # list equality + POOL_DIM in filename. If any source .npy file (hidden
    # states, E_base, constraint) or the R0001 global-mean C_t constant is newer
    # than the cache, return None to force a re-load. Prevents stale quadruplets
    # after ET recovery, Run 0001 Pass 4 re-computation, or any single-run
    # re-collection from silently poisoning Run 0042/34/46 fractions.
    try:
        _cache_mtime = os.path.getmtime(path)
        _npy_pattern = os.path.join(hidden_dir, '*.npy')
        for _npy in glob.glob(_npy_pattern):
            # Skip the cache file itself (not .npy) and other qcache files
            if os.path.basename(_npy).startswith('_qcache_'):
                continue
            if os.path.getmtime(_npy) > _cache_mtime:
                try:
                    os.remove(path)
                except Exception:
                    pass
                return None
        # Also check the base sibling dir for E_base files (ET recovery writes here)
        try:
            _cond_dir    = os.path.dirname(hidden_dir)
            _size_dir    = os.path.dirname(os.path.dirname(_cond_dir))
            _base_hid    = os.path.join(_size_dir, 'base',
                                        os.path.basename(_cond_dir), 'hidden_states')
            if os.path.isdir(_base_hid):
                for _npy in glob.glob(os.path.join(_base_hid, '*_et_base.npy')):
                    if os.path.getmtime(_npy) > _cache_mtime:
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        return None
        except Exception:
            pass
    except Exception:
        pass
    try:
        d = np.load(path, allow_pickle=False)
        cached_runs = sorted(d['source_runs'].tolist())
        if cached_runs != sorted(source_runs):
            os.remove(path)
            return None
        # Force all arrays into memory — NpzFile is lazy, indexing d['key'][i]
        # re-decompresses the full array on every access. At dim=1024 × 22800 rows
        # that's 90K+ decompressions of a 90MB matrix. Load once → index fast.
        _sp = np.array(d['s_prev'])
        _sn = np.array(d['s_next'])
        _et = np.array(d['e_t'])
        _ct = np.array(d['c_t'])
        _hr = np.array(d['has_real_ct'])
        _pt = np.array(d['prompt_tokens'])
        _rn = np.array(d['run_nums'])
        _tr = np.array(d['trials'])
        _tn = np.array(d['turns'])
        d.close()
        n = len(_rn)
        rows = []
        for i in range(n):
            c_t = _ct[i] if _hr[i] else None
            rows.append({
                's_prev': _sp[i], 's_next': _sn[i],
                'e_t': _et[i], 'c_t': c_t,
                'has_real_ct': bool(_hr[i]),
                'ct_source': 'cached',
                'prompt_tokens': float(_pt[i]),
                'run_num': int(_rn[i]),
                'trial': int(_tr[i]),
                'turn': int(_tn[i]),
            })
        ui.ok(f"  [cache] Loaded {n} quadruplets from {os.path.basename(path)} "
              f"({os.path.getsize(path)/1024/1024:.1f}MB)")
        return rows
    except Exception:
        try: os.remove(path)
        except OSError: pass  # v0.83.1: tighten cleanup-loader except
        return None

def _qcache_save(hidden_dir, source_runs, require_ct, rows):
    """Save quadruplets to .npz cache. Cleans up stale caches in same directory."""
    if POOL_DIM > _CACHE_MAX_DIM or not rows:
        return
    path = _qcache_path(hidden_dir, source_runs, require_ct)
    try:
        dim = len(rows[0]['s_prev'])
        c_t = np.zeros((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            if r['c_t'] is not None:
                c_t[i] = r['c_t']
        np.savez_compressed(path,
            s_prev=np.stack([r['s_prev'] for r in rows]),
            s_next=np.stack([r['s_next'] for r in rows]),
            e_t=np.stack([r['e_t'] for r in rows]),
            c_t=c_t,
            has_real_ct=np.array([r['has_real_ct'] for r in rows], dtype=bool),
            prompt_tokens=np.array([r.get('prompt_tokens', np.nan) for r in rows], dtype=np.float32),
            run_nums=np.array([r['run_num'] for r in rows], dtype=np.int32),
            trials=np.array([r['trial'] for r in rows], dtype=np.int32),
            turns=np.array([r['turn'] for r in rows], dtype=np.int32),
            source_runs=np.array(sorted(source_runs), dtype=np.int32))
        ui.msg(f"  [cache] Saved {len(rows)} quadruplets → {os.path.basename(path)} "
               f"({os.path.getsize(path)/1024/1024:.1f}MB)")
        # v0.75.1.0: clean up stale caches — keep only the one we just wrote.
        # One cache per temperature directory. No accumulation, instant re-entry.
        for stale in glob.glob(os.path.join(hidden_dir, '_qcache_*.npz')):
            if os.path.abspath(stale) != os.path.abspath(path):
                try: os.remove(stale)
                except OSError: pass  # v0.83.1: tighten cleanup-stale except
    except Exception as e:
        ui.warn(f"  [cache] Save failed: {e}")


def _load_quadruplets(hidden_dir, source_runs, model_name, require_ct=True):
    """Load (s_prev, e_t, c_t, s_next, prompt_tokens) quadruplets.

    prompt_tokens: scalar count of full formatted input tokens at each turn.
    Saved by run_generation into every CSV row. Used as a scalar covariate
    in _run_four_models to absorb any residual prompt-length variance that
    survives token matching (e.g. system-prompt differences across runs).

    ct_source field per row:
      'per_trial'   — real per-trial C_t from kind='C' embedding file (3WAY runs)
      'global_mean' — global mean C_t from R0001 constant file (2WAY runs backfilled)
      'none'        — no C_t available; row will not enter three-way OLS

    v40.0.0: 2WAY runs (1, 2, 6-9) now receive backfilled C_t from the Run 0001
    global mean constant (R0001_{model_name}_ct_global_mean.npy) when it exists.
    This elevates them from legacy two-predictor rows to full three-predictor rows
    for the purpose of Run 0042/34. The ct_source field distinguishes per-trial from
    global-mean rows so downstream analysis can report the distinction.

    BUG-ET-DIR fix: E_base files for recovery runs (1-9, 15-17) are now written
    to the base-variant sibling dir by _run_et_recovery. Derive it here and try
    it first for E_base lookups; fall back to hidden_dir (abliterated) for runs
    whose recovery hasn't yet happened or for Run 0001 which writes E_base in-place.
    """
    import pandas as pd

    mn = sanitize(model_name)

    # v0.71.0.1: Detect actual model_name from files on disk. If session model_name
    # doesn't match filenames (e.g. dashboard saveModel() overwrote it), use the
    # name found in the files for ALL file access in this function.
    _effective_mn = mn
    _first_run = source_runs[0] if source_runs else 3
    try:
        _first_run_int = int(_first_run)
    except (ValueError, TypeError):
        _first_run_int = 3
    _det_pfx = run_prefix(_first_run_int)
    _rid4 = f"{_first_run_int:04d}"
    # v0.79.4.0: try 4-digit (canonical) first, fall back to 2-digit (legacy)
    _det_pattern = os.path.join(hidden_dir, f"{_det_pfx}{_rid4}_{mn}_trial*_turn01.npy")
    if not glob.glob(_det_pattern):
        _det_pattern = os.path.join(hidden_dir, f"{_det_pfx}{_first_run_int:02d}_{mn}_trial*_turn01.npy")
    if not glob.glob(_det_pattern):
        # Fallback with wildcard model name — try 4-digit then 2-digit
        for _pat in [f"{_det_pfx}{_rid4}_*_trial*_turn01.npy",
                     f"{_det_pfx}{_first_run_int:02d}_*_trial*_turn01.npy"]:
            _det_fallback = os.path.join(hidden_dir, _pat)
            _det_files = [f for f in sorted(glob.glob(_det_fallback))
                          if not f.endswith('_alllayers.npy')
                          and '_emb.npy' not in f
                          and '_et_base.npy' not in f
                          and '_constraint.npy' not in f]
            if _det_files:
                # Extract effective model name based on whichever prefix matched
                _bn = os.path.basename(_det_files[0])
                for _split_key in [f"{_det_pfx}{_rid4}_", f"{_det_pfx}{_first_run_int:02d}_"]:
                    if _split_key in _bn:
                        _effective_mn = _bn.split(_split_key)[1].split("_trial")[0]
                        break
                if _effective_mn != mn:
                    pass  # silently use disk name — dashboard display name often drifts
                break

    # ── Cache check (v0.71.0.12) ──────────────────────────────────────────
    cached = _qcache_load(hidden_dir, source_runs, require_ct)
    if cached is not None:
        return cached

    # Derive base sibling hidden dir from the abliterated hidden_dir path:
    #   .../size/abliterated/cond/hidden_states  -> .../size/base/cond/hidden_states
    try:
        _cond_dir    = os.path.dirname(hidden_dir)
        _size_dir    = os.path.dirname(os.path.dirname(_cond_dir))
        base_hid_dir = os.path.join(_size_dir, 'base',
                                    os.path.basename(_cond_dir), 'hidden_states')
    except Exception:
        base_hid_dir = hidden_dir  # safe fallback

    # Load the global mean C_t constant from Run 0001.
    _global_ct_path = os.path.join(hidden_dir, f"R0001_{_effective_mn}_ct_global_mean.npy")
    _global_ct = None
    if os.path.exists(_global_ct_path):
        try:
            _global_ct = np.load(_global_ct_path)
            ui.msg(f"  Global mean C_t loaded from Run 0001: {_global_ct_path}")
        except Exception:
            ui.warn(f"  Failed to load global mean C_t: {_global_ct_path}")

    # Build two lookups from CSV data:
    #   pt_lookup: (run_num, trial, turn)       — plain trial, works for single-condition runs
    #   ft_lookup: (run_num, file_trial, turn)  — offset trial matching .npy filenames
    #
    # Multi-condition runs (26, 28, 41) name their .npy files with global_trial /
    # file_trial (an offset encoding condition × n_trials + trial) to avoid BUG 5
    # filename collisions, but store plain trial in the CSV for analysis readability.
    # When _load_quadruplets extracts the trial id from a .npy filename it gets the
    # offset value — which never matches the plain-trial CSV key, causing all
    # prompt_tokens lookups for these runs to NaN-impute.  ft_lookup fixes this by
    # keying on the file_trial column written by runners.py (v24.5 fix).
    csv_dir = os.path.join(os.path.dirname(hidden_dir), 'csv')
    pt_lookup: dict = {}
    ft_lookup: dict = {}
    if os.path.isdir(csv_dir):
        for f in sorted(glob.glob(os.path.join(csv_dir, '*.csv'))):
            try:
                df = pd.read_csv(f, usecols=lambda c: c in
                                 ('run_mode', 'trial', 'file_trial', 'turn', 'prompt_tokens'),
                                 on_bad_lines='skip')
                if not all(c in df.columns for c in ('run_mode', 'trial', 'turn', 'prompt_tokens')):
                    continue
                has_ft = 'file_trial' in df.columns
                for _, row in df.iterrows():
                    run_n = int(row['run_mode'])
                    trn   = int(row['turn'])
                    pt    = float(row['prompt_tokens'])
                    pt_lookup[(run_n, int(row['trial']), trn)] = pt
                    if has_ft and pd.notna(row.get('file_trial')):
                        ft_lookup[(run_n, int(row['file_trial']), trn)] = pt
            except Exception:
                continue

    rows = []
    for run_num in source_runs:
        pfx      = run_prefix(run_num)
        pattern  = os.path.join(hidden_dir, f"{pfx}{run_num:04d}_{_effective_mn}_trial*_turn01.npy")
        trial_files = sorted(glob.glob(pattern))
        if not trial_files:
            ui.warn(f"  Run {run_num}: no files — skipping")
            continue

        trial_ids = []
        for f in trial_files:
            try:
                trial_ids.append(int(os.path.basename(f).split("_trial")[1].split("_")[0]))
            except Exception:
                continue

        n_loaded = 0
        for trial in trial_ids:
            # Determine C_t source for this run
            c_t        = None
            has_real_ct = False
            ct_source  = 'none'

            if require_ct:
                # Attempt per-trial C_t first (3WAY runs: 3,4,5,15,16,17,19,26,28)
                c_t = load_embedding(hidden_dir, run_num, _effective_mn, trial, 1, kind='C')
                if c_t is not None:
                    has_real_ct = True
                    ct_source   = 'per_trial'
                elif _global_ct is not None and run_num in SOURCE_RUNS_2WAY:
                    # Backfill with global mean C_t from Run 0001 for 2WAY runs
                    c_t         = _global_ct
                    has_real_ct = True
                    ct_source   = 'global_mean'

            prev_s = None
            for turn in range(1, QUADRUPLET_MAX_TURNS + 1):
                curr_s = load_hidden_states(run_num, _effective_mn, hidden_dir, trial=trial, turn=turn)
                if curr_s is None: break
                # v41.0.1: try kind='E_base' first (base-model hidden states,
                # written by Run 0001 three-model pass or the Runs 0004-0032 recovery pass).
                # BUG-ET-DIR fix: recovery runs write E_base to the base sibling dir.
                # Try base_hid_dir first, then hidden_dir (abliterated) as fallback
                # for Run 0001 (which writes in-place) and any not-yet-recovered runs.
                # Fall back to kind='E' (_emb.npy) if no E_base found anywhere.
                e_t = load_embedding(base_hid_dir, run_num, _effective_mn, trial, turn, kind='E_base')
                if e_t is None:
                    e_t = load_embedding(hidden_dir, run_num, _effective_mn, trial, turn, kind='E_base')
                if e_t is None:
                    e_t = load_embedding(hidden_dir, run_num, _effective_mn, trial, turn, kind='E')
                if e_t is None:
                    prev_s = curr_s; continue
                if prev_s is not None:
                    pt = ft_lookup.get((run_num, trial, turn),
                         pt_lookup.get((run_num, trial, turn), float('nan')))
                    rows.append({
                        "run_num": run_num, "trial": trial, "turn": turn,
                        "s_prev": _pool(prev_s), "e_t": _pool(e_t),
                        "c_t": _pool(c_t) if (c_t is not None and has_real_ct) else None,
                        "s_next": _pool(curr_s), "has_real_ct": has_real_ct,
                        "ct_source": ct_source,
                        "prompt_tokens": pt,
                    })
                    n_loaded += 1
                prev_s = curr_s
        ui.msg(f"  Run {run_num}: {n_loaded} quadruplets ({len(trial_ids)} trials)")
    _qcache_save(hidden_dir, source_runs, require_ct, rows)
    return rows


def _run_four_models(rows_3way, rows_2way):
    """Fit four nested OLS models for the E+C+R decomposition.

    prompt_tokens is included as a scalar covariate in every model.
    This absorbs any residual prompt-length variance that survives
    token matching — primarily system-prompt length differences across
    runs (introspection vs arithmetic system prompts differ slightly).
    ΔR²_internal is therefore: variance in s_next attributable to s_prev
    AFTER E_t, C_t, AND prompt length are already accounted for.

    Belt (token matching at collection) + suspenders (length covariate
    in regression). Both applied because all runs are uncollected and
    the correct scientific approach is to eliminate the confound at
    both the data and model level.
    """
    def _pt(rows):
        """prompt_tokens column as (N,1) array; NaN rows filled with column mean."""
        arr = np.array([r['prompt_tokens'] for r in rows], dtype=np.float32).reshape(-1, 1)
        col_mean = float(np.nanmean(arr))
        arr[np.isnan(arr)] = col_mean
        return arr

    Xe3 = np.stack([r['e_t']    for r in rows_3way])
    Xs3 = np.stack([r['s_prev'] for r in rows_3way])
    Xc3 = np.stack([r['c_t']    for r in rows_3way])
    Xp3 = _pt(rows_3way)
    y3  = np.stack([r['s_next'] for r in rows_3way])

    all_rows = rows_3way + rows_2way
    Xe_all = np.stack([r['e_t']    for r in all_rows])
    Xs_all = np.stack([r['s_prev'] for r in all_rows])
    Xp_all = _pt(all_rows)
    y_all  = np.stack([r['s_next'] for r in all_rows])

    # All four models now include prompt_tokens (Xp) as baseline covariate.
    # Model A: E_t + prompt_len            (external only)
    # Model B: E_t + S_prev + prompt_len   (external + internal)
    # Model C: E_t + C_t + prompt_len      (external + constraint)
    # Model D: E_t + C_t + S_prev + prompt_len  (full)
    r2_A = _fit_ols(np.hstack([Xe3, Xp3]),              y3)
    r2_B = _fit_ols(np.hstack([Xe3, Xs3, Xp3]),         y3)
    r2_C = _fit_ols(np.hstack([Xe3, Xc3, Xp3]),         y3)
    r2_D = _fit_ols(np.hstack([Xe3, Xc3, Xs3, Xp3]),    y3)
    r2_S = _fit_ols(np.hstack([Xs3, Xp3]),               y3)  # S-only for interaction info
    r2_A_leg = _fit_ols(np.hstack([Xe_all, Xp_all]),             y_all)
    r2_B_leg = _fit_ols(np.hstack([Xe_all, Xs_all, Xp_all]),     y_all)
    return {
        'r2_A': r2_A, 'r2_B': r2_B, 'r2_C': r2_C, 'r2_D': r2_D, 'r2_S': r2_S,
        'delta_internal':   r2_D - r2_C,
        'delta_constraint': r2_C - r2_A,
        'r2_A_leg': r2_A_leg, 'r2_B_leg': r2_B_leg,
        'delta_legacy': r2_B_leg - r2_A_leg,
        'n_3way': len(rows_3way), 'n_all': len(all_rows),
    }


def _perm_test_3way(rows_3way, target, seed=None, n_perm=None, _parallel=False):
    """Permutation test for ΔR² (internal or constraint).

    v0.58.0.8: rewritten for speed. Previous versions called _fit_ols or
    _fit_ols_fast inside the permutation loop — each call did SVD-based lstsq
    on a 36,000 × 3,073 matrix (~3s per fit, 200 iters = 10 min per test).

    New approach:
      1. Normal equations (X^T X, Cholesky solve) instead of SVD-based lstsq.
         DGEMM for X^T X has 3-5x better cache performance than SVD.
      2. Precompute invariant blocks: F^T F, S^T S, F^T y are constant across
         permutations (row permutation preserves Gram matrices). Only cross-terms
         F^T S_perm and S_perm^T y change per iteration.
      3. Compute R² directly from normal equations: R² = sum(Xty * coef) / ss_tot.
         No prediction step (X @ coef), no residuals. Eliminates the largest matmul.

    Expected: ~5-8x faster per iteration. Full Run 0042 main pass in 3-5 minutes.
    """
    if n_perm is None:
        n_perm = N_PERMUTE
    _seed = (0 if target == 'internal' else 1) if seed is None else seed
    rng = np.random.default_rng(_seed)

    from sklearn.preprocessing import StandardScaler

    Xe = np.stack([r['e_t']    for r in rows_3way])
    Xs = np.stack([r['s_prev'] for r in rows_3way])
    Xc = np.stack([r['c_t']    for r in rows_3way])
    y  = np.stack([r['s_next'] for r in rows_3way])
    pt = np.array([r['prompt_tokens'] for r in rows_3way], dtype=np.float32).reshape(-1, 1)
    pt[np.isnan(pt)] = float(np.nanmean(pt))

    # Pre-scale all components ONCE.
    sx_y = StandardScaler(); ys = sx_y.fit_transform(y)
    sx_e  = StandardScaler(); Xe_s = sx_e.fit_transform(Xe)
    sx_c  = StandardScaler(); Xc_s = sx_c.fit_transform(Xc)
    sx_s  = StandardScaler(); Xs_s = sx_s.fit_transform(Xs)
    sx_pt = StandardScaler(); pt_s = sx_pt.fit_transform(pt)

    # Observed delta: full _fit_ols (with cond check — diagnostic, runs twice only).
    if target == 'internal':
        r2_base_obs = _fit_ols(np.hstack([Xe, Xc, pt]), y)
        observed    = _fit_ols(np.hstack([Xe, Xc, Xs, pt]), y) - r2_base_obs
        F = np.hstack([Xe_s, Xc_s, pt_s])   # fixed columns
        S = Xs_s                              # permuted columns
    else:
        r2_base_obs = _fit_ols(np.hstack([Xe, pt]), y)
        observed    = _fit_ols(np.hstack([Xe, Xc, pt]), y) - r2_base_obs
        F = np.hstack([Xe_s, pt_s])
        S = Xc_s

    # ── Precompute invariant blocks ──────────────────────────────────────
    # F^T F and S^T S are constant: row permutation P satisfies P^T P = I,
    # so (PS)^T(PS) = S^T S. F^T y is constant (F and y never change).
    FtF = F.T @ F          # (n_fixed × n_fixed) — constant
    StS = S.T @ S          # (n_perm × n_perm) — invariant under row permutation
    Fty = F.T @ ys         # (n_fixed × k) — constant
    ss_tot = float(np.sum(ys ** 2))  # ys is zero-mean from StandardScaler

    n_f = F.shape[1]
    n_s = S.shape[1]
    n_total = n_f + n_s

    def _r2_normal_equations(S_perm):
        """R² via normal equations, no SVD, no prediction step.
        R² = sum(Xty * coef) / ss_tot when y is zero-mean.
        """
        # Cross-terms (only things that change per permutation)
        FtS = F.T @ S_perm                       # n_fixed × n_perm
        Sty = S_perm.T @ ys                      # n_perm × k

        # Assemble normal equations: (X^T X) coef = X^T y
        XtX = np.empty((n_total, n_total), dtype=F.dtype)
        XtX[:n_f, :n_f] = FtF
        XtX[:n_f, n_f:] = FtS
        XtX[n_f:, :n_f] = FtS.T
        XtX[n_f:, n_f:] = StS

        Xty = np.empty((n_total, ys.shape[1]), dtype=F.dtype)
        Xty[:n_f] = Fty
        Xty[n_f:] = Sty

        # Cholesky solve (3-5x faster than SVD for SPD systems)
        try:
            coef = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Singular — fall back to lstsq (shouldn't happen with scaled data)
            coef = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        # R² = explained_variance / total_variance
        # For zero-mean y: ss_explained = sum(Xty * coef) (Frobenius inner product)
        return float(np.sum(Xty * coef)) / ss_tot

    # ── Permutation loop ─────────────────────────────────────────────────
    iter_seeds = rng.integers(0, 2**31, size=n_perm)
    null_deltas = []
    for iseed in iter_seeds:
        _rng = np.random.default_rng(int(iseed))
        perm_idx = _rng.permutation(len(S))
        null_deltas.append(_r2_normal_equations(S[perm_idx]) - r2_base_obs)

    return observed, float(np.mean(np.array(null_deltas) >= observed))


def _conclusion(p_int, d_int, p_con, d_con):
    r_real = p_int < ALPHA and d_int > 0.01
    c_real = p_con < ALPHA and d_con > 0.01
    if r_real and c_real:
        return (f"Both R and C are real, separable predictors. "
                f"ΔR²_internal={d_int:.4f} (p={p_int:.3f}), "
                f"ΔR²_constraint={d_con:.4f} (p={p_con:.3f}). "
                f"E+C+R=1 decomposition is empirically supported.")
    elif r_real:
        return (f"R is real (ΔR²={d_int:.4f}, p={p_int:.3f}) but C_t does not add "
                f"independent predictive power (p={p_con:.3f}).")
    elif c_real:
        return (f"C_t is a real predictor (ΔR²={d_con:.4f}, p={p_con:.3f}) but "
                f"ΔR²_internal is not significant (p={p_int:.3f}). "
                f"Prior results may be constraint compliance, not internal state share.")
    else:
        return (f"Neither ΔR²_internal (p={p_int:.3f}) nor ΔR²_constraint (p={p_con:.3f}) "
                f"is significant. Insufficient data or decomposition not recoverable.")


def _load_run23_condition_map(csv_dir):
    """Build a {file_trial: confound_condition} lookup from Q28_introspection.csv.

    Run 0023 uses trial_offset=cond_idx*n_trials so hidden state files are named
    with file_trial, not plain trial. _load_quadruplets extracts file_trial from
    the .npy filename. This map lets us tag each Run 0023 row in rows_3way with its
    confound condition without assuming n_trials=100.

    Only 'semantic' and 'neutral' appear here — 'none' has no sys_prompt, so no
    C_t is saved and those rows never reach rows_3way.
    """
    import pandas as pd
    csv_path = os.path.join(csv_dir, "Q0023_introspection.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        df = df.replace("NA", float('nan'))
        if not all(c in df.columns for c in ('file_trial', 'confound_condition')):
            return {}
        df = df[df.get('priming', pd.Series('0', index=df.index)) != '1'] \
            if 'priming' in df.columns else df
        return dict(zip(df['file_trial'].astype(float).astype(int), df['confound_condition']))
    except Exception:
        return {}


def _decompose_condition(rows_3way_base, rows_23_cond, cond_name):
    """Run OLS + permutation tests for one Run 0023 confound condition.

    Combines the Run 0023 condition-specific rows with the full non-28 3WAY base
    so the decomposition measures delta_r2_internal at that condition's C_t
    without losing cross-run variance structure.

    Returns a dict ready to embed in Q33_decomposition.json.
    """
    rows = rows_3way_base + rows_23_cond
    if len(rows) < 20:
        return {"n_rows": len(rows), "error": "insufficient data"}
    m     = _run_four_models(rows, [])
    d_int, p_int = _perm_test_3way(rows, 'internal',    seed=10)
    d_con, p_con = _perm_test_3way(rows, 'constraint',  seed=11)
    return {
        "confound_condition": cond_name,
        "n_rows": len(rows),
        "n_run23_rows": len(rows_23_cond),
        "r2_A": m['r2_A'], "r2_C": m['r2_C'], "r2_D": m['r2_D'],
        "delta_r2_internal":  {"value": float(d_int), "p_value": float(p_int),
                               "significant": bool(p_int < ALPHA)},
        "delta_r2_constraint":{"value": float(d_con), "p_value": float(p_con),
                               "significant": bool(p_con < ALPHA)},
        "verdict": _conclusion(p_int, d_int, p_con, d_con),
    }


def _run_decomposition(session, paths):
    hidden_dir = paths['hidden']
    ana_dir    = paths['analysis']
    csv_dir    = os.path.join(os.path.dirname(hidden_dir), 'csv')
    os.makedirs(ana_dir, exist_ok=True)
    out_file = os.path.join(ana_dir, "Q0042_decomposition.json")
    model_name = session.get('model_name', '')

    ui.section("Run 0042 — Canonical E+C+R Decomposition (no GPU)")
    ui.msg("Supersedes Runs 0047 and 0048.")
    ui.blank()
    _temp33 = session.get('temperature', 0.0)

    def _merge_q33(updates):
        """Merge updates into Q33 partial JSON, preserving existing fields."""
        try:
            existing = {}
            if os.path.exists(out_file):
                with open(out_file) as _mf:
                    existing = json.load(_mf)
            existing.update(updates)
            with open(out_file, 'w') as _mf:
                json.dump(existing, _mf, indent=2, default=str)
        except Exception:
            pass

    # ── Early completion: all stages done in partial → write final, skip quadruplet load ──
    if os.path.exists(out_file):
        try:
            with open(out_file) as _ecf33:
                _ec33 = json.load(_ecf33)
            if (_ec33.get('status') == 'running'
                    and _ec33.get('r2_D_full_decomposition') is not None
                    and _ec33.get('h16_confound_decomposition')
                    and _ec33.get('linearity_check')
                    and _ec33.get('interaction_info')):
                ui.ok("  All stages complete in partial — writing final output.")
                _ec33.pop('status', None)
                with open(out_file, 'w') as _wf33:
                    json.dump(_ec33, _wf33, indent=2, default=str)
                ui.ok(f"  Saved: {out_file}")
                return
        except Exception:
            pass

    _analysis_status(33, "Loading quadruplets", 10, temperature=_temp33)

    ui.msg("Loading 3-way quadruplets (S_{t-1}, E_t, C_t, S_t)...")
    rows_3way_raw = _load_quadruplets(hidden_dir, SOURCE_RUNS_3WAY, model_name, require_ct=True)
    # Rows with per-trial C_t are primary 3WAY rows.
    # Rows backfilled with global mean C_t (ct_source='global_mean') from SOURCE_RUNS_2WAY
    # are also included in the three-way OLS — they have real C_t, just not per-trial.
    rows_3way = [r for r in rows_3way_raw if r['has_real_ct'] and r['c_t'] is not None]

    ui.msg("Loading 2-way triplets for legacy continuity (no C_t available)...")
    rows_2way_raw = _load_quadruplets(hidden_dir, SOURCE_RUNS_2WAY, model_name, require_ct=True)
    # Rows that got global_mean C_t are promoted to 3WAY. Rows with no C_t remain 2WAY.
    rows_promoted = [r for r in rows_2way_raw if r['has_real_ct'] and r['c_t'] is not None]
    rows_2way     = [r for r in rows_2way_raw if not r['has_real_ct'] or r['c_t'] is None]
    if rows_promoted:
        rows_3way = rows_3way + rows_promoted

    n_per_trial  = sum(1 for r in rows_3way if r.get('ct_source') == 'per_trial')
    n_global_ct  = sum(1 for r in rows_3way if r.get('ct_source') == 'global_mean')

    ui.msg(f"  3-way rows (per-trial C_t):   {n_per_trial}")
    ui.msg(f"  3-way rows (global-mean C_t): {n_global_ct}")
    ui.msg(f"  3-way rows total:             {len(rows_3way)}")
    ui.msg(f"  2-way rows (no C_t):          {len(rows_2way)}")
    ui.blank()

    if len(rows_3way) < 50:
        choice = ui.srx_prompt(
            f"Insufficient 3-way quadruplets for Run 0042: {len(rows_3way)} found, 50 required.\n"
            "  C_t embeddings needed from Runs 0013-0015, 26, and 28.\n"
            "  Running anyway will likely produce unreliable decomposition."
        )
        if choice == 's': return
        if choice == 'x': raise SystemExit
        # 'r' → continue with whatever data exists


    _analysis_status(33, "Fitting OLS models", 40, temperature=_temp33)
    ui.msg("Fitting four OLS models (pooled)...")

    # Resume: check if partial results exist
    _r33_partial = {}
    if os.path.exists(out_file):
        try:
            with open(out_file) as _rf33:
                _r33_partial = json.load(_rf33)
        except Exception:
            _r33_partial = {}

    # v0.75.1.0: removed status=='running' gate. Completed JSONs have status
    # popped, so the old guard rejected valid cached OLS data on re-entry.
    # The data presence check is sufficient — if r2_D exists, the OLS is done.
    if _r33_partial.get('r2_D_full_decomposition') is not None:
        m = {
            'r2_A': _r33_partial.get('r2_A_external_only', 0),
            'r2_B': _r33_partial.get('r2_B_external_plus_internal', 0),
            'r2_C': _r33_partial.get('r2_C_external_plus_constraint', 0),
            'r2_D': _r33_partial.get('r2_D_full_decomposition', 0),
            'r2_S': _r33_partial.get('r2_S_internal_only', 0),
            'r2_A_leg': _r33_partial.get('legacy_2way', {}).get('r2_A', 0),
            'r2_B_leg': _r33_partial.get('legacy_2way', {}).get('r2_B', 0),
            'delta_legacy': _r33_partial.get('legacy_2way', {}).get('delta', 0),
        }
        d_int = _r33_partial.get('delta_r2_internal', {}).get('value', 0)
        p_int = _r33_partial.get('delta_r2_internal', {}).get('p_value', 1)
        d_con = _r33_partial.get('delta_r2_constraint', {}).get('value', 0)
        p_con = _r33_partial.get('delta_r2_constraint', {}).get('p_value', 1)
        ui.ok(f"  Resuming: OLS + perm tests loaded from partial")
        # r2_S may be absent from old partials — fit it now if needed
        if m['r2_S'] == 0 and rows_3way:
            _pt33r = lambda rows: np.array(
                [r.get('prompt_tokens', 0.0) for r in rows], dtype=np.float32).reshape(-1, 1)
            Xs3r = np.stack([r['s_prev'] for r in rows_3way])
            Xp3r = _pt33r(rows_3way)
            Xp3r[np.isnan(Xp3r)] = float(np.nanmean(Xp3r)) if not np.all(np.isnan(Xp3r)) else 0.0
            y3r = np.stack([r['s_next'] for r in rows_3way])
            m['r2_S'] = _fit_ols(np.hstack([Xs3r, Xp3r]), y3r)
            ui.msg(f"  Fitted S-only model: R²_S = {m['r2_S']:.6f}")
    else:
        m = _run_four_models(rows_3way, rows_2way)
        _check_pause()

        _analysis_status(33, "Permutation tests", 60, temperature=_temp33)
        ui.msg(f"Running permutation tests (n={N_PERMUTE})...")
        d_int, p_int = _perm_test_3way(rows_3way, 'internal')
        _check_pause()
        d_con, p_con = _perm_test_3way(rows_3way, 'constraint')
        _check_pause()

        # Incremental save — OLS + perm tests done
        try:
            _p33 = {
                'run_num': 33, 'status': 'running',
                'n_3way': len(rows_3way), 'n_2way': len(rows_2way),
                'r2_A_external_only': m['r2_A'], 'r2_B_external_plus_internal': m['r2_B'],
                'r2_C_external_plus_constraint': m['r2_C'], 'r2_D_full_decomposition': m['r2_D'],
                'r2_S_internal_only': m['r2_S'],
                'delta_r2_internal': {'value': float(d_int), 'p_value': float(p_int),
                                       'significant': bool(p_int < ALPHA)},
                'delta_r2_constraint': {'value': float(d_con), 'p_value': float(p_con),
                                         'significant': bool(p_con < ALPHA)},
                'legacy_2way': {'r2_A': m['r2_A_leg'], 'r2_B': m['r2_B_leg'],
                                'delta': m['delta_legacy']},
            }
            with open(out_file, 'w') as _pf33:
                json.dump(_p33, _pf33, indent=2, default=str)
        except Exception:
            pass

    # ── H17 direct test: per-condition decomposition for Run 0023 ─────────────
    # Resume: skip if partial already has h16 data
    if (_r33_partial.get('h16_confound_decomposition')
            and _r33_partial.get('h16_verdict')):
        h16_decomp = _r33_partial['h16_confound_decomposition']
        h16_verdict = _r33_partial['h16_verdict']
        ui.ok("  H16: loaded from partial")
    else:
        ui.msg("Running per-condition decomposition for H17 (Run 0023 confound isolation)...")
        cond_map   = _load_run23_condition_map(csv_dir)
        rows_23    = [r for r in rows_3way if r['run_num'] == 23]  # v0.79.4.0: was 28 (confound)
        rows_base  = [r for r in rows_3way if r['run_num'] != 23]  # v0.79.4.0: was 28
        h16_decomp = {}
        for cond_name in ('semantic', 'neutral'):
            rows_cond = [r for r in rows_23
                         if cond_map.get(r['trial']) == cond_name]
            if rows_cond:
                ui.msg(f"  {cond_name}: {len(rows_cond)} Run 0023 rows + {len(rows_base)} base rows")
                h16_decomp[cond_name] = _decompose_condition(rows_base, rows_cond, cond_name)
            else:
                ui.msg(f"  {cond_name}: no rows found (Run 0023 not yet collected for this condition)")
                h16_decomp[cond_name] = {"confound_condition": cond_name, "n_rows": 0,
                                         "error": "no data"}
            _check_pause()
        semantic_sig = h16_decomp.get('semantic', {}).get(
            'delta_r2_internal', {}).get('significant', None)
        neutral_sig  = h16_decomp.get('neutral',  {}).get(
            'delta_r2_internal', {}).get('significant', None)
        if semantic_sig is True and neutral_sig is True:
            h16_verdict = "delta_r2_internal significant in both conditions — system prompt confound does not explain R"
        elif semantic_sig is False:
            h16_verdict = "delta_r2_internal collapses under semantic system prompt — confound may explain observed R"
        elif semantic_sig is None or neutral_sig is None:
            h16_verdict = "insufficient data for one or both conditions"
        else:
            h16_verdict = "mixed — significant in neutral but not semantic; system prompt partially confounds R"
        # Incremental save — H16 done
        _merge_q33({'h16_confound_decomposition': h16_decomp, 'h16_verdict': h16_verdict})

    # ── Linearity validation: Ridge vs MLP ──────────────────────────────────────
    # Resume: skip if partial already has linearity data
    if (_r33_partial.get('linearity_check')
            and 'error' not in _r33_partial.get('linearity_check', {})):
        linearity_check = _r33_partial['linearity_check']
        ui.ok("  Linearity check: loaded from partial")
    else:
        ui.section("Linearity Validation — Ridge vs MLP")
        linearity_check = {}
        try:
            from sklearn.linear_model import Ridge as _Ridge_lc
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler as _SS_lc
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score as _r2_lc

            _pt_lc = lambda rows: np.array(
                [r.get('prompt_tokens', 0.0) for r in rows], dtype=np.float32).reshape(-1, 1)
            Xe_lc = np.stack([r['e_t'] for r in rows_3way])
            Xc_lc = np.stack([r['c_t'] for r in rows_3way])
            Xs_lc = np.stack([r['s_prev'] for r in rows_3way])
            Xp_lc = _pt_lc(rows_3way)
            Xp_lc[np.isnan(Xp_lc)] = float(np.nanmean(Xp_lc)) if not np.all(np.isnan(Xp_lc)) else 0.0
            y_lc = np.stack([r['s_next'] for r in rows_3way])
            X_full = np.hstack([Xe_lc, Xc_lc, Xs_lc, Xp_lc])

            X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_lc, test_size=0.2, random_state=42)
            sx_lc = _SS_lc(); sy_lc = _SS_lc()
            X_tr_s = sx_lc.fit_transform(X_tr); X_te_s = sx_lc.transform(X_te)
            y_tr_s = sy_lc.fit_transform(y_tr); y_te_s = sy_lc.transform(y_te)

            _ridge = _Ridge_lc(alpha=0.01).fit(X_tr_s, y_tr_s)
            ridge_r2 = float(_r2_lc(y_te_s, _ridge.predict(X_te_s)))
            ui.msg(f"  Ridge R² (test):  {ridge_r2:.6f}")

            _configs = {
                'mlp_256':    {'hidden_layer_sizes': (256,),    'label': 'MLP-256 (1 layer)'},
                'mlp_64':     {'hidden_layer_sizes': (64,),     'label': 'MLP-64 (1 layer)'},
                'mlp_128_64': {'hidden_layer_sizes': (128, 64), 'label': 'MLP-128-64 (2 layers)'},
            }
            mlp_results = {}
            for key, cfg in _configs.items():
                _mlp = MLPRegressor(
                    hidden_layer_sizes=cfg['hidden_layer_sizes'],
                    early_stopping=True, validation_fraction=0.1,
                    max_iter=500, random_state=42)
                _mlp.fit(X_tr_s, y_tr_s)
                mlp_r2 = float(_r2_lc(y_te_s, _mlp.predict(X_te_s)))
                gap = mlp_r2 - ridge_r2
                mlp_results[key] = {
                    'label': cfg['label'],
                    'r2_test': mlp_r2,
                    'gap_vs_ridge': gap,
                    'within_001': abs(gap) < 0.01,
                }
                ui.msg(f"  {cfg['label']} R² (test): {mlp_r2:.6f}  gap: {gap:+.6f}"
                       f"  {'✓' if abs(gap) < 0.01 else '✗'}")

            all_within = all(v['within_001'] for v in mlp_results.values())
            max_gap = max(abs(v['gap_vs_ridge']) for v in mlp_results.values())
            if all_within:
                lc_verdict = f"Linearity validated: all MLP configs within 0.01 of Ridge (max gap={max_gap:.4f})"
            else:
                violators = [v['label'] for v in mlp_results.values() if not v['within_001']]
                lc_verdict = f"Nonlinearity detected in {', '.join(violators)} (max gap={max_gap:.4f})"
            ui.msg(f"  VERDICT: {lc_verdict}")

            linearity_check = {
                'ridge_r2_test': ridge_r2,
                'n_train': len(X_tr), 'n_test': len(X_te),
                'mlp_configs': mlp_results,
                'all_within_001': all_within,
                'max_gap': max_gap,
                'verdict': lc_verdict,
            }
        except Exception as _lc_err:
            ui.warn(f"  Linearity check failed: {_lc_err}")
            linearity_check = {'error': str(_lc_err)}
        # Incremental save — linearity done
        _merge_q33({'linearity_check': linearity_check})

    # ── Interaction information: ordering sensitivity of decomposition ─────────
    # Resume: skip if partial already has interaction info data
    if (_r33_partial.get('interaction_info')
            and 'error' not in _r33_partial.get('interaction_info', {})):
        interaction_info = _r33_partial['interaction_info']
        ui.ok("  Interaction info: loaded from partial")
    else:
        ui.section("Interaction Information — Ordering Sensitivity")
        interaction_info = {}
        try:
            r2_E = m['r2_A']   # E only
            r2_S = m['r2_S']   # S only
            r2_SE = m['r2_B']  # S + E joint (no C)
            n_dim = len(rows_3way[0]['s_next']) if rows_3way else 1
            # Clamp R² to avoid log(0)
            _clamp = lambda x: min(max(x, 1e-10), 1.0 - 1e-10)
            mi_S  = -(n_dim / 2.0) * np.log(1.0 - _clamp(r2_S))
            mi_E  = -(n_dim / 2.0) * np.log(1.0 - _clamp(r2_E))
            mi_SE = -(n_dim / 2.0) * np.log(1.0 - _clamp(r2_SE))
            II = float(mi_S + mi_E - mi_SE)
            II_frac = float(II / mi_SE) if mi_SE > 0 else float('nan')
            ui.msg(f"  R²_E (E only):   {r2_E:.6f}  →  Î(S_t+1; E_t)  = {mi_E:.4f}")
            ui.msg(f"  R²_S (S only):   {r2_S:.6f}  →  Î(S_t+1; S_t)  = {mi_S:.4f}")
            ui.msg(f"  R²_SE (S+E):     {r2_SE:.6f}  →  Î(S_t+1; S,E) = {mi_SE:.4f}")
            ui.msg(f"  II = {II:.4f}  (fraction of joint: {II_frac:.4f})")
            if abs(II_frac) < 0.05:
                ii_verdict = f"II near zero ({II_frac:.4f}): unique information dominates, ordering insensitive, R valid"
            elif II_frac > 0.05:
                ii_verdict = f"Positive II ({II_frac:.4f}): redundancy detected, E-first ordering may inflate E"
            else:
                ii_verdict = f"Negative II ({II_frac:.4f}): synergy detected, decomposition may miss hidden structure"
            ui.msg(f"  VERDICT: {ii_verdict}")
            interaction_info = {
                'r2_E': float(r2_E), 'r2_S': float(r2_S), 'r2_SE': float(r2_SE),
                'mi_E': float(mi_E), 'mi_S': float(mi_S), 'mi_SE': float(mi_SE),
                'II': II, 'II_fraction': II_frac,
                'n_dim': n_dim, 'verdict': ii_verdict,
                'method': 'linear_proxy',
            }

            # ── kNN non-parametric MI (validates linear proxy) ────────────────
            ui.msg("  kNN non-parametric MI estimation...")
            try:
                from sklearn.decomposition import PCA
                from sklearn.neighbors import KNeighborsRegressor
                from sklearn.metrics import r2_score as _r2_knn
                from sklearn.model_selection import train_test_split as _tts_knn
                _KNN_DIM = 32
                _KNN_PER_RUN = 500
                _knn_rng = np.random.default_rng(2024)

                _by_run = {}
                for _ri, _row in enumerate(rows_3way):
                    _rn = _row.get('run_num', 0)
                    _by_run.setdefault(_rn, []).append(_ri)
                _knn_idx = []
                for _rn in sorted(_by_run):
                    _pool = _by_run[_rn]
                    _n_take = min(_KNN_PER_RUN, len(_pool))
                    _knn_idx.extend(_knn_rng.choice(_pool, size=_n_take, replace=False).tolist())
                _knn_rng.shuffle(_knn_idx)
                ui.msg(f"    Stratified sample: {len(_knn_idx)} rows from {len(_by_run)} runs")

                def _knn_ii(idx_list, label='pooled'):
                    """Per-row pooled kNN-MI computation.

                    NOTE TO READERS: The apparatus has THREE different
                    MI estimator families with similar names. This
                    function (_knn_ii) is Family 1.

                    Family 1 -- Cover-Thomas on kNN-regressor R^2:
                      - This function (_knn_ii in analysis.py).
                      - MI = -(dim/2) * log(1 - R^2) where R^2 is from
                        a kNN regressor on the joint distribution.
                      - Assumes Gaussian residuals.
                      - Convention: redundancy-positive (see CONVENTION
                        block below).
                      - Used for: per-cell ii_fraction_knn /
                        redundancy_fraction_knn writes to results.json
                        measurements.

                    Family 2 -- sklearn mutual_info_regression with PCA:
                      - run_kraskov_anchor._kraskov_mi
                      - run_kraskov_spike._kraskov_mi_share
                      - sklearn KSG-equivalent estimator on PCA-reduced
                        features.
                      - Convention: synergy-positive (standard II).
                      - Used for: per-cell p_anchor (kraskov_anchor),
                        V5b parametric sweep (kraskov_spike).

                    Family 3 -- KSG-1 with discrete-dispatcher:
                      - run_split_pca_selection._knn_mi
                      - run_knn_mi_reliability._knn_mi
                      - Dispatches: exact contingency-table MI for
                        integer-lattice; KSG-1 for continuous.
                      - Convention: synergy-positive.
                      - Used for: V5b/V5e calibration, MI reliability
                        validation.

                    These are NOT interchangeable: each handles a
                    different data regime and produces values on
                    different scales. Pick the family that matches the
                    data type, not the one with the most familiar name.

                    CONVENTION: This function computes the
                    redundancy-positive form:
                        II = mi_S + mi_E - mi_SE
                    which is the NEGATION of standard interaction
                    information. In this convention:
                      - II positive  -> redundancy (marginals duplicate
                        info that the joint already carries)
                      - II negative  -> synergy (joint exceeds sum of
                        marginals)
                    Pure XOR (V5d) -> very negative (synergy saturation).
                    Pure redundancy (S = E = Y) -> approaches +1.

                    II_fraction = II / mi_SE is the per-cell summary
                    written to results.json as ii_fraction_knn. Same
                    convention.

                    The MI estimates themselves use the
                    Cover-Thomas Gaussian formula on kNN-regressor R^2,
                    NOT the KSG estimator. KSG-specific bias corrections
                    (e.g. Gao 2015) do not apply to this estimator.

                    See audit_report_v0_82_0_26.md (CHUNK 1) for the
                    full sign convention audit and downstream consumer
                    inventory.
                    """
                    _S = np.stack([rows_3way[i]['s_prev'] for i in idx_list])
                    _E = np.stack([rows_3way[i]['e_t'] for i in idx_list])
                    _Y = np.stack([rows_3way[i]['s_next'] for i in idx_list])
                    _dim = min(_KNN_DIM, len(idx_list) - 1, _S.shape[1])
                    if _dim < 2 or len(idx_list) < 100:
                        return None
                    _Sp = PCA(n_components=_dim, random_state=42).fit_transform(_S)
                    _Ep = PCA(n_components=_dim, random_state=42).fit_transform(_E)
                    _Yp = PCA(n_components=_dim, random_state=42).fit_transform(_Y)
                    _SEp = np.hstack([_Sp, _Ep])
                    _clamp_knn = lambda x: min(max(x, 1e-10), 1.0 - 1e-10)
                    def _knn_r2(X_all, Y_all):
                        Xtr, Xte, ytr, yte = _tts_knn(X_all, Y_all, test_size=0.2, random_state=42)
                        _k = min(5, len(Xtr) - 1)
                        if _k < 1: return 0.0
                        mdl = KNeighborsRegressor(n_neighbors=_k).fit(Xtr, ytr)
                        return float(_r2_knn(yte, mdl.predict(Xte)))
                    r2_s = _knn_r2(_Sp, _Yp)
                    r2_e = _knn_r2(_Ep, _Yp)
                    r2_se = _knn_r2(_SEp, _Yp)
                    mi_s = -(_dim / 2.0) * np.log(1.0 - _clamp_knn(r2_s))
                    mi_e = -(_dim / 2.0) * np.log(1.0 - _clamp_knn(r2_e))
                    mi_se = -(_dim / 2.0) * np.log(1.0 - _clamp_knn(r2_se))
                    _ii = float(mi_s + mi_e - mi_se)
                    _frac = float(_ii / mi_se) if mi_se > 0 else float('nan')
                    return {'mi_S': float(mi_s), 'mi_E': float(mi_e), 'mi_SE': float(mi_se),
                            'II': _ii, 'II_fraction': _frac, 'n': len(idx_list),
                            'r2_S': r2_s, 'r2_E': r2_e, 'r2_SE': r2_se}

                _knn_pooled = _knn_ii(_knn_idx, 'pooled')
                if _knn_pooled:
                    _II_knn = _knn_pooled['II']
                    _II_knn_frac = _knn_pooled['II_fraction']
                    ui.msg(f"    kNN pooled: MI(S)={_knn_pooled['mi_S']:.4f}  MI(E)={_knn_pooled['mi_E']:.4f}"
                           f"  MI(S,E)={_knn_pooled['mi_SE']:.4f}  II={_II_knn:.4f}  frac={_II_knn_frac:.4f}")
                else:
                    _II_knn_frac = float('nan')

                _COND_RUNS = {
                    'introspection': {6, 7, 8, 13, 14, 15},
                    'null': {1}, 'temperature': {3}, 'confound': {23},
                }
                _knn_per_cond = {}
                for _cname, _cruns in _COND_RUNS.items():
                    _cidx = [i for i in _knn_idx if rows_3way[i].get('run_num', 0) in _cruns]
                    if len(_cidx) >= 100:
                        _cr = _knn_ii(_cidx, _cname)
                        if _cr:
                            _knn_per_cond[_cname] = _cr
                            ui.msg(f"    kNN {_cname:14s}: II={_cr['II']:.4f}  frac={_cr['II_fraction']:.4f}  (n={_cr['n']})")

                if np.isnan(_II_knn_frac):
                    _agree = True
                else:
                    _both_near_zero = abs(II_frac) < 0.05 and abs(_II_knn_frac) < 0.05
                    _both_positive = II_frac > 0.05 and _II_knn_frac > 0.05
                    _both_negative = II_frac < -0.05 and _II_knn_frac < -0.05
                    _agree = _both_near_zero or _both_positive or _both_negative
                if _agree:
                    ui.msg(f"    Linear and kNN AGREE — reporting linear proxy")
                else:
                    ui.msg(f"    Linear and kNN DISAGREE — reporting both")
                    ii_verdict = (f"Methods disagree: linear II_frac={II_frac:.4f}, "
                                  f"kNN II_frac={_II_knn_frac:.4f}. Interpret with caution.")

                interaction_info['knn'] = {
                    'pooled': _knn_pooled,
                    'per_condition': _knn_per_cond,
                    'n_samples': len(_knn_idx), 'pca_dim': _KNN_DIM,
                    'agrees_with_linear': _agree,
                }
                interaction_info['verdict'] = ii_verdict
            except Exception as _knn_err:
                ui.warn(f"    kNN MI failed: {_knn_err} — linear proxy only")
                interaction_info['knn'] = {'error': str(_knn_err)}
        except Exception as _ii_err:
            ui.warn(f"  Interaction information failed: {_ii_err}")
            interaction_info = {'error': str(_ii_err)}
        # Incremental save — interaction info done
        _merge_q33({'interaction_info': interaction_info})

    results = {
        "run_num": 33,
        "n_3way": len(rows_3way), "n_2way": len(rows_2way),
        "r2_A_external_only":             m['r2_A'],
        "r2_B_external_plus_internal":    m['r2_B'],
        "r2_C_external_plus_constraint":  m['r2_C'],
        "r2_D_full_decomposition":        m['r2_D'],
        "r2_S_internal_only":             m['r2_S'],
        "delta_r2_internal":  {"value": float(d_int), "p_value": float(p_int),
                               "significant": bool(p_int < ALPHA)},
        "delta_r2_constraint":{"value": float(d_con), "p_value": float(p_con),
                               "significant": bool(p_con < ALPHA)},
        "legacy_2way": {"r2_A": m['r2_A_leg'], "r2_B": m['r2_B_leg'],
                        "delta": m['delta_legacy']},
        "verdict": {
            "r_is_separable_from_c": bool(p_int < ALPHA and d_int > 0.01),
            "c_is_measurable":       bool(p_con < ALPHA and d_con > 0.01),
            "conclusion": _conclusion(p_int, d_int, p_con, d_con),
        },
        # H17 direct test — per-condition decomposition (INF-1 fix, v29.7)
        "h16_confound_decomposition": h16_decomp,
        "h16_verdict": h16_verdict,
        "linearity_check": linearity_check,
        "interaction_info": interaction_info,
    }

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    ui.blank()
    ui.section("Results")
    ui.msg(f"  R²_A (E only):     {m['r2_A']:.4f}")
    ui.msg(f"  R²_B (E + S):      {m['r2_B']:.4f}")
    ui.msg(f"  R²_C (E + C):      {m['r2_C']:.4f}")
    ui.msg(f"  R²_D (E + C + S):  {m['r2_D']:.4f}")
    ui.blank()
    si = "✓ SIGNIFICANT" if p_int < ALPHA else "✗ not significant"
    sc = "✓ SIGNIFICANT" if p_con < ALPHA else "✗ not significant"
    ui.msg(f"  ΔR²_internal:    {d_int:.4f}  p={p_int:.4f}  {si}")
    ui.msg(f"  ΔR²_constraint:  {d_con:.4f}  p={p_con:.4f}  {sc}")
    ui.blank()
    ui.msg("  H17 per-condition (confound isolation):")
    for cond, cd in h16_decomp.items():
        if 'error' not in cd:
            dr = cd.get('delta_r2_internal', {})
            sig = "✓" if dr.get('significant') else "✗"
            ui.msg(f"    {cond:10s}  ΔR²_int={dr.get('value', float('nan')):.4f}"
                   f"  p={dr.get('p_value', float('nan')):.4f}  {sig}")
        else:
            ui.msg(f"    {cond:10s}  {cd['error']}")
    ui.msg(f"  H17 verdict: {h16_verdict}")
    ui.blank()
    ui.msg(f"  VERDICT: {results['verdict']['conclusion']}")
    ui.blank()
    ui.ok(f"Saved: {out_file}")


# ══════════════════════════════════════════════════════════════════
# RUN 34 — SOBOL CAUSAL PARTITION
# H22: E, C, R causal fractions can be estimated as a proper partition
# summing to 1 via permutation-based variance attribution (Sobol first-order).
# Requires same quadruplet data as Run 0042.
# No GPU. Post-hoc on saved hidden states.

def _permutation_sensitivity(rows_3way, component, n_perm=N_PERMUTE_SOBOL, seed=42, _parallel=False):
    """
    Estimate first-order permutation sensitivity index for component in {E, C, S_prev}.
    Method: fit a Ridge model on full data, then permute the target component across
    trials and measure variance drop in S_next predictions.
    Effect = baseline_var - permuted_var (averaged over n_perm shuffles).

    NOTE: This is fixed-model permutation sensitivity, not proper Sobol indices.
    Proper Sobol indices (Saltelli, 2002) require independent Monte Carlo sampling
    from the joint distribution. This method systematically overestimates attributed
    variance because the model was trained on the data being permuted. Keys are
    named perm_sens_* throughout to distinguish from proper Sobol decompositions.
    Renamed from _sobol_first_order in v15.0.

    v18.2: prompt_tokens (Xp) added as scalar covariate — mirrors the suspenders
    fix applied to _fit_perm_model / _permutation_sensitivity_on in v17.1.
    Run 0051 (pooled) now calls this function with consistent methodology so its
    fractions are directly comparable to Run 0043. Xp is never permuted.
    Feature layout: [Xe, Xc, Xs, Xp].
    """
    Xe = np.stack([r['e_t']    for r in rows_3way])
    Xs = np.stack([r['s_prev'] for r in rows_3way])
    Xc = np.stack([r['c_t']    for r in rows_3way])
    y  = np.stack([r['s_next'] for r in rows_3way])
    Xp = np.array([r.get('prompt_tokens', 0.0) for r in rows_3way],
                  dtype=np.float32).reshape(-1, 1)
    Xp[np.isnan(Xp)] = float(np.nanmean(Xp)) if not np.all(np.isnan(Xp)) else 0.0

    # Baseline: R² of full model predictions
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score as _r2
    full_X = np.hstack([Xe, Xc, Xs, Xp])          # layout [Xe, Xc, Xs, Xp]
    sx = StandardScaler()
    sy = StandardScaler()
    full_Xs = sx.fit_transform(full_X)
    ys = sy.fit_transform(y)
    mdl = Ridge(alpha=0.01).fit(full_Xs, ys)
    baseline_pred = mdl.predict(full_Xs)
    baseline_r2   = float(_r2(ys, baseline_pred))
    # baseline_var kept for backward compat with interaction_mass calculation
    baseline_var  = float(np.var(baseline_pred))

    # R² drop — Xp column is held fixed (control variable).
    # Use R² drop instead of variance drop: R² drop is always positive for a
    # real predictor and sign-invariant to Ridge's scaled prediction space.
    # Variance drop can be negative when permuted predictions spread out more
    # than baseline predictions (overfitting in standardized space).
    # RNG-PERM-CORR fix (v30.9): use passed seed so E/C/R draws are independent.
    rng = np.random.default_rng(seed)
    # Generate per-iteration seeds for reproducibility under parallelism
    iter_seeds = rng.integers(0, 2**31, size=n_perm)

    def _one_perm(iseed):
        _rng = np.random.default_rng(iseed)
        perm_idx = _rng.permutation(len(rows_3way))
        if component == 'E':
            perm_X = np.hstack([Xe[perm_idx], Xc, Xs, Xp])
        elif component == 'C':
            perm_X = np.hstack([Xe, Xc[perm_idx], Xs, Xp])
        else:
            perm_X = np.hstack([Xe, Xc, Xs[perm_idx], Xp])
        perm_Xs   = sx.transform(perm_X)
        perm_pred = mdl.predict(perm_Xs)
        return baseline_r2 - float(_r2(ys, perm_pred))

    if _parallel:
        try:
            from joblib import Parallel, delayed
            drop_samples = Parallel(n_jobs=-1)(
                delayed(_one_perm)(int(s)) for s in iter_seeds)
        except ImportError:
            drop_samples = [_one_perm(int(s)) for s in iter_seeds]
    else:
        drop_samples = [_one_perm(int(s)) for s in iter_seeds]

    effect = float(np.mean(drop_samples))
    effect_std = float(np.std(drop_samples))
    return effect, effect_std, baseline_var


def _fit_perm_model(rows_3way):
    """Fit the Ridge model on training rows for held-out evaluation (H28 / Run 0033).

    Separated from _permutation_sensitivity so the same fitted model can be
    evaluated on both in-sample data and held-out (Run 0033) data. Returns the
    fitted model and scalers so _permutation_sensitivity_on can apply them
    to any row set without re-fitting.

    Returns: (mdl, sx, sy, baseline_var)
      mdl          — fitted Ridge(alpha=0.01)
      sx           — StandardScaler fitted on [Xe, Xc, Xs, Xp] (input space)
      sy           — StandardScaler fitted on y (output space)
      baseline_var — variance of in-sample predictions (reference point)

    v17.1: prompt_tokens (Xp) added as scalar covariate — suspenders fix.
    Mirrors _run_four_models and _perm_test_3way in Run 0042. Xp is never
    permuted; it controls for residual prompt-length variance that survives
    token matching (e.g. system-prompt length differences across runs).
    Feature layout [Xe, Xc, Xs, Xp] must match _permutation_sensitivity_on
    exactly — the scaler sx is fitted here and applied there without re-fitting.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    Xe = np.stack([r['e_t']    for r in rows_3way])
    Xs = np.stack([r['s_prev'] for r in rows_3way])
    Xc = np.stack([r['c_t']    for r in rows_3way])
    y  = np.stack([r['s_next'] for r in rows_3way])
    Xp = np.array([r.get('prompt_tokens', 0.0) for r in rows_3way],
                  dtype=np.float32).reshape(-1, 1)
    Xp[np.isnan(Xp)] = float(np.nanmean(Xp)) if not np.all(np.isnan(Xp)) else 0.0
    full_X  = np.hstack([Xe, Xc, Xs, Xp])
    sx = StandardScaler()
    sy = StandardScaler()
    full_Xs = sx.fit_transform(full_X)
    ys      = sy.fit_transform(y)
    mdl     = Ridge(alpha=0.01).fit(full_Xs, ys)
    baseline_pred = mdl.predict(full_Xs)
    baseline_var  = float(np.var(baseline_pred))
    return mdl, sx, sy, baseline_var


def _permutation_sensitivity_on(rows, component, mdl, sx, sy=None, n_perm=N_PERMUTE_SOBOL, seed=99, _parallel=False):
    """Evaluate permutation sensitivity of a pre-fit model on the given rows.

    Used for Run 0043 held-out evaluation (H28): fit model on training rows
    via _fit_perm_model, then call this function with the held-out (Run 0033)
    rows to measure permutation sensitivity without in-sample bias.

    Unlike _permutation_sensitivity, this function NEVER fits a model — it
    applies the provided (mdl, sx, sy) fitted on training data to new rows.

    v0.58.0.0 FIX-10: sy (target scaler) now accepted and used. Held-out rows
    DO have s_next (they are full quadruplets from Run 0033). Prior code assumed
    "held-out rows have no target y" and fell back to |variance drop| — a
    different metric than the R² drop used in-sample, biasing H28 comparisons.
    Now uses R² drop when sy is provided (identical method to in-sample).
    Falls back to |variance drop| only if sy is None (backward compat).

    component: 'E', 'C', or 'S_prev' (= R).
    seed: default 99 (backward compat). Callers pass distinct seeds per
    component (99/100/101 in-sample, 102/103/104 held-out — RNG-PERM-CORR fix,
    v30.9) so E/C/R permutation draws are independent.

    v17.1: prompt_tokens (Xp) added as scalar covariate — suspenders fix.
    Xp is never permuted. Feature layout [Xe, Xc, Xs, Xp] matches
    _fit_perm_model exactly so sx.transform() receives the correct shape.
    """
    Xe = np.stack([r['e_t']    for r in rows])
    Xs = np.stack([r['s_prev'] for r in rows])
    Xc = np.stack([r['c_t']    for r in rows])
    Xp = np.array([r.get('prompt_tokens', 0.0) for r in rows], dtype=np.float32).reshape(-1, 1)
    Xp[np.isnan(Xp)] = float(np.nanmean(Xp)) if not np.all(np.isnan(Xp)) else 0.0

    from sklearn.metrics import r2_score as _r2_h
    full_X   = np.hstack([Xe, Xc, Xs, Xp])
    full_Xs  = sx.transform(full_X)          # pre-fit scaler — no re-fitting
    baseline_pred = mdl.predict(full_Xs)
    baseline_var  = float(np.var(baseline_pred))

    # Scale held-out y with the TRAINING sy scaler (no re-fitting).
    # R² drop is now identical to the in-sample method.
    use_r2 = False
    ys_h   = None
    baseline_r2_h = None
    if sy is not None:
        y = np.stack([r['s_next'] for r in rows])
        ys_h = sy.transform(y)
        baseline_r2_h = float(_r2_h(ys_h, baseline_pred))
        use_r2 = True

    # Generate per-iteration seeds for reproducibility under parallelism
    rng = np.random.default_rng(seed)
    iter_seeds = rng.integers(0, 2**31, size=n_perm)

    def _one_perm_on(iseed):
        _rng = np.random.default_rng(iseed)
        perm_idx = _rng.permutation(len(rows))
        if component == 'E':
            perm_X = np.hstack([Xe[perm_idx], Xc, Xs, Xp])
        elif component == 'C':
            perm_X = np.hstack([Xe, Xc[perm_idx], Xs, Xp])
        else:
            perm_X = np.hstack([Xe, Xc, Xs[perm_idx], Xp])
        perm_Xs  = sx.transform(perm_X)
        perm_pred = mdl.predict(perm_Xs)
        if use_r2:
            return baseline_r2_h - float(_r2_h(ys_h, perm_pred))
        else:
            return abs(baseline_var - float(np.var(perm_pred)))

    if _parallel:
        try:
            from joblib import Parallel, delayed
            drop_samples = Parallel(n_jobs=-1)(
                delayed(_one_perm_on)(int(s)) for s in iter_seeds)
        except ImportError:
            drop_samples = [_one_perm_on(int(s)) for s in iter_seeds]
    else:
        drop_samples = [_one_perm_on(int(s)) for s in iter_seeds]

    effect     = float(np.mean(drop_samples))
    effect_std = float(np.std(drop_samples))
    return effect, effect_std, baseline_var


def _run_permutation_sensitivity(session, paths):
    """Run 0043 — Permutation sensitivity causal partition + held-out validation (H22 + H28).

    Two-stage analysis:
      Stage 1 (existing): fit Ridge on SOURCE_RUNS_3WAY, compute in-sample
        permutation sensitivity fractions. Preserved for continuity.
      Stage 2 (new v16.0): load Run 0033 held-out rows. Evaluate the SAME fitted
        model on held-out data using _permutation_sensitivity_on. Fractions from
        unseen trajectories break the fixed-model in-sample bias.

    H22 is answered by Stage 1 (R fraction > 0.05).
    H28 is answered by Stage 2: held-out fractions matching in-sample within
    ±0.05 = partition externally validated. Divergence > 0.10 = overfit.

    Run 0033 data is OPTIONAL — if absent, Stage 2 is skipped. Backward compatible.
    """
    hidden_dir = paths['hidden']
    ana_dir    = paths['analysis']
    os.makedirs(ana_dir, exist_ok=True)
    out_file   = os.path.join(ana_dir, "Q0043_sobol_partition.json")
    model_name = session.get('model_name', '')

    ui.section("Run 0043 — Permutation Sensitivity Partition (no GPU)")
    ui.msg("Stage 1: in-sample fractions. Stage 2: held-out validation via Run 0033 (v16.0).")
    ui.msg("This is the run that earns E+C+R=1 as written.")
    _temp34 = session.get('temperature', 0.0)

    # ── Early completion: if bootstrap is done, write final output without reloading ──
    _N_BOOTSTRAP = 500
    if os.path.exists(out_file):
        try:
            with open(out_file) as _ecf:
                _ec = json.load(_ecf)
            _ba = _ec.get('_boot_arrays', {})
            _has_fracs = _ec.get('perm_sens_R', {}).get('fraction') is not None
            _boot_done = len(_ba.get('R', [])) >= _N_BOOTSTRAP
            # v0.75.1.0: removed status=='running' gate — same fix as Run 0042.
            if _has_fracs and _boot_done:
                ui.ok("  Bootstrap already complete — writing final output from checkpoint.")
                # Extract everything from partial
                effect_E = _ec['perm_sens_E']['effect']
                effect_C = _ec['perm_sens_C']['effect']
                effect_R = _ec['perm_sens_R']['effect']
                std_E = _ec['perm_sens_E']['std']
                std_C = _ec['perm_sens_C']['std']
                std_R = _ec['perm_sens_R']['std']
                frac_E = _ec['perm_sens_E']['fraction']
                frac_C = _ec['perm_sens_C']['fraction']
                frac_R = _ec['perm_sens_R']['fraction']
                total = effect_E + effect_C + effect_R
                interaction_mass = _ec.get('interaction_mass_unattributed', float('nan'))
                verdict = _ec.get('verdict', '')
                # Bootstrap CIs — flat format matching normal path
                boot_E_arr = np.array(_ba['E'])
                boot_C_arr = np.array(_ba['C'])
                boot_R_arr = np.array(_ba['R'])
                bootstrap_ci = {
                    'n_bootstrap': len(boot_R_arr),
                    'E_ci_lower': float(np.percentile(boot_E_arr, 2.5)),
                    'E_ci_upper': float(np.percentile(boot_E_arr, 97.5)),
                    'E_ci_median': float(np.median(boot_E_arr)),
                    'C_ci_lower': float(np.percentile(boot_C_arr, 2.5)),
                    'C_ci_upper': float(np.percentile(boot_C_arr, 97.5)),
                    'C_ci_median': float(np.median(boot_C_arr)),
                    'R_ci_lower': float(np.percentile(boot_R_arr, 2.5)),
                    'R_ci_upper': float(np.percentile(boot_R_arr, 97.5)),
                    'R_ci_median': float(np.median(boot_R_arr)),
                    'R_excludes_zero': bool(np.percentile(boot_R_arr, 2.5) > 0),
                }
                baseline_var = _ec.get('baseline_var', 0)
                results = {
                    'run_num': 34, 'n_3way': _ec.get('n_3way', 0), 'n_perm': _ec.get('n_perm', 0),
                    'perm_sens_E': {'effect': effect_E, 'std': std_E, 'fraction': frac_E},
                    'perm_sens_C': {'effect': effect_C, 'std': std_C, 'fraction': frac_C},
                    'perm_sens_R': {'effect': effect_R, 'std': std_R, 'fraction': frac_R},
                    'interaction_mass_unattributed': interaction_mass,
                    'baseline_var': baseline_var,
                    'bootstrap_ci': bootstrap_ci, 'n_bootstrap': _N_BOOTSTRAP,
                    'verdict': verdict,
                }
                with open(out_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                ui.ok(f"  E={frac_E:.3f}  C={frac_C:.3f}  R={frac_R:.3f}")
                ui.ok(f"Saved: {out_file}")
                return
        except Exception:
            pass

    _analysis_status(34, "Loading quadruplets", 5, temperature=_temp34)
    ui.blank()

    # ── Stage 1: training data ────────────────────────────────────────
    rows_3way = _load_quadruplets(hidden_dir, SOURCE_RUNS_3WAY, model_name, require_ct=True)
    rows_3way = [r for r in rows_3way if r['has_real_ct'] and r['c_t'] is not None]
    ui.msg(f"  Training rows (SOURCE_RUNS_3WAY): {len(rows_3way)}")

    if len(rows_3way) < 50:
        choice = ui.srx_prompt(
            f"Insufficient training rows for Run 0043: {len(rows_3way)} found, 50 required.\n"
            "  Runs 0003, 0023, and Runs 0006-0008 must complete first.\n"
            "  Running anyway will produce unreliable partition fractions."
        )
        if choice == 's': return
        if choice == 'x': raise SystemExit

    # ── Stage 2: held-out data (Run 0033) ──────────────────────────────
    rows_heldout = _load_quadruplets(hidden_dir, SOURCE_RUNS_VALIDATION, model_name, require_ct=True)
    rows_heldout = [r for r in rows_heldout if r['has_real_ct'] and r['c_t'] is not None]
    has_heldout  = len(rows_heldout) >= 20
    ui.msg(f"  Held-out rows   (Run 0033):         {len(rows_heldout)}"
           + ("" if has_heldout else "  [not yet available — Stage 2 skipped]"))
    ui.blank()

    # Fit Ridge once on training. Same model used for both stages.
    ui.msg("Fitting Ridge model on training data...")
    mdl_train, sx_train, sy_train, bv_train = _fit_perm_model(rows_3way)

    # v0.71.0.10: Resume from partial — skip in-sample permutation if fractions exist
    _resumed_34 = False
    effect_E = effect_C = effect_R = std_E = std_C = std_R = bv_E = bv_C = bv_R = 0.0
    if os.path.exists(out_file):
        try:
            with open(out_file) as _rf34:
                _p34 = json.load(_rf34)
            # v0.75.1.0: removed status=='running' gate — same fix as Run 0042.
            if _p34.get('perm_sens_R', {}).get('fraction') is not None:
                effect_E = _p34['perm_sens_E']['effect']
                effect_C = _p34['perm_sens_C']['effect']
                effect_R = _p34['perm_sens_R']['effect']
                std_E = _p34['perm_sens_E']['std']
                std_C = _p34['perm_sens_C']['std']
                std_R = _p34['perm_sens_R']['std']
                bv_E = bv_C = bv_R = bv_train
                _resumed_34 = True
                ui.ok(f"  Resuming: in-sample fractions loaded from partial "
                      f"(E={effect_E/sum([effect_E,effect_C,effect_R]):.3f} "
                      f"C={effect_C/sum([effect_E,effect_C,effect_R]):.3f} "
                      f"R={effect_R/sum([effect_E,effect_C,effect_R]):.3f})")
        except Exception:
            pass

    if not _resumed_34:
        ui.msg(f"Computing in-sample permutation sensitivity (n_perm={N_PERMUTE_SOBOL} per component)...")
        # RNG-PERM-CORR fix (v30.9): seeds 99/100/101 — independent draws per component.
        effect_E, std_E, bv_E = _permutation_sensitivity_on(rows_3way, 'E',      mdl_train, sx_train, sy=sy_train, seed=99)
        ui.msg(f"  E effect: {effect_E:.6f}  (std={std_E:.6f})")
        _check_pause()
        effect_C, std_C, bv_C = _permutation_sensitivity_on(rows_3way, 'C',      mdl_train, sx_train, sy=sy_train, seed=100)
        ui.msg(f"  C effect: {effect_C:.6f}  (std={std_C:.6f})")
        _check_pause()
        effect_R, std_R, bv_R = _permutation_sensitivity_on(rows_3way, 'S_prev', mdl_train, sx_train, sy=sy_train, seed=101)
        ui.msg(f"  R effect: {effect_R:.6f}  (std={std_R:.6f})")
        _check_pause()
    ui.blank()

    baseline_var = float(np.mean([bv_E, bv_C, bv_R]))
    total = effect_E + effect_C + effect_R
    if total > 0:
        frac_E = effect_E / total
        frac_C = effect_C / total
        frac_R = effect_R / total
    else:
        frac_E = frac_C = frac_R = float('nan')

    # interaction_mass: fraction of baseline variance NOT explained by first-order effects.
    # Must be computed from raw effects / baseline_var BEFORE renormalization —
    # after renorm frac_E+frac_C+frac_R = 1.0 by definition, useless for this.
    if baseline_var > 0:
        interaction_mass = float(1.0 - total / baseline_var)
    else:
        interaction_mass = float('nan')

    no_nan = all(v == v for v in [frac_E, frac_C, frac_R])
    verdict = (
        f"E={frac_E:.3f}  C={frac_C:.3f}  R={frac_R:.3f}  "
        f"(in-sample renormalized permutation sensitivity). "
        f"Interaction mass not attributed: {interaction_mass:.3f}. "
        f"R fraction={frac_R:.3f} — conditional MI share attributable to prior state "
        f"under permutation null."
    ) if no_nan else "Could not compute in-sample partition — check data."

    # Incremental save — in-sample fractions available even if bootstrap/held-out crashes
    # PRESERVE existing _boot_arrays from prior checkpoint
    try:
        _partial34 = {
            'run_num': 34, 'status': 'running',
            'n_3way': len(rows_3way), 'n_perm': N_PERMUTE_SOBOL,
            'perm_sens_E': {'effect': effect_E, 'std': std_E, 'fraction': frac_E},
            'perm_sens_C': {'effect': effect_C, 'std': std_C, 'fraction': frac_C},
            'perm_sens_R': {'effect': effect_R, 'std': std_R, 'fraction': frac_R},
            'interaction_mass_unattributed': interaction_mass,
            'baseline_var': baseline_var,
            'verdict': verdict,
        }
        # Merge with existing file to keep _boot_arrays
        if os.path.exists(out_file):
            try:
                with open(out_file) as _ef34:
                    _existing = json.load(_ef34)
                if '_boot_arrays' in _existing:
                    _partial34['_boot_arrays'] = _existing['_boot_arrays']
            except Exception:
                pass
        with open(out_file, 'w') as _pf:
            json.dump(_partial34, _pf, indent=2, default=str)
    except Exception:
        pass

    _analysis_status(34, "In-sample permutation", 30,
                     f"E={frac_E:.3f} C={frac_C:.3f} R={frac_R:.3f}", _temp34)

    # ── Bootstrap CIs on E/C/R fractions (v0.71.0.0) ─────────────────
    # Resample the quadruplet pool with replacement, recompute permutation
    # sensitivity each time, report 2.5th and 97.5th percentiles.
    # The CI excluding zero for R at each temperature is the claim.
    # v0.71.0.8: m-out-of-n bootstrap (Politis & Romano 1994). Cap resample at
    # 5000 rows AND re-pool to POOL_DIM dims. The in-sample fractions are computed at
    # full POOL_DIM — bootstrap only measures fraction *stability*, not precision.
    # At 512 dims × 5000 rows: ~1.5s/iter = ~12min. Conservative: wider CIs at
    # lower dim means if R excludes zero here, it excludes zero at full dim.
    _N_BOOTSTRAP = 500
    _BOOT_N_PERM = 20   # reduced permutations per bootstrap iteration
    _BOOT_MAX_ROWS = 5000  # cap per-iteration sample size
    _BOOT_DIM = POOL_DIM  # must match in-sample dimension for valid CIs
    bootstrap_ci = {}
    if len(rows_3way) >= 50 and no_nan:
        _boot_n = min(len(rows_3way), _BOOT_MAX_ROWS)
        # Pre-pool all rows to bootstrap dimension once
        _boot_pool = _repool_rows(rows_3way, _BOOT_DIM)
        _speed_note = f" (m={_boot_n}, dim={_BOOT_DIM})"

        # Resume: check for saved bootstrap arrays in partial JSON
        boot_E, boot_C, boot_R = [], [], []
        _boot_start = 0
        if os.path.exists(out_file):
            try:
                with open(out_file) as _rbf:
                    _bp = json.load(_rbf)
                # v0.75.1.0: removed status=='running' gate.
                if _bp.get('_boot_arrays'):
                    _ba = _bp['_boot_arrays']
                    boot_E = _ba.get('E', [])
                    boot_C = _ba.get('C', [])
                    boot_R = _ba.get('R', [])
                    _boot_start = len(boot_R)
                    if _boot_start > 0:
                        ui.ok(f"  Resuming bootstrap from iteration {_boot_start}/{_N_BOOTSTRAP}")
            except Exception:
                pass

        ui.msg(f"  Bootstrap CIs: {_N_BOOTSTRAP} iterations × {_BOOT_N_PERM} perms/component{_speed_note}...")
        import time as _bt
        _bt0 = _bt.time()
        boot_rng = np.random.default_rng(7777)
        # Advance RNG past already-completed iterations
        for _skip_i in range(_boot_start):
            boot_rng.integers(0, len(_boot_pool), size=_boot_n)
        for bi in range(_boot_start, _N_BOOTSTRAP):
            # Resample with replacement from pre-pooled rows
            idx = boot_rng.integers(0, len(_boot_pool), size=_boot_n)
            boot_rows = [_boot_pool[i] for i in idx]
            # Compute permutation sensitivity with reduced n_perm
            # Parallelize 3 independent components for ~2-3x speedup
            try:
                be, _, _ = _permutation_sensitivity(boot_rows, 'E',      n_perm=_BOOT_N_PERM, seed=bi*3)
                bc, _, _ = _permutation_sensitivity(boot_rows, 'C',      n_perm=_BOOT_N_PERM, seed=bi*3+1)
                br, _, _ = _permutation_sensitivity(boot_rows, 'S_prev', n_perm=_BOOT_N_PERM, seed=bi*3+2)
                bt_total = be + bc + br
                if bt_total > 0:
                    boot_E.append(be / bt_total)
                    boot_C.append(bc / bt_total)
                    boot_R.append(br / bt_total)
            except Exception:
                continue
            if (bi + 1) % 25 == 0:
                _bt_pct = int(30 + (bi + 1) / _N_BOOTSTRAP * 30)
                _analysis_status(34, f"Bootstrap {bi+1}/{_N_BOOTSTRAP}", _bt_pct,
                                 f"{_bt.time()-_bt0:.0f}s elapsed", _temp34)
                _check_pause()
                ui.msg(f"    {bi+1}/{_N_BOOTSTRAP} ({_bt.time()-_bt0:.0f}s)")
                # Checkpoint: save bootstrap arrays to partial JSON
                try:
                    with open(out_file) as _rcf:
                        _cp = json.load(_rcf)
                    _cp['_boot_arrays'] = {'E': boot_E, 'C': boot_C, 'R': boot_R}
                    with open(out_file, 'w') as _wcf:
                        json.dump(_cp, _wcf, indent=2, default=str)
                except Exception:
                    pass
        _bt_elapsed = _bt.time() - _bt0
        if len(boot_R) >= 50:
            boot_E_arr = np.array(boot_E)
            boot_C_arr = np.array(boot_C)
            boot_R_arr = np.array(boot_R)
            bootstrap_ci = {
                "n_bootstrap": len(boot_R),
                "n_perm_per_iter": _BOOT_N_PERM,
                "resample_size": _boot_n,
                "boot_dim": _BOOT_DIM,
                "elapsed_sec": round(_bt_elapsed, 1),
                "E_ci_lower": float(np.percentile(boot_E_arr, 2.5)),
                "E_ci_upper": float(np.percentile(boot_E_arr, 97.5)),
                "E_ci_median": float(np.median(boot_E_arr)),
                "C_ci_lower": float(np.percentile(boot_C_arr, 2.5)),
                "C_ci_upper": float(np.percentile(boot_C_arr, 97.5)),
                "C_ci_median": float(np.median(boot_C_arr)),
                "R_ci_lower": float(np.percentile(boot_R_arr, 2.5)),
                "R_ci_upper": float(np.percentile(boot_R_arr, 97.5)),
                "R_ci_median": float(np.median(boot_R_arr)),
                "R_excludes_zero": bool(np.percentile(boot_R_arr, 2.5) > 0),
            }
            ui.ok(f"  Bootstrap ({len(boot_R)} iters, {_bt_elapsed:.0f}s): "
                  f"R = [{bootstrap_ci['R_ci_lower']:.3f}, {bootstrap_ci['R_ci_upper']:.3f}] "
                  f"{'(excludes zero ✓)' if bootstrap_ci['R_excludes_zero'] else '(includes zero ✗)'}")
        else:
            ui.warn(f"  Bootstrap: only {len(boot_R)} valid iterations — CI unreliable")
            bootstrap_ci = {"n_bootstrap": len(boot_R), "error": "insufficient valid iterations"}
    else:
        ui.msg("  Bootstrap CIs: skipped (insufficient data or NaN fractions)")
    ui.blank()

    # ── Held-out stage ────────────────────────────────────────────────
    heldout_results = {}
    h28_verdict     = "Run 0033 not yet available — H28 pending."

    if has_heldout:
        _analysis_status(34, "Held-out validation (H28)", 70, temperature=_temp34)
        ui.msg(f"Computing held-out permutation sensitivity "
               f"(n={len(rows_heldout)} rows, same fitted model, n_perm={N_PERMUTE_SOBOL})...")
        # RNG-PERM-CORR fix (v30.9): seeds 102/103/104 — independent from each other
        # and from the in-sample seeds 99/100/101.
        eff_E_h, std_E_h, bv_E_h = _permutation_sensitivity_on(rows_heldout, 'E',      mdl_train, sx_train, sy=sy_train, seed=102)
        ui.msg(f"  E held-out: {eff_E_h:.6f}  (std={std_E_h:.6f})")
        _check_pause()
        eff_C_h, std_C_h, bv_C_h = _permutation_sensitivity_on(rows_heldout, 'C',      mdl_train, sx_train, sy=sy_train, seed=103)
        ui.msg(f"  C held-out: {eff_C_h:.6f}  (std={std_C_h:.6f})")
        _check_pause()
        eff_R_h, std_R_h, bv_R_h = _permutation_sensitivity_on(rows_heldout, 'S_prev', mdl_train, sx_train, sy=sy_train, seed=104)
        ui.msg(f"  R held-out: {eff_R_h:.6f}  (std={std_R_h:.6f})")
        _check_pause()
        ui.blank()

        bv_h    = float(np.mean([bv_E_h, bv_C_h, bv_R_h]))
        total_h = eff_E_h + eff_C_h + eff_R_h
        if total_h > 0:
            frac_E_h = eff_E_h / total_h
            frac_C_h = eff_C_h / total_h
            frac_R_h = eff_R_h / total_h
        else:
            frac_E_h = frac_C_h = frac_R_h = float('nan')

        im_h = float(1.0 - total_h / bv_h) if bv_h > 0 else float('nan')

        no_nan_h = all(v == v for v in [frac_E_h, frac_C_h, frac_R_h, frac_E, frac_C, frac_R])
        if no_nan_h:
            max_div = max(abs(frac_E_h - frac_E),
                         abs(frac_C_h - frac_C),
                         abs(frac_R_h - frac_R))
            if max_div < 0.05:
                h28_status = "supported"
                h28_note   = "< 0.05 — partition externally validated"
            elif max_div > 0.10:
                h28_status = "disproven"
                h28_note   = "> 0.10 — in-sample fractions overfit"
            else:
                h28_status = "inconclusive"
                h28_note   = "borderline — interpret with caution"
            h28_verdict = (
                f"H28 {h28_status.upper()}: "
                f"held-out E={frac_E_h:.3f} C={frac_C_h:.3f} R={frac_R_h:.3f}  "
                f"vs in-sample E={frac_E:.3f} C={frac_C:.3f} R={frac_R:.3f}. "
                f"Max divergence={max_div:.3f} ({h28_note})."
            )
        else:
            h28_status  = "inconclusive"
            max_div     = float('nan')
            h28_verdict = "H28: NaN in fractions — insufficient held-out data."

        heldout_results = {
            "n_heldout": len(rows_heldout),
            "baseline_var_heldout": bv_h,
            "heldout_perm_sens_E": {"effect": eff_E_h, "std": std_E_h, "fraction": frac_E_h},
            "heldout_perm_sens_C": {"effect": eff_C_h, "std": std_C_h, "fraction": frac_C_h},
            "heldout_perm_sens_R": {"effect": eff_R_h, "std": std_R_h, "fraction": frac_R_h},
            "interaction_mass_heldout": im_h,
            "h28_max_fraction_divergence": float(max_div) if max_div == max_div else float('nan'),
            "h28_status": h28_status,
        }

    results = {
        "run_num": 34,
        "n_3way": len(rows_3way),
        "n_perm": N_PERMUTE_SOBOL,
        "baseline_var": baseline_var,
        # In-sample results (primary, backward compatible keys preserved)
        "perm_sens_E": {"effect": effect_E, "std": std_E, "fraction": frac_E},
        "perm_sens_C": {"effect": effect_C, "std": std_C, "fraction": frac_C},
        "perm_sens_R": {"effect": effect_R, "std": std_R, "fraction": frac_R},
        "sobol_E":     {"effect": effect_E, "std": std_E, "fraction": frac_E},
        "sobol_C":     {"effect": effect_C, "std": std_C, "fraction": frac_C},
        "sobol_R":     {"effect": effect_R, "std": std_R, "fraction": frac_R},
        "total_first_order_raw": total,
        "total_first_order_fraction": float(total / baseline_var) if baseline_var > 0 else float('nan'),
        "interaction_mass_unattributed": interaction_mass,
        "verdict": verdict,
        "bootstrap_ci": bootstrap_ci,
        "interpretation": (
            "In-sample fractions are renormalized fixed-model permutation sensitivity indices. "
            "NOT proper Sobol indices — proper Sobol (Saltelli 2002) requires independent "
            "Monte Carlo sampling from the joint distribution. This method may overestimate "
            "attributed variance because the model was trained on the permuted data. "
            "v16.0: heldout_perm_sens_* uses the same fitted model on Run 0033 data "
            "(trajectories the Ridge model never saw). H28: if held-out fractions match "
            "in-sample within ±0.05, the partition is externally validated. "
            "Divergence > 0.10 invalidates it. Interaction mass reported pre-renormalization."
        ),
        # Held-out results (v16.0 — present only if Run 0033 data available)
        **heldout_results,
        "h28_verdict": h28_verdict,
    }

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    ui.blank()
    ui.section("In-Sample Permutation Sensitivity Results")
    ui.msg(f"  E (external input):      {frac_E:.3f}  ({effect_E:.6f} raw)")
    ui.msg(f"  C (constraint pressure): {frac_C:.3f}  ({effect_C:.6f} raw)")
    ui.msg(f"  R (internal state share):  {frac_R:.3f}  ({effect_R:.6f} raw)")
    ui.blank()
    ui.msg(f"  Baseline var:     {baseline_var:.6f}")
    ui.msg(f"  First-order mass: {float(total/baseline_var) if baseline_var > 0 else float('nan'):.3f}")
    ui.msg(f"  Interaction mass: {interaction_mass:.3f}")
    ui.blank()
    ui.msg(f"  H22 VERDICT: {verdict}")

    if has_heldout:
        ui.blank()
        ui.section("Held-Out Results (H28)")
        ui.msg(f"  E held-out: {frac_E_h:.3f}  (in-sample: {frac_E:.3f}  Δ={abs(frac_E_h-frac_E):.3f})")
        ui.msg(f"  C held-out: {frac_C_h:.3f}  (in-sample: {frac_C:.3f}  Δ={abs(frac_C_h-frac_C):.3f})")
        ui.msg(f"  R held-out: {frac_R_h:.3f}  (in-sample: {frac_R:.3f}  Δ={abs(frac_R_h-frac_R):.3f})")
        ui.blank()
        ui.msg(f"  H28 VERDICT: {h28_verdict}")

    ui.blank()
    ui.ok(f"Saved: {out_file}")


# ══════════════════════════════════════════════════════════════════
# RUN 40 — POOLED CROSS-DIRECTORY E+C+R DECOMPOSITION + SOBOL
# ══════════════════════════════════════════════════════════════════
# Standalone post-hoc run. Walks all temperature directories under
# base/{family}/{size}/{variant}/, pools quadruplets, then runs the
# full decomposition (Run 0042 logic) and Sobol partition (Run 0043 logic)
# on the combined dataset. Outputs to pooled/ directory.
# Run once after all temperature rounds are complete.

def _run_pooled(session, paths):
    from cartography import DATA, get_pooled_paths, get_family_size_dir

    family     = session.get('model_family', 'llama')
    size       = session.get('model_size', '8b')
    variant    = session.get('model_variant', 'abliterated')
    # BUG-FOUND-5 (fixed v36.5): 'instruct' fallback → 'abliterated'.
    # _run_pooled walks base/{family}/{size}/{variant}/ — wrong variant = zero conditions found.
    model_name = session.get('model_name', '')

    pooled_paths = get_pooled_paths(family, size, variant)
    out_dir = pooled_paths['analysis']
    out_33  = os.path.join(out_dir, "Q0051_pooled_decomposition.json")
    out_34  = os.path.join(out_dir, "Q0051_pooled_sobol.json")

    ui.section("Run 0051 — Pooled E+C+R Decomposition + Permutation Sensitivity (all rounds)")
    ui.msg("Walks all temperature directories. Pools quadruplets. No GPU.")
    ui.blank()

    base_variant_dir = os.path.join(get_family_size_dir(family, size), variant)
    if not os.path.isdir(base_variant_dir):
        choice = ui.srx_prompt(
            f"No data directory found: {base_variant_dir}\n"
            "  Temperature rounds (Run 0003) must be collected before pooling."
        )
        if choice in ('s', 'x'):
            if choice == 'x': raise SystemExit
            return

    all_rows_3way = []
    all_rows_2way = []
    conditions_found = []

    for cond in sorted(os.listdir(base_variant_dir)):
        if cond == 'pooled':
            continue
        hidden_dir = os.path.join(base_variant_dir, cond, 'hidden_states')
        if not os.path.isdir(hidden_dir):
            continue
        conditions_found.append(cond)
        ui.msg(f"  [{cond}]")
        rows_3 = _load_quadruplets(hidden_dir, SOURCE_RUNS_3WAY, model_name, require_ct=True)
        rows_3 = [r for r in rows_3 if r['has_real_ct'] and r['c_t'] is not None]
        rows_2 = _load_quadruplets(hidden_dir, SOURCE_RUNS_2WAY, model_name, require_ct=False)
        for r in rows_3: r['source_condition'] = cond
        for r in rows_2: r['source_condition'] = cond
        all_rows_3way.extend(rows_3)
        all_rows_2way.extend(rows_2)
        ui.msg(f"    3-way: {len(rows_3)}   2-way: {len(rows_2)}")

    ui.blank()
    ui.msg(f"  Conditions pooled  : {len(conditions_found)}  ({', '.join(conditions_found)})")
    ui.msg(f"  Total 3-way rows   : {len(all_rows_3way)}")
    ui.msg(f"  Total 2-way rows   : {len(all_rows_2way)}")
    ui.blank()

    if len(conditions_found) < 3:
        ui.err(f"Run 0051 requires at least 3 temperature conditions. Found {len(conditions_found)}.")
        ui.msg("  Complete more temperature rounds before running the pooled decomposition.")
        return

    # ── Gate: require all 6 temperature rounds, override at 3+ ────────
    _REQUIRED_CONDS = {'deterministic', 'temp_0.2', 'temp_0.4', 'temp_0.6', 'temp_0.8', 'temp_1.0'}
    _missing_conds = _REQUIRED_CONDS - set(conditions_found)
    if _missing_conds:
        ui.warn(f"Run 0051 expects ALL 6 temperature rounds. Missing: {sorted(_missing_conds)}")
        ui.warn(f"  Proceeding with {len(conditions_found)}/6 rounds — results are preliminary.")
        ui.blank()

    if len(all_rows_3way) < 50:
        choice = ui.srx_prompt(
            f"Insufficient 3-way quadruplets for pooled analysis: {len(all_rows_3way)} found, 50 required.\n"
            "  Complete more temperature rounds (Run 0003 + Run 0023) before running Run 0051.\n"
            "  Running anyway will likely produce unreliable fractions."
        )
        if choice in ('s', 'x'):
            if choice == 'x': raise SystemExit
            return

    # ── Decomposition (Run 0042 logic) ──────────────────────────────
    # Stage resume: if decomposition JSON exists, skip OLS + perm tests
    _decomp_done = False
    if os.path.exists(out_33):
        try:
            with open(out_33) as _rd40:
                _d40 = json.load(_rd40)
            if _d40.get('r2_D_full_decomposition') is not None:
                _decomp_done = True
                ui.ok(f"  Decomposition already complete — skipping OLS + perm tests")
                results_33 = _d40
        except Exception:
            pass

    if not _decomp_done:
        ui.msg("Fitting four OLS models on pooled data...")
        m = _run_four_models(all_rows_3way, all_rows_2way)

        ui.msg(f"Running permutation tests (n={N_PERMUTE}) ...")
        d_int, p_int = _perm_test_3way(all_rows_3way, 'internal')
        d_con, p_con = _perm_test_3way(all_rows_3way, 'constraint')

        results_33 = {
            "run_num":              40,
            "label":                "pooled_decomposition",
            "conditions_pooled":    conditions_found,
            "n_conditions":         len(conditions_found),
            "n_3way":               len(all_rows_3way),
            "n_2way":               len(all_rows_2way),
            "r2_A_external_only":             m['r2_A'],
            "r2_B_external_plus_internal":    m['r2_B'],
            "r2_C_external_plus_constraint":  m['r2_C'],
            "r2_D_full_decomposition":        m['r2_D'],
            "delta_r2_internal":   {"value": float(d_int), "p_value": float(p_int),
                                    "significant": bool(p_int < ALPHA)},
            "delta_r2_constraint": {"value": float(d_con), "p_value": float(p_con),
                                    "significant": bool(p_con < ALPHA)},
            "legacy_2way":         {"r2_A": m['r2_A_leg'], "r2_B": m['r2_B_leg'],
                                    "delta": m['delta_legacy']},
            "verdict": {
                "r_is_separable_from_c": bool(p_int < ALPHA and d_int > 0.01),
                "c_is_measurable":       bool(p_con < ALPHA and d_con > 0.01),
                "conclusion":            _conclusion(p_int, d_int, p_con, d_con),
            },
        }

        with open(out_33, 'w') as f:
            json.dump(results_33, f, indent=2, default=str)

        ui.blank()
        ui.section("Pooled Decomposition Results")
        ui.msg(f"  R²_A (E only):     {m['r2_A']:.4f}")
        ui.msg(f"  R²_B (E + S):      {m['r2_B']:.4f}")
        ui.msg(f"  R²_C (E + C):      {m['r2_C']:.4f}")
        ui.msg(f"  R²_D (E + C + S):  {m['r2_D']:.4f}")
        ui.blank()
        si = "✓ SIGNIFICANT" if p_int < ALPHA else "✗ not significant"
        sc = "✓ SIGNIFICANT" if p_con < ALPHA else "✗ not significant"
        ui.msg(f"  ΔR²_internal:    {d_int:.4f}  p={p_int:.4f}  {si}")
        ui.msg(f"  ΔR²_constraint:  {d_con:.4f}  p={p_con:.4f}  {sc}")
        ui.blank()
        ui.msg(f"  VERDICT: {results_33['verdict']['conclusion']}")

        ui.ok(f"Saved: {out_33}")
    _check_pause()

    # ── Permutation sensitivity partition (Run 0043 logic) ────────────
    ui.blank()
    ui.section("Pooled Permutation Sensitivity Partition")
    ui.msg(f"  n_perm={N_PERMUTE_SOBOL} per component  |  {len(all_rows_3way)} pooled rows")
    ui.blank()

    # RNG-PERM-CORR fix (v30.9): seeds 42/43/44 — independent draws per component.
    effect_E, std_E, bv_E = _permutation_sensitivity(all_rows_3way, 'E',      N_PERMUTE_SOBOL, seed=42)
    ui.msg(f"  E effect: {effect_E:.6f}  (std={std_E:.6f})")
    _check_pause()
    effect_C, std_C, bv_C = _permutation_sensitivity(all_rows_3way, 'C',      N_PERMUTE_SOBOL, seed=43)
    ui.msg(f"  C effect: {effect_C:.6f}  (std={std_C:.6f})")
    _check_pause()
    effect_R, std_R, bv_R = _permutation_sensitivity(all_rows_3way, 'S_prev', N_PERMUTE_SOBOL, seed=44)
    ui.msg(f"  R effect: {effect_R:.6f}  (std={std_R:.6f})")
    _check_pause()
    ui.blank()

    baseline_var = float(np.mean([bv_E, bv_C, bv_R]))
    total = effect_E + effect_C + effect_R
    if total > 0:
        frac_E = effect_E / total
        frac_C = effect_C / total
        frac_R = effect_R / total
    else:
        frac_E = frac_C = frac_R = float('nan')

    interaction_mass = (
        float(1.0 - total / baseline_var) if baseline_var > 0 else float('nan')
    )

    no_nan = all(v == v for v in [frac_E, frac_C, frac_R])
    verdict_sobol = (
        f"E={frac_E:.3f}  C={frac_C:.3f}  R={frac_R:.3f}  "
        f"(pooled renormalized permutation sensitivity, {len(all_rows_3way)} rows, "
        f"{len(conditions_found)} temperature rounds). "
        f"Interaction mass: {interaction_mass:.3f}. "
        f"R fraction > 0.05: {frac_R > 0.05}."
    ) if no_nan else "Could not compute partition — check data."

    results_34 = {
        "run_num":                       40,
        "label":                         "pooled_perm_sens",
        "conditions_pooled":             conditions_found,
        "n_conditions":                  len(conditions_found),
        "n_3way":                        len(all_rows_3way),
        "n_perm":                        N_PERMUTE_SOBOL,
        "baseline_var":                  baseline_var,
        "perm_sens_E": {"effect": effect_E, "std": std_E, "fraction": frac_E},
        "perm_sens_C": {"effect": effect_C, "std": std_C, "fraction": frac_C},
        "perm_sens_R": {"effect": effect_R, "std": std_R, "fraction": frac_R},
        # Legacy keys preserved for backward compat
        "sobol_E": {"effect": effect_E, "std": std_E, "fraction": frac_E},
        "sobol_C": {"effect": effect_C, "std": std_C, "fraction": frac_C},
        "sobol_R": {"effect": effect_R, "std": std_R, "fraction": frac_R},
        "total_first_order_raw":         total,
        "total_first_order_fraction":    float(total / baseline_var) if baseline_var > 0 else float('nan'),
        "interaction_mass_unattributed": interaction_mass,
        "verdict":                       verdict_sobol,
        "interpretation": (
            "Pooled fixed-model permutation sensitivity indices across all completed temperature rounds. "
            "NOT proper Sobol indices — see Run 0043 interpretation for caveat. "
            "Each round contributes independently collected quadruplets. "
            "Higher N stabilises fraction estimates relative to per-round Run 0043. "
            "Fractions renormalized to sum to 1. Interaction mass reported separately."
        ),
    }

    with open(out_34, 'w') as f:
        json.dump(results_34, f, indent=2, default=str)

    ui.section("Pooled Permutation Sensitivity Results")
    ui.msg(f"  E (external input):      {frac_E:.3f}  ({effect_E:.6f} raw)")
    ui.msg(f"  C (constraint pressure): {frac_C:.3f}  ({effect_C:.6f} raw)")
    ui.msg(f"  R (internal state share):  {frac_R:.3f}  ({effect_R:.6f} raw)")
    ui.blank()
    ui.msg(f"  Baseline var:     {baseline_var:.6f}")
    ui.msg(f"  First-order mass: {float(total/baseline_var) if baseline_var > 0 else float('nan'):.3f}")
    ui.msg(f"  Interaction mass: {interaction_mass:.3f}")
    ui.blank()
    ui.msg(f"  VERDICT: {verdict_sobol}")
    ui.blank()
    ui.ok(f"Saved: {out_34}")

    # ── Cross-temperature E(T), C(T), R(T) synthesis ──────────────────
    # Read per-round Run 0043 results from each temperature directory.
    # Produces the temperature curve showing how the decomposition shifts.
    ui.blank()
    ui.section("Cross-Temperature Decomposition Curve")

    from cartography import get_paths as _gp_curve
    _TEMP_ORDER = [
        (0.0, 'deterministic'), (0.2, 'temp_0.2'), (0.4, 'temp_0.4'),
        (0.6, 'temp_0.6'),     (0.8, 'temp_0.8'),  (1.0, 'temp_1.0'),
    ]
    temp_curve = []
    for temp_val, cond_name in _TEMP_ORDER:
        round_paths = _gp_curve(family, size, variant, temp_val, create_dirs=False)
        r43_file = os.path.join(round_paths['analysis'], "Q0043_sobol_partition.json")
        r42_file = os.path.join(round_paths['analysis'], "Q0042_decomposition.json")
        entry = {"temperature": temp_val, "condition": cond_name}
        # Run 0043 fractions
        if os.path.exists(r43_file):
            try:
                with open(r43_file) as _f43:
                    d43 = json.load(_f43)
                entry["E"] = d43.get("perm_sens_E", {}).get("fraction", float('nan'))
                entry["C"] = d43.get("perm_sens_C", {}).get("fraction", float('nan'))
                entry["R"] = d43.get("perm_sens_R", {}).get("fraction", float('nan'))
                entry["n_3way"] = d43.get("n_3way", 0)
                entry["interaction_mass"] = d43.get("interaction_mass_unattributed", float('nan'))
                # Held-out fractions if available
                if "heldout_perm_sens_R" in d43:
                    entry["R_heldout"] = d43["heldout_perm_sens_R"].get("fraction", float('nan'))
                    entry["E_heldout"] = d43["heldout_perm_sens_E"].get("fraction", float('nan'))
                    entry["C_heldout"] = d43["heldout_perm_sens_C"].get("fraction", float('nan'))
                entry["r34_present"] = True
            except Exception as _e34:
                entry["r34_present"] = False
                entry["r34_error"] = str(_e34)
        else:
            entry["r34_present"] = False
        # Run 0042 ΔR² for complementary view
        if os.path.exists(r42_file):
            try:
                with open(r42_file) as _f42:
                    d42 = json.load(_f42)
                entry["delta_r2_internal"] = d42.get("delta_r2_internal", {}).get("value", float('nan'))
                entry["delta_r2_constraint"] = d42.get("delta_r2_constraint", {}).get("value", float('nan'))
                entry["r2_D"] = d42.get("r2_D_full_decomposition", float('nan'))
                entry["r33_present"] = True
            except Exception:
                entry["r33_present"] = False
        else:
            entry["r33_present"] = False
        temp_curve.append(entry)

    # Display the curve
    ui.msg(f"  {'T':>4}  {'E':>6}  {'C':>6}  {'R':>6}  {'R_held':>7}  {'ΔR²_int':>8}  {'N':>5}")
    ui.msg(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*5}")
    for e in temp_curve:
        if not e.get('r34_present'):
            ui.warn(f"  {e['temperature']:4.1f}  — Run 0043 not found —")
            continue
        _E = e.get('E', float('nan'))
        _C = e.get('C', float('nan'))
        _R = e.get('R', float('nan'))
        _Rh = e.get('R_heldout', float('nan'))
        _dr = e.get('delta_r2_internal', float('nan'))
        _n = e.get('n_3way', 0)
        _Rh_str = f"{_Rh:.3f}" if _Rh == _Rh else "  —  "
        _dr_str = f"{_dr:.4f}" if _dr == _dr else "   —   "
        ui.msg(f"  {e['temperature']:4.1f}  {_E:6.3f}  {_C:6.3f}  {_R:6.3f}  {_Rh_str:>7}  {_dr_str:>8}  {_n:5d}")
    ui.blank()

    # Compute trend statistics
    _temps = [e['temperature'] for e in temp_curve if e.get('r34_present')]
    _Rs = [e['R'] for e in temp_curve if e.get('r34_present') and e.get('R') == e.get('R')]
    r_trend = {}
    if len(_Rs) >= 3:
        _t_arr = np.array([e['temperature'] for e in temp_curve
                           if e.get('r34_present') and e.get('R') == e.get('R')])
        _r_arr = np.array(_Rs)
        # Linear trend
        if len(_t_arr) >= 2:
            _slope, _intercept = np.polyfit(_t_arr, _r_arr, 1)
            r_trend["linear_slope"] = float(_slope)
            r_trend["linear_intercept"] = float(_intercept)
        # Peak R
        _peak_idx = int(np.argmax(_r_arr))
        r_trend["R_min"] = float(np.min(_r_arr))
        r_trend["R_max"] = float(np.max(_r_arr))
        r_trend["R_range"] = float(np.max(_r_arr) - np.min(_r_arr))
        r_trend["R_peak_temperature"] = float(_t_arr[_peak_idx])
        r_trend["R_mean"] = float(np.mean(_r_arr))

        ui.msg(f"  R range:  {r_trend['R_min']:.3f} – {r_trend['R_max']:.3f}  "
               f"(Δ={r_trend['R_range']:.3f})")
        ui.msg(f"  R peak:   T={r_trend['R_peak_temperature']:.1f}  "
               f"(R={r_trend['R_max']:.3f})")
        ui.msg(f"  R mean:   {r_trend['R_mean']:.3f}")
        if 'linear_slope' in r_trend:
            direction = "increases" if r_trend['linear_slope'] > 0.005 else \
                        "decreases" if r_trend['linear_slope'] < -0.005 else "stable"
            ui.msg(f"  R trend:  {direction} with temperature "
                   f"(slope={r_trend['linear_slope']:.4f})")
    ui.blank()

    # Save cross-temperature results
    out_curve = os.path.join(out_dir, "Q0051_temperature_curve.json")
    curve_results = {
        "run_num": 40,
        "label": "cross_temperature_curve",
        "n_temperatures": len(temp_curve),
        "curve": temp_curve,
        "r_trend": r_trend,
        "interpretation": (
            "Per-temperature E+C+R fractions read from each round's Q34_sobol_partition.json. "
            "Shows how the decomposition shifts across temperature. R(T) curve reveals whether "
            "internal state share strengthens, weakens, or peaks at intermediate temperature. "
            "Linear slope > 0.005 = R increases with T. Peak location identifies optimal "
            "temperature for internal trajectory expression."
        ),
    }
    with open(out_curve, 'w') as f:
        json.dump(curve_results, f, indent=2, default=str)
    ui.ok(f"Saved: {out_curve}")


# ══════════════════════════════════════════════════════════════════
# RUN 45 — POOL_DIM CALIBRATION SWEEP
# ══════════════════════════════════════════════════════════════════
# One-time-per-model calibration. Determines the optimal POOL_DIM
# by sweeping [64, 128, 256, 512, 1024, 2048, 4096] and finding
# where R fraction plateaus. All subsequent analysis runs inherit
# the result. Only runs at T=0.0 (deterministic).

_POOL_SWEEP_DIMS = [64, 128, 256, 512, 1024, 2048, 4096]
_SWEEP_N_PERMUTE_OLS  = 200   # stability test, not significance
_SWEEP_N_PERMUTE_SENS = 100


def load_optimal_pool_dim(ana_dir, temperature=None):
    """Read optimal POOL_DIM from Q45 calibration.

    v0.65.1.2 fix: old Q45 JSONs (pre-dim₉₅) report optimal_pool_dim=4096
    from the broken plateau detection. This function now computes dim₉₅ on
    the fly from the stored results array if the dim_95 key is missing.
    No rerun of Run 0041 required.
    """
    f = os.path.join(ana_dir, "Q0041_pool_dim_sweep.json")
    if not os.path.exists(f):
        return None
    try:
        with open(f) as fp:
            d = json.load(fp)

        # Reject incremental saves only if results are incomplete
        if d.get('status') == 'running':
            _partial_results = d.get('results', [])
            _partial_valid = [r for r in _partial_results if 'error' not in r
                              and r.get('frac_R') is not None
                              and r.get('frac_R') == r.get('frac_R')]
            if len(_partial_valid) < 3:
                # Too few dims completed — genuinely incomplete
                return None
            # Enough results to compute dim₉₅ — fall through to old-format path

        # New format (v0.65+): dim_95 key present
        if 'dim_95' in d and d['dim_95'] is not None:
            return int(d['dim_95'])

        # Old format: compute dim₉₅ from results array
        results = d.get('results', [])
        valid = [r for r in results if 'error' not in r
                 and r.get('frac_R') is not None
                 and r.get('frac_R') == r.get('frac_R')]  # NaN check
        if not valid:
            val = d.get('optimal_pool_dim')
            return int(val) if val is not None else None

        R_max = max(r['frac_R'] for r in valid)
        threshold = 0.95 * R_max
        for r in valid:
            if r['frac_R'] >= threshold:
                return int(r['pool_dim'])

        # Fallback to argmax
        best = max(valid, key=lambda r: r['frac_R'])
        return int(best['pool_dim'])
    except Exception:
        return None


def _ensure_calibrated(session, paths):
    """Check for Q45 calibration. Returns True if calibrated, False if not.
    When False, caller should trigger Run 0041 or abort.
    Q45 runs per-temperature — each temp directory needs its own calibration."""
    ana_dir = paths['analysis']
    dim = load_optimal_pool_dim(ana_dir)
    if dim is not None:
        global POOL_DIM
        POOL_DIM = dim
        return True
    return False


def _repool_rows(rows_full, dim, full_dim=None):
    """Re-pool pre-loaded full-dimension rows to a smaller POOL_DIM in memory."""
    if full_dim is None:
        # Detect from first row
        full_dim = len(rows_full[0]['s_prev']) if rows_full else 4096
    if dim >= full_dim:
        return rows_full
    idx = np.linspace(0, full_dim - 1, dim, dtype=int)
    out = []
    for r in rows_full:
        out.append({
            's_prev': r['s_prev'][idx],
            's_next': r['s_next'][idx],
            'e_t':    r['e_t'][idx],
            'c_t':    r['c_t'][idx],
            'has_real_ct':    r['has_real_ct'],
            'prompt_tokens':  r.get('prompt_tokens', 0.0),
            'run_num':        r.get('run_num'),
            'trial':          r.get('trial'),
            'turn':           r.get('turn'),
        })
    return out


def _sweep_one_dim_mem(dim, rows_full):
    """Ridge partition at one POOL_DIM. No OLS, no permutation tests.
    Run 0041 tests stability (shape of curve), not significance.
    One Ridge fit + 3 single-shuffle R² drops per dimension."""
    rows_3way = _repool_rows(rows_full, dim)

    if len(rows_3way) < 50:
        return {'pool_dim': dim, 'error': 'insufficient_data', 'n_3way': len(rows_3way)}

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score as _r2s

    Xe = np.stack([r['e_t']    for r in rows_3way])
    Xs = np.stack([r['s_prev'] for r in rows_3way])
    Xc = np.stack([r['c_t']    for r in rows_3way])
    y  = np.stack([r['s_next'] for r in rows_3way])
    Xp = np.array([r.get('prompt_tokens', 0.0) for r in rows_3way],
                  dtype=np.float32).reshape(-1, 1)
    Xp[np.isnan(Xp)] = float(np.nanmean(Xp)) if not np.all(np.isnan(Xp)) else 0.0

    full_X = np.hstack([Xe, Xc, Xs, Xp])
    sx = StandardScaler()
    sy = StandardScaler()
    full_Xs = sx.fit_transform(full_X)
    ys = sy.fit_transform(y)
    mdl = Ridge(alpha=0.01).fit(full_Xs, ys)
    baseline_pred = mdl.predict(full_Xs)
    baseline_r2 = float(_r2s(ys, baseline_pred))
    baseline_var = float(np.var(baseline_pred))

    # 10 shuffles per component for stable-ish fractions (still fast)
    dim_e = Xe.shape[1]
    dim_c = Xc.shape[1]
    dim_s = Xs.shape[1]

    def _avg_drop(col_start, col_end, n_shuf=10):
        rng = np.random.default_rng(col_start + dim)
        drops = []
        for _ in range(n_shuf):
            pX = full_Xs.copy()
            pX[:, col_start:col_end] = pX[rng.permutation(len(rows_3way)), col_start:col_end]
            drops.append(baseline_r2 - float(_r2s(ys, mdl.predict(pX))))
        return max(float(np.mean(drops)), 0.0)

    effect_E = _avg_drop(0, dim_e)
    effect_C = _avg_drop(dim_e, dim_e + dim_c)
    effect_R = _avg_drop(dim_e + dim_c, dim_e + dim_c + dim_s)

    total = effect_E + effect_C + effect_R
    if total > 0:
        frac_E = effect_E / total
        frac_C = effect_C / total
        frac_R = effect_R / total
    else:
        frac_E = frac_C = frac_R = float('nan')

    interaction_mass = float(1.0 - total / baseline_var) if baseline_var > 0 else float('nan')

    return {
        'pool_dim': dim,
        'n_3way': len(rows_3way),
        'r2_full': baseline_r2,
        'frac_E': frac_E, 'frac_C': frac_C, 'frac_R': frac_R,
        'effect_E': effect_E, 'effect_C': effect_C, 'effect_R': effect_R,
        'baseline_var': baseline_var,
        'interaction_mass': interaction_mass,
    }


def _run_pool_sweep(session, paths):
    global POOL_DIM

    temp = session.get('temperature', 0.0)

    hidden_dir = paths['hidden']
    ana_dir    = paths['analysis']
    model_name = session.get('model_name', '')
    os.makedirs(ana_dir, exist_ok=True)
    out_file = os.path.join(ana_dir, "Q0041_pool_dim_sweep.json")

    ui.section("Run 0041 — POOL_DIM Calibration Sweep")
    ui.msg("One-time calibration per model family. Determines optimal POOL_DIM.")
    ui.msg(f"Dimensions: {_POOL_SWEEP_DIMS}")
    ui.msg(f"Reduced permutation counts: OLS={_SWEEP_N_PERMUTE_OLS}, sensitivity={_SWEEP_N_PERMUTE_SENS}")
    ui.msg("3-way rows only (matches Run 0043 methodology). No GPU.")
    ui.blank()

    original_pool_dim = POOL_DIM

    # Load full-dimension data ONCE (max POOL_DIM = no downsampling).
    # All subsequent dimensions re-pool in memory — zero disk I/O after this.
    # v0.71.0.6: detect actual hidden dim from data (Gemma=3584, LLaMA=4096).
    POOL_DIM = 99999  # effectively "no pooling" — load raw vectors
    ui.msg("  Loading quadruplets at full dimension...")
    rows_full_raw = _load_quadruplets(hidden_dir, SOURCE_RUNS_3WAY, model_name, require_ct=True)
    rows_full = [r for r in rows_full_raw if r['has_real_ct'] and r['c_t'] is not None]

    # Detect actual hidden dimension from loaded data
    _actual_hidden_dim = 4096  # fallback
    if rows_full:
        _actual_hidden_dim = len(rows_full[0]['s_prev'])
    ui.msg(f"  Detected hidden dimension: {_actual_hidden_dim}")

    # Filter sweep dims to only include dims <= actual hidden dim
    sweep_dims = [d for d in _POOL_SWEEP_DIMS if d <= _actual_hidden_dim]
    if _actual_hidden_dim not in sweep_dims:
        sweep_dims.append(_actual_hidden_dim)
        sweep_dims.sort()
    ui.msg(f"  {len(rows_full)} 3-way rows loaded. Sweeping {len(sweep_dims)} dimensions: {sweep_dims}")
    ui.blank()

    # Sequential: each dim does 1 Ridge fit + 30 shuffle-predicts. Fast.
    # v0.71.0.11: Resume from partial results if interrupted.
    _temp45 = session.get('temperature', 0.0)
    results = []
    _resumed_dims = set()
    if os.path.exists(out_file):
        try:
            with open(out_file) as _rf:
                _parsed = json.load(_rf)
            if _parsed.get('results'):
                for r in _parsed['results']:
                    if 'pool_dim' in r:
                        results.append(r)
                        _resumed_dims.add(r['pool_dim'])
                if _resumed_dims:
                    ui.ok(f"  Resuming: {len(_resumed_dims)} dims already complete: "
                          f"{sorted(_resumed_dims)}")
        except Exception:
            ui.warn(f"  Could not read partial Q45 — starting fresh")

    for i, dim in enumerate(sweep_dims, 1):
        if dim in _resumed_dims:
            ui.msg(f"  [{i}/{len(sweep_dims)}] POOL_DIM = {dim}... (cached)")
            continue
        pct = int(i / len(sweep_dims) * 100)
        _analysis_status(45, f"Dim {i}/{len(sweep_dims)} ({dim})", pct,
                         f"POOL_DIM={dim}", _temp45)
        ui.msg(f"  [{i}/{len(sweep_dims)}] POOL_DIM = {dim}...")
        r = _sweep_one_dim_mem(dim, rows_full)
        results.append(r)
        if 'error' not in r:
            ui.msg(f"    R²={r['r2_full']:.4f}  E={r['frac_E']:.3f}  "
                   f"C={r['frac_C']:.3f}  R={r['frac_R']:.3f}")
        # Incremental save — partial results visible + crash-safe
        try:
            _partial = {'run_num': 45, 'status': 'running',
                        'dims_complete': i, 'dims_total': len(sweep_dims),
                        'results': results}
            with open(out_file, 'w') as _pf:
                json.dump(_partial, _pf, indent=2, default=str)
        except Exception:
            pass
        _check_pause()

    # Restore original while we compute verdict
    POOL_DIM = original_pool_dim

    # Sort results by pool_dim (resumed + new may be out of order)
    results.sort(key=lambda r: r.get('pool_dim', 0))

    # ── Summary ──────────────────────────────────────────────────────
    ui.section("POOL_DIM Sweep — Summary")
    ui.msg(f"  {'DIM':>5s}  {'R²':>6s}  {'E':>5s}  {'C':>5s}  {'R':>5s}  {'inter':>6s}")
    ui.msg(f"  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
    for r in results:
        if 'error' in r:
            ui.msg(f"  {r['pool_dim']:>5d}  — insufficient data —")
            continue
        ui.msg(f"  {r['pool_dim']:>5d}  {r['r2_full']:>6.4f}  "
               f"{r['frac_E']:>5.3f}  {r['frac_C']:>5.3f}  {r['frac_R']:>5.3f}  "
               f"{r['interaction_mass']:>6.3f}")

    # ── dim₉₅ selection: smallest dim where R >= 95% of peak R ────────
    # The plateau detection (pre-v0.65) assumed fractions converge monotonically
    # with dimension. They don't — R swings by 0.2+ between adjacent dims because
    # linspace subsampling selects different neurons at each dim. The default-to-max
    # fallback (4096) often reports LOWER R than the actual peak.
    #
    # dim₉₅ is pre-registered, mechanical, not cherry-pickable, same for every
    # condition. It picks a dimension in the stable region near the peak, not the
    # peak itself (which could be a spike). All hypothesis thresholds evaluate at
    # R(dim₉₅) — same scale as the original R fractions, no reinterpretation needed.
    valid = [r for r in results if 'error' not in r]
    r_fracs = [r['frac_R'] for r in valid if r.get('frac_R') == r.get('frac_R')]

    if valid and r_fracs:
        R_max    = max(r_fracs)
        dim_star = valid[r_fracs.index(R_max)]['pool_dim']  # argmax dim
        threshold = 0.95 * R_max

        # Smallest dim where R >= 95% of peak
        dim_95       = dim_star  # fallback to argmax
        R_at_dim95   = R_max
        for r in valid:
            if r.get('frac_R', 0) >= threshold:
                dim_95     = r['pool_dim']
                R_at_dim95 = r['frac_R']
                break

        r_range = max(r_fracs) - min(r_fracs)
        r_std   = float(np.std(r_fracs))

        if r_range < 0.03:
            verdict = (f"R fraction stable across dimensions (range={r_range:.4f}). "
                       f"dim₉₅={dim_95}, R*={R_at_dim95:.3f}.")
        elif r_range < 0.10:
            verdict = (f"R fraction moderately dimension-sensitive (range={r_range:.4f}). "
                       f"Peak R={R_max:.3f} at dim={dim_star}. "
                       f"dim₉₅={dim_95}, R*={R_at_dim95:.3f}.")
        else:
            verdict = (f"R fraction dimension-sensitive (range={r_range:.4f}). "
                       f"Peak R={R_max:.3f} at dim={dim_star}. "
                       f"dim₉₅={dim_95}, R*={R_at_dim95:.3f}.")

        optimal_dim = dim_95
    else:
        R_max = R_at_dim95 = r_range = r_std = float('nan')
        dim_star = dim_95 = sweep_dims[-1]
        optimal_dim = dim_95
        verdict = "Insufficient sweep points for dim₉₅ selection."

    ui.blank()
    ui.msg(f"  R_max:    {R_max:.3f}  at dim={dim_star}")
    ui.msg(f"  dim₉₅:   {dim_95}  (smallest dim ≥ 95% of peak)")
    ui.msg(f"  R*:      {R_at_dim95:.3f}  (R at dim₉₅ — used for all stats)")
    ui.msg(f"  VERDICT: {verdict}")
    ui.blank()

    # Set globally for this session
    POOL_DIM = optimal_dim

    sweep_output = {
        'run_num': 45,
        'pool_dims_tested': sweep_dims,
        'hidden_dim': _actual_hidden_dim,
        'results': results,
        'optimal_pool_dim': optimal_dim,
        'dim_95': dim_95,
        'R_at_dim95': float(R_at_dim95),
        'R_at_peak': float(R_max),
        'R_max': float(R_max),
        'dim_star': dim_star,
        'dim95_ratio': float(R_at_dim95 / R_max) if (R_max == R_max and R_max > 0) else float('nan'),
        'r_fraction_range': float(r_range) if r_range == r_range else float('nan'),
        'r_fraction_std':   float(r_std) if r_std == r_std else float('nan'),
        'verdict': verdict,
        'dim95_rationale': (
            f"dim₉₅ = {dim_95}: smallest dimension where R ≥ 95% of peak R. "
            f"Peak R = {R_max:.4f} at dim={dim_star}. R at dim₉₅ = {R_at_dim95:.4f}. "
            f"Ratio = {float(R_at_dim95/R_max) if (R_max==R_max and R_max>0) else float('nan'):.4f} (≥ 0.95 by construction)."
        ),
    }

    with open(out_file, 'w') as f:
        json.dump(sweep_output, f, indent=2, default=str)
    ui.ok(f"Saved: {out_file}")
    ui.ok(f"All subsequent analysis runs will use POOL_DIM={optimal_dim} (dim₉₅).")


# ══════════════════════════════════════════════════════════════════
# RUN 46 — PER-SOURCE-RUN R FRACTIONS
# ══════════════════════════════════════════════════════════════════
# Answers: "Which conditions produce the highest R?"
# Same quadruplet pool as Run 0043, but subsetted by source run.
# Per-subset permutation sensitivity → per-condition E+C+R fractions.
# No GPU. Runs at any temperature. Uses current round's data.

_MIN_QUADRUPLETS_PER_RUN = 30  # minimum for meaningful permutation test

def _run_per_condition_r(session, paths, force_dim=None, run_label=44, out_name="Q0044_per_condition_R.json"):
    global POOL_DIM
    _saved_dim = POOL_DIM
    if force_dim is not None:
        POOL_DIM = force_dim

    hidden_dir = paths['hidden']
    ana_dir    = paths['analysis']
    model_name = session.get('model_name', '')
    os.makedirs(ana_dir, exist_ok=True)
    out_file = os.path.join(ana_dir, out_name)

    _dim_note = f" (forced POOL_DIM={force_dim})" if force_dim else ""
    ui.section(f"Run {run_label} — Per-Source-Run R Fractions{_dim_note} (no GPU)")
    ui.msg("Same quadruplet pool as Run 0043, subsetted by source run.")
    ui.msg("Answers: which conditions produce the highest R?")
    if force_dim:
        ui.msg(f"  Fixed dimension: POOL_DIM={force_dim} (overrides dim₉₅={_saved_dim})")
    ui.blank()

    # Load quadruplets — same logic as Run 0043
    ui.msg("Loading 3-way quadruplets...")
    rows_3way_raw = _load_quadruplets(hidden_dir, SOURCE_RUNS_3WAY, model_name, require_ct=True)
    rows_3way = [r for r in rows_3way_raw if r['has_real_ct'] and r['c_t'] is not None]

    if len(rows_3way) < _MIN_QUADRUPLETS_PER_RUN:
        ui.err(f"Insufficient quadruplets: {len(rows_3way)} (need {_MIN_QUADRUPLETS_PER_RUN})")
        ui.msg("  Run 0042/34 first to ensure quadruplet data is available.")
        return

    ui.msg(f"  Total 3-way quadruplets: {len(rows_3way)}")
    ui.blank()

    # Group by source run
    by_run = {}
    for r in rows_3way:
        rn = r.get('run_num', 0)
        by_run.setdefault(rn, []).append(r)

    # RUN_LABELS for readable output. v0.79.4.0: renumbered.
    _LABELS = {
         6: "Introspection A (direct)",      7: "Introspection B (memory)",
         8: "Introspection C (enforcer)",   13: "Priming neutral",
        14: "Priming cooperative",          15: "Priming resistant",
         1: "Null trivariant",               3: "Temperature grid",
        23: "Confound isolation",
    }

    ui.msg(f"  {'Run':>4}  {'Label':<32}  {'N':>5}  {'E':>6}  {'C':>6}  {'R':>6}")
    ui.msg(f"  {'─'*4}  {'─'*32}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}")

    # Resume: load completed conditions from partial
    per_run_results = []
    _completed_runs = set()
    if os.path.exists(out_file):
        try:
            with open(out_file) as _rpf:
                _partial46 = json.load(_rpf)
            # v0.75.1.0: removed status=='running' gate.
            if _partial46.get('per_run'):
                per_run_results = _partial46['per_run']
                _completed_runs = {r['run_num'] for r in per_run_results}
                if _completed_runs:
                    ui.ok(f"  Resuming: {len(_completed_runs)} conditions already complete")
        except Exception:
            pass

    _n_runs46 = len(by_run)
    _temp46 = session.get('temperature', 0.0)
    for _ri46, rn in enumerate(sorted(by_run.keys()), 1):
        if rn in _completed_runs:
            _existing = next((r for r in per_run_results if r['run_num'] == rn), None)
            _lbl = _existing.get('label', f'Run {rn}') if _existing else f'Run {rn}'
            ui.msg(f"  {rn:4d}  {_lbl:<32}  (cached)")
            continue
        _analysis_status(run_label, f"Condition {_ri46}/{_n_runs46}", int(_ri46/_n_runs46*100),
                         _LABELS.get(rn, f"Run {rn}"), _temp46)
        rows = by_run[rn]
        label = _LABELS.get(rn, f"Run {rn}")
        if len(rows) < _MIN_QUADRUPLETS_PER_RUN:
            ui.msg(f"  {rn:4d}  {label:<32}  {len(rows):5d}  — insufficient data —")
            per_run_results.append({
                "run_num": rn, "label": label, "n_quadruplets": len(rows),
                "skipped": True, "reason": f"< {_MIN_QUADRUPLETS_PER_RUN} quadruplets"
            })
            continue

        # Permutation sensitivity per source run
        n_perm = min(N_PERMUTE_SOBOL, 100)  # faster per-subset, still meaningful
        eff_E, std_E, bv_E = _permutation_sensitivity(rows, 'E',      n_perm, seed=42)
        eff_C, std_C, bv_C = _permutation_sensitivity(rows, 'C',      n_perm, seed=43)
        eff_R, std_R, bv_R = _permutation_sensitivity(rows, 'S_prev', n_perm, seed=44)

        total = eff_E + eff_C + eff_R
        if total > 0:
            fE, fC, fR = eff_E/total, eff_C/total, eff_R/total
        else:
            fE = fC = fR = float('nan')

        ui.msg(f"  {rn:4d}  {label:<32}  {len(rows):5d}  {fE:6.3f}  {fC:6.3f}  {fR:6.3f}")

        per_run_results.append({
            "run_num": rn, "label": label, "n_quadruplets": len(rows),
            "skipped": False,
            "frac_E": fE, "frac_C": fC, "frac_R": fR,
            "effect_E": eff_E, "effect_C": eff_C, "effect_R": eff_R,
            "std_E": std_E, "std_C": std_C, "std_R": std_R,
            "n_perm": n_perm,
        })
        _check_pause()
        # Incremental save after each condition
        try:
            _inc46 = {'run_num': run_label, 'status': 'running',
                      'pool_dim': POOL_DIM, 'per_run': per_run_results}
            with open(out_file, 'w') as _wf46:
                json.dump(_inc46, _wf46, indent=2, default=str)
        except Exception:
            pass

    ui.blank()

    # Sort by R fraction for summary
    ranked = sorted([r for r in per_run_results if not r.get('skipped')],
                    key=lambda r: r.get('frac_R', 0), reverse=True)
    if ranked:
        ui.section("R Ranking (highest to lowest)")
        for r in ranked:
            ui.msg(f"  R={r['frac_R']:.3f}  {r['label']}  (N={r['n_quadruplets']})")
        ui.blank()
        ui.msg(f"  Highest R: {ranked[0]['label']} ({ranked[0]['frac_R']:.3f})")
        ui.msg(f"  Lowest R:  {ranked[-1]['label']} ({ranked[-1]['frac_R']:.3f})")
        ui.msg(f"  Range:     {ranked[0]['frac_R'] - ranked[-1]['frac_R']:.3f}")

    # Run 0023 constraint variance correction (v0.71.0.0)
    # Run 0023 (was R28) has 3 system prompt conditions → ~2-3x the quadruplets.
    # Subsample to match median N and recompute.
    run23_correction = {}
    r23_entry = next((r for r in per_run_results if r.get('run_num') == 23 and not r.get('skipped')), None)
    if r23_entry:
        other_ns = [r['n_quadruplets'] for r in per_run_results
                    if r.get('run_num') != 23 and not r.get('skipped') and r['n_quadruplets'] > 0]
        if other_ns:
            median_n = int(np.median(other_ns))
            r23_n = r23_entry['n_quadruplets']
            if r23_n > median_n * 1.5:
                ui.msg(f"  Run 0023 has {r23_n} quadruplets vs median {median_n} — subsampling...")
                r23_rows = [r for r in rows_3way if r['run_num'] == 23]
                if len(r23_rows) >= median_n:
                    _sub_rng = np.random.default_rng(2323)
                    idx = _sub_rng.choice(len(r23_rows), size=median_n, replace=False)
                    r23_sub = [r23_rows[i] for i in idx]
                    try:
                        se23, _, _ = _permutation_sensitivity(r23_sub, 'E', n_perm=50, seed=230)
                        sc23, _, _ = _permutation_sensitivity(r23_sub, 'C', n_perm=50, seed=231)
                        sr23, _, _ = _permutation_sensitivity(r23_sub, 'S_prev', n_perm=50, seed=232)
                        t23 = se23 + sc23 + sr23
                        sub_R = sr23 / t23 if t23 > 0 else float('nan')
                        run23_correction = {
                            'run23_n_original': r23_n,
                            'run23_n_subsampled': median_n,
                            'run23_R_original': r23_entry['frac_R'],
                            'run23_R_subsampled': float(sub_R),
                            'median_other_n': median_n,
                            'note': (f"Run 0023 has {r28_n} quadruplets due to 3 system prompt conditions. "
                                     f"Subsampling to {median_n} (median of other runs) yields R = {sub_R:.3f} "
                                     f"(original R = {r23_entry['frac_R']:.3f}). "
                                     f"{'Low R is NOT a sample size artefact.' if abs(sub_R - r23_entry['frac_R']) < 0.05 else 'Sample size may affect R estimate.'}"),
                        }
                        ui.ok(f"  Run 0023 subsample R = {sub_R:.3f} (original {r23_entry['frac_R']:.3f})")
                    except Exception as _e28:
                        ui.warn(f"  Run 0023 subsample failed: {_e28}")

    results = {
        "run_num": 46,
        "label": "per_condition_R",
        "n_total_quadruplets": len(rows_3way),
        "n_source_runs": len(by_run),
        "pool_dim": POOL_DIM,
        "per_run": per_run_results,
        "ranking": [{"run_num": r["run_num"], "label": r["label"], "frac_R": r["frac_R"]}
                    for r in ranked] if ranked else [],
        "run23_n_correction": run23_correction,
        "interpretation": (
            "Per-source-run E+C+R fractions from the same quadruplet pool as Run 0043. "
            "Shows which experimental conditions produce the highest internal state share (R). "
            "Introspection conditions expected to show higher R than arithmetic or null. "
            "Priming conditions expected to show R variation by prime type. "
            "Fractions per subset are noisier than pooled Run 0043 (fewer quadruplets per group)."
        ),
    }

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    ui.blank()
    ui.ok(f"Saved: {out_file}")
    POOL_DIM = _saved_dim



# ══════════════════════════════════════════════════════════════════
# RUN 56 — MLP PERMUTATION SENSITIVITY DECOMPOSITION (v0.76.1.0)
# ══════════════════════════════════════════════════════════════════
# Parallel MLP-based E+C+R decomposition. Mirrors Run 0043 (pooled) and
# Run 0044 (per-condition) using MLPRegressor in place of Ridge.
#
# 56a (pooled)        — one MLP fit per (architecture, seed) on SOURCE_RUNS_3WAY.
#                       Mirrors Run 0043 Stage 1. Reads Q34_sobol_partition.json
#                       for ridge_reference and pool_dim.
# 56b (per-condition) — one MLP fit per (source_run, architecture, seed).
#                       Mirrors Run 0044. Reads Q46_per_condition_R.json for
#                       ridge_reference.
#
# Both passes use the same fit + permutation primitives. Permutation procedure
# matches Run 0043/46 exactly (row index shuffle, scalers frozen, R² drop on
# standardized target). Only the model class differs.
#
# Compute budget: ~3 architectures × 3 seeds × (1 fit + 3 components × 50 perms)
# per scope. Pooled ≈ 9 fits/temp. Per-condition ≈ 9 conditions × 9 fits ≈ 81
# fits/temp. On RTX 3080 CPU path at pool_dim=64-128 and n≈1300 rows/cond:
# roughly 30-60 min per temp. Incremental save after every (arch, seed).
#
# H50 relationship: H50 (linearity_check) already fits these three
# architectures at a single seed with train/test split for a scalar verdict.
# Run 0046 fits on full data (no holdout), 3 seeds, and runs full permutation
# sensitivity. The two are complementary: H50 is a linearity gate, Run 0046 is
# a quantitative fraction comparison.

N_PERMUTE_MLP = 50
_MLP_ARCHITECTURES = {
    "64":     {"hidden_layer_sizes": (64,),     "label": "MLP-64"},
    "256":    {"hidden_layer_sizes": (256,),    "label": "MLP-256"},
    "128-64": {"hidden_layer_sizes": (128, 64), "label": "MLP-128-64"},
}
_MLP_SEEDS = [42, 43, 44]


def _fit_mlp_decomposition(rows, arch_key, seed):
    """Fit one MLP on the [Xe, Xc, Xs, Xp] → y_next decomposition.

    Mirrors _fit_perm_model (Ridge) exactly: same feature layout, same
    StandardScaler fit on inputs and targets, same rows, no holdout.
    Returns (mdl, sx, sy, baseline_r2, baseline_var, training_meta).

    Prompt tokens (Xp) included as scalar covariate, held fixed during
    permutation downstream. Scalers fit ONCE here, re-used on permuted
    inputs by _mlp_permutation_sensitivity_on.

    training_meta records final epoch, loss tail, and a convergence flag
    so the batch summary can report cases where early stopping did not
    trigger within max_iter=500.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score as _r2

    cfg = _MLP_ARCHITECTURES[arch_key]

    Xe = np.stack([r['e_t']    for r in rows])
    Xc = np.stack([r['c_t']    for r in rows])
    Xs = np.stack([r['s_prev'] for r in rows])
    y  = np.stack([r['s_next'] for r in rows])
    Xp = np.array([r.get('prompt_tokens', 0.0) for r in rows],
                  dtype=np.float32).reshape(-1, 1)
    Xp[np.isnan(Xp)] = float(np.nanmean(Xp)) if not np.all(np.isnan(Xp)) else 0.0

    full_X = np.hstack([Xe, Xc, Xs, Xp])
    sx = StandardScaler()
    sy = StandardScaler()
    full_Xs = sx.fit_transform(full_X)
    ys = sy.fit_transform(y)

    # early_stopping=True carves its own 10% internal validation split from
    # full_Xs for stopping. That's fine — it's sklearn's default convergence
    # heuristic and is seed-reproducible via random_state.
    mdl = MLPRegressor(
        hidden_layer_sizes=cfg['hidden_layer_sizes'],
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        batch_size=64,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,     # matches handoff patience=20
        random_state=seed,
    )
    mdl.fit(full_Xs, ys)

    baseline_pred = mdl.predict(full_Xs)
    baseline_r2   = float(_r2(ys, baseline_pred))
    baseline_var  = float(np.var(baseline_pred))

    # Training diagnostics — converged flag fires when early stopping kicked
    # in before max_iter. If not converged we still use the model but flag it.
    _n_iter = int(getattr(mdl, 'n_iter_', 0))
    _loss_curve = getattr(mdl, 'loss_curve_', [])
    _val_curve  = getattr(mdl, 'validation_scores_', [])
    training_meta = {
        'final_iter': _n_iter,
        'converged': _n_iter < 500,
        'loss_tail': [float(v) for v in _loss_curve[-5:]] if _loss_curve else [],
        'val_score_tail': [float(v) for v in _val_curve[-5:]] if _val_curve else [],
    }
    return mdl, sx, sy, baseline_r2, baseline_var, training_meta


def _mlp_permutation_sensitivity_on(rows, component, mdl, sx, sy,
                                     n_perm=N_PERMUTE_MLP, seed=99):
    """Permutation sensitivity on a pre-fit MLP.

    Direct parallel of _permutation_sensitivity_on (Ridge). Same row
    indexing, same seed pattern via rng.integers, same sx/sy frozen,
    same R² drop metric on standardized target. Only mdl class differs.

    Returns (effect, std, neg_drop_count, per_iter_drops). per_iter_drops
    returned so the caller can log the distribution if needed.
    """
    from sklearn.metrics import r2_score as _r2

    Xe = np.stack([r['e_t']    for r in rows])
    Xc = np.stack([r['c_t']    for r in rows])
    Xs = np.stack([r['s_prev'] for r in rows])
    Xp = np.array([r.get('prompt_tokens', 0.0) for r in rows],
                  dtype=np.float32).reshape(-1, 1)
    Xp[np.isnan(Xp)] = float(np.nanmean(Xp)) if not np.all(np.isnan(Xp)) else 0.0
    y  = np.stack([r['s_next'] for r in rows])

    full_X  = np.hstack([Xe, Xc, Xs, Xp])
    full_Xs = sx.transform(full_X)
    ys      = sy.transform(y)
    baseline_pred = mdl.predict(full_Xs)
    baseline_r2   = float(_r2(ys, baseline_pred))

    rng = np.random.default_rng(seed)
    iter_seeds = rng.integers(0, 2**31, size=n_perm)

    drops = []
    neg_count = 0
    for iseed in iter_seeds:
        _rng = np.random.default_rng(int(iseed))
        perm_idx = _rng.permutation(len(rows))
        if component == 'E':
            perm_X = np.hstack([Xe[perm_idx], Xc, Xs, Xp])
        elif component == 'C':
            perm_X = np.hstack([Xe, Xc[perm_idx], Xs, Xp])
        else:  # 'S_prev'
            perm_X = np.hstack([Xe, Xc, Xs[perm_idx], Xp])
        perm_Xs   = sx.transform(perm_X)
        perm_pred = mdl.predict(perm_Xs)
        drop      = baseline_r2 - float(_r2(ys, perm_pred))
        if drop < 0:
            neg_count += 1
            drop = 0.0          # clip per handoff spec — prevents negative
                                # drops from inflating the denominator during
                                # renormalization
        drops.append(drop)

    return float(np.mean(drops)), float(np.std(drops)), neg_count, drops


def _mlp_one_arch_seed(rows, arch_key, seed, n_perm):
    """Full single (architecture, seed) evaluation: fit + permute E/C/R.

    Returns a dict matching the per-entry schema of Run 0046's mlp_results list.
    """
    # Deterministic per-seed permutation seeds — independent streams per
    # component so E/C/R row shuffles don't correlate.
    perm_seed_base = seed * 100 + 1000
    mdl, sx, sy, r2_full, _bv, tmeta = _fit_mlp_decomposition(rows, arch_key, seed)

    eE, sE, negE, _ = _mlp_permutation_sensitivity_on(
        rows, 'E',      mdl, sx, sy, n_perm=n_perm, seed=perm_seed_base + 1)
    eC, sC, negC, _ = _mlp_permutation_sensitivity_on(
        rows, 'C',      mdl, sx, sy, n_perm=n_perm, seed=perm_seed_base + 2)
    eR, sR, negR, _ = _mlp_permutation_sensitivity_on(
        rows, 'S_prev', mdl, sx, sy, n_perm=n_perm, seed=perm_seed_base + 3)

    total = eE + eC + eR
    if total > 0:
        fE, fC, fR = eE / total, eC / total, eR / total
    else:
        fE = fC = fR = float('nan')

    return {
        'architecture':  arch_key,
        'seed':          int(seed),
        'r2_full':       r2_full,
        'effect_E':      eE, 'effect_C': eC, 'effect_R': eR,
        'std_E':         sE, 'std_C':    sC, 'std_R':    sR,
        'frac_E':        fE, 'frac_C':   fC, 'frac_R':   fR,
        'neg_drop_count': {'E': negE, 'C': negC, 'S': negR},
        'training':      tmeta,
    }


def _mlp_summarize(entries, ridge_fracs):
    """Mean/std across (arch, seed) entries + gap vs ridge.

    entries: list of _mlp_one_arch_seed outputs, possibly including error entries.
    ridge_fracs: dict {'E': float, 'C': float, 'R': float} from Q34 or Q46.

    Skips entries that have an 'error' key (failed fits) and reports how many
    were skipped so downstream can flag incomplete summaries.
    """
    valid = [e for e in entries if 'error' not in e
             and all(not np.isnan(e.get(f'frac_{c}', float('nan')))
                     for c in 'ECR')]
    n_failed = len(entries) - len(valid)
    if not valid:
        return {
            'n_entries': len(entries), 'n_valid': 0, 'n_failed': n_failed,
            'mean': {'E': float('nan'), 'C': float('nan'), 'R': float('nan')},
            'std':  {'E': float('nan'), 'C': float('nan'), 'R': float('nan')},
            'r2_full_mean': float('nan'),
            'ridge_gap': {'E': float('nan'), 'C': float('nan'), 'R': float('nan')},
        }
    fE_arr = np.array([e['frac_E'] for e in valid])
    fC_arr = np.array([e['frac_C'] for e in valid])
    fR_arr = np.array([e['frac_R'] for e in valid])
    r2_arr = np.array([e['r2_full'] for e in valid])
    mean = {'E': float(np.mean(fE_arr)),
            'C': float(np.mean(fC_arr)),
            'R': float(np.mean(fR_arr))}
    std  = {'E': float(np.std(fE_arr)),
            'C': float(np.std(fC_arr)),
            'R': float(np.std(fR_arr))}
    gap = {c: (mean[c] - float(ridge_fracs.get(c, float('nan'))))
           for c in 'ECR'}
    return {
        'n_entries': len(entries), 'n_valid': len(valid), 'n_failed': n_failed,
        'mean': mean, 'std': std,
        'r2_full_mean': float(np.mean(r2_arr)),
        'ridge_gap': gap,
    }


def _load_ridge_reference_pooled(ana_dir):
    """Read Q34_sobol_partition.json and extract pooled fractions + R²."""
    q43 = os.path.join(ana_dir, "Q0043_sobol_partition.json")
    if not os.path.exists(q43):
        return None
    try:
        with open(q43) as f:
            d = json.load(f)
        fE = d.get('perm_sens_E', {}).get('fraction')
        fC = d.get('perm_sens_C', {}).get('fraction')
        fR = d.get('perm_sens_R', {}).get('fraction')
        if fE is None or fC is None or fR is None:
            return None
        # Run 0043 doesn't store r2_full explicitly; reconstruct from effects
        # relative to baseline_var if needed, else leave None.
        return {
            'E': float(fE), 'C': float(fC), 'R': float(fR),
            'n_3way':     d.get('n_3way'),
            'source':     'Q0043_sobol_partition.json',
        }
    except Exception:
        return None


def _load_ridge_reference_per_condition(ana_dir):
    """Read Q46_per_condition_R.json and return map {run_num: fracs_dict}."""
    q44 = os.path.join(ana_dir, "Q0044_per_condition_R.json")
    if not os.path.exists(q44):
        return {}
    try:
        with open(q44) as f:
            d = json.load(f)
        out = {}
        for entry in d.get('per_run', []):
            if entry.get('skipped'):
                continue
            rn = entry.get('run_num')
            if rn is None:
                continue
            fE = entry.get('frac_E'); fC = entry.get('frac_C'); fR = entry.get('frac_R')
            if fE is None or fC is None or fR is None:
                continue
            out[int(rn)] = {
                'E': float(fE), 'C': float(fC), 'R': float(fR),
                'label': entry.get('label', f'Run {rn}'),
                'n_quadruplets': entry.get('n_quadruplets'),
                'source': 'Q0044_per_condition_R.json',
            }
        return out
    except Exception:
        return {}


def _run_mlp_decomposition(session, paths,
                            n_perm=N_PERMUTE_MLP,
                            do_per_condition=True):
    """Run 0046 — MLP permutation sensitivity decomposition.

    Produces Q56_mlp_decomposition.json with two sections:
      'pooled'        — 56a mirror of Run 0043 (all SOURCE_RUNS_3WAY rows)
      'per_condition' — 56b mirror of Run 0044 (per source run)

    do_per_condition=False skips 56b entirely for compute budget. 56a alone is
    cheaper and often sufficient to confirm Ridge adequacy. Batch driver can
    toggle off after first model shows small gaps.

    Resume-safe: partial results flushed after every (arch, seed) completion.
    Re-entry loads partial, skips completed entries.

    Prerequisites:
      - Run 0042 complete at this temperature (quadruplets, POOL_DIM)
      - Run 0043 complete at this temperature (ridge_reference for pooled)
      - Run 0044 complete at this temperature (ridge_reference for per_condition)
        — only required if do_per_condition=True
    """
    hidden_dir = paths['hidden']
    ana_dir    = paths['analysis']
    os.makedirs(ana_dir, exist_ok=True)
    out_file   = os.path.join(ana_dir, "Q0046_mlp_decomposition.json")
    model_name = session.get('model_name', '')
    temp       = session.get('temperature', 0.0)

    ui.section("Run 0046 — MLP Permutation Sensitivity Decomposition (no GPU)")
    ui.msg("Parallel MLP decomposition. Mirrors Run 0043 (pooled) + Run 0044 (per-condition).")
    ui.msg(f"Architectures: {list(_MLP_ARCHITECTURES.keys())}  Seeds: {_MLP_SEEDS}  Perms/component: {n_perm}")
    if not do_per_condition:
        ui.msg("  Per-condition pass DISABLED — 56a pooled only.")
    ui.blank()

    # ── Load ridge references ────────────────────────────────────────────
    ridge_pooled = _load_ridge_reference_pooled(ana_dir)
    if ridge_pooled is None:
        ui.err("Run 0043 output not found or incomplete — 56a cannot compute ridge_gap.")
        ui.err("  Run 0043 must complete before Run 0046.")
        return
    ui.msg(f"  Ridge (Run 0043): E={ridge_pooled['E']:.3f}  C={ridge_pooled['C']:.3f}  R={ridge_pooled['R']:.3f}")

    # v0.79.6.1: Run 0044 is a hard prerequisite for Run 0046.
    # Per-condition MLP (56b) must run alongside pooled MLP (56a). A Q0046
    # that claims status='complete' but carries only pooled results is
    # misleading — master JSON ingestion picks it up, ridge_vs_mlp aggregates
    # get computed against an incomplete denominator, and the conjecture_1
    # verdict is evaluated on a subset. Per paper Sections 2.10 / 3.7 / 8.1,
    # the Ridge-vs-MLP comparison is load-bearing for the H50 finding and
    # must cover both pooled and per-condition surfaces. Ground-truth
    # alignment: code must not flip a run to 'complete' when its data isn't.
    # Refuse to proceed if Run 0044 output isn't present; user reruns 0044
    # first. The do_per_condition parameter is preserved for signature
    # compatibility but no longer gates the per-condition load.
    ridge_per_cond = _load_ridge_reference_per_condition(ana_dir)
    if not ridge_per_cond:
        ui.err("Run 0044 output not found or empty — per-condition Ridge reference unavailable.")
        ui.err("  Run 0046 requires BOTH pooled (56a) and per-condition (56b) passes.")
        ui.err("  Run 0044 must complete at this temperature before Run 0046 can proceed.")
        return
    do_per_condition = True
    ui.msg(f"  Ridge (Run 0044): {len(ridge_per_cond)} conditions available")
    ui.blank()

    # ── Resume ───────────────────────────────────────────────────────────
    partial = {}
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                partial = json.load(f)
            if partial.get('status') == 'complete':
                ui.ok("Run 0046 already complete — skipping.")
                return
            ui.msg(f"  Resuming from partial: pooled={len(partial.get('pooled',{}).get('results',[]))} entries, "
                   f"per_condition={len(partial.get('per_condition',{}) or {})} conditions")
        except Exception:
            partial = {}

    # ── Pooled data load ──────────────────────────────────────────────────
    ui.msg("Loading pooled quadruplets (SOURCE_RUNS_3WAY)...")
    rows_3way = _load_quadruplets(hidden_dir, SOURCE_RUNS_3WAY, model_name, require_ct=True)
    rows_3way = [r for r in rows_3way if r['has_real_ct'] and r['c_t'] is not None]
    ui.msg(f"  Pooled rows: {len(rows_3way)}")
    if len(rows_3way) < 50:
        ui.err(f"  Insufficient pooled rows ({len(rows_3way)} < 50) — aborting.")
        return
    ui.blank()

    # Build skeleton
    result = {
        'run_num': 56, 'status': 'running',
        'model_name':  model_name,
        'temperature': float(temp),
        'pool_dim':    int(POOL_DIM),
        'n_permutations': int(n_perm),
        'architectures':  list(_MLP_ARCHITECTURES.keys()),
        'seeds':          list(_MLP_SEEDS),
        'n_pooled_rows':  len(rows_3way),
        'pooled':         partial.get('pooled', {'ridge_reference': ridge_pooled,
                                                 'results': []}),
        'per_condition':  partial.get('per_condition', {}) if do_per_condition else None,
    }
    # Ensure nested shape
    if 'results' not in result['pooled']:
        result['pooled']['results'] = []
    if 'ridge_reference' not in result['pooled']:
        result['pooled']['ridge_reference'] = ridge_pooled

    def _flush():
        try:
            with open(out_file, 'w') as wf:
                json.dump(result, wf, indent=2, default=str)
        except Exception as _ef:
            ui.warn(f"  [flush] save failed: {_ef}")

    # ── 56a: pooled ───────────────────────────────────────────────────────
    ui.section("56a — Pooled MLP decomposition")
    completed_pooled = {(e['architecture'], e['seed']) for e in result['pooled']['results']}
    total_pooled = len(_MLP_ARCHITECTURES) * len(_MLP_SEEDS)
    pooled_done = len(completed_pooled)

    for arch_key in _MLP_ARCHITECTURES:
        for seed in _MLP_SEEDS:
            if (arch_key, seed) in completed_pooled:
                ui.msg(f"  [{arch_key:6s} s{seed}]  (cached)")
                continue
            _analysis_status(56, f"56a {arch_key} seed{seed}",
                             int(pooled_done / total_pooled * 50), temperature=temp)
            try:
                entry = _mlp_one_arch_seed(rows_3way, arch_key, seed, n_perm)
                ui.msg(f"  [{arch_key:6s} s{seed}]  R²={entry['r2_full']:.4f}  "
                       f"E={entry['frac_E']:.3f}  C={entry['frac_C']:.3f}  R={entry['frac_R']:.3f}  "
                       f"neg={entry['neg_drop_count']}")
                if not entry['training']['converged']:
                    ui.warn(f"    [{arch_key} s{seed}] did not converge within max_iter=500")
            except Exception as _fe:
                ui.warn(f"  [{arch_key} s{seed}] FAILED: {_fe}")
                entry = {'architecture': arch_key, 'seed': int(seed),
                         'error': str(_fe)}
            result['pooled']['results'].append(entry)
            pooled_done += 1
            _flush()
            _check_pause()

    # Pooled summary
    ridge_fracs = {c: ridge_pooled[c] for c in 'ECR'}
    result['pooled']['summary'] = _mlp_summarize(result['pooled']['results'], ridge_fracs)
    ps = result['pooled']['summary']
    ui.msg(f"  POOLED SUMMARY: E={ps['mean']['E']:.3f}±{ps['std']['E']:.3f}  "
           f"C={ps['mean']['C']:.3f}±{ps['std']['C']:.3f}  "
           f"R={ps['mean']['R']:.3f}±{ps['std']['R']:.3f}")
    ui.msg(f"  RIDGE GAP:      E={ps['ridge_gap']['E']:+.4f}  "
           f"C={ps['ridge_gap']['C']:+.4f}  R={ps['ridge_gap']['R']:+.4f}")
    _flush()
    ui.blank()

    # ── 56b: per-condition ────────────────────────────────────────────────
    if do_per_condition:
        ui.section("56b — Per-condition MLP decomposition")
        # Group rows by source run — same logic as Run 0044
        by_run = {}
        for r in rows_3way:
            by_run.setdefault(r.get('run_num', 0), []).append(r)

        # v0.79.4.0: renumbered to new IDs.
        _LABELS46 = {
             6: "Introspection A",      7: "Introspection B",
             8: "Introspection C",     13: "Priming neutral",
            14: "Priming cooperative", 15: "Priming resistant",
             1: "Null trivariant",      3: "Temperature grid",
            23: "Confound isolation",
        }

        if result['per_condition'] is None:
            result['per_condition'] = {}

        cond_keys = sorted(by_run.keys())
        for ci, rn in enumerate(cond_keys, 1):
            rows_c  = by_run[rn]
            label_c = _LABELS46.get(rn, f"Run {rn}")
            key_c   = f"run{rn:02d}"

            if len(rows_c) < _MIN_QUADRUPLETS_PER_RUN:
                ui.msg(f"  {rn:4d} {label_c:<22}  N={len(rows_c):4d}  — insufficient, skipping")
                result['per_condition'][key_c] = {
                    'run_num': rn, 'label': label_c, 'n_quadruplets': len(rows_c),
                    'skipped': True,
                    'reason': f'< {_MIN_QUADRUPLETS_PER_RUN} quadruplets',
                }
                _flush()
                continue

            # Ridge reference for this condition (may be absent if Run 0044 skipped it)
            ridge_c = ridge_per_cond.get(rn)
            if ridge_c is None:
                ui.msg(f"  {rn:4d} {label_c:<22}  N={len(rows_c):4d}  — no Ridge reference, skipping")
                result['per_condition'][key_c] = {
                    'run_num': rn, 'label': label_c, 'n_quadruplets': len(rows_c),
                    'skipped': True, 'reason': 'no Ridge reference in Q46',
                }
                _flush()
                continue

            # Resume per (run_num, arch, seed)
            existing = result['per_condition'].get(key_c, {})
            existing_results = existing.get('results', []) if not existing.get('skipped') else []
            completed_c = {(e['architecture'], e['seed']) for e in existing_results}

            entries_c = list(existing_results)
            for arch_key in _MLP_ARCHITECTURES:
                for seed in _MLP_SEEDS:
                    if (arch_key, seed) in completed_c:
                        continue
                    _analysis_status(56, f"56b {label_c} {arch_key} s{seed}",
                                     50 + int(ci / max(len(cond_keys),1) * 50),
                                     temperature=temp)
                    try:
                        entry = _mlp_one_arch_seed(rows_c, arch_key, seed, n_perm)
                    except Exception as _fe:
                        ui.warn(f"    [{label_c} {arch_key} s{seed}] FAILED: {_fe}")
                        entry = {'architecture': arch_key, 'seed': int(seed),
                                 'error': str(_fe)}
                    entries_c.append(entry)
                    # Update result in place after each entry so partial is complete
                    result['per_condition'][key_c] = {
                        'run_num':         rn,
                        'label':           label_c,
                        'n_quadruplets':   len(rows_c),
                        'skipped':         False,
                        'ridge_reference': ridge_c,
                        'results':         entries_c,
                    }
                    _flush()
                    _check_pause()

            # Per-condition summary
            ridge_fracs_c = {c: ridge_c[c] for c in 'ECR'}
            result['per_condition'][key_c]['summary'] = _mlp_summarize(entries_c, ridge_fracs_c)
            cs = result['per_condition'][key_c]['summary']
            # Only print if we actually computed anything
            if cs.get('n_valid', 0) > 0:
                ui.msg(f"  {rn:4d} {label_c:<22}  N={len(rows_c):4d}  "
                       f"E={cs['mean']['E']:.3f} C={cs['mean']['C']:.3f} R={cs['mean']['R']:.3f}  "
                       f"gap R={cs['ridge_gap']['R']:+.3f}")
            _flush()

    # ── Top-level mlp_summary ────────────────────────────────────────────
    top_summary = {
        'pooled': {
            'mean':      result['pooled']['summary']['mean'],
            'std':       result['pooled']['summary']['std'],
            'ridge_gap': result['pooled']['summary']['ridge_gap'],
        }
    }
    if do_per_condition and result['per_condition']:
        # Average ridge_gap across all valid conditions
        gaps_E, gaps_C, gaps_R = [], [], []
        for key_c, cd in result['per_condition'].items():
            if cd.get('skipped'):
                continue
            summ = cd.get('summary', {})
            if summ.get('n_valid', 0) == 0:
                continue
            rg = summ.get('ridge_gap', {})
            if np.isnan(rg.get('E', float('nan'))): continue
            gaps_E.append(rg['E']); gaps_C.append(rg['C']); gaps_R.append(rg['R'])
        if gaps_E:
            top_summary['per_condition_ridge_gap_mean'] = {
                'E': float(np.mean(gaps_E)),
                'C': float(np.mean(gaps_C)),
                'R': float(np.mean(gaps_R)),
                'n_conditions_used': len(gaps_E),
            }
    result['mlp_summary'] = top_summary
    result['status'] = 'complete'
    _flush()

    ui.blank()
    ui.ok(f"Run 0046 complete. Saved: {out_file}")

# ══════════════════════════════════════════════════════════════════
# RUN 47 — CROSS-TEMPERATURE SYNTHESIS
# ══════════════════════════════════════════════════════════════════
# Answers the remaining paper questions by reading CSVs across all
# temperature directories. No GPU. Produces one JSON with sections:
#   1. Priming vulnerability (Runs 0013/0014/0015)
#   2. Contradiction phase analysis (Run 0026)
#   3. Per-run similarity summary (all runs, all temperatures)
#   4. Coherence levels power by condition (Run 0021)
#   5. Three-variant comparison (Run 0001)

def _run_cross_temp_synthesis(session, paths):
    import csv as _csv
    from cartography import DATA, get_paths as _gp_synth, RUN_CSV, condition_name, get_family_size_dir, run_mode_matches

    family     = session.get('model_family', 'llama')
    size       = session.get('model_size', '8b')
    variant    = session.get('model_variant', 'abliterated')
    model_name = session.get('model_name', '')

    pooled_dir = os.path.join(get_family_size_dir(family, size), variant, 'pooled', 'analysis')
    os.makedirs(pooled_dir, exist_ok=True)
    out_file = os.path.join(pooled_dir, "Q0050_cross_temp_synthesis.json")

    ui.section("Run 0050 — Cross-Temperature Synthesis (no GPU)")
    ui.msg("Reads CSVs across all temperature directories.")
    ui.blank()

    _TEMPS = [
        (0.0, 'deterministic'), (0.2, 'temp_0.2'), (0.4, 'temp_0.4'),
        (0.6, 'temp_0.6'), (0.8, 'temp_0.8'), (1.0, 'temp_1.0'),
    ]

    # ── Gate: require all 6 temperature rounds, override at 3+ ────────
    base_dir = os.path.join(get_family_size_dir(family, size), variant)
    _found_conds = set()
    if os.path.isdir(base_dir):
        for d in os.listdir(base_dir):
            if d != 'pooled' and os.path.isdir(os.path.join(base_dir, d, 'csv')):
                _found_conds.add(d)
    _REQUIRED_CONDS = {'deterministic', 'temp_0.2', 'temp_0.4', 'temp_0.6', 'temp_0.8', 'temp_1.0'}
    _missing_conds = _REQUIRED_CONDS - _found_conds
    if len(_found_conds) < 3:
        ui.err(f"Run 0050 requires at least 3 temperature conditions. Found {len(_found_conds)}.")
        ui.msg("  Complete more temperature rounds before running cross-temp synthesis.")
        return
    if _missing_conds:
        ui.warn(f"Run 0050 expects ALL 6 temperature rounds. Missing: {sorted(_missing_conds)}")
        ui.warn(f"  Proceeding with {len(_found_conds)}/6 rounds — results are preliminary.")
        ui.blank()

    def _read_csv(csv_path, want_cols=None):
        """Read CSV, return list of row dicts. Filters to want_cols if specified."""
        rows = []
        if not os.path.exists(csv_path):
            return rows
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    if row.get('priming') == '1':
                        continue
                    rows.append(row)
        except Exception:
            pass
        return rows

    def _safe_float(v):
        try:
            f = float(v)
            return f if f == f else None  # NaN check
        except (ValueError, TypeError):
            return None

    results = {"run_num": 47, "label": "cross_temp_synthesis"}

    # ═══════════════════════════════════════════════════════════════
    # Section 1: Priming Vulnerability (Runs 0013/0014/0015)
    # ═══════════════════════════════════════════════════════════════
    ui.section("Section 1 — Priming Vulnerability (Runs 0013/0014/0015)")
    _PRIME_LABELS = {13: 'neutral', 14: 'cooperative', 15: 'resistant'}
    priming_results = []

    for temp, cond in _TEMPS:
        temp_paths = _gp_synth(family, size, variant, temp, create_dirs=False)
        csv_dir = temp_paths.get('csv', '')
        temp_entry = {"temperature": temp}
        for rn, prime_label in _PRIME_LABELS.items():
            csv_name = RUN_CSV.get(rn)
            if not csv_name:
                continue
            rows = _read_csv(os.path.join(csv_dir, csv_name))
            rows = [r for r in rows if run_mode_matches(r.get('run_mode', ''), rn)]
            similarities = [_safe_float(r.get('state_similarity_index')) for r in rows]
            similarities = [t for t in similarities if t is not None]
            disruption_flags = [_safe_float(r.get('disruption_flag')) for r in rows]
            disruption_flags = [r for r in disruption_flags if r is not None]
            temp_entry[f"{prime_label}_mean_similarity"] = float(np.mean(similarities)) if similarities else None
            temp_entry[f"{prime_label}_std_similarity"] = float(np.std(similarities)) if similarities else None
            temp_entry[f"{prime_label}_disruption_rate"] = float(np.mean(disruption_flags)) if disruption_flags else None
            temp_entry[f"{prime_label}_n"] = len(similarities)
        priming_results.append(temp_entry)

    results["priming_vulnerability"] = priming_results

    # Display
    ui.msg(f"  {'T':>4}  {'Neutral Similarity':>12}  {'Coop Similarity':>12}  {'Resist Similarity':>12}  {'N rc':>8}  {'C rc':>8}  {'R rc':>8}")
    for e in priming_results:
        _nt = e.get('neutral_mean_similarity')
        _ct = e.get('cooperative_mean_similarity')
        _rt = e.get('resistant_mean_similarity')
        _nr = e.get('neutral_disruption_rate')
        _cr = e.get('cooperative_disruption_rate')
        _rr = e.get('resistant_disruption_rate')
        ui.msg(f"  {e['temperature']:4.1f}  "
               f"{_nt:12.4f}  {_ct:12.4f}  {_rt:12.4f}  "
               f"{_nr:8.3f}  {_cr:8.3f}  {_rr:8.3f}"
               if all(v is not None for v in [_nt,_ct,_rt,_nr,_cr,_rr])
               else f"  {e['temperature']:4.1f}  — incomplete data —")
    ui.blank()

    _check_pause()

    # ═══════════════════════════════════════════════════════════════
    # Section 2: Contradiction Phase Analysis (Run 0026)
    # ═══════════════════════════════════════════════════════════════
    ui.section("Section 2 — Contradiction Phase Analysis (Run 0026)")
    contradiction_results = []

    for temp, cond in _TEMPS:
        temp_paths = _gp_synth(family, size, variant, temp, create_dirs=False)
        csv_path = os.path.join(temp_paths.get('csv', ''), RUN_CSV.get("0026", ''))
        rows = _read_csv(csv_path)
        rows = [r for r in rows if r.get('run_mode', '') in ('0026', '44')]
        if not rows:
            contradiction_results.append({"temperature": temp, "n": 0})
            continue

        # Group by r_condition
        by_cond = {}
        for r in rows:
            c = r.get('r_condition', 'unknown')
            by_cond.setdefault(c, []).append(r)

        temp_entry = {"temperature": temp, "conditions": {}}
        for rc, cond_rows in sorted(by_cond.items()):
            pre = [_safe_float(r.get('state_similarity_index')) for r in cond_rows
                   if r.get('pre_contradiction') == '1']
            pre = [t for t in pre if t is not None]
            post = [_safe_float(r.get('state_similarity_index')) for r in cond_rows
                    if r.get('post_contradiction') == '1']
            post = [t for t in post if t is not None]
            contra = [_safe_float(r.get('state_similarity_index')) for r in cond_rows
                      if r.get('contradiction_turn') == '1']
            contra = [t for t in contra if t is not None]
            temp_entry["conditions"][rc] = {
                "pre_similarity_mean": float(np.mean(pre)) if pre else None,
                "post_similarity_mean": float(np.mean(post)) if post else None,
                "contradiction_similarity_mean": float(np.mean(contra)) if contra else None,
                "similarity_drop": float(np.mean(pre) - np.mean(post)) if pre and post else None,
                "n_pre": len(pre), "n_post": len(post),
            }
        contradiction_results.append(temp_entry)

    results["contradiction_analysis"] = contradiction_results

    ui.msg(f"  {'T':>4}  {'Condition':<20}  {'Pre Similarity':>8}  {'Post Similarity':>8}  {'Drop':>8}")
    for e in contradiction_results:
        for rc, d in e.get("conditions", {}).items():
            pre = d.get('pre_similarity_mean')
            post = d.get('post_similarity_mean')
            drop = d.get('similarity_drop')
            if pre is not None and post is not None:
                ui.msg(f"  {e['temperature']:4.1f}  {rc:<20}  {pre:8.4f}  {post:8.4f}  {drop:+8.4f}")
    ui.blank()

    _check_pause()

    # ═══════════════════════════════════════════════════════════════
    # Section 3: Per-Run similarity Summary (all runs, all temperatures)
    # ═══════════════════════════════════════════════════════════════
    ui.section("Section 3 — Per-Run similarity Summary")
    similarity_summary = {}  # {run_num: {temp: {mean_similarity, mean_sim, n}}}

    for temp, cond in _TEMPS:
        temp_paths = _gp_synth(family, size, variant, temp, create_dirs=False)
        csv_dir = temp_paths.get('csv', '')
        if not os.path.isdir(csv_dir):
            continue
        for fname in os.listdir(csv_dir):
            if not fname.endswith('.csv'):
                continue
            rows = _read_csv(os.path.join(csv_dir, fname))
            if not rows:
                continue
            # Group by run_mode
            by_run = {}
            for r in rows:
                rm = r.get('run_mode', '')
                try:
                    rm = int(rm)
                except (ValueError, TypeError):
                    continue
                by_run.setdefault(rm, []).append(r)
            for rm, run_rows in by_run.items():
                similarities = [_safe_float(r.get('state_similarity_index')) for r in run_rows]
                similarities = [t for t in similarities if t is not None]
                sims = [_safe_float(r.get('layer_sim_mean')) for r in run_rows]
                sims = [s for s in sims if s is not None]
                if rm not in similarity_summary:
                    similarity_summary[rm] = {}
                similarity_summary[rm][temp] = {
                    "mean_similarity": float(np.mean(similarities)) if similarities else None,
                    "mean_sim": float(np.mean(sims)) if sims else None,
                    "n": len(similarities),
                }

    results["similarity_summary"] = {str(k): v for k, v in similarity_summary.items()}

    # Print compact table
    ui.msg(f"  {'Run':>4}  " + "  ".join(f"T={t:.1f}" for t, _ in _TEMPS))
    for rn in sorted(similarity_summary.keys()):
        vals = []
        for temp, _ in _TEMPS:
            d = similarity_summary[rn].get(temp, {})
            m = d.get('mean_similarity')
            vals.append(f"{m:.3f}" if m is not None else "  —  ")
        ui.msg(f"  {rn:4d}  " + "  ".join(f"{v:>5}" for v in vals))
    ui.blank()

    _check_pause()

    # ═══════════════════════════════════════════════════════════════
    # Section 4: Power by R Condition (Run 0021)
    # ═══════════════════════════════════════════════════════════════
    ui.section("Section 4 — Power by R Condition (Run 0021)")
    power_results = []

    for temp, cond in _TEMPS:
        temp_paths = _gp_synth(family, size, variant, temp, create_dirs=False)
        csv_path = os.path.join(temp_paths.get('csv', ''), RUN_CSV.get("0021", ''))
        rows = _read_csv(csv_path)
        rows = [r for r in rows if r.get('run_mode', '') in ('0021', '31')]
        if not rows:
            power_results.append({"temperature": temp, "n": 0})
            continue

        by_cond = {}
        for r in rows:
            c = r.get('r_condition', 'unknown')
            by_cond.setdefault(c, []).append(r)

        temp_entry = {"temperature": temp, "conditions": {}}
        for rc, cond_rows in sorted(by_cond.items()):
            powers = [_safe_float(r.get('peak_gpu_power')) for r in cond_rows]
            powers = [p for p in powers if p is not None]
            similarities = [_safe_float(r.get('state_similarity_index')) for r in cond_rows]
            similarities = [t for t in similarities if t is not None]
            temp_entry["conditions"][rc] = {
                "mean_power": float(np.mean(powers)) if powers else None,
                "mean_similarity": float(np.mean(similarities)) if similarities else None,
                "n": len(powers),
            }
        power_results.append(temp_entry)

    results["coherence_levels_power"] = power_results

    ui.msg(f"  {'T':>4}  {'Condition':<12}  {'Power':>8}  {'Similarity':>8}  {'N':>6}")
    for e in power_results:
        for rc, d in e.get("conditions", {}).items():
            pw = d.get('mean_power')
            tc = d.get('mean_similarity')
            n = d.get('n', 0)
            if pw is not None:
                ui.msg(f"  {e['temperature']:4.1f}  {rc:<12}  {pw:8.1f}  {tc:8.4f}  {n:6d}"
                       if tc is not None
                       else f"  {e['temperature']:4.1f}  {rc:<12}  {pw:8.1f}  {'—':>8}  {n:6d}")
    ui.blank()

    _check_pause()

    # ═══════════════════════════════════════════════════════════════
    # Section 5: Three-Variant Comparison (Run 0001)
    # ═══════════════════════════════════════════════════════════════
    ui.section("Section 5 — Three-Variant Comparison (Run 0001)")
    # Run 0001 is temperature-independent (collected once, copied to all rounds)
    # Read from deterministic directory
    variant_results = {}
    det_paths = _gp_synth(family, size, variant, 0.0, create_dirs=False)
    r1_csv = os.path.join(det_paths.get('csv', ''), RUN_CSV.get("0001", ''))
    rows = _read_csv(r1_csv)
    rows_1 = [r for r in rows if r.get('run_mode', '') in ('0001', '19')]  # accept both forms pre/post migration

    by_model = {}
    for r in rows_1:
        m = r.get('model', 'unknown')
        by_model.setdefault(m, []).append(r)

    for m, mrows in sorted(by_model.items()):
        similarities = [_safe_float(r.get('state_similarity_index')) for r in mrows]
        similarities = [t for t in similarities if t is not None]
        sims = [_safe_float(r.get('layer_sim_mean')) for r in mrows]
        sims = [s for s in sims if s is not None]
        ents = [_safe_float(r.get('mean_logit_entropy')) for r in mrows]
        ents = [e for e in ents if e is not None]
        variant_results[m] = {
            "mean_similarity": float(np.mean(similarities)) if similarities else None,
            "mean_sim": float(np.mean(sims)) if sims else None,
            "mean_entropy": float(np.mean(ents)) if ents else None,
            "n": len(similarities),
        }
        ui.msg(f"  {m:<40}  SIM={np.mean(similarities):.4f}  SIM={np.mean(sims):.4f}  "
               f"ENT={np.mean(ents):.4f}  N={len(similarities)}"
               if similarities and sims and ents
               else f"  {m:<40}  — insufficient data —")

    results["three_variant"] = variant_results
    ui.blank()

    _check_pause()

    # ═══════════════════════════════════════════════════════════════
    # Section 6: Compute Efficiency (Run 0025)
    # ═══════════════════════════════════════════════════════════════
    ui.section("Section 6 — Compute Efficiency (Run 0025)")
    compression_results = []

    for temp, cond in _TEMPS:
        temp_paths = _gp_synth(family, size, variant, temp, create_dirs=False)
        csv_path = os.path.join(temp_paths.get('csv', ''), RUN_CSV.get("0025", ''))
        rows = _read_csv(csv_path)
        rows = [r for r in rows if r.get('run_mode', '') in ('0025', '43')]
        if not rows:
            compression_results.append({"temperature": temp, "n": 0})
            continue

        by_cond = {}
        for r in rows:
            c = r.get('r_condition', 'unknown')
            by_cond.setdefault(c, []).append(r)

        temp_entry = {"temperature": temp, "conditions": {}}
        for rc, cond_rows in sorted(by_cond.items()):
            arith_rows = [r for r in cond_rows if r.get('phase') == 'arithmetic']
            correct = sum(1 for r in arith_rows if r.get('correct', '0') == '1')
            total_arith = len(arith_rows)
            powers = [_safe_float(r.get('peak_gpu_power')) for r in arith_rows]
            powers = [p for p in powers if p is not None]
            elapsed = [_safe_float(r.get('elapsed_sec')) for r in arith_rows]
            elapsed = [e for e in elapsed if e is not None]
            # joules = power × time for each turn
            joules = []
            for r in arith_rows:
                p = _safe_float(r.get('peak_gpu_power'))
                e = _safe_float(r.get('elapsed_sec'))
                if p is not None and e is not None:
                    joules.append(p * e)
            total_joules = sum(joules) if joules else None
            accuracy = correct / total_arith if total_arith > 0 else None
            compute_per_correct = (total_joules / correct) if (total_joules and correct > 0) else None

            temp_entry["conditions"][rc] = {
                "n_arithmetic": total_arith,
                "correct": correct,
                "accuracy": accuracy,
                "total_joules": total_joules,
                "compute_per_correct": compute_per_correct,
                "mean_power": float(np.mean(powers)) if powers else None,
            }
        compression_results.append(temp_entry)

    results["compute_efficiency"] = compression_results

    ui.msg(f"  {'T':>4}  {'Condition':<20}  {'Acc':>6}  {'J/correct':>10}  {'Power':>8}")
    for e in compression_results:
        for rc, d in e.get("conditions", {}).items():
            acc = d.get('accuracy')
            jpc = d.get('compute_per_correct')
            pw = d.get('mean_power')
            if acc is not None:
                ui.msg(f"  {e['temperature']:4.1f}  {rc:<20}  {acc:6.2f}  "
                       f"{jpc:10.1f}  {pw:8.1f}"
                       if jpc is not None
                       else f"  {e['temperature']:4.1f}  {rc:<20}  {acc:6.2f}  "
                       f"{'—':>10}  {pw:8.1f}" if pw is not None
                       else f"  {e['temperature']:4.1f}  {rc:<20}  {acc:6.2f}")
    ui.blank()

    _check_pause()

    # ═══════════════════════════════════════════════════════════════
    # Section 7: Context Saturation — Persistence Modes (Run 0024)
    # ═══════════════════════════════════════════════════════════════
    ui.section("Section 7 — Context Saturation: Persistence Modes (Run 0024)")
    persistence_results = []

    for temp, cond in _TEMPS:
        temp_paths = _gp_synth(family, size, variant, temp, create_dirs=False)
        csv_path = os.path.join(temp_paths.get('csv', ''), RUN_CSV.get("0024", ''))
        rows = _read_csv(csv_path)
        rows = [r for r in rows if r.get('run_mode', '') in ('0024', '29')]
        if not rows:
            persistence_results.append({"temperature": temp, "n": 0})
            continue

        by_mode = {}
        for r in rows:
            m = r.get('history_mode', 'unknown')
            by_mode.setdefault(m, []).append(r)

        temp_entry = {"temperature": temp, "modes": {}}
        for mode, mode_rows in sorted(by_mode.items()):
            similarities = [_safe_float(r.get('state_similarity_index')) for r in mode_rows]
            similarities = [t for t in similarities if t is not None]
            # Early vs late similarity to measure decay rate
            early = [_safe_float(r.get('state_similarity_index')) for r in mode_rows
                     if r.get('turn', '0') in ('1','2','3')]
            early = [t for t in early if t is not None]
            late = [_safe_float(r.get('state_similarity_index')) for r in mode_rows
                    if r.get('turn', '0') in ('11','12','13')]
            late = [t for t in late if t is not None]
            decay = (float(np.mean(early)) - float(np.mean(late))) if early and late else None

            temp_entry["modes"][mode] = {
                "mean_similarity": float(np.mean(similarities)) if similarities else None,
                "early_similarity": float(np.mean(early)) if early else None,
                "late_similarity": float(np.mean(late)) if late else None,
                "similarity_decay": decay,
                "n": len(similarities),
            }
        persistence_results.append(temp_entry)

    results["persistence_modes"] = persistence_results

    ui.msg(f"  {'T':>4}  {'Mode':<12}  {'Mean Similarity':>9}  {'Early':>7}  {'Late':>7}  {'Decay':>7}")
    for e in persistence_results:
        for mode, d in e.get("modes", {}).items():
            mt = d.get('mean_similarity')
            et = d.get('early_similarity')
            lt = d.get('late_similarity')
            dc = d.get('similarity_decay')
            if mt is not None:
                ui.msg(f"  {e['temperature']:4.1f}  {mode:<12}  {mt:9.4f}  {et:7.4f}  {lt:7.4f}  {dc:+7.4f}"
                       if all(v is not None for v in [et, lt, dc])
                       else f"  {e['temperature']:4.1f}  {mode:<12}  {mt:9.4f}")
    ui.blank()

    _check_pause()

    # ═══════════════════════════════════════════════════════════════
    # Section 8: Condition Transfer (Run 0036)
    # ═══════════════════════════════════════════════════════════════
    ui.section("Section 8 — Condition Transfer: Introspection → Arithmetic (Run 0036)")
    transfer_results = []

    for temp, cond in _TEMPS:
        temp_paths = _gp_synth(family, size, variant, temp, create_dirs=False)
        csv_path = os.path.join(temp_paths.get('csv', ''), RUN_CSV.get("0036", ''))
        rows = _read_csv(csv_path)
        rows = [r for r in rows if r.get('run_mode', '') in ('0036', '38')]
        if not rows:
            transfer_results.append({"temperature": temp, "n": 0})
            continue

        by_cond = {}
        for r in rows:
            c = r.get('condition', 'unknown')
            by_cond.setdefault(c, []).append(r)

        temp_entry = {"temperature": temp, "conditions": {}}
        for cname, cond_rows in sorted(by_cond.items()):
            # Split at turn 7 (SWITCH_AT)
            pre = [r for r in cond_rows if int(r.get('turn', '0')) < 7]
            post = [r for r in cond_rows if int(r.get('turn', '0')) >= 7]
            pre_sims = [_safe_float(r.get('state_similarity_index')) for r in pre]
            pre_sims = [t for t in pre_sims if t is not None]
            post_sims = [_safe_float(r.get('state_similarity_index')) for r in post]
            post_sims = [t for t in post_sims if t is not None]
            transfer_delta = (float(np.mean(post_sims)) - float(np.mean(pre_sims))) if pre_sims and post_sims else None

            temp_entry["conditions"][cname] = {
                "pre_switch_similarity": float(np.mean(pre_sims)) if pre_sims else None,
                "post_switch_similarity": float(np.mean(post_sims)) if post_sims else None,
                "transfer_delta": transfer_delta,
                "n_pre": len(pre_sims), "n_post": len(post_sims),
            }
        transfer_results.append(temp_entry)

    results["condition_transfer"] = transfer_results

    ui.msg(f"  {'T':>4}  {'Condition':<28}  {'Pre Similarity':>8}  {'Post Similarity':>8}  {'Delta':>8}")
    for e in transfer_results:
        for cname, d in e.get("conditions", {}).items():
            pre = d.get('pre_switch_similarity')
            post = d.get('post_switch_similarity')
            delta = d.get('transfer_delta')
            if pre is not None and post is not None:
                ui.msg(f"  {e['temperature']:4.1f}  {cname:<28}  {pre:8.4f}  {post:8.4f}  {delta:+8.4f}")
    ui.blank()

    # ═══════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════
    results["interpretation"] = (
        "Cross-temperature synthesis answering the core paper questions. "
        "Section 1: Priming vulnerability — resistant prime should damage Similarity most. "
        "Section 2: Contradiction — Similarity should drop at contradiction turn. "
        "Section 3: Per-run similarity — master summary of trajectory consistency across all conditions. "
        "Section 4: Power by R condition — high_r should differ from low_r in power usage. "
        "Section 5: Three-variant — abliterated vs instruct vs base similarity comparison. "
        "Section 6: Compute efficiency — joules per correct answer by R condition. "
        "Section 7: Persistence modes — full/last/summary context, tests KV cache confound. "
        "Section 8: Condition transfer — does introspection R carry into arithmetic?"
    )

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    ui.blank()
    ui.ok(f"Saved: {out_file}")


# ══════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════

def run(run_num: int, session: dict, paths: dict):
    assert run_num in (47, 48, 49, 42, 43, 51, 41, 44, 50, 45, 52, 53, 54, 46), \
        f"analysis.py handles runs 0047, 0048, 0049-0043, 0051, 0041-0050, 0045-0054, 0046 — not {run_num}"

    _temp = session.get('temperature', 0.0)
    ana_dir = paths.get('analysis', '')

    # ── Resume-aware completeness check (v0.75.1.0) ─────────────────────
    # Replaces the v0.71.0.12 file-existence gate that caused silent skips
    # when new analysis stages were added to existing runs (e.g. H50/H51
    # added to Run 0042 after LLaMA's initial analysis).
    #
    # Each analysis run declares required fields. If the output JSON exists
    # but is missing any required field (or a field contains 'error'), the
    # run re-enters and internal incremental logic handles skipping completed
    # stages. Runs without incremental logic will recompute fully — this is
    # correct: the old output was incomplete.
    #
    # Runs not in REQUIRED_FIELDS fall back to the legacy check (file exists
    # + status != 'running') for backward compatibility.
    REQUIRED_FIELDS = {
        # v0.79.4.0 renumber:
        #   old 33 (decomposition)      → new 42
        #   old 34 (perm sensitivity)   → new 43
        #   old 40 (pooled decomp+perm) → new 51
        #   old 56 (MLP decomposition)  → new 46
        42: ['r2_D_full_decomposition', 'linearity_check', 'interaction_info',
             'h16_confound_decomposition'],
        43: ['perm_sens_R', 'bootstrap_ci'],
        51: ['perm_sens_R'],
        46: ['pooled', 'mlp_summary'],
    }

    from cartography import ANALYSIS_JSON as _AJ_GATE
    _out_name = _AJ_GATE.get(run_num)
    if _out_name:
        _out_path = os.path.join(ana_dir, _out_name)
        if os.path.exists(_out_path):
            try:
                with open(_out_path) as _fg:
                    _dg = json.load(_fg)
                if _dg.get('status') == 'running':
                    ui.msg(f"Run {run_num}: partial ({_out_name} status=running) — resuming.")
                    pass  # fall through to re-enter
                elif run_num in REQUIRED_FIELDS:
                    _missing = []
                    for _rf in REQUIRED_FIELDS[run_num]:
                        _val = _dg.get(_rf)
                        if _val is None:
                            _missing.append(_rf)
                        elif isinstance(_val, dict) and 'error' in _val:
                            _missing.append(f"{_rf}(error)")
                    if _missing:
                        ui.msg(f"Run {run_num}: incomplete — missing {', '.join(_missing)}. Re-entering.")
                        pass  # fall through to re-enter
                    else:
                        ui.ok(f"Run {run_num}: all required fields present — skipping.")
                        return
                else:
                    # Legacy fallback for runs without declared required fields
                    ui.ok(f"Run {run_num}: already complete ({_out_name}) — skipping.")
                    return
            except Exception:
                pass  # corrupt/truncated → fall through to re-run

    # Write running marker + initial status
    _analysis_marker(ana_dir, run_num, start=True)
    _analysis_status(run_num, "Starting...", 0, temperature=_temp)

    # All analysis runs except 45, 49, and 50-52 require POOL_DIM calibration.
    # Run 0045 forces POOL_DIM=64 regardless of dim₉₅.
    if run_num not in (41, 45, 52, 53, 54):
        if not _ensure_calibrated(session, paths):
            ui.err(f"Run 0041 (POOL_DIM calibration) required at T={_temp} before running analysis.")
            ui.err(f"  Run 0041 first, then re-run {run_num}.")
            _analysis_marker(ana_dir, run_num, start=False)
            return
        ui.msg(f"  [POOL_DIM={POOL_DIM}]")

    try:
        if run_num in (47, 48):
            _run_granger(run_num, session, paths)
        elif run_num == 49:
            _run_baseline_swap(session, paths)
        elif run_num == 42:
            _run_decomposition(session, paths)
        elif run_num == 43:
            _run_permutation_sensitivity(session, paths)
        elif run_num == 51:
            _run_pooled(session, paths)
        elif run_num == 41:
            _run_pool_sweep(session, paths)
        elif run_num == 44:
            _run_per_condition_r(session, paths)
        elif run_num == 45:
            _run_per_condition_r(session, paths, force_dim=64, run_label=45,
                                out_name="Q0045_fixed_dim_per_condition_R.json")
        elif run_num == 50:
            _run_cross_temp_synthesis(session, paths)
        elif run_num in (52, 53, 54):
            _run_cross_model(run_num, session, paths)
        elif run_num == 46:
            # v0.76.1.0: MLP permutation sensitivity — pooled (56a) + per-condition (56b).
            # do_per_condition can be toggled via session['_r56_per_condition'] — batch
            # drivers set False after first model shows tight Ridge-MLP agreement.
            _do_pc = session.get('_r56_per_condition', True)
            _run_mlp_decomposition(session, paths, do_per_condition=_do_pc)
    finally:
        _analysis_marker(ana_dir, run_num, start=False)
        _analysis_status(run_num, "Complete", 100, temperature=_temp)


# ══════════════════════════════════════════════════════════════════
# RUNS 50-52 — CROSS-MODEL COMPARISON (v0.71.0.0)
# ══════════════════════════════════════════════════════════════════
# Auto-discover models by scanning base/ for completed analysis.
# Run 0052: pairwise R fraction comparison.
# Run 0053: Kendall's W concordance on condition rankings.
# Run 0054: paper summary table (Sections 5 & 6 schema).

def _discover_models(data_dir):
    """Scan base/ for model directories with completed Q34 analysis."""
    models = []
    if not os.path.isdir(data_dir):
        return models
    for family in sorted(os.listdir(data_dir)):
        fam_dir = os.path.join(data_dir, family)
        if not os.path.isdir(fam_dir):
            continue
        for size in sorted(os.listdir(fam_dir)):
            sz_dir = os.path.join(fam_dir, size)
            if not os.path.isdir(sz_dir):
                continue
            for variant in sorted(os.listdir(sz_dir)):
                var_dir = os.path.join(sz_dir, variant)
                if not os.path.isdir(var_dir) or variant == 'base':
                    continue
                # Check if at least one temperature has Q34
                has_q34 = False
                for cond in os.listdir(var_dir):
                    if cond == 'pooled':
                        continue
                    ana_dir = os.path.join(var_dir, cond, 'analysis')
                    if os.path.isdir(ana_dir):
                        for f in os.listdir(ana_dir):
                            if 'Q34' in f and f.endswith('.json'):
                                has_q34 = True
                                break
                    if has_q34:
                        break
                if has_q34:
                    models.append({'family': family, 'size': size, 'variant': variant,
                                   'path': var_dir, 'label': f"{family}/{size}/{variant}"})
    return models


def _load_model_summary(model, data_dir):
    """Load per-temperature summary for one model. Returns list of per-temp dicts."""
    from cartography import get_paths as _gp_cm
    _TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for temp in _TEMPS:
        paths = _gp_cm(model['family'], model['size'], model['variant'], temp, create_dirs=False)
        ana = paths['analysis']
        entry = {'temperature': temp, 'model': model['label']}

        # Q34 fractions + bootstrap
        q43 = None
        q43_path = os.path.join(ana, 'Q0043_sobol_partition.json')
        if os.path.exists(q43_path):
            try:
                with open(q43_path) as f: q43 = json.load(f)
            except Exception: pass
        if q43:
            entry['E'] = q43.get('perm_sens_E', {}).get('fraction')
            entry['C'] = q43.get('perm_sens_C', {}).get('fraction')
            entry['R'] = q43.get('perm_sens_R', {}).get('fraction')
            bc = q43.get('bootstrap_ci', {})
            entry['R_bootstrap_lower'] = bc.get('R_ci_lower')
            entry['R_bootstrap_upper'] = bc.get('R_ci_upper')
            entry['h28_divergence'] = q43.get('h28_max_fraction_divergence')
            entry['h28_status'] = q43.get('h28_status')

        # Q33 decomposition
        q42 = None
        q42_path = os.path.join(ana, 'Q0042_decomposition.json')
        if os.path.exists(q42_path):
            try:
                with open(q42_path) as f: q42 = json.load(f)
            except Exception: pass
        if q42:
            dr2 = q42.get('delta_r2_internal', {})
            entry['delta_r2_internal'] = dr2.get('value')
            entry['delta_r2_p'] = dr2.get('p_value')
            dc = q42.get('delta_r2_constraint', {})
            entry['delta_r2_constraint'] = dc.get('value')
            entry['delta_r2_constraint_p'] = dc.get('p_value')

        # Run 0017 patching output change rate
        csv_dir = paths.get('csv', '')
        r17_csv = os.path.join(csv_dir, 'R0017_patching.csv')
        if os.path.exists(r17_csv):
            try:
                import pandas as _pd_cm
                df17 = _pd_cm.read_csv(r17_csv, dtype=str, keep_default_na=False)
                patched = df17[df17['patch_mode'].isin(['partial', 'full'])]
                if 'output_changed' in patched.columns:
                    oc = _pd_cm.to_numeric(patched['output_changed'], errors='coerce').dropna()
                    if len(oc) >= 10:
                        entry['output_change_rate'] = float(oc.mean() * 100)
            except Exception: pass

        # Run 0018 peak layer
        r18_csv = os.path.join(csv_dir, 'R0018_layer_isolation.csv')
        if os.path.exists(r18_csv):
            try:
                import pandas as _pd_18
                df18 = _pd_18.read_csv(r18_csv, dtype=str, keep_default_na=False)
                patched42 = df18[df18['patch_layer'] != 'none']
                if 'output_changed' in patched42.columns:
                    layer_rates = {}
                    for layer in ['L8', 'L16', 'L24', 'L31']:
                        sub = _pd_18.to_numeric(
                            patched42[patched42['patch_layer'] == layer]['output_changed'],
                            errors='coerce').dropna()
                        if len(sub) >= 5:
                            layer_rates[layer] = float(sub.mean() * 100)
                    if layer_rates:
                        entry['peak_layer'] = max(layer_rates, key=layer_rates.get)
            except Exception: pass

        rows.append(entry)
    return rows


def _run_cross_model(run_num, session, paths):
    """Runs 0052-0054: cross-model comparison. All three share the same discovery + loading."""
    from cartography import DATA, get_pooled_paths

    ui.section(f"Run {run_num} — Cross-Model Comparison (v0.71.0.0)")

    models = _discover_models(DATA)
    ui.msg(f"  Discovered {len(models)} models:")
    for m in models:
        ui.msg(f"    {m['label']}")
    ui.blank()

    if len(models) < 1:
        ui.err("No models with completed analysis found in base/")
        return

    # Load per-model summaries
    all_summaries = {}  # {model_label: [per-temp dicts]}
    for m in models:
        rows = _load_model_summary(m, DATA)
        all_summaries[m['label']] = rows
        n_valid = sum(1 for r in rows if r.get('R') is not None)
        ui.msg(f"  {m['label']}: {n_valid}/6 temperatures with R fractions")

    # Output directory: pooled/analysis of first model (or session model)
    family = session.get('model_family', models[0]['family'])
    size = session.get('model_size', models[0]['size'])
    variant = session.get('model_variant', models[0]['variant'])
    pooled = get_pooled_paths(family, size, variant)
    out_dir = pooled['analysis']

    if run_num == 54 or run_num == 52:
        # Run 0054 (and 50): full summary table
        # Flatten all summaries into the paper schema
        paper_table = []
        for label, rows in all_summaries.items():
            for r in rows:
                paper_table.append(r)

        # Cross-model Kendall W for R ordering across temperatures
        kendall_w = float('nan')
        if len(models) >= 2:
            # Each model is a "judge", each temperature is an "item"
            # Rank R values across temperatures for each model
            _TEMPS_W = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            rank_matrix = []
            for label, rows in all_summaries.items():
                r_by_temp = {r['temperature']: r.get('R') for r in rows}
                r_vals = [r_by_temp.get(t) for t in _TEMPS_W]
                if all(v is not None and v == v for v in r_vals):
                    order = np.argsort(np.argsort(r_vals)) + 1
                    rank_matrix.append(order.tolist())
            if len(rank_matrix) >= 2:
                k = len(rank_matrix)
                n = len(rank_matrix[0])
                rank_arr = np.array(rank_matrix, dtype=float)
                col_sums = rank_arr.sum(axis=0)
                grand_mean = col_sums.mean()
                S = float(np.sum((col_sums - grand_mean) ** 2))
                denom = (k ** 2) * (n ** 3 - n)
                kendall_w = float(S * 12 / denom) if denom > 0 else float('nan')

        out_path = os.path.join(out_dir, 'Q0054_cross_model_summary.json')
        out_data = {
            'run_num': 52,
            'models': [m['label'] for m in models],
            'n_models': len(models),
            'table': paper_table,
            'kendall_w_R_ordering': kendall_w,
            'interpretation': (
                "Cross-model summary for paper Sections 5 & 6. Each row contains "
                "one model at one temperature with the full schema: E, C, R, ΔR², "
                "H28 divergence, output change rate, peak layer, bootstrap CIs. "
                f"Kendall W = {kendall_w:.3f} — "
                f"{'strong concordance' if kendall_w > 0.7 else 'weak concordance'} "
                "in R ordering across temperatures between models."
            ),
        }
        with open(out_path, 'w') as f:
            json.dump(out_data, f, indent=2, default=str)
        ui.ok(f"Run 0054 summary: {out_path}")
        if kendall_w == kendall_w:
            ui.msg(f"  Kendall W (R ordering concordance): {kendall_w:.3f}")

    if run_num == 53:
        # Run 0053: Kendall W on per-condition R rankings across models
        # For each model, load Q46 per-condition R at T=0.2, rank conditions
        from cartography import get_paths as _gp51
        condition_ranks = []
        for m in models:
            p51 = _gp51(m['family'], m['size'], m['variant'], 0.2, create_dirs=False)
            q44_path = os.path.join(p51['analysis'], 'Q0044_per_condition_R.json')
            if os.path.exists(q44_path):
                try:
                    with open(q44_path) as f: q44 = json.load(f)
                    per_run = q44.get('per_run', [])
                    valid = [e for e in per_run if not e.get('skipped') and e.get('frac_R') is not None]
                    if valid:
                        valid.sort(key=lambda e: e.get('run_num', 0))
                        r_vals = [e['frac_R'] for e in valid]
                        ranks = (np.argsort(np.argsort(r_vals)) + 1).tolist()
                        condition_ranks.append({
                            'model': m['label'],
                            'conditions': [e.get('label', f"R{e['run_num']}") for e in valid],
                            'r_fractions': r_vals,
                            'ranks': ranks,
                        })
                except Exception:
                    pass

        out_path = os.path.join(out_dir, 'Q0053_condition_concordance.json')
        with open(out_path, 'w') as f:
            json.dump({'run_num': 51, 'models': condition_ranks,
                       'n_models': len(condition_ranks)}, f, indent=2, default=str)
        ui.ok(f"Run 0053 concordance: {out_path}")

    ui.blank()


def _run_layer_isolation_analysis(session, paths):
    """Run 0018 analysis pass — produce Q0018_layer_isolation.json from
    R0018_layer_isolation.csv.

    Per-layer aggregates: output_change_rate, n_trials, n_changed, p_binomial.

    v0.80.0.34: production wire-in. Layer set discovered from the CSV's
    patch_layer column, not hardcoded. Different model sizes have different
    layer counts (LLaMA 8B has 32 transformer blocks, Gemma 2B has 18) —
    discovery handles all configurations correctly. Binomial null: one-sided
    test against the unpatched-control rate (patch_layer='none'). When that
    baseline is 0 (always, in this experiment), the test is degenerate and
    p_binomial is None for all layers. The JSON's binomial_null field marks
    the regime.
    """
    import pandas as pd
    from scipy.stats import binomtest

    hidden_dir = paths['hidden']
    ana_dir    = paths['analysis']
    csv_dir    = os.path.join(os.path.dirname(hidden_dir), 'csv')
    os.makedirs(ana_dir, exist_ok=True)
    out_file = os.path.join(ana_dir, 'Q0018_layer_isolation.json')
    csv_path = os.path.join(csv_dir, 'R0018_layer_isolation.csv')

    ui.section("Run 0018 — Layer Causal Sufficiency Analysis (no GPU)")

    if not os.path.exists(csv_path):
        ui.warn(f"  R0018_layer_isolation.csv not found at {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as e:
        ui.err(f"  Failed to read CSV: {e}")
        return None

    if 'patch_layer' not in df.columns or 'output_changed' not in df.columns:
        ui.err(f"  Missing required columns. Saw: {list(df.columns)[:8]}")
        return None

    df['_oc'] = pd.to_numeric(df['output_changed'], errors='coerce')
    df = df.dropna(subset=['_oc'])

    baseline = df[df['patch_layer'] == 'none']
    n_baseline = int(len(baseline))
    p_null = float(baseline['_oc'].mean()) if n_baseline > 0 else None

    patched_layers = df[df['patch_layer'] != 'none']['patch_layer'].unique()

    def _layer_depth(L):
        digits = ''.join(c for c in str(L) if c.isdigit())
        try:
            return int(digits) if digits else 9999
        except ValueError:
            return 9999

    layers_sorted = sorted(patched_layers, key=_layer_depth)

    per_layer = {}
    for layer in layers_sorted:
        sub = df[df['patch_layer'] == layer]['_oc']
        n_trials = int(len(sub))
        if n_trials < 5:
            continue
        n_changed = int(sub.sum())
        rate = n_changed / n_trials

        p_bin = None
        if p_null is not None and 0.0 < p_null < 1.0:
            try:
                p_bin = float(binomtest(n_changed, n_trials, p_null,
                                         alternative='greater').pvalue)
            except Exception:
                p_bin = None

        per_layer[str(layer)] = {
            'output_change_rate': float(rate),
            'n_trials':           n_trials,
            'n_changed':          n_changed,
            'p_binomial':         p_bin,
        }

    binomial_null_kind = (
        'vs_baseline_rate'
        if (p_null is not None and 0.0 < p_null < 1.0)
        else 'degenerate_baseline_test_skipped'
    )

    out = {
        'run_num':           18,
        'schema_version':    '0.80.0-q18.1',
        'per_layer':         per_layer,
        'baseline': {
            'rate':              p_null,
            'n_trials':          n_baseline,
            'patch_layer_value': 'none',
        },
        'binomial_null':     binomial_null_kind,
        'n_layers_analyzed': len(per_layer),
        'source_csv':        os.path.basename(csv_path),
    }

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)

    ui.ok(f"  Q0018_layer_isolation.json written. {len(per_layer)} layers analyzed.")
    return out
