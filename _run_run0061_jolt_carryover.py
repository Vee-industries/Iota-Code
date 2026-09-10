"""
_run_run0061_jolt_carryover.py -- Item 13 hidden-state cohort dispersion analysis.

Tests whether the four shock_variant cohorts in Run 0039 produce reliably
different hidden states at post-shock turns 14, 15, 16 (k = 1, 2, 3) under
the stateless-turn condition with byte-identical test-turn prompts (turn
13+k input is fixed JOLT_PROMPTS, only what happened at turn 13 varies;
every turn is a full re-prefill of the threaded transcript).

Per cell, per k:
  - Load R0039_jolt.csv for trial -> shock_variant mapping
  - Load hidden state for (trial, turn=13+k) for every trial
  - Group by shock_variant (4 cohorts of ~25 trials each)
  - Compute F-like statistic = between-cohort SS / within-cohort SS
  - Permutation test: shuffle shock_variant labels 1000 times, recompute
  - p-value = fraction of perms where shuffled stat >= observed

Aggregation:
  - fleet-mean p at each k
  - n_cells_reliable at p < 0.05 per k
  - decay characterization: does fleet-mean p increase k=1 -> k=3?
  - per-cell monotonic-decay flag

Output: data/paper/behavioral_trial_response/run_0061_jolt_carryover_analysis.json

ASCII-only. No GPU needed; pure CSV + .npy file reads + numpy.
"""

import os, sys, json, glob, time, re

# UTF-8 + warning silence (matches start_here.py convention)
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from cartography import get_paths, run_prefix


# ===== Configuration =====
# Default source is Run 0039 (thematic recovery, Paper A §3 headline test).
# Override via --run <NN> on the command line to point at a sibling run
# with the same protocol structure (currently: Run 0060, non-thematic
# recovery, the §7 follow-up). The output filename incorporates the run
# number so the two analyses don't collide.
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument('--run', type=int, default=39,
                help='source run number (39 = thematic R0039, '
                     '60 = non-thematic R0060)')
_args, _unknown = _p.parse_known_args()

RUN_NUM       = _args.run
RUN_PFX       = run_prefix(RUN_NUM)  # 'Q' for runs > 21, 'R' for proof runs
# _run_jolt writes its CSV with a hardcoded 'R' prefix regardless of
# run_prefix(), so the CSV path uses 'R' rather than RUN_PFX. Hidden-state
# .npy files use RUN_PFX naturally via save_npy.
CSV_PFX       = 'R'
SHOCK_AT      = 13       # turn at which shock is injected
N_TURNS       = 16       # total turns under both R0039 and R0060
N_COHORTS     = 4        # SHOCK_VARIANTS length (asserted at runners_prompts.py:143)
K_VALUES      = [1, 2, 3]
N_PERMUTATIONS = 1000
PERM_SEED     = 42
P_RELIABLE    = 0.05

# Output filename varies by source run so R0039 and R0060 outputs don't
# overwrite each other. The Phase D analyzer reads both for the side-by-side
# spontaneous-retention vs input-triggered-reactivation comparison.
_OUT_BASENAME = (f'run_0061_jolt_carryover_analysis.json' if RUN_NUM == 39
                 else f'run_0061_jolt_carryover_analysis_r{RUN_NUM:04d}.json')
OUT_REL_PATH  = os.path.join('data', 'paper', 'behavioral_trial_response',
                              _OUT_BASENAME)


# ===== Display-name discovery (reuse the hardened logic from Item 12) =====
def discover_display_name(hidden_dir):
    """Mode-vote across files to pick the canonical model display tag.
    Hardens against stale per-run tag conventions; matches the
    _run_reviewer_three_response.py:_discover_display_name fix shipped in
    v0.82.0.26.
    """
    if not os.path.isdir(hidden_dir):
        return None
    pat = os.path.join(hidden_dir, f'{RUN_PFX}{RUN_NUM:04d}_*_trial*_turn*.npy')
    files = sorted(glob.glob(pat))
    if not files:
        return None
    rx = re.compile(rf'^{RUN_PFX}{RUN_NUM:04d}_(.+)_trial\d+_turn\d+\.npy$')
    candidates = [f for f in files
                  if not os.path.basename(f).endswith(
                      ('_et_base.npy', '_it_instruct.npy', '_alllayers.npy'))]
    if not candidates:
        return None
    n = len(candidates)
    if n <= 200:
        sample = candidates
    else:
        idxs = [int(n * i / 200) for i in range(200)]
        sample = [candidates[i] for i in idxs]
    import collections as _co
    tags = []
    for f in sample:
        m = rx.match(os.path.basename(f))
        if m:
            tags.append(m.group(1))
    if not tags:
        return None
    return _co.Counter(tags).most_common(1)[0][0]


# ===== Cell discovery =====
def discover_cells():
    """Find all (family, size, variant, temp) cells with a Run 0039 CSV.
    Return list of (cell_key, family, size, variant, temp) tuples sorted
    by cell_key.
    """
    data_root = os.path.join(ROOT, 'data')
    cells = []
    pat = os.path.join(data_root, '*', '*', '*', '*', 'csv',
                        f'{CSV_PFX}{RUN_NUM:04d}_jolt.csv')
    for csv_path in sorted(glob.glob(pat)):
        # csv_path like: data/llama/8b_4bit/abliterated/temp_0.2/csv/R0039_jolt.csv
        parts = os.path.normpath(csv_path).split(os.sep)
        if len(parts) < 6:
            continue
        # walking back from csv/R0039_jolt.csv
        temp_dir = parts[-3]
        variant = parts[-4]
        size    = parts[-5]
        family  = parts[-6]
        # Map temp_dir to numeric temperature
        if temp_dir == 'deterministic':
            temp = 0.0
            t_tag = 't00'
        elif temp_dir.startswith('temp_'):
            try:
                temp = float(temp_dir.split('_', 1)[1])
                t_tag = f't{int(round(temp * 10)):02d}'
            except ValueError:
                continue
        else:
            continue
        # Cell key matches Item 12 convention: family_size_variant_t##
        # but size has '_4bit' or '_fp16' which writerbot writes as 'q4'/'fp16'
        # Use Item 12 format: gemma_2b_q4_abliterated_t00 etc.
        size_tag = size.replace('_4bit', '_q4').replace('_8bit', '_q8')
        cell_key = f"{family}_{size_tag}_{variant}_{t_tag}"
        cells.append((cell_key, family, size, variant, temp))
    return cells


# ===== Per-cell statistic =====
def f_like_statistic(vectors, labels):
    """F-like statistic: between-cohort SS / within-cohort SS in cosine-
    distance space.

    vectors: (n, d) array of cohort vectors
    labels:  (n,) array of cohort labels (integers)

    Returns scalar: SS_between / max(SS_within, eps).
    """
    # Normalize to unit length so dot product = cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    V = vectors / norms

    grand_mean = V.mean(axis=0)
    # Re-project grand mean to unit; if it's zero (degenerate), fall back.
    gn = np.linalg.norm(grand_mean)
    if gn > 0:
        grand_mean = grand_mean / gn

    unique_labels = np.unique(labels)
    ss_between = 0.0
    ss_within  = 0.0
    for lab in unique_labels:
        mask = labels == lab
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        V_g = V[mask]
        m_g = V_g.mean(axis=0)
        mn = np.linalg.norm(m_g)
        if mn > 0:
            m_g_unit = m_g / mn
        else:
            m_g_unit = m_g
        # Between: cohort-mean vs grand-mean cosine distance, weighted by n_g
        cos_bg = float(np.dot(m_g_unit, grand_mean))
        cos_bg = max(-1.0, min(1.0, cos_bg))
        ss_between += n_g * (1.0 - cos_bg)
        # Within: per-vector vs cohort-mean cosine distance
        cos_within = V_g @ m_g_unit  # (n_g,)
        cos_within = np.clip(cos_within, -1.0, 1.0)
        ss_within += float(np.sum(1.0 - cos_within))

    eps = 1e-12
    return ss_between / max(ss_within, eps)


def permutation_p(vectors, labels, n_perms=N_PERMUTATIONS, seed=PERM_SEED):
    """Permutation test on f_like_statistic. Returns (observed_stat, p_value).
    """
    observed = f_like_statistic(vectors, labels)
    rng = np.random.RandomState(seed)
    perm_stats = np.empty(n_perms, dtype=np.float64)
    perm_labels = labels.copy()
    for i in range(n_perms):
        rng.shuffle(perm_labels)
        perm_stats[i] = f_like_statistic(vectors, perm_labels)
    # one-sided p: fraction of perms >= observed (use +1/+1 to avoid p=0)
    n_ge = int(np.sum(perm_stats >= observed))
    p = (n_ge + 1) / (n_perms + 1)
    return float(observed), float(p)


# ===== Per-cell processing =====
def process_cell(cell_key, family, size, variant, temp, progress):
    """Load CSV + hidden states for one cell, compute stat + p-value at each k.

    Returns dict: {
        'k1': {'observed_stat': ..., 'p_value': ..., 'n_trials': ..., 'cohort_sizes': [...]},
        'k2': {...},
        'k3': {...},
        'shock_at': 13,
        'cohort_count': 4,
    }
    or {'skipped': '<reason>'} on failure.
    """
    paths = get_paths(family, size, variant, temp, create_dirs=False)
    csv_path = os.path.join(paths['csv'], f'{CSV_PFX}{RUN_NUM:04d}_jolt.csv')
    hidden_dir = paths['hidden']

    if not os.path.isfile(csv_path):
        return {'skipped': f'CSV missing: {csv_path}'}
    if not os.path.isdir(hidden_dir):
        return {'skipped': f'hidden_dir missing: {hidden_dir}'}

    # Load CSV. Columns we need: trial, turn, shock_variant.
    df = pd.read_csv(csv_path)
    needed = {'trial', 'turn', 'shock_variant'}
    missing = needed - set(df.columns)
    if missing:
        return {'skipped': f'CSV columns missing: {sorted(missing)}'}

    # Discover model display name from the actual filenames in hidden_dir.
    display = discover_display_name(hidden_dir)
    if display is None:
        return {'skipped': 'no canonical R0039_*.npy files in hidden_dir'}

    result = {
        'shock_at': SHOCK_AT,
        'cohort_count': N_COHORTS,
        'model_display_tag': display,
    }

    for k in K_VALUES:
        target_turn = SHOCK_AT + k
        # Pull per-trial rows at target_turn
        df_k = df[df['turn'] == target_turn].copy()
        if len(df_k) == 0:
            result[f'k{k}'] = {'skipped': f'no rows at turn={target_turn}'}
            continue

        vectors = []
        labels  = []
        miss_count = 0
        for _, row in df_k.iterrows():
            trial = int(row['trial'])
            sv    = int(row['shock_variant'])
            fname = f'{RUN_PFX}{RUN_NUM:04d}_{display}_trial{trial:04d}_turn{target_turn:02d}.npy'
            fpath = os.path.join(hidden_dir, fname)
            if not os.path.isfile(fpath):
                miss_count += 1
                continue
            try:
                vec = np.load(fpath).flatten().astype(np.float32)
            except Exception:
                miss_count += 1
                continue
            vectors.append(vec)
            labels.append(sv)

        if len(vectors) < N_COHORTS * 5:
            result[f'k{k}'] = {
                'skipped': f'insufficient trials ({len(vectors)}) for {N_COHORTS}-cohort test',
                'rows_in_csv': int(len(df_k)),
                'files_missing': miss_count,
            }
            continue

        # Check that all cohorts are represented and shapes match
        labels_arr = np.array(labels, dtype=np.int32)
        unique_labels = np.unique(labels_arr)
        if len(unique_labels) != N_COHORTS:
            result[f'k{k}'] = {
                'skipped': f'expected {N_COHORTS} cohorts, found {len(unique_labels)}',
            }
            continue

        # Stack into matrix (require shape consistency)
        first_shape = vectors[0].shape
        if not all(v.shape == first_shape for v in vectors):
            result[f'k{k}'] = {
                'skipped': 'inconsistent hidden-state shapes across trials',
            }
            continue
        V = np.stack(vectors).astype(np.float32)

        cohort_sizes = [int(np.sum(labels_arr == lab)) for lab in unique_labels]

        # Run the permutation test
        observed, p = permutation_p(V, labels_arr,
                                       n_perms=N_PERMUTATIONS,
                                       seed=PERM_SEED + k)
        result[f'k{k}'] = {
            'observed_stat': observed,
            'p_value': p,
            'n_trials': int(len(vectors)),
            'cohort_sizes': cohort_sizes,
            'files_missing': miss_count,
            'pool_dim': int(first_shape[0]),
        }

    return result


# ===== Aggregation =====
def aggregate_fleet(per_cell):
    """Compute fleet-mean p, n_reliable, monotonic-decay characterization."""
    out = {}
    for k in K_VALUES:
        ks = f'k{k}'
        ps = []
        for cell_key, cell_res in per_cell.items():
            if 'skipped' in cell_res:
                continue
            kres = cell_res.get(ks, {})
            if 'skipped' in kres or 'p_value' not in kres:
                continue
            ps.append(kres['p_value'])
        out[ks] = {
            'fleet_mean_p': float(np.mean(ps)) if ps else None,
            'fleet_median_p': float(np.median(ps)) if ps else None,
            'n_cells_with_p': len(ps),
            'n_cells_reliable_at_0_05': int(sum(1 for p in ps if p < P_RELIABLE)),
        }

    # Monotonic decay: per-cell flag = is p_k1 <= p_k2 <= p_k3?
    decay_flags = []
    for cell_key, cell_res in per_cell.items():
        if 'skipped' in cell_res:
            continue
        try:
            p1 = cell_res['k1']['p_value']
            p2 = cell_res['k2']['p_value']
            p3 = cell_res['k3']['p_value']
        except (KeyError, TypeError):
            continue
        decay_flags.append({
            'cell': cell_key,
            'p_k1': p1, 'p_k2': p2, 'p_k3': p3,
            'monotonic_decay': bool(p1 <= p2 <= p3),
            'monotonic_increase': bool(p1 >= p2 >= p3),
        })

    n_with_all_three = len(decay_flags)
    n_monotonic_decay = sum(1 for d in decay_flags if d['monotonic_decay'])
    out['decay_characterization'] = {
        'n_cells_with_all_three_k': n_with_all_three,
        'n_cells_monotonic_decay': n_monotonic_decay,
        'fraction_monotonic_decay': (n_monotonic_decay / n_with_all_three
                                     if n_with_all_three else None),
        'fleet_mean_p_trajectory': [
            out[f'k{k}']['fleet_mean_p'] for k in K_VALUES
        ],
        'fleet_mean_p_monotonic_increase': (
            out['k1']['fleet_mean_p'] is not None
            and out['k2']['fleet_mean_p'] is not None
            and out['k3']['fleet_mean_p'] is not None
            and out['k1']['fleet_mean_p'] <= out['k2']['fleet_mean_p'] <= out['k3']['fleet_mean_p']
        ),
        'per_cell_decay': decay_flags,
    }
    return out


# ===== Main =====
def main():
    t0 = time.time()
    print()
    print("=" * 70)
    print(f" Run 0061 -- Jolt-carryover hidden-state cohort dispersion")
    print(f" iota v0.82.0.26, started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 70)
    print()

    # Discover cells
    cells = discover_cells()
    print(f"  [+ {time.time()-t0:6.1f}s] discovered {len(cells)} cells with R{RUN_NUM:04d}_jolt.csv")
    for ck, *_ in cells:
        print(f"      {ck}")
    print()

    # Process each cell
    per_cell = {}
    for i_cell, (cell_key, family, size, variant, temp) in enumerate(cells):
        t_cell = time.time()
        def progress(msg):
            print(f"  [+ {time.time()-t0:6.1f}s]   {msg}", flush=True)
        progress(f"[{i_cell+1}/{len(cells)}] {cell_key}")
        try:
            result = process_cell(cell_key, family, size, variant, temp, progress)
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {'skipped': f'EXCEPTION: {type(e).__name__}: {e}'}

        per_cell[cell_key] = result
        if 'skipped' in result:
            progress(f"    SKIPPED: {result['skipped']}")
        else:
            line_parts = []
            for k in K_VALUES:
                kres = result.get(f'k{k}', {})
                if 'p_value' in kres:
                    line_parts.append(f"k{k}: p={kres['p_value']:.4f} (n={kres['n_trials']})")
                elif 'skipped' in kres:
                    line_parts.append(f"k{k}: SKIPPED ({kres['skipped']})")
            progress(f"    {'  |  '.join(line_parts)}")
            progress(f"    ({time.time()-t_cell:.1f}s)")

    # Aggregate
    print()
    print(f"  [+ {time.time()-t0:6.1f}s] aggregating fleet stats...")
    aggregates = aggregate_fleet(per_cell)
    n_processed = sum(1 for r in per_cell.values() if 'skipped' not in r)
    print(f"  [+ {time.time()-t0:6.1f}s] cells processed: {n_processed}/{len(cells)}")
    for k in K_VALUES:
        ks = f'k{k}'
        a = aggregates[ks]
        fmp = a['fleet_mean_p']
        fmp_str = f"{fmp:.4f}" if fmp is not None else "None"
        print(f"    k={k}: fleet_mean_p={fmp_str}, "
              f"reliable@0.05: {a['n_cells_reliable_at_0_05']}/{a['n_cells_with_p']}")
    dc = aggregates['decay_characterization']
    print(f"    monotonic decay (p_k1<=p_k2<=p_k3): "
          f"{dc['n_cells_monotonic_decay']}/{dc['n_cells_with_all_three_k']} cells; "
          f"fleet trajectory monotonic: {dc['fleet_mean_p_monotonic_increase']}")

    # Write output
    out_path = os.path.join(ROOT, OUT_REL_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        'phase': 'run_0061_jolt_carryover',
        'iota_version': '0.82.0.26',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source_run': RUN_NUM,
        'shock_at': SHOCK_AT,
        'n_turns': N_TURNS,
        'n_cohorts': N_COHORTS,
        'k_values': K_VALUES,
        'n_permutations': N_PERMUTATIONS,
        'p_reliable_threshold': P_RELIABLE,
        'cells_total': len(cells),
        'cells_processed': n_processed,
        'cells': per_cell,
        'aggregates': aggregates,
    }
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=float)
    os.replace(tmp, out_path)
    print()
    print(f"  [+ {time.time()-t0:6.1f}s] wrote {out_path}")
    print()
    print("=" * 70)
    print(" DONE")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
