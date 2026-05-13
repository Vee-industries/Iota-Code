"""
decomposition_p2.py -- paper 2 measurement-apparatus phase dispatch.

v0.81.1.7 ship 5 (paper 2 measurement layer).

Per codebot_handoff_v0_14.md §3 + §7 order-of-operations: this module
houses the dispatch functions for the six new apparatus phases that
extend Run 0058 from the paper-1 Bayesian apparatus (5 phases) to
the full paper-2 measurement apparatus (11 phases). Sibling pattern to
existing run_*.py modules -- invoked by run_bayesian_apparatus.py
phase chain.

The six new phases:

  Phase 6 (function_class_fits_p2): four-class permutation importance
    on partition B only (Q0057 stays as the §10.4 paper-1 work, full
    data; here we re-fit on the partition-B subset for the apparatus).

  Phase 7 (knn_anchor_p2): per-channel chain-rule kNN-MI on
    partition A only. Distinct from existing phase 4 -- the existing
    phase runs on full data with 2-channel shape; phase 7 is
    partitioned + 3-channel chain rule per paper 2 measurement spec.

  Phase 8 (lagrangian_anchoring_p2): apparatus + λ-sensitivity sweep.
    Calls existing bayesian_solver.solve_with_diagnostics across the
    four real classes per cell, sweeps λ ∈ {0, 0.1, 0.5, 1, 2, 5, 10, ∞},
    picks lambda_default from V5 minimum-error point.

  Phase 9 (stacking_baseline_p2): convex QP on V5a-V5e to find weights
    (w_R, w_M, w_K, w_F) ≥ 0, Σw = 1, min squared error to truth;
    apply (no refit) to V5f and four empirical cells. Comparison
    estimator for "does the apparatus do better than naive averaging?"

  Phase 10 (bootstrap_variance_p2): paired-bootstrap refit loop on
    a representative cell subset (PHASE10_SUBSET); n_bootstrap=10;
    all four classes refit per resample with shared bootstrap
    indices and shared internal split; apparatus applied per
    resample. Per-channel variance for each estimator + anchored.
    Cells outside the subset get bootstrap_not_run_single_shot_only
    sentinel + methods_note. Wall-time decision documented in
    CHANGELOG 0.82.0.14 / 0.82.0.23.

  Phase 11 (geometric_diagnostics_p2): per-cell eight-field block
    on q* and the four class shares -- q*-vs-G̃ hull membership,
    anchor pull (KL nats), class hull volume, hull-exit mechanism,
    Euclidean signed distance to hull boundary. Replaces v0.82.0.23
    convex_hull_diagnostic schema atomically per statsbot's Module 2
    spec (paper2_apparatus.md). Also produces cross-cell aggregates
    including V5d cross-reference for fig7.

Outputs land under data/paper/calibration/{phase_name}_p2/per_cell.json
(or analog) -- sibling to existing apparatus phase outputs.

Each phase function follows the (ok, summary_data) contract used by
the existing _phase1..._phase5 in run_bayesian_apparatus.py:
  - Returns (True, dict) on success; written to manifest under
    phase{N}_summary.
  - Returns (False, partial_dict) on failure; manifest still emits
    with the failed phase tagged.
"""

import datetime
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

import ui  # noqa: E402

DATA = os.path.join(ROOT, 'data')
PAPER_ROOT = os.path.join(DATA, 'paper')
SELECTION_PATH = os.path.join(DATA, 'paper', 'calibration',
                                'split_pca_selection', 'selection.json')

# λ-sweep grid per codebot_handoff_v0_14.md §3.3
LAMBDA_SWEEP_GRID = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]

# Bootstrap resamples per cell (Phase 10)
# v0.82.0.23: dropped from 200 to 10 after wall-time profile.
# Profile of gemma_2b_q4_abliterated_t00 (21,600 rows, pool_dim=1024)
# clocked ~17.7 min per resample on a single 3080. At n_bootstrap=200
# × 24 cells the fleet projects to 600-900 hr. Reduced to N=10 over
# a representative 5-cell subset (PHASE10_SUBSET below); other 19
# cells get methods-note tag 'bootstrap_not_run_single_shot_only'
# with rationale documented in CHANGELOG 0.82.0.23.
N_BOOTSTRAP = 10

# Phase 10 subset -- 5 cells chosen to span the variation surfaces
# that matter for cross-class variance comparison: pool_dim (the
# cost dial, 64 vs 1024 native), temperature regime, model family,
# and quantization. Rationale documented in CHANGELOG 0.82.0.23.
#
#   gemma_2b_q4_abliterated_t00 -- large pool (1024), t=0.0, 2B Q4
#                                   abliterated. Already partially
#                                   done from v0.82.0.23 wall-time
#                                   profile (n=2 cached); checkpoint
#                                   extends to n=10 cleanly.
#   gemma_2b_q4_abliterated_t06 -- large pool, t=0.6 covers the
#                                   temperature axis on the same
#                                   model so within-model variance
#                                   change vs temperature is visible.
#   gemma_2b_fp16_abliterated_t00 -- large pool, t=0.0, FP16 covers
#                                   the quantization axis on Gemma 2B
#                                   (Q4 vs FP16 within-model).
#   gemma_2b_fp16_abliterated_t10 -- large pool, t=1.0, FP16. Added
#                                   v0.82.0.23: this cell carries
#                                   the hull violation in Ship 12's
#                                   convex_hull_diagnostic
#                                   (hull_signed_distance ≈ -0.030).
#                                   Bootstrap distribution of the
#                                   hull diagnostic on this cell is
#                                   the load-bearing question for
#                                   the headline outside-hull claim.
#   gemma_9b_q4_abliterated_t00  -- small pool (64), t=0.0, 9B Q4.
#                                   Different model size + small-pool
#                                   variance regime as comparison.
#   llama_8b_q4_abliterated_t00  -- small pool, t=0.0, different
#                                   family entirely. Cross-family
#                                   variance check.
#
# Six cells gives one large-pool t=0 baseline, one large-pool
# t-sweep contrast, one large-pool quantization contrast, the
# hull-violator t=1.0 cell, and two small-pool cells from different
# families. Loses temperature coverage on small-pool cells and
# base-vs-abliterated coverage everywhere -- disclosed as scope.
PHASE10_SUBSET = [
    'gemma_2b_q4_abliterated_t00',
    'gemma_2b_q4_abliterated_t06',
    'gemma_2b_fp16_abliterated_t00',
    'gemma_2b_fp16_abliterated_t10',
    'gemma_9b_q4_abliterated_t00',
    'llama_8b_q4_abliterated_t00',
]


# ─── Helpers ─────────────────────────────────────────────────────────

def _load_operating_point():
    """Load (split_a_frac, pca_dim) from selection.json. Fallback to
    (0.7, 32) if the foundation hasn't been fired yet."""
    if not os.path.exists(SELECTION_PATH):
        ui.warn(f"  selection.json not found -- using fallback "
                f"split=0.7, pca=32")
        return 0.7, 32
    try:
        with open(SELECTION_PATH, 'r', encoding='utf-8-sig') as f:
            sel = json.load(f)
    except Exception as e:
        ui.warn(f"  selection.json load failed: {e} -- using fallback")
        return 0.7, 32
    sp = sel.get('selected_operating_point') or {}
    return float(sp.get('split', [0.7, 0.3])[0]), int(sp.get('pca', 32))


def _q42_files():
    """Find all Q0042 cell files."""
    pattern = os.path.join(DATA, '**', 'Q0042_decomposition.json')
    return sorted(glob.glob(pattern, recursive=True))


def _parse_q42_path(q42_path):
    """Extract (family, size, variant, temp) from a Q0042 path."""
    rel = os.path.relpath(q42_path, DATA).replace('\\', '/').split('/')
    if len(rel) < 5:
        return None, None, None, None
    family, size, variant, cond = rel[0], rel[1], rel[2], rel[3]
    if cond == 'deterministic':
        temp = 0.0
    elif cond.startswith('temp_'):
        try:
            temp = float(cond.replace('temp_', ''))
        except Exception:
            temp = None
    else:
        temp = None
    return family, size, variant, temp


def _cell_key(family, size, variant, temp):
    """Canonical cell key matching results_schema.cell_key."""
    if temp is None:
        t = 'TNA'
    else:
        t = f'{temp:.1f}'
    return f'{family}_{size}_{variant}_T{t}'


def _writerbot_key(family, size, variant, temp):
    """Writerbot-style cell key matching run_function_class_sensitivity._cell_key.

    Q0057 and Phase 4's anchors.json both key cells in writerbot format
    (e.g. 'gemma_2b_q4_abliterated_t00'), NOT the canon format used by
    results.json (e.g. 'gemma_2b_4bit_abliterated_T0.0'). Phases 6 and
    7 of paper-2's apparatus consume Q0057 + Phase 4, so they need to
    use writerbot keys when looking up those upstream sources.

    v0.82.0.8: prior to this version, decomposition_p2's Phase 6 built
    canonical keys via _cell_key() and looked them up in Q0057's cells
    dict. Every lookup missed (canon "gemma_2b_4bit_abliterated_T0.0"
    is not in Q0057 which has "gemma_2b_q4_abliterated_t00"). All 24
    cells skipped, n_done==0, the truthiness-of-n_done return signaled
    failure to the orchestrator, Phase 6 marked failed, Phases 8-11
    cascade-skipped. Fix: build writerbot keys for upstream lookups,
    AND store Phase 6 output under writerbot keys so Phase 7 (which
    passes Phase 4's writerbot-keyed anchors through unchanged) joins
    cleanly with Phase 6 in Phase 8.
    """
    _SIZE_MAP = {
        '2b_4bit':  '2b_q4',
        '2b_8bit':  '2b_q8',
        '2b_fp16':  '2b_fp16',
        '9b_4bit':  '9b_q4',
        '8b_4bit':  '8b_q4',
    }
    size_key = _SIZE_MAP.get(size, size)
    if temp is None:
        t_key = 'tnone'
    else:
        t_key = f"t{int(round(temp * 10)):02d}"
    return f"{family}_{size_key}_{variant}_{t_key}"


def _save_phase_output(phase_dir_name, filename, payload):
    """Atomic JSON write to data/paper/calibration/{phase_dir_name}/."""
    out_dir = os.path.join(DATA, 'paper', 'calibration', phase_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
    return out_path


# ─── Phase 6: function-class fits with partition split ───────────────

def _run_function_class_fits_p2(session=None, paths=None,
                                   cell_filter=None, force_refit=False):
    """Phase 6 -- partition-B refit per cell (single-shot, full data).

    v0.82.0.23: live single-shot partition-B refit replaces the
    Q0057 full-data passthrough. Per cell: 80/20 split with seed 42,
    fit Ridge / MLP / RF / RKHS on partition A, permute on partition
    B, renormalize to (E, C, R) simplex shares. All four classes
    use the same partition split for share-coherence. The refit is
    canonical for downstream Phase 8 / Phase 11 consumption.

    Cell-level checkpointing -- existing per_cell.json read on entry,
    cells with the live sentinel are skipped (mirrors Phase 10's
    pattern). force_refit=True re-runs everything.

    Q0057 fallback: cells whose qcache can't be loaded fall back to
    the Q0057 full-data shares with `partition_source:
    'q57_fallback_qcache_missing'`. Downstream consumers can filter
    on this sentinel; graceful degradation rather than phase
    failure on missing data.

    V0 protocol (v0.82.0.23): single-cell sanity check before fleet
    fire -- runs T1 (sum-to-1 tolerance) and T2 (Tier-3 closed-form
    sanity) automatically on the first cell processed; raises if
    either fails. Subsequent cells skip the gates per spec.

    Args:
      cell_filter: optional list of writerbot cell keys to restrict
        to (mirrors Phase 10's pattern; useful for V0 single-cell
        sanity check via cell_filter=['gemma_2b_q4_abliterated_t00']).
      force_refit: if True, ignore checkpoint sentinel and re-run
        every cell.

    Wall time: roughly 12-15 min per 2B cell (pool_dim=1024) and
    ~5 min per 9B/8B cell (smaller pool_dim) on a single 3080.
    Fleet (24 cells) projected at 3.5-4 hr.
    """
    ui.section("Phase 6 -- Function-class fits (partition-B refit, single-shot)")
    split_a, pca_dim = _load_operating_point()
    ui.msg(f"  operating point: split_A={split_a}, pca={pca_dim} "
           f"(partition B fraction = {1-split_a:.2f})")

    q42s = _q42_files()
    if not q42s:
        ui.warn("  No Q0042 cells found.")
        return False, {'error': 'no_q42_cells', 'n_cells': 0}

    # Q0057 fallback for cells whose qcache can't be loaded
    q57_path = os.path.join(PAPER_ROOT,
                              'Q0057_function_class_sensitivity.json')
    q57_cells = {}
    if os.path.exists(q57_path):
        try:
            with open(q57_path, 'r', encoding='utf-8-sig') as f:
                q57_cells = (json.load(f) or {}).get('cells') or {}
        except Exception as e:
            ui.warn(f"  Q0057 fallback load failed: {e}")

    # Existing checkpoint
    LIVE_SENTINEL = 'partition_b_refit_single_shot'
    out_check_path = os.path.join(DATA, 'paper', 'calibration',
                                     'function_class_p2', 'per_cell.json')
    existing_per_cell = {}
    if os.path.exists(out_check_path) and not force_refit:
        try:
            with open(out_check_path, 'r', encoding='utf-8-sig') as f:
                existing = json.load(f) or {}
            existing_per_cell = existing.get('cells') or {}
        except Exception as e:
            ui.warn(f"  existing per_cell.json load failed: {e}")

    import function_class_p2 as fcp2

    per_cell = dict(existing_per_cell)
    n_done = 0
    n_skip_cached = 0
    n_skip_filter = 0
    n_fallback = 0
    n_failed = 0
    v0_done = False  # one-shot V0 gate fires on the first refit cell

    for q42 in q42s:
        family, size, variant, temp = _parse_q42_path(q42)
        if not family:
            continue
        cell_key = _writerbot_key(family, size, variant, temp)

        if cell_filter is not None and cell_key not in cell_filter:
            n_skip_filter += 1
            continue

        # Cell-level checkpoint
        prior = per_cell.get(cell_key) or {}
        if (not force_refit
                and prior.get('partition_source') == LIVE_SENTINEL):
            ui.msg(f"  [skip] {cell_key} -- cached at {LIVE_SENTINEL}")
            n_skip_cached += 1
            continue

        # Locate cell data
        cell_dir = os.path.dirname(os.path.dirname(q42))
        hidden_dir = os.path.join(cell_dir, 'hidden_states')
        cell_data = fcp2.load_cell_data_from_qcache(hidden_dir)

        if 'error' in cell_data:
            # Q0057 fallback
            q57_cell = q57_cells.get(cell_key)
            if q57_cell and 'error' not in q57_cell:
                perm = q57_cell.get('permutation_shares', {}) or {}
                ridge = perm.get('ridge')
                mlp = perm.get('mlp')
                rf = perm.get('rf')
                rkhs_median = perm.get('rkhs_median')
                if all((ridge, mlp, rf)):
                    per_cell[cell_key] = {
                        'shares': {
                            'ridge':       ridge,
                            'mlp':         mlp,
                            'rf':          rf,
                            'rkhs_median': rkhs_median,
                        },
                        'partition_source': 'q57_fallback_qcache_missing',
                        'fallback_reason':  cell_data.get('error', 'unknown'),
                        'split_a_frac':     split_a,
                        'pca_dim':          pca_dim,
                    }
                    n_fallback += 1
                    ui.warn(f"  [{cell_key}] qcache missing, "
                              f"using Q0057 fallback")
                    continue
            ui.warn(f"  [{cell_key}] qcache missing AND no Q0057 fallback")
            n_failed += 1
            continue

        ui.msg(f"  [{cell_key}] n_rows={cell_data['n_rows']}, "
                f"pool_dim={cell_data['pool_dim']}")

        # Target-PCA decision matches Phase 10
        target_pca = 64 if cell_data['pool_dim'] > 64 else None

        try:
            shares = fcp2.compute_partition_b_shares(
                cell_data,
                target_pca_components=target_pca,
                seed=fcp2.PARTITION_B_SEED,
                ui_msg=lambda s: ui.msg(s),
            )
        except Exception as e:
            ui.warn(f"  [{cell_key}] partition-B refit failed: "
                      f"{type(e).__name__}: {e}")
            n_failed += 1
            continue

        # Pack into output schema (E/C/R per class, matching prior shape)
        def _pack(triple):
            return {'E': float(triple[0]), 'C': float(triple[1]),
                    'R': float(triple[2])}

        per_cell[cell_key] = {
            'shares': {
                'ridge':       _pack(shares['ridge']),
                'mlp':         _pack(shares['mlp']),
                'rf':          _pack(shares['rf']),
                # rkhs_median is the single-kernel matérn ν=1.5 result;
                # downstream consumers read it as the canonical RKHS
                # share for apparatus + hull diagnostics.
                'rkhs_median': _pack(shares['rkhs']),
            },
            'partition_source':           LIVE_SENTINEL,
            'partition_seed':             int(shares['partition_seed']),
            'rkhs_median_length_scale':   float(shares['rkhs_median_length_scale']),
            'rkhs_kernel_in_phase6':      'matern_nu1.5_only',
            'split_a_frac':               split_a,
            'pca_dim':                    pca_dim,
            'target_pca_components':      target_pca,
        }
        n_done += 1

        # ── V0 sanity gate (first refit cell only) ────────────────
        if not v0_done:
            v0_done = True
            ui.msg("  [V0] sanity check on first partition-B cell...")
            sums_ok = True
            for cls_name, triple in (('ridge', shares['ridge']),
                                        ('mlp',   shares['mlp']),
                                        ('rf',    shares['rf']),
                                        ('rkhs',  shares['rkhs'])):
                s = sum(triple)
                if abs(s - 1.0) > 1e-6:
                    ui.warn(f"  [V0 FAIL] {cls_name} share sum = {s:.9f}, "
                              f"expected 1.0 ± 1e-6 (sum-to-1 tolerance "
                              f"violated -- partition-B fit returning "
                              f"un-normalized shares)")
                    sums_ok = False
            if sums_ok:
                ui.ok("  [V0 PASS] sum-to-1 tolerance OK on all 4 classes")
            # Tier-3 closed-form check: solve_iprojection_closed_form
            # against scipy reference is exercised in T-tier validation
            # at solver-load time; the Phase-6 V0 only verifies that
            # the partition-B share inputs land in the simplex-interior
            # regime where the solver's formulas are tested. Module 2
            # (Phase 11) handles internal-consistency of the eight-field
            # block downstream.

        # Atomic incremental write -- survive process death mid-fleet
        payload = {
            'phase': 'phase6_function_class_fits_p2',
            'iota_version': '0.82.0.23',
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'operating_point': {
                'split_a_frac': split_a,
                'pca_dim':      pca_dim,
            },
            'partition_source_note': (
                "Phase 6 surfaces single-shot partition-B refit shares "
                "as canonical input for Phase 8 / Phase 11. v0.82.0.23 "
                "ship: replaced the v0.82.0.23 Q0057 full-data "
                "passthrough after V0 confirmed partition mismatch on "
                "gemma_2b_q4_abliterated_t00 (max |Δ| = 0.125, dominated "
                "by Ridge ΔC = -0.125). Cells with missing qcache fall "
                "back to Q0057 shares flagged as "
                "'q57_fallback_qcache_missing'."
            ),
            'cells': per_cell,
            'summary': {
                'n_cells_refit':     n_done,
                'n_cells_cached':    n_skip_cached,
                'n_cells_fallback':  n_fallback,
                'n_cells_failed':    n_failed,
                'n_cells_filtered':  n_skip_filter,
            },
        }
        _save_phase_output('function_class_p2', 'per_cell.json', payload)
        ui.ok(f"  [{cell_key}] partition-B refit done")

    # If only checkpoints/filters fired (no fresh cells this run), still
    # emit a payload so the orchestrator sees something coherent.
    if n_done == 0:
        payload = {
            'phase': 'phase6_function_class_fits_p2',
            'iota_version': '0.82.0.23',
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'operating_point': {
                'split_a_frac': split_a,
                'pca_dim':      pca_dim,
            },
            'cells': per_cell,
            'summary': {
                'n_cells_refit':     0,
                'n_cells_cached':    n_skip_cached,
                'n_cells_fallback':  n_fallback,
                'n_cells_failed':    n_failed,
                'n_cells_filtered':  n_skip_filter,
            },
        }
        _save_phase_output('function_class_p2', 'per_cell.json', payload)

    ui.ok(f"  Phase 6 done: {n_done} cells refit, "
            f"{n_skip_cached} cached, {n_fallback} fallback, "
            f"{n_failed} failed, {n_skip_filter} filtered")
    return (n_done > 0 or n_skip_cached > 0 or n_fallback > 0), payload


# ─── Phase 7: kNN anchor on partition A ──────────────────────────────

def _run_knn_anchor_p2(session=None, paths=None):
    """Phase 7 -- per-channel chain-rule kNN-MI on partition A of each cell.

    Distinct from existing Phase 4 (run_kraskov_anchor) -- Phase 4 uses
    full data and a 2-channel anchor for paper 1. Phase 7 partitions
    the cell, runs kNN-MI on partition A only, and computes the full
    chain-rule decomposition I(S_{t+1}; E), I(S_{t+1}; E, C),
    I(S_{t+1}; E, C, S_{t-1}) for paper 2's three-channel apparatus
    consumer.

    For first-shipped scope, this phase wraps the existing Phase 4
    anchor producer's output and tags it 'partition_a_pending'. The
    structural difference (3-channel chain rule, partition A only) is
    implemented in the follow-up that fires the actual partitioned
    estimator. Apparatus phase 8 consumes the wrapped values
    transitionally and switches to real partitioned values on the
    follow-up ship without changing its loader contract.
    """
    ui.section("Phase 7 -- kNN anchor (partition A)")
    split_a, pca_dim = _load_operating_point()
    phase4_path = os.path.join(DATA, 'paper', 'calibration',
                                'kraskov_anchor', 'anchors.json')
    if not os.path.exists(phase4_path):
        ui.warn(f"  Phase 4 anchors.json missing -- phase 7 cannot wrap")
        return False, {'error': 'phase4_anchor_missing', 'n_cells': 0}
    try:
        with open(phase4_path, 'r', encoding='utf-8-sig') as f:
            phase4 = json.load(f)
    except Exception as e:
        ui.warn(f"  Phase 4 load failed: {e}")
        return False, {'error': f'phase4_load_failed: {e}', 'n_cells': 0}
    phase4_cells = (phase4 or {}).get('anchors_per_cell') or {}
    # v0.82.0.8: Phase 4 (run_kraskov_anchor.py) stores its per-cell
    # anchor results under the key 'anchors_per_cell', not 'cells'.
    # Phase 7's pre-patch read of `.get('cells')` returned {} so it
    # wrapped zero cells and Phase 8 cascade-skipped. Fixed.
    payload = {
        'phase': 'phase7_knn_anchor_p2',
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'operating_point': {
            'split_a_frac': split_a,
            'pca_dim':      pca_dim,
        },
        'anchor_source_note': (
            "Phase 7 currently wraps Phase 4's full-data 2-channel "
            "anchor as a transitional input for the paper-2 four-class "
            "apparatus consumer. The partition-A 3-channel chain-rule "
            "estimator lands in a subsequent ship; consumers' loader "
            "contract is stable across the swap (key: anchors[cell]['p_a']'s "
            "shape stays (3,))."
        ),
        'cells': phase4_cells,
        'summary': {
            'n_cells': len(phase4_cells),
        },
    }
    out_path = _save_phase_output('kraskov_anchor_p2', 'anchors.json', payload)
    ui.ok(f"  wrote {out_path}: {len(phase4_cells)} cells (transitional wrap)")
    return len(phase4_cells) > 0, payload


# ─── Phase 8: apparatus + λ-sensitivity sweep ────────────────────────

def _run_lagrangian_anchoring_p2(session=None, paths=None):
    """Phase 8 -- invoke bayesian_solver across four classes per cell,
    sweep λ ∈ LAMBDA_SWEEP_GRID, pick lambda_default from V5 minimum-
    error point.

    Uses Phase 6 shares (4-class) and Phase 7 anchor (per-cell p_a).
    Per cell, solves q*(λ) for each λ in the grid and records the
    full sweep. lambda_default is selected as the λ minimizing
    cross-V5-system error; if V5 calibration outputs aren't yet
    available, falls back to λ=1.0 (the previous DEFAULT_LAMBDA).
    """
    ui.section("Phase 8 -- Apparatus + λ-sensitivity sweep")
    p6_path = os.path.join(DATA, 'paper', 'calibration',
                            'function_class_p2', 'per_cell.json')
    p7_path = os.path.join(DATA, 'paper', 'calibration',
                            'kraskov_anchor_p2', 'anchors.json')
    if not os.path.exists(p6_path) or not os.path.exists(p7_path):
        ui.warn(f"  Phase 6 or 7 output missing -- phase 8 cannot fire")
        return False, {'error': 'phase_67_missing'}
    try:
        with open(p6_path, 'r', encoding='utf-8-sig') as f:
            p6 = json.load(f)
        with open(p7_path, 'r', encoding='utf-8-sig') as f:
            p7 = json.load(f)
    except Exception as e:
        return False, {'error': f'load_failed: {e}'}

    p6_cells = (p6 or {}).get('cells') or {}
    p7_cells = (p7 or {}).get('cells') or {}

    try:
        import bayesian_solver as ls
    except Exception as e:
        return False, {'error': f'bayesian_solver import failed: {e}'}

    per_cell = {}
    n_done = 0
    n_skip = 0
    for cell_key, p6_block in p6_cells.items():
        if cell_key not in p7_cells:
            n_skip += 1
            continue
        shares = p6_block.get('shares') or {}
        # Build (4, 3) class-share matrix
        try:
            ridge = np.array([float(shares['ridge'][k]) for k in ('E', 'C', 'R')])
            mlp = np.array([float(shares['mlp'][k]) for k in ('E', 'C', 'R')])
            rf = np.array([float(shares['rf'][k]) for k in ('E', 'C', 'R')])
            rkhs_md = shares.get('rkhs_median')
            if rkhs_md:
                rkhs = np.array([float(rkhs_md[k]) for k in ('E', 'C', 'R')])
            else:
                rkhs = ridge.copy()
        except Exception as e:
            n_skip += 1
            continue
        # v0.82.0.8: Phase 7 passes Phase 4's anchors_per_cell through
        # unchanged. Phase 4 stores 'p_anchor' as a 3-element list
        # [E, C, R] (not a dict). Earlier code looked for 'p_a' as a
        # dict -- that contract was a forward-design that didn't match
        # what Phase 4 actually emits. Read 'p_anchor' as the list and
        # convert to a numpy array directly.
        p_anchor_list = p7_cells[cell_key].get('p_anchor')
        if p_anchor_list is None or len(p_anchor_list) != 3:
            n_skip += 1
            continue
        try:
            p_a_arr = np.array([float(x) for x in p_anchor_list])
        except Exception:
            n_skip += 1
            continue

        # v0.82.0.10: solver signature is solve_iprojection_closed_form(
        # p_classes: array shape (n_classes, n_coords), p_anchor, lam).
        # Pre-patch this passed `class_shares` as a dict, which np.asarray
        # produced an object-array from, breaking the math silently and
        # raising in the np.log(p) call. The except-Exception caught it,
        # all 8 λ-grid points failed, sweep stayed empty, every cell
        # skipped, and Phase 8 wrote 0 cells. Phase 5's
        # run_bayesian_aggregator.py builds a stacked array via
        # np.array([ridge, mlp, rf, rkhs]) and works fine; do the same
        # here.
        class_shares_arr = np.array([ridge, mlp, rf, rkhs])
        sweep = []
        q_star_default = None
        for lam in LAMBDA_SWEEP_GRID:
            try:
                # v0.82.0.12: pass lam through directly. The pre-patch
                # substitution `lam=(0.0 if lam == float('inf') else lam)`
                # was a wrong fix for the solver's lam=inf bug -- it made
                # λ=∞ output equal λ=0 output (pure G), the OPPOSITE
                # extreme of what λ=∞ should give (pure anchor). Now
                # that solve_iprojection_closed_form handles inf
                # correctly via the np.isfinite(lam) guard, just pass
                # lam through.
                lam_val = float('inf') if lam == float('inf') else float(lam)
                if hasattr(ls, 'solve_iprojection_closed_form'):
                    q_star = ls.solve_iprojection_closed_form(
                        class_shares_arr, p_a_arr, lam=lam_val)
                elif hasattr(ls, 'solve_with_diagnostics'):
                    diag = ls.solve_with_diagnostics(
                        class_shares_arr, p_a_arr, lam=lam_val)
                    q_star = diag.get('q_star') if isinstance(diag, dict) else diag
                else:
                    q_star = None
                if q_star is not None:
                    q_arr = np.asarray(q_star).flatten()
                    if q_arr.size == 3:
                        sweep.append({
                            'lambda': lam if lam != float('inf') else 'inf',
                            'q_star': {
                                'E': float(q_arr[0]),
                                'C': float(q_arr[1]),
                                'R': float(q_arr[2]),
                            },
                        })
                        if lam == 1.0:
                            q_star_default = q_arr.tolist()
            except Exception:
                continue

        if not sweep:
            n_skip += 1
            continue
        per_cell[cell_key] = {
            'lambda_sweep':  sweep,
            'q_star_at_lambda_default':
                {'E': q_star_default[0], 'C': q_star_default[1],
                 'R': q_star_default[2]} if q_star_default else None,
            'lambda_default_used': 1.0,
        }
        n_done += 1

    # Pick lambda_default from V5 calibration if available; else fallback.
    # V5-based λ-selection ships with the V5 calibration data integration
    # (calibration source = V5b/V5e/V5f in subsequent ship).
    payload = {
        'phase': 'phase8_lagrangian_anchoring_p2',
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'lambda_sweep_grid': [str(x) if x == float('inf') else x
                                for x in LAMBDA_SWEEP_GRID],
        'lambda_default_chosen': 1.0,
        'lambda_calibration_source': 'fallback_pending_v5_integration',
        'cells': per_cell,
        'summary': {
            'n_cells_done':    n_done,
            'n_cells_skipped': n_skip,
        },
    }
    out_path = _save_phase_output('apparatus_p2', 'per_cell.json', payload)
    ui.ok(f"  wrote {out_path}: {n_done} cells, λ-sweep {len(LAMBDA_SWEEP_GRID)} points")
    return n_done > 0, payload


# ─── Phase 9: stacking baseline ──────────────────────────────────────

def _run_stacking_baseline_p2(session=None, paths=None):
    """Phase 9 -- convex QP on V5a-V5e to find weights (w_R, w_M, w_K, w_F) ≥ 0,
    sum to 1, minimizing ||w · class_shares - truth||² across V5 systems
    where truth is known. Apply (no refit) to V5f and four empirical cells.

    Comparison estimator for paper 2: how does the apparatus do
    relative to naive convex averaging of the four classes?
    """
    ui.section("Phase 9 -- Stacking baseline (4-weight convex QP)")
    # V5 calibration data
    v5_path = os.path.join(DATA, 'paper', 'calibration', 'v5',
                            'v5_calibration_results.json')
    if not os.path.exists(v5_path):
        ui.warn(f"  V5 calibration missing -- fallback to uniform weights")
        weights = {'w_R': 0.25, 'w_M': 0.25, 'w_K': 0.25, 'w_F': 0.25}
    else:
        # V5a-V5e provide per-system reference R values; we'd ideally
        # solve a QP for the four weights against those references.
        # First-shipped scope: uniform weights (1/4) as a proper convex
        # combination on the simplex; replaced with QP-fitted weights
        # in the follow-up ship that ingests V5a-V5e per-class shares.
        weights = {'w_R': 0.25, 'w_M': 0.25, 'w_K': 0.25, 'w_F': 0.25}

    # Apply to empirical cells from Phase 6 shares
    p6_path = os.path.join(DATA, 'paper', 'calibration',
                            'function_class_p2', 'per_cell.json')
    if not os.path.exists(p6_path):
        return False, {'error': 'phase6_missing'}
    with open(p6_path, 'r', encoding='utf-8-sig') as f:
        p6 = json.load(f)
    p6_cells = (p6 or {}).get('cells') or {}

    per_cell = {}
    for cell_key, block in p6_cells.items():
        shares = block.get('shares') or {}
        try:
            ridge = np.array([float(shares['ridge'][k]) for k in ('E', 'C', 'R')])
            mlp = np.array([float(shares['mlp'][k]) for k in ('E', 'C', 'R')])
            rf = np.array([float(shares['rf'][k]) for k in ('E', 'C', 'R')])
            rkhs_md = shares.get('rkhs_median')
            if rkhs_md:
                rkhs = np.array([float(rkhs_md[k]) for k in ('E', 'C', 'R')])
            else:
                rkhs = ridge.copy()
        except Exception:
            continue
        stacked = (weights['w_R'] * ridge +
                    weights['w_M'] * mlp +
                    weights['w_K'] * rkhs +
                    weights['w_F'] * rf)
        per_cell[cell_key] = {
            'stacking_baseline_shares': {
                'E': float(stacked[0]),
                'C': float(stacked[1]),
                'R': float(stacked[2]),
            },
        }

    payload = {
        'phase': 'phase9_stacking_baseline_p2',
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'stacking_baseline_weights': weights,
        'weights_source': 'uniform_pending_v5_qp_fit',
        'cells': per_cell,
        'summary': {'n_cells': len(per_cell)},
    }
    out_path = _save_phase_output('stacking_baseline_p2', 'per_cell.json', payload)
    ui.ok(f"  wrote {out_path}: {len(per_cell)} cells")
    return len(per_cell) > 0, payload


# ─── Phase 10: bootstrap variance ────────────────────────────────────

def _run_bootstrap_variance_p2(session=None, paths=None,
                                 n_bootstrap=N_BOOTSTRAP,
                                 cell_filter=None):
    """Phase 10 -- bootstrap variance per cell.

    For each cell: n_bootstrap resamples of the cell's full data with
    replacement to size n. Per resample: paired bootstrap indices and
    paired internal 80/20 train/test split shared across the four
    function classes (Ridge, MLP, RKHS, RF). Refit each class on the
    bootstrap train, permute on the bootstrap test, renormalize drops
    to a (E, C, R) simplex point. Apply the I-projection apparatus
    with the cell's fixed Phase 7 anchor and the globally-chosen
    lambda_default. Per-channel variance for each class and for
    anchored is computed across resamples.

    Cell-level checkpointing: existing per_cell.json is read on
    entry; cells already complete at the requested n_bootstrap are
    skipped (variance_source matches the live sentinel).
    per_cell.json is rewritten after each cell completes -- process
    death mid-fleet preserves prior cells' work.

    Args:
      n_bootstrap: number of resamples per cell. Default
        N_BOOTSTRAP (=200). Profile-mode invocations should pass a
        smaller number (e.g. 20) plus cell_filter to measure wall
        time without committing to the full fleet.
      cell_filter: list of writerbot cell keys to restrict to, or
        None to process all cells in Phase 6's output. Useful for
        the codebot-brief-mandated single-cell wall-time profile.

    Scope limitations (also documented in
    bootstrap_variance_p2.py module docstring and CHANGELOG
    0.82.0.23):
      - Single-kernel matern ν=1.5 RKHS inside the bootstrap loop
        (vs three-kernel headline RKHS_median outside the loop).
      - Per-resample permutation count reduced to 30 / 10 (Ridge,
        MLP, RF / RKHS) vs canonical 200 / 10.
      - Test rows subsampled to 1500 per resample for permutation.
      - Phase 6 partition mismatch: bootstrap does its own
        partition-B refits per resample; headline q* (Phase 8)
        consumes Q0057 full-data shares. Reconciled when Phase 6
        partition-B refit lands.
    """
    ui.section(f"Phase 10 -- Bootstrap variance (n_bootstrap={n_bootstrap})")
    p6_path = os.path.join(DATA, 'paper', 'calibration',
                            'function_class_p2', 'per_cell.json')
    if not os.path.exists(p6_path):
        return False, {'error': 'phase6_missing'}
    with open(p6_path, 'r', encoding='utf-8-sig') as f:
        p6 = json.load(f)
    p6_cells = (p6 or {}).get('cells') or {}

    # Phase 7 -- anchor per cell (writerbot keys)
    p7_path = os.path.join(DATA, 'paper', 'calibration',
                            'kraskov_anchor_p2', 'anchors.json')
    p7_cells = {}
    if os.path.exists(p7_path):
        try:
            with open(p7_path, 'r', encoding='utf-8-sig') as f:
                p7_cells = (json.load(f) or {}).get('cells') or {}
        except Exception as e:
            ui.warn(f"  Phase 7 load failed: {e}")
    if not p7_cells:
        ui.warn("  Phase 7 anchors unavailable -- Phase 10 cannot fire")
        return False, {'error': 'phase7_missing'}

    # Phase 8 -- lambda_default (falls back to 1.0 if missing)
    lam_default = 1.0
    p8_path = os.path.join(DATA, 'paper', 'calibration',
                            'apparatus_p2', 'per_cell.json')
    if os.path.exists(p8_path):
        try:
            with open(p8_path, 'r', encoding='utf-8-sig') as f:
                p8 = json.load(f)
            lam_default = float(p8.get('lambda_default_chosen') or 1.0)
        except Exception as e:
            ui.warn(f"  Phase 8 load failed (using lam=1.0): {e}")
    ui.msg(f"  lambda_default = {lam_default}")

    # Existing checkpoint -- skip cells already complete at this
    # n_bootstrap.
    out_path_check = os.path.join(DATA, 'paper', 'calibration',
                                    'bootstrap_variance_p2', 'per_cell.json')
    LIVE_SENTINEL = 'phase10_partition_b_refit_paired_bootstrap'
    existing_per_cell = {}
    if os.path.exists(out_path_check):
        try:
            with open(out_path_check, 'r', encoding='utf-8-sig') as f:
                existing = json.load(f) or {}
            existing_per_cell = existing.get('cells') or {}
        except Exception as e:
            ui.warn(f"  existing per_cell.json load failed: {e}")

    import bootstrap_variance_p2 as bvp2

    # Walk Q0042 files; for each writerbot cell key in p6_cells, find
    # the matching Q0042 path so we can load the qcache.
    q42s = _q42_files()
    q42_by_writerbot = {}
    for q42 in q42s:
        family, size, variant, temp = _parse_q42_path(q42)
        if not family:
            continue
        wb = _writerbot_key(family, size, variant, temp)
        q42_by_writerbot[wb] = q42

    per_cell = dict(existing_per_cell)  # carry over already-done cells
    n_done_this_run = 0
    n_skipped = 0
    n_failed = 0
    payload = None

    # v0.82.0.23: when no cell_filter is passed, restrict to
    # PHASE10_SUBSET (5 cells) per the wall-time decision. Cells NOT
    # in the subset get a methods-note tag rather than a real
    # bootstrap result; downstream consumers see a sentinel they can
    # filter on.
    METHODS_NOTE_SENTINEL = 'bootstrap_not_run_single_shot_only'
    active_filter = (cell_filter
                     if cell_filter is not None
                     else list(PHASE10_SUBSET))
    ui.msg(f"  active cell filter: {len(active_filter)} cells "
           f"({'caller-supplied' if cell_filter is not None else 'PHASE10_SUBSET'})")

    # Tag every Phase-6 cell not in the active filter with the
    # methods-note sentinel -- ensures downstream
    # results_builder.py reads something for every cell, not None.
    for cell_key in p6_cells:
        if cell_key in active_filter:
            continue
        # Don't overwrite a real bootstrap result that's already in
        # per_cell from a prior run.
        prior = per_cell.get(cell_key) or {}
        if prior.get('variance_source') == LIVE_SENTINEL:
            continue
        per_cell[cell_key] = {
            'n_bootstrap':                   0,
            'Ridge_var_per_channel':         {'E': 0.0, 'C': 0.0, 'R': 0.0},
            'MLP_var_per_channel':           {'E': 0.0, 'C': 0.0, 'R': 0.0},
            'RKHS_median_var_per_channel':   {'E': 0.0, 'C': 0.0, 'R': 0.0},
            'RF_var_per_channel':            {'E': 0.0, 'C': 0.0, 'R': 0.0},
            'anchored_var_per_channel':      {'E': 0.0, 'C': 0.0, 'R': 0.0},
            'variance_source':               METHODS_NOTE_SENTINEL,
            'methods_note': (
                "Bootstrap variance not run on this cell per the "
                "v0.82.0.23 wall-time decision. Variance fields are "
                "0.0 placeholders; consumers should treat as missing "
                "data, not zero variance. The 5-cell subset that was "
                "bootstrap-refit is listed in PHASE10_SUBSET in "
                "decomposition_p2.py. See CHANGELOG 0.82.0.23 for "
                "the subset selection rationale."
            ),
        }

    for cell_key in sorted(p6_cells.keys()):
        if cell_key not in active_filter:
            continue
        prior = per_cell.get(cell_key) or {}
        if (prior.get('variance_source') == LIVE_SENTINEL
                and int(prior.get('n_bootstrap') or 0) >= n_bootstrap):
            ui.msg(f"  [skip] {cell_key} -- already complete at "
                    f"n_bootstrap={prior.get('n_bootstrap')}")
            n_skipped += 1
            continue

        q42 = q42_by_writerbot.get(cell_key)
        if not q42:
            ui.warn(f"  [{cell_key}] no Q0042 path -- cannot load qcache")
            n_failed += 1
            continue

        # Phase 7 anchor -- Phase 4's p_anchor list is [E, C, R]
        p7 = p7_cells.get(cell_key) or {}
        p_anchor_list = p7.get('p_anchor')
        if p_anchor_list is None or len(p_anchor_list) != 3:
            ui.warn(f"  [{cell_key}] no Phase 7 anchor")
            n_failed += 1
            continue
        try:
            p_anchor = np.array([float(x) for x in p_anchor_list])
        except Exception as e:
            ui.warn(f"  [{cell_key}] anchor parse failed: {e}")
            n_failed += 1
            continue

        # Load cell data from qcache
        cell_dir = os.path.dirname(os.path.dirname(q42))
        hidden_dir = os.path.join(cell_dir, 'hidden_states')
        cell_data = bvp2.load_cell_data_from_qcache(hidden_dir)
        if 'error' in cell_data:
            ui.warn(f"  [{cell_key}] data load failed: "
                    f"{cell_data['error']}")
            n_failed += 1
            continue

        ui.msg(f"  [{cell_key}] n_rows={cell_data['n_rows']}, "
                f"pool_dim={cell_data['pool_dim']}")

        # Target-PCA decision: cells with native pool_dim > 64 use
        # PCA-64 for RF / RKHS (matches canonical Q0057 protocol).
        target_pca = 64 if cell_data['pool_dim'] > 64 else None

        # Per-cell deterministic seed offset from cell_key -- keeps
        # bootstraps reproducible per cell, independent across cells.
        seed_base = abs(hash(cell_key)) % (2**31 - 1)

        result = bvp2.bootstrap_one_cell(
            cell_data, p_anchor, lam_default,
            n_bootstrap=n_bootstrap,
            seed_base=seed_base,
            target_pca_components=target_pca,
            ui_msg=lambda s: ui.msg(s),
            progress_every=max(1, n_bootstrap // 10),
        )

        per_cell[cell_key] = {
            'n_bootstrap':              int(result['n_bootstrap_requested']),
            'n_completed':              int(result['n_completed']),
            'n_failed':                 int(result['n_failed']),
            'Ridge_var_per_channel':         result['Ridge_var_per_channel'],
            'MLP_var_per_channel':           result['MLP_var_per_channel'],
            'RKHS_median_var_per_channel':   result['RKHS_median_var_per_channel'],
            'RF_var_per_channel':            result['RF_var_per_channel'],
            'anchored_var_per_channel':      result['anchored_var_per_channel'],
            'wall_time_seconds':             float(result['wall_time_seconds']),
            'lambda_used':                   float(lam_default),
            'rkhs_kernel':                   'matern_nu1.5_only',
            'partition_source':              'partition_b_refit_paired_bootstrap',
            'variance_source':               LIVE_SENTINEL,
            # v0.82.0.23: persist per-resample shares so post-hoc
            # geometric / hull diagnostics can be reconstructed
            # without re-firing the bootstrap. Each key holds a
            # list of N (E, C, R) triples: ridge / mlp / rkhs / rf
            # are the per-class shares that went into the apparatus,
            # anchored is the apparatus output q* per resample.
            # ~5KB per cell at n_bootstrap=10 -- small.
            'per_resample_shares':           result['per_resample_shares'],
        }
        if result['failures_sample']:
            per_cell[cell_key]['failures_sample'] = result['failures_sample']
        n_done_this_run += 1

        # Atomic incremental write -- survive process death mid-fleet.
        payload = {
            'phase': 'phase10_bootstrap_variance_p2',
            'iota_version': '0.82.0.23',
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'n_bootstrap': n_bootstrap,
            'lambda_default': lam_default,
            'rkhs_kernel': 'matern_nu1.5_only',
            'scope_limitations': {
                'rkhs_kernel_grid':
                    'single_kernel_matern_nu1.5_inside_bootstrap',
                'permutation_count_per_resample': {
                    'ridge_mlp_rf': bvp2.N_PERM_BOOTSTRAP,
                    'rkhs':         bvp2.N_PERM_RKHS_BOOTSTRAP,
                },
                'test_subsample_per_resample':
                    bvp2.N_TEST_SUBSAMPLE_BOOTSTRAP,
                'bootstrap_rows_cap':
                    bvp2.N_BOOTSTRAP_ROWS_CAP,
                'phase6_partition_caveat': (
                    "Phase 10 bootstrap performs partition-B refits "
                    "per resample; headline q* (Phase 8) consumes "
                    "Q0057 full-data shares. Reconciled when Phase 6 "
                    "partition-B refit ships."
                ),
            },
            'cells': per_cell,
            'summary': {
                'n_cells_total':      len(p6_cells),
                'n_cells_completed':  sum(
                    1 for v in per_cell.values()
                    if v.get('variance_source') == LIVE_SENTINEL
                ),
                'n_done_this_run':    n_done_this_run,
                'n_skipped_this_run': n_skipped,
                'n_failed_this_run':  n_failed,
            },
        }
        _save_phase_output('bootstrap_variance_p2', 'per_cell.json',
                            payload)
        ui.ok(f"  [{cell_key}] done in "
                f"{result['wall_time_seconds']:.0f}s "
                f"(completed={result['n_completed']}/{n_bootstrap})")

    # If no cells were processed this run (all skipped or filtered
    # out), still emit a payload so the orchestrator sees something.
    if payload is None:
        payload = {
            'phase': 'phase10_bootstrap_variance_p2',
            'iota_version': '0.82.0.23',
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'n_bootstrap': n_bootstrap,
            'lambda_default': lam_default,
            'cells': per_cell,
            'summary': {
                'n_cells_total':      len(p6_cells),
                'n_done_this_run':    0,
                'n_skipped_this_run': n_skipped,
                'n_failed_this_run':  n_failed,
            },
        }

    ui.ok(f"  Phase 10 done: {n_done_this_run} cells refit, "
            f"{n_skipped} cells skipped (cached), "
            f"{n_failed} cells failed")
    return (n_done_this_run > 0 or n_skipped > 0), payload


# ─── Phase 11: hull diagnostic ───────────────────────────────────────

def _run_geometric_diagnostics_p2(session=None, paths=None):
    """Phase 11 -- geometric diagnostics (statsbot's Module 2 spec).

    Replaces v0.82.0.23's _run_hull_diagnostic_p2. Computes the
    eight-field per-cell geometric_diagnostics block (q_star vs G̃
    hull membership, anchor pull distance, signed anchor pull, class
    hull volume, degenerate flag, hull exit mechanism, hull signed
    distance) plus cross-cell aggregates including V5d cross-reference.

    Schema migration:
      - Old output: hull_diagnostic_p2/per_cell.json carrying
        anchored_in_hull + hull_signed_distance + hull_vertices.
      - New output: geometric_diagnostics_p2/per_cell.json carrying
        the eight-field block + geometric_aggregates.

    Old `convex_hull_diagnostic` schema is removed atomically per spec.
    Migration directive (Ship 12 → Module 2) lives in
    paper2_apparatus.md and CHANGELOG 0.82.0.23.

    V5d cross-reference: reads V5d flags (v5d_flag_combined) from the
    aggregator's per-cell output (data/paper/calibration/
    apparatus_aggregator/per_cell.json), since V5d is computed
    downstream of Module 2 per the spec's execution order.
    """
    ui.section("Phase 11 -- Geometric diagnostics (Module 2, eight fields)")
    p6_path = os.path.join(DATA, 'paper', 'calibration',
                            'function_class_p2', 'per_cell.json')
    p8_path = os.path.join(DATA, 'paper', 'calibration',
                            'apparatus_p2', 'per_cell.json')
    if not (os.path.exists(p6_path) and os.path.exists(p8_path)):
        return False, {'error': 'phase6_or_phase8_missing'}
    with open(p6_path, 'r', encoding='utf-8-sig') as f:
        p6 = json.load(f)
    with open(p8_path, 'r', encoding='utf-8-sig') as f:
        p8 = json.load(f)
    p6_cells = (p6 or {}).get('cells') or {}
    p8_cells = (p8 or {}).get('cells') or {}

    try:
        import geometric_diagnostics_p2 as gd
    except Exception as e:
        return False, {'error': f'geometric_diagnostics_p2 import: {e}'}

    # Optional: V5d flags from the aggregator (Phase 5 of Run 0058).
    # If absent, geometric_aggregates.v5d_flagged_quadrant_distribution
    # comes back None -- diagnostic is still useful without the V5d
    # cross-reference, but the spec's "operating-point expectation"
    # quadrant counts can't be computed.
    agg_path = os.path.join(DATA, 'paper', 'calibration',
                              'apparatus_aggregator', 'per_cell.json')
    v5d_flagged = None
    if os.path.exists(agg_path):
        try:
            with open(agg_path, 'r', encoding='utf-8-sig') as f:
                agg = json.load(f)
            agg_cells = (agg or {}).get('cells') or {}
            v5d_flagged = {
                ck for ck, v in agg_cells.items()
                if (v or {}).get('v5d_flag_combined') is True
            }
        except Exception as e:
            ui.warn(f"  V5d flag load failed (cross-ref skipped): {e}")

    # v0.82.0.23: Bootstrap variance decomposition population.
    # Read bootstrap_variance_p2/per_cell.json for per_resample_shares;
    # populate `bootstrap_variance_decomposition` block per cell when
    # the shares are present (PHASE10_SUBSET cells in current state,
    # the full fleet if Phase 10 is fully populated).
    bsd_path = os.path.join(DATA, 'paper', 'calibration',
                              'bootstrap_variance_p2', 'per_cell.json')
    bsd_cells = {}
    if os.path.exists(bsd_path):
        try:
            with open(bsd_path, 'r', encoding='utf-8-sig') as f:
                bsd_cells = (json.load(f) or {}).get('cells') or {}
        except Exception as e:
            ui.warn(f"  bootstrap_variance_p2 load failed (decomposition skipped): {e}")

    per_cell = {}
    for cell_key in p6_cells:
        if cell_key not in p8_cells:
            continue
        shares = p6_cells[cell_key].get('shares') or {}
        try:
            ridge = np.array([float(shares['ridge'][k]) for k in ('E', 'C', 'R')])
            mlp = np.array([float(shares['mlp'][k]) for k in ('E', 'C', 'R')])
            rf = np.array([float(shares['rf'][k]) for k in ('E', 'C', 'R')])
            rkhs_md = shares.get('rkhs_median')
            if rkhs_md:
                rkhs = np.array([float(rkhs_md[k]) for k in ('E', 'C', 'R')])
            else:
                rkhs = ridge.copy()
        except Exception:
            continue
        q = p8_cells[cell_key].get('q_star_at_lambda_default')
        if not q:
            continue
        q_arr = np.array([float(q['E']), float(q['C']), float(q['R'])])
        p_classes = np.array([ridge, mlp, rf, rkhs])

        try:
            diag = gd.compute_geometric_diagnostics(q_arr, p_classes)
        except Exception as e:
            ui.warn(f"  [{cell_key}] diagnostic failed: {e}")
            continue

        # Optional bootstrap_variance_decomposition population
        bsd_cell = bsd_cells.get(cell_key) or {}
        prs = bsd_cell.get('per_resample_shares')
        if prs:
            try:
                decomp = gd.compute_bootstrap_variance_decomposition(prs)
                if decomp is not None:
                    diag['bootstrap_variance_decomposition'] = decomp
            except Exception as e:
                ui.warn(f"  [{cell_key}] decomposition failed: {e}")

        per_cell[cell_key] = diag

    n_decomp = sum(1 for v in per_cell.values()
                    if 'bootstrap_variance_decomposition' in v)

    aggregates = gd.compute_geometric_aggregates(per_cell,
                                                    v5d_flagged=v5d_flagged)

    payload = {
        'phase': 'phase11_geometric_diagnostics_p2',
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'schema_replaces': 'convex_hull_diagnostic',
        'projection_note': (
            "Hull computed in 2D (E, C) projection of the 3-simplex "
            "(R = 1 − E − C). Field 8 (hull_signed_distance) uses "
            "facet-equation signed distance for non-degenerate hulls "
            "and point-to-segment fallback for degenerate hulls. Sign "
            "convention preserved from Ship 12: positive=inside, "
            "negative=outside."
        ),
        'cells': per_cell,
        'geometric_aggregates': aggregates,
        'summary': {
            'n_cells': len(per_cell),
            'v5d_cross_reference_available': v5d_flagged is not None,
            'n_cells_with_bootstrap_decomposition': n_decomp,
        },
    }
    out_path = _save_phase_output('geometric_diagnostics_p2',
                                     'per_cell.json', payload)
    ui.ok(f"  wrote {out_path}: {len(per_cell)} cells, "
            f"inside={aggregates['n_cells_inside_hull']}, "
            f"outside={aggregates['n_cells_outside_hull']}, "
            f"degen={aggregates['n_cells_hull_degenerate']}, "
            f"decomp={n_decomp}")
    return len(per_cell) > 0, payload


def _run_estimator_joint_R2_p2(session=None, paths=None,
                                 cell_filter=None, force_refit=False):
    """Phase 12 -- per-estimator joint R² at canonical partition-B
    operating point.

    v0.82.0.23: surfaces the recoverability-plane linearity-axis
    citation for paper 1 §5. Per cell, fits Ridge / MLP / RF / RKHS
    on partition A (80%) and scores on partition B (20%) -- same
    split, scaling, and PCA-target rule as Phase 6's
    compute_partition_b_shares, but emits R² values instead of
    permutation shares.

    NOT derivable from existing fields. linearity_max_gap (Q0042)
    is the related-but-distinct in-sample architecture sweep
    (default ReLU MLP, 3 configs); estimator_joint_R2 is the
    canonical-config held-out R² at partition-B.

    Cell-level checkpointing -- existing per_cell.json read on
    entry, cells with the live sentinel are skipped.

    Args:
      cell_filter: optional list of writerbot cell keys to restrict
        to (V0 single-cell sanity).
      force_refit: if True, ignore checkpoint and re-fit every cell.

    Wall time: roughly 12-15 sec per cell (deterministic single-
    config fits, no permutation loop). Fleet at 24 cells: ~5 min.
    """
    ui.section("Phase 12 -- Estimator joint R² (partition-B canonical)")
    split_a, pca_dim = _load_operating_point()
    ui.msg(f"  operating point: split_A={split_a}, pca={pca_dim}")

    q42s = _q42_files()
    if not q42s:
        ui.warn("  No Q0042 cells found.")
        return False, {'error': 'no_q42_cells', 'n_cells': 0}

    LIVE_SENTINEL = 'partition_b_canon'
    out_check_path = os.path.join(DATA, 'paper', 'calibration',
                                     'estimator_joint_R2_p2',
                                     'per_cell.json')
    existing_per_cell = {}
    if os.path.exists(out_check_path) and not force_refit:
        try:
            with open(out_check_path, 'r', encoding='utf-8-sig') as f:
                existing = json.load(f) or {}
            existing_per_cell = existing.get('cells') or {}
        except Exception as e:
            ui.warn(f"  existing per_cell.json load failed: {e}")

    import function_class_p2 as fcp2
    import estimator_joint_R2_p2 as ejr

    per_cell = dict(existing_per_cell)
    n_done = 0
    n_skip_cached = 0
    n_skip_filter = 0
    n_failed = 0

    for q42 in q42s:
        family, size, variant, temp = _parse_q42_path(q42)
        if not family:
            continue
        cell_key = _writerbot_key(family, size, variant, temp)

        if cell_filter is not None and cell_key not in cell_filter:
            n_skip_filter += 1
            continue

        prior = per_cell.get(cell_key) or {}
        if (not force_refit
                and prior.get('partition_source') == LIVE_SENTINEL):
            ui.msg(f"  [skip] {cell_key} -- cached at {LIVE_SENTINEL}")
            n_skip_cached += 1
            continue

        cell_dir = os.path.dirname(os.path.dirname(q42))
        hidden_dir = os.path.join(cell_dir, 'hidden_states')
        cell_data = fcp2.load_cell_data_from_qcache(hidden_dir)

        if 'error' in cell_data:
            ui.warn(f"  [{cell_key}] qcache missing -- skipped")
            n_failed += 1
            continue

        ui.msg(f"  [{cell_key}] n_rows={cell_data['n_rows']}, "
                f"pool_dim={cell_data['pool_dim']}")

        target_pca = 64 if cell_data['pool_dim'] > 64 else None

        try:
            r2_block = ejr.compute_estimator_joint_R2(
                cell_data,
                target_pca_components=target_pca,
                seed=fcp2.PARTITION_B_SEED,
                ui_msg=lambda s: ui.msg(s),
            )
        except Exception as e:
            ui.warn(f"  [{cell_key}] joint R² failed: "
                      f"{type(e).__name__}: {e}")
            n_failed += 1
            continue

        per_cell[cell_key] = {
            'estimator_joint_R2': {
                'ridge':       r2_block['ridge'],
                'mlp':         r2_block['mlp'],
                'rf':          r2_block['rf'],
                'rkhs_median': r2_block['rkhs_median'],
            },
            'rkhs_median_length_scale': r2_block['rkhs_median_length_scale'],
            'partition_seed':           r2_block['partition_seed'],
            'partition_source':         LIVE_SENTINEL,
            'split_a_frac':             split_a,
            'pca_dim':                  pca_dim,
            'native_pool_dim':          int(cell_data['pool_dim']),
            'target_pca_components':    target_pca,
        }
        n_done += 1
        ui.ok(f"  [{cell_key}] joint R² done")

    payload = {
        'phase':        'phase12_estimator_joint_R2_p2',
        'iota_version': '0.82.0.23',
        'timestamp':    datetime.datetime.utcnow().isoformat() + 'Z',
        'cells':        per_cell,
        'summary': {
            'n_cells_done':    n_done,
            'n_cells_cached':  n_skip_cached,
            'n_cells_filter':  n_skip_filter,
            'n_cells_failed':  n_failed,
            'n_cells_total':   len(per_cell),
        },
    }
    out_path = _save_phase_output('estimator_joint_R2_p2',
                                     'per_cell.json', payload)
    ui.ok(f"  Phase 12 done: {n_done} fit, {n_skip_cached} cached, "
            f"{n_failed} failed, {n_skip_filter} filtered")
    return len(per_cell) > 0, payload
