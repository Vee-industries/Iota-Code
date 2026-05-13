"""
run_bayesian_aggregator.py — Phase 5 of the Bayesian apparatus.

For each real cell:
  1. Load four-class shares (Ridge, MLP, RF, RKHS placeholder) from Q0057.
  2. Load Kraskov anchor from Phase 4 output.
  3. Apply λ_anchor (calibrated value, default = 1.0 for first ship).
  4. Solve I-projection via bayesian_solver.solve_with_diagnostics.
  5. Apply V5d threshold flags from Phase 3 output.
  6. Emit per-cell q*, anchor_bounds (or rank for ordinal), V5d flags.

Output: data/paper/calibration/apparatus_aggregator/per_cell.json
"""
import json
import os
import sys
import time
import datetime

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import ui  # noqa: E402
import bayesian_solver as ls  # noqa: E402

DATA = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(DATA, 'paper', 'calibration', 'apparatus_aggregator')
OUT_PATH = os.path.join(OUT_DIR, 'per_cell.json')

# Default λ_anchor for first ship. Calibration sweep is a follow-up
# (extension of v5_synthetic_calibration.py); for now use λ=1.0 which
# is the balanced anchor-weighting case.
DEFAULT_LAMBDA = 1.0


def _load_q57_shares():
    p = os.path.join(DATA, 'paper', 'Q0057_function_class_sensitivity.json')
    if not os.path.exists(p):
        return {}
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            q = json.load(f)
    except Exception:
        return {}
    shares = {}
    for cell_key, cell in (q.get('cells') or {}).items():
        perm = cell.get('permutation_shares', {})
        ridge = perm.get('ridge')
        mlp = perm.get('mlp')
        rf = perm.get('rf')
        rkhs_median = perm.get('rkhs_median')
        if not all((ridge, mlp, rf)):
            continue
        def _to_arr(d):
            return [float(d['E']), float(d['C']), float(d['R'])]
        # v0.81.1.4: real RKHS if present; Ridge fallback otherwise
        # (transitional state until full Q0057 re-fit completes)
        rkhs_arr = _to_arr(rkhs_median) if rkhs_median else _to_arr(ridge)
        shares[cell_key] = {
            'ridge': _to_arr(ridge),
            'mlp':   _to_arr(mlp),
            'rf':    _to_arr(rf),
            'rkhs':  rkhs_arr,
        }
    return shares


def _load_anchors():
    p = os.path.join(DATA, 'paper', 'calibration', 'kraskov_anchor', 'anchors.json')
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        return None


def _load_thresholds():
    p = os.path.join(DATA, 'paper', 'calibration', 'v5d_threshold', 'calibration.json')
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            d = json.load(f)
        return float(d.get('tau_spread', 0.5)), float(d.get('tau_amgm', 0.5))
    except Exception:
        return None, None


def _load_spike_bucket():
    p = os.path.join(DATA, 'paper', 'calibration', 'kraskov_spike', 'spike_result.json')
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            d = json.load(f)
        bucket = d.get('part_a_bias_measurement', {}).get('bucket')
        beta = d.get('part_a_bias_measurement', {}).get('beta_overall')
        return bucket, float(beta) if beta is not None else None
    except Exception:
        return None, None


def main():
    t_total = time.time()
    ui.section("Run 0058 Phase 5 — Aggregator over real cells")
    os.makedirs(OUT_DIR, exist_ok=True)

    bucket, beta = _load_spike_bucket()
    if bucket is None:
        ui.err("No bucket from Phase 1 spike — cannot proceed.")
        return 1
    ui.msg(f"Bucket: {bucket} (β={beta:.4f})")

    shares = _load_q57_shares()
    if not shares:
        ui.err("No Q0057 shares — Phase 5 cannot proceed.")
        return 1
    ui.msg(f"Loaded shares for {len(shares)} cells from Q0057.")

    anchors_doc = _load_anchors()
    if not anchors_doc or not anchors_doc.get('anchors_per_cell'):
        ui.err("No anchors from Phase 4 — Phase 5 cannot proceed.")
        return 1
    anchors = anchors_doc['anchors_per_cell']
    ui.msg(f"Loaded {len(anchors)} anchors from Phase 4.")

    tau_spread, tau_amgm = _load_thresholds()
    if tau_spread is None:
        ui.warn("No V5d thresholds from Phase 3 — using defaults τ_spread=0.05, τ_amgm=0.10")
        tau_spread, tau_amgm = 0.05, 0.10
    ui.msg(f"Thresholds: τ_spread={tau_spread:.4f}, τ_amgm={tau_amgm:.4f}")

    lam = DEFAULT_LAMBDA
    ui.msg(f"λ_anchor = {lam} (default; calibration sweep follow-up)")

    cells_out = {}
    n_done = 0
    n_skipped = 0
    n_v5d_flagged = 0

    for cell_key in sorted(shares.keys()):
        if cell_key not in anchors:
            ui.warn(f"  {cell_key}: no anchor — skipped")
            n_skipped += 1
            continue
        cell_shares = shares[cell_key]
        p_classes = np.array([cell_shares['ridge'], cell_shares['mlp'],
                              cell_shares['rf'], cell_shares['rkhs']])
        p_anchor = np.array(anchors[cell_key]['p_anchor'])

        try:
            result = ls.solve_with_diagnostics(p_classes, p_anchor, lam,
                                                beta=beta, bucket=bucket)
            spread = result['spread_js']
            amgm = result['solver_diagnostics']['am_gm_log_ratio_C']
            v5d_flag_spread = bool(spread > tau_spread)
            v5d_flag_amgm = bool(abs(amgm) > tau_amgm)
            v5d_flag_combined = v5d_flag_spread or v5d_flag_amgm
            if v5d_flag_combined:
                n_v5d_flagged += 1
            result['v5d_flag_spread'] = v5d_flag_spread
            result['v5d_flag_amgm'] = v5d_flag_amgm
            result['v5d_flag_combined'] = v5d_flag_combined
            result['tau_spread_used'] = float(tau_spread)
            result['tau_amgm_used'] = float(tau_amgm)
            cells_out[cell_key] = result
            n_done += 1
            q = result['q_star']
            flag_str = " [V5d]" if v5d_flag_combined else ""
            ui.msg(f"  {cell_key}: q*=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}]  "
                   f"spread={spread:.4f}{flag_str}")
        except Exception as e:
            import traceback as _tb
            ui.err(f"  {cell_key}: FAIL: {type(e).__name__}: {e}")
            for line in _tb.format_exc().splitlines()[-5:]:
                ui.err(f"    {line}")
            n_skipped += 1

    out = {
        'phase': 'phase5_aggregator',
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'bucket': bucket,
        'beta_overall': beta,
        'lambda_used': lam,
        'tau_spread': tau_spread,
        'tau_amgm': tau_amgm,
        'n_cells_evaluated': n_done,
        'n_cells_skipped': n_skipped,
        'n_v5d_flagged': n_v5d_flagged,
        'cells': cells_out,
    }
    # v0.83.2: atomic write for aggregator output
    from results_schema import atomic_json_dump
    atomic_json_dump(out, OUT_PATH, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_total
    ui.ok(f"Phase 5 complete. {n_done} aggregated, {n_skipped} skipped, "
          f"{n_v5d_flagged} V5d-flagged. Elapsed: {elapsed:.0f}s")
    ui.ok(f"Wrote {OUT_PATH}")
    if n_done == 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
