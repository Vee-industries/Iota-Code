"""
run_bayesian_apparatus.py — Run 0058 orchestrator.

v0.81.1.0: phases 1-5 implemented (paper-1 Bayesian apparatus).
v0.81.1.7: phases 6-11 added (paper-2 measurement layer).

Phase chain:
  Phase 1: kraskov bias spike + linearization sweep
           emits: data/paper/calibration/kraskov_spike/spike_result.json
           locks: bucket assignment

  Phase 2: solver module 3-tier validation harness
           emits: data/paper/calibration/lagrangian_solver/validation.json

  Phase 3: V5d threshold τ calibration
           emits: data/paper/calibration/v5d_threshold/calibration.json

  Phase 4: Kraskov anchor producer over real cells (paper-1 anchor)
           emits: data/paper/calibration/kraskov_anchor/anchors.json

  Phase 5: aggregator over real cells (paper-1 apparatus output)
           emits: data/paper/calibration/apparatus_aggregator/per_cell.json

  ── Paper 2 measurement layer ──
  Phase 6: function-class fits with partition split (4 classes)
           emits: data/paper/calibration/function_class_p2/per_cell.json

  Phase 7: kNN anchor on partition A (3-channel chain rule)
           emits: data/paper/calibration/kraskov_anchor_p2/anchors.json

  Phase 8: apparatus + λ-sensitivity sweep
           emits: data/paper/calibration/apparatus_p2/per_cell.json

  Phase 9: stacking baseline (4-weight convex QP)
           emits: data/paper/calibration/stacking_baseline_p2/per_cell.json

  Phase 10: bootstrap variance — paired-bootstrap refit loop on a
            6-cell representative subset (PHASE10_SUBSET in
            decomposition_p2.py); n_bootstrap=10 by default (12 in the released run, via IOTA_N_BOOTSTRAP); all four
            classes (Ridge / MLP / RKHS / RF) refit per resample.
            Other 18 cells get bootstrap_not_run_single_shot_only
            sentinel + methods_note. Per-resample shares persisted
            for post-hoc geometric-on-bootstrap analysis.
            emits: data/paper/calibration/bootstrap_variance_p2/per_cell.json

  Phase 11: geometric diagnostics (Module 2 — eight fields per cell:
            q* vs G̃ hull membership, anchor pull KL nats, class hull
            volume, hull-exit mechanism, Euclidean signed distance to
            hull boundary, plus cross-cell aggregates with V5d
            cross-reference for fig7).
            emits: data/paper/calibration/geometric_diagnostics_p2/per_cell.json

  Phase 12: write Q0058_apparatus_manifest.json summarizing all phases.

Each phase hard-gates on the previous one having produced its output.
On any phase failure, manifest still emits with phase status set to
'failed', so downstream consumers see partial state.

Phases 6-11 are NON-BLOCKING for the paper-1 apparatus claim — if
phases 1-5 succeed but 6+ fail, manifest reports paper-1 'done' and
paper-2 'failed' independently.
"""
import json
import os
import sys
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import ui  # noqa: E402

DATA = os.path.join(ROOT, 'data')
PAPER_ROOT = os.path.join(DATA, 'paper')
MANIFEST_PATH = os.path.join(PAPER_ROOT, 'Q0058_apparatus_manifest.json')


def _phase1():
    ui.section("Phase 1 — Kraskov spike + linearization sweep")
    try:
        import run_kraskov_spike
        rc = run_kraskov_spike.main()
    except SystemExit as _se:
        rc = _se.code or 0
    if rc != 0:
        return False, None
    spike_path = os.path.join(DATA, 'paper', 'calibration',
                               'kraskov_spike', 'spike_result.json')
    if not os.path.exists(spike_path):
        return False, None
    with open(spike_path, 'r', encoding='utf-8') as f:
        return True, json.load(f)


def _phase2():
    ui.section("Phase 2 — Solver 3-tier validation harness")
    out_dir = os.path.join(DATA, 'paper', 'calibration', 'lagrangian_solver')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'validation.json')
    try:
        import bayesian_solver as ls
        result = ls.run_validation_harness()
        result['phase'] = 'phase2_solver_module'
        result['iota_version'] = '0.82.0.23'
        result['timestamp'] = datetime.datetime.utcnow().isoformat() + 'Z'
        # v0.83.2: atomic write for per-cell bayesian result
        from results_schema import atomic_json_dump
        atomic_json_dump(result, out_path, indent=2, ensure_ascii=False)
        if result['all_tiers_pass']:
            ui.ok(f"  All 3 tiers + mirror-descent cross-check passed.")
            ui.ok(f"  Wrote {out_path}")
            return True, result
        else:
            ui.err(f"  Validation failed: see {out_path}")
            return False, result
    except Exception as e:
        ui.err(f"  Phase 2 exception: {type(e).__name__}: {e}")
        return False, None


def _phase3():
    ui.section("Phase 3 — V5d threshold calibration")
    try:
        import run_v5d_threshold_calibration
        rc = run_v5d_threshold_calibration.main()
    except SystemExit as _se:
        rc = _se.code or 0
    if rc != 0:
        return False, None
    p = os.path.join(DATA, 'paper', 'calibration', 'v5d_threshold',
                     'calibration.json')
    if not os.path.exists(p):
        return False, None
    with open(p, 'r', encoding='utf-8') as f:
        return True, json.load(f)


def _phase4():
    ui.section("Phase 4 — Kraskov anchor producer over real cells")
    try:
        import run_kraskov_anchor
        rc = run_kraskov_anchor.main()
    except SystemExit as _se:
        rc = _se.code or 0
    if rc != 0:
        return False, None
    p = os.path.join(DATA, 'paper', 'calibration', 'kraskov_anchor',
                     'anchors.json')
    if not os.path.exists(p):
        return False, None
    with open(p, 'r', encoding='utf-8') as f:
        return True, json.load(f)


def _phase5():
    ui.section("Phase 5 — Aggregator over real cells")
    try:
        import run_bayesian_aggregator
        rc = run_bayesian_aggregator.main()
    except SystemExit as _se:
        rc = _se.code or 0
    if rc != 0:
        return False, None
    p = os.path.join(DATA, 'paper', 'calibration', 'apparatus_aggregator',
                     'per_cell.json')
    if not os.path.exists(p):
        return False, None
    with open(p, 'r', encoding='utf-8') as f:
        return True, json.load(f)


def _emit_manifest(phase_states, phase1_data, phase5_data, paper2_data=None):
    ui.section("Phase 12 — Manifest emission")
    bucket = beta = None
    if phase1_data:
        bucket = phase1_data.get('part_a_bias_measurement', {}).get('bucket')
        beta = phase1_data.get('part_a_bias_measurement', {}).get('beta_overall')
    n_cells = n_v5d = None
    if phase5_data:
        n_cells = phase5_data.get('n_cells_evaluated')
        n_v5d = phase5_data.get('n_v5d_flagged')
    manifest = {
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'apparatus_phase_status': phase_states,
        'phase1_summary': {
            'bucket': bucket,
            'beta_overall': beta,
            'spike_result_path': 'data/paper/calibration/kraskov_spike/spike_result.json',
        },
        'phase5_summary': {
            'n_cells_evaluated': n_cells,
            'n_v5d_flagged': n_v5d,
            'per_cell_path': 'data/paper/calibration/apparatus_aggregator/per_cell.json',
        },
    }
    # v0.81.1.7: paper-2 phase summaries
    if paper2_data:
        for phase_n, key, path_suffix in [
            (6, 'phase6', 'function_class_p2/per_cell.json'),
            (7, 'phase7', 'kraskov_anchor_p2/anchors.json'),
            (8, 'phase8', 'apparatus_p2/per_cell.json'),
            (9, 'phase9', 'stacking_baseline_p2/per_cell.json'),
            (10, 'phase10', 'bootstrap_variance_p2/per_cell.json'),
            (11, 'phase11', 'geometric_diagnostics_p2/per_cell.json'),
        ]:
            data = paper2_data.get(key) or {}
            manifest[f'{key}_summary'] = {
                'n_cells': (data.get('summary') or {}).get('n_cells')
                            if isinstance(data.get('summary'), dict) else None,
                'output_path': f'data/paper/calibration/{path_suffix}',
            }
    # v0.83.2: atomic write for bayesian manifest (load-bearing state)
    from results_schema import atomic_json_dump
    atomic_json_dump(manifest, MANIFEST_PATH, indent=2, ensure_ascii=False)
    ui.ok(f"Wrote {MANIFEST_PATH}")


def main():
    ui.section("Run 0058 — Lagrangian Apparatus (full pipeline)")

    phase_states = {
        'phase1_kraskov_spike':       'pending',
        'phase2_solver_module':       'pending',
        'phase3_v5d_threshold':       'pending',
        'phase4_anchor_producer':     'pending',
        'phase5_aggregator':          'pending',
        # v0.81.1.7 paper-2 layer
        'phase6_function_class_p2':   'pending',
        'phase7_knn_anchor_p2':       'pending',
        'phase8_apparatus_p2':        'pending',
        'phase9_stacking_baseline_p2': 'pending',
        'phase10_bootstrap_variance_p2': 'pending',
        'phase11_geometric_diagnostics_p2': 'pending',
    }

    p1_ok, p1_data = _phase1()
    phase_states['phase1_kraskov_spike'] = 'done' if p1_ok else 'failed'
    if not p1_ok:
        _emit_manifest(phase_states, p1_data, None)
        sys.exit(1)

    p2_ok, _ = _phase2()
    phase_states['phase2_solver_module'] = 'done' if p2_ok else 'failed'
    if not p2_ok:
        _emit_manifest(phase_states, p1_data, None)
        sys.exit(1)

    p3_ok, _ = _phase3()
    phase_states['phase3_v5d_threshold'] = 'done' if p3_ok else 'failed'
    if not p3_ok:
        _emit_manifest(phase_states, p1_data, None)
        sys.exit(1)

    p4_ok, _ = _phase4()
    phase_states['phase4_anchor_producer'] = 'done' if p4_ok else 'failed'
    if not p4_ok:
        _emit_manifest(phase_states, p1_data, None)
        sys.exit(1)

    p5_ok, p5_data = _phase5()
    phase_states['phase5_aggregator'] = 'done' if p5_ok else 'failed'

    # ── Paper-2 layer (phases 6-11). Non-blocking on paper-1 success: ──
    # if phases 1-5 succeeded the paper-1 apparatus claim is complete
    # regardless of paper-2 outcome. Phases 6-11 chain off Phase 5
    # outputs (transitionally) and Phase 6/7 outputs (mutually).
    paper2_data = {}
    if p5_ok:
        try:
            import decomposition_p2 as dp2

            p6_ok, p6_data = dp2._run_function_class_fits_p2()
            phase_states['phase6_function_class_p2'] = 'done' if p6_ok else 'failed'
            paper2_data['phase6'] = p6_data

            p7_ok, p7_data = dp2._run_knn_anchor_p2()
            phase_states['phase7_knn_anchor_p2'] = 'done' if p7_ok else 'failed'
            paper2_data['phase7'] = p7_data

            if p6_ok and p7_ok:
                p8_ok, p8_data = dp2._run_lagrangian_anchoring_p2()
                phase_states['phase8_apparatus_p2'] = 'done' if p8_ok else 'failed'
                paper2_data['phase8'] = p8_data
            else:
                phase_states['phase8_apparatus_p2'] = 'skipped_upstream'

            if p6_ok:
                p9_ok, p9_data = dp2._run_stacking_baseline_p2()
                phase_states['phase9_stacking_baseline_p2'] = 'done' if p9_ok else 'failed'
                paper2_data['phase9'] = p9_data

                p10_ok, p10_data = dp2._run_bootstrap_variance_p2()
                phase_states['phase10_bootstrap_variance_p2'] = 'done' if p10_ok else 'failed'
                paper2_data['phase10'] = p10_data
            else:
                phase_states['phase9_stacking_baseline_p2'] = 'skipped_upstream'
                phase_states['phase10_bootstrap_variance_p2'] = 'skipped_upstream'

            if phase_states.get('phase8_apparatus_p2') == 'done':
                p11_ok, p11_data = dp2._run_geometric_diagnostics_p2()
                phase_states['phase11_geometric_diagnostics_p2'] = 'done' if p11_ok else 'failed'
                paper2_data['phase11'] = p11_data
            else:
                phase_states['phase11_geometric_diagnostics_p2'] = 'skipped_upstream'

        except Exception as _p2_e:
            ui.err(f"Paper-2 phases failed: {type(_p2_e).__name__}: {_p2_e}")
            import traceback as _tb
            _tb.print_exc()
            for k in list(phase_states.keys()):
                if k.startswith('phase') and 'p2' in k and phase_states[k] == 'pending':
                    phase_states[k] = 'failed'
    else:
        for k in list(phase_states.keys()):
            if 'p2' in k:
                phase_states[k] = 'skipped_upstream'

    _emit_manifest(phase_states, p1_data, p5_data, paper2_data=paper2_data)

    if not p5_ok:
        sys.exit(1)

    ui.ok("Run 0058 complete. Apparatus operational.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
