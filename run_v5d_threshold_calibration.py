"""
run_v5d_threshold_calibration.py -- Phase 3 of the Bayesian apparatus.

Calibrates two thresholds against V5d-style synthetic ground truth:
  τ_spread on `spread_js` (max pairwise JS over four classes)
  τ_amgm   on `|am_gm_log_ratio_C|` (AM-GM divergence on C coord)

Calibration target: τ such that V5d-true cells flag at >90% rate
and V5b-low-synergy cells flag at <10% rate. ROC-style threshold pick.

Output: data/paper/calibration/v5d_threshold/calibration.json
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
OUT_DIR = os.path.join(DATA, 'paper', 'calibration', 'v5d_threshold')
OUT_PATH = os.path.join(OUT_DIR, 'calibration.json')


def _generate_synthetic_4class(synergy, n_cells, seed=0, eps=1e-6):
    """Generate 4-class share matrices for V5b-low-synergy or V5d-high-synergy
    regimes. Synergy ∈ [0, 1]. Low synergy → classes agree (low spread, low am-gm).
    High synergy → classes disagree on C in the AM-GM signature pattern.
    """
    rng = np.random.RandomState(seed)
    cells = []
    for i in range(n_cells):
        # Base shares
        base = rng.dirichlet(np.array([3.0, 1.5, 2.5]))
        # Per-class perturbations grow with synergy
        # V5d signature: classes diverge on C coord specifically
        classes = []
        for c_idx in range(4):
            noise = rng.normal(0, 0.05 * (1 + synergy), size=3)
            # C coordinate gets extra divergence at high synergy
            noise[1] += (c_idx - 1.5) * 0.1 * synergy
            p = base + noise
            p = np.clip(p, eps, 1 - eps)
            p = p / p.sum()
            classes.append(p)
        cells.append(np.array(classes))
    return cells


def _measure_thresholds_for_regime(cells, regime_label):
    """For each cell, compute spread_js and |am_gm_log_ratio_C|.
    Returns sorted arrays for ROC analysis.
    """
    spread = []
    amgm = []
    for p_classes in cells:
        spread.append(ls.pairwise_max_js(p_classes))
        amgm.append(abs(ls.am_gm_log_ratio_C(p_classes)))
    return np.array(spread), np.array(amgm)


def _roc_threshold(positives, negatives, target_pos_rate=0.90, target_neg_rate=0.10):
    """Pick threshold τ such that:
      - fraction of positives >= τ is at least target_pos_rate
      - fraction of negatives >= τ is at most target_neg_rate
    Sweep candidates and pick best ROC point. Returns (tau, pos_rate, neg_rate).
    """
    candidates = np.linspace(min(positives.min(), negatives.min()),
                             max(positives.max(), negatives.max()), 200)
    best = None
    for tau in candidates:
        pos_rate = float((positives >= tau).mean())
        neg_rate = float((negatives >= tau).mean())
        # Score: maximize (pos_rate - neg_rate) subject to pos_rate >= target
        if pos_rate >= target_pos_rate and neg_rate <= target_neg_rate:
            score = pos_rate - neg_rate
            if best is None or score > best[3]:
                best = (float(tau), pos_rate, neg_rate, score)
    if best is None:
        # Relax: pick threshold that maximizes Youden's J = pos_rate - neg_rate
        for tau in candidates:
            pos_rate = float((positives >= tau).mean())
            neg_rate = float((negatives >= tau).mean())
            score = pos_rate - neg_rate
            if best is None or score > best[3]:
                best = (float(tau), pos_rate, neg_rate, score)
    tau, pos_rate, neg_rate, _ = best
    return tau, pos_rate, neg_rate


def main():
    t0 = time.time()
    ui.section("Run 0058 Phase 3 -- V5d threshold calibration")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Generate ground-truth ensembles
    n_cells = 200
    ui.msg(f"Generating {n_cells} V5b-low-synergy cells (negatives)...")
    cells_neg = _generate_synthetic_4class(synergy=0.05, n_cells=n_cells, seed=42)
    ui.msg(f"Generating {n_cells} V5d-true (high-synergy) cells (positives)...")
    cells_pos = _generate_synthetic_4class(synergy=0.7, n_cells=n_cells, seed=43)

    spread_neg, amgm_neg = _measure_thresholds_for_regime(cells_neg, 'low_synergy')
    spread_pos, amgm_pos = _measure_thresholds_for_regime(cells_pos, 'v5d_true')

    ui.msg(f"  low_synergy: spread mean={spread_neg.mean():.4f}, amgm mean={amgm_neg.mean():.4f}")
    ui.msg(f"  v5d_true:    spread mean={spread_pos.mean():.4f}, amgm mean={amgm_pos.mean():.4f}")

    # ROC threshold picks
    tau_spread, ps_pos, ps_neg = _roc_threshold(spread_pos, spread_neg)
    tau_amgm, pa_pos, pa_neg = _roc_threshold(amgm_pos, amgm_neg)

    ui.ok(f"τ_spread = {tau_spread:.4f}  (positives flag {ps_pos*100:.1f}%, negatives flag {ps_neg*100:.1f}%)")
    ui.ok(f"τ_amgm   = {tau_amgm:.4f}  (positives flag {pa_pos*100:.1f}%, negatives flag {pa_neg*100:.1f}%)")

    out = {
        'phase': 'phase3_v5d_threshold',
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'tau_spread': float(tau_spread),
        'tau_amgm': float(tau_amgm),
        'v5d_true_spread_flag_rate': float(ps_pos),
        'v5b_low_synergy_spread_flag_rate': float(ps_neg),
        'v5d_true_amgm_flag_rate': float(pa_pos),
        'v5b_low_synergy_amgm_flag_rate': float(pa_neg),
        'n_cells_per_regime': n_cells,
        'spread_means': {'low_synergy': float(spread_neg.mean()),
                         'v5d_true': float(spread_pos.mean())},
        'amgm_means': {'low_synergy': float(amgm_neg.mean()),
                       'v5d_true': float(amgm_pos.mean())},
        'calibration_target': {
            'positive_flag_rate_target': 0.90,
            'negative_flag_rate_target': 0.10,
        },
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - t0
    ui.ok(f"Phase 3 complete in {elapsed:.1f}s")
    ui.ok(f"Wrote {OUT_PATH}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
