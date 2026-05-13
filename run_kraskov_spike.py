"""
run_kraskov_spike.py — Phase 1 of the Bayesian apparatus.

Two diagnostics in one ~15 minute run:

  PART A — Kraskov bias measurement.
    Runs Kraskov-MI estimation (sklearn's mutual_info_regression,
    KSG-equivalent at default parameters) on V5b synthetic data
    where ground-truth (E, S) shares are computable analytically
    from the V5b parameterization. Compares estimated MI shares
    against ground-truth shares. Returns a single bias scalar β
    representing mean L1 distance, plus per-config breakdown.

    β decides the bucket:
      strong  (β <= 0.10): anchor contributes quantitative shares
      narrow  (0.10 < β <= 0.20): anchor contributes weighted-quantitative
                                   with empirical bounds on share accuracy
      ordinal (β > 0.20):  anchor contributes ordinal information only

  PART B — Linearization sweep.
    For each (β, λ) combination on a 6×3 grid, solves the
    I-projection optimum twice — once with biased anchor, once
    with unbiased — and compares the closed-form linearized bound
    against the true L1 distance. Confirms or refines the bucket
    boundary as either β-only or (β, λ)-jointly defined.

Output: data/paper/calibration/kraskov_spike/spike_result.json
        data/paper/calibration/kraskov_spike/.cache_key.json

Prereqs:
  - V5b synthetic data is regenerable on demand (V5b is fast — no
    cached state needed; we run it inline).

Cache key: hash of this script + hash of v5_synthetic_calibration.py.
Stamps cache_key.json on success so the calibration freshness
mechanism in export_stats.py knows the spike is current.

Cost:
  - Part A: ~30s/config × 4 configs = ~2 min
  - Part B: 6 β × 3 λ × 2 solves = 36 closed-form solver calls,
            each <100ms. Total ~5s.
  - Output JSON: ~50KB.

Roughly a 3-minute run end-to-end on a modern machine.
"""
import json
import os
import sys
import time
import datetime
import hashlib

import numpy as np

# ─────────────────────────────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import ui  # noqa: E402

DATA = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(DATA, 'paper', 'calibration', 'kraskov_spike')
OUT_PATH = os.path.join(OUT_DIR, 'spike_result.json')
CACHE_KEY_PATH = os.path.join(OUT_DIR, '.cache_key.json')

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
#  Cache key
# ─────────────────────────────────────────────────────────────────────

def _file_hash(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _build_cache_key():
    return {
        'script_hash': _file_hash(os.path.join(ROOT, 'run_kraskov_spike.py')),
        'upstream': {
            'v5_synthetic_calibration.py':
                _file_hash(os.path.join(ROOT, 'v5_synthetic_calibration.py')),
        },
    }


# ─────────────────────────────────────────────────────────────────────
#  Part A — V5b bias measurement
# ─────────────────────────────────────────────────────────────────────

def _v5b_ground_truth_shares(S_traj, E_traj, A_matrix, B_matrix):
    """Empirically-grounded ground-truth (E, S) shares from V5b
    generated data.

    v0.81.0.1: replaced the closed-form `b²/(a²+b²)` formula. That
    formula assumed A's spectral norm and B's Frobenius norm were
    dimensionally comparable; they aren't, and the formula
    disagreed with actual V5b variance shares by up to 24
    percentage points on the balanced config.

    The chain-rule decomposition's reference in the linearizable
    regime is the variance contribution of each input to the
    pre-tanh linear combination:
        AS_t = A @ S[t]   (variance contribution from prior state)
        BE_t = B @ E[t]   (variance contribution from input embedding)
    The shares Var(AS) / (Var(AS) + Var(BE)) and the symmetric for E
    are what an unbiased estimator should recover under linearization.
    Computing them from the actual generated data is the most honest
    reference — it captures whatever V5b's specific parameterization
    actually produces, including the spectral-vs-Frobenius scaling
    that the closed-form formula missed.

    Inputs:
      S_traj:   shape (n_steps+1, dim) — generated state trajectory
      E_traj:   shape (n_steps,   dim) — input embedding trajectory
      A_matrix: shape (dim, dim)       — V5b's recurrent matrix
      B_matrix: shape (dim, dim)       — V5b's input matrix
    """
    AS = S_traj[:-1] @ A_matrix.T
    BE = E_traj      @ B_matrix.T
    var_AS = float(np.var(AS))
    var_BE = float(np.var(BE))
    total = var_AS + var_BE
    if total <= 0:
        return 0.5, 0.5
    return float(var_BE / total), float(var_AS / total)


def _generate_v5b_data(a_scale, b_scale, dim=8, n_steps=20000, seed=42):
    """Reproduce V5b's synthetic generator with the same parameters used
    by run_v5b() in v5_synthetic_calibration.py. Independent reimplementation
    so the spike doesn't depend on V5b's internal output format.

    v0.81.0.1: now also returns A and B matrices so the caller can
    compute empirical variance-share ground truth. That replaces the
    closed-form formula which was off by up to 24 percentage points
    on the balanced config.
    """
    rng = np.random.RandomState(seed)
    A_raw = rng.randn(dim, dim)
    A_raw = A_raw / np.max(np.abs(np.linalg.eigvals(A_raw)))
    A = a_scale * A_raw
    B_raw = rng.randn(dim, dim)
    B = b_scale * B_raw / np.linalg.norm(B_raw)
    S = np.zeros((n_steps + 1, dim))
    E = rng.randn(n_steps, dim)
    S[0] = rng.randn(dim) * 0.1
    for t in range(n_steps):
        S[t + 1] = np.tanh(A @ S[t] + B @ E[t]) + rng.randn(dim) * 0.1
    return S, E, A, B


def _kraskov_mi_share(X, Y, n_neighbors=5):
    """Mean Kraskov-MI estimate across output dims, then sum.
    Uses sklearn's KSG-equivalent estimator. PCA-32 reduction on Y
    if Y has more than 32 columns (matches the project's existing
    knn_pca_dim=32 convention).
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.decomposition import PCA
    if Y.shape[1] > 32:
        Y = PCA(n_components=32, random_state=42).fit_transform(Y)
    if X.shape[1] > 32:
        X = PCA(n_components=32, random_state=42).fit_transform(X)
    # mutual_info_regression returns I(X[:,i]; y) for each input column
    # given a 1-D y. For multi-D y we sum across output dims (assumes
    # output dims are approximately independent under PCA — reasonable
    # for the V5b regime).
    mi_per_input = np.zeros(X.shape[1])
    for j in range(Y.shape[1]):
        mi_j = mutual_info_regression(
            X, Y[:, j], n_neighbors=n_neighbors, random_state=42)
        mi_per_input += mi_j
    # Total MI across all (X dim, Y dim) pairs, summed
    return float(mi_per_input.sum())


def _measure_bias_one_config(a_scale, b_scale, config_label):
    """Run Kraskov-MI on one V5b config and return the bias diagnostic.

    v0.81.0.1: ground truth now computed from the actual V5b generator
    matrices and trajectories, not the closed-form formula that was
    off by up to 24 pp.
    """
    ui.msg(f"  config={config_label!r}: a={a_scale}, b={b_scale}")
    t0 = time.time()
    S, E, A, B = _generate_v5b_data(a_scale, b_scale)
    Y = S[1:]
    X_S = S[:-1]
    X_E = E
    mi_E = _kraskov_mi_share(X_E, Y)
    mi_S = _kraskov_mi_share(X_S, Y)
    total = mi_E + mi_S
    if total <= 0:
        ui.warn(f"    MI total non-positive ({total}); config skipped")
        return None
    est_share_E = mi_E / total
    est_share_S = mi_S / total
    gt_share_E, gt_share_S = _v5b_ground_truth_shares(S, E, A, B)
    bias_E = abs(est_share_E - gt_share_E)
    bias_S = abs(est_share_S - gt_share_S)
    bias = (bias_E + bias_S) / 2.0
    dt = time.time() - t0
    ui.msg(f"    estimated: E={est_share_E:.3f} S={est_share_S:.3f}  "
           f"truth: E={gt_share_E:.3f} S={gt_share_S:.3f}  "
           f"bias={bias:.4f}  ({dt:.1f}s)")
    return {
        'config_label': config_label,
        'a_scale': a_scale,
        'b_scale': b_scale,
        'estimated_share_E': est_share_E,
        'estimated_share_S': est_share_S,
        'ground_truth_share_E': gt_share_E,
        'ground_truth_share_S': gt_share_S,
        'ground_truth_method': 'empirical_variance_contribution_v0_81_0_1',
        'bias_E': float(bias_E),
        'bias_S': float(bias_S),
        'bias_mean': float(bias),
        'mi_E_raw': float(mi_E),
        'mi_S_raw': float(mi_S),
        'wall_clock_s': dt,
    }


def part_a_bias_measurement():
    """Part A: measure Kraskov bias on V5b configs."""
    ui.section("Part A — Kraskov bias measurement on V5b")
    configs = [
        ('E-dominant', 0.2, 0.8),
        ('balanced',   0.5, 0.5),
        ('S-dominant', 0.8, 0.2),
        ('strong-S',   0.95, 0.1),
    ]
    per_config = []
    for label, a, b in configs:
        result = _measure_bias_one_config(a, b, label)
        if result is not None:
            per_config.append(result)
    if not per_config:
        ui.err("No V5b configs produced usable bias measurements.")
        sys.exit(1)
    bias_overall = float(np.mean([c['bias_mean'] for c in per_config]))
    ui.ok(f"Overall mean bias (β) = {bias_overall:.4f}")
    return per_config, bias_overall


# ─────────────────────────────────────────────────────────────────────
#  Bucket assignment
# ─────────────────────────────────────────────────────────────────────

def _assign_bucket(beta):
    if beta <= 0.10:
        return 'strong'
    if beta <= 0.20:
        return 'narrow'
    return 'ordinal'


# ─────────────────────────────────────────────────────────────────────
#  Part B — Linearization sweep
# ─────────────────────────────────────────────────────────────────────

def _solve_iprojection_closed_form(p_classes, p_anchor, lam, eps=1e-6):
    """Closed-form I-projection optimum on the simplex.

    q*_i ∝ G_i^(1/(1+λ)) * p_a,i^(λ/(1+λ))
        G_i = (∏_f p_f,i)^(1/4)

    p_classes: shape (n_classes, 3) — class-by-coordinate probabilities
    p_anchor:  shape (3,) — anchor probabilities
    """
    p = np.clip(p_classes, eps, 1 - eps)
    a = np.clip(p_anchor, eps, 1 - eps)
    G = np.exp(np.mean(np.log(p), axis=0))  # geometric mean over classes
    exp1 = 1.0 / (1.0 + lam)
    exp2 = lam / (1.0 + lam)
    q_unnorm = (G ** exp1) * (a ** exp2)
    q = q_unnorm / q_unnorm.sum()
    return q


def _linearization_sweep():
    """Part B: sweep over (β, λ) to validate the linearized bound.

    For each (β, λ), construct synthetic 4-class data with anchor at
    a known biased point, solve with biased and unbiased anchors,
    measure true vs. simplex-projected linearized bound.

    v0.81.0.4: antisymmetric bias model — corrected δ_C formula.
    The 0.81.0.2 spec used δ_C = -(δ_E + δ_R) · q*_C / (q*_E + q*_R)
    derived assuming SYMMETRIC bias [β, 0, β], where renormalization
    forces C to absorb the imbalance. The sweep injects ANTISYMMETRIC
    bias [β, 0, -β] which preserves anchor sum, so renormalization
    is a no-op and δ_C reduces to simplex closure:
        δ_E = +sens · q*_E / p_a,E · β
        δ_R = -sens · q*_R / p_a,R · β   (signed, antisymmetric)
        δ_C = -(δ_E + δ_R)               (simplex closure only)
        bound = |δ_E| + |δ_C| + |δ_R|
    Numerical L1 matches this formula within 1% at all small β,
    vs ~26% systematic offset under the symmetric-derived formula.

    Why antisymmetric: Kraskov bias on (E, S) projections has no
    privileged direction; realistic Kraskov bias profile across
    channels is approximately antisymmetric or independent.
    Symmetric bias would couple to the Z0/Z(λ) integration-cost
    rescaling and double-count that disclosure.

    Earlier history:
    - 0.81.0.0/0.81.0.1: bound formula omitted δ_C entirely (~53%
      systematic underestimate)
    - 0.81.0.2: added δ_C with renormalization factor (correct for
      symmetric bias, ~26% high for antisymmetric)
    - 0.81.0.4: simplex-closure δ_C, antisymmetric bias model

    bound_holds semantics also fixed: a "bound" must upper-bound
    the true L1 distance, so bound_holds requires bound >= true_L1
    (within numerical tolerance) AND bound <= 1.5 * true_L1 (not
    too loose). The 0.81.0.0/0.81.0.1 two-sided 10% tolerance was
    rewarding bounds that systematically underestimated the truth.
    """
    ui.section("Part B — Linearization sweep over (β, λ) grid")
    beta_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    lambda_grid = [0.1, 1.0, 10.0]
    # Reference 4-class shares (slightly heterogeneous to make the
    # sensitivity non-trivial)
    p_classes = np.array([
        [0.45, 0.20, 0.35],   # ridge
        [0.50, 0.18, 0.32],   # mlp
        [0.42, 0.22, 0.36],   # rf
        [0.48, 0.19, 0.33],   # rkhs
    ])
    p_anchor_unbiased = np.array([0.46, 0.22, 0.32])
    rows = []
    for beta in beta_grid:
        # Inject bias on E, R coordinates only (C reconstructed via
        # geometric mean per the apparatus spec — bias on C is zero
        # by construction)
        # Antisymmetric bias model: E coordinate biased +β, R biased -β,
        # C unchanged (kNN-MI doesn't bias C — anchor's C is constructed
        # from the geometric class mean per the apparatus spec, not from
        # kNN). Anchor sum preserved at 1.
        bias_vec_signed = np.array([beta, 0.0, -beta])
        p_anchor_biased = p_anchor_unbiased + bias_vec_signed
        p_anchor_biased = np.clip(p_anchor_biased, 1e-6, 1 - 1e-6)
        p_anchor_biased = p_anchor_biased / p_anchor_biased.sum()
        for lam in lambda_grid:
            q_unbiased = _solve_iprojection_closed_form(
                p_classes, p_anchor_unbiased, lam)
            q_biased = _solve_iprojection_closed_form(
                p_classes, p_anchor_biased, lam)
            true_distance = float(np.abs(q_biased - q_unbiased).sum())

            # Simplex-projected linearized bound under antisymmetric
            # bias model [β, 0, -β] (lagrangianbot, 0.81.0.4).
            #
            # Antisymmetric bias preserves anchor sum at 1 (no
            # renormalization correction), so δ_C is determined purely
            # by simplex closure: δ_C = -(δ_E + δ_R). The q*_C / (q*_E +
            # q*_R) factor in the original spec was a renormalization
            # artifact that doesn't apply here.
            #
            # δ_R carries the antisymmetric sign (E up, R down), so:
            #   δ_E = +sens · q*_E / p_a,E · β
            #   δ_R = -sens · q*_R / p_a,R · β
            #   δ_C = -(δ_E + δ_R)         ≈ 0 to first order when
            #                                q*/p_a ≈ 1 on both
            sens = (lam / (1.0 + lam))
            delta_E = sens * q_unbiased[0] / p_anchor_unbiased[0] * beta
            delta_R = -sens * q_unbiased[2] / p_anchor_unbiased[2] * beta
            delta_C = -(delta_E + delta_R)
            linearized_bound = float(abs(delta_E) + abs(delta_C) + abs(delta_R))

            # bound_holds: bound must upper-bound truth and not be too loose.
            # Tolerances: bound >= 0.95 * true (allow 5% numerical slack on
            # the lower side from higher-order terms in the linearization),
            # bound <= 1.5 * true (not loose by more than 50%).
            if true_distance > 0:
                relative_error = (linearized_bound - true_distance) / true_distance
                upper_bounds = (linearized_bound >= 0.95 * true_distance)
                tight_enough = (linearized_bound <= 1.5 * true_distance)
                bound_holds = bool(upper_bounds and tight_enough)
            else:
                relative_error = 0.0
                bound_holds = True

            rows.append({
                'beta': float(beta),
                'lambda': float(lam),
                'true_l1_distance': true_distance,
                'linearized_bound': linearized_bound,
                'delta_E': float(delta_E),
                'delta_C': float(delta_C),
                'delta_R': float(delta_R),
                'relative_error': float(relative_error),
                'bound_holds': bound_holds,
                'bias_model': 'antisymmetric_E_R',
                'bound_method': 'simplex_closure_antisymmetric_v0_81_0_4',
            })
            ui.msg(f"  β={beta:.2f} λ={lam:>4.1f}  "
                   f"true={true_distance:.4f}  bound={linearized_bound:.4f}  "
                   f"rel_err={relative_error:+.3f}  "
                   f"holds={'yes' if bound_holds else 'NO'}")
    return rows


# ─────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    ui.section("Run 0058 Phase 1 — Kraskov spike + linearization sweep")
    ui.msg(f"Output: {OUT_PATH}")

    per_config, beta = part_a_bias_measurement()
    bucket = _assign_bucket(beta)
    ui.ok(f"Bucket: {bucket} (β = {beta:.4f})")

    sweep_rows = _linearization_sweep()
    n_holds = sum(1 for r in sweep_rows if r['bound_holds'])
    ui.ok(f"Linearization sweep: {n_holds}/{len(sweep_rows)} (β, λ) "
          f"combinations have bound in [0.95·true, 1.5·true]")

    out = {
        'phase': 'phase1_kraskov_spike',
        'iota_version': '0.82.0.23',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'part_a_bias_measurement': {
            'per_config': per_config,
            'beta_overall': beta,
            'bucket': bucket,
            'bucket_thresholds': {
                'strong_max':  0.10,
                'narrow_max':  0.20,
            },
        },
        'part_b_linearization_sweep': {
            'rows': sweep_rows,
            'n_holds': n_holds,
            'n_total': len(sweep_rows),
            'bound_holds_lower_factor': 0.95,
            'bound_holds_upper_factor': 1.5,
        },
        'apparatus_phase_status': {
            'phase1_kraskov_spike':       'done',
            'phase2_solver_module':       'pending',
            'phase3_v5d_threshold':       'pending',
            'phase4_anchor_producer':     'pending',
            'phase5_aggregator':          'pending',
        },
    }
    # v0.83.2: atomic write replaces flush+fsync block (stronger guarantee)
    from results_schema import atomic_json_dump
    atomic_json_dump(out, OUT_PATH, indent=2, ensure_ascii=False)

    # Stamp cache key for freshness mechanism
    with open(CACHE_KEY_PATH, 'w', encoding='utf-8') as f:
        json.dump(_build_cache_key(), f, indent=2)

    elapsed = time.time() - t_total
    ui.ok(f"Phase 1 complete in {elapsed:.1f}s")
    ui.ok(f"Wrote {OUT_PATH}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
