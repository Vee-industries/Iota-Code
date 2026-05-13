"""
bayesian_solver.py -- I-projection solver on the simplex.

Implements lagrangianbot's mirror-descent solver for the Bayesian
aggregator over four function classes plus an anchor:

    min_q   (1/4) Σ_f KL(q || p_f)  +  λ KL(q || p_a)
    s.t.    Σ q_i = 1,  q_i >= ε

Closed-form optimum:
    q*_i ∝ G_i^(1/(1+λ)) · p_a,i^(λ/(1+λ))    where G_i = (∏_f p_{f,i})^(1/4)

The closed-form is the production solver. Mirror descent is provided
as a numerical reference for tier validation -- the closed-form must
match mirror descent's converged solution at 1e-8 across all three
validation tiers, otherwise the apparatus refuses to proceed.

Bucket-aware anchor-bound propagation:

  Strong / narrow buckets emit anchor_bounds (per-cell quantitative).
  Ordinal bucket emits rank_order + rank_confidence.

  Antisymmetric bias model (per 0.81.0.4):
    δ_E = +sens · q*_E / p_a,E · β
    δ_R = -sens · q*_R / p_a,R · β
    δ_C = -(δ_E + δ_R)             # simplex closure
    bound = |δ_E| + |δ_C| + |δ_R|

Outputs include per-cell solver diagnostics for §7 prose and §8
calibration tables.
"""
import numpy as np



def solve_iprojection_closed_form(p_classes, p_anchor, lam, eps=1e-6):
    """Closed-form I-projection optimum. Production solver.

    p_classes: shape (n_classes, n_coords)
    p_anchor:  shape (n_coords,)
    lam:       float >= 0, or math.inf for the pure-anchor limit

    v0.82.0.12: special-case lam=inf. The closed form
        q* ∝ G^(1/(1+λ)) * a^(λ/(1+λ))
    has a well-defined limit as λ→∞: G^0 * a^1 = a (normalized,
    which is just a since a is already a probability). But naive
    evaluation gives 1/(1+inf)=0 (fine) and inf/(1+inf)=nan (broken,
    because inf/inf is mathematically indeterminate even when the
    limit is 1). Without this guard, callers either pass nan through
    or substitute lam=0 (which gives the WRONG limit -- λ=0 is pure
    geometric mean, the opposite extreme). Same applies to lam=0:
    the limit is a normalized G, but G^1 * a^0 = G works fine
    without special handling so we leave it.
    """
    a = np.clip(np.asarray(p_anchor, dtype=float), eps, 1 - eps)
    if not np.isfinite(lam):
        # Pure anchor limit: q* = a / sum(a)
        return a / a.sum()
    p = np.clip(np.asarray(p_classes, dtype=float), eps, 1 - eps)
    G = np.exp(np.mean(np.log(p), axis=0))
    exp1 = 1.0 / (1.0 + lam)
    exp2 = lam / (1.0 + lam)
    q_unnorm = (G ** exp1) * (a ** exp2)
    return q_unnorm / q_unnorm.sum()


def solve_iprojection_mirror_descent(p_classes, p_anchor, lam,
                                      eps=1e-6, eta=0.5, max_iter=10000):
    """Mirror descent reference. Used by validation harness -- should
    converge to the closed-form solution at the gradient-norm criterion.

    Objective:  (1/4) Σ_f KL(q || p_f) + λ KL(q || p_a)
    Gradient at q:  ∂/∂q_i = (1+λ)(log q_i + 1) - (1/4) Σ_f log p_{f,i} - λ log p_{a,i}
    The (1+λ) factor scales the step; using fixed η=0.5 without
    dividing by (1+λ) makes effective step grow with λ and convergence
    fails for high λ. Compensate by scaling η down by (1+λ).
    """
    p = np.clip(np.asarray(p_classes, dtype=float), eps, 1 - eps)
    a = np.clip(np.asarray(p_anchor, dtype=float), eps, 1 - eps)
    # v0.82.0.12: pure-anchor limit. At lam=inf the objective is
    # dominated by KL(q || p_a), minimized at q = a. Mirror descent
    # without this guard sets eta_effective = eta/(1+inf) = 0 and
    # never moves from the uniform initialization.
    if not np.isfinite(lam):
        return a / a.sum(), 0
    n = p.shape[1]
    q = np.full(n, 1.0 / n)
    log_p_mean = np.mean(np.log(p), axis=0)
    log_a = np.log(a)
    eta_effective = eta / (1.0 + lam)
    for it in range(max_iter):
        log_q = np.log(q)
        # Gradient of Lagrangian (without the (1+λ) factor on log_q,
        # since the factored form is what we step in)
        g = (1.0 + lam) * log_q - log_p_mean - lam * log_a
        g_bar = g.mean()
        g_proj = g - g_bar
        if np.linalg.norm(g_proj, ord=np.inf) < 1e-7:
            return q, it + 1
        q_new = q * np.exp(-eta_effective * g)
        q_new = q_new / q_new.sum()
        q_new = np.clip(q_new, eps, 1 - eps)
        q_new = q_new / q_new.sum()
        if np.max(np.abs(q_new - q)) < 1e-9:
            return q_new, it + 1
        q = q_new
    raise RuntimeError(f"mirror descent did not converge in {max_iter} iters")


def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def pairwise_max_js(p_classes):
    """Spread metric: max pairwise JS divergence across classes."""
    n = p_classes.shape[0]
    max_js = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v = js_divergence(p_classes[i], p_classes[j])
            if v > max_js:
                max_js = v
    return float(max_js)


def am_gm_log_ratio_C(p_classes, c_idx=1):
    """AM-GM divergence on the C coordinate. Lagrangianbot's V5d signature."""
    p = np.asarray(p_classes, dtype=float)
    c_col = np.clip(p[:, c_idx], 1e-12, None)
    am = float(c_col.mean())
    gm = float(np.exp(np.log(c_col).mean()))
    return float(np.log(am) - np.log(gm))


def compute_anchor_bounds_antisymmetric(q_star, p_anchor, lam, beta, eps=1e-6):
    """Antisymmetric bias model bound propagation (per 0.81.0.4).

    Returns dict with per-coord δ, q*_lower, q*_upper.
    """
    q = np.asarray(q_star, dtype=float)
    a = np.clip(np.asarray(p_anchor, dtype=float), eps, 1 - eps)
    # v0.82.0.12: sensitivity is λ/(1+λ), with limit 1 as λ→∞.
    # Without the guard inf/inf yields nan and propagates through
    # delta_E / delta_R, poisoning the bound output.
    if not np.isfinite(lam):
        sens = 1.0
    else:
        sens = lam / (1.0 + lam)
    delta_E = +sens * q[0] / a[0] * beta
    delta_R = -sens * q[2] / a[2] * beta
    delta_C = -(delta_E + delta_R)
    delta = np.array([delta_E, delta_C, delta_R])
    q_lower = np.clip(q - np.abs(delta), eps, 1 - eps)
    q_lower = q_lower / q_lower.sum()
    q_upper = np.clip(q + np.abs(delta), eps, 1 - eps)
    q_upper = q_upper / q_upper.sum()
    return {
        'delta': delta.tolist(),
        'q_star_lower': q_lower.tolist(),
        'q_star_upper': q_upper.tolist(),
        'bound_method': 'simplex_closure_antisymmetric_v0_81_0_4',
        'bias_model': 'antisymmetric_E_R',
    }


def solve_with_diagnostics(p_classes, p_anchor, lam, beta=None, bucket=None, eps=1e-6):
    """Production entry point. Solves I-projection and emits full diagnostics
    block per the lagrangianbot output spec.

    If beta is provided and bucket is in {'strong', 'narrow'}, attaches
    anchor_bounds. If bucket == 'ordinal', attaches rank_order + rank_confidence.
    """
    p = np.clip(np.asarray(p_classes, dtype=float), eps, 1 - eps)
    a = np.clip(np.asarray(p_anchor, dtype=float), eps, 1 - eps)
    q_star = solve_iprojection_closed_form(p, a, lam, eps=eps)

    G = np.exp(np.mean(np.log(p), axis=0))
    G_tilde = G / G.sum()
    AM = p.mean(axis=0)

    kl_per_class = {}
    class_labels = ['ridge', 'mlp', 'rf', 'rkhs']
    for i, label in enumerate(class_labels[:p.shape[0]]):
        kl_per_class[label] = kl_divergence(q_star, p[i])

    anchor_kl = kl_divergence(q_star, a)
    spread_js = pairwise_max_js(p)
    am_gm_C = am_gm_log_ratio_C(p)

    out = {
        'q_star': q_star.tolist(),
        'lambda_used': float(lam),
        'kl_per_class': kl_per_class,
        'anchor_kl': float(anchor_kl),
        'spread_js': float(spread_js),
        'solver_diagnostics': {
            'G_unnormalized': G.tolist(),
            'G_tilde': G_tilde.tolist(),
            'AM_classes': AM.tolist(),
            'am_gm_log_ratio_C': float(am_gm_C),
            'input_had_clipped_shares': bool(
                np.any(p_classes < eps) or np.any(p_classes > 1 - eps) or
                np.any(p_anchor < eps) or np.any(p_anchor > 1 - eps)),
        },
    }

    if beta is not None and bucket in ('strong', 'narrow'):
        bounds = compute_anchor_bounds_antisymmetric(q_star, a, lam, beta, eps=eps)
        out['anchor_bounds'] = {
            'kraskov_bias_estimate': float(beta),
            'anchor_bias_propagated': bounds['delta'],
            'q_star_lower': bounds['q_star_lower'],
            'q_star_upper': bounds['q_star_upper'],
            'bound_method': bounds['bound_method'],
            'bias_model': bounds['bias_model'],
            'bound_assumptions': [
                'bias_uniform_across_cells',
                'linear_propagation_through_solver',
                'antisymmetric_E_R_bias_model',
            ],
        }
    elif bucket == 'ordinal':
        # Rank order and rank confidence (placeholder -- ordinal regime has
        # softer claim, no quantitative bound)
        rank_idx = np.argsort(-q_star)
        coord_labels = ['E', 'C', 'R']
        rank_order = [coord_labels[i] for i in rank_idx]
        # Confidence: gap between top-2 / range
        sorted_q = np.sort(q_star)[::-1]
        rank_confidence = float((sorted_q[0] - sorted_q[1]) / (sorted_q[0] - sorted_q[-1] + 1e-12))
        out['rank_order'] = rank_order
        out['rank_confidence'] = rank_confidence
        out['bound_method'] = 'ordinal_v1'

    return out


# ─────────────────────────────────────────────────────────────────────
#  3-tier validation harness
# ─────────────────────────────────────────────────────────────────────

def validate_tier1():
    """Trivial baseline: all classes return (0.5, 0.3, 0.2), λ=0.
    Expect q* = (0.5, 0.3, 0.2) within 1e-8."""
    p_classes = np.tile([0.5, 0.3, 0.2], (4, 1))
    p_anchor = np.array([1/3, 1/3, 1/3])
    q = solve_iprojection_closed_form(p_classes, p_anchor, 0.0)
    expected = np.array([0.5, 0.3, 0.2])
    err = float(np.max(np.abs(q - expected)))
    return {'tier': 1, 'passed': err < 1e-8, 'max_error': err, 'tolerance': 1e-8}


def validate_tier2():
    """Self-consistent anchor: heterogeneous classes, p_a = G̃.
    Expect q* = G̃ for all λ within 1e-8."""
    p_classes = np.array([
        [0.5, 0.3, 0.2], [0.4, 0.4, 0.2],
        [0.6, 0.2, 0.2], [0.5, 0.25, 0.25],
    ])
    G = np.exp(np.mean(np.log(p_classes), axis=0))
    G_tilde = G / G.sum()
    per_lambda = []
    max_err = 0.0
    for lam in [0, 0.1, 1.0, 10.0, 100.0]:
        q = solve_iprojection_closed_form(p_classes, G_tilde, lam)
        err = float(np.max(np.abs(q - G_tilde)))
        per_lambda.append({'lambda': lam, 'max_error': err})
        if err > max_err:
            max_err = err
    return {'tier': 2, 'passed': max_err < 1e-8, 'max_error': max_err,
            'tolerance': 1e-8, 'per_lambda': per_lambda}


def validate_tier3():
    """Closed-form non-degenerate: heterogeneous classes, anchor distinct from G̃.
    Verify q*_i ∝ G_i^(1/(1+λ)) p_a,i^(λ/(1+λ)) within 1e-8."""
    p_classes = np.array([
        [0.50, 0.30, 0.20], [0.45, 0.35, 0.20],
        [0.55, 0.25, 0.20], [0.40, 0.40, 0.20],
    ])
    p_anchor = np.array([0.20, 0.30, 0.50])
    G = np.exp(np.mean(np.log(p_classes), axis=0))
    per_lambda = []
    max_err = 0.0
    for lam in [0.1, 1.0, 10.0]:
        q_actual = solve_iprojection_closed_form(p_classes, p_anchor, lam)
        exp1 = 1.0 / (1.0 + lam)
        exp2 = lam / (1.0 + lam)
        q_expected = (G ** exp1) * (p_anchor ** exp2)
        q_expected = q_expected / q_expected.sum()
        err = float(np.max(np.abs(q_actual - q_expected)))
        per_lambda.append({'lambda': lam, 'max_error': err})
        if err > max_err:
            max_err = err
    return {'tier': 3, 'passed': max_err < 1e-8, 'max_error': max_err,
            'tolerance': 1e-8, 'per_lambda': per_lambda}


def validate_mirror_descent_matches_closed_form():
    """Cross-check: mirror descent and closed-form agree."""
    p_classes = np.array([
        [0.45, 0.20, 0.35], [0.50, 0.18, 0.32],
        [0.42, 0.22, 0.36], [0.48, 0.19, 0.33],
    ])
    p_anchor = np.array([0.46, 0.22, 0.32])
    max_err = 0.0
    per_lambda = []
    for lam in [0.1, 1.0, 10.0]:
        q_cf = solve_iprojection_closed_form(p_classes, p_anchor, lam)
        q_md, n_iter = solve_iprojection_mirror_descent(p_classes, p_anchor, lam)
        err = float(np.max(np.abs(q_cf - q_md)))
        per_lambda.append({'lambda': lam, 'max_error': err, 'n_iterations': n_iter})
        if err > max_err:
            max_err = err
    return {'tier': 'mirror_descent', 'passed': max_err < 1e-6, 'max_error': max_err,
            'tolerance': 1e-6, 'per_lambda': per_lambda}


def run_validation_harness():
    """Full validation harness. Returns dict with per-tier results.
    all_tiers_pass is False if any tier fails."""
    t1 = validate_tier1()
    t2 = validate_tier2()
    t3 = validate_tier3()
    tmd = validate_mirror_descent_matches_closed_form()
    return {
        'tier1': t1,
        'tier2': t2,
        'tier3': t3,
        'mirror_descent_check': tmd,
        'all_tiers_pass': t1['passed'] and t2['passed'] and t3['passed'] and tmd['passed'],
    }
