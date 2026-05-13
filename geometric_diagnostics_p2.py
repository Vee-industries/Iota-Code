"""
geometric_diagnostics_p2.py -- Module 2 of the paper-2 apparatus.

v0.82.0.23 (paper-2 measurement layer).

Per-cell topology and magnitude diagnostics on:
  - the recovered partition q* (from Module 1 / Phase 8),
  - the four class shares {p_f} (from Phase 6),
  - the renormalized geometric class mean G̃ (computed locally
    here from p_f; same definition as solver_diagnostics.G_tilde).

Eight fields per cell (statsbot's spec):

  Field 1: q_star_inside_class_hull   -- bool
  Field 2: geometric_mean_inside_class_hull -- bool
  Field 3: anchor_pull_distance       -- float (nats), KL(q* || G̃)
  Field 4: signed_anchor_pull         -- float (nats), signed by Field 1
  Field 5: class_hull_volume          -- float (E,C)-area
  Field 6: hull_degenerate            -- bool
  Field 7: hull_exit_mechanism        -- str ("none" / "anchor_pull" /
                                              "geometric_mean_exit")
  Field 8: hull_signed_distance       -- float, Euclidean to hull
                                         boundary, signed by Field 1
                                         (Ship 12 sign convention)

Replaces Ship 12's `convex_hull_diagnostic` schema atomically. Field 1
preserves Ship 12's `anchored_in_hull` semantics; Field 8 preserves
Ship 12's `hull_signed_distance` sign convention (positive=inside,
negative=outside). Migration directive in paper2_apparatus.md.

Cross-cell aggregates also computed here -- counts, mechanism counts,
quadrant distribution, V5d cross-reference. The V5d-flagged quadrant
distribution reads V5d flags from the aggregator's per-cell output
(v5d_flag_combined), not from Phase 11's own output, since V5d is a
downstream-of-Module-2 concept per the spec.

V0 protocol -- every numerical claim that landed in this module was
independently re-verified before implementation per statsbot's
spec-drift catch list. The G1-G4 test harness asserts the verified
values, not the spec literals; if a future spec edit changes a
worked example, the test catches the inconsistency at run time.

Dependencies: numpy, scipy.spatial.ConvexHull. No new heavy deps.
"""

import numpy as np
from scipy.spatial import ConvexHull


# ── Constants ────────────────────────────────────────────────────────

EPS = 1e-6           # KL stability floor; matches aggregator pipeline
HULL_TOL = 1e-9      # tolerance for hull-boundary equality
DEDUP_TOL = 10       # decimal places for dedup of class points


# ── G̃ computation (canonical) ───────────────────────────────────────

def compute_g_tilde(p_classes, eps=EPS):
    """Renormalized geometric class mean.

    G̃_i = (∏_f p_{f,i})^(1/n_classes) / Σ_j (∏_f p_{f,j})^(1/n_classes)

    Same formula as solver_diagnostics.G_tilde in bayesian_solver.py;
    duplicated locally so Module 2 doesn't depend on the solver's
    diagnostics block being threaded through Phase 8 / Phase 11.

    Returns 3-vector summing to 1.
    """
    p = np.clip(np.asarray(p_classes, dtype=float), eps, None)
    n = p.shape[0]
    G = np.prod(p, axis=0) ** (1.0 / n)
    return G / G.sum()


# ── Hull membership + signed distance ────────────────────────────────

def _hull_membership_and_signed_distance(point_3d, p_classes,
                                            eps=EPS, hull_tol=HULL_TOL):
    """Test whether point_3d (a simplex point) lies inside the convex
    hull of p_classes (a (k, 3) array of simplex points), and compute
    Euclidean signed distance to the hull boundary in (E, C) projection.

    Returns dict with:
      inside      bool  -- True if inside hull (≥ boundary within tol).
      signed_dist float -- Euclidean distance to nearest facet,
                          signed positive=inside, negative=outside
                          per Ship 12 convention.
      degenerate  bool  -- True if hull is colinear or coincident.
      hull_volume float -- (E,C)-area of the hull (0 if degenerate).

    Algorithm matches statsbot's spec literally:
      - Drop R, work in (E, C). All five points sum to 1, so dropping
        R is information-preserving for hull topology.
      - Non-degenerate hull: scipy.spatial.ConvexHull, signed distance
        via hull.equations facets:
            signed_dist = -(eqs[:, :2] @ q + eqs[:, 2]).min()
        (positive when inside, negative when outside).
      - Segment-degenerate: point-to-segment distance; sign by whether
        q lies on the segment within tolerance.
      - Point-degenerate: Euclidean distance to the single point;
        sign positive iff distance ≤ tol.
    """
    pts = np.asarray(p_classes, dtype=float)
    pts2d = pts[:, :2]
    q2d = np.asarray(point_3d, dtype=float)[:2]

    # Detect degeneracy via dedup of (E, C) projection
    unique_pts = np.unique(np.round(pts2d, decimals=DEDUP_TOL), axis=0)

    # Point-degenerate (all classes coincide in (E, C) projection)
    if len(unique_pts) == 1:
        d = float(np.linalg.norm(q2d - unique_pts[0]))
        inside = d <= hull_tol
        return {
            'inside':      inside,
            'signed_dist': d if inside else -d,
            'degenerate':  True,
            'hull_volume': 0.0,
        }

    # Segment-degenerate (two distinct points, hull is a 1D segment)
    if len(unique_pts) == 2:
        p0, p1 = unique_pts[0], unique_pts[1]
        v = p1 - p0
        v2 = float(np.dot(v, v))
        if v2 < hull_tol ** 2:
            # Effectively a point
            d = float(np.linalg.norm(q2d - p0))
            inside = d <= hull_tol
            return {
                'inside':      inside,
                'signed_dist': d if inside else -d,
                'degenerate':  True,
                'hull_volume': 0.0,
            }
        t_unclamped = float(np.dot(q2d - p0, v) / v2)
        t_clamped = max(0.0, min(1.0, t_unclamped))
        foot = p0 + t_clamped * v
        d = float(np.linalg.norm(q2d - foot))
        inside = (d <= hull_tol) and (0.0 <= t_unclamped <= 1.0)
        return {
            'inside':      inside,
            'signed_dist': d if inside else -d,
            'degenerate':  True,
            'hull_volume': 0.0,
        }

    # Non-degenerate (≥ 3 distinct points → planar hull)
    try:
        hull = ConvexHull(pts2d)
    except Exception:
        # Fall back to degenerate handling if scipy fails
        d = float(np.linalg.norm(q2d - pts2d.mean(axis=0)))
        return {
            'inside':      False,
            'signed_dist': -d,
            'degenerate':  True,
            'hull_volume': 0.0,
        }

    # signed[j] = signed distance to facet j (positive when inside)
    signed = -(hull.equations[:, :2] @ q2d + hull.equations[:, 2])
    signed_dist = float(signed.min())
    inside = signed_dist >= -hull_tol
    return {
        'inside':      bool(inside),
        'signed_dist': signed_dist,
        'degenerate':  False,
        'hull_volume': float(hull.volume),
    }


# ── KL distance with epsilon-clip ────────────────────────────────────

def _kl_clipped(p, q, eps=EPS):
    """KL(p || q) = Σ p_i log(p_i / q_i), with both clipped at eps."""
    p_c = np.clip(np.asarray(p, dtype=float), eps, None)
    q_c = np.clip(np.asarray(q, dtype=float), eps, None)
    return float(np.sum(p_c * np.log(p_c / q_c)))


# ── Per-cell diagnostics ─────────────────────────────────────────────

def compute_geometric_diagnostics(q_star, p_classes, G_tilde=None,
                                     eps=EPS):
    """Compute the eight-field geometric diagnostics block per spec.

    Args:
      q_star: 3-vector (E, C, R), the recovered partition.
      p_classes: (k, 3) array, the four class shares.
      G_tilde: optional 3-vector, the renormalized geometric class
        mean. If None, computed locally from p_classes.
      eps: KL/clip tolerance.

    Returns dict matching the spec's output block exactly:
      q_star_inside_class_hull, geometric_mean_inside_class_hull,
      anchor_pull_distance, signed_anchor_pull, class_hull_volume,
      hull_degenerate, hull_exit_mechanism, hull_signed_distance.
    """
    q = np.asarray(q_star, dtype=float)
    p = np.asarray(p_classes, dtype=float)
    G = compute_g_tilde(p, eps=eps) if G_tilde is None else \
            np.asarray(G_tilde, dtype=float)

    # Field 1, 5, 6, 8: q* hull membership + signed distance + volume
    q_hull = _hull_membership_and_signed_distance(q, p, eps=eps)

    # Field 2: G̃ hull membership (same hull, different test point)
    G_hull = _hull_membership_and_signed_distance(G, p, eps=eps)

    # Field 3: KL(q* || G̃) -- anchor pull distance in nats
    pull_distance = _kl_clipped(q, G, eps=eps)

    # Field 4: signed_anchor_pull = ±pull_distance, sign by Field 1
    signed_pull = pull_distance if q_hull['inside'] else -pull_distance

    # Field 7: hull_exit_mechanism (categorical attribution)
    if q_hull['inside']:
        mechanism = 'none'
    elif G_hull['inside']:
        mechanism = 'anchor_pull'
    else:
        mechanism = 'geometric_mean_exit'

    return {
        'q_star_inside_class_hull':         bool(q_hull['inside']),
        'geometric_mean_inside_class_hull': bool(G_hull['inside']),
        'anchor_pull_distance':             float(pull_distance),
        'signed_anchor_pull':               float(signed_pull),
        'class_hull_volume':                float(q_hull['hull_volume']),
        'hull_degenerate':                  bool(q_hull['degenerate']),
        'hull_exit_mechanism':              mechanism,
        'hull_signed_distance':             float(q_hull['signed_dist']),
    }


# ── Cross-cell aggregates ────────────────────────────────────────────

def compute_geometric_aggregates(per_cell, v5d_flagged=None):
    """Cross-cell aggregates over the per-cell diagnostics block.

    Args:
      per_cell: dict of {cell_key: geometric_diagnostics_block}.
      v5d_flagged: optional set/list of cell keys flagged by V5d
        (spread_js > τ_spread or |amgm| > τ_amgm). If None,
        v5d_flagged_quadrant_distribution is None in the output.

    Returns dict matching the spec's geometric_aggregates block.

    Quadrants: split by inside/outside hull (Field 1) × small/large
    anchor pull (Field 3 split at cross-cell median).
    """
    if v5d_flagged is None:
        v5d_set = None
    else:
        v5d_set = set(v5d_flagged)

    n_inside = sum(1 for v in per_cell.values()
                       if v.get('q_star_inside_class_hull'))
    n_outside = sum(1 for v in per_cell.values()
                        if not v.get('q_star_inside_class_hull'))
    n_degen = sum(1 for v in per_cell.values()
                       if v.get('hull_degenerate'))

    mech_counts = {'none': 0, 'anchor_pull': 0, 'geometric_mean_exit': 0}
    for v in per_cell.values():
        m = v.get('hull_exit_mechanism', 'none')
        if m in mech_counts:
            mech_counts[m] += 1

    pulls = [v.get('anchor_pull_distance', 0.0) for v in per_cell.values()]
    if pulls:
        mean_pull   = float(np.mean(pulls))
        median_pull = float(np.median(pulls))
        max_pull    = float(np.max(pulls))
        max_cell = max(per_cell.items(),
                          key=lambda kv: kv[1].get('anchor_pull_distance', 0.0))[0]
    else:
        mean_pull = median_pull = max_pull = 0.0
        max_cell = None

    # Quadrant assignment: inside × {small, large} where small/large
    # is split at cross-cell median pull.
    def _quad(block, median):
        inside = block.get('q_star_inside_class_hull')
        large = block.get('anchor_pull_distance', 0.0) > median
        if inside and not large:  return 'inside_small'
        if inside and large:      return 'inside_large'
        if not inside and not large: return 'outside_small'
        return 'outside_large'

    quad_counts = {'inside_small': 0, 'inside_large': 0,
                    'outside_small': 0, 'outside_large': 0}
    v5d_quad = {'inside_small': 0, 'inside_large': 0,
                 'outside_small': 0, 'outside_large': 0}
    for ck, v in per_cell.items():
        q = _quad(v, median_pull)
        quad_counts[q] += 1
        if v5d_set is not None and ck in v5d_set:
            v5d_quad[q] += 1

    return {
        'n_cells_inside_hull':       n_inside,
        'n_cells_outside_hull':      n_outside,
        'n_cells_hull_degenerate':   n_degen,
        'hull_exit_mechanism_counts': mech_counts,
        'mean_anchor_pull_distance':   mean_pull,
        'median_anchor_pull_distance': median_pull,
        'max_anchor_pull_distance':    max_pull,
        'max_anchor_pull_cell':        max_cell,
        'quadrant_counts':             quad_counts,
        'v5d_flagged_quadrant_distribution':
            v5d_quad if v5d_set is not None else None,
    }


# ── Bootstrap variance decomposition (v0.82.0.23 schema extension) ──

def compute_bootstrap_variance_decomposition(per_resample_shares,
                                                eps=EPS, hull_tol=HULL_TOL):
    """Decompose hull-distance bootstrap variance into q*-only,
    hull-only, and cross-term components.

    Per statsbot's spec (v0.82.0.23 CHANGELOG, Module 2 schema
    extension): for each resample i in the bootstrap output, three
    counterfactuals are evaluated:

      observed[i]    = hull_signed_distance(q*_i, hull_i)
      qstar_only[i]  = hull_signed_distance(q*_i, mean(hull))
      hull_only[i]   = hull_signed_distance(mean(q*), hull_i)

    Then:

      var(observed) = var(qstar_only) + var(hull_only) + cross_term

    where cross_term captures the q*-vs-hull anti-correlation
    that's the I-projection apparatus's damping signature.

    Empirically: hull-only variance exceeds observed variance
    (hull_frac > 1) on cells where q* tracks the hull boundary,
    with negative cross terms -- q* is anti-correlated with hull
    movement under bootstrap resampling, real apparatus damping
    rather than noise.

    Args:
      per_resample_shares: dict with keys 'ridge', 'mlp', 'rf',
        'rkhs', 'anchored', each value a list of n_resamples
        x 3-vector entries (E, C, R simplex points).

    Returns:
      dict with q_star_movement_var, hull_movement_var, cross_term,
      n_resamples, hull_only_frac, q_star_only_frac, cross_frac.
      All variances in (E, C) Euclidean units squared. The fractions
      are unitless and sum to 1 + cross_frac (with cross typically
      negative); hull_only_frac > 1 indicates strong damping.

    Returns None if shares are missing or n_resamples < 2.
    """
    if not per_resample_shares:
        return None
    try:
        ridge = np.asarray(per_resample_shares['ridge'], dtype=float)
        mlp   = np.asarray(per_resample_shares['mlp'], dtype=float)
        rf    = np.asarray(per_resample_shares['rf'], dtype=float)
        rkhs  = np.asarray(per_resample_shares['rkhs'], dtype=float)
        qstar = np.asarray(per_resample_shares['anchored'], dtype=float)
    except (KeyError, ValueError, TypeError):
        return None
    n = len(qstar)
    if n < 2 or any(arr.shape[0] != n for arr in (ridge, mlp, rf, rkhs)):
        return None

    def _hsd(q2d, pts2d):
        """Internal hull-signed-distance for (E, C) projection."""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts2d)
            signed = -(hull.equations[:, :2] @ q2d + hull.equations[:, 2])
            return float(signed.min())
        except Exception:
            return float('nan')

    # Observed: each resample uses its own (q*_i, hull_i)
    observed = np.array([
        _hsd(qstar[i, :2],
             np.stack([ridge[i], mlp[i], rkhs[i], rf[i]])[:, :2])
        for i in range(n)
    ])

    # q*-only counterfactual: hull held FIXED at across-resample mean
    fixed_classes_2d = np.stack([
        ridge.mean(axis=0), mlp.mean(axis=0),
        rkhs.mean(axis=0), rf.mean(axis=0),
    ])[:, :2]
    qstar_only = np.array([
        _hsd(qstar[i, :2], fixed_classes_2d) for i in range(n)
    ])

    # hull-only counterfactual: q* held FIXED at across-resample mean
    qstar_mean_2d = qstar.mean(axis=0)[:2]
    hull_only = np.array([
        _hsd(qstar_mean_2d,
             np.stack([ridge[i], mlp[i], rkhs[i], rf[i]])[:, :2])
        for i in range(n)
    ])

    var_obs = float(np.var(observed,    ddof=1))
    var_q   = float(np.var(qstar_only,  ddof=1))
    var_h   = float(np.var(hull_only,   ddof=1))
    cross   = var_obs - var_q - var_h

    if var_obs > 0:
        q_frac    = var_q / var_obs
        hull_frac = var_h / var_obs
        cross_frac = cross / var_obs
    else:
        q_frac = hull_frac = cross_frac = float('nan')

    return {
        'q_star_movement_var': var_q,
        'hull_movement_var':   var_h,
        'cross_term':          float(cross),
        'observed_var':        var_obs,
        'n_resamples':         int(n),
        'q_star_only_frac':    float(q_frac),
        'hull_only_frac':      float(hull_frac),
        'cross_frac':          float(cross_frac),
        'bootstrap_median_signed_distance': float(np.median(observed)),
        'bootstrap_n_inside': int((observed >= -hull_tol).sum()),
    }

