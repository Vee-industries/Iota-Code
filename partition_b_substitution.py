"""
partition_b_substitution.py -- §9.2 + §9.3 evidence base for results.json.

v0.82.0.23 (paper-2 measurement layer).

Joins three source files into a per-cell `partition_b_substitution_diagnostics`
schema block plus a cross-cell `partition_b_substitution_aggregates` block:

  - data/paper/Q0057_function_class_sensitivity.json (PRE shares)
  - data/paper/calibration/function_class_p2/per_cell.json (POST shares)
  - data/paper/calibration/kraskov_anchor_p2/anchors.json (Phase 7 anchor)

Surfaced fields per cell:

  share_deltas:                ΔE, ΔC, ΔR per class (post − pre, simplex-normalized)
  abs_dC_by_class:             |ΔC| per class for the swing-linearity reading
  strict_ordering_holds:       bool, |ΔC| Ridge > MLP > RKHS > RF
  two_tier_ordering_holds:     bool, Ridge > RF AND MLP > RKHS AND RKHS > RF
                               (per item-4 fleet finding: 24/24 cells, the
                                empirically-supported claim)
  g_tilde_pre, g_tilde_post:   renormalized geometric class means
  g_tilde_shift_norm:          ‖G̃_post − G̃_pre‖₂ in 3D simplex
  g_tilde_shift_dC_dominance:  |ΔC| / ‖G̃_shift‖, mean 0.75 fleet-wide
  kl_pre_anchor, kl_post_anchor:
                               KL(G̃_•‖p_anchor); kl_delta_anchor positive
                               on 18/24 cells means G̃ moved further from
                               anchor under partition-B
  anchor_alignment_cosine:     cos(G̃_shift, p_anchor − G̃_pre) in 3D
                               simplex. Positive on 16/24 cells with mean
                               +0.21 -- the C-deflation reframing
                               REPLACED the original "anchor-aligned overfit"
                               framing (which would have predicted negative
                               cosine fleet-wide). The cosine-negative
                               metric was retained because it discriminated
                               between two candidate mechanisms, not
                               because it confirmed one. Phase 7 Step 3
                               sets p_a,C = G̃_C, so the anchor doesn't
                               pull on C by construction; an anchor-direction
                               C-coordinate alignment mechanism is
                               structurally impossible by design. The
                               C-deflation mechanism (Held-out evaluation
                               deflates Q0057's in-sample over-attribution
                               to C; G̃_shift dominated by ΔC; hulls
                               contract in (E, C) projection) is what the
                               data supports.

Cross-cell aggregates: fleet counts of strict/two-tier ordering holding;
fleet-mean |ΔC| per class (Ridge ≈ 0.125, MLP ≈ 0.10, RKHS ≈ 0.017, RF ≈ 0.0004
-- the two orders of magnitude separation between linear-leaning and
tree-based methods that's the §9.2 fleet-supported claim);
fleet-mean dC dominance (≈0.75); fleet counts of KL-further-from-anchor
and cosine-negative.

Mechanism reframing recorded in schema docstrings, not just session
history. Future readers should be able to reconstruct from the schema
that the cosine-negative metric was a discrimination test that
falsified the original mechanism, replaced by the C-deflation reading
that's empirically supported by three quantitative predictions all
holding in the data (ΔC dominates G̃_shift, swings cluster on
linear-leaning methods, tree-based methods partition-stable on C).
"""

import json
import os

import numpy as np


def _norm_share(triple):
    """Normalize {E,C,R} dict to a simplex 3-vector summing to 1."""
    if isinstance(triple, dict):
        a = np.array([triple.get('E', 0.0),
                      triple.get('C', 0.0),
                      triple.get('R', 0.0)], dtype=float)
    else:
        a = np.asarray(triple, dtype=float)
    s = a.sum()
    return a / s if s > 0 else a


def _g_tilde(p_classes_4x3, eps=1e-6):
    """Renormalized geometric class mean."""
    p = np.clip(np.asarray(p_classes_4x3), eps, None)
    G = np.prod(p, axis=0) ** (1.0 / p.shape[0])
    return G / G.sum()


def _kl(p, q, eps=1e-6):
    p_c = np.clip(np.asarray(p), eps, None)
    q_c = np.clip(np.asarray(q), eps, None)
    return float(np.sum(p_c * np.log(p_c / q_c)))


def _load_q57(paper_root):
    p = os.path.join(paper_root, 'Q0057_function_class_sensitivity.json')
    if not os.path.exists(p):
        return {}
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            return (json.load(f) or {}).get('cells') or {}
    except Exception:
        return {}


def _load_p6(paper_root):
    p = os.path.join(paper_root, 'calibration', 'function_class_p2',
                     'per_cell.json')
    if not os.path.exists(p):
        return {}
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            return (json.load(f) or {}).get('cells') or {}
    except Exception:
        return {}


def _load_p7(paper_root):
    p = os.path.join(paper_root, 'calibration', 'kraskov_anchor_p2',
                     'anchors.json')
    if not os.path.exists(p):
        return {}
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            return (json.load(f) or {}).get('cells') or {}
    except Exception:
        return {}


def compute_per_cell_block(cell_key, q57_cells, p6_cells, p7_cells):
    """Compute the per-cell partition_b_substitution_diagnostics block.

    Returns None when source data is missing for this cell.
    """
    pre_root = (q57_cells.get(cell_key) or {}).get('permutation_shares')
    post_root = (p6_cells.get(cell_key) or {}).get('shares')
    p_anchor = (p7_cells.get(cell_key) or {}).get('p_anchor')
    if not (pre_root and post_root and p_anchor):
        return None

    classes = ('ridge', 'mlp', 'rf', 'rkhs_median')
    try:
        pre_shares = {c: _norm_share(pre_root[c]) for c in classes}
        post_shares = {c: _norm_share(post_root[c]) for c in classes}
    except Exception:
        return None
    p_anchor_arr = np.asarray(p_anchor, dtype=float)

    # Per-class deltas
    deltas = {c: post_shares[c] - pre_shares[c] for c in classes}
    abs_dC = {c: float(abs(deltas[c][1])) for c in classes}

    R = abs_dC['ridge']
    M = abs_dC['mlp']
    K = abs_dC['rkhs_median']
    F = abs_dC['rf']

    strict_ordering_holds = bool(R > M > K > F)
    two_tier_ordering_holds = bool(R > F and M > K and K > F)

    # G̃ shift + anchor analysis
    pre_classes_arr = np.stack([pre_shares[c]  for c in classes])
    post_classes_arr = np.stack([post_shares[c] for c in classes])
    G_pre  = _g_tilde(pre_classes_arr)
    G_post = _g_tilde(post_classes_arr)
    G_shift = G_post - G_pre
    G_shift_norm = float(np.linalg.norm(G_shift))
    if G_shift_norm > 0:
        dC_dominance = float(abs(G_shift[1]) / G_shift_norm)
    else:
        dC_dominance = 0.0

    kl_pre_anchor  = _kl(G_pre,  p_anchor_arr)
    kl_post_anchor = _kl(G_post, p_anchor_arr)

    anchor_pull = p_anchor_arr - G_pre
    pn = np.linalg.norm(anchor_pull)
    if G_shift_norm > 0 and pn > 0:
        cos = float(np.dot(G_shift, anchor_pull) / (G_shift_norm * pn))
    else:
        cos = 0.0

    return {
        'share_deltas': {c: {'E': float(deltas[c][0]),
                              'C': float(deltas[c][1]),
                              'R': float(deltas[c][2])}
                          for c in classes},
        'abs_dC_by_class': {
            'ridge':       float(R),
            'mlp':         float(M),
            'rkhs_median': float(K),
            'rf':          float(F),
        },
        'strict_ordering_holds':   strict_ordering_holds,
        'two_tier_ordering_holds': two_tier_ordering_holds,
        'g_tilde_pre':  [float(x) for x in G_pre],
        'g_tilde_post': [float(x) for x in G_post],
        'g_tilde_shift_norm': G_shift_norm,
        'g_tilde_shift_dC_dominance': dC_dominance,
        'kl_pre_anchor':  float(kl_pre_anchor),
        'kl_post_anchor': float(kl_post_anchor),
        'kl_delta_anchor': float(kl_post_anchor - kl_pre_anchor),
        'anchor_alignment_cosine': cos,
    }


def compute_aggregates(per_cell_blocks):
    """Cross-cell aggregates over the per-cell partition-B substitution
    diagnostic blocks.

    The cosine-negative count is preserved because it was the original
    'anchor-aligned overfit' mechanism's prediction; the C-deflation
    reframing replaced that mechanism after the apparatus's Phase 7
    Step 3 construction (p_a,C = G̃_C) was identified as ruling out
    anchor-direction C-alignment by design. Future readers should be
    able to tell from the schema that this metric was retained because
    it discriminated between two candidate mechanisms, not because it
    confirmed one.
    """
    if not per_cell_blocks:
        return {}

    blocks = [v for v in per_cell_blocks.values() if v is not None]
    n = len(blocks)
    if n == 0:
        return {}

    n_strict = sum(1 for v in blocks if v.get('strict_ordering_holds'))
    n_two_tier = sum(1 for v in blocks if v.get('two_tier_ordering_holds'))

    fleet_mean_abs_dC = {}
    for cls in ('ridge', 'mlp', 'rkhs_median', 'rf'):
        vals = [v['abs_dC_by_class'].get(cls, 0.0) for v in blocks]
        fleet_mean_abs_dC[cls] = float(np.mean(vals)) if vals else 0.0

    dC_dom_vals = [v['g_tilde_shift_dC_dominance'] for v in blocks]
    fleet_mean_dC_dominance = float(np.mean(dC_dom_vals)) if dC_dom_vals else 0.0

    n_kl_further = sum(1 for v in blocks if v.get('kl_delta_anchor', 0.0) > 0)
    n_cos_neg    = sum(1 for v in blocks if v.get('anchor_alignment_cosine', 0.0) < 0)

    return {
        'n_cells': n,
        'n_strict_ordering':   n_strict,
        'n_two_tier_ordering': n_two_tier,
        'fleet_mean_abs_dC':   fleet_mean_abs_dC,
        'fleet_mean_g_tilde_shift_dC_dominance': fleet_mean_dC_dominance,
        'n_g_tilde_moved_further_from_anchor': n_kl_further,
        'n_anchor_aligned_cosine_negative':    n_cos_neg,
        'cosine_negative_history_note': (
            "n_anchor_aligned_cosine_negative records cells where "
            "cos(G̃_shift, p_anchor − G̃_pre) < 0. The original "
            "'anchor-aligned overfit' mechanism (Q0057 G̃ overfit "
            "toward anchor → Phase 6 deflates AWAY from anchor) "
            "predicted this count near 24/24. Observed value is "
            "lower because the apparatus's Phase 7 Step 3 sets "
            "p_a,C = G̃_C by construction, ruling out anchor-direction "
            "C-coordinate alignment as a mechanism. The C-deflation "
            "reading replaced the original mechanism: held-out "
            "evaluation deflates Q0057's in-sample C over-attribution "
            "(visible in fleet_mean_abs_dC: linear-leaning methods "
            "two orders of magnitude larger than tree-based), G̃_shift "
            "dominated by ΔC (fleet_mean_g_tilde_shift_dC_dominance), "
            "hulls contract in (E, C) projection. The cosine metric "
            "is retained as the discrimination record."
        ),
    }


def build_for_cell(cell_key, paper_root, _cache={}):
    """Convenience entry point for results_builder.py to call per cell.

    Loads source files once and caches; returns the per-cell block or
    None.
    """
    if 'q57' not in _cache:
        _cache['q57'] = _load_q57(paper_root)
        _cache['p6']  = _load_p6(paper_root)
        _cache['p7']  = _load_p7(paper_root)
    return compute_per_cell_block(cell_key, _cache['q57'],
                                    _cache['p6'], _cache['p7'])


def build_aggregates(paper_root):
    """Build the cross-cell aggregates block for results_builder to
    surface under apparatus.partition_b_substitution_aggregates.
    """
    q57 = _load_q57(paper_root)
    p6  = _load_p6(paper_root)
    p7  = _load_p7(paper_root)

    keys = sorted(set(q57.keys()) & set(p6.keys()) & set(p7.keys()))
    blocks = {}
    for ck in keys:
        b = compute_per_cell_block(ck, q57, p6, p7)
        if b is not None:
            blocks[ck] = b
    return compute_aggregates(blocks)
