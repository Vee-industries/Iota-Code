"""
tests/test_geometric_diagnostics_p2.py — G1–G4 validation harness.

Per statsbot's spec (paper2_apparatus.md Module 2 validation), all
four tiers must pass before downstream consumers run.

  G1 — Identical classes (degenerate hull)
  G2 — Self-consistent anchor (q* = G̃ across all λ)
  G3 — Strong anchor pull (q* far outside tight class hull)
  G4 — Near-degenerate hull, geometric mean exit

V0 protocol: numerical values asserted here were independently
re-verified before the implementation was written. Spec literals are
NOT load-bearing — the tests assert the computed values, so a future
spec edit that drifts a worked example won't pass-by-coincidence.

Run from project root:
    python tests/test_geometric_diagnostics_p2.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import geometric_diagnostics_p2 as gd
import bayesian_solver as ls


# ── G1 ───────────────────────────────────────────────────────────────

def test_g1_identical_classes():
    """G1 — four identical classes p_f = (0.4, 0.3, 0.3).

    Hull degenerates to a single point. q* = G̃ = (0.4, 0.3, 0.3).
    Both hull-membership flags True (point coincides with itself).
    """
    print("\n[G1] identical classes (0.4, 0.3, 0.3)")
    p = np.array([[0.4, 0.3, 0.3]] * 4)
    p_a = np.array([0.4, 0.3, 0.3])
    q_star = ls.solve_iprojection_closed_form(p, p_a, lam=1.0)

    diag = gd.compute_geometric_diagnostics(q_star, p)

    expected = {
        'q_star_inside_class_hull':         True,
        'geometric_mean_inside_class_hull': True,
        'anchor_pull_distance':             0.0,
        'signed_anchor_pull':               0.0,
        'class_hull_volume':                0.0,
        'hull_degenerate':                  True,
        'hull_exit_mechanism':              'none',
    }
    failures = []
    for k, v in expected.items():
        actual = diag[k]
        if isinstance(v, float):
            ok = abs(actual - v) < 1e-9
        else:
            ok = actual == v
        if not ok:
            failures.append(f"{k}: expected {v}, got {actual}")
    # hull_signed_distance — magnitude < tolerance
    if abs(diag['hull_signed_distance']) > 1e-6:
        failures.append(
            f"hull_signed_distance: |{diag['hull_signed_distance']}| > 1e-6"
        )
    print(f"      diag: {diag}")
    return failures


# ── G2 ───────────────────────────────────────────────────────────────

def test_g2_self_consistent_anchor():
    """G2 — distinct classes, p_a = G̃, sweep λ. q* = G̃ for all λ.

    Asserts solver self-consistency (Module 1 T2 territory; restated
    here per spec for completeness) AND records hull diagnostics
    without asserting them as preconditions.
    """
    print("\n[G2] self-consistent anchor across λ ∈ {0, 0.1, 1, 10, 100}")
    p = np.array([
        [0.40, 0.30, 0.30],
        [0.35, 0.40, 0.25],
        [0.45, 0.25, 0.30],
        [0.50, 0.20, 0.30],
    ])
    G_tilde = gd.compute_g_tilde(p)
    p_a = G_tilde.copy()

    failures = []
    for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
        q_star = ls.solve_iprojection_closed_form(p, p_a, lam=lam)
        if np.max(np.abs(q_star - G_tilde)) > 1e-8:
            failures.append(
                f"λ={lam}: q* not equal to G̃ within 1e-8 "
                f"(q*={q_star}, G̃={G_tilde})"
            )
        diag = gd.compute_geometric_diagnostics(q_star, p, G_tilde=G_tilde)
        # Assertions: pull = 0, signed pull = 0
        if abs(diag['anchor_pull_distance']) > 1e-9:
            failures.append(
                f"λ={lam}: anchor_pull_distance = "
                f"{diag['anchor_pull_distance']:.4e} > 1e-9"
            )
        if abs(diag['signed_anchor_pull']) > 1e-9:
            failures.append(
                f"λ={lam}: signed_anchor_pull = "
                f"{diag['signed_anchor_pull']:.4e} > 1e-9"
            )
        # Recording-only fields per spec — not asserted as preconditions
        print(f"      λ={lam:6.1f}: inside_hull={diag['q_star_inside_class_hull']}, "
                f"signed_dist={diag['hull_signed_distance']:+.4f}, "
                f"mechanism={diag['hull_exit_mechanism']}")
    return failures


# ── G3 ───────────────────────────────────────────────────────────────

def test_g3_strong_anchor_pull():
    """G3 — clustered classes around (0.5, 0.3, 0.2), anchor far at
    (0.1, 0.1, 0.8), λ=10. q* pulled outside the tight class hull.
    """
    print("\n[G3] strong anchor pull (cluster, anchor=(0.1,0.1,0.8), λ=10)")
    cluster = np.array([0.5, 0.3, 0.2])
    spread = 0.01
    p = np.array([
        cluster + np.array([+spread, -spread/2, -spread/2]),
        cluster + np.array([-spread, +spread/2, -spread/2]),
        cluster + np.array([+spread/2, +spread, -3*spread/2]),
        cluster + np.array([-spread/2, -spread/2, +spread]),
    ])
    p = p / p.sum(axis=1, keepdims=True)
    p_a = np.array([0.1, 0.1, 0.8])
    lam = 10.0

    q_star = ls.solve_iprojection_closed_form(p, p_a, lam=lam)
    diag = gd.compute_geometric_diagnostics(q_star, p)

    print(f"      q* = ({q_star[0]:.3f}, {q_star[1]:.3f}, {q_star[2]:.3f})")
    print(f"      diag: inside={diag['q_star_inside_class_hull']}, "
            f"G̃_inside={diag['geometric_mean_inside_class_hull']}, "
            f"pull={diag['anchor_pull_distance']:.4f}, "
            f"signed_dist={diag['hull_signed_distance']:.4f}")
    print(f"      mechanism: {diag['hull_exit_mechanism']}")

    failures = []
    if diag['q_star_inside_class_hull']:
        failures.append("q_star_inside_class_hull should be False (q* far outside)")
    if not diag['geometric_mean_inside_class_hull']:
        failures.append("geometric_mean_inside_class_hull should be True (clustered classes have G̃ in their tight hull)")
    if diag['anchor_pull_distance'] <= 0.5:
        failures.append(
            f"anchor_pull_distance should be > 0.5, got "
            f"{diag['anchor_pull_distance']:.4f}"
        )
    if diag['signed_anchor_pull'] >= 0:
        failures.append(
            f"signed_anchor_pull should be negative (q* outside), got "
            f"{diag['signed_anchor_pull']:.4f}"
        )
    if diag['hull_exit_mechanism'] != 'anchor_pull':
        failures.append(
            f"hull_exit_mechanism should be 'anchor_pull', got "
            f"'{diag['hull_exit_mechanism']}'"
        )
    if diag['hull_signed_distance'] >= 0:
        failures.append(
            f"hull_signed_distance should be << 0 (q* far outside hull), got "
            f"{diag['hull_signed_distance']:.4f}"
        )
    return failures


# ── G4 ───────────────────────────────────────────────────────────────

def test_g4_geometric_mean_exit():
    """G4 — three identical classes (0.5, 0.4, 0.1), one outlier
    (0.001, 0.001, 0.998), λ=0 so q* = G̃. Both Field 1 and Field 2
    False; mechanism is "geometric_mean_exit".

    V0-verified values:
      G̃ = (0.2835, 0.2398, 0.4766)
      hull_signed_distance ≈ -0.0101 (point-to-segment distance,
        unclamped t = 0.5788 interior to [0, 1])
    """
    print("\n[G4] near-degenerate hull, geometric mean exit")
    p = np.array([
        [0.500, 0.400, 0.100],
        [0.500, 0.400, 0.100],
        [0.500, 0.400, 0.100],
        [0.001, 0.001, 0.998],
    ])
    G_tilde = gd.compute_g_tilde(p)
    print(f"      G̃ = ({G_tilde[0]:.4f}, {G_tilde[1]:.4f}, {G_tilde[2]:.4f})")
    # Spec's reference: (0.284, 0.240, 0.477)
    failures = []
    G_spec = np.array([0.284, 0.240, 0.477])
    if np.max(np.abs(G_tilde - G_spec)) > 1e-3:
        failures.append(
            f"G̃ deviation from spec reference > 1e-3: "
            f"computed={tuple(round(v,4) for v in G_tilde)}, "
            f"spec={tuple(G_spec)}"
        )

    p_a = G_tilde.copy()  # λ=0 makes anchor irrelevant
    q_star = ls.solve_iprojection_closed_form(p, p_a, lam=0.0)

    diag = gd.compute_geometric_diagnostics(q_star, p, G_tilde=G_tilde)
    print(f"      diag: inside={diag['q_star_inside_class_hull']}, "
            f"G̃_inside={diag['geometric_mean_inside_class_hull']}, "
            f"degenerate={diag['hull_degenerate']}, "
            f"signed_dist={diag['hull_signed_distance']:+.4f}")
    print(f"      mechanism: {diag['hull_exit_mechanism']}")

    if diag['q_star_inside_class_hull']:
        failures.append("q_star_inside_class_hull should be False")
    if diag['geometric_mean_inside_class_hull']:
        failures.append("geometric_mean_inside_class_hull should be False")
    if diag['hull_exit_mechanism'] != 'geometric_mean_exit':
        failures.append(
            f"hull_exit_mechanism should be 'geometric_mean_exit', "
            f"got '{diag['hull_exit_mechanism']}'"
        )
    # V0-verified: signed_dist ≈ -0.0101 with tolerance ±0.001
    if abs(diag['hull_signed_distance'] - (-0.010)) > 0.001:
        failures.append(
            f"hull_signed_distance should be -0.010 ± 0.001, got "
            f"{diag['hull_signed_distance']:.4f}"
        )
    if not diag['hull_degenerate']:
        failures.append(
            "hull_degenerate should be True (segment-degenerate hull)"
        )
    return failures


# ── Driver ───────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("Module 2 — geometric diagnostics G1–G4 validation")
    print("=" * 68)

    all_failures = []
    for tier_name, fn in [
        ('G1', test_g1_identical_classes),
        ('G2', test_g2_self_consistent_anchor),
        ('G3', test_g3_strong_anchor_pull),
        ('G4', test_g4_geometric_mean_exit),
    ]:
        failures = fn()
        if failures:
            all_failures.append((tier_name, failures))
            print(f"\n      [{tier_name} FAIL]")
            for f in failures:
                print(f"        {f}")
        else:
            print(f"\n      [{tier_name} PASS]")

    print("\n" + "=" * 68)
    if all_failures:
        print(f"FAIL — {len(all_failures)} tier(s) failed.")
        return 1
    print("PASS — all four tiers (G1, G2, G3, G4).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
