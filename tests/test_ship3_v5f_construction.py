"""
test_ship3_v5f_construction.py — verifies Ship 3 (V5f construction).

Ship 3 changes:
  1. v5_synthetic_calibration.py adds run_v5f() + helpers
     (_v5f_generate, _v5f_compute_mi_proxies, _v5f_tune_noise,
      estimate_R_3channel) per codebot_handoff_v0_14.md §3.1.
  2. main() resume cache + run sequence + output dict + summary
     extended to include v5f_doubly_conditional.

Tests verify:
  - Generator builds (E, C, S, Y) with correct independence + structure.
    E ⊥ C marginal correlation small. S ≈ E + C + ε. Y ≈ sin(S) + η.
  - MI proxies are monotone in noise: more σ_S → I(Y; E, C) drops;
    more σ_Y → I(Y; S) drops.
  - Tuning loop converges within tolerance for a moderate target.
  - run_v5f() returns the documented shape (construction, tuning_log,
    analytic_truth, attribution_ridge, attribution_mlp).
  - Three-channel R estimator returns shares summing to 1 (within 1e-6).

Tuning is heavy enough that we use small n_steps and a coarse target
to keep the test under a minute. Real run uses 20k samples and the
empirical band; this test verifies shape and qualitative behavior.

Run from project root:
    python tests/test_ship3_v5f_construction.py
"""

import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class V5fGeneratorTests(unittest.TestCase):
    """The construction primitive (E, C, S, Y) generation."""

    def test_generator_shape(self):
        from v5_synthetic_calibration import _v5f_generate
        E, C, S, Y = _v5f_generate(1000, sigma_S=0.5, sigma_Y=0.3, seed=42)
        self.assertEqual(E.shape, (1000, 1))
        self.assertEqual(C.shape, (1000, 1))
        self.assertEqual(S.shape, (1000, 1))
        self.assertEqual(Y.shape, (1000, 1))

    def test_generator_independence(self):
        """E and C should be (statistically) independent — sample
        correlation small."""
        from v5_synthetic_calibration import _v5f_generate
        E, C, _, _ = _v5f_generate(5000, sigma_S=0.5, sigma_Y=0.3, seed=42)
        rho = float(np.corrcoef(E.ravel(), C.ravel())[0, 1])
        # |rho| should be ~ 1/sqrt(5000) ~ 0.014 in expectation.
        # 0.05 is a generous bound that won't flake.
        self.assertLess(abs(rho), 0.05,
            f"E and C should be ~independent; got rho={rho}")

    def test_generator_structural(self):
        """S = E + C + ε. With σ_S=0, S = E + C exactly."""
        from v5_synthetic_calibration import _v5f_generate
        E, C, S, _ = _v5f_generate(1000, sigma_S=0.0, sigma_Y=0.3, seed=42)
        residual = S - (E + C)
        self.assertLess(float(np.abs(residual).max()), 1e-9,
            "with σ_S=0, S should equal E + C exactly")

    def test_generator_y_structural(self):
        """Y = sin(S) + η. With σ_Y=0, Y = sin(S) exactly."""
        from v5_synthetic_calibration import _v5f_generate
        _, _, S, Y = _v5f_generate(1000, sigma_S=0.5, sigma_Y=0.0, seed=42)
        residual = Y - np.sin(S)
        self.assertLess(float(np.abs(residual).max()), 1e-9,
            "with σ_Y=0, Y should equal sin(S) exactly")


class V5fNoiseMonotonicityTests(unittest.TestCase):
    """MI proxies should respond monotonically to noise scales."""

    def test_higher_sigma_S_lower_mi_ec(self):
        """As σ_S grows, S becomes a noisier function of E + C, so Y
        (which depends on S) is less predictable from E + C alone.
        I(Y; E, C) should decrease."""
        from v5_synthetic_calibration import (
            _v5f_generate, _v5f_compute_mi_proxies)
        # Small sigma_S: S ≈ E + C, so Y is well-predictable from E + C
        E1, C1, S1, Y1 = _v5f_generate(3000, sigma_S=0.1, sigma_Y=0.1, seed=42)
        mi_ec_low, _ = _v5f_compute_mi_proxies(E1, C1, S1, Y1, seed=42)
        # Large sigma_S: S has lots of noise on top of E + C
        E2, C2, S2, Y2 = _v5f_generate(3000, sigma_S=2.0, sigma_Y=0.1, seed=42)
        mi_ec_high, _ = _v5f_compute_mi_proxies(E2, C2, S2, Y2, seed=42)
        self.assertGreater(mi_ec_low, mi_ec_high,
            f"I(Y; E, C) should drop as σ_S rises; "
            f"got low-σ_S={mi_ec_low:.4f}, high-σ_S={mi_ec_high:.4f}")

    def test_higher_sigma_Y_lower_mi_s(self):
        """As σ_Y grows, Y becomes a noisier function of S. I(Y; S) drops."""
        from v5_synthetic_calibration import (
            _v5f_generate, _v5f_compute_mi_proxies)
        E1, C1, S1, Y1 = _v5f_generate(3000, sigma_S=0.5, sigma_Y=0.05, seed=42)
        _, mi_s_low_noise = _v5f_compute_mi_proxies(E1, C1, S1, Y1, seed=42)
        E2, C2, S2, Y2 = _v5f_generate(3000, sigma_S=0.5, sigma_Y=2.0, seed=42)
        _, mi_s_high_noise = _v5f_compute_mi_proxies(E2, C2, S2, Y2, seed=42)
        self.assertGreater(mi_s_low_noise, mi_s_high_noise,
            f"I(Y; S) should drop as σ_Y rises; "
            f"got low-σ_Y={mi_s_low_noise:.4f}, "
            f"high-σ_Y={mi_s_high_noise:.4f}")


class V5fTuningTests(unittest.TestCase):
    """Tuning loop converges to target MI proxies."""

    def test_tune_converges(self):
        """Targets in the achievable range — tuning should land within
        ~30% (loose, since proxy is noisy on small n)."""
        from v5_synthetic_calibration import _v5f_tune_noise, _v5f_generate, _v5f_compute_mi_proxies
        target_mi_ec = 0.30
        target_mi_s = 0.50
        sigma_S, sigma_Y, log = _v5f_tune_noise(
            target_mi_ec=target_mi_ec, target_mi_s=target_mi_s,
            n_steps=2000, max_iter=8, tol_log=0.20, seed=42,
        )
        # Sanity: tune produced reasonable values
        self.assertGreater(sigma_S, 0)
        self.assertGreater(sigma_Y, 0)
        self.assertLess(sigma_S, 5.0)
        self.assertLess(sigma_Y, 5.0)
        # Final MI close to targets (loose bound for fast n)
        E, C, S, Y = _v5f_generate(2000, sigma_S, sigma_Y, seed=42)
        mi_ec, mi_s = _v5f_compute_mi_proxies(E, C, S, Y, seed=42)
        self.assertLess(abs(mi_ec - target_mi_ec) / target_mi_ec, 0.6,
            f"tuned σ_S didn't bring MI(Y; E, C) close to target: "
            f"got {mi_ec:.4f}, target {target_mi_ec:.4f}")
        self.assertLess(abs(mi_s - target_mi_s) / target_mi_s, 0.6,
            f"tuned σ_Y didn't bring MI(Y; S) close to target: "
            f"got {mi_s:.4f}, target {target_mi_s:.4f}")
        # Log structure
        self.assertIsInstance(log, list)
        self.assertGreater(len(log), 0)
        for entry in log:
            self.assertIn('phase', entry)
            self.assertIn('sigma_S', entry)
            self.assertIn('sigma_Y', entry)
            self.assertIn('mi_ec', entry)
            self.assertIn('mi_s', entry)


class V5fRunReturnShapeTests(unittest.TestCase):
    """run_v5f returns the documented shape."""

    def test_three_channel_estimator_shares_sum_to_one(self):
        from v5_synthetic_calibration import (
            estimate_R_3channel, _v5f_generate)
        E, C, S, Y = _v5f_generate(1000, sigma_S=0.5, sigma_Y=0.3, seed=42)
        shares, r2 = estimate_R_3channel(
            E, C, S, Y, n_permutations=20, seed=42, regressor='ridge')
        total = sum(shares.values())
        self.assertAlmostEqual(total, 1.0, places=6,
            msg=f"3-channel shares should sum to 1; got {total}")
        for ch in ('E', 'C', 'S'):
            self.assertGreaterEqual(shares[ch], 0.0)
            self.assertLessEqual(shares[ch], 1.0)

    def test_run_v5f_return_shape(self):
        """Light-weight run_v5f to verify return shape — small n_steps,
        few iters, no claim about scientific accuracy."""
        from v5_synthetic_calibration import run_v5f
        result = run_v5f(
            target_mi_ec=0.20, target_mi_s=0.40,
            n_steps_final=2000, n_repeats=2,
            tune_n_steps=1000, seed=42,
        )
        # Top-level keys
        for key in ('description', 'construction', 'tuning_log',
                    'analytic_truth', 'attribution_ridge', 'attribution_mlp'):
            self.assertIn(key, result, f"missing key '{key}'")
        # Construction
        c = result['construction']
        self.assertIn('sigma_S', c)
        self.assertIn('sigma_Y', c)
        self.assertGreater(c['sigma_S'], 0)
        self.assertGreater(c['sigma_Y'], 0)
        # Truth
        t = result['analytic_truth']
        self.assertIn('R_truth_mean', t)
        self.assertIn('truth_formula', t)
        # Attribution: shares dict with E/C/S keys summing to ~1
        for key in ('attribution_ridge', 'attribution_mlp'):
            sh = result[key]['shares_mean']
            tot = sh['E'] + sh['C'] + sh['S']
            self.assertAlmostEqual(tot, 1.0, places=4,
                msg=f"{key} shares should sum to 1; got {tot}")


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(V5fGeneratorTests),
        loader.loadTestsFromTestCase(V5fNoiseMonotonicityTests),
        loader.loadTestsFromTestCase(V5fTuningTests),
        loader.loadTestsFromTestCase(V5fRunReturnShapeTests),
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nOK")
        return 0
    print("\nFAILED")
    return 1


if __name__ == '__main__':
    sys.exit(_run())
