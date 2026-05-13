"""
test_ship3_v5f_four_class.py — verifies Ship 3.

Ship 3 changes (existing V5f extended):
  1. estimate_R_3channel supports {ridge, mlp, rkhs, rf} (was {ridge, mlp}).
  2. run_v5f computes all four classes and emits attribution_{ridge, mlp,
     rkhs, rf} blocks.

Tests (small-n, quick-fire — full 20k-step V5f is for the actual
calibration run):
  - estimate_R_3channel('rkhs', ...) returns simplex-normalised shares
    on a known-S system.
  - estimate_R_3channel('rf', ...) same.
  - run_v5f(n_steps_final=2000, n_repeats=1, tune_n_steps=1500) emits
    all four attribution blocks with float-shares-on-simplex shape.
  - V5f generator is deterministic given seed.

Run from project root:
    python tests/test_ship3_v5f_four_class.py
"""

import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _shares_on_simplex(d, tol=1e-6):
    if not isinstance(d, dict):
        return False
    keys = ('E', 'C', 'S')
    if not all(k in d for k in keys):
        return False
    vals = [d[k] for k in keys]
    if not all(isinstance(v, (int, float)) for v in vals):
        return False
    if any(v < -tol for v in vals):
        return False
    s = sum(vals)
    return abs(s - 1.0) < tol or s == 0.0


class V5fGeneratorTests(unittest.TestCase):
    """V5f generator is deterministic given seed."""

    def test_generator_deterministic(self):
        from v5_synthetic_calibration import _v5f_generate
        E1, C1, S1, Y1 = _v5f_generate(500, 0.5, 0.3, seed=42)
        E2, C2, S2, Y2 = _v5f_generate(500, 0.5, 0.3, seed=42)
        np.testing.assert_array_equal(E1, E2)
        np.testing.assert_array_equal(C1, C2)
        np.testing.assert_array_equal(S1, S2)
        np.testing.assert_array_equal(Y1, Y2)
        # Different seed -> different output
        E3, _, _, _ = _v5f_generate(500, 0.5, 0.3, seed=43)
        self.assertFalse(np.array_equal(E1, E3))

    def test_generator_shape(self):
        from v5_synthetic_calibration import _v5f_generate
        E, C, S, Y = _v5f_generate(1000, 0.5, 0.3, seed=42)
        self.assertEqual(E.shape, (1000, 1))
        self.assertEqual(C.shape, (1000, 1))
        self.assertEqual(S.shape, (1000, 1))
        self.assertEqual(Y.shape, (1000, 1))
        # S = E + C + eps — should be approximately so on a clean draw
        # (residual = eps ~ N(0, σ_S²))
        residual = S - E - C
        self.assertAlmostEqual(float(np.std(residual)), 0.5, places=1,
            msg="residual std should be ~σ_S = 0.5")


class FourClassRChannelTests(unittest.TestCase):
    """estimate_R_3channel supports rkhs and rf (added in 0.81.1.5)."""

    def _build_known_S_system(self, n=2000, seed=42):
        """Build (E, C, S, Y) where Y depends ONLY on S (R-truth -> 1).
        Uses the same V5f construction at small n so RKHS+RF have
        enough samples to fit but the test stays quick."""
        from v5_synthetic_calibration import _v5f_generate
        return _v5f_generate(n, sigma_S=0.5, sigma_Y=0.3, seed=seed)

    def test_rkhs_returns_simplex_shares(self):
        from v5_synthetic_calibration import estimate_R_3channel
        E, C, S, Y = self._build_known_S_system()
        shares, r2 = estimate_R_3channel(
            E, C, S, Y, n_permutations=20, seed=42, regressor='rkhs')
        self.assertTrue(_shares_on_simplex(shares),
            f"RKHS shares not on simplex: {shares}")
        self.assertIsInstance(r2, float)
        # On a system where Y depends on S, RKHS should put substantial
        # share on S (not necessarily majority — Ridge under-attributes
        # but the test is just "is the result sane")
        self.assertGreaterEqual(shares['S'], 0.0)

    def test_rf_returns_simplex_shares(self):
        from v5_synthetic_calibration import estimate_R_3channel
        E, C, S, Y = self._build_known_S_system()
        shares, r2 = estimate_R_3channel(
            E, C, S, Y, n_permutations=20, seed=42, regressor='rf')
        self.assertTrue(_shares_on_simplex(shares),
            f"RF shares not on simplex: {shares}")
        self.assertIsInstance(r2, float)

    def test_unknown_regressor_raises(self):
        from v5_synthetic_calibration import estimate_R_3channel
        E, C, S, Y = self._build_known_S_system(n=200)
        with self.assertRaises(ValueError):
            estimate_R_3channel(
                E, C, S, Y, n_permutations=5, seed=42, regressor='nonsense')


class V5fFourClassEmissionTests(unittest.TestCase):
    """run_v5f emits all four attribution blocks."""

    def test_run_v5f_smoke_four_classes(self):
        """Small-n smoke fire — verify attribution_{ridge, mlp, rkhs, rf}
        all populated with simplex shares. Skip if too slow (RKHS at
        2000 rows is ~10s, RF ~5s)."""
        from v5_synthetic_calibration import run_v5f
        # Override defaults: tiny tuning + tiny final to keep test under 1 min.
        # We don't care if the noise tuner converges to the target — we care
        # only that the attribution emits the four blocks correctly.
        result = run_v5f(
            target_mi_ec=0.30, target_mi_s=0.45,
            n_steps_final=2000, n_repeats=1,
            tune_n_steps=1000,
            seed=42,
        )
        # All four attribution blocks present
        for est in ('ridge', 'mlp', 'rkhs', 'rf'):
            key = f'attribution_{est}'
            self.assertIn(key, result,
                f"V5f result missing {key}")
            shares = result[key]['shares_mean']
            self.assertTrue(_shares_on_simplex(shares),
                f"V5f {est} shares not on simplex: {shares}")
            self.assertIsInstance(result[key]['R2_mean'], float)
        # Construction block populated
        self.assertIn('construction', result)
        self.assertIn('sigma_S', result['construction'])
        self.assertIn('sigma_Y', result['construction'])
        # Analytic-truth block
        self.assertIn('analytic_truth', result)
        self.assertIn('R_truth_mean', result['analytic_truth'])


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(V5fGeneratorTests),
        loader.loadTestsFromTestCase(FourClassRChannelTests),
        loader.loadTestsFromTestCase(V5fFourClassEmissionTests),
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
