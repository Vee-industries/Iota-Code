"""
test_ship4_split_pca_selection_and_reliability.py — verifies Ship 4.

Ship 4 (already built by earlier turn — this test verifies):
  1. run_split_pca_selection is importable, _knn_mi works, generators
     produce sane shapes, _select_operating_point picks correctly under
     synthetic input.
  2. run_knn_mi_reliability is importable, _bootstrap_term returns
     well-formed dict, _load_operating_point falls back gracefully.
  3. export_stats._CALIBRATION_SCRIPTS has 6 entries (v5, ridge_bias,
     toy_nonlinearity, channel_marginal, split_pca_selection,
     knn_mi_reliability).
  4. _load_methodology_calibration emits both new keys.

Run from project root:
    python tests/test_ship4_split_pca_selection_and_reliability.py
"""

import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FoundationsRegistryTests(unittest.TestCase):
    """Foundations registry grew from 4 to 6."""

    def test_six_foundations(self):
        from export_stats import _CALIBRATION_SCRIPTS
        expected = {
            'v5', 'ridge_bias', 'toy_nonlinearity', 'channel_marginal',
            'split_pca_selection', 'knn_mi_reliability',
        }
        self.assertEqual(set(_CALIBRATION_SCRIPTS.keys()), expected)

    def test_upstream_dependencies_correct(self):
        from export_stats import _CALIBRATION_UPSTREAMS
        self.assertEqual(
            _CALIBRATION_UPSTREAMS['split_pca_selection'],
            ['v5_synthetic_calibration.py'])
        self.assertEqual(
            _CALIBRATION_UPSTREAMS['knn_mi_reliability'],
            ['v5_synthetic_calibration.py',
             'run_split_pca_selection.py'])

    def test_methodology_calibration_emits_new_keys(self):
        from export_stats import _load_methodology_calibration
        block = _load_methodology_calibration()
        self.assertIn('split_pca_selection', block)
        self.assertIn('knn_mi_reliability', block)


class SplitPcaSelectionTests(unittest.TestCase):
    """run_split_pca_selection module is sane."""

    def test_module_importable(self):
        import run_split_pca_selection
        self.assertTrue(hasattr(run_split_pca_selection, 'main'))
        self.assertTrue(hasattr(run_split_pca_selection, '_knn_mi'))
        self.assertTrue(hasattr(run_split_pca_selection,
            '_select_operating_point'))

    def test_knn_mi_on_independent_data_near_zero(self):
        """KSG-MI on independent X, Y should be near zero (small positive
        bias is expected for finite n; aim < 0.1 nats at n=500)."""
        from run_split_pca_selection import _knn_mi
        rng = np.random.RandomState(42)
        x = rng.randn(500, 1)
        y = rng.randn(500, 1)
        mi = _knn_mi(x, y, k=5)
        self.assertLess(abs(mi), 0.15,
            f"MI on independent data should be ~0; got {mi}")

    def test_knn_mi_on_dependent_data_positive(self):
        """KSG-MI on Y = X + noise should be substantially positive."""
        from run_split_pca_selection import _knn_mi
        rng = np.random.RandomState(42)
        x = rng.randn(500, 1)
        y = x + 0.3 * rng.randn(500, 1)
        mi = _knn_mi(x, y, k=5)
        self.assertGreater(mi, 0.5,
            f"MI on Y=X+small_noise should be large; got {mi}")

    def test_v5e_generator_shape(self):
        from run_split_pca_selection import _generate_v5e_for_grid
        S_prev, E, S_next = _generate_v5e_for_grid(1000, alpha=0.5, seed=42)
        self.assertEqual(S_prev.shape, (1000, 1))
        self.assertEqual(E.shape, (1000, 1))
        self.assertEqual(S_next.shape, (1000, 1))

    def test_select_operating_point_picks_largest_passing(self):
        """Synthetic grid where 70/30 + 24 passes and 80/20 + 24 fails:
        selector picks 70/30+24."""
        from run_split_pca_selection import _select_operating_point
        # Build a fake grid_results with deliberately structured errors:
        # - 80/20 cells: median 0.15 (FAIL)
        # - 70/30 cells: median 0.05, p95 0.12 (PASS)
        # - 60/40 cells: median 0.04, p95 0.10 (PASS — but smaller split)
        grid_results = []
        for split in [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]:
            for pca in [16, 24, 32, 48]:
                if split == (0.8, 0.2):
                    errs = [0.15, 0.18, 0.20]  # FAIL p95
                elif split == (0.7, 0.3):
                    errs = [0.04, 0.05, 0.06]  # PASS
                else:
                    errs = [0.03, 0.04, 0.05]  # PASS, smaller split
                grid_results.append({
                    'split': list(split),
                    'pca':   pca,
                    'system': 'fake',
                    'per_term': {'fake_term': {'errors_per_repeat': errs,
                                                 'mean_abs_error': float(np.mean(errs))}},
                })
        selected, _ = _select_operating_point(grid_results)
        self.assertIsNotNone(selected, "should have picked a passing cell")
        self.assertEqual(selected['split'], [0.7, 0.3],
            f"expected 70/30 (largest passing); got {selected['split']}")


class KnnMiReliabilityTests(unittest.TestCase):
    """run_knn_mi_reliability module is sane."""

    def test_module_importable(self):
        import run_knn_mi_reliability
        self.assertTrue(hasattr(run_knn_mi_reliability, 'main'))
        self.assertTrue(hasattr(run_knn_mi_reliability, '_knn_mi'))
        self.assertTrue(hasattr(run_knn_mi_reliability, '_bootstrap_term'))
        self.assertTrue(hasattr(run_knn_mi_reliability,
            '_load_operating_point'))

    def test_load_operating_point_fallback(self):
        """Fallback to (0.7, 32) when selection.json missing."""
        from unittest.mock import patch
        import run_knn_mi_reliability as _r
        with patch.object(_r, 'SELECTION_PATH',
                            '/nonexistent/path/selection.json'):
            split, pca = _r._load_operating_point()
        self.assertEqual(split, 0.7)
        self.assertEqual(pca, 32)

    def test_bootstrap_term_returns_well_formed(self):
        """_bootstrap_term returns a dict with the right keys."""
        from run_knn_mi_reliability import _bootstrap_term
        rng = np.random.RandomState(42)
        X = rng.randn(500, 2)
        y = X[:, :1] + 0.3 * rng.randn(500, 1)
        out = _bootstrap_term(
            X, y, 'I(Y;E)',
            idx_func_x=lambda x: x[:, :1],
            n_subsample=300, n_bootstrap=3,
            truth_value=0.7, seed=42,
        )
        self.assertIn('median_rel_err', out)
        self.assertIn('p95_rel_err', out)
        self.assertIn('n_valid', out)
        self.assertGreaterEqual(out['n_valid'], 0)


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(FoundationsRegistryTests),
        loader.loadTestsFromTestCase(SplitPcaSelectionTests),
        loader.loadTestsFromTestCase(KnnMiReliabilityTests),
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
