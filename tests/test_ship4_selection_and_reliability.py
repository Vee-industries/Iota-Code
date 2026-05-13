"""
test_ship4_selection_and_reliability.py — verifies Ship 4.

Ship 4 changes:
  1. run_split_pca_selection.py — new foundation; runs grid of (split, pca)
     on V5e+V5f, picks largest split where median<10% AND p95<20%.
  2. run_knn_mi_reliability.py — new foundation; bootstrap kNN-MI at the
     selected operating point, reports per-term median + p95 rel error.
  3. export_stats._CALIBRATION_SCRIPTS extended with both as foundations.
  4. _load_methodology_calibration extended to surface both artifacts.

Tests:
  - kNN-MI estimator returns finite values on a Gaussian known-answer
    pair where MI is analytically computable.
  - The selection loop runs through --quick mode without crashing and
    writes selection.json.
  - The reliability loop runs through --quick mode and writes
    reliability.json.
  - Foundations registry has both new entries.
  - _load_methodology_calibration includes both keys in the returned block.

Run from project root:
    python tests/test_ship4_selection_and_reliability.py
"""

import os
import sys
import unittest
import numpy as np
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class KSGEstimatorTests(unittest.TestCase):
    """Sanity check on the kNN-MI estimator used by both new foundations."""

    def test_independent_pair_low_mi(self):
        """Independent Gaussians: MI should be ~ 0."""
        from run_split_pca_selection import _knn_mi
        rng = np.random.RandomState(42)
        x = rng.standard_normal((1000, 1))
        y = rng.standard_normal((1000, 1))
        mi = _knn_mi(x, y, k=5)
        self.assertLess(abs(mi), 0.1,
            f"independent pair MI should be ~0; got {mi}")

    def test_perfectly_correlated_high_mi(self):
        """y = x: MI is unbounded analytically; KSG returns a large
        finite value. Just check it's positive and bigger than the
        independent case."""
        from run_split_pca_selection import _knn_mi
        rng = np.random.RandomState(42)
        x = rng.standard_normal((1000, 1))
        y = x + 0.01 * rng.standard_normal((1000, 1))  # near-deterministic
        mi = _knn_mi(x, y, k=5)
        self.assertGreater(mi, 1.0,
            f"highly correlated pair MI should be >>0; got {mi}")

    def test_bivariate_gaussian_known_answer(self):
        """Bivariate Gaussian with correlation rho: MI = -0.5 ln(1 - rho²).
        For rho = 0.7, MI = -0.5 * ln(1 - 0.49) ≈ 0.336 nats. KSG should
        be within ~30% (finite-sample bias)."""
        from run_split_pca_selection import _knn_mi
        rng = np.random.RandomState(42)
        rho = 0.7
        x = rng.standard_normal((2000, 1))
        y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal((2000, 1))
        mi = _knn_mi(x, y, k=5)
        truth = -0.5 * np.log(1 - rho**2)
        self.assertLess(abs(mi - truth) / truth, 0.30,
            f"KSG MI should be within 30% of analytic for rho=0.7: "
            f"got {mi}, analytic {truth}")


class SelectionRunTests(unittest.TestCase):
    """run_split_pca_selection.main() goes end-to-end in --quick mode."""

    def test_selection_quick_writes_output(self):
        with tempfile.TemporaryDirectory() as tmpd:
            sys.argv = ['run_split_pca_selection.py',
                        '--out', tmpd, '--quick']
            from run_split_pca_selection import main as _main
            try:
                _main()
            except SystemExit as e:
                # main returns int via sys.exit (0 or 1); both are fine
                # here, we just want to know the run completed without
                # crashing.
                self.assertIn(e.code, (0, 1, None))

            # Verify selection.json was written
            out_path = os.path.join(tmpd, 'selection.json')
            self.assertTrue(os.path.exists(out_path))
            import json
            with open(out_path) as f:
                sel = json.load(f)
            for key in ('grid', 'tolerance', 'cell_summary', 'grid_results',
                        'selected_operating_point', 'metadata'):
                self.assertIn(key, sel)
            # selected_operating_point has split + pca
            sp = sel['selected_operating_point']
            self.assertIn('split', sp)
            self.assertIn('pca', sp)
            # cache_key.json was also written
            self.assertTrue(os.path.exists(
                os.path.join(tmpd, '.cache_key.json')))


class ReliabilityRunTests(unittest.TestCase):
    """run_knn_mi_reliability.main() runs end-to-end in --quick mode."""

    def test_reliability_quick_writes_output(self):
        with tempfile.TemporaryDirectory() as tmpd:
            sys.argv = ['run_knn_mi_reliability.py',
                        '--out', tmpd, '--quick']
            from run_knn_mi_reliability import main as _main
            try:
                _main()
            except SystemExit as e:
                self.assertIn(e.code, (0, 1, None))
            out_path = os.path.join(tmpd, 'reliability.json')
            self.assertTrue(os.path.exists(out_path))
            import json
            with open(out_path) as f:
                rel = json.load(f)
            for key in ('description', 'operating_point', 'systems',
                        'summary', 'metadata'):
                self.assertIn(key, rel)
            self.assertIn('v5e', rel['systems'])
            self.assertIn('v5f', rel['systems'])


class CalibrationRegistryTests(unittest.TestCase):
    """Foundations registry has both new entries."""

    def test_calibration_scripts_registered(self):
        from export_stats import _CALIBRATION_SCRIPTS
        self.assertIn('split_pca_selection', _CALIBRATION_SCRIPTS,
            "split_pca_selection should be in _CALIBRATION_SCRIPTS")
        self.assertIn('knn_mi_reliability', _CALIBRATION_SCRIPTS,
            "knn_mi_reliability should be in _CALIBRATION_SCRIPTS")
        # Six total now: v5 / ridge_bias / toy_nonlinearity / channel_marginal
        # + split_pca_selection + knn_mi_reliability
        self.assertEqual(len(_CALIBRATION_SCRIPTS), 6,
            f"expected 6 foundations entries, got {len(_CALIBRATION_SCRIPTS)}")

    def test_upstream_chain(self):
        from export_stats import _CALIBRATION_UPSTREAMS
        # split_pca_selection depends on v5_synthetic_calibration.py
        self.assertIn('split_pca_selection', _CALIBRATION_UPSTREAMS)
        self.assertIn('v5_synthetic_calibration.py',
                       _CALIBRATION_UPSTREAMS['split_pca_selection'])
        # knn_mi_reliability depends on both
        self.assertIn('knn_mi_reliability', _CALIBRATION_UPSTREAMS)
        ups = _CALIBRATION_UPSTREAMS['knn_mi_reliability']
        self.assertIn('v5_synthetic_calibration.py', ups)
        self.assertIn('run_split_pca_selection.py', ups)

    def test_load_methodology_calibration_block_keys(self):
        """_load_methodology_calibration returns block with new keys
        seeded as None when artifacts absent."""
        from export_stats import _load_methodology_calibration
        block = _load_methodology_calibration()
        # Both new keys present (will be None unless artifacts exist).
        self.assertIn('split_pca_selection', block)
        self.assertIn('knn_mi_reliability', block)


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(KSGEstimatorTests),
        loader.loadTestsFromTestCase(SelectionRunTests),
        loader.loadTestsFromTestCase(ReliabilityRunTests),
        loader.loadTestsFromTestCase(CalibrationRegistryTests),
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
