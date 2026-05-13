"""
test_ship2_schema_and_rkhs.py — verifies Ship 2.

Ship 2 changes:
  1. results_schema.SCHEMA_VERSION bumped 0.80.0 → 0.81.0.
  2. paper2_apparatus_metadata top-level block added to schema (additive).
  3. Q0057 grows RKHS column (rkhs_median in permutation_shares).
  4. _build_cell_entry accepts rkhs_result and emits four-class shape.
  5. run_kraskov_anchor._load_function_class_shares reads real RKHS when
     present, falls back to Ridge clone when absent.
  6. run_bayesian_aggregator._load_q57_shares same.

Tests:
  - Schema version bumped.
  - 0.80.0-shape document still validates (additive only).
  - paper2_apparatus_metadata accepted as optional top-level field.
  - _build_cell_entry produces a cell with rkhs_median in
    permutation_shares when rkhs_result provided; with None when not.
  - apparatus loaders pick rkhs_median from permutation_shares when
    present; fall back to Ridge when absent.

Run from project root:
    python tests/test_ship2_schema_and_rkhs.py
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class SchemaBumpTests(unittest.TestCase):
    """Schema 0.80.0 → 0.81.0."""

    def test_schema_version_bumped(self):
        from results_schema import SCHEMA_VERSION
        self.assertEqual(SCHEMA_VERSION, "0.81.0",
            f"SCHEMA_VERSION should be '0.81.0', got '{SCHEMA_VERSION}'")

    def test_minimal_document_validates(self):
        """A minimal 0.80.0-shape document validates under 0.81.0 (additive)."""
        from results_schema import build_schema
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = build_schema()
        doc = {
            "schema_version": "0.81.0",
            "iota_version":   "0.81.1.4",
            "generated_at":   "2026-04-29T00:00:00Z",
            "cells":          {},
            "cross_cell_aggregates": {},
            "synthetic_calibration": None,
            "toy_nonlinearity":      None,
            "methodology_calibration": {},
        }
        jsonschema.validate(instance=doc, schema=schema)

    def test_paper2_apparatus_metadata_optional(self):
        """paper2_apparatus_metadata is optional (absent in 0.80.0 docs).
        When present, valid shape validates."""
        from results_schema import build_schema
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = build_schema()
        # Absent — should validate
        doc_absent = {
            "schema_version": "0.81.0", "iota_version": "0.81.1.4",
            "generated_at": "2026-04-29T00:00:00Z",
            "cells": {}, "cross_cell_aggregates": {},
            "synthetic_calibration": None, "toy_nonlinearity": None,
            "methodology_calibration": {},
        }
        jsonschema.validate(instance=doc_absent, schema=schema)
        # Present with all fields populated — should validate
        doc_present = dict(doc_absent)
        doc_present["paper2_apparatus_metadata"] = {
            "lambda_default": 1.0,
            "lambda_calibration_source": "v5b",
            "split_ratio": [0.7, 0.3],
            "pca_dim_selected": 32,
            "n_anchor": 5000,
            "v5f_construction_artifact_path": "v5f_construction_artifact.md",
        }
        jsonschema.validate(instance=doc_present, schema=schema)
        # Present with null — should validate
        doc_null = dict(doc_absent)
        doc_null["paper2_apparatus_metadata"] = None
        jsonschema.validate(instance=doc_null, schema=schema)


class RKHSCellEntryTests(unittest.TestCase):
    """_build_cell_entry emits four-class shape when rkhs_result provided."""

    def _stub_rf_result(self):
        return {
            'pool_dim': 64,
            'n_rows': 1000,
            'r2_rf_E': 0.20, 'r2_rf_S': 0.30,
            'rf_joint_R2_train': 0.45,
            'rf_joint_R2_held_out': 0.40,
            'rf_partition': {'E': 0.30, 'C': 0.30, 'R': 0.40},
            'rf_partition_raw_drops': {'E': 0.05, 'C': 0.05, 'R': 0.07},
            'native_pool_dim': 64,
            'rf_target_dim': 64,
            'rf_target_pca_applied': False,
            'rf_target_pca_explained_variance_ratio': None,
        }

    def _stub_rkhs_result(self):
        return {
            'pool_dim': 64,
            'n_rows': 1000,
            'n_train_subsampled': 5000,
            'median_pairwise_distance': 11.5,
            'rkhs_median_r2_test': 0.42,
            'rkhs_median_partition': {'E': 0.32, 'C': 0.28, 'R': 0.40},
            'rkhs_range_per_channel': {
                'E': {'min': 0.28, 'max': 0.36},
                'C': {'min': 0.24, 'max': 0.32},
                'R': {'min': 0.36, 'max': 0.44},
            },
            'rkhs_per_kernel': [
                {'kernel_id': 'matern_nu0.5', 'r2_test': 0.40,
                 'partition': {'E': 0.30, 'C': 0.28, 'R': 0.42}},
            ],
            'native_pool_dim': 64,
            'rkhs_target_dim': 64,
            'rkhs_target_pca_applied': False,
            'rkhs_target_pca_explained_variance_ratio': None,
        }

    def test_build_cell_entry_with_rkhs(self):
        from run_function_class_sensitivity import _build_cell_entry
        canon_csv_row = {
            'r2_ridge_E': 0.10, 'r2_ridge_S': 0.15,
            'r2_mlp_E':   0.20, 'r2_mlp_S':   0.30,
        }
        ridge_partition = {'E': 0.40, 'C': 0.30, 'R': 0.30}
        mlp_partition   = {'E': 0.35, 'C': 0.25, 'R': 0.40}
        cell = _build_cell_entry(
            'gemma', '2b', 'abliterated', 'temp_0', 0.0,
            self._stub_rf_result(), canon_csv_row,
            ridge_partition, mlp_partition,
            'pulled_from_results_json (test)',
            rkhs_result=self._stub_rkhs_result(),
        )
        self.assertNotIn('error', cell)
        # Four-class permutation_shares
        ps = cell['permutation_shares']
        self.assertIn('ridge', ps)
        self.assertIn('mlp', ps)
        self.assertIn('rf', ps)
        self.assertIn('rkhs_median', ps,
            "permutation_shares should include rkhs_median in 0.81.1.4")
        self.assertIsNotNone(ps['rkhs_median'])
        self.assertAlmostEqual(ps['rkhs_median']['R'], 0.40, places=8)
        # rkhs_kernel_grid block populated
        rkg = cell.get('rkhs_kernel_grid')
        self.assertIsNotNone(rkg, "rkhs_kernel_grid should be populated")
        self.assertEqual(rkg['n_train_subsampled'], 5000)
        self.assertIn('per_kernel', rkg)

    def test_build_cell_entry_without_rkhs(self):
        """Backwards compatible: rkhs_result=None produces null rkhs_median."""
        from run_function_class_sensitivity import _build_cell_entry
        canon_csv_row = {
            'r2_ridge_E': 0.10, 'r2_ridge_S': 0.15,
            'r2_mlp_E':   0.20, 'r2_mlp_S':   0.30,
        }
        cell = _build_cell_entry(
            'gemma', '2b', 'abliterated', 'temp_0', 0.0,
            self._stub_rf_result(), canon_csv_row,
            {'E': 0.40, 'C': 0.30, 'R': 0.30},
            {'E': 0.35, 'C': 0.25, 'R': 0.40},
            'pulled_from_results_json (test)',
            rkhs_result=None,
        )
        ps = cell['permutation_shares']
        self.assertIn('rkhs_median', ps)
        self.assertIsNone(ps['rkhs_median'])


class ApparatusLoaderTests(unittest.TestCase):
    """Apparatus loaders pick real RKHS when present."""

    def test_kraskov_anchor_loader_uses_real_rkhs(self):
        """_load_function_class_shares reads rkhs_median field when present;
        Ridge fallback when absent."""
        # Build a synthetic Q0057 in a temp dir, point the loader at it
        import json
        import tempfile
        from unittest.mock import patch
        # Cell with real rkhs_median present
        q57_with_rkhs = {
            'function_classes': ['ridge', 'mlp', 'rf', 'rkhs_median'],
            'cells': {
                'gemma_2b_4bit_abliterated_T0.0': {
                    'pool_dim': 64,
                    'permutation_shares': {
                        'ridge': {'E': 0.40, 'C': 0.30, 'R': 0.30},
                        'mlp':   {'E': 0.35, 'C': 0.25, 'R': 0.40},
                        'rf':    {'E': 0.30, 'C': 0.30, 'R': 0.40},
                        'rkhs_median': {'E': 0.32, 'C': 0.28, 'R': 0.40},
                    },
                },
            },
        }
        # Cell with no rkhs_median (transitional state)
        q57_no_rkhs = {
            'function_classes': ['ridge', 'mlp', 'rf'],
            'cells': {
                'gemma_2b_4bit_abliterated_T0.0': {
                    'pool_dim': 64,
                    'permutation_shares': {
                        'ridge': {'E': 0.40, 'C': 0.30, 'R': 0.30},
                        'mlp':   {'E': 0.35, 'C': 0.25, 'R': 0.40},
                        'rf':    {'E': 0.30, 'C': 0.30, 'R': 0.40},
                    },
                },
            },
        }

        for label, q57_obj, expected_rkhs_R in [
            ('with rkhs_median', q57_with_rkhs, 0.40),
            ('no rkhs_median (Ridge fallback)', q57_no_rkhs, 0.30),
        ]:
            with tempfile.TemporaryDirectory() as tmpd:
                p_dir = os.path.join(tmpd, 'paper')
                os.makedirs(p_dir)
                with open(os.path.join(p_dir, 'Q0057_function_class_sensitivity.json'), 'w') as f:
                    json.dump(q57_obj, f)
                # Patch the DATA constant the loader uses
                from run_kraskov_anchor import _load_function_class_shares
                import run_kraskov_anchor as _rka
                with patch.object(_rka, 'DATA', tmpd):
                    shares = _load_function_class_shares(
                        'gemma_2b_4bit_abliterated_T0.0')
                self.assertIsNotNone(shares,
                    f"loader returned None for case '{label}'")
                self.assertAlmostEqual(shares['rkhs'][2], expected_rkhs_R,
                    places=8,
                    msg=f"case '{label}': rkhs[R] expected "
                        f"{expected_rkhs_R}, got {shares['rkhs'][2]}")

    def test_aggregator_loader_uses_real_rkhs(self):
        """run_bayesian_aggregator._load_q57_shares same."""
        import json
        import tempfile
        from unittest.mock import patch
        q57 = {
            'function_classes': ['ridge', 'mlp', 'rf', 'rkhs_median'],
            'cells': {
                'gemma_2b_4bit_abliterated_T0.0': {
                    'pool_dim': 64,
                    'permutation_shares': {
                        'ridge': {'E': 0.40, 'C': 0.30, 'R': 0.30},
                        'mlp':   {'E': 0.35, 'C': 0.25, 'R': 0.40},
                        'rf':    {'E': 0.30, 'C': 0.30, 'R': 0.40},
                        'rkhs_median': {'E': 0.32, 'C': 0.28, 'R': 0.40},
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpd:
            p_dir = os.path.join(tmpd, 'paper')
            os.makedirs(p_dir)
            with open(os.path.join(p_dir, 'Q0057_function_class_sensitivity.json'), 'w') as f:
                json.dump(q57, f)
            from run_bayesian_aggregator import _load_q57_shares
            import run_bayesian_aggregator as _rla
            with patch.object(_rla, 'DATA', tmpd):
                shares = _load_q57_shares()
            self.assertIn('gemma_2b_4bit_abliterated_T0.0', shares)
            cell_shares = shares['gemma_2b_4bit_abliterated_T0.0']
            self.assertAlmostEqual(cell_shares['rkhs'][2], 0.40, places=8)


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(SchemaBumpTests),
        loader.loadTestsFromTestCase(RKHSCellEntryTests),
        loader.loadTestsFromTestCase(ApparatusLoaderTests),
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
