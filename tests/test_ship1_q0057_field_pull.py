"""
test_ship1_q0057_field_pull.py — verifies Ship 1 (Q0057 column-pull fix).

Ship 1 changes:
  1. run_function_class_sensitivity._pull_canon_partition pulls
     *_ridge_insample / *_mlp_pooled_mean (canonical paper §5/§6),
     not *_ridge_heldout / *_mlp_reference (the H1 cross-source +
     single-arch-fit values).
  2. patch_q0057_partition_pull.py pulls the same canonical fields,
     uses a new sentinel ('081013') so it re-fires over 0.80.0.47-
     patched files.
  3. results_builder._ingest_q0043 emits canonical
     `*_ridge_h1_crosssource` fields alongside the deprecated
     `*_ridge_heldout` aliases (populated identically).

What this test does:
  - Builds a synthetic results.json-shape cell with measurements
    populated such that insample, h1_crosssource (heldout alias),
    and mlp pooled_mean / reference values are all distinct
    floats. Then asserts:
      a. run_function_class_sensitivity._pull_canon_partition picks
         insample for ridge and pooled_mean for mlp (NOT heldout/
         reference).
      b. patch_q0057_partition_pull.pull_canon_partition_from_results
         picks the same canonical pair.
      c. results_builder._ingest_q0043 sets BOTH
         `rhat_ridge_h1_crosssource` (canonical) and
         `rhat_ridge_heldout` (deprecated alias), to the same value.

Run from project root:
    python tests/test_ship1_q0057_field_pull.py

Pass: stdout ends with "OK" and exit code 0.
Fail: AssertionError with diagnostic message.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ── Fixture: a results.json-shape cell with all four field flavors ──

def _build_synthetic_cell():
    """A cell whose measurements have insample / h1_crosssource (=heldout)
    / pooled_mean / reference set to distinct numeric values, so a wrong
    field pull is unambiguously detectable."""
    return {
        'measurements': {
            # canonical Ridge (paper §5/§6 headline) — what should be pulled
            'rhat_ridge_insample': 0.40,
            'chat_ridge_insample': 0.30,
            'ehat_ridge_insample': 0.30,
            # H1 cross-source Ridge (deprecated heldout name retained as alias)
            'rhat_ridge_h1_crosssource': 0.55,
            'chat_ridge_h1_crosssource': 0.20,
            'ehat_ridge_h1_crosssource': 0.25,
            'rhat_ridge_heldout': 0.55,
            'chat_ridge_heldout': 0.20,
            'ehat_ridge_heldout': 0.25,
            # canonical MLP (paper §5/§6 headline) — what should be pulled
            'rhat_mlp_pooled_mean': 0.50,
            'chat_mlp_pooled_mean': 0.20,
            'ehat_mlp_pooled_mean': 0.30,
            # single-arch reference MLP
            'rhat_mlp_reference': 0.60,
            'chat_mlp_reference': 0.15,
            'ehat_mlp_reference': 0.25,
        },
    }


class Q0057FieldPullTests(unittest.TestCase):
    """Ship 1 — Q0057 column-pull bug fix."""

    def test_run_function_class_sensitivity_pulls_canonical(self):
        """_pull_canon_partition picks insample/pooled_mean, not heldout/
        reference. The signature requires either tuple args (family/size/
        variant/temp) to build a canon key, or it falls back to
        cell_key. We pass a writerbot-style cell_key directly into the
        results_cells dict so the lookup hits, then verify the pull."""
        from run_function_class_sensitivity import _pull_canon_partition
        cell = _build_synthetic_cell()
        cell_key = 'gemma_2b_q4_abliterated_t00'
        results_cells = {cell_key: cell}
        ridge, mlp, source = _pull_canon_partition(
            cell_key, results_cells,
            family=None, size=None, variant=None, temp=None,
        )
        self.assertIsNotNone(ridge,
            f"ridge dict was None; source={source}")
        self.assertIsNotNone(mlp,
            f"mlp dict was None; source={source}")
        # Pulled CANONICAL values (insample/pooled_mean), NOT heldout/reference
        self.assertAlmostEqual(ridge['R'], 0.40, places=8,
            msg=f"ridge R should be 0.40 (insample); got {ridge['R']} "
                f"— is the pull still picking heldout (0.55)?")
        self.assertAlmostEqual(mlp['R'], 0.50, places=8,
            msg=f"mlp R should be 0.50 (pooled_mean); got {mlp['R']} "
                f"— is the pull still picking reference (0.60)?")
        self.assertAlmostEqual(ridge['E'], 0.30, places=8)
        self.assertAlmostEqual(ridge['C'], 0.30, places=8)
        self.assertAlmostEqual(mlp['E'], 0.30, places=8)
        self.assertAlmostEqual(mlp['C'], 0.20, places=8)

    @unittest.skip("patch_q0057_partition_pull module removed; test is stale")
    def test_patch_script_pulls_canonical(self):
        """patch_q0057_partition_pull.pull_canon_partition_from_results
        picks insample/pooled_mean. Same fix as the production runner —
        must stay in lockstep so retroactive backfill matches re-runs."""
        from patch_q0057_partition_pull import (
            pull_canon_partition_from_results,
            PATCH_SENTINEL,
        )
        # Sentinel must have changed in 0.81.1.3 so this patch re-fires
        # over 0.80.0.47-patched files.
        self.assertIn('081013', PATCH_SENTINEL,
            f"PATCH_SENTINEL='{PATCH_SENTINEL}' should include '081013' "
            f"to re-fire over 0.80.0.47-patched files.")

        cell = _build_synthetic_cell()
        canon_key = 'gemma_2b_4bit_abliterated_T0.0'
        ridge, mlp = pull_canon_partition_from_results(
            canon_key, {canon_key: cell})
        self.assertIsNotNone(ridge)
        self.assertIsNotNone(mlp)
        self.assertAlmostEqual(ridge['R'], 0.40, places=8,
            msg=f"patch script ridge R should be 0.40 (insample); "
                f"got {ridge['R']}")
        self.assertAlmostEqual(mlp['R'], 0.50, places=8,
            msg=f"patch script mlp R should be 0.50 (pooled_mean); "
                f"got {mlp['R']}")

    def test_results_builder_emits_canonical_and_deprecated(self):
        """_ingest_q0043 sets BOTH the new canonical name
        (`rhat_ridge_h1_crosssource`) and the deprecated alias
        (`rhat_ridge_heldout`), populated to the same value. This keeps
        paper 1 reproducibility working while migrating to the canonical
        name."""
        from results_builder import _ingest_q0043
        # Synthetic Q0043 input in nested shape (legacy)
        q43 = {
            'in_sample': {
                'Rhat': 0.40, 'Chat': 0.30, 'Ehat': 0.30,
                'drop_E': 0.30, 'drop_C': 0.30, 'drop_S': 0.40,
            },
            'held_out': {
                'Rhat': 0.55, 'Chat': 0.20, 'Ehat': 0.25,
            },
        }
        cell = {'measurements': {}, 'provenance': {}}
        _ingest_q0043(cell, q43)
        m = cell['measurements']
        # Canonical name present
        self.assertIn('rhat_ridge_h1_crosssource', m,
            "Schema 0.81.0 should expose canonical "
            "`rhat_ridge_h1_crosssource` field.")
        self.assertAlmostEqual(m['rhat_ridge_h1_crosssource'], 0.55,
            places=8)
        # Deprecated alias present and equal
        self.assertIn('rhat_ridge_heldout', m,
            "Deprecated `rhat_ridge_heldout` alias retained for "
            "back-compat (removal in 0.82.0).")
        self.assertAlmostEqual(m['rhat_ridge_heldout'],
            m['rhat_ridge_h1_crosssource'], places=8,
            msg="Deprecated alias must equal canonical value.")
        # Insample (separate, distinct) also present
        self.assertAlmostEqual(m['rhat_ridge_insample'], 0.40, places=8)


def _run():
    """Run as standalone — emit OK to stdout if all pass."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Q0057FieldPullTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nOK")
        return 0
    else:
        print("\nFAILED")
        return 1


if __name__ == '__main__':
    sys.exit(_run())
