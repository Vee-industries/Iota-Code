"""
results_schema.py — schema, encoder, validator for data/paper/results.json (v0.81).

Responsibilities:
  - SCHEMA_VERSION constant
  - JSON encoder: numpy scalars → Python, NaN → null, Inf → "inf"/"-inf",
    ndarrays → {"_type": "ndarray", "dtype", "shape", "data"}
  - jsonschema validator with blocking semantics
  - Per-cell flat key builder
  - Schema document (as a Python dict that jsonschema consumes)

Blocking: validate_results_json() raises ValueError on any schema violation.
build_master_results in export_stats.py calls this AFTER constructing the
aggregate but BEFORE writing to disk. Invalid structure never ships.

v0.81.0 additions (additive only — 0.80.0 documents validate cleanly):
  Top-level:
    - paper2_apparatus_metadata: lambda_default, lambda_calibration_source,
      split_ratio, pca_dim_selected, n_anchor, v5f_construction_artifact_path
  Per-cell measurements (paper 2 measurement layer):
    - rf_shares, rkhs_shares (the four-class lineup, real RKHS not placeholder)
    - anchored_shares (Ridge / MLP / RKHS_median / RF, four-class anchored)
    - geometric_diagnostics (eight-field block on q* + class shares:
      hull membership, anchor pull, hull volume, exit mechanism, signed
      distance — replaces v0.82.0.23 convex_hull_diagnostic per
      statsbot's Module 2 spec). v0.82.0.23: optional
      bootstrap_variance_decomposition sub-field populated when
      Phase 10's per_resample_shares are available — surfaces the
      apparatus's coupled-stabilization signature (q* anti-correlated
      with hull boundary movement under bootstrap resampling, with
      hull-only variance exceeding observed variance — the I-projection
      MAP aggregator damping hull-distance variance through the
      geometric blend).
    - partition_b_substitution_diagnostics (v0.82.0.23): per-cell join
      of Q0057 in-sample shares, partition-B held-out shares (Phase 6
      output), and Phase 7 anchor. Surfaces §9.2 evidence (per-class
      ΔE/ΔC/ΔR; abs_dC_by_class; strict_ordering_holds and
      two_tier_ordering_holds — the fleet-wide two-tier claim is
      empirically supported on 24/24 cells, the strict four-way ordering
      Ridge>MLP>RKHS>RF holds on 13/24) and §9.3 C-deflation evidence
      (g_tilde_pre/post; g_tilde_shift_dC_dominance — fleet mean 0.75
      with |ΔC| dominating G̃_shift on 22/24; kl_pre/post_anchor +
      kl_delta_anchor; anchor_alignment_cosine retained as
      discrimination record for the original 'anchor-aligned overfit'
      mechanism that Phase 7 Step 3's p_a,C = G̃_C construction ruled
      out by design).
    - estimator_joint_R2 (v0.82.0.23): per-estimator joint R² (train +
      held-out) for Ridge, MLP, RF, RKHS_median at the apparatus's
      canonical operating point (partition-B, split=0.8, pca=48,
      PARTITION_B_SEED=42, RIDGE_ALPHA=0.01, MLP_HYPERS including
      canon tanh activation). These are the canonical linearity-axis
      position values for the recoverability plane (paper 1 §5).
      Related but DISTINCT field:
      cells.<key>.measurements.linearity_max_gap captures Q0042
      collection-time architecture sweep (3 MLP configs vs Ridge,
      ReLU activation, in-sample fit) used as a sensitivity check
      that the canonical-config R² values are not architecture
      artifacts. linearity_max_gap is NOT derivable from
      estimator_joint_R2 (different MLP activation, architecture
      sweep vs single config, in-sample vs held-out at canon).
      Both fields are independently meaningful for §5
      recoverability-plane positioning.
    - bootstrap_variance (per-channel variance for each estimator + anchored)
    - stacking_baseline_shares + stacking_baseline_weights (w_R/w_M/w_K/w_F)
    - lambda_sweep (list of (lambda, q_star) tuples per cell)
  Apparatus-level aggregates (cross_cell_aggregates.apparatus):
    - phase10_bootstrap_variance_p2.selection_rationale (v0.82.0.23):
      structured block documenting why the 6-cell PHASE10_SUBSET was
      chosen (subset_size, model_coverage, temperature_coverage,
      selection_criterion, wall_time_constraint, original_changelog_ref,
      scope_acknowledged). Schema-self-documentation of subset
      sampling so §6.6 / §10 prose can cite without reconstructing
      from session history.
    - phase11_geometric_diagnostics_p2.geometric_aggregates.v5d_flagged_quadrant_distribution_note
      (v0.82.0.23): inline metadata defining v5d_flagged_quadrant_distribution
      operationally — Phase 5 aggregator's runtime check comparing
      cell's V5d-spread amgm against Phase 3's calibrated τ_spread or
      τ_amgm. Flag means SIGNATURE MATCH (cell exhibits V5d's
      variance-spread pattern), NOT regime-membership. §5.4 / §6.4
      prose cites 'cells with V5d-signature match' rather than
      'cells in V5d failure regime'.
    - partition_b_substitution_aggregates (v0.82.0.23): cross-cell
      §9.2 + §9.3 evidence base.
  Naming canonicalization:
    - rhat_ridge_h1_crosssource (canonical) added alongside rhat_ridge_heldout
      (deprecated alias retained for paper 1 reproducibility, removal in 0.82.0)
"""

import json
import math
import re

import numpy as np

SCHEMA_VERSION = "0.81.0"

# v0.82.0.12: single source of truth for iota_version across results.json
# producers. Pre-patch this lived hardcoded in export_stats.py (line ~5840
# as "_iota_version = 0.80.0.0"), and the version-bump regex sweep we use
# at every ship missed it. Result: schema_version moved to 0.81.0 in
# Ship 2 but iota_version stayed pinned at 0.80.0.0 in every results.json
# emitted since. Co-locating with SCHEMA_VERSION here means the bump
# sweep catches it (it already touches results_schema.py), and any other
# producer can import this constant rather than re-stating the version.
IOTA_VERSION = "0.82.0.23"


# ══════════════════════════════════════════════════════════════════════
#  Encoder — numpy, NaN, Infinity, ndarray
# ══════════════════════════════════════════════════════════════════════

def _encode_value(v):
    """Recursively normalise a value for JSON serialisation.
    - numpy scalars → Python scalars
    - NaN → None  (becomes null in JSON)
    - ±Inf → "inf" / "-inf" strings
    - ndarrays → {"_type": "ndarray", "dtype", "shape", "data"}
    - dict/list recurse
    - everything else passes through
    """
    # numpy scalar
    if isinstance(v, np.generic):
        v = v.item()
    # float checks (after numpy conversion)
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return round(v, 6)
    # int passes through
    if isinstance(v, (int, bool)):
        return v
    # ndarray
    if isinstance(v, np.ndarray):
        return {
            "_type": "ndarray",
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "data":  [_encode_value(x) for x in v.flatten().tolist()],
        }
    # dict
    if isinstance(v, dict):
        return {str(k): _encode_value(val) for k, val in v.items()}
    # list / tuple
    if isinstance(v, (list, tuple)):
        return [_encode_value(x) for x in v]
    # str / None passes through
    if isinstance(v, (str, type(None))):
        return v
    # fallback — repr as string
    try:
        return str(v)
    except Exception:
        return None


class IOTAEncoder(json.JSONEncoder):
    """JSON encoder that pre-processes with _encode_value. Use as
    json.dumps(obj, cls=IOTAEncoder, allow_nan=False)."""

    def iterencode(self, o, _one_shot=False):
        # Pre-process the entire object before iterencode sees it so NaN/Inf
        # conversions are settled.
        return super().iterencode(_encode_value(o), _one_shot=_one_shot)


def atomic_json_dump(obj, path, **kwargs):
    """Write JSON atomically. v0.83.2.

    Writes to {path}.tmp first, then os.replace() to {path}. If the
    process is interrupted mid-write (Ctrl+C, crash, power loss), the
    original file at {path} is preserved unchanged. Without this
    pattern, an interrupted write leaves a corrupt JSON file that
    breaks downstream runs.

    Pass through all kwargs to json.dump (e.g. indent, default,
    ensure_ascii, cls=IOTAEncoder).
    """
    import json, os
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, **kwargs)
    os.replace(tmp, path)


def dump_results(obj, fp, **kwargs):
    """Write obj to file-like fp as JSON using IOTAEncoder. Sets indent=2
    by default, allow_nan=False (IOTAEncoder has already sanitized)."""
    kwargs.setdefault('indent', 2)
    kwargs.setdefault('allow_nan', False)
    json.dump(obj, fp, cls=IOTAEncoder, **kwargs)



# ══════════════════════════════════════════════════════════════════════
#  Cell key builder
# ══════════════════════════════════════════════════════════════════════

def cell_key(family, size, variant, temperature):
    """Flat cell key: {family}_{size}_{variant}_T{temp}.

    size is expected to include quant (e.g. "8b_4bit"). temperature is
    formatted to one decimal (e.g., T0.0, T0.2, T1.0).

    This is the APPARATUS CANON form. See cell_key_writerbot() for the
    alternate form used in side-files and bot-to-bot dispatches, and
    translate_cell_key() for universal conversion.
    """
    t = f"{float(temperature):.1f}" if temperature is not None else "TNA"
    return f"{family}_{size}_{variant}_T{t}"


# v0.83: Cell-key translation layer
# Three forms exist in the apparatus:
#   1. APPARATUS_CANON: gemma_9b_4bit_abliterated_T0.4 (4bit, T0.4)
#      - produced by cell_key()
#      - used in measurements blocks of results.json / results4th.json
#   2. WRITERBOT:       gemma_9b_q4_abliterated_t04   (q4, t04)
#      - used in side-files, bot-to-bot dispatches, decomposition_p2
#   3. PHASE5_HYBRID:   gemma_9b_q4_abliterated_T0.4  (q4, T0.4)
#      - produced incidentally by phase5_aggregator.cells path
#      - DEPRECATED; new code should not produce this form
# Use translate_cell_key() to convert between any two forms.

_QUANT_CANON_TO_WB = {'4bit': 'q4', '8bit': 'q8', 'fp16': 'fp16'}
_QUANT_WB_TO_CANON = {'q4': '4bit', 'q8': '8bit', 'fp16': 'fp16'}


def cell_key_writerbot(family, size, variant, temperature):
    """Writerbot-form cell key: {family}_{size_bare}_{quant_wb}_{variant}_t{TT}.

    Examples:
      cell_key_writerbot('gemma', '9b_4bit', 'abliterated', 0.4)
        -> 'gemma_9b_q4_abliterated_t04'
      cell_key_writerbot('gemma', '2b_fp16', 'abliterated', 0.0)
        -> 'gemma_2b_fp16_abliterated_t00'
    """
    if '_' in size:
        size_bare, quant_canon = size.rsplit('_', 1)
    else:
        size_bare, quant_canon = size, 'fp16'
    quant_wb = _QUANT_CANON_TO_WB.get(quant_canon, quant_canon)
    if temperature is None:
        t_suffix = 'tNA'
    else:
        t_int = int(round(float(temperature) * 10))
        t_suffix = f't{t_int:02d}'
    return f"{family}_{size_bare}_{quant_wb}_{variant}_{t_suffix}"


def parse_cell_key_writerbot(key):
    """Parse writerbot-form key. Returns dict matching parse_cell_key()."""
    parts = key.split('_')
    if len(parts) != 5:
        return None
    family, size_bare, quant_wb, variant, t_suffix = parts
    quant_canon = _QUANT_WB_TO_CANON.get(quant_wb, quant_wb)
    if not t_suffix.startswith('t'):
        return None
    try:
        t_int = int(t_suffix[1:])
    except ValueError:
        return None
    return {
        'family': family,
        'size': f"{size_bare}_{quant_canon}",
        'variant': variant,
        'temperature': t_int / 10.0,
    }


def translate_cell_key(key, target_form='canon'):
    """Translate a cell key to the target form. Accepts any of the three
    known forms and returns the requested form.

    target_form: one of 'canon' (apparatus canon, default), 'writerbot',
                 'phase5_hybrid'.

    Returns the translated key, or the original key if it cannot be parsed.
    """
    # Try writerbot parse first (matches t04 suffix)
    parts = key.split('_')
    if len(parts) == 5 and parts[-1].startswith('t') and parts[-1][1:].isdigit():
        parsed = parse_cell_key_writerbot(key)
    else:
        parsed = parse_cell_key(key)
        # Handle phase5_hybrid (has T0.4 but quant is in writerbot form)
        if parsed and '_' in parsed['size']:
            size_bare, quant = parsed['size'].rsplit('_', 1)
            if quant in _QUANT_WB_TO_CANON:
                # was hybrid form; convert quant to canon
                parsed['size'] = f"{size_bare}_{_QUANT_WB_TO_CANON[quant]}"

    if parsed is None:
        return key  # cannot parse; return as-is

    if target_form == 'canon':
        return cell_key(parsed['family'], parsed['size'],
                        parsed['variant'], parsed['temperature'])
    elif target_form == 'writerbot':
        return cell_key_writerbot(parsed['family'], parsed['size'],
                                    parsed['variant'], parsed['temperature'])
    elif target_form == 'phase5_hybrid':
        # q4 + T0.4 hybrid
        size_bare, quant_canon = parsed['size'].rsplit('_', 1)
        quant_wb = _QUANT_CANON_TO_WB.get(quant_canon, quant_canon)
        t = f"{float(parsed['temperature']):.1f}"
        return f"{parsed['family']}_{size_bare}_{quant_wb}_{parsed['variant']}_T{t}"
    else:
        raise ValueError(f"target_form must be canon|writerbot|phase5_hybrid, got {target_form!r}")



def parse_cell_key(key):
    """Inverse of cell_key. Returns dict with family/size/variant/temperature
    or None if the key doesn't parse."""
    m = re.match(r'^(?P<family>[^_]+)_(?P<size>[^_]+_[^_]+)_(?P<variant>[^_]+)_T(?P<temp>[-0-9.]+)$', key)
    if not m:
        return None
    d = m.groupdict()
    try:
        d['temperature'] = float(d['temp'])
    except ValueError:
        d['temperature'] = None
    d.pop('temp', None)
    return d


# ══════════════════════════════════════════════════════════════════════
#  Schema definition (jsonschema-compatible)
# ══════════════════════════════════════════════════════════════════════

_NUMBER_OR_NULL   = {"anyOf": [{"type": "number"}, {"type": "null"},
                                {"type": "string", "enum": ["inf", "-inf"]}]}
_INTEGER_OR_NULL  = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
_STRING_OR_NULL   = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_BOOLEAN_OR_NULL  = {"anyOf": [{"type": "boolean"}, {"type": "null"}]}


def _ndarray_schema():
    return {
        "type": "object",
        "required": ["_type", "dtype", "shape", "data"],
        "properties": {
            "_type": {"type": "string", "const": "ndarray"},
            "dtype": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
            "data":  {"type": "array"},
        },
    }


_CELL_SCHEMA = {
    "type": "object",
    "required": ["identifiers", "provenance", "measurements", "derived", "meta"],
    "additionalProperties": False,
    "properties": {
        "identifiers": {
            "type": "object",
            "required": ["family", "size", "variant", "temperature"],
            "properties": {
                "family":       {"type": "string"},
                "size":         {"type": "string"},
                "quantization": _STRING_OR_NULL,
                "variant":      {"type": "string"},
                "temperature":  {"type": "number"},
            },
            "additionalProperties": True,
        },
        "provenance": {
            "type": "object",
            "properties": {
                "n_rows_3way":        _INTEGER_OR_NULL,
                "pool_dim_used":      _INTEGER_OR_NULL,
                "ct_source_counts":   {"type": ["object", "null"]},
                "source_run_csvs":    {"type": ["array", "null"]},
                "source_analysis_jsons": {"type": ["array", "null"]},
                "collected_at":       _STRING_OR_NULL,
            },
            "additionalProperties": True,
        },
        "measurements": {
            "type": "object",
            "additionalProperties": True,
        },
        "derived": {
            "type": "object",
            "additionalProperties": True,
        },
        "meta": {
            "type": "object",
            "additionalProperties": True,
        },
    },
}


def build_schema():
    """Returns the jsonschema doc for the master results.json."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title":   "IOTA Results (master paper JSON)",
        "type":    "object",
        "required": ["schema_version", "iota_version", "generated_at",
                     "cells", "cross_cell_aggregates",
                     "synthetic_calibration", "toy_nonlinearity",
                     "methodology_calibration"],
        "additionalProperties": True,
        "properties": {
            "schema_version": {"type": "string"},
            "iota_version":   {"type": "string"},
            "git_commit":     {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "generated_at":   {"type": "string"},
            "cells": {
                "type": "object",
                "patternProperties": {r'^.+$': _CELL_SCHEMA},
                "additionalProperties": False,
            },
            "cross_cell_aggregates": {"type": "object"},
            "synthetic_calibration": {"type": ["object", "null"]},
            "toy_nonlinearity":      {"type": ["object", "null"]},
            "methodology_calibration": {"type": "object"},
            "paper_sections":        {"type": ["object", "null"]},
            "legacy_models":         {"type": ["object", "null"]},
            # v0.81.0 additions: paper 2 measurement layer
            # Optional — absent in 0.80.0 documents (validates cleanly).
            # Present once paper-2 phases (V5f construction, selection
            # diagnostics, kNN reliability, p2 apparatus phases) have run.
            "paper2_apparatus_metadata": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "lambda_default":            _NUMBER_OR_NULL,
                            "lambda_calibration_source": _STRING_OR_NULL,
                            "split_ratio":               {"type": ["array", "string", "null"]},
                            "pca_dim_selected":          _INTEGER_OR_NULL,
                            "n_anchor":                  _INTEGER_OR_NULL,
                            "v5f_construction_artifact_path": _STRING_OR_NULL,
                        },
                        "additionalProperties": True,
                    },
                ],
            },
        },
    }


# ══════════════════════════════════════════════════════════════════════
#  Validator
# ══════════════════════════════════════════════════════════════════════

class SchemaValidationError(ValueError):
    """Raised when a results.json fails schema validation. Blocking."""


def validate_results(obj):
    """Validate obj against the results schema. Raises SchemaValidationError
    on failure. Returns True on success.

    Uses jsonschema if available; falls back to a minimal structural check
    if the library is missing (never silent — fallback prints a warning)."""
    schema = build_schema()
    try:
        import jsonschema
    except ImportError:
        # Minimal structural fallback. Accepting this rather than crashing
        # so the framework still works on machines without jsonschema —
        # but warn loudly because the fallback is not a real validator.
        import sys
        print("! jsonschema not installed — falling back to minimal structural check. "
              "Install jsonschema for full validation: pip install jsonschema",
              file=sys.stderr)
        return _minimal_validate(obj)

    try:
        jsonschema.validate(instance=obj, schema=schema)
    except jsonschema.ValidationError as e:
        raise SchemaValidationError(
            f"results.json schema validation failed at "
            f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}"
        ) from e
    return True


def _minimal_validate(obj):
    """Structural fallback when jsonschema isn't installed."""
    required_root = ['schema_version', 'iota_version', 'generated_at',
                     'cells', 'cross_cell_aggregates',
                     'methodology_calibration']
    for k in required_root:
        if k not in obj:
            raise SchemaValidationError(f"missing root key: {k}")
    if not isinstance(obj['cells'], dict):
        raise SchemaValidationError("cells must be an object")
    for ck, cv in obj['cells'].items():
        for required in ['identifiers', 'provenance', 'measurements',
                         'derived', 'meta']:
            if required not in cv:
                raise SchemaValidationError(
                    f"cell {ck!r} missing required key: {required}")
    return True


# ══════════════════════════════════════════════════════════════════════
#  Empty-cell constructor (for cells not yet computed)
# ══════════════════════════════════════════════════════════════════════

def empty_cell(family, size, variant, temperature, quantization=None):
    """Blank cell with all required sub-structures. Populated progressively
    as source JSONs are ingested."""
    return {
        "identifiers": {
            "family":       family,
            "size":         size,
            "quantization": quantization,
            "variant":      variant,
            "temperature":  float(temperature) if temperature is not None else None,
        },
        "provenance": {
            "n_rows_3way":          None,
            "pool_dim_used":        None,
            "ct_source_counts":     None,
            "source_run_csvs":      [],
            "source_analysis_jsons": [],
            "collected_at":         None,
        },
        "measurements": {},
        "derived":      {},
        "meta":         {},
    }
