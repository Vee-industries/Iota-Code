"""
results_builder.py -- assemble data/paper/results.json from source JSONs.

Reads per-cell analysis JSONs (Q0042, Q0043, Q0044, Q0046, Q0018, Q0023, etc.)
plus methodology calibration artifacts, populates the flat per-cell schema
defined in results_schema.py, derives the computed fields, and writes the
validated results.json.

Called from export_stats.build_all_masters in 0.80+ (replaces the legacy
per-model master_results.json aggregation path at the root level -- per-model
masters are still used for discovery/iteration but the aggregate writes
through here).
"""

import datetime
import glob
import json
import os
import re

import results_schema as _sch


# ══════════════════════════════════════════════════════════════════════
#  JSON reading with BOM / NaN tolerance
# ══════════════════════════════════════════════════════════════════════

def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            raw = f.read()
        raw = re.sub(r'\bNaN\b', 'null', raw)
        raw = re.sub(r'\bInfinity\b', 'null', raw)
        raw = re.sub(r'\b-Infinity\b', 'null', raw)
        return json.loads(raw)
    except Exception:
        return None


def _read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        import csv as _csv
        with open(path, 'r', encoding='utf-8-sig') as f:
            return list(_csv.DictReader(f))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  Cell discovery
# ══════════════════════════════════════════════════════════════════════

def discover_cells(data_root):
    """Walk data/ for Q0042_decomposition.json files. Each found path
    identifies one (family, size, variant, condition/temp) cell.

    Returns list of dicts with family, size, variant, condition, temperature,
    analysis_dir, hidden_dir.
    """
    found = []
    pattern = os.path.join(data_root, '**', 'Q0042_decomposition.json')
    for fp in sorted(glob.glob(pattern, recursive=True)):
        rel = os.path.relpath(fp, data_root).replace('\\', '/').split('/')
        if len(rel) < 6:
            continue
        family, size, variant, cond = rel[0], rel[1], rel[2], rel[3]
        if cond == 'deterministic':
            temp = 0.0
        elif cond.startswith('temp_'):
            try:    temp = float(cond.replace('temp_', ''))
            except: continue
        else:
            continue
        ana_dir = os.path.dirname(fp)
        cell_dir = os.path.dirname(ana_dir)
        hidden_dir = os.path.join(cell_dir, 'hidden_states')
        found.append({
            'family': family, 'size': size, 'variant': variant,
            'condition': cond, 'temperature': temp,
            'analysis_dir': ana_dir, 'hidden_dir': hidden_dir,
        })
    return found


# ══════════════════════════════════════════════════════════════════════
#  Quantization parser -- splits "8b_4bit" → ("8b", "4bit")
# ══════════════════════════════════════════════════════════════════════

def _split_quant(size_str):
    if '_' in size_str:
        parts = size_str.rsplit('_', 1)
        return parts[0], parts[1]
    return size_str, None


# ══════════════════════════════════════════════════════════════════════
#  Per-cell ingestion
# ══════════════════════════════════════════════════════════════════════

def _ingest_q0042(cell, q42):
    """OLS decomposition (r2_A/B/C/D/S, delta_r2_internal, delta_r2_constraint)."""
    if not q42:
        return
    m = cell['measurements']
    p = cell['provenance']

    p['n_rows_3way']   = q42.get('n_3way')
    p['pool_dim_used'] = q42.get('pool_dim')

    for k in ('r2_A_external_only', 'r2_B_external_plus_internal',
              'r2_C_external_plus_constraint', 'r2_D_full_decomposition',
              'r2_S_internal_only'):
        m[k] = q42.get(k)
    # Short aliases for the paper/derivations
    m['r2_A'] = m.get('r2_A_external_only')
    m['r2_B'] = m.get('r2_B_external_plus_internal')
    m['r2_C'] = m.get('r2_C_external_plus_constraint')
    m['r2_D'] = m.get('r2_D_full_decomposition')
    m['r2_S'] = m.get('r2_S_internal_only')
    # v0.80.0.29: delta_r2_internal/constraint can be either a scalar
    # (older R0042 output path, analysis.py:299) or a dict like
    # {'value': float, 'p_value': float, 'significant': bool}
    # (analysis.py:1052 path). Normalize: store the scalar in the main
    # field, preserve the rich form under `_full` so significance info
    # isn't lost. Without this, cross_cell_aggregates' isinstance(int,
    # float) filter dropped the dict-form values silently and the
    # `delta_r2_internal_mean_per_temperature` block stayed empty.
    def _scalar(v):
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, dict):
            inner = v.get('value')
            return inner if isinstance(inner, (int, float)) else None
        return None
    _di_raw = q42.get('delta_r2_internal')
    _dc_raw = q42.get('delta_r2_constraint')
    m['delta_r2_internal']   = _scalar(_di_raw)
    m['delta_r2_constraint'] = _scalar(_dc_raw)
    if isinstance(_di_raw, dict):
        m['delta_r2_internal_full'] = _di_raw
    if isinstance(_dc_raw, dict):
        m['delta_r2_constraint_full'] = _dc_raw

    # Interaction info
    ii = q42.get('interaction_info') or {}
    # v0.83: redundancy_*_linear is the new canonical name (matches the
    # redundancy-positive convention computed by II = mi_S + mi_E - mi_SE).
    # ii_*_linear retained as deprecated alias for backward compatibility
    # with existing readers (writerbot/statsbot integrations).
    _ii_lin_val = ii.get('II')
    _ii_frac_lin_val = ii.get('II_fraction')
    m['redundancy_linear'] = _ii_lin_val
    m['redundancy_fraction_linear'] = _ii_frac_lin_val
    m['ii_linear'] = _ii_lin_val                # deprecated alias (v0.83+)
    m['ii_fraction_linear']  = _ii_frac_lin_val # deprecated alias (v0.83+)
    m['mi_E_linear'] = ii.get('mi_E')
    m['mi_S_linear'] = ii.get('mi_S')
    m['mi_SE_linear'] = ii.get('mi_SE')
    m['r2_E_linear'] = ii.get('r2_E')
    m['r2_S_linear_ii'] = ii.get('r2_S')
    m['r2_SE_linear'] = ii.get('r2_SE')

    knn = ii.get('knn') or {}
    pooled = knn.get('pooled') or {}
    m['mi_E_knn'] = pooled.get('mi_E') if isinstance(pooled, dict) else None
    m['mi_S_knn'] = pooled.get('mi_S') if isinstance(pooled, dict) else None
    m['mi_SE_knn'] = pooled.get('mi_SE') if isinstance(pooled, dict) else None
    # v0.83: redundancy_*_knn is the new canonical name (matches the
    # redundancy-positive convention computed in _knn_ii). ii_*_knn
    # retained as deprecated alias for backward compatibility.
    _ii_knn_val = pooled.get('II') if isinstance(pooled, dict) else None
    _ii_frac_knn_val = pooled.get('II_fraction') if isinstance(pooled, dict) else None
    m['redundancy_knn'] = _ii_knn_val
    m['redundancy_fraction_knn'] = _ii_frac_knn_val
    m['ii_knn'] = _ii_knn_val                  # deprecated alias (v0.83+)
    m['ii_fraction_knn'] = _ii_frac_knn_val    # deprecated alias (v0.83+)
    m['r2_E_knn'] = pooled.get('r2_E') if isinstance(pooled, dict) else None
    m['r2_S_knn'] = pooled.get('r2_S') if isinstance(pooled, dict) else None
    m['knn_pca_dim']  = knn.get('pca_dim')
    m['n_knn_samples']= knn.get('n_samples') if isinstance(knn.get('n_samples'), (int, float)) else knn.get('n_samples')
    m['knn_agrees_with_linear'] = knn.get('agrees_with_linear')

    # Per-condition kNN breakdown (keep as-is, verbose but structured)
    m['knn_per_condition'] = knn.get('per_condition') if isinstance(knn.get('per_condition'), dict) else None

    # Linearity check (H50 inputs)
    lc = q42.get('linearity_check') or {}
    m['r2_ridge_test'] = lc.get('r2_ridge')
    m['r2_mlp_256_test'] = lc.get('r2_mlp')  # analysis.py uses a single MLP; mlp_256 is the reference
    m['linearity_max_gap'] = lc.get('max_gap')
    m['linearity_all_within_001'] = lc.get('all_within_001')

    # H16 confound decomposition
    m['h16_confound_decomposition'] = q42.get('h16_confound_decomposition')


def _ingest_q0043(cell, q43):
    """Ridge permutation sensitivity -- Rhat, Chat, Ehat insample + heldout +
    raw drops + H28 validation outputs.

    v0.80.0.34: accepts both shapes --
      legacy nested:  {'in_sample': {'Rhat': ..., 'drop_E': ...}, 'held_out': ...}
      flat perm_sens: {'perm_sens_R': {'fraction': ..., 'effect': ..., 'std': ...},
                       'heldout_perm_sens_R': {...}, ...}
    Normalizes at ingest. Same Class 8 pattern as the Q0046 legacy-payload
    remap (FINDINGS.md, 0.79.6.6).
    """
    if not q43:
        return
    m = cell['measurements']

    flat_shape = ('perm_sens_R' in q43) or ('perm_sens_E' in q43)
    nested_shape = ('in_sample' in q43) or ('insample' in q43)

    if flat_shape and not nested_shape:
        def _flat_block(prefix):
            E = q43.get(f'{prefix}perm_sens_E') or {}
            C = q43.get(f'{prefix}perm_sens_C') or {}
            R = q43.get(f'{prefix}perm_sens_R') or {}
            return {
                'Rhat':       R.get('fraction'),
                'Chat':       C.get('fraction'),
                'Ehat':       E.get('fraction'),
                'drop_E':     E.get('effect'),
                'drop_C':     C.get('effect'),
                'drop_S':     R.get('effect'),
                'drop_E_std': E.get('std'),
                'drop_C_std': C.get('std'),
                'drop_S_std': R.get('std'),
            }
        ins = _flat_block('')
        ho  = _flat_block('heldout_')
    else:
        ins = q43.get('in_sample') or q43.get('insample') or {}
        ho  = q43.get('held_out')  or q43.get('heldout')  or {}

    m['rhat_ridge_insample'] = ins.get('Rhat') or ins.get('rhat')
    m['chat_ridge_insample'] = ins.get('Chat') or ins.get('chat')
    m['ehat_ridge_insample'] = ins.get('Ehat') or ins.get('ehat')
    m['drop_E_ridge']        = ins.get('drop_E')
    m['drop_C_ridge']        = ins.get('drop_C')
    m['drop_S_ridge']        = ins.get('drop_S')
    m['drop_E_ridge_std']    = ins.get('drop_E_std')
    m['drop_C_ridge_std']    = ins.get('drop_C_std')
    m['drop_S_ridge_std']    = ins.get('drop_S_std')

    # v0.81.1.3: canonical naming. The `*_ridge_heldout` fields are
    # actually the H1 cross-source quantity (Ridge fit on the 9-source pool,
    # applied to Run 0033 -- the seed-shifted twin). They are NOT the headline
    # Ridge values reported in paper 1 §5/§6 (those are the *_insample fields
    # above, the 80/20 test). The "heldout" name is structurally misleading
    # -- it reads as "the held-out 20% test from the in-sample 80/20" but it
    # is a different thing entirely. Schema 0.81.0 introduces the canonical
    # name `*_ridge_h1_crosssource` and keeps `*_ridge_heldout` as a
    # deprecated alias populated identically. Consumers should migrate to
    # the h1_crosssource name; the heldout alias is retained to keep paper 1
    # reproducibility from breaking. Removal target: schema 0.82.0.
    _h1_R = ho.get('Rhat') or ho.get('rhat')
    _h1_C = ho.get('Chat') or ho.get('chat')
    _h1_E = ho.get('Ehat') or ho.get('ehat')
    m['rhat_ridge_h1_crosssource']  = _h1_R
    m['chat_ridge_h1_crosssource']  = _h1_C
    m['ehat_ridge_h1_crosssource']  = _h1_E
    # Deprecated aliases (populated identically; remove in 0.82.0)
    m['rhat_ridge_heldout']  = _h1_R
    m['chat_ridge_heldout']  = _h1_C
    m['ehat_ridge_heldout']  = _h1_E

    ci = q43.get('bootstrap_ci') or {}
    if ci:
        m['bootstrap_ci'] = ci

    m['h28_max_fraction_divergence']   = q43.get('h28_max_fraction_divergence')
    m['h28_status']                    = q43.get('h28_status')
    m['h28_verdict']                   = q43.get('h28_verdict')
    m['interaction_mass_heldout']      = q43.get('interaction_mass_heldout')
    m['interaction_mass_unattributed'] = q43.get('interaction_mass_unattributed')

    if 'n_3way' in q43:
        cell['provenance'].setdefault('n_rows_3way', q43.get('n_3way'))
    if 'n_perm' in q43:
        m['n_perm_q43'] = q43.get('n_perm')



def _ingest_q0046(cell, q46):
    """MLP permutation sensitivity -- per-config Rhat/Ĉ/Ê + R² + raw drops."""
    if not q46:
        return
    m = cell['measurements']

    pooled = q46.get('pooled') or {}
    # The pooled block's `results` array holds one entry per (architecture, seed).
    # We aggregate by architecture.
    results = pooled.get('results') if isinstance(pooled, dict) else None
    summary = pooled.get('summary') if isinstance(pooled, dict) else None

    if isinstance(summary, dict):
        mean = summary.get('mean') or {}
        m['rhat_mlp_pooled_mean'] = mean.get('R')
        m['chat_mlp_pooled_mean'] = mean.get('C')
        m['ehat_mlp_pooled_mean'] = mean.get('E')
        std = summary.get('std') or {}
        m['rhat_mlp_pooled_std']  = std.get('R')
        gap = summary.get('ridge_gap') or {}
        m['mlp_vs_ridge_gap_E_pooled'] = gap.get('E')
        m['mlp_vs_ridge_gap_C_pooled'] = gap.get('C')
        m['mlp_vs_ridge_gap_R_pooled'] = gap.get('R')
        m['r2_full_mlp_mean'] = summary.get('r2_full_mean')

    # Per-architecture aggregation
    arch_agg = {}
    if isinstance(results, list):
        for entry in results:
            if not isinstance(entry, dict):
                continue
            arch = str(entry.get('architecture', 'unknown'))
            arch_agg.setdefault(arch, {
                'frac_E': [], 'frac_C': [], 'frac_R': [], 'r2_full': [],
            })
            arch_agg[arch]['frac_E'].append(entry.get('frac_E'))
            arch_agg[arch]['frac_C'].append(entry.get('frac_C'))
            arch_agg[arch]['frac_R'].append(entry.get('frac_R'))
            arch_agg[arch]['r2_full'].append(entry.get('r2_full'))

    # Reduce per-arch across seeds
    def _mean_of(lst):
        lst = [x for x in lst if isinstance(x, (int, float))]
        return (sum(lst) / len(lst)) if lst else None

    per_arch = {}
    for arch, d in arch_agg.items():
        per_arch[arch] = {
            'rhat_mlp':    _mean_of(d['frac_R']),
            'chat_mlp':    _mean_of(d['frac_C']),
            'ehat_mlp':    _mean_of(d['frac_E']),
            'r2_full':     _mean_of(d['r2_full']),
            'n_seeds':     len(d['frac_R']),
        }
    m['mlp_per_architecture'] = per_arch

    # Designated reference -- "256" if present, else "128-64", else first key
    ref_key = None
    for candidate in ('256', '128-64', '64'):
        if candidate in per_arch:
            ref_key = candidate
            break
    if ref_key is None and per_arch:
        ref_key = next(iter(per_arch))
    m['mlp_reference_architecture'] = ref_key
    if ref_key:
        m['rhat_mlp_reference'] = per_arch[ref_key].get('rhat_mlp')
        m['chat_mlp_reference'] = per_arch[ref_key].get('chat_mlp')
        m['ehat_mlp_reference'] = per_arch[ref_key].get('ehat_mlp')
        m['r2_mlp_reference']   = per_arch[ref_key].get('r2_full')

    # Per-condition (56b) -- raw passthrough for the paper's per-condition story
    m['q46_per_condition'] = q46.get('per_condition')


def _ingest_q0044(cell, q44):
    """Ridge per-condition reference. Mostly used by H50/H51 to know
    what's expected of 56b."""
    if not q44:
        return
    m = cell['measurements']
    m['q44_per_run_count'] = sum(
        1 for e in (q44.get('per_run') or [])
        if isinstance(e, dict)
            and not e.get('skipped')
            and e.get('run_num') is not None
    )


def _ingest_q0018(cell, q18):
    """Layer-isolation causal profile.

    v0.80.0.35: surface binomial regime + baseline + per-layer trial counts.
    The source JSON has these fields; the previous ingester pulled only
    output_change_rate and p_binomial. Regime + counts let consumers (paper
    prose, figure subscripts) distinguish degenerate-by-design cells from
    cells with a meaningful binomial null without re-deriving the regime
    from the per-layer p_binomial null pattern.
    """
    if not q18:
        return
    m = cell['measurements']

    per_layer = q18.get('per_layer') or q18.get('layer_results')
    if isinstance(per_layer, dict):
        m['layer_output_change_rate'] = {
            k: (v.get('output_change_rate') if isinstance(v, dict) else v)
            for k, v in per_layer.items()
        }
        m['layer_binomial_p'] = {
            k: (v.get('p_binomial') if isinstance(v, dict) else None)
            for k, v in per_layer.items()
        }
        m['layer_n_trials'] = {
            k: (v.get('n_trials') if isinstance(v, dict) else None)
            for k, v in per_layer.items()
        }
        m['layer_n_changed'] = {
            k: (v.get('n_changed') if isinstance(v, dict) else None)
            for k, v in per_layer.items()
        }

    m['layer_binomial_regime']    = q18.get('binomial_null')
    bl = q18.get('baseline') or {}
    m['layer_baseline_rate']      = bl.get('rate')
    m['layer_baseline_n_trials']  = bl.get('n_trials')


def _ingest_channel_marginal(cell, channel_rows):
    """Pull this cell's channel_marginal row out of the calibration CSV."""
    if not channel_rows:
        return
    ids = cell['identifiers']
    match = None
    for row in channel_rows:
        if (row.get('family')    == ids['family']
            and row.get('size')    == ids['size']
            and row.get('variant') == ids['variant']):
            try:
                if abs(float(row.get('temperature') or -99) - ids['temperature']) < 1e-6:
                    match = row
                    break
            except (TypeError, ValueError):
                continue
    if match is None:
        return
    m = cell['measurements']
    def _f(k):
        v = match.get(k)
        if v is None or v == '':
            return None
        try:    return float(v)
        except: return v
    m['r2_ridge_E_only'] = _f('r2_ridge_E')
    m['r2_ridge_S_only'] = _f('r2_ridge_S')
    m['r2_mlp_E_only']   = _f('r2_mlp_E')
    m['r2_mlp_S_only']   = _f('r2_mlp_S')
    m['gap_E']           = _f('gap_E')
    m['gap_S']           = _f('gap_S')
    m['asymmetry']       = _f('asymmetry')
    m['asymmetry_knn']   = _f('knn_asymmetry')


# ══════════════════════════════════════════════════════════════════════
#  compute_derived -- functions of measurements
# ══════════════════════════════════════════════════════════════════════

def compute_derived(cell):
    """Populate cell['derived'] from cell['measurements']. Idempotent."""
    m = cell['measurements']
    d = {}

    # MLP vs Ridge share gaps (pooled reference)
    # v0.80.0.30: was `rhat_mlp_reference - rhat_ridge_insample`. The
    # canonical gap definition (analysis.py:3360) is
    # `pooled_mean - ridge`, so subtracting from reference produces a
    # value offset by `reference - pooled_mean` -- not the gap. Use
    # pooled_mean to compute the derived gap so it matches the stored
    # `mlp_vs_ridge_gap_R_pooled`. Falls back to direct gap if the raw
    # subtraction can't be computed.
    rhat_r = m.get('rhat_ridge_insample')
    pooled = m.get('rhat_mlp_pooled_mean')
    stored_gap = m.get('mlp_vs_ridge_gap_R_pooled')
    if isinstance(rhat_r, (int, float)) and isinstance(pooled, (int, float)):
        d['rhat_gap_mlp_minus_ridge'] = pooled - rhat_r
    elif isinstance(stored_gap, (int, float)):
        d['rhat_gap_mlp_minus_ridge'] = stored_gap

    # MLP vs Ridge R² gap
    r2_r = m.get('r2_ridge_test')
    r2_m = m.get('r2_mlp_256_test')
    if isinstance(r2_r, (int, float)) and isinstance(r2_m, (int, float)):
        d['r2_gap_mlp_minus_ridge'] = r2_m - r2_r

    # H50: within_001 across all fit MLPs (delegated to linearity_check already;
    # re-expose here as derived for consumer clarity)
    d['h50_passes'] = bool(m.get('linearity_all_within_001')) \
                      if m.get('linearity_all_within_001') is not None else None
    max_gap = m.get('linearity_max_gap')
    if max_gap is not None:
        d['mlp_delta_r2'] = max_gap
        if isinstance(max_gap, (int, float)):
            d['h50_verdict'] = ('pass' if max_gap <= 0.01 else 'fail')

    # H51: linear and kNN both below threshold
    thresh = 0.10
    # v0.83: prefer new names, fall back to deprecated aliases.
    lin_frac = m.get('redundancy_fraction_linear', m.get('ii_fraction_linear'))
    knn_frac = m.get('redundancy_fraction_knn',    m.get('ii_fraction_knn'))
    d['h51_threshold'] = thresh
    if isinstance(lin_frac, (int, float)):
        d['h51_linear_passes'] = abs(lin_frac) < thresh
    if isinstance(knn_frac, (int, float)):
        d['h51_knn_passes']    = abs(knn_frac) < thresh
    if isinstance(lin_frac, (int, float)) and isinstance(knn_frac, (int, float)):
        d['h51_passes_both']   = (abs(lin_frac) < thresh) and (abs(knn_frac) < thresh)

    # Channel-marginal reconstruction
    gE = m.get('gap_E'); gS = m.get('gap_S')
    if isinstance(gE, (int, float)) and isinstance(gS, (int, float)):
        d['channel_asymmetry'] = gE - gS
        d['channel_asymmetry_sign'] = ('E-nonlinear' if (gE - gS) > 0.03
                                        else 'S-nonlinear' if (gE - gS) < -0.03
                                        else 'symmetric')

    cell['derived'] = d


# ══════════════════════════════════════════════════════════════════════
#  Cross-cell aggregates
# ══════════════════════════════════════════════════════════════════════

def _read_apparatus_phase3():
    """Read Phase 3 V5d threshold calibration result if present."""
    import json as _json, os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    p = _os.path.join(here, 'data', 'paper', 'calibration',
                      'v5d_threshold', 'calibration.json')
    if not _os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            data = _json.load(f)
    except Exception:
        return None
    return {
        'tau_spread': data.get('tau_spread'),
        'tau_amgm': data.get('tau_amgm'),
        'v5d_true_spread_flag_rate': data.get('v5d_true_spread_flag_rate'),
        'v5b_low_synergy_spread_flag_rate': data.get('v5b_low_synergy_spread_flag_rate'),
        'iota_version_at_emit': data.get('iota_version'),
        'timestamp': data.get('timestamp'),
    }


def _read_apparatus_phase5():
    """Read Phase 5 aggregator output if present. Returns full per-cell
    block plus cross-cell summary."""
    import json as _json, os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    p = _os.path.join(here, 'data', 'paper', 'calibration',
                      'apparatus_aggregator', 'per_cell.json')
    if not _os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            data = _json.load(f)
    except Exception:
        return None
    return {
        'bucket':         data.get('bucket'),
        'lambda_used':    data.get('lambda_used'),
        'tau_spread':     data.get('tau_spread'),
        'tau_amgm':       data.get('tau_amgm'),
        'n_cells_evaluated': data.get('n_cells_evaluated'),
        'n_v5d_flagged':  data.get('n_v5d_flagged'),
        'cells':          data.get('cells', {}),
        'iota_version_at_emit': data.get('iota_version'),
        'timestamp':      data.get('timestamp'),
    }


def _read_paper2_phase(phase_dir, fname):
    """Generic reader for paper-2 phase outputs (phases 6-11). Returns
    the parsed JSON dict (top-level, including 'cells' subdict) or None
    if the file is missing/unparseable.

    v0.81.1.7 ship 5: paper-2 phases 6-11 land under
    data/paper/calibration/{phase_dir}/{fname}. results.json's
    cross_cell_aggregates.apparatus.{phase_key} surfaces the summary
    block; per-cell measurements get the relevant slice spread into
    cell['measurements'] in build_results."""
    import json as _json, os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    p = _os.path.join(here, 'data', 'paper', 'calibration',
                      phase_dir, fname)
    if not _os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            return _json.load(f)
    except Exception:
        return None


def _read_apparatus_phase6():
    """Phase 6 -- function-class fits on partition B."""
    return _read_paper2_phase('function_class_p2', 'per_cell.json')


def _read_apparatus_phase7():
    """Phase 7 -- kNN anchor on partition A."""
    return _read_paper2_phase('kraskov_anchor_p2', 'anchors.json')


def _read_apparatus_phase8():
    """Phase 8 -- λ-sensitivity sweep + lambda_default selection."""
    return _read_paper2_phase('apparatus_p2', 'per_cell.json')


def _read_apparatus_phase9():
    """Phase 9 -- stacking baseline."""
    return _read_paper2_phase('stacking_baseline_p2', 'per_cell.json')


def _read_apparatus_phase10():
    """Phase 10 -- bootstrap variance."""
    return _read_paper2_phase('bootstrap_variance_p2', 'per_cell.json')


def _read_apparatus_phase11():
    """Phase 11 -- hull diagnostic."""
    return _read_paper2_phase('geometric_diagnostics_p2', 'per_cell.json')


def _read_apparatus_phase12():
    """Phase 12 -- estimator joint R² at canonical partition-B."""
    return _read_paper2_phase('estimator_joint_R2_p2', 'per_cell.json')


def _read_paper2_foundations():
    """Read split_pca_selection + knn_mi_reliability foundations to
    populate paper2_apparatus_metadata top-level block. Returns dict
    with the metadata fields (lambda_default, split_ratio, pca_dim,
    n_anchor, v5f_construction_artifact_path); None when absent."""
    import json as _json, os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    sel_p = _os.path.join(here, 'data', 'paper', 'calibration',
                            'split_pca_selection', 'selection.json')
    rel_p = _os.path.join(here, 'data', 'paper', 'calibration',
                            'knn_mi_reliability', 'reliability.json')
    apparatus_p2_p = _os.path.join(here, 'data', 'paper', 'calibration',
                                     'apparatus_p2', 'per_cell.json')
    sel = rel = ap2 = None
    for path, slot in ((sel_p, 'sel'), (rel_p, 'rel'),
                        (apparatus_p2_p, 'ap2')):
        if _os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    if slot == 'sel': sel = _json.load(f)
                    elif slot == 'rel': rel = _json.load(f)
                    else: ap2 = _json.load(f)
            except Exception:
                pass
    if not (sel or rel or ap2):
        return None
    sp = (sel or {}).get('selected_operating_point') or {}
    out = {
        'split_ratio':        sp.get('split'),
        'pca_dim_selected':   sp.get('pca'),
    }
    op = (rel or {}).get('operating_point') or {}
    if op:
        out['n_anchor'] = op.get('n_subsample')
    if ap2:
        out['lambda_default']            = ap2.get('lambda_default_chosen')
        out['lambda_calibration_source'] = ap2.get('lambda_calibration_source')
    out['v5f_construction_artifact_path'] = 'v5f_construction_artifact.md'
    return out


def _read_apparatus_phase1():
    """Read Phase 1 spike result if present and return its summary block.

    v0.81.0.0: Phase 1 of the lagrangian apparatus emits its bucket
    assignment + linearization sweep into a dedicated calibration
    JSON. Build_results lifts the summary into
    cross_cell_aggregates.apparatus.phase1_spike for paper-write
    convenience.

    Returns None if the file isn't there (Phase 1 hasn't fired yet).
    """
    import json as _json, os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    p = _os.path.join(here, 'data', 'paper', 'calibration',
                      'kraskov_spike', 'spike_result.json')
    if not _os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8-sig') as f:
            data = _json.load(f)
    except Exception:
        return None
    return {
        'bucket':       data.get('part_a_bias_measurement', {}).get('bucket'),
        'beta_overall': data.get('part_a_bias_measurement', {}).get('beta_overall'),
        'n_v5b_configs_tested':
            len(data.get('part_a_bias_measurement', {}).get('per_config', [])),
        'linearization_sweep_holds':
            data.get('part_b_linearization_sweep', {}).get('n_holds'),
        'linearization_sweep_total':
            data.get('part_b_linearization_sweep', {}).get('n_total'),
        'iota_version_at_emit': data.get('iota_version'),
        'timestamp':    data.get('timestamp'),
    }


def _read_function_class_sensitivity():
    """Read Q0057_function_class_sensitivity.json if present and return its
    cross_cell_aggregates summary block + per-cell sign_check rollup.
    Returns None if the file isn't there (Run 0057 hasn't fired yet).

    v0.80.0.38: added. The full per-cell breakdown stays in Q0057's own
    JSON (it's large and paper-write doesn't need it inline). Only the
    summary block lands in results.json's cross_cell_aggregates so
    figure scripts and §5.4 prose generation can reference it directly.
    """
    import json as _json, os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    q57 = _os.path.join(here, 'data', 'paper', 'Q0057_function_class_sensitivity.json')
    if not _os.path.exists(q57):
        return None
    try:
        with open(q57, 'r', encoding='utf-8-sig') as f:
            data = _json.load(f)
    except Exception:
        return None

    summary = {
        'cross_cell_aggregates': data.get('cross_cell_aggregates', {}),
        'function_classes':      data.get('function_classes', []),
        'metadata':              data.get('metadata', {}),
        'n_cells_with_full_data': len([
            k for k, v in data.get('cells', {}).items() if 'error' not in v
        ]),
    }
    return summary


def _compute_h58_partition_sanity(cells):
    """H0_58 partition sanity check -- verify renormalized E+C+R fractions
    sum to 1.0 within numerical tolerance across all cells and estimator
    variants. Pre-registered thresholds (see IOTA_Hypotheses_v10.md):
      per-cell:    |E+C+R - 1.0| <= 0.01
      pooled mean: |mean(deviation)| <= 0.005

    Stored as cross_cell_aggregates.h58_partition_sanity. Read by figure
    captions and paper §5.4 sanity-check prose. Failure of this check
    invalidates downstream R-spread / Ridge-vs-MLP comparisons because
    those rest on the partition being well-formed.

    v0.80.0.36: added.
    """
    import statistics as _stats

    PER_CELL_THRESHOLD    = 0.01
    POOLED_MEAN_THRESHOLD = 0.005

    VARIANTS = [
        ("ridge_insample",  "ehat_ridge_insample",  "chat_ridge_insample",  "rhat_ridge_insample"),
        ("ridge_heldout",   "ehat_ridge_heldout",   "chat_ridge_heldout",   "rhat_ridge_heldout"),
        ("mlp_pooled_mean", "ehat_mlp_pooled_mean", "chat_mlp_pooled_mean", "rhat_mlp_pooled_mean"),
        ("mlp_reference",   "ehat_mlp_reference",   "chat_mlp_reference",   "rhat_mlp_reference"),
    ]

    out = {
        "thresholds": {
            "per_cell_abs_dev_max":    PER_CELL_THRESHOLD,
            "pooled_mean_abs_dev_max": POOLED_MEAN_THRESHOLD,
        },
        "n_cells_total": len(cells),
        "variants":      {},
        "overall":       None,
    }

    all_pass = True

    for vname, ke, kc, kr in VARIANTS:
        deviations = []
        per_cell = {}
        missing = []

        for cell_key, cell in cells.items():
            m = cell.get("measurements", {})
            e, c, r = m.get(ke), m.get(kc), m.get(kr)
            if e is None or c is None or r is None:
                missing.append(cell_key)
                continue
            try:
                s = float(e) + float(c) + float(r)
            except (TypeError, ValueError):
                missing.append(cell_key)
                continue
            d = s - 1.0
            deviations.append(d)
            per_cell[cell_key] = {
                "E":         float(e),
                "C":         float(c),
                "R":         float(r),
                "sum":       s,
                "deviation": d,
            }

        if not deviations:
            out["variants"][vname] = {
                "fields":                     {"E": ke, "C": kc, "R": kr},
                "n_cells_evaluated":          0,
                "n_cells_missing":            len(missing),
                "missing_cells":              missing,
                "per_cell_max_abs_deviation": None,
                "pooled_mean_deviation":      None,
                "pooled_std_deviation":       None,
                "per_cell_pass":              None,
                "pooled_mean_pass":           None,
                "verdict":                    "SKIPPED",
                "per_cell":                   {},
            }
            continue

        per_cell_max_abs = max(abs(d) for d in deviations)
        pooled_mean = sum(deviations) / len(deviations)
        pooled_std  = _stats.stdev(deviations) if len(deviations) > 1 else 0.0

        per_cell_pass = per_cell_max_abs <= PER_CELL_THRESHOLD
        pooled_pass   = abs(pooled_mean) <= POOLED_MEAN_THRESHOLD
        variant_pass  = per_cell_pass and pooled_pass
        if not variant_pass:
            all_pass = False

        out["variants"][vname] = {
            "fields":                     {"E": ke, "C": kc, "R": kr},
            "n_cells_evaluated":          len(deviations),
            "n_cells_missing":            len(missing),
            "missing_cells":              missing,
            "per_cell_max_abs_deviation": per_cell_max_abs,
            "pooled_mean_deviation":      pooled_mean,
            "pooled_std_deviation":       pooled_std,
            "per_cell_pass":              per_cell_pass,
            "pooled_mean_pass":           pooled_pass,
            "verdict":                    "PASS" if variant_pass else "FAIL",
            "per_cell":                   per_cell,
        }

    out["overall"] = "PASS" if all_pass else "FAIL"
    return out


def compute_cross_cell_aggregates(cells):
    """Compute cross-cell paper-headline statistics."""
    agg = {}

    # v0.80.0.30: Ridge values fall back to pooled-mean-minus-gap when
    # Run 0043 is absent. Mirrors fig1/fig3/fig4 helper. CRITICAL: the
    # gap is `pooled_mean - ridge` (analysis.py:3360), so the only valid
    # MLP denominator for the subtraction is `rhat_mlp_pooled_mean`,
    # NOT `rhat_mlp_reference`. Reference architecture and pooled mean
    # differ by several percent in typical Run 0046 output; using the
    # wrong one offsets every Ridge value by that delta.
    def _real(x):
        return (isinstance(x, (int, float))
                and not (isinstance(x, float) and x != x))

    def _ridge_of(measurements):
        direct = measurements.get('rhat_ridge_insample')
        if _real(direct):
            return direct
        pm  = measurements.get('rhat_mlp_pooled_mean')
        gap = measurements.get('mlp_vs_ridge_gap_R_pooled')
        if _real(pm) and _real(gap):
            return pm - gap
        return None

    def _mlp_of(measurements):
        # v0.80.0.30: keep reference-first (matches pre-0.80.0.29
        # behavior). Changing to pooled-first would alter the
        # `rhat_mlp_spread_at_T1.0` value already known to readers of
        # results.json. The pooled-vs-reference choice for paired
        # bar plots is a separate decision left to the figure scripts.
        ref = measurements.get('rhat_mlp_reference')
        if _real(ref):
            return ref
        pm = measurements.get('rhat_mlp_pooled_mean')
        return pm if _real(pm) else None

    # Ridge spread at T=1.0 vs MLP spread at T=1.0
    r_at_t10 = []
    m_ref_at_t10 = []
    m_pool_at_t10 = []
    keys_at_t10 = []
    for key, cell in cells.items():
        if abs(cell['identifiers'].get('temperature', -99) - 1.0) > 1e-6:
            continue
        keys_at_t10.append(key)
        r  = _ridge_of(cell['measurements'])
        m_ref  = cell['measurements'].get('rhat_mlp_reference')
        m_pool = cell['measurements'].get('rhat_mlp_pooled_mean')
        if _real(r):       r_at_t10.append(r)
        if _real(m_ref):   m_ref_at_t10.append(m_ref)
        if _real(m_pool):  m_pool_at_t10.append(m_pool)

    def _spread(lst):
        return (max(lst) - min(lst)) if lst else None

    agg['rhat_ridge_spread_at_T1.0']     = _spread(r_at_t10)
    # v0.80.0.33: surface BOTH MLP spread variants. They differ because
    # individual cells have asymmetric reference-vs-pooled deltas (LLaMA
    # in particular shows architecture-sensitivity at T=1.0). Reference
    # spread answers "how much do models disagree at one chosen
    # architecture?"; pooled spread answers "how much do models disagree
    # in their architecture-averaged R?". Pooled is the more honest
    # headline for the paper since it doesn't privilege one architecture
    # choice. Reference is what fig1's MLP bars actually show.
    agg['rhat_mlp_spread_at_T1.0']         = _spread(m_ref_at_t10)
    agg['rhat_mlp_spread_at_T1.0_pooled']  = _spread(m_pool_at_t10)
    agg['n_cells_at_T1.0']                 = len(r_at_t10)
    # v0.80.0.30: expose which cell keys actually contributed to the
    # T=1.0 aggregates. Helps the user audit when the spread number
    # doesn't match a hand-computed value -- points directly at any
    # contaminating cell (e.g. a non-paper model that has analysis
    # data, or a stale legacy cell key).
    agg['cells_at_T1.0'] = sorted(keys_at_t10)

    # Per-temperature mean delta_r2_internal
    per_t = {}
    for key, cell in cells.items():
        t = cell['identifiers'].get('temperature')
        if t is None:
            continue
        v = cell['measurements'].get('delta_r2_internal')
        if _real(v):
            per_t.setdefault(str(t), []).append(v)
    agg['delta_r2_internal_mean_per_temperature'] = {
        k: (sum(vs) / len(vs)) for k, vs in per_t.items() if vs
    }

    # H50 pass count per temperature
    h50_per_t = {}
    for key, cell in cells.items():
        t = cell['identifiers'].get('temperature')
        p = cell['derived'].get('h50_passes')
        if t is None or p is None:
            continue
        tk = str(t)
        h50_per_t.setdefault(tk, {'pass': 0, 'fail': 0, 'total': 0})
        h50_per_t[tk]['total'] += 1
        if p:   h50_per_t[tk]['pass'] += 1
        else:   h50_per_t[tk]['fail'] += 1
    agg['h50_status_per_temperature'] = h50_per_t

    # Convergence claim status -- at each T, do all cells' Rhat values fall
    # within 0.05 of each other?
    conv = {}
    for tk, vals in (
        ('0.0', None), ('0.2', None), ('0.4', None),
        ('0.6', None), ('0.8', None), ('1.0', None),
    ):
        rs = []
        for key, cell in cells.items():
            t = cell['identifiers'].get('temperature')
            if t is None or abs(t - float(tk)) > 1e-6:
                continue
            r = _mlp_of(cell['measurements'])
            if _real(r):
                rs.append(r)
        if len(rs) >= 2:
            spread = max(rs) - min(rs)
            conv[tk] = {
                'n':      len(rs),
                'spread': spread,
                'status': ('supported' if spread < 0.05
                          else 'mixed' if spread < 0.15
                          else 'rejected'),
            }
    agg['convergence_status_per_temperature'] = conv

    # v0.80.0.38: function-class sensitivity (Run 0057). Read Q0057
    # JSON if present and aggregate summary into cross_cell_aggregates.
    # Run 0059 (paper assembly, was 0058) won't fire without Q0057 (hard prereq),
    # so under normal operation this field will be populated. Returning
    # None is acceptable for ad-hoc / partial-run invocations of
    # build_results that bypass the Run 0059 dispatch path.
    _fcs = _read_function_class_sensitivity()
    if _fcs is not None:
        agg["function_class_sensitivity"] = _fcs

    # v0.81.0.0: Phase 1 of the lagrangian apparatus. Bucket assignment
    # + linearization sweep summary lives at calibration/kraskov_spike/
    # but a paper-write summary lifts here for the merged paper's §7
    # disclosure paragraph and §8 calibration table.
    _ph1 = _read_apparatus_phase1()
    if _ph1 is not None:
        agg.setdefault("apparatus", {})["phase1_spike"] = _ph1

    # v0.81.1.0: Phase 3 + Phase 5 ingestion
    _ph3 = _read_apparatus_phase3()
    if _ph3 is not None:
        agg.setdefault("apparatus", {})["phase3_v5d_threshold"] = _ph3
    _ph5 = _read_apparatus_phase5()
    if _ph5 is not None:
        agg.setdefault("apparatus", {})["phase5_aggregator"] = _ph5

    # v0.81.1.7 ship 5: paper-2 measurement-layer phases 6-11. Each
    # surfaces a summary block + per-cell rollup. Per-cell measurements
    # gain rkhs_shares / anchored_shares / lambda_sweep / hull /
    # bootstrap_variance / stacking_baseline_shares spread into
    # cell['measurements'] in build_results below.
    for phase_n, key, reader in [
        (6, 'phase6_function_class_p2',     _read_apparatus_phase6),
        (7, 'phase7_kraskov_anchor_p2',     _read_apparatus_phase7),
        (8, 'phase8_apparatus_p2',          _read_apparatus_phase8),
        (9, 'phase9_stacking_baseline_p2',  _read_apparatus_phase9),
        (10, 'phase10_bootstrap_variance_p2', _read_apparatus_phase10),
        (11, 'phase11_geometric_diagnostics_p2',  _read_apparatus_phase11),
        (12, 'phase12_estimator_joint_R2_p2',     _read_apparatus_phase12),
    ]:
        block = reader()
        if block is None:
            continue
        # Cross-cell summary slot (drop the heavy per-cell payload from
        # the cross-cell view; it's spread into cell['measurements'] below)
        slot = {
            'phase':        block.get('phase'),
            'iota_version': block.get('iota_version'),
            'timestamp':    block.get('timestamp'),
            'summary':      block.get('summary'),
            'output_path':  f'data/paper/calibration/{key.split("_", 1)[1]}/'
                            f'{"anchors" if "anchor" in key else "per_cell"}.json',
        }
        # v0.82.0.23: Phase 11 carries cross-cell geometric_aggregates
        # (counts, mechanism breakdown, quadrant distribution, V5d
        # cross-reference). Surface alongside summary for fig7
        # consumption.
        if 'geometric_aggregates' in block:
            slot['geometric_aggregates'] = block['geometric_aggregates']
        agg.setdefault("apparatus", {})[key] = slot

    # v0.82.0.23: Item 4 -- PHASE10_SUBSET selection rationale.
    # Lifts the rationale from CHANGELOG 0.82.0.19 / decomposition_p2.py
    # comments into structured schema, so future readers don't have
    # to reconstruct from session history. Schema-self-documentation
    # precedent: cosine_negative_history_note (v0.82.0.20).
    p10 = (agg.get("apparatus") or {}).get(
        "phase10_bootstrap_variance_p2") or {}
    if p10:
        p10['selection_rationale'] = {
            'subset_size': 6,
            'subset_cells': [
                'gemma_2b_q4_abliterated_t00',
                'gemma_2b_q4_abliterated_t06',
                'gemma_2b_fp16_abliterated_t00',
                'gemma_2b_fp16_abliterated_t10',
                'gemma_9b_q4_abliterated_t00',
                'llama_8b_q4_abliterated_t00',
            ],
            'model_coverage': (
                "Three model families: Gemma 2B (Q4 + FP16 variants -- "
                "covers quantization axis), Gemma 9B Q4 (covers model "
                "size axis at fixed quantization), Llama 8B Q4 (covers "
                "architecture family axis). Loses Gemma 9B FP16 and "
                "Llama 8B FP16 -- disclosed as scope."
            ),
            'temperature_coverage': (
                "Five cells at T=0.0 (deterministic baseline), one cell "
                "at T=0.6 (gemma_2b_q4 only -- the within-model "
                "temperature contrast), one cell at T=1.0 "
                "(gemma_2b_fp16_t10 -- the hull-violator). Temperature "
                "coverage is concentrated at T=0.0 to ground the cross-"
                "cell variance comparison; the t=0.6 and t=1.0 cells "
                "anchor temperature-axis trajectories."
            ),
            'selection_criterion': (
                "Span the variation surfaces that matter for cross-class "
                "variance comparison: pool_dim (cost dial -- 64 vs 1024 "
                "native), temperature regime, model family, "
                "quantization. Subset chosen to give one large-pool "
                "T=0 baseline, one large-pool T-sweep contrast, one "
                "large-pool quantization contrast, the hull-violator "
                "T=1.0 cell, and two small-pool cells from different "
                "families."
            ),
            'wall_time_constraint': (
                "n_bootstrap=200 × 24 cells projected to 600-900 hr on "
                "single 3080 (gemma_2b_q4_t00 ~17.7 min/resample at "
                "21,600 rows × pool_dim=1024). Reduced to N_BOOTSTRAP=10 "
                "× 6-cell subset = ~60 cell-resample units, fitting "
                "within available wall-time budget. Other 18 cells "
                "carry methods-note tag "
                "'bootstrap_not_run_single_shot_only' as scope "
                "limitation."
            ),
            'original_changelog_ref': (
                "v0.82.0.19 CHANGELOG: 'Phase 10 bootstrap variance '"
                "'(N=10 across 6-cell subset)' ship. Per-cell rationale "
                "comments live in decomposition_p2.py at PHASE10_SUBSET "
                "definition (line ~101)."
            ),
            'scope_acknowledged': (
                "Subset selection is non-random -- chosen by axis "
                "coverage, not by stratified sampling. Cross-cell "
                "variance claims hold within the subset; fleet-wide "
                "claims require either the methods-note framing or "
                "explicit re-firing on additional cells. §10 follow-up: "
                "extend bootstrap to fleet via wall-time budget "
                "increase or n_bootstrap reduction (n=5 across 24 "
                "cells = ~120 cell-resample units)."
            ),
        }

    # v0.82.0.23: Item 5 -- v5d_flagged_quadrant_distribution_note.
    # Inline metadata at Phase 11 schema location, naming the
    # operational meaning of the v5d flag (signature-match runtime
    # check, not regime-membership) so §5.4 / §6.4 prose can cite
    # the field without reconstructing the meaning from session.
    p11 = (agg.get("apparatus") or {}).get(
        "phase11_geometric_diagnostics_p2") or {}
    if p11 and 'geometric_aggregates' in p11:
        p11['geometric_aggregates']['v5d_flagged_quadrant_distribution_note'] = (
            "v5d_flagged_quadrant_distribution counts cells flagged by "
            "the V5d-spread amgm signature-match check at Phase 5 "
            "aggregator runtime. Operationally: Phase 5 compares each "
            "cell's V5d-spread amgm metric against Phase 3's calibrated "
            "thresholds (τ_spread, τ_amgm); cells whose metric crosses "
            "either threshold are flagged. The flag is a SIGNATURE "
            "MATCH -- i.e., this cell exhibits the variance-spread "
            "pattern V5d's calibration regime predicts -- NOT a regime-"
            "membership claim that the cell sits in V5d's failure "
            "regime. Flagged and non-flagged cells distribute across "
            "all four hull-membership quadrants (inside_small, "
            "inside_large, outside_small, outside_large) because the "
            "signature-match is independent of geometric position. "
            "§5.4 / §6.4 prose: cite 'cells with V5d-signature match' "
            "rather than 'cells in V5d failure regime'."
        )

    # v0.82.0.23: cross-cell partition_b_substitution_aggregates.
    # §9.2 fleet-supported claim (two-tier ordering, per-class fleet-
    # mean |ΔC|) and §9.3 C-deflation reframing (G̃-shift dC dominance,
    # KL-further-from-anchor count, cosine-negative count retained as
    # discrimination record). See partition_b_substitution.py docstring
    # for the cosine-negative metric's mechanism-discrimination role.
    try:
        import os as _os_pbs2
        import partition_b_substitution as _pbs
        _here_pbs2 = _os_pbs2.path.dirname(
            _os_pbs2.path.abspath(__file__))
        _paper_root_pbs2 = _os_pbs2.path.join(_here_pbs2,
                                                  'data', 'paper')
        _pbs_aggs = _pbs.build_aggregates(_paper_root_pbs2)
        if _pbs_aggs:
            agg.setdefault("apparatus", {})[
                "partition_b_substitution_aggregates"] = _pbs_aggs
    except Exception:
        pass  # graceful: missing source files = no aggregate block

    # v0.80.0.36: H0_58 partition sanity check on E+C+R fractions.
    # Verifies that the renormalized partition sums to 1.0 within
    # tolerance across all cells and 4 estimator variants. See
    # IOTA_Hypotheses_v10.md and paper_framing.md §5.4 for thresholds
    # and interpretation.
    agg["h58_partition_sanity"] = _compute_h58_partition_sanity(cells)

    return agg


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

def build_results(data_root, iota_version, methodology_calibration_block=None,
                  git_commit=None):
    """Construct the full results dict. Caller writes it to disk after
    validation.

    Args:
      data_root: absolute path to DATA directory
      iota_version: version string for the root field
      methodology_calibration_block: pre-loaded calibration dict (from
        export_stats._load_methodology_calibration). Can be None; the
        schema requires the key but allows its contents to be null.
      git_commit: optional git SHA for root field

    Returns the full dict, pre-validation.
    """
    cells = {}
    discovered = discover_cells(data_root)

    for cell_info in discovered:
        key = _sch.cell_key(
            cell_info['family'], cell_info['size'],
            cell_info['variant'], cell_info['temperature'])
        size_bare, quant = _split_quant(cell_info['size'])
        cell = _sch.empty_cell(
            family=cell_info['family'],
            size=cell_info['size'],
            variant=cell_info['variant'],
            temperature=cell_info['temperature'],
            quantization=quant,
        )
        cell['identifiers']['condition'] = cell_info['condition']

        # Ingest source JSONs from the analysis dir
        ad = cell_info['analysis_dir']
        _ingest_q0042(cell, _read_json(os.path.join(ad, 'Q0042_decomposition.json')))
        _ingest_q0043(cell, _read_json(os.path.join(ad, 'Q0043_sobol_partition.json')))
        _ingest_q0044(cell, _read_json(os.path.join(ad, 'Q0044_per_condition_R.json')))
        _ingest_q0046(cell, _read_json(os.path.join(ad, 'Q0046_mlp_decomposition.json')))
        _ingest_q0018(cell, _read_json(os.path.join(ad, 'Q0018_layer_isolation.json')))

        # Provenance
        source_jsons = []
        for fname in ('Q0042_decomposition.json', 'Q0043_sobol_partition.json',
                      'Q0044_per_condition_R.json', 'Q0046_mlp_decomposition.json',
                      'Q0018_layer_isolation.json'):
            fp = os.path.join(ad, fname)
            if os.path.exists(fp):
                try:
                    rel = os.path.relpath(fp, data_root)
                except ValueError:
                    rel = fp
                source_jsons.append(rel.replace('\\', '/'))
        cell['provenance']['source_analysis_jsons'] = source_jsons

        cells[key] = cell

    # Pull channel_marginal rows from the calibration CSV (top-level,
    # not per-cell-dir) and spread into the cells that match.
    ch_path = os.path.join(data_root, 'paper', 'calibration',
                           'channel_marginal', 'channel_marginal_nonlinearity.csv')
    ch_rows = _read_csv(ch_path)
    for cell in cells.values():
        _ingest_channel_marginal(cell, ch_rows)

    # v0.81.1.7 ship 5: spread paper-2 phase 6-11 per-cell payloads into
    # each cell['measurements']. Each cell is keyed by canonical
    # cell_key (results_schema.cell_key); phase outputs use the same key,
    # so the merge is direct. Missing cells (phase didn't see them) get
    # the relevant measurement keys left absent -- schema is permissive
    # (additionalProperties: True on measurements).
    # v0.82.0.11: Phase 6-11 outputs are keyed in writerbot format
    # (e.g. 'gemma_2b_q4_abliterated_t00') because Phase 6 reads from
    # Q0057 which is also writerbot-keyed. The cells dict here is
    # keyed in canon format (e.g. 'gemma_2b_4bit_abliterated_T0.0').
    # Pre-patch the lookup `ck in (_p8.get('cells') or {})` missed
    # every cell, no Phase 6-11 measurements got merged into
    # results.json, fig6 read empty lambda_sweep, plot was empty.
    # Translate canon -> writerbot before each lookup.
    _SIZE_REMAP_TO_WRITERBOT = {
        '2b_4bit':  '2b_q4',
        '2b_8bit':  '2b_q8',
        '2b_fp16':  '2b_fp16',
        '9b_4bit':  '9b_q4',
        '8b_4bit':  '8b_q4',
    }
    def _ck_to_writerbot(canon_key):
        """canon 'gemma_2b_4bit_abliterated_T0.0' ->
        writerbot 'gemma_2b_q4_abliterated_t00'."""
        parsed = _sch.parse_cell_key(canon_key)
        if not parsed:
            return None
        size_wb = _SIZE_REMAP_TO_WRITERBOT.get(parsed['size'], parsed['size'])
        try:
            t_val = float(parsed['temperature'])
            t_wb = f"t{int(round(t_val * 10)):02d}"
        except (TypeError, ValueError):
            t_wb = 'tnone'
        return f"{parsed['family']}_{size_wb}_{parsed['variant']}_{t_wb}"

    _p6 = _read_apparatus_phase6()
    _p8 = _read_apparatus_phase8()
    _p9 = _read_apparatus_phase9()
    _p10 = _read_apparatus_phase10()
    _p11 = _read_apparatus_phase11()
    _p12 = _read_apparatus_phase12()
    for ck, cell in cells.items():
        m = cell.setdefault('measurements', {})
        wb = _ck_to_writerbot(ck)
        # Phase 6 -- four-class shares on partition B
        if _p6 and wb and wb in (_p6.get('cells') or {}):
            block = _p6['cells'][wb] or {}
            shares = block.get('shares') or {}
            if shares.get('rkhs_median'):
                m['rkhs_shares_partition_b'] = shares['rkhs_median']
            if shares.get('rf'):
                m['rf_shares_partition_b'] = shares['rf']
        # Phase 8 -- apparatus q* + λ-sweep
        if _p8 and wb and wb in (_p8.get('cells') or {}):
            block = _p8['cells'][wb] or {}
            if block.get('q_star_at_lambda_default'):
                m['anchored_shares'] = block['q_star_at_lambda_default']
            if block.get('lambda_sweep'):
                m['lambda_sweep'] = block['lambda_sweep']
        # Phase 9 -- stacking baseline
        if _p9 and wb and wb in (_p9.get('cells') or {}):
            block = _p9['cells'][wb] or {}
            if block.get('stacking_baseline_shares'):
                m['stacking_baseline_shares'] = block['stacking_baseline_shares']
            # Weights are global, not per-cell -- surface in metadata
            # block, not in cell measurements.
        # Phase 10 -- bootstrap variance
        if _p10 and wb and wb in (_p10.get('cells') or {}):
            block = _p10['cells'][wb] or {}
            # v0.82.0.23: cells with variance_source ==
            # 'bootstrap_not_run_single_shot_only' carry 0.0
            # placeholders for the variance fields on disk. Translate
            # those placeholders to None in the merged results.json so
            # downstream consumers can't accidentally treat the
            # placeholder as zero variance. The methods_note field is
            # surfaced alongside so the reason is co-located with the
            # signal.
            vsrc = block.get('variance_source')
            is_placeholder = (
                vsrc == 'bootstrap_not_run_single_shot_only'
            )
            var_keys = (
                'Ridge_var_per_channel',
                'MLP_var_per_channel',
                'RKHS_median_var_per_channel',
                'RF_var_per_channel',
                'anchored_var_per_channel',
            )
            if is_placeholder:
                m['bootstrap_variance'] = {
                    'n_bootstrap':     block.get('n_bootstrap'),
                    'variance_source': vsrc,
                    'methods_note':    block.get('methods_note'),
                    **{k: None for k in var_keys},
                }
            else:
                m['bootstrap_variance'] = {
                    'n_bootstrap':     block.get('n_bootstrap'),
                    'variance_source': vsrc,
                    **{k: block.get(k) for k in var_keys},
                }
        # Phase 11 -- geometric diagnostics (Module 2)
        # v0.82.0.23: schema migration from convex_hull_diagnostic
        # (Ship 12, deprecated) to geometric_diagnostics (eight fields).
        # Per statsbot's migration directive (paper2_apparatus.md):
        #   convex_hull_diagnostic.anchored_in_hull → geometric_diagnostics.q_star_inside_class_hull
        #   convex_hull_diagnostic.hull_signed_distance → geometric_diagnostics.hull_signed_distance
        # Both renames are semantically identical (sign convention
        # preserved). hull_vertices is dropped -- Module 2's eight-field
        # block carries strictly more information.
        if _p11 and wb and wb in (_p11.get('cells') or {}):
            block = _p11['cells'][wb] or {}
            gd_block = {
                k: block.get(k) for k in (
                    'q_star_inside_class_hull',
                    'geometric_mean_inside_class_hull',
                    'anchor_pull_distance',
                    'signed_anchor_pull',
                    'class_hull_volume',
                    'hull_degenerate',
                    'hull_exit_mechanism',
                    'hull_signed_distance',
                )
            }
            # v0.82.0.23: optional bootstrap_variance_decomposition
            # populated when per_resample_shares are available. Surfaced
            # alongside the eight-field block.
            if 'bootstrap_variance_decomposition' in block:
                gd_block['bootstrap_variance_decomposition'] = \
                    block['bootstrap_variance_decomposition']
            m['geometric_diagnostics'] = gd_block

        # v0.82.0.23: partition_b_substitution_diagnostics -- joins
        # Q0057 (pre) + function_class_p2 (post) + kraskov_anchor
        # (Phase 7 anchor) into a per-cell block surfacing §9.2 swing-
        # linearity (per-class ΔE/ΔC/ΔR + ordering booleans) and §9.3
        # C-deflation evidence (G̃ shift, KL deltas, anchor cosine).
        if wb:
            try:
                import os as _os_pbs
                import partition_b_substitution as pbs
                _here_pbs = _os_pbs.path.dirname(
                    _os_pbs.path.abspath(__file__))
                _paper_root_pbs = _os_pbs.path.join(_here_pbs,
                                                       'data', 'paper')
                pbs_block = pbs.build_for_cell(wb, _paper_root_pbs)
                if pbs_block is not None:
                    m['partition_b_substitution_diagnostics'] = pbs_block
            except Exception:
                pass  # graceful: missing source files = no block

        # v0.82.0.23: estimator_joint_R2 -- per-cell joint R² (train +
        # held-out) for Ridge, MLP, RF, RKHS_median at the canonical
        # partition-B operating point (split=0.8, pca=48,
        # PARTITION_B_SEED=42, RIDGE_ALPHA=0.01, MLP_HYPERS canon
        # tanh activation). Surfaces the recoverability-plane
        # linearity-axis citation for paper 1 §5.
        # Distinct from cells.<key>.measurements.linearity_max_gap
        # (Q0042) -- see results_schema.py docstring for relationship.
        if _p12 and wb and wb in (_p12.get('cells') or {}):
            block = _p12['cells'][wb] or {}
            ejr_block = block.get('estimator_joint_R2')
            if ejr_block is not None:
                m['estimator_joint_R2'] = ejr_block

    # Compute derived per cell
    for cell in cells.values():
        compute_derived(cell)

    # Cross-cell aggregates
    cross = compute_cross_cell_aggregates(cells)

    # Synthetic calibration -- V5 passthrough + ridge_bias summary
    synth = None
    v5_path = os.path.join(data_root, 'paper', 'calibration', 'v5',
                           'v5_calibration_results.json')
    synth_v5 = _read_json(v5_path)
    if synth_v5 is not None:
        synth = {'v5': synth_v5}
    # ridge_bias rows
    rb_path = os.path.join(data_root, 'paper', 'calibration', 'ridge_bias',
                           'results.csv')
    rb_rows = _read_csv(rb_path)
    if rb_rows:
        if synth is None:
            synth = {}
        synth['ridge_bias_rows'] = rb_rows

    # Toy nonlinearity
    toy = None
    tn_csv = os.path.join(data_root, 'paper', 'calibration', 'toy_nonlinearity',
                          'results.csv')
    tn_rows = _read_csv(tn_csv)
    tn_summary_path = os.path.join(data_root, 'paper', 'calibration',
                                   'toy_nonlinearity', 'summary.txt')
    tn_summary = None
    if os.path.exists(tn_summary_path):
        try:
            with open(tn_summary_path, 'r', encoding='utf-8-sig') as f:
                tn_summary = f.read()
        except Exception:
            pass
    if tn_rows or tn_summary:
        toy = {'rows': tn_rows, 'summary_text': tn_summary}

    # v0.81.1.7 ship 5: paper2_apparatus_metadata top-level block.
    # Populated from the foundations (selection.json, reliability.json)
    # and apparatus_p2 outputs. Optional in schema; absent when paper-2
    # foundations haven't been fired yet.
    paper2_meta = _read_paper2_foundations()

    out = {
        'schema_version': _sch.SCHEMA_VERSION,
        'iota_version':   iota_version,
        'git_commit':     git_commit,
        'generated_at':   datetime.datetime.now().isoformat(),
        'cells':          cells,
        'cross_cell_aggregates':    cross,
        'synthetic_calibration':    synth,
        'toy_nonlinearity':         toy,
        'methodology_calibration':  methodology_calibration_block or {},
    }
    if paper2_meta is not None:
        out['paper2_apparatus_metadata'] = paper2_meta
    return out


def write_results(results_dict, out_path):
    """Validate then write results.json to out_path. Blocking -- raises
    SchemaValidationError if validation fails, nothing is written."""
    _sch.validate_results(results_dict)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        _sch.dump_results(results_dict, f)
    return out_path
