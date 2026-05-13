"""
fig3_mechanism_falsification.py -- paper figure 3.

Two-panel scatter. Four points per panel (the four configurations at T=1.0).
Left: Ridge-to-MLP R-hat drop vs kNN II fraction (redundancy hypothesis).
Right: same drop vs channel-marginal asymmetry (nonlinearity hypothesis).

Usage:
  python fig3_mechanism_falsification.py
  python fig3_mechanism_falsification.py --override
"""

import argparse
import sys

import figures_common as fc

# v0.80.0.20: force UTF-8 stdout/stderr on Windows. Default Python on
# Windows uses cp1252 for stdout, which crashes on Greek letters,
# arrows, and pictographs (⚠ → σ ρ etc.) that calibration scripts use.
# Reconfigure to UTF-8 with replacement chars as final safety net.
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    import io as _io
    if hasattr(_sys.stdout, 'buffer'):
        _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(_sys.stderr, 'buffer'):
        _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')


CONFIGS = [
    ('llama_8b_4bit_abliterated',  'LLaMA 8B Q4',       'llama_8b'),
    ('gemma_9b_4bit_abliterated',  'Gemma 9B Q4',       'gemma_9b'),
    ('gemma_2b_4bit_abliterated',  'Gemma 2B Q4',       'gemma_2b_q4'),
    ('gemma_2b_fp16_abliterated',  'Gemma 2B FP16',     'gemma_2b_fp16'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--override', action='store_true')
    args = ap.parse_args()

    fc.check_calibration_or_exit(override=args.override)
    fc.setup_nature_rcparams()

    results, rhash = fc.load_results()
    cells = results.get('cells', {})

    pts = []  # one per config at T=1.0
    for key_prefix, label, color_key in CONFIGS:
        ck = f'{key_prefix}_T1.0'
        cell = cells.get(ck)
        if cell is None:
            continue
        m = cell.get('measurements', {})
        d = cell.get('derived', {})
        # v0.80.0.30: drop = rhat_mlp_pooled - rhat_ridge.
        # Prefer the gap value directly (definitionally that delta).
        # Fall back to derived rhat_gap_mlp_minus_ridge (same thing,
        # different name from older builders). Last resort: subtract
        # rhat_ridge_insample from rhat_mlp_pooled_mean -- must use
        # pooled, not reference, because the gap is defined against
        # the pooled mean.
        drop = m.get('mlp_vs_ridge_gap_R_pooled')
        if not isinstance(drop, (int, float)):
            drop = d.get('rhat_gap_mlp_minus_ridge')
        if not isinstance(drop, (int, float)):
            r  = m.get('rhat_ridge_insample')
            pm = m.get('rhat_mlp_pooled_mean')
            if isinstance(r, (int, float)) and isinstance(pm, (int, float)):
                drop = pm - r
        ii_knn   = m.get('redundancy_fraction_knn', m.get('ii_fraction_knn'))  # v0.83 deprecation shim
        asym     = m.get('asymmetry')  # channel-marginal
        pts.append({
            'label':  label,
            'color':  fc.COLORS[color_key],
            'drop':   drop,
            'ii_knn': ii_knn,
            'asym':   asym,
        })

    import matplotlib.pyplot as plt
    import numpy as np

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(180 * fc.MM_TO_INCHES, 80 * fc.MM_TO_INCHES))

    # Panel a: redundancy hypothesis
    for p in pts:
        if p['ii_knn'] is None or p['drop'] is None:
            continue
        axL.scatter(p['ii_knn'], p['drop'], s=50, color=p['color'],
                    edgecolor='black', linewidth=0.5, zorder=3)
        axL.annotate(p['label'], (p['ii_knn'], p['drop']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=6)
    axL.axhline(0, color='gray', linewidth=0.4, linestyle='--')
    axL.set_xlabel('kNN redundancy fraction $(I(Y;S) + I(Y;E) - I(Y;S,E)) / I(Y;S,E)$')
    axL.set_ylabel(r'$\hat{R}_{MLP} - \hat{R}_{Ridge}$ at T=1.0')
    axL.text(0.02, 0.98, 'a', transform=axL.transAxes,
             fontsize=8, fontweight='bold', va='top', ha='left')
    axL.set_title('Redundancy hypothesis', fontsize=7, pad=2)

    # Panel b: nonlinearity hypothesis
    for p in pts:
        if p['asym'] is None or p['drop'] is None:
            continue
        axR.scatter(p['asym'], p['drop'], s=50, color=p['color'],
                    edgecolor='black', linewidth=0.5, zorder=3)
        axR.annotate(p['label'], (p['asym'], p['drop']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=6)
    axR.axhline(0, color='gray', linewidth=0.4, linestyle='--')
    axR.axvline(0, color='gray', linewidth=0.4, linestyle='--')
    axR.set_xlabel(r'Channel-marginal asymmetry ($gap_E - gap_S$)')
    axR.set_ylabel(r'$\hat{R}_{MLP} - \hat{R}_{Ridge}$ at T=1.0')
    axR.text(0.02, 0.98, 'b', transform=axR.transAxes,
             fontsize=8, fontweight='bold', va='top', ha='left')
    axR.set_title('Channel-marginal nonlinearity hypothesis', fontsize=7, pad=2)

    fig.tight_layout()

    fc.save_figure(
        fig, 'fig3_mechanism_falsification',
        source_fields=[
            'cells.{key}.measurements.ii_fraction_knn',
            'cells.{key}.measurements.asymmetry',
            'cells.{key}.derived.rhat_gap_mlp_minus_ridge',
        ],
        cell_filter={'temperature': 1.0,
                     'configurations': [k for k, _, _ in CONFIGS]},
        caption_draft=(
            'Two mechanism candidates for the Ridge-vs-MLP R-hat gap. '
            '(a) Redundancy hypothesis: the drop should scale with the '
            'kNN-estimated interaction information if E and S carry '
            'overlapping information. (b) Channel-marginal nonlinearity: '
            'the drop should scale with gap_E - gap_S if only one '
            'channel carries nonlinear structure Ridge cannot capture.'
        ),
        results_hash=rhash,
    )
    print(f"fig3 written. results.json hash: {rhash[:16]}...")


if __name__ == '__main__':
    main()
