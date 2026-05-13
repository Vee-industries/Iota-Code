"""
fig1_ridge_vs_mlp_by_temperature.py -- paper figure 1.

Four-panel bar chart. One panel per configuration. Each panel shows
paired Ridge R-hat and MLP R-hat across six temperatures.

Usage:
  python fig1_ridge_vs_mlp_by_temperature.py
  python fig1_ridge_vs_mlp_by_temperature.py --override   # run without calibration
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
    ('llama_8b_4bit_abliterated',  'LLaMA 3 8B Q4'),
    ('gemma_9b_4bit_abliterated',  'Gemma 2 9B Q4'),
    ('gemma_2b_4bit_abliterated',  'Gemma 2 2B Q4'),
    ('gemma_2b_fp16_abliterated',  'Gemma 2 2B FP16'),
]
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--override', action='store_true',
                    help='generate without calibration (not publication-valid)')
    args = ap.parse_args()

    fc.check_calibration_or_exit(override=args.override)
    fc.setup_nature_rcparams()

    results, rhash = fc.load_results()
    cells = results.get('cells', {})

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(
        2, 2, figsize=(180 * fc.MM_TO_INCHES, 120 * fc.MM_TO_INCHES),
        sharey=True, sharex=True)
    axes = axes.flatten()

    bar_w = 0.35
    xs = np.arange(len(TEMPERATURES))

    for panel_idx, (key_prefix, label) in enumerate(CONFIGS):
        ax = axes[panel_idx]
        ridge_vals = []
        mlp_vals   = []
        for t in TEMPERATURES:
            cell_key = f'{key_prefix}_T{t:.1f}'
            cell = cells.get(cell_key)
            if cell is None:
                ridge_vals.append(np.nan)
                mlp_vals.append(np.nan)
                continue
            m = cell.get('measurements', {})
            r  = fc.cell_rhat_ridge(m)
            mm = fc.cell_rhat_mlp(m)
            ridge_vals.append(r if isinstance(r, (int, float)) else np.nan)
            mlp_vals.append(mm if isinstance(mm, (int, float)) else np.nan)

        ax.bar(xs - bar_w/2, ridge_vals, bar_w,
               color=fc.COLORS['ridge'], label='Ridge', linewidth=0.4,
               edgecolor='black')
        ax.bar(xs + bar_w/2, mlp_vals, bar_w,
               color=fc.COLORS['mlp'],   label='MLP',   linewidth=0.4,
               edgecolor='black')

        ax.set_title(label, fontsize=7, pad=2)
        ax.set_ylim(0, 0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{t:.1f}' for t in TEMPERATURES])
        # Panel label
        ax.text(0.02, 0.98, chr(ord('a') + panel_idx),
                transform=ax.transAxes, fontsize=8, fontweight='bold',
                va='top', ha='left')

    # Bottom row x labels, left col y labels
    for ax in axes[2:]:
        ax.set_xlabel('Temperature (T)')
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(r'$\hat{R}$ (remainder share)')

    # Single legend across the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=6)

    fig.tight_layout(rect=(0, 0, 1, 0.98))

    fc.save_figure(
        fig, 'fig1_ridge_vs_mlp_by_temperature',
        source_fields=[
            'cells.{key}.measurements.rhat_ridge_insample',
            'cells.{key}.measurements.rhat_mlp_reference',
        ],
        cell_filter={'configurations': [k for k, _ in CONFIGS],
                     'temperatures':   TEMPERATURES},
        caption_draft=(
            'Remainder share (R-hat) under Ridge versus MLP permutation '
            'importance, by decoding temperature, across four model '
            'configurations. The Ridge estimator compresses cross-'
            'architecture differences toward a common ceiling; the MLP '
            'reference preserves genuine architectural spread.'
        ),
        results_hash=rhash,
    )
    print(f"fig1 written. results.json hash: {rhash[:16]}...")


if __name__ == '__main__':
    main()
