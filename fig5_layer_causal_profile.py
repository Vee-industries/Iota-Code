"""
fig5_layer_causal_profile.py -- paper figure 5.

Four-panel grid (one per configuration). Grouped bars: per-layer output
change rate from Run 0018 single-layer patching, grouped by temperature.

Usage:
  python fig5_layer_causal_profile.py
  python fig5_layer_causal_profile.py --override
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
    ('llama_8b_4bit_abliterated',  'LLaMA 3 8B Q4',     'llama_8b'),
    ('gemma_9b_4bit_abliterated',  'Gemma 2 9B Q4',     'gemma_9b'),
    ('gemma_2b_4bit_abliterated',  'Gemma 2 2B Q4',     'gemma_2b_q4'),
    ('gemma_2b_fp16_abliterated',  'Gemma 2 2B FP16',   'gemma_2b_fp16'),
]
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Fallback layer labels if results.json doesn't expose them -- analysis.py
# Run 0018 uses {L8, L16, L24, L31} on 8B models and similar patterns
# elsewhere.
DEFAULT_LAYERS = ['L8', 'L16', 'L24', 'L31']


def _collect_layers(cells, key_prefix):
    """Discover the set of layer names present across this config's cells."""
    seen = []
    for t in TEMPERATURES:
        ck = f'{key_prefix}_T{t:.1f}'
        cell = cells.get(ck)
        if cell is None:
            continue
        lr = cell.get('measurements', {}).get('layer_output_change_rate')
        if isinstance(lr, dict):
            for k in lr:
                if k not in seen:
                    seen.append(k)
    return seen or DEFAULT_LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--override', action='store_true')
    args = ap.parse_args()

    fc.check_calibration_or_exit(override=args.override)
    fc.setup_nature_rcparams()

    results, rhash = fc.load_results()
    cells = results.get('cells', {})

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    fig, axes = plt.subplots(
        2, 2, figsize=(180 * fc.MM_TO_INCHES, 130 * fc.MM_TO_INCHES),
        sharey=True)
    axes = axes.flatten()

    # Temperature color scale
    t_cmap = mpl.cm.viridis

    # v0.80.0.29: per-cell data presence check up front. If no cell has
    # any layer_output_change_rate populated, render a clear "data not
    # collected" annotation instead of blank axes that look like a
    # rendering failure. Run 0018 (single-layer patching) is the only
    # source for this figure; if it hasn't been collected on any model,
    # the figure has nothing to show and that should read as such.
    any_data = False
    for key_prefix, _, _ in CONFIGS:
        for t in TEMPERATURES:
            ck = f'{key_prefix}_T{t:.1f}'
            cell = cells.get(ck)
            if cell is None:
                continue
            lr = cell.get('measurements', {}).get('layer_output_change_rate')
            if isinstance(lr, dict) and lr:
                any_data = True
                break
        if any_data:
            break

    for panel_idx, (key_prefix, label, _) in enumerate(CONFIGS):
        ax = axes[panel_idx]
        layers = _collect_layers(cells, key_prefix)
        n_l = len(layers)
        xs = np.arange(n_l)
        bar_w = 0.8 / max(len(TEMPERATURES), 1)

        for t_idx, t in enumerate(TEMPERATURES):
            ck = f'{key_prefix}_T{t:.1f}'
            cell = cells.get(ck)
            if cell is None:
                continue
            lr = cell.get('measurements', {}).get('layer_output_change_rate') or {}
            vals = [lr.get(L, np.nan) for L in layers]
            offset = (t_idx - len(TEMPERATURES) / 2 + 0.5) * bar_w
            color = t_cmap(t_idx / max(len(TEMPERATURES) - 1, 1))
            ax.bar(xs + offset, vals, bar_w, color=color,
                   linewidth=0.3, edgecolor='black',
                   label=f'T={t:.1f}' if panel_idx == 0 else None)

        ax.set_xticks(xs)
        ax.set_xticklabels(layers)
        ax.set_title(label, fontsize=7, pad=2)
        ax.axhline(0.05, color='red', linewidth=0.6,
                   linestyle=':', alpha=0.6)  # significance threshold
        ax.text(0.02, 0.98, chr(ord('a') + panel_idx),
                transform=ax.transAxes, fontsize=8, fontweight='bold',
                va='top', ha='left')
        if not any_data:
            ax.text(0.5, 0.5,
                    'Run 0018 not collected\n(layer-isolation patching data unavailable)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=7, color='gray', style='italic')

    for ax in axes[2:]:
        ax.set_xlabel('Layer')
    for ax in (axes[0], axes[2]):
        ax.set_ylabel('Output change rate')

    # Legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.02), ncol=len(TEMPERATURES),
                   frameon=False, fontsize=6)

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    fc.save_figure(
        fig, 'fig5_layer_causal_profile',
        source_fields=[
            'cells.{key}.measurements.layer_output_change_rate',
            'cells.{key}.measurements.layer_binomial_p',
        ],
        cell_filter={'configurations': [k for k, _, _ in CONFIGS],
                     'temperatures':   TEMPERATURES},
        caption_draft=(
            'Single-layer causal patching results (Run 0018). Per-layer '
            'output change rate under noise injection at a single layer, '
            'grouped by decoding temperature. Dashed red line at 0.05 '
            'marks the binomial significance threshold. The L16-at-low-T '
            'vs L31-at-high-T crossover is the paper\'s causal finding.'
        ),
        results_hash=rhash,
    )
    print(f"fig5 written. results.json hash: {rhash[:16]}...")


if __name__ == '__main__':
    main()
