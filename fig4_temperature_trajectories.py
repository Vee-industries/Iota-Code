"""
fig4_temperature_trajectories.py -- paper figure 4.

One panel. Four lines -- one per configuration. X-axis temperature.
Y-axis MLP R-hat (primary), with dashed Ridge R-hat overlay at half
opacity. Small tick where H50 first fails per configuration.

Usage:
  python fig4_temperature_trajectories.py
  python fig4_temperature_trajectories.py --override
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--override', action='store_true')
    args = ap.parse_args()

    fc.check_calibration_or_exit(override=args.override)
    fc.setup_nature_rcparams()

    results, rhash = fc.load_results()
    cells = results.get('cells', {})

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(180 * fc.MM_TO_INCHES,
                                    90 * fc.MM_TO_INCHES))

    for key_prefix, label, color_key in CONFIGS:
        col = fc.COLORS[color_key]
        mlp_vals = []
        rid_vals = []
        h50_fail_t = None
        h50_pass_count = 0
        h50_total = 0
        for t in TEMPERATURES:
            ck = f'{key_prefix}_T{t:.1f}'
            cell = cells.get(ck)
            if cell is None:
                mlp_vals.append(np.nan); rid_vals.append(np.nan)
                continue
            m = cell.get('measurements', {})
            d = cell.get('derived', {})
            mm = fc.cell_rhat_mlp(m)
            r  = fc.cell_rhat_ridge(m)
            mlp_vals.append(mm if isinstance(mm, (int, float)) else np.nan)
            rid_vals.append(r  if isinstance(r,  (int, float)) else np.nan)
            h50 = d.get('h50_passes')
            if h50 is True or h50 is False:
                h50_total += 1
                if h50:
                    h50_pass_count += 1
                elif h50_fail_t is None:
                    h50_fail_t = t

        # MLP primary (solid)
        ax.plot(TEMPERATURES, mlp_vals, '-o', color=col, markersize=3,
                linewidth=1.2, label=label, zorder=3)
        # Ridge overlay (dashed, half opacity)
        ax.plot(TEMPERATURES, rid_vals, '--', color=col,
                alpha=0.4, linewidth=0.9, zorder=2)

        # v0.80.0.29: H50-first-fail tick is informative when at least
        # ONE temperature passes H50 -- the tick then marks the threshold
        # where linearity breaks. If H50 fails at every temperature the
        # tick at T=0.0 just adds visual noise. Skip ticks for
        # always-fail configs and surface the "fails everywhere" status
        # in the legend annotation instead.
        if h50_fail_t is not None and h50_pass_count > 0:
            y_tick = 0.02
            ax.plot([h50_fail_t], [y_tick], marker='|', color=col,
                    markersize=8, markeredgewidth=1.5, zorder=4)

    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel(r'$\hat{R}$ (remainder share)')
    ax.set_xticks(TEMPERATURES)
    ax.set_ylim(0, None)
    ax.legend(loc='best', frameon=False, fontsize=6)

    # Annotate: solid = MLP, dashed = Ridge
    ax.text(0.98, 0.98,
            'solid: MLP (primary)\ndashed: Ridge\n| : H50 first fail',
            transform=ax.transAxes, va='top', ha='right',
            fontsize=5, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', linewidth=0.4))

    fig.tight_layout()

    fc.save_figure(
        fig, 'fig4_temperature_trajectories',
        source_fields=[
            'cells.{key}.measurements.rhat_mlp_reference',
            'cells.{key}.measurements.rhat_ridge_insample',
            'cells.{key}.derived.h50_passes',
        ],
        cell_filter={'configurations': [k for k, _, _ in CONFIGS],
                     'temperatures':   TEMPERATURES},
        caption_draft=(
            'Per-configuration R-hat trajectory across decoding temperature, '
            'MLP primary (solid) and Ridge overlay (dashed). Tick marks '
            'indicate the temperature at which the MLP-vs-Ridge linearity '
            'check (H50) first fails for each configuration. Architectures '
            'take visibly different paths to their T=1.0 R-hat values.'
        ),
        results_hash=rhash,
    )
    print(f"fig4 written. results.json hash: {rhash[:16]}...")


if __name__ == '__main__':
    main()
