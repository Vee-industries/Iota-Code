"""
fig2_toy_heatmap.py -- paper figure 2.

3x3 heatmap of Rhat_gap (MLP S-share - Ridge S-share) over the toy's
(nl_S, nl_E) sweep. Diverging colormap centered at zero.

Usage:
  python fig2_toy_heatmap.py
  python fig2_toy_heatmap.py --override
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--override', action='store_true')
    args = ap.parse_args()

    fc.check_calibration_or_exit(override=args.override)
    fc.setup_nature_rcparams()

    results, rhash = fc.load_results()
    toy = results.get('toy_nonlinearity') or {}
    rows = toy.get('rows') or []
    if not rows:
        print("! no toy_nonlinearity rows in results.json -- "
              "did toy_nonlinearity_asymmetry.py run?", file=sys.stderr)
        sys.exit(2)

    import matplotlib.pyplot as plt
    import numpy as np

    # Aggregate Rhat_gap per (nl_S, nl_E) across seeds
    grid = {}
    for row in rows:
        try:
            s = float(row.get('nl_S'))
            e = float(row.get('nl_E'))
            g = float(row.get('Rhat_gap'))
        except (TypeError, ValueError):
            continue
        grid.setdefault((s, e), []).append(g)

    s_levels = sorted({s for (s, _) in grid})
    e_levels = sorted({e for (_, e) in grid})
    if not s_levels or not e_levels:
        print("! no (nl_S, nl_E) rows parsed", file=sys.stderr)
        sys.exit(2)

    mat = np.full((len(s_levels), len(e_levels)), np.nan)
    for i, s in enumerate(s_levels):
        for j, e in enumerate(e_levels):
            vals = grid.get((s, e))
            if vals:
                mat[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(90 * fc.MM_TO_INCHES,
                                    90 * fc.MM_TO_INCHES))
    vmax = max(0.05, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap=fc.COLORS['diverging'], vmin=-vmax, vmax=vmax,
                   aspect='equal', origin='lower')
    ax.set_xticks(range(len(e_levels)))
    ax.set_xticklabels([f'{v:.2f}' for v in e_levels])
    ax.set_yticks(range(len(s_levels)))
    ax.set_yticklabels([f'{v:.2f}' for v in s_levels])
    ax.set_xlabel(r'$nl_E$ (E-channel nonlinearity)')
    ax.set_ylabel(r'$nl_S$ (S-channel nonlinearity)')

    # Cell annotations
    for i in range(len(s_levels)):
        for j in range(len(e_levels)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = 'white' if abs(v) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                    fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\hat{R}_{MLP} - \hat{R}_{Ridge}$', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.text(0.02, 0.98, 'a', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left')

    fig.tight_layout()

    fc.save_figure(
        fig, 'fig2_toy_heatmap',
        source_fields=[
            'toy_nonlinearity.rows[*].nl_S',
            'toy_nonlinearity.rows[*].nl_E',
            'toy_nonlinearity.rows[*].Rhat_gap',
        ],
        caption_draft=(
            'Ridge-vs-MLP share-gap under channel-marginal nonlinearity. '
            'Negative values (red): Ridge over-attributes to S, the pattern '
            'produced when E carries nonlinear structure Ridge cannot see. '
            'Mechanism: permuting a nonlinearly-contributing E on a Ridge '
            'fit loses less R^2 than under an MLP fit, inflating Ridge R.'
        ),
        results_hash=rhash,
    )
    print(f"fig2 written. results.json hash: {rhash[:16]}...")


if __name__ == '__main__':
    main()
