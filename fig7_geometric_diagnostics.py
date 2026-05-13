"""
fig7_geometric_diagnostics.py -- paper-2 figure 7.

2D scatter, one point per cell:
  x-axis: signed_anchor_pull (nats, signed by hull membership)
  y-axis: class_hull_volume (dimensionless area on (E, C) projection)
  color : V5d flag (red = flagged, blue = unflagged)
  outline: hull_exit_mechanism
            none                  -- no outline
            anchor_pull           -- dashed outline
            geometric_mean_exit   -- dotted outline

Reads from the merged results.json (post-Run 0059):
  - cells[*].measurements.geometric_diagnostics (eight-field block)
  - apparatus_metadata.spike.bucket / beta (for caption legend)

Apparatus bucket goes in the figure caption / inset legend, not as a
marker channel. Per spec: four channels per cell on the scatter is
the design ceiling.

Usage:
  python fig7_geometric_diagnostics.py
  python fig7_geometric_diagnostics.py --override
"""

import argparse
import sys

import figures_common as fc

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    import io as _io
    if hasattr(_sys.stdout, 'buffer'):
        _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer,
                                          encoding='utf-8', errors='replace')
    if hasattr(_sys.stderr, 'buffer'):
        _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer,
                                          encoding='utf-8', errors='replace')


CAPTION = (
    "Figure 7. Per-cell geometric diagnostics on the apparatus output. "
    "x-axis: signed anchor pull (nats), KL(q* || G̃) signed by whether "
    "q* lies inside the convex hull of the four class shares "
    "(positive = inside, negative = outside). y-axis: class hull volume "
    "(dimensionless area on the (E, C) simplex projection). Color: V5d "
    "high-spread / am-gm flag (red = flagged, blue = unflagged). "
    "Outline: hull exit mechanism -- solid (none, q* inside hull), "
    "dashed (anchor_pull, q* pulled across hull boundary by anchor), "
    "dotted (geometric_mean_exit, G̃ already outside hull from "
    "near-degenerate class configuration). Apparatus is in the strong "
    "bucket (β = 0.0386, Run 0058 Phase 1 spike): chain-rule-consistent "
    "partition recovery; anchor_bounds schema dormant."
)


def _v5d_color(v5d_flag):
    return '#d62728' if v5d_flag else '#1f77b4'


def _outline_style(mechanism):
    if mechanism == 'anchor_pull':
        return 'dashed'
    if mechanism == 'geometric_mean_exit':
        return 'dotted'
    return None  # 'none' -> no outline


def render(results, out_basename='fig7_geometric_diagnostics'):
    import matplotlib.pyplot as plt

    fc.setup_nature_rcparams()

    cells = (results or {}).get('cells') or {}
    points = []
    for ck, cell in cells.items():
        m = (cell or {}).get('measurements') or {}
        gd = m.get('geometric_diagnostics') or {}
        if gd.get('signed_anchor_pull') is None or \
           gd.get('class_hull_volume') is None:
            continue
        points.append({
            'cell_key':    ck,
            'x':           float(gd['signed_anchor_pull']),
            'y':           float(gd['class_hull_volume']),
            'mechanism':   gd.get('hull_exit_mechanism', 'none'),
            'inside_hull': bool(gd.get('q_star_inside_class_hull', False)),
            'v5d_flagged': bool(_cell_v5d(cell)),
        })

    if not points:
        print("fig7: no cells with geometric_diagnostics data -- "
                "nothing to plot.")
        return None

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Scatter, one call per (mechanism × V5d) combination so legends
    # can reference each style.
    for v5d in (False, True):
        for mech in ('none', 'anchor_pull', 'geometric_mean_exit'):
            sel = [p for p in points
                   if p['v5d_flagged'] == v5d and p['mechanism'] == mech]
            if not sel:
                continue
            xs = [p['x'] for p in sel]
            ys = [p['y'] for p in sel]
            color = _v5d_color(v5d)
            outline_style = _outline_style(mech)
            edge_kwargs = {}
            if outline_style is not None:
                edge_kwargs = {
                    'edgecolors': 'black',
                    'linestyle':  outline_style,
                    'linewidths': 1.0,
                }
            ax.scatter(xs, ys, c=color, s=55, alpha=0.85,
                        marker='o', zorder=3, **edge_kwargs)

    # x=0 line marks hull boundary
    ax.axvline(0.0, color='#888', linewidth=0.6, linestyle='-', zorder=1)

    ax.set_xlabel('signed anchor pull (nats; sign = q* hull membership)')
    ax.set_ylabel('class hull volume (E, C area)')
    ax.grid(True, alpha=0.3, zorder=0)

    # Inline annotations
    ax.text(0.02, 0.98,
              'right of zero: q* inside hull\nleft of zero: q* outside hull',
              transform=ax.transAxes, fontsize=7, verticalalignment='top',
              color='#444',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7,
                          pad=2))

    # Custom legend: color → V5d, line style → mechanism
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                 markersize=8, label='V5d unflagged'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                 markersize=8, label='V5d flagged'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaa',
                 markeredgecolor='black', markeredgewidth=1.0,
                 markersize=8, label='solid: none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaa',
                 markeredgecolor='black', markeredgewidth=1.0,
                 linestyle='dashed', markersize=8, label='dashed: anchor_pull'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaa',
                 markeredgecolor='black', markeredgewidth=1.0,
                 linestyle='dotted', markersize=8,
                 label='dotted: geometric_mean_exit'),
    ]
    ax.legend(handles=legend_handles, loc='best', fontsize=7,
              framealpha=0.9)

    fig.tight_layout()

    return fc.save_figure(
        fig, out_basename,
        source_fields=[
            'cells.<cell>.measurements.geometric_diagnostics.signed_anchor_pull',
            'cells.<cell>.measurements.geometric_diagnostics.class_hull_volume',
            'cells.<cell>.measurements.geometric_diagnostics.hull_exit_mechanism',
            'cells.<cell>.measurements.geometric_diagnostics.q_star_inside_class_hull',
            'cells.<cell>.measurements.v5d_flag_combined',
        ],
        caption_draft=CAPTION,
    )


def _cell_v5d(cell):
    """Pull V5d flag from cell's measurements. Falls back to False."""
    m = (cell or {}).get('measurements') or {}
    flag = m.get('v5d_flag_combined')
    if flag is None:
        # Try alternate paths (apparatus aggregator, etc.)
        flag = m.get('v5d_flag') or m.get('v5d_flagged')
    return bool(flag) if flag is not None else False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--override', action='store_true',
                          help='ignore calibration check')
    args = parser.parse_args()

    fc.check_calibration_or_exit(override=args.override)
    results = fc.load_results()
    out = render(results)
    if out:
        print(f"  fig7: wrote {out['svg']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
