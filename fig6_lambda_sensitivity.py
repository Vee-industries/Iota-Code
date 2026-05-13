"""
fig6_lambda_sensitivity.py -- paper 2 figure: λ-sensitivity sweep.

v0.81.1.7 ship 5. Single-panel line chart: anchored q*_R vs λ for each
of the four empirical cells. λ axis on log scale. Reads
cell['measurements']['lambda_sweep'] (populated by Phase 8 via
results_builder Ship 5).

Usage:
  python fig6_lambda_sensitivity.py
  python fig6_lambda_sensitivity.py --override

If lambda_sweep isn't in any cells (Phase 8 hasn't fired), the script
exits with a clear notice -- no silent empty plot.
"""
import argparse
import math
import os
import sys

import figures_common as fc

# Force UTF-8 stdout (Windows safety)
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


CONFIGS = [
    ('llama_8b_4bit_abliterated',  'LLaMA 3 8B Q4',  'llama_8b'),
    ('gemma_9b_4bit_abliterated',  'Gemma 2 9B Q4',  'gemma_9b'),
    ('gemma_2b_fp16_abliterated',  'Gemma 2 2B FP16', 'gemma_2b_fp16'),
    ('gemma_2b_4bit_abliterated',  'Gemma 2 2B Q4',  'gemma_2b_q4'),
]
# v0.82.0.11: temps are paper-1 canon (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
# Pre-patch fig6 used [0.0, 0.4, 0.7, 1.0, 1.3, 1.6] which is fig4's
# trajectory grid; it doesn't match the cells the apparatus actually
# produces. With the wrong temp list, every results.json lookup
# missed and fig6 produced an empty plot even when measurements
# existed.
TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _gather_lambda_sweep(results, config_key, temp):
    """Return list of (lambda_float, q_R_float) tuples for one
    (config, temp) cell. lambda='inf' is mapped to math.inf."""
    cell = (results.get('cells') or {}).get(f'{config_key}_T{temp:.1f}')
    if not cell:
        return []
    sweep = (cell.get('measurements') or {}).get('lambda_sweep')
    if not sweep:
        return []
    out = []
    for entry in sweep:
        lam = entry.get('lambda')
        if lam == 'inf':
            lam_f = float('inf')
        else:
            try:
                lam_f = float(lam)
            except (TypeError, ValueError):
                continue
        q = entry.get('q_star') or {}
        try:
            q_R = float(q.get('R'))
        except (TypeError, ValueError):
            continue
        out.append((lam_f, q_R))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--override', action='store_true',
                        help='Bypass calibration freshness gate')
    args = parser.parse_args()

    fc.check_calibration_or_exit(override=args.override)
    fc.setup_nature_rcparams()
    import matplotlib.pyplot as plt
    import numpy as np

    results, results_hash = fc.load_results()
    fig, ax = plt.subplots(figsize=(180 * fc.MM_TO_INCHES, 90 * fc.MM_TO_INCHES))

    n_cells_drawn = 0
    palette = list(fc.OKABE_ITO.values())[:len(CONFIGS)]
    for ci, (key, label, _) in enumerate(CONFIGS):
        # Collect (λ, q_R) lines per temperature for this config
        for ti, temp in enumerate(TEMPS):
            sweep = _gather_lambda_sweep(results, key, temp)
            if not sweep:
                continue
            sweep.sort(key=lambda p: p[0] if math.isfinite(p[0]) else 1e9)
            xs = []
            ys = []
            for lam_f, q_R in sweep:
                if not math.isfinite(lam_f):
                    # cap inf at 100 for plotting
                    xs.append(100.0)
                else:
                    xs.append(max(lam_f, 0.01))  # log scale needs positive
                ys.append(q_R)
            ax.plot(xs, ys,
                    color=palette[ci], alpha=0.4 + 0.1 * ti,
                    linewidth=0.7,
                    label=label if ti == 0 else None)
            n_cells_drawn += 1

    if n_cells_drawn == 0:
        print("[fig6] No lambda_sweep entries in any cells -- Phase 8 has not")
        print("       fired yet. Run 0058 with the paper-2 layer "
              "(>= v0.81.1.7) generates these.")
        return 1

    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda$ (log scale)')
    ax.set_ylabel(r'$q^*_R$ (anchored R-share)')
    ax.set_title(r'$\lambda$-sensitivity of anchored $q^*_R$ across cells')
    ax.legend(loc='best', fontsize=6, framealpha=0.7)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5,
                linewidth=0.5)
    ax.text(1.05, ax.get_ylim()[1] * 0.95, r'$\lambda_{\rm default}=1$',
            fontsize=6, alpha=0.6, va='top')
    ax.set_ylim(0, 1)

    fc.save_figure(
        fig, 'fig6_lambda_sensitivity',
        source_fields=['cells.<key>.measurements.lambda_sweep'],
        caption_draft=(
            "Anchored R-share q*_R as a function of the apparatus "
            "regularization parameter λ for each empirical cell. "
            "λ=0 = unanchored (q*_i = G_i); λ=∞ = pure anchor (q*_i = p_a,i). "
            "Default λ=1 marked. Temperature lines per cell shown with "
            "increasing alpha; configurations shown with separate colors. "
            "Schema field: cells.<key>.measurements.lambda_sweep."
        ),
        results_hash=results_hash,
    )
    print(f"[fig6] drew {n_cells_drawn} (config, temperature) lines")
    return 0


if __name__ == '__main__':
    sys.exit(main())
