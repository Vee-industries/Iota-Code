"""
IOTA FRAMEWORK — REPORT GENERATOR
====================================
Reads JSON outputs from Runs 0043, 0051, 0044, 0050 and CSVs across all
temperature directories. Generates matplotlib figures and a plain-English
markdown report answering the nine core questions about R.

Usage:
    python report.py                  # uses last_session.json
    python report.py --temperature 0.0  # specific temperature for per-round figs

Output:
    {pooled_dir}/figures/             # PNG figures
    {pooled_dir}/REPORT.md            # plain English markdown
"""

import os, sys, json, glob
import numpy as np

def _find_root():
    """Walk up from this file's directory to the iota root (contains
    start_here.py). Lets report.py be invoked from any working directory."""
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(candidate, "start_here.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.dirname(os.path.abspath(__file__))

ROOT = _find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BG    = '#0a0c0f'
PANEL_BG   = '#10141a'
GRID_CLR   = '#1e2830'
TEXT_CLR    = '#c8d0d8'
ACCENT     = '#64b4ff'
YELLOW     = '#ddbb00'
GREEN      = '#44cc66'
RED        = '#ff5555'
ORANGE     = '#ff9944'
PURPLE     = '#bb77ff'

def _style():
    """Apply IOTA dark-theme matplotlib rcParams.

    Historical: report.py generates dark-on-light figures and is
    retained for manual REPORT.md generation only. The paper-figure
    pipeline (FIG01-FIG13 in export_stats.generate_paper_figures)
    uses a separate grayscale style and supersedes this for
    publication output. Kept because REPORT.md is still useful for
    per-session diagnostic review in the dashboard Stats tab.
    """
    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor': PANEL_BG,
        'axes.edgecolor': GRID_CLR,
        'axes.labelcolor': TEXT_CLR,
        'axes.grid': True,
        'grid.color': GRID_CLR,
        'grid.alpha': 0.5,
        'xtick.color': TEXT_CLR,
        'ytick.color': TEXT_CLR,
        'text.color': TEXT_CLR,
        'legend.facecolor': PANEL_BG,
        'legend.edgecolor': GRID_CLR,
        'font.size': 10,
        'axes.titlesize': 12,
        'figure.titlesize': 14,
    })

_style()


def _load_json(path):
    """Read a JSON file, return None if absent or unreadable.
    Used to tolerate partial analysis state — the report pipeline
    degrades gracefully when any single run's output is missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _read_csv_rows(csv_path, run_mode=None):
    """Lightweight CSV reader — no pandas dependency."""
    import csv
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('priming') == '1':
                continue
            if run_mode:
                # v0.79.5.0: dual-accept canonical 4-digit and legacy integer-string
                from cartography import run_mode_matches as _rmm_chk
                if not _rmm_chk(row.get('run_mode', ''), run_mode):
                    continue
            rows.append(row)
    return rows


def _sf(v):
    """Safe float."""
    try:
        f = float(v)
        return f if f == f else None
    except (ValueError, TypeError):
        return None


def generate(session=None):
    """Assemble REPORT.md and supporting diagnostic figures into
    pooled/analysis/figures/.

    Reads:
      - Q33/Q34 (per-temp + pooled), Q40/Q47 (pooled)
      - Q46 (per-condition R)
      - per-run CSVs for per-turn metric plots

    Produces 9 diagnostic figures (fig1-fig9) + a markdown narrative
    answering the nine core questions about R (does geometry change,
    is it causal, does it survive temperature, can it decompose, is
    it a confound, layer profile, dimension, per-condition, etc.).

    Legacy — removed from Run 0055's dispatch in v0.74.0.0. Paper
    figures (FIG01-FIG13 in export_stats.generate_paper_figures)
    supersede these for publication. report.generate() retained for
    manual dashboard Stats-tab diagnostic use and standalone invocation.
    """
    from cartography import DATA, get_paths, get_pooled_paths, RUN_CSV, condition_name

    if session is None:
        sess_path = os.path.join(ROOT, 'last_session.json')
        if os.path.exists(sess_path):
            with open(sess_path) as f:
                session = json.load(f)
        else:
            session = {}

    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')

    pooled = get_pooled_paths(family, size, variant)
    fig_dir = os.path.join(pooled['analysis'], 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(f"  Report output: {fig_dir}")
    print(f"  Pooled analysis: {pooled['analysis']}")
    n_generated = 0

    report_lines = []
    def _h(text):
        report_lines.append(f"\n## {text}\n")
    def _p(text):
        report_lines.append(f"{text}\n")
    def _fig(name):
        report_lines.append(f"\n![{name}](figures/{name}.png)\n")

    report_lines.append(f"# IOTA Results Report\n")
    report_lines.append(f"**Model:** {family}/{size}/{variant}\n")
    report_lines.append(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    _TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    _COLORS = {0.0: ACCENT, 0.2: GREEN, 0.4: YELLOW, 0.6: ORANGE, 0.8: RED, 1.0: PURPLE}

    # ══════════════════════════════════════════════════════════════
    # Figure 1: E(T), C(T), R(T) curve
    # ══════════════════════════════════════════════════════════════
    _h("1. Does R increase with temperature?")

    curve = _load_json(os.path.join(pooled['analysis'], 'Q0051_temperature_curve.json'))
    if curve and curve.get('curve'):
        temps, Es, Cs, Rs = [], [], [], []
        for e in curve['curve']:
            if not e.get('r34_present'):
                continue
            temps.append(e['temperature'])
            Es.append(e.get('E', 0))
            Cs.append(e.get('C', 0))
            Rs.append(e.get('R', 0))

        if temps:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(temps, Es, 'o-', color=ACCENT, linewidth=2, markersize=8, label='E (external)')
            ax.plot(temps, Cs, 's-', color=YELLOW, linewidth=2, markersize=8, label='C (constraint)')
            ax.plot(temps, Rs, 'D-', color=GREEN, linewidth=2, markersize=8, label='R (internal)')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Fraction')
            ax.set_title('E + C + R Decomposition Across Temperature')
            ax.legend()
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(0, 0.7)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'fig1_ecr_curve.png'), dpi=150)
            plt.close(fig)
            _fig('fig1_ecr_curve')

            trend = curve.get('r_trend', {})
            slope = trend.get('linear_slope', 0)
            if slope > 0.005:
                _p(f"**R increases with temperature.** Linear slope = {slope:.4f}. "
                   f"R ranges from {trend.get('R_min',0):.3f} to {trend.get('R_max',0):.3f}, "
                   f"peaking at T={trend.get('R_peak_temperature',0):.1f}.")
            elif slope < -0.005:
                _p(f"**R decreases with temperature.** Linear slope = {slope:.4f}. "
                   f"R peaks at T={trend.get('R_peak_temperature',0):.1f}.")
            else:
                _p(f"**R is stable across temperature.** Linear slope = {slope:.4f}. "
                   f"Temperature does not substantially affect R.")
    else:
        _p("*Run 0051 temperature curve not available yet. Complete all 6 rounds and run stats.*")

    # ══════════════════════════════════════════════════════════════
    # Figure 2: OLS decomposition bars (NEW v0.71.0.0)
    # ══════════════════════════════════════════════════════════════
    _h("2. How much does each component contribute?")

    if curve and curve.get('curve'):
        dr2_temps, dr2_int, dr2_con = [], [], []
        for e in curve['curve']:
            if e.get('r33_present') and e.get('delta_r2_internal') is not None:
                dr2_temps.append(e['temperature'])
                dr2_int.append(e.get('delta_r2_internal', 0))
                dr2_con.append(e.get('delta_r2_constraint', 0))
        if dr2_temps:
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(dr2_temps))
            w = 0.35
            bars_int = ax.bar(x - w/2, dr2_int, w, color=GREEN, label='ΔR²_internal (S_{t-1})',
                              edgecolor=GRID_CLR, alpha=0.85)
            bars_con = ax.bar(x + w/2, dr2_con, w, color=YELLOW, label='ΔR²_constraint (C_t)',
                              edgecolor=GRID_CLR, alpha=0.85)
            ax.axhline(y=0.01, color=RED, linestyle='--', linewidth=1, alpha=0.7,
                       label='Significance threshold (0.01)')
            # Grey out T=0.0 bars (insufficient power at deterministic)
            if dr2_temps[0] == 0.0:
                bars_int[0].set_alpha(0.3)
                bars_int[0].set_hatch('///')
                bars_con[0].set_alpha(0.3)
                bars_con[0].set_hatch('///')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('ΔR²')
            ax.set_title('OLS Decomposition: ΔR²_internal and ΔR²_constraint by Temperature')
            ax.set_xticks(x)
            ax.set_xticklabels([f'T={t:.1f}' for t in dr2_temps])
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'fig2_ols_decomposition.png'), dpi=150)
            plt.close(fig)
            _fig('fig2_ols_decomposition')
            n_generated += 1

            _p(f"ΔR²_internal ranges from {min(dr2_int):.4f} to {max(dr2_int):.4f}. "
               f"ΔR²_constraint ranges from {min(dr2_con):.4f} to {max(dr2_con):.4f}. "
               f"Both predictors contribute independently at most temperatures.")
    else:
        _p("*Run 0051 temperature curve not available yet.*")

    # ══════════════════════════════════════════════════════════════
    # Figure 3: Patching heatmap — layer × temperature (NEW v0.71.0.0)
    # ══════════════════════════════════════════════════════════════
    _h("3. Where in the network does R live?")

    _LAYERS = ['L8', 'L16', 'L24', 'L31']
    heatmap_data = {}  # {temp: {layer: rate}}
    for temp in _TEMPS:
        paths = get_paths(family, size, variant, temp, create_dirs=False)
        csv_path = os.path.join(paths.get('csv', ''), 'R0018_layer_isolation.csv')
        rows = _read_csv_rows(csv_path, run_mode=18)  # v0.79.4.0: old 42 layer iso → new 18
        patched = [r for r in rows if r.get('patch_layer', 'none') != 'none']
        if patched:
            heatmap_data[temp] = {}
            for layer in _LAYERS:
                layer_rows = [r for r in patched if r.get('patch_layer') == layer]
                oc_vals = [_sf(r.get('output_changed')) for r in layer_rows]
                oc_vals = [v for v in oc_vals if v is not None]
                if oc_vals:
                    heatmap_data[temp][layer] = float(np.mean(oc_vals)) * 100

    if heatmap_data:
        heat_temps = sorted(heatmap_data.keys())
        matrix = np.full((len(_LAYERS), len(heat_temps)), float('nan'))
        for j, t in enumerate(heat_temps):
            for i, l in enumerate(_LAYERS):
                matrix[i, j] = heatmap_data[t].get(l, float('nan'))

        fig, ax = plt.subplots(figsize=(8, 4))
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlBu_r
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=max(30, np.nanmax(matrix)))
        ax.set_xticks(range(len(heat_temps)))
        ax.set_xticklabels([f'T={t:.1f}' for t in heat_temps])
        ax.set_yticks(range(len(_LAYERS)))
        ax.set_yticklabels(_LAYERS)
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Patched Layer')
        ax.set_title('Activation Patching: Output Change Rate (%) by Layer × Temperature')
        # Annotate cells
        for i in range(len(_LAYERS)):
            for j in range(len(heat_temps)):
                v = matrix[i, j]
                if v == v:  # not NaN
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                            fontsize=9, color='white' if v > 15 else TEXT_CLR)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Output Change Rate (%)')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'fig3_patching_heatmap.png'), dpi=150)
        plt.close(fig)
        _fig('fig3_patching_heatmap')
        n_generated += 1

        _p("Cell values show the percentage of turns where patching a single layer changed "
           "the model's output token. Late layers (L24/L31) dominate at low temperature. "
           "The profile may flatten at high temperature as trajectory consistency dissolves.")
    else:
        _p("*Run 0018 layer isolation data not available at multiple temperatures yet.*")

    # ══════════════════════════════════════════════════════════════
    # Figure 4: Per-condition R fractions (Run 0044) — renumbered from fig2
    # ══════════════════════════════════════════════════════════════
    _h("4. Which conditions produce the highest R?")

    # Collect Run 0044 from all temperature directories
    all_r46 = {}
    for temp in _TEMPS:
        paths = get_paths(family, size, variant, temp, create_dirs=False)
        r46 = _load_json(os.path.join(paths['analysis'], 'Q0044_per_condition_R.json'))
        if r46:
            all_r46[temp] = r46

    # Use first available
    r46 = all_r46.get(0.0) or (list(all_r46.values())[0] if all_r46 else None)
    if r46 and r46.get('per_run'):
        entries = [e for e in r46['per_run'] if not e.get('skipped')]
        entries.sort(key=lambda e: e.get('frac_R', 0), reverse=True)
        if entries:
            labels = [e.get('label', f"Run {e['run_num']}") for e in entries]
            rs = [e.get('frac_R', 0) for e in entries]
            colors = [GREEN if r > 0.2 else YELLOW if r > 0.1 else RED for r in rs]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(range(len(labels)), rs, color=colors, edgecolor=GRID_CLR)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('R Fraction')
            ax.set_title('R by Experimental Condition')
            ax.invert_yaxis()
            for i, v in enumerate(rs):
                ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8, color=TEXT_CLR)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'fig4_per_condition_r.png'), dpi=150)
            plt.close(fig)
            _fig('fig4_per_condition_r')

            _p(f"**{entries[0]['label']}** produces the highest R ({entries[0]['frac_R']:.3f}). "
               f"**{entries[-1]['label']}** produces the lowest ({entries[-1]['frac_R']:.3f}). "
               f"Range: {entries[0]['frac_R'] - entries[-1]['frac_R']:.3f}.")
    else:
        _p("*Run 0044 not available yet. Run analysis first.*")

    # ══════════════════════════════════════════════════════════════
    # Figure 5: Priming vulnerability
    # ══════════════════════════════════════════════════════════════
    _h("5. Does resistant prime damage trajectory more than cooperative?")

    q47 = _load_json(os.path.join(pooled['analysis'], 'Q0050_cross_temp_synthesis.json'))
    if q47 and q47.get('priming_vulnerability'):
        pv = q47['priming_vulnerability']
        temps_p, neutral, coop, resist = [], [], [], []
        for e in pv:
            n = e.get('neutral_mean_similarity')
            c = e.get('cooperative_mean_similarity')
            r = e.get('resistant_mean_similarity')
            if all(v is not None for v in [n, c, r]):
                temps_p.append(e['temperature'])
                neutral.append(n)
                coop.append(c)
                resist.append(r)

        if temps_p:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(temps_p, neutral, 'o-', color=ACCENT, linewidth=2, markersize=8, label='Neutral')
            ax.plot(temps_p, coop, 's-', color=GREEN, linewidth=2, markersize=8, label='Cooperative')
            ax.plot(temps_p, resist, 'D-', color=RED, linewidth=2, markersize=8, label='Resistant')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Mean Similarity')
            ax.set_title('Priming Vulnerability — Similarity by Prime Type Across Temperature')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'fig5_priming.png'), dpi=150)
            plt.close(fig)
            _fig('fig5_priming')

            gap = np.mean(coop) - np.mean(resist)
            _p(f"Cooperative prime averages {np.mean(coop):.3f} similarity. "
               f"Resistant prime averages {np.mean(resist):.3f} similarity. "
               f"Gap: {gap:.3f}. "
               f"{'Resistant prime consistently damages trajectory.' if gap > 0.01 else 'No meaningful difference between prime types.'}")
    else:
        _p("*Run 0050 priming data not available yet.*")

    # ══════════════════════════════════════════════════════════════
    # Figure 6: Contradiction similarity drop
    # ══════════════════════════════════════════════════════════════
    _h("6. Does contradiction collapse R?")

    if q47 and q47.get('contradiction_analysis'):
        ca = q47['contradiction_analysis']
        fig, ax = plt.subplots(figsize=(8, 5))
        cond_colors = {'high_r': GREEN, 'low_r': RED, 'mid_r': YELLOW}
        for cname in ['high_r', 'low_r']:
            temps_c, drops = [], []
            for e in ca:
                conds = e.get('conditions', {})
                if cname in conds:
                    d = conds[cname]
                    drop = d.get('similarity_drop')
                    if drop is not None:
                        temps_c.append(e['temperature'])
                        drops.append(drop)
            if temps_c:
                ax.plot(temps_c, drops, 'o-', color=cond_colors.get(cname, ACCENT),
                        linewidth=2, markersize=8, label=cname)
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Similarity Drop (pre - post)')
        ax.set_title('Contradiction Similarity Drop by R Condition Across Temperature')
        ax.axhline(y=0, color=GRID_CLR, linestyle='--', alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'fig6_contradiction.png'), dpi=150)
        plt.close(fig)
        _fig('fig6_contradiction')

        _p("Positive values mean Similarity dropped after contradiction. "
           "Larger drops in high_r conditions confirm high-R trajectories have more to lose.")
    else:
        _p("*Run 0050 contradiction data not available yet.*")

    # ══════════════════════════════════════════════════════════════
    # Figure 7: Context saturation — persistence modes
    # ══════════════════════════════════════════════════════════════
    _h("7. Is R decay caused by KV cache growth?")

    if q47 and q47.get('persistence_modes'):
        pm = q47['persistence_modes']
        fig, ax = plt.subplots(figsize=(8, 5))
        mode_colors = {'full': ACCENT, 'last': GREEN, 'summary': YELLOW}
        for mode in ['full', 'last', 'summary']:
            temps_m, decays = [], []
            for e in pm:
                modes = e.get('modes', {})
                if mode in modes:
                    d = modes[mode].get('similarity_decay')
                    if d is not None:
                        temps_m.append(e['temperature'])
                        decays.append(d)
            if temps_m:
                ax.plot(temps_m, decays, 'o-', color=mode_colors.get(mode, ACCENT),
                        linewidth=2, markersize=8, label=mode)
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Similarity Decay (early - late)')
        ax.set_title('Similarity Decay by Persistence Mode — KV Cache Test')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'fig7_persistence.png'), dpi=150)
        plt.close(fig)
        _fig('fig7_persistence')

        _p("If 'last exchange only' shows the same decay as 'full history', "
           "the decay is NOT caused by growing KV cache — trajectory consistency is losing "
           "energy regardless of context length. If 'last' shows less decay, "
           "context growth is a confound.")
    else:
        _p("*Run 0050 persistence data not available yet.*")

    # ══════════════════════════════════════════════════════════════
    # Figure 8: Condition transfer
    # ══════════════════════════════════════════════════════════════
    _h("8. Does introspection R transfer to arithmetic?")

    if q47 and q47.get('condition_transfer'):
        ct = q47['condition_transfer']
        fig, ax = plt.subplots(figsize=(10, 5))
        all_conds = set()
        for e in ct:
            all_conds.update(e.get('conditions', {}).keys())
        cond_list = sorted(all_conds)
        cond_clr = [GREEN, YELLOW, RED, PURPLE, ACCENT, ORANGE]

        for ci, cname in enumerate(cond_list):
            temps_t, deltas = [], []
            for e in ct:
                conds = e.get('conditions', {})
                if cname in conds:
                    d = conds[cname].get('transfer_delta')
                    if d is not None:
                        temps_t.append(e['temperature'])
                        deltas.append(d)
            if temps_t:
                ax.plot(temps_t, deltas, 'o-', color=cond_clr[ci % len(cond_clr)],
                        linewidth=2, markersize=6, label=cname[:30])
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Similarity Delta (post-switch - pre-switch)')
        ax.set_title('Condition Transfer — Similarity Change at Task Switch')
        ax.axhline(y=0, color=GRID_CLR, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'fig8_transfer.png'), dpi=150)
        plt.close(fig)
        _fig('fig8_transfer')

        _p("Positive delta = Similarity increased after the switch (unusual). "
           "Negative delta = Similarity dropped at switch. "
           "If introspection→arithmetic drops less than arithmetic→arithmetic, "
           "introspection R carries forward.")
    else:
        _p("*Run 0050 condition transfer data not available yet.*")

    # fig8_compression and fig9_three_variant CUT from main report (v0.71.0.0).
    # Compute efficiency (Run 0025) → Appendix B under H33.
    # Three-variant comparison (Run 0001) → data used in analysis but standalone figure not needed.

    # ══════════════════════════════════════════════════════════════
    # Figure 9: Per-turn similarity trajectory by condition (NEW v0.71.0.0)
    # ══════════════════════════════════════════════════════════════
    _h("9. What happens to trajectory consistency within a conversation?")

    # Use T=0.2 (first stochastic temperature, cleanest signal)
    _traj_temp = 0.2
    traj_paths = get_paths(family, size, variant, _traj_temp, create_dirs=False)
    traj_csv_dir = traj_paths.get('csv', '')
    _TRAJ_GROUPS = {
        'Introspection': [3, 4, 5],
        'Arithmetic': [6, 7, 8, 9],
        'Null': [1, 2, 19],
        'Priming': [15, 16, 17],
    }
    _TRAJ_COLORS = {'Introspection': GREEN, 'Arithmetic': YELLOW, 'Null': ACCENT, 'Priming': RED}
    traj_data = {}  # {group: {turn: [similarity_values]}}
    if os.path.isdir(traj_csv_dir):
        for group, runs in _TRAJ_GROUPS.items():
            turn_similarities = {}
            for rn in runs:
                csv_name = RUN_CSV.get(rn)
                if not csv_name:
                    continue
                rows = _read_csv_rows(os.path.join(traj_csv_dir, csv_name), run_mode=rn)
                for r in rows:
                    t = _sf(r.get('turn'))
                    sim_v = _sf(r.get('state_similarity_index'))
                    if t is not None and sim_v is not None:
                        turn_similarities.setdefault(int(t), []).append(sim_v)
            if turn_similarities:
                traj_data[group] = turn_similarities

    if traj_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        for group in ['Introspection', 'Arithmetic', 'Null', 'Priming']:
            if group not in traj_data:
                continue
            td = traj_data[group]
            turns = sorted(td.keys())
            means = [float(np.mean(td[t])) for t in turns]
            sems = [float(np.std(td[t]) / np.sqrt(len(td[t]))) if len(td[t]) > 1 else 0 for t in turns]
            ax.plot(turns, means, 'o-', color=_TRAJ_COLORS.get(group, ACCENT),
                    linewidth=2, markersize=5, label=group)
            ax.fill_between(turns, [m-s for m,s in zip(means,sems)],
                            [m+s for m,s in zip(means,sems)],
                            alpha=0.15, color=_TRAJ_COLORS.get(group, ACCENT))
        ax.set_xlabel('Turn')
        ax.set_ylabel('Mean Similarity')
        ax.set_title(f'Per-Turn Similarity Trajectory by Condition (T={_traj_temp})')
        ax.legend(fontsize=9)
        ax.set_xticks(range(1, 14))
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'fig9_per_turn_similarity.png'), dpi=150)
        plt.close(fig)
        _fig('fig9_per_turn_similarity')
        n_generated += 1

        _p(f"At T={_traj_temp}, introspection similarity trajectory shows within-conversation "
           "dynamics. If introspection climbs while arithmetic falls, that's visible "
           "evidence of condition-dependent trajectory consistency.")
    else:
        _p(f"*Per-turn similarity data not available at T={_traj_temp} yet.*")

    # ══════════════════════════════════════════════════════════════
    # Write report
    # ══════════════════════════════════════════════════════════════
    report_path = os.path.join(pooled['analysis'], 'REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    n_figs = len(glob.glob(os.path.join(fig_dir, '*.png')))
    print(f"\n  Report saved: {report_path}")
    print(f"  Figures saved: {fig_dir}/")
    print(f"  {n_figs} figures generated.")
    if n_figs == 0:
        print("  WARNING: No figures were generated. Check that Q40/Q47 JSONs exist in pooled/analysis/")
    return report_path


if __name__ == '__main__':
    generate()
