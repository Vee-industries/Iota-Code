"""
IOTA FRAMEWORK -- EXPORT: STATS + FIGURES
==========================================
v37.2

Figures saved to: {condition_dir}/visuals/
  Named: H{nn}_{short_name}.png   ← one per hypothesis minimum
         FIG_SUMMARY_hypothesis_outcomes.png  ← composite summary

Graphs tell the truth regardless of what the truth is.
A null result graphs as well as a positive one.

E is for export. Everything that ran, summarized.

Standalone:
    python export_stats.py
"""

import os, sys, glob, json, re, warnings

def _find_root():
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
sys.path.insert(0, ROOT)

from cartography import get_paths, run_mode_mask, run_mode_mask_any, run_id_pad
import ui

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Scientific publication style (v0.72.2.0)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'axes.grid': False,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
})

# B&W-friendly palette -- distinguishable in grayscale via markers/hatching
_BW_COLORS = ['#000000', '#555555', '#999999', '#CCCCCC']
_BW_MARKERS = ['o', 's', '^', 'D', 'v', 'P']
_BW_HATCHES = ['', '///', '...', 'xxx', '\\\\\\', '+++']
_BW_LINESTYLES = ['-', '--', ':', '-.']

# Global model style registry -- consistent symbols across ALL figures.
# Key: (family, size). Every figure uses get_model_style() not loop index.
_MODEL_STYLES = {
    ('llama', '8b'):  {'color': '#000000', 'marker': 'o', 'ls': '-',  'hatch': ''},
    ('gemma', '8b'):  {'color': '#555555', 'marker': 's', 'ls': '--', 'hatch': '///'},
    ('gemma', '3b'):  {'color': '#999999', 'marker': '^', 'ls': ':',  'hatch': '...'},
    ('qwen', '0.5b'): {'color': '#333333', 'marker': 'D', 'ls': '-.', 'hatch': 'xxx'},
    ('qwen', '1.5b'): {'color': '#CCCCCC', 'marker': 'v', 'ls': '-',  'hatch': '\\\\\\'},
}

def _get_model_style(model_info, fallback_idx=0):
    """Return consistent style dict for a model. Uses registry, falls back to index."""
    key = (model_info.get('family', ''), model_info.get('size', ''))
    if key in _MODEL_STYLES:
        return _MODEL_STYLES[key]
    i = fallback_idx
    return {'color': _BW_COLORS[i % len(_BW_COLORS)],
            'marker': _BW_MARKERS[i % len(_BW_MARKERS)],
            'ls': _BW_LINESTYLES[i % len(_BW_LINESTYLES)],
            'hatch': _BW_HATCHES[i % len(_BW_HATCHES)]}

import seaborn as sns
from scipy import stats as sp_stats

sns.set_theme(style="whitegrid", font_scale=1.1)
sns.set_palette([_BW_COLORS[0], _BW_COLORS[1], _BW_COLORS[2], _BW_COLORS[3]])
FIGSIZE  = (10, 6)
DPI      = 300
PAL      = {"introspect": "#000000", "null": "#555555",
            "arithmetic": "#999999", "priming": "#BBBBBB",
            "other": "#333333"}

# ── Hypothesis registry ───────────────────────────────────────────────────────
# Each entry: H_id → (label, source_runs, key_metric, direction)

HYPOTHESES = {
    "H01":  ("Geometry Is Condition-Invariant",              [4, 5, 1],        "layer_sim_mean",              "positive"),
    # v0.77.1.0: renamed from "Is Degenerate" -- code supports when 0.5 ≤ entropy
    # ≤ 8.0, i.e. NOT degenerate. Prior name was semantically inverted against
    # the test. Framework convention is "supported = favored finding"; favored
    # here is working-range entropy. (Was H02 in pre-0.77 naming, renumbered
    # H54 in 0.77.0.0 surgery, renamed in 0.77.1.0.)
    "H54":  ("Baseline Entropy Is Not Degenerate",           [4, 5, 1],        "mean_logit_entropy",          "neutral"),
    "H03":  ("Introspection Has No Geometric Effect",        [6, 7, 8],         "state_similarity_index",                   "positive"),
    # v0.77.1.0: renamed from "Is Condition-Invariant" -- code runs ttest with
    # direction='positive' on signal_entropy_ratio. An elevation test cannot
    # falsify invariance; the prior name misdescribed the test. (Was H04 →
    # renumbered H55 in 0.77.0.0 → renamed in 0.77.1.0.)
    "H55":  ("Signal Entropy Ratio Is Elevated By Introspection",   [6, 7, 8],  "signal_entropy_ratio",                   "positive"),
    "H05":  ("Contradiction Has No Geometric Effect",        [28],              "disruption_flag",                    "positive"),
    "H06":  ("Similarity Does Not Decay With Context",              [22],              "state_similarity_index",                   "negative"),
    "H56":  ("No Similarity Drop Early To Late",                    [23],              "state_similarity_index",                   "negative"),
    "H08":  ("Throughline Has No Trajectory Effect",         [13, 14, 15],      "state_similarity_index",                   "positive"),
    "H09":  ("Task Demand Geometry Equals Introspection",    [9, 10, 11, 12],      "state_similarity_index",                   "negative"),
    "H10":  ("Disruption Fully Resets Trajectory",           [39, 40],          "layer_sim_mean",              "positive"),
    "H11":  ("Prior State Adds No Predictive Power",         [25, 27],          "delta_r2_internal",           "positive"),
    "H12":  ("Similarity Is Explained By Token Statistics",         [32],              "state_similarity_index",                   "neutral"),
    "H13":  ("Similarity Collapses Under Temperature Variation",    [3],              "state_similarity_index",                   "positive"),
    # v0.77.1.0: renamed from "Is Layer-Uniform" -- code at the H14 inference
    # block supports when peak_idx >= n_layers/2, a concentration test. A
    # concentration test is not a uniformity test.
    "H14":  ("Cross-Turn Similarity Concentrates In Late Layers", [27],          "layer_sim_mean",              "positive"),
    # v0.77.1.0: renamed from "Is Layer-Uniform" -- code supports when
    # late − early > 0.02, a directional gradient test. Gradient is not
    # uniformity. (Was H15 → renumbered H57 in 0.77.0.0 → renamed in 0.77.1.0.)
    "H57":  ("Turn-1 Similarity Shows Early-To-Late Gradient",      [24],          "layer_sim_turn1_mean",        "neutral"),
    "H16":  ("Permuted Component Adds No Variance",          [32],              "state_similarity_index",                   "positive"),
    # H16 source is Run 0049 (baseline_swap.json). Runs 0004-0002 are upstream (supply the CSV rows
    # that Run 0049 permutes) but are not direct data sources for this hypothesis.
    # dependency_map.py correctly lists data_runs:[32], upstream_runs:[1..20].
    # Prior value list(range(1,32)) = [1..31] was wrong -- BUG-FOUND-4 (fixed v36.5).
    "H17":  ("Consistency Is Fully Explained By System Prompt",[28],              "delta_r2_internal_per_condition", "neutral"),
    "H18":  ("History Mode Has No Trajectory Effect",        [24],              "state_similarity_index",                   "positive"),
    "H19":  ("Two-Instance Coupling Has No Geometric Effect",  [20],              "coupling_score",              "positive"),
    "H20":  ("R Condition Has No Signal-Per-Watt Effect",    [21],              "signal_per_watt",                "positive"),  # OUT OF PAPER SCOPE -- Appendix A only
    "H21":  ("Prior State Adds No Power Over E_t And C_t",  [33],              "delta_r2_internal",           "positive"),
    # H22 REMOVED as standalone hypothesis (v0.71.0.0): E/C/R fractions are a
    # threshold check on Run 0043 output, not an independent test. Fractions kept
    # in _infer_outcomes for Q34 interpretation; H28 is the real validation guard.
    "H23":  ("Hesitation Is Uncorrelated With Disruption",   [38],              "onset_delay_ratio",            "positive"),
    # H24 REMOVED as standalone hypothesis (v0.71.0.0): redundant with H14/H15.
    # All three test the same wrong prediction (peak at middle layers). H14 result
    # reported; H24 data folded into one sentence in appendix.
    "H25":  ("Entropy Shape Is Uncorrelated With R",         [37],              "mean_logit_entropy",          "negative"),
    "H26":  ("R Does Not Persist Across Condition Switch",   [36],              "state_similarity_index",                   "positive"),
    "H27":  ("Output Similarity Is Uncorrelated With R",     [35],              "output_sim_prev",             "positive"),
    "H28":  ("In-Sample Fractions Do Not Generalise",        [41],              "h28_max_fraction_divergence", "negative"),
    # H28: Run 0033 held-out set. Null = in-sample fractions overfit (divergence > 0.10).
    # Supported = max divergence < 0.05 across E/C/R fractions.
    # Metric direction "negative" = lower divergence = better.
    "H29":  ("No Single Layer Is Causally Sufficient",       [18],              "output_change_rate_max",      "positive"),
    # H29: Run 0018 single-layer patching. Null = no individual layer sufficient (all rates <= 2%).
    # Supported = at least one layer shows output_change_rate > 5% (p < 0.05 vs null).
    # Metric: max output_change_rate across L8/L16/L24/L31.
    "H38": (
        "Patched State Does Not Change Output",
        [17],
        "output_change_rate",
        "positive",
    ),
    # ── v32.0 additions -- zero extra data cost ─────────────────────────────────
    "H30":  ("Output Turn-1 Self-Referentiality Grows With Turns",  [39],  "output_sim_turn1",   "positive"),
    # H30: Run 0035 already saves output_sim_turn1 per row. Null: cosine sim of current
    # output embedding to turn-1 output embedding shows no increase over turns.
    # Supported = positive slope (linregress state_similarity_index ~ turn, introspection arm).
    "H31":  ("Arithmetic Accuracy Is Uncorrelated With Trajectory Consistency", [9, 10, 11, 12], "pearsonr_correct_similarity", "positive"),
    # H31: Run 0009-9 save `correct` flag (1/0) per turn. Null: r(state_similarity_index, correct) is
    # not significant. Supported = Pearson r > 0.15 and p < 0.05 -- high-R trajectories
    # produce more accurate answers, demonstrating functional consequence of R.
    "H32":  ("Shock Phrasing Has No Differential Effect On Recovery", [10], "shock_variant_anova_p", "positive"),
    # H32: Run 0039 cycles 4 SHOCK_VARIANTS per trial. Null: recovery_delta (post-shock
    # sim - shock-turn sim) is identical across all four phrasings (ANOVA p ≥ 0.05).
    # Supported = ANOVA non-significant -- recovery is phrasing-robust.
    # Disproven = one or more variants shows significantly different recovery arc.
    # H38: Run 0017 multi-layer activation patching. Null = grafting high-R geometry
    # has no causal effect on output (change rate <= 2%, indistinguishable from noise).
    # Supported = rate >= 10% in partial or full patch_mode (p < 0.05, binomial vs 10%).
    # H38 prediction: >= 10% of turns change output. Prior finding: 15.4% (pre-Bug-M data).
    # output_changed absent from pre-v30.9 R21_patching.csv rows -- degrades gracefully to pending.
    # Re-collect Run 0017 after Run 0006 completes with save_all_layers=True before citing result.
    # ── v35.0 additions -- Phase 4: Coherence Transfer and Contradiction Recovery ───────
    "H33":  ("Condition A Has No Task-Equivalent Compute Advantage", [43], "compute_per_correct_ratio", "negative"),  # OUT OF PAPER SCOPE -- Appendix A only
    # H33: Run 0025. 8 enforcer priming turns (1 token each) + 5 arithmetic turns.
    # Null: compute_per_correct does not differ across conditions, or condition_a
    # correct_rate is significantly lower.
    # Supported: compute_per_correct(condition_a) ≤ 40% of condition_c (p < 0.05)
    # AND correct_rate not significantly lower. This is the compute efficiency measurement.
    # Metric: ratio condition_a / condition_c compute_per_correct (lower = better).
    "H34":  ("Disruption Magnitude Does Not Track Contradiction Response",              [22],  "disruption_magnitude_delta",    "positive"),
    # H34: Run 0028 -- zero new data cost. disruption_magnitude (continuous) at contradiction turn
    # vs pre-contradiction baseline. H05 tests binary disruption_flag clustering; H34 tests
    # the continuous signal magnitude -- is disruption_magnitude a continuous geometric disruption signal?
    # Supported: paired t-test within-trial, disruption_magnitude at turn 7 significantly higher
    # than mean(turns 1–6), p < 0.05 and Δ > 0.01.
    # Implication: if supported, disruption_magnitude is a continuous signal computed from per-turn hidden states.
    "H35a": ("R Level Has No Effect On Contradiction Recovery (Group)",    [44],  "recovery_similarity_delta_group","positive"),
    "H35b": ("R Level Has No Effect On Contradiction Recovery (Per-Trial)", [44],  "recovery_similarity_delta_corr","positive"),
    # H35a: Group comparison -- high_r vs low_r recovery_delta (t-test).
    # H35b: Per-trial correlation -- Pearson r(pre_sim, recovery_delta).
    # Split (v0.71.0.0): these are two different tests giving two different answers.
    # H35a (group) rejects the null at every temperature. H35b (per-trial) does not.
    # Reporting them as one ambiguous result was misleading.
    "H36":  ("Causal Effect Of Patching Is Temperature-Invariant",   [21],  "patch_temp_slope",  "negative"),
    # H36: Run 0017 across all temperature rounds. Null: output_change_rate does not vary
    # with temperature (Kruskal-Wallis p >= 0.05). Supported: output_change_rate decreases
    # monotonically with temperature -- Pearson r(temperature, output_change_rate) < -0.8.
    # Requires temperature written to CSV (v54.2.2+). Pre-v54.2.2 data not usable.
    "H37":  ("Layer Causal Sufficiency Is Temperature-Invariant",    [42],  "layer_temp_interaction_p", "negative"),
    # H37: Run 0018 across all temperature rounds. Null: per-layer output_change_rate profile
    # does not vary with temperature (no layer × temperature interaction).
    # Supported: two-way ANOVA layer × temperature interaction p < 0.05.
    # Profile flattens at high temperature -- late-layer causal dominance dissolves.
    # Requires temperature written to CSV (v54.2.2+). Pre-v54.2.2 data not usable.

    # ── v44.0.0 -- FIND-H09-GAP fix ────────────────────────────────────────────
    "H39": ("Impossibility Has No Geometric Effect",             [29, 30, 31], "state_similarity_index", "positive"),
    # H39: Runs 0029-0031 (impossibility cluster). Null: impossible-input geometry
    # indistinguishable from null baseline [4,5,1]. Supported: significant difference
    # (either direction) found between impossibility runs and null. Disproven: no effect.

    # ── v0.71.0.0 -- H40 random noise control ───────────────────────────────────
    "H40": ("Random Noise Patching Matches Real Patching Effect",  [19], "output_change_rate_noise", "negative"),
    # H40: Run 0019 (random noise patching baseline). Null: random noise patches produce
    # the same output change rate as real geometry patches (Run 0017). If so, Run 0017's
    # causal claim is confounded -- any perturbation would do. Two-proportion z-test:
    # H38 rate vs H40 rate. Supported = H38 rate significantly > H40 rate (p < 0.05).
    # Disproven = rates indistinguishable -- patching effect is non-specific.

    # ── v0.77.0.0 -- planned hypotheses (catalog entries, inference pending) ────
    # These entries define the hypothesis schema for measurements that are
    # specified in IOTA_Hypotheses_v10.md but do not yet have inference blocks
    # implemented in _compute_outcomes(). They will report status="pending"
    # until their inference is written.
    "H22": ("E, C, R Permutation Fractions Are Equal",              [34], "frac_R",                         "positive"),
    "H24": ("Cross-Turn Similarity Has No Layer Locality",          [34], "layer_sim_mean",                 "positive"),
    "H41": ("System Prompt Has No Geometric Effect",                [20], "state_similarity_index",         "positive"),
    "H42": ("Similarity Does Not Accumulate With Turn Count",       [19], "state_similarity_index",         "negative"),
    "H43": ("R Fraction Is Sensitive To Projection Dimension",      [45], "frac_R",                         "negative"),
    "H44": ("R Fraction Is Condition-Invariant",                    [46], "frac_R",                         "positive"),
    "H45": ("Resistant Prime Does Not Damage Trajectory More Than Cooperative", [47], "disruption_flag", "positive"),
    "H46": ("Contradiction Does Not Drop Similarity",               [47], "state_similarity_index",         "positive"),
    "H48": ("R Decay Is Explained By KV Cache Growth",              [47], "state_similarity_index",         "positive"),
    "H49": ("Introspection R Does Not Transfer To Arithmetic",      [47], "state_similarity_index",         "positive"),
    # v0.77.1.0: renamed H50/H51 to null-direction-correct form. Prior names
    # ("Improves ... By More Than 0.01", "Exceeds 5% Of Joint MI") stated the
    # alternative hypothesis, not the null. Framework convention is that the
    # registry name states the null. Code logic unchanged.
    "H50": ("MLP Does Not Improve On Ridge By More Than 0.01",      [42], "mlp_delta_r2",                   "negative"),
    "H51": ("Interaction Information Below 5% Of Joint MI",         [42], "interaction_info_fraction",      "neutral"),
    "H52": ("Disruption Magnitude Is Stationary Across Non-Contradiction Turns", [22], "disruption_magnitude_stationarity", "positive"),
    "H53": ("Cross-Stochasticity Disruption Rate Is Monotonic",     [22, 35], "disruption_flag_temp_monotonicity", "positive"),
    "H58": ("E + C + R Fractions Sum To 1.0 Within Tolerance",      [34], "ecr_sum_deviation",              "neutral"),
    # Appendix A (out of scope for measurement paper) -- schema kept for UI completeness
    "H47": ("Coherence Transfer Is Not Compute-Efficient",          [47], "compute_per_correct_ratio",       "negative"),  # OUT OF PAPER SCOPE -- Appendix A only
}



def _read_csv_robust(fpath: str) -> pd.DataFrame:
    """Read a v52 universal-schema CSV as DataFrame. All rows are header-width."""
    try:
        return pd.read_csv(fpath, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame()


def load_all(csv_dir: str) -> pd.DataFrame:
    dfs = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        df = _read_csv_robust(f)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    if 'priming' in combined.columns:
        combined = combined[combined['priming'].astype(str) != '1']
    # v52: replace "NA" sentinel (structurally inapplicable fields) with NaN
    combined = combined.replace("NA", float('nan'))
    # Known string/JSON columns -- never coerce to numeric
    _STRING_COLS = frozenset({
        'output','prompt','model','source_file',
        'layer_sim_prev_profile','layer_sim_t1_profile',
        'entropy_trajectory','first_token_top50',
        'layer_sim_prev_profile','layer_sim_t1_profile','layer_sim_depth_profile',
        'r_condition','condition','history_mode','confound_condition',
        'instance','temperature_condition','patch_mode','patch_layer','phase',
        'expected',
    })
    # Coerce everything that isn't a known string column
    for col in combined.columns:
        if col not in _STRING_COLS:
            try:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
            except Exception:
                pass
    return combined


def load_granger(ana_dir: str, temperature: float = None) -> dict:
    """Load Granger probe JSON results."""
    results = {}
    for label, run_num in [("A", 47), ("B", 48)]:  # v0.79.4.0: (B) old 27 → new 48
        f = os.path.join(ana_dir, f"Q{run_num:04d}_granger_{label}.json")
        if os.path.exists(f):
            with open(f) as fp:
                results[label] = json.load(fp)
    return results


def load_baseline_swap(ana_dir: str, temperature: float = None) -> dict:
    f = os.path.join(ana_dir, "Q0049_baseline_swap.json")
    if os.path.exists(f):
        with open(f) as fp:
            return json.load(fp)
    return {}


def load_decomposition(ana_dir: str, temperature: float = None) -> dict:
    """Load Run 0042 E+C+R 3-way decomposition results."""
    f = os.path.join(ana_dir, "Q0042_decomposition.json")
    if os.path.exists(f):
        with open(f) as fp:
            return json.load(fp)
    return {}


def load_permutation_sensitivity(ana_dir: str, temperature: float = None) -> dict:
    """Load Run 0043 permutation sensitivity partition results."""
    f = os.path.join(ana_dir, "Q0043_sobol_partition.json")
    if os.path.exists(f):
        with open(f) as fp:
            return json.load(fp)
    return {}


def load_pooled_permutation_sensitivity(pooled_ana_dir: str) -> dict:
    """Load Run 0051 pooled permutation sensitivity results.
    Reads Q40_pooled_sobol.json from the pooled analysis directory
    (.../pooled/analysis/), which is separate from the per-condition
    analysis/ directory used by load_permutation_sensitivity. Bug V fix (v25.7).
    Pooled dir has no temperature prefix -- no T-prefix needed.
    """
    f = os.path.join(pooled_ana_dir, "Q0051_pooled_sobol.json")
    if os.path.exists(f):
        with open(f) as fp:
            return json.load(fp)
    return {}


def save_fig(fig, vis_dir: str, name: str):
    os.makedirs(vis_dir, exist_ok=True)
    path = os.path.join(vis_dir, name + ".png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def _condition_color(run_mode):
    if run_mode in [4, 5, 1]:   return PAL["null"]
    if run_mode in [6, 7, 8]:    return PAL["introspect"]
    if run_mode in [9, 10, 11, 12]: return PAL["arithmetic"]
    if run_mode in [13, 14, 15]: return PAL["priming"]
    return PAL["other"]


def _outcome_color(status):
    return {"supported": "#2ecc71", "disproven": "#e74c3c",
            "inconclusive": "#95a5a6", "pending": "#bdc3c7"}.get(status, "#bdc3c7")


# ── Individual hypothesis figures ─────────────────────────────────────────────

def fig_similarity_by_cluster(df, vis_dir):
    """H01, H03, H09: similarity mean by run cluster."""
    if 'state_similarity_index' not in df.columns or 'run_mode' not in df.columns:
        return None
    grouped = (df.groupby('run_mode')['state_similarity_index']
                 .agg(['mean', 'sem'])
                 .reset_index())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors  = [_condition_color(r) for r in grouped['run_mode']]
    ax.bar(grouped['run_mode'].astype(str), grouped['mean'],
           yerr=grouped['sem'], color=colors, capsize=4)
    ax.set_xlabel("Run")
    ax.set_ylabel("Similarity (mean ± SEM)")
    ax.set_title("H01 / H03 / H09 -- Similarity by Run")
    patches = [mpatches.Patch(color=v, label=k) for k, v in PAL.items()]
    ax.legend(handles=patches, fontsize=8)
    return save_fig(fig, vis_dir, "H01_H03_H09_similarity_by_run")


def fig_similarity_trajectory(df, vis_dir, run_nums, h_id, title):
    """similarity over turns for given runs. Per-condition with overall mean."""
    if 'state_similarity_index' not in df.columns or 'turn' not in df.columns:
        return None
    sub = df[run_mode_mask_any(df['run_mode'], run_nums)].copy()
    if sub.empty:
        return None
    sub['state_similarity_index'] = pd.to_numeric(sub['state_similarity_index'], errors='coerce')
    sub['turn'] = pd.to_numeric(sub['turn'], errors='coerce')
    sub = sub.dropna(subset=['state_similarity_index', 'turn'])
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    # Per-condition thin lines
    # v0.79.4.0: renumbered. Old {3,4,5,1,2,19} → new {6,7,8,4,5,1}.
    _COND_LABELS = {6: 'Intro-Direct', 7: 'Intro-Indirect', 8: 'Intro-Reflective',
                    4: 'Baseline-A', 5: 'Baseline-B', 1: 'Null (3-variant)'}
    from cartography import run_mode_mask as _rmm_es1  # v0.79.5.2: dual-accept
    for ri, rn in enumerate(run_nums):
        rsub = sub[_rmm_es1(sub['run_mode'], rn)]
        if rsub.empty:
            continue
        grp = rsub.groupby('turn')['state_similarity_index'].mean().reset_index()
        c = _BW_COLORS[ri % len(_BW_COLORS)]
        mk = _BW_MARKERS[ri % len(_BW_MARKERS)]
        lbl = _COND_LABELS.get(rn, f'Run {rn}')
        ax.plot(grp['turn'], grp['state_similarity_index'], marker=mk, color=c,
                linewidth=0.8, markersize=4, alpha=0.5, label=lbl)

    # Overall mean + SEM (bold)
    grouped = sub.groupby('turn')['state_similarity_index'].agg(['mean', 'sem']).reset_index()
    ax.plot(grouped['turn'], grouped['mean'], color='black', linewidth=2,
            marker='o', markersize=5, label='Overall mean', zorder=5)
    ax.fill_between(grouped['turn'],
                    grouped['mean'] - grouped['sem'],
                    grouped['mean'] + grouped['sem'],
                    alpha=0.15, color='black')

    ax.set_xlabel("Turn")
    ax.set_ylabel("Similarity")
    ax.set_title(f"{h_id} -- {title}")
    ax.legend(fontsize=7, ncol=2)
    # Auto y-axis: pad around data range
    vals = sub['state_similarity_index'].dropna()
    if len(vals) > 0:
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        pad = max((hi - lo) * 0.1, 0.01)
        ax.set_ylim(lo - pad, hi + pad)
    plt.tight_layout()
    return save_fig(fig, vis_dir, f"{h_id}_{title.lower().replace(' ','_')}")


def fig_disruption_events(df, vis_dir, run_nums, h_id, title):
    """Disruption event rate by turn."""
    if 'disruption_flag' not in df.columns or 'turn' not in df.columns:
        return None
    sub = df[run_mode_mask_any(df['run_mode'], run_nums)]
    if sub.empty:
        return None
    grouped = sub.groupby('turn')['disruption_flag'].mean().reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(grouped['turn'], grouped['disruption_flag'], color=PAL["introspect"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Disruption Event Rate")
    ax.set_title(f"{h_id} -- {title}")
    ax.set_ylim(0, 1)
    return save_fig(fig, vis_dir, f"{h_id}_{title.lower().replace(' ','_')}")


def fig_perturbation_recovery(df, vis_dir):
    """H10: layer_sim_mean across turns for Runs 0039 and 0040, with shock and recovery marked.

    Shows the full arc: pre-shock trajectory, shock-turn drop, and (for Run 0039)
    the recovery window in turns 14-16. Run 0040 shows turns 1-13 with shock at turn 5.
    Both runs plotted on same axes for timing comparison.
    """
    sub = df[run_mode_mask_any(df['run_mode'], [39, 40])]
    if sub.empty or 'layer_sim_mean' not in sub.columns or 'turn' not in sub.columns:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = {39: PAL["introspect"], 40: PAL["priming"]}
    shock_at = {39: 13, 40: 5}
    for run_num in [39, 40]:
        from cartography import run_mode_mask as _rmm_es2  # v0.79.5.2: dual-accept
        rsub = sub[_rmm_es2(sub['run_mode'], run_num)]
        if rsub.empty:
            continue
        grouped = rsub.groupby('turn')['layer_sim_mean'].agg(['mean', 'sem']).reset_index()
        ax.plot(grouped['turn'], grouped['mean'], marker='o', ms=4,
                color=colors[run_num], label=f"Run {run_num}")
        ax.fill_between(grouped['turn'],
                        grouped['mean'] - grouped['sem'],
                        grouped['mean'] + grouped['sem'],
                        alpha=0.15, color=colors[run_num])
        # Mark shock turn
        st = shock_at[run_num]
        shock_row = grouped[grouped['turn'] == st]
        if not shock_row.empty:
            ax.axvline(st, color=colors[run_num], linestyle='--', alpha=0.5,
                       label=f"Run {run_num} shock (t={st})")
    # Mark Run 0039 recovery window
    ax.axvspan(13.5, 16.5, alpha=0.07, color=PAL["arithmetic"], label="Run 0039 recovery window")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Layer Sim Mean (cross-turn cosine)")
    ax.set_title("H10 -- Disruption Fully Resets Trajectory: Shock and Recovery Arc")
    ax.legend(fontsize=8)
    return save_fig(fig, vis_dir, "H10_perturbation_recovery")


def fig_self_reference_phase(df, vis_dir):
    """H05: state_similarity_index before/at/after contradiction turn (contextual geometry view).

    NOT IN PIPELINE -- retained for future figure generation.
    """
    if 'contradiction_turn' not in df.columns:
        return None
    sub = df[run_mode_mask(df['run_mode'], 22)]
    if sub.empty:
        return None
    sub = sub.copy()
    sub['phase'] = 'pre'
    sub.loc[sub['contradiction_turn'] == 1, 'phase'] = 'contradiction'
    sub.loc[sub.get('post_contradiction', pd.Series(0, index=sub.index)) == 1, 'phase'] = 'post'
    order  = ['pre', 'contradiction', 'post']
    means  = sub.groupby('phase')['state_similarity_index'].mean().reindex(order)
    sems   = sub.groupby('phase')['state_similarity_index'].sem().reindex(order)
    colors = [PAL["null"], PAL["priming"], PAL["arithmetic"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(order, means, yerr=sems, color=colors, capsize=4)
    ax.set_ylabel("State Similarity Index")
    ax.set_title("H05 -- State Similarity by Phase (geometry context; primary test is disruption_flag chi-square)")
    return save_fig(fig, vis_dir, "H05_self_reference_phase")


def fig_disruption_events_by_phase(df, vis_dir):
    """H05 (primary): Disruption event rate before/at/after contradiction turn.

    This figure visualises the ACTUAL test statistic for H05. The chi-square test in
    _infer_outcomes compares disruption_flag at contradiction_turn==1 vs contradiction_turn==0.
    This figure shows the three-way split (pre / contradiction / post) so the reader can
    see whether the elevation is localised to the injection turn or lingers post-turn.
    Added v32.0 to align figure with inference. INF-H05-FIG fix.
    """
    if 'contradiction_turn' not in df.columns or 'disruption_flag' not in df.columns:
        return None
    sub = df[run_mode_mask(df['run_mode'], 22)].copy()
    if sub.empty:
        return None
    sub['phase'] = 'pre'
    sub.loc[sub['contradiction_turn'] == 1, 'phase'] = 'contradiction'
    sub.loc[sub.get('post_contradiction', pd.Series(0, index=sub.index)) == 1, 'phase'] = 'post'
    order  = ['pre', 'contradiction', 'post']
    means  = sub.groupby('phase')['disruption_flag'].mean().reindex(order)
    sems   = sub.groupby('phase')['disruption_flag'].sem().reindex(order)
    colors = [PAL["null"], PAL["priming"], PAL["arithmetic"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(order, means, yerr=sems, color=colors, capsize=4)
    ax.set_ylabel("Disruption Event Rate (proportion of turns)")
    ax.set_ylim(0, min(1.0, (means.max() if not means.isna().all() else 0.5) + 0.2))
    ax.set_title("H05 -- Disruption Event Rate by Phase (primary inference figure)\n"
                 "Contradiction injection expected to cluster disruption events")
    return save_fig(fig, vis_dir, "H05_disruption_events_by_phase")


def fig_saturation(df, vis_dir):
    """H06: similarity decay over 30 turns."""
    sub = df[run_mode_mask(df['run_mode'], 23)]
    if sub.empty or 'state_similarity_index' not in df.columns:
        return None
    grouped = sub.groupby('turn')['state_similarity_index'].agg(['mean','sem']).reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(grouped['turn'], grouped['mean'], marker='o', ms=3, color=PAL["introspect"])
    ax.fill_between(grouped['turn'],
                    grouped['mean'] - grouped['sem'],
                    grouped['mean'] + grouped['sem'],
                    alpha=0.2, color=PAL["introspect"])
    ax.axvline(20, color='#333333', linestyle='--', alpha=0.5, label='Turn 20')
    ax.set_xlabel("Turn")
    ax.set_ylabel("Similarity")
    ax.set_title("H06 -- Similarity Does Not Decay With Context: Similarity Over 30 Turns")
    ax.legend()
    return save_fig(fig, vis_dir, "H06_saturation_similarity_decay")


def fig_temperature(df, vis_dir):
    """H13: similarity by temperature condition."""
    # BUG 3 fix: Run 0003 saves column as 'temperature' not 'temperature_condition'.
    sub = df[run_mode_mask(df['run_mode'], 3)]
    if sub.empty or 'temperature' not in sub.columns:
        return None
    grouped = sub.groupby('temperature')['state_similarity_index'].agg(['mean','sem']).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(grouped['temperature'].astype(str), grouped['mean'],
           yerr=grouped['sem'], capsize=4, color=PAL["arithmetic"])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Similarity")
    ax.set_title("H13 -- Similarity Collapses Under Temperature Variation: Similarity by Temperature")
    return save_fig(fig, vis_dir, "H13_temperature_similarity")


def fig_granger(granger_data, vis_dir):
    """H11: ΔR² internal from Granger probe."""
    if not granger_data:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    x_labels, r2_ext, r2_both, deltas = [], [], [], []
    for label, d in granger_data.items():
        x_labels.append(f"Run {d.get('run_num','?')}\n({label})")
        r2_ext.append(d.get('r2_external_only', 0))
        r2_both.append(d.get('r2_external_plus_internal', 0))
        deltas.append(d.get('delta_r2_internal', 0))
    x = np.arange(len(x_labels))
    w = 0.35
    ax.bar(x - w/2, r2_ext,  width=w, label='R² (E_t only)',    color=PAL["null"])
    ax.bar(x + w/2, r2_both, width=w, label='R² (E_t + S_{t-1})', color=PAL["introspect"])
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_ylabel("R²")
    ax.set_title("H11 -- Prior State Adds No Predictive Power: Granger ΔR² Internal")
    ax.legend()
    for i, d in enumerate(deltas):
        ax.annotate(f"Δ={d:.4f}", xy=(x[i], max(r2_ext[i], r2_both[i]) + 0.01),
                    ha='center', fontsize=9)
    return save_fig(fig, vis_dir, "H11_granger_delta_r2")


def fig_confound(df, vis_dir, decomp=None):
    """H17: dual-panel -- primary: ΔR²_internal per confound condition (Run 0042 decomp),
    secondary: similarity by system prompt condition from Run 0023 CSV (context only).

    FIX-3 (v34.2): prior version plotted only similarity from Run 0023 CSV. The H17
    inference block (INF-1 fix v29.7) reads delta_r2_internal per condition from the
    Run 0042 decomposition JSON -- a completely different metric, different run, different
    pipeline stage. Figure and inference were misaligned: same class of bug fixed for
    H05 (v32.0 FIX-3) and H25 (v32.0 FIX-12). This is the last unresolved case.
    Primary panel now matches what the verdict is actually based on.
    Secondary similarity bars retained with explicit '(context only; not the inference test)'
    label. Primary renders a 'pending' placeholder if decomp data is absent.
    """
    has_decomp = False
    sem_val = neu_val = float('nan')
    if decomp:
        h16_cd = decomp.get('h16_confound_decomposition', {})
        sem_dr2 = h16_cd.get('semantic', {}).get('delta_r2_internal', {})
        neu_dr2 = h16_cd.get('neutral',  {}).get('delta_r2_internal', {})
        sem_val = sem_dr2.get('value', float('nan'))
        neu_val = neu_dr2.get('value', float('nan'))
        has_decomp = not (np.isnan(sem_val) and np.isnan(neu_val))

    sub = df[run_mode_mask(df['run_mode'], 23)] if not df.empty else pd.DataFrame()
    has_csv = not sub.empty and 'confound_condition' in sub.columns

    if not has_decomp and not has_csv:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Primary panel: ΔR²_internal per condition from Run 0042 decomposition ──
    ax0 = axes[0]
    if has_decomp:
        conds  = ['semantic', 'neutral']
        vals   = [sem_val, neu_val]
        colors = [PAL["priming"], PAL["null"]]
        bars   = ax0.bar(conds, vals, color=colors, capsize=4)
        # v34.2: use np.nanmax to avoid order-dependent NaN poisoning of max().
        # max([nan, x]) returns nan; max([x, nan]) returns x -- silent breakage.
        valid_vals = [v for v in vals if not np.isnan(v)]
        _ann_offset = float(np.nanmax(np.abs(valid_vals))) * 0.05 if valid_vals else 0.01
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax0.annotate(f"{v:.4f}", xy=(bar.get_x() + bar.get_width()/2,
                             v + _ann_offset), ha='center', fontsize=9)
        ax0.set_ylabel("ΔR²_internal")
        ax0.set_xlabel("System Prompt Condition")
        ax0.set_title("H17 PRIMARY -- ΔR²_internal per condition\n(Run 0042 decomposition -- inference test)")
        ax0.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    else:
        ax0.text(0.5, 0.5, "Pending: Run 0042 decomposition\nnot yet complete",
                 ha='center', va='center', fontsize=11, color='grey',
                 transform=ax0.transAxes)
        ax0.set_title("H17 PRIMARY -- ΔR²_internal per condition\n(Run 0042 decomposition -- inference test)")
        ax0.set_xticks([])

    # ── Secondary panel: similarity from Run 0023 CSV ──
    ax1 = axes[1]
    if has_csv:
        order   = ['semantic', 'neutral', 'none']
        grouped = sub.groupby('confound_condition')['state_similarity_index'].agg(['mean','sem']).reindex(order).reset_index()
        colors  = [PAL["priming"], PAL["null"], PAL["arithmetic"]]
        ax1.bar(grouped['confound_condition'], grouped['mean'],
                yerr=grouped['sem'], color=colors, capsize=4)
        ax1.set_ylabel("Similarity")
        ax1.set_xlabel("System Prompt Condition")
    else:
        ax1.text(0.5, 0.5, "Pending: Run 0023 not yet complete",
                 ha='center', va='center', fontsize=11, color='grey',
                 transform=ax1.transAxes)
        ax1.set_xticks([])
    ax1.set_title("H17 SECONDARY -- similarity by condition\n(context only; not the inference test)")

    fig.suptitle("H17 -- Coherence Is Fully Explained By System Prompt: C_t Confound Isolation",
                 fontsize=11)
    fig.tight_layout()
    return save_fig(fig, vis_dir, "H17_confound_isolation")


def fig_persistence(df, vis_dir):
    """H18: similarity by history mode (run 0024)."""
    sub = df[run_mode_mask(df['run_mode'], 29)]
    if sub.empty or 'history_mode' not in sub.columns:
        return None
    order   = ['full', 'last', 'summary']
    grouped = sub.groupby('history_mode')['state_similarity_index'].agg(['mean','sem']).reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(grouped['history_mode'], grouped['mean'],
           yerr=grouped['sem'], capsize=4, color=PAL["introspect"])
    ax.set_xlabel("History Mode")
    ax.set_ylabel("Similarity")
    ax.set_title("H18 -- History Mode Has No Trajectory Effect: Similarity by History Mode")
    return save_fig(fig, vis_dir, "H18_persistence_similarity")


def fig_cross_instance(df, vis_dir):
    """H19: Geometric coupling over turns (run 0020).

    NOT IN PIPELINE -- retained for future figure generation.
    """
    sub = df[run_mode_mask(df['run_mode'], 30)]
    if sub.empty or 'coupling_score' not in sub.columns:
        return None
    sub = sub.copy()
    sub['coupling_score'] = pd.to_numeric(sub['coupling_score'], errors='coerce')
    grouped = sub.groupby('turn')['coupling_score'].agg(['mean','sem']).reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(grouped['turn'], grouped['mean'], marker='o', color=PAL["introspect"])
    ax.fill_between(grouped['turn'],
                    grouped['mean'] - grouped['sem'],
                    grouped['mean'] + grouped['sem'],
                    alpha=0.2, color=PAL["introspect"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Coupling Score (cosine)")
    ax.set_title("H19 -- Two-Instance Coupling Has No Geometric Effect: Cross-Instance Coupling Score")
    return save_fig(fig, vis_dir, "H19_cross_instance")


def fig_coherence_levels(df, vis_dir):
    """H20: signal_per_watt by R condition (run 0021).

    NOT IN PIPELINE -- retained for future figure generation.
    """
    sub = df[run_mode_mask(df['run_mode'], 31)]
    if sub.empty or 'signal_per_watt' not in sub.columns:
        return None
    order   = ['high_r', 'mid_r', 'low_r']
    grouped = sub.groupby('r_condition')['signal_per_watt'].agg(['mean','sem']).reindex(order).reset_index()
    colors  = [PAL["introspect"], PAL["null"], PAL["arithmetic"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(grouped['r_condition'], grouped['mean'],
           yerr=grouped['sem'], color=colors, capsize=4)
    ax.set_xlabel("R Condition")
    ax.set_ylabel("Signal / Watt")
    ax.set_title("H20 -- R Condition Has No Signal-Per-Watt Effect: Signal per Watt")
    return save_fig(fig, vis_dir, "H20_power_by_condition")


def fig_sobol(sobol_data, vis_dir, suffix=""):
    """H22: Sobol first-order causal fractions E/C/R."""
    if not sobol_data:
        return None
    se = sobol_data.get('sobol_E', {})
    sc = sobol_data.get('sobol_C', {})
    sr = sobol_data.get('sobol_R', {})
    fracs = [se.get('fraction', float('nan')),
             sc.get('fraction', float('nan')),
             sr.get('fraction', float('nan'))]
    if all(v != v for v in fracs):  # all NaN
        return None
    labels = ['E (external)', 'C (constraint)', 'R (internal)']
    colors = [PAL["null"], PAL["priming"], PAL["introspect"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, fracs, color=colors)
    ax.set_ylabel("Permutation Sensitivity Fraction")
    src_label = "Pooled (Run 0051)" if suffix else "Run 0043"
    ax.set_title(f"H22 -- E, C, R Permutation Fractions Are Equal: Permutation Sensitivity Partition\n({src_label})")
    ax.set_ylim(0, 1)
    for bar_, v in zip(bars, fracs):
        if v == v:
            ax.text(bar_.get_x() + bar_.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    return save_fig(fig, vis_dir, f"H22_sobol_partition{suffix}")


def fig_linearity(decomp, vis_dir):
    """Linearity validation: Ridge vs MLP R² comparison."""
    lc = decomp.get('linearity_check', {}) if decomp else {}
    if not lc or 'error' in lc:
        return None
    ridge = lc.get('ridge_r2_test', float('nan'))
    configs = lc.get('mlp_configs', {})
    if not configs:
        return None
    labels = ['Ridge']
    vals = [ridge]
    colors = [PAL['null']]
    _mlp_colors = [PAL['introspect'], PAL['priming'], PAL['other']]
    for i, (k, v) in enumerate(sorted(configs.items())):
        labels.append(v.get('label', k))
        vals.append(v.get('r2_test', float('nan')))
        colors.append(_mlp_colors[i % len(_mlp_colors)])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("R² (test set)")
    ax.set_title("Linearity Validation -- Ridge vs MLP\n(Model D features: E+C+S → S_next)")
    # Set y-axis to show detail around the values
    valid = [v for v in vals if v == v]
    if valid:
        lo = min(valid) - 0.02
        ax.set_ylim(max(0, lo), 1.0)
    for bar_, v in zip(bars, vals):
        if v == v:
            ax.text(bar_.get_x() + bar_.get_width()/2, v + 0.002,
                    f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    # Draw 0.01 threshold band around Ridge
    if ridge == ridge:
        ax.axhspan(ridge - 0.01, ridge + 0.01, alpha=0.1, color='#888888',
                   label='±0.01 threshold')
        ax.legend(fontsize=8, loc='lower right')
    verdict = lc.get('verdict', '')
    ax.text(0.5, 0.02, verdict, transform=ax.transAxes, ha='center',
            fontsize=7, color='gray', style='italic')
    return save_fig(fig, vis_dir, "linearity_validation")


def fig_interaction_info(decomp, vis_dir):
    """Interaction information: MI decomposition showing ordering sensitivity.
    Shows both linear proxy and kNN non-parametric estimates when available."""
    ii = decomp.get('interaction_info', {}) if decomp else {}
    if not ii or 'error' in ii:
        return None
    knn = ii.get('knn', {})
    knn_pooled = knn.get('pooled', {}) if knn and 'error' not in knn else {}
    has_knn = bool(knn_pooled and 'II' in knn_pooled)

    if has_knn:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for ax, mi_s, mi_e, mi_se, ii_val, ii_f, title in [
            (ax1, ii.get('mi_S', 0), ii.get('mi_E', 0),
             ii.get('mi_SE', 0), ii.get('II', 0), ii.get('II_fraction', 0),
             'Linear Proxy (Gaussian)'),
            (ax2, knn_pooled.get('mi_S', 0), knn_pooled.get('mi_E', 0),
             knn_pooled.get('mi_SE', 0), knn_pooled.get('II', 0), knn_pooled.get('II_fraction', 0),
             f"kNN Non-Parametric (n={knn.get('n_samples', '?')}, d={knn.get('pca_dim', '?')})")]:
            labels = ['Î(S;S_t)', 'Î(S;E_t)', 'Î(S;S,E)', 'II']
            vals = [mi_s, mi_e, mi_se, ii_val]
            colors = [PAL['introspect'], PAL['null'], PAL['priming'],
                      '#AAAAAA' if abs(ii_f) < 0.05 else '#333333']
            bars = ax.bar(labels, vals, color=colors)
            ax.set_ylabel("MI (nats)")
            ax.set_title(title)
            ax.axhline(0, color='gray', linewidth=0.5)
            for bar_, v in zip(bars, vals):
                _off = max(abs(v) * 0.02, 0.01)
                ax.text(bar_.get_x() + bar_.get_width()/2, v + (_off if v >= 0 else -_off),
                        f"{v:.3f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
        agrees = knn.get('agrees_with_linear', True)
        _tag = "AGREE" if agrees else "DISAGREE"
        fig.suptitle(f"Interaction Information -- Methods {_tag}", fontsize=11, y=1.02)
        verdict = ii.get('verdict', '')
        fig.text(0.5, -0.02, verdict, ha='center', fontsize=7, color='gray', style='italic')
        fig.tight_layout()
    else:
        labels = ['Î(S;S_t)', 'Î(S;E_t)', 'Î(S;S,E)', 'II']
        vals = [ii.get('mi_S', 0), ii.get('mi_E', 0), ii.get('mi_SE', 0), ii.get('II', 0)]
        colors = [PAL['introspect'], PAL['null'], PAL['priming'],
                  '#AAAAAA' if abs(ii.get('II_fraction', 1)) < 0.05 else '#333333']
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylabel("MI proxy (nats)")
        ax.set_title("Interaction Information -- Ordering Sensitivity (Linear Proxy Only)")
        ax.axhline(0, color='gray', linewidth=0.5)
        for bar_, v in zip(bars, vals):
            _off = max(abs(v) * 0.02, 0.01)
            ax.text(bar_.get_x() + bar_.get_width()/2, v + (_off if v >= 0 else -_off),
                    f"{v:.3f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
        verdict = ii.get('verdict', '')
        ax.text(0.5, 0.02, verdict, transform=ax.transAxes, ha='center',
                fontsize=7, color='gray', style='italic')
    return save_fig(fig, vis_dir, "interaction_information")


def fig_hesitation(df, vis_dir):
    """H23: Hesitation ratio by turn (run 0038)."""
    sub = df[run_mode_mask(df['run_mode'], 35)]
    if sub.empty or 'onset_delay_ratio' not in sub.columns:
        return None
    grouped = sub.groupby('turn')['onset_delay_ratio'].agg(['mean','sem']).reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(grouped['turn'], grouped['mean'], marker='o', color=PAL["introspect"])
    ax.fill_between(grouped['turn'],
                    grouped['mean'] - grouped['sem'],
                    grouped['mean'] + grouped['sem'],
                    alpha=0.2, color=PAL["introspect"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Onset Delay Ratio (first-token latency / mean interval)")
    ax.set_title("H23 -- Hesitation Is Uncorrelated With Disruption: First-Token Latency vs Turn")
    return save_fig(fig, vis_dir, "H23_onset_delay_ratio")


def fig_layer_depth(df, vis_dir):
    """H24: Layer similarity depth profile -- where does trajectory consistency live?"""
    sub = df[run_mode_mask(df['run_mode'], 36)]
    if sub.empty or 'layer_sim_depth_profile' not in sub.columns:
        return None
    profiles = []
    for raw in sub['layer_sim_depth_profile'].dropna():
        try:
            profiles.append(json.loads(raw))
        except Exception:
            continue
    if not profiles:
        return None
    max_len = max(len(p) for p in profiles)
    matrix  = np.full((len(profiles), max_len), np.nan)
    for i, p in enumerate(profiles):
        matrix[i, :len(p)] = p
    mean_profile = np.nanmean(matrix, axis=0)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(range(len(mean_profile)), mean_profile, color=PAL["introspect"])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Cosine Similarity to Previous Turn")
    ax.set_title("H24 -- Cross-Turn Similarity Has No Layer Locality: Depth Profile")
    return save_fig(fig, vis_dir, "H24_layer_depth_profile")


def fig_entropy_shape(df, vis_dir):
    """H25: Within-turn entropy trajectory -- dual panel: primary=state_similarity_index split, secondary=condition.

    INF-H25-SPLIT fix (v32.0): prior figure split by 'condition' (introspection vs null);
    inference splits by state_similarity_index median. These test different questions and could produce
    contradictory signals without anyone noticing. Fix: primary panel matches inference
    (similarity median split). Secondary panel retains condition split for context.
    """
    sub = df[run_mode_mask(df['run_mode'], 37)]
    if sub.empty or 'entropy_trajectory' not in sub.columns:
        return None
    has_sim = 'state_similarity_index' in sub.columns and not sub['state_similarity_index'].isna().all()
    has_cond = 'condition' in sub.columns

    def _mean_traj(group):
        trajs = []
        for raw in group['entropy_trajectory'].dropna():
            try:
                t = json.loads(raw)
                if len(t) > 1:
                    trajs.append(t)
            except Exception:
                continue
        if not trajs:
            return None
        max_len = max(len(t) for t in trajs)
        mat = np.full((len(trajs), max_len), np.nan)
        for i, t in enumerate(trajs):
            mat[i, :len(t)] = t
        return np.nanmean(mat, axis=0)

    n_panels = 2 if (has_sim and has_cond) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes, None]

    # Panel A (primary) -- state_similarity_index median split
    ax = axes[0]
    if has_sim:
        sim_med = sub['state_similarity_index'].median()
        for group, label, color in [
            (sub[sub['state_similarity_index'] > sim_med],  'high-R (sim>median)', PAL["introspect"]),
            (sub[sub['state_similarity_index'] <= sim_med], 'low-R  (sim≤median)', PAL["null"]),
        ]:
            mt = _mean_traj(group)
            if mt is not None:
                ax.plot(range(len(mt)), mt, label=label, color=color, marker='o', ms=3)
    ax.set_xlabel("Token Position Within Turn")
    ax.set_ylabel("Logit Entropy")
    ax.set_title("H25 -- PRIMARY: Entropy by similarity Median\n(matches inference test)")
    ax.legend()

    # Panel B (secondary) -- condition split
    if axes[1] is not None and has_cond:
        ax2 = axes[1]
        for cond, color in [('introspection', PAL["introspect"]), ('null', PAL["null"])]:
            csub = sub[sub['condition'] == cond]
            mt = _mean_traj(csub)
            if mt is not None:
                ax2.plot(range(len(mt)), mt, label=cond, color=color, marker='o', ms=3)
        ax2.set_xlabel("Token Position Within Turn")
        ax2.set_ylabel("Logit Entropy")
        ax2.set_title("H25 -- SECONDARY: Entropy by Condition\n(context only; not the inference test)")
        ax2.legend()

    fig.suptitle("H25 -- Entropy Shape Is Uncorrelated With R", fontsize=12)
    fig.tight_layout()
    return save_fig(fig, vis_dir, "H25_entropy_shape")


def fig_condition_transfer(df, vis_dir):
    """H26: similarity across turns for all 4 arms of run 0036.
    Primary comparison: transfer vs null_to_arithmetic post-switch (difference = pure R inertia).
    Secondary: both vs introspection_only and arithmetic_only baselines.
    """
    sub = df[run_mode_mask(df['run_mode'], 38)]
    if sub.empty or 'condition' not in sub.columns or 'turn' not in sub.columns:
        return None
    grouped = sub.groupby(['condition', 'turn'])['state_similarity_index'].mean().reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style_map = [
        ('introspection_only',  PAL["introspect"],  '-',  'Introspection only'),
        ('arithmetic_only',     PAL["arithmetic"],  '-',  'Arithmetic only'),
        ('transfer',            PAL["priming"],     '-',  'Transfer (intro → arith)'),
        ('null_to_arithmetic',  PAL["null"],        '--', 'Null → arith (shock control)'),
    ]
    for cond, color, ls, label in style_map:
        csub = grouped[grouped['condition'] == cond]
        if csub.empty:
            continue
        ax.plot(csub['turn'], csub['state_similarity_index'], label=label,
                marker='o', color=color, linestyle=ls)
    ax.axvline(x=6.5, color='grey', linestyle=':', linewidth=1, label='Switch (turn 7)')
    ax.set_xlabel("Turn")
    ax.set_ylabel("Similarity")
    ax.set_title("H26 -- R Does Not Persist Across Condition Switch: Similarity by Arm Post-Switch\n"
                 "(transfer − null_to_arith = pure trajectory inertia)")
    ax.legend(fontsize=8)
    return save_fig(fig, vis_dir, "H26_condition_transfer")


def fig_output_similarity(df, vis_dir):
    """H27: Output self-similarity over turns (run 0035)."""
    sub = df[run_mode_mask(df['run_mode'], 39)]
    if sub.empty or 'output_sim_prev' not in sub.columns:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cond, color in [('introspection', PAL["introspect"]), ('null', PAL["null"])]:
        csub = sub[sub.get('condition', pd.Series(dtype=str)) == cond] if 'condition' in sub.columns else pd.DataFrame()
        if csub.empty:
            continue
        grouped = csub.groupby('turn')['output_sim_prev'].agg(['mean','sem']).reset_index()
        ax.plot(grouped['turn'], grouped['mean'], label=cond, color=color, marker='o')
        ax.fill_between(grouped['turn'],
                        grouped['mean'] - grouped['sem'],
                        grouped['mean'] + grouped['sem'],
                        alpha=0.2, color=color)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Output Cosine Similarity to Prior Turn")
    ax.set_title("H27 -- Output Similarity Is Uncorrelated With R: Output Cosine Similarity by Condition")
    ax.legend()
    return save_fig(fig, vis_dir, "H27_output_self_similarity")


def _load_all_temps_run(run_num: int, session: dict = None) -> pd.DataFrame:
    """Load and pool CSV rows for run_num across all temperature directories.

    Used by H36/H37 figures which need cross-temperature data not available
    in the single-round df passed to the figure functions.

    Falls back to empty DataFrame on any error -- figure functions handle this
    gracefully by returning None.
    """
    try:
        from cartography import DATA, get_family_size_dir
        import json as _j
        if session:
            _fam  = session.get('model_family', 'llama')
            _sz   = session.get('model_size', '8b')
            _var  = session.get('model_variant', 'abliterated')
        else:
            _sp = os.path.join(ROOT, 'last_session.json')
            _s  = _j.load(open(_sp)) if os.path.exists(_sp) else {}
            _fam  = _s.get('model_family', 'llama')
            _sz   = _s.get('model_size', '8b')
            _var  = _s.get('model_variant', 'abliterated')
        from cartography import RUN_CSV
        _csv_name = RUN_CSV.get(run_num)
        if not _csv_name:
            return pd.DataFrame()
        _base_v = os.path.join(get_family_size_dir(_fam, _sz), _var)
        _frames = []
        for _cond in (sorted(os.listdir(_base_v)) if os.path.isdir(_base_v) else []):
            if _cond == 'pooled':
                continue
            _f = os.path.join(_base_v, _cond, 'csv', _csv_name)
            if os.path.exists(_f):
                _df = _read_csv_robust(_f)
                if not _df.empty:
                    _frames.append(_df)
        return pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fig_layer_isolation(df18, vis_dir):
    """H29: Per-layer output change rate from Run 0018 (layer causal sufficiency).

    Bar chart: one bar per patch layer (L8/L16/L24/L31), height = output_change_rate.
    Error bars = 95% CI (Wilson interval, binomial proportion).
    Dashed horizontal line at 2% = noise floor threshold.
    Color-coded: bars exceeding 5% threshold highlighted.
    """
    if df18.empty or 'patch_layer' not in df18.columns:
        return None

    # Patched rows only -- exclude none (baseline)
    patched = df18[df18['patch_layer'] != 'none'].copy()
    if patched.empty or 'output_changed' not in patched.columns:
        return None

    layers = ['L8', 'L16', 'L24', 'L31']
    rates, ci_lo, ci_hi = [], [], []

    for layer in layers:
        sub = patched[patched['patch_layer'] == layer]['output_changed'].dropna()
        if len(sub) == 0:
            rates.append(float('nan')); ci_lo.append(0); ci_hi.append(0)
            continue
        n = len(sub); k = sub.sum()
        p = k / n
        # Wilson score 95% CI
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
        rates.append(p * 100)
        ci_lo.append(max(0, (centre - margin) * 100))
        ci_hi.append(min(100, (centre + margin) * 100))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    THRESHOLD = 5.0
    colors = [PAL["introspect"] if r > THRESHOLD else PAL["null"]
              for r in rates]
    x = range(len(layers))
    bars = ax.bar(x, rates, color=colors, alpha=0.8, width=0.6)
    for i, (r, lo, hi) in enumerate(zip(rates, ci_lo, ci_hi)):
        if not np.isnan(r):
            ax.errorbar(i, r, yerr=[[r - lo], [hi - r]],
                        fmt='none', color='black', capsize=4, linewidth=1.5)
    ax.axhline(2.0, color='grey', linestyle='--', linewidth=1,
               label='2% noise floor')
    ax.axhline(THRESHOLD, color='#333333', linestyle=':', linewidth=1,
               label='5% sufficiency threshold')
    ax.set_xticks(list(x))
    ax.set_xticklabels(layers)
    ax.set_xlabel("Patched Layer")
    ax.set_ylabel("Output Change Rate (%)")
    ax.set_title("H29 -- No Single Layer Is Causally Sufficient: Per-Layer Output Change Rate\n"
                 "Blue = causally sufficient (>5%); orange = insufficient")
    ax.legend(fontsize=8)
    return save_fig(fig, vis_dir, "H29_layer_causal_sufficiency")


# ── Summary figure ────────────────────────────────────────────────────────────

def fig_patching_causal(df21, vis_dir):
    """H38: Output change rate per patch_mode from Run 0017 (activation patching).

    Bar chart: partial and full patch modes, height = output_change_rate (%).
    Error bars = 95% CI (Wilson interval). Dashed lines at 2% noise floor and
    10% H38 prediction threshold.
    Returns None gracefully if output_changed column absent (pre-v30.9 data).
    """
    if df21.empty or 'patch_mode' not in df21.columns:
        return None
    patched = df21[df21['patch_mode'].isin(['partial', 'full'])].copy()
    if patched.empty or 'output_changed' not in patched.columns:
        return None
    # Coerce output_changed to numeric -- R21 short rows can have string values
    patched['output_changed'] = pd.to_numeric(patched['output_changed'], errors='coerce')
    patched = patched.dropna(subset=['output_changed'])
    if patched.empty:
        return None

    modes = ['partial', 'full']
    rates, ci_lo, ci_hi, ns = [], [], [], []
    for mode in modes:
        sub = patched[patched['patch_mode'] == mode]['output_changed'].dropna()
        if len(sub) == 0:
            rates.append(float('nan')); ci_lo.append(0); ci_hi.append(0); ns.append(0)
            continue
        n = len(sub); k = int(sub.sum()); p = k / n
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
        rates.append(p * 100)
        ci_lo.append(max(0, (centre - margin) * 100))
        ci_hi.append(min(100, (centre + margin) * 100))
        ns.append(n)

    if all(np.isnan(r) for r in rates):
        return None

    NOISE_FLOOR = 2.0
    PREDICTION  = 10.0
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = [PAL["introspect"] if (not np.isnan(r) and r >= PREDICTION) else PAL["null"]
              for r in rates]
    x = range(len(modes))
    ax.bar(x, [r if not np.isnan(r) else 0 for r in rates], color=colors, alpha=0.8, width=0.5)
    for i, (r, lo, hi) in enumerate(zip(rates, ci_lo, ci_hi)):
        if not np.isnan(r):
            ax.errorbar(i, r, yerr=[[r - lo], [hi - r]],
                        fmt="none", color="black", capsize=4, linewidth=1.5)
            ax.text(i, r + 1.5, f"{r:.1f}% (n={ns[i]})",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(NOISE_FLOOR, color="grey", linestyle="--", linewidth=1,
               label=f"{NOISE_FLOOR}% noise floor")
    ax.axhline(PREDICTION, color="red", linestyle=":", linewidth=1.2,
               label=f"{PREDICTION}% prediction threshold (H38)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.capitalize() for m in modes])
    ax.set_xlabel("Patch mode")
    ax.set_ylabel("Output change rate (%)")
    ax.set_title("H38 -- Run 0017 Activation Patching: Output Change Rate by Mode")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max((r for r in rates if not np.isnan(r)), default=0) + 15, 20))
    plt.tight_layout()
    return save_fig(fig, vis_dir, "H38_patching_output_change")
    # DEAD CODE REMOVED (v36.6 FINDING-1): lines 899–901 were exact duplicates of
    # the three lines above, placed after the return statement -- unreachable by construction.
    # Copy-paste artifact from the original fig authoring. No runtime or inference impact.


def fig_patching_temp_sensitivity(df21, vis_dir):
    """H36: output_change_rate from Run 0017 by temperature round.

    Line chart: x=temperature, y=output_change_rate (%), separate lines for
    partial and full patch modes. Tests whether causal effect declines with temp.
    Requires temperature column (v54.2.2+).
    """
    if df21.empty or 'patch_mode' not in df21.columns:
        return None
    if 'temperature' not in df21.columns or 'output_changed' not in df21.columns:
        return None
    patched = df21[df21['patch_mode'].isin(['partial', 'full'])].copy()
    patched['temperature']    = pd.to_numeric(patched['temperature'],    errors='coerce')
    patched['output_changed'] = pd.to_numeric(patched['output_changed'], errors='coerce')
    patched = patched.dropna(subset=['temperature', 'output_changed'])
    if patched.empty:
        return None
    temps = sorted(patched['temperature'].unique())
    if len(temps) < 2:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for mode, color in [('partial', PAL['introspect']), ('full', PAL['priming'])]:
        rates, xs = [], []
        for t in temps:
            sub = patched[(patched['patch_mode'] == mode) &
                          (patched['temperature'] == t)]['output_changed'].dropna()
            if len(sub) >= 10:
                rates.append(float(sub.mean() * 100))
                xs.append(t)
        if xs:
            ax.plot(xs, rates, marker='o', color=color, label=mode.capitalize(), linewidth=2)
    ax.axhline(10.0, color='#333333', linestyle=':', linewidth=1, label='10% prediction threshold')
    ax.axhline(2.0,  color='grey', linestyle='--', linewidth=1, label='2% noise floor')
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Output Change Rate (%)")
    ax.set_title("H36 -- Causal Effect Of Patching: Output Change Rate by Temperature\n"
                 "Prediction: rate declines monotonically with temperature (r < −0.8)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return save_fig(fig, vis_dir, "H36_patching_temp_sensitivity")


def fig_layer_temp_sensitivity(df18, vis_dir):
    """H37: per-layer output_change_rate from Run 0018 by temperature round.

    Heatmap: rows = layers (L8/L16/L24/L31), columns = temperature rounds,
    cells = output_change_rate (%). Shows whether the layer profile flattens
    at high temperature. Requires temperature column (v54.2.2+).
    """
    if df18.empty or 'patch_layer' not in df18.columns:
        return None
    if 'temperature' not in df18.columns or 'output_changed' not in df18.columns:
        return None
    patched = df18[df18['patch_layer'] != 'none'].copy()
    patched['temperature']    = pd.to_numeric(patched['temperature'],    errors='coerce')
    patched['output_changed'] = pd.to_numeric(patched['output_changed'], errors='coerce')
    patched = patched.dropna(subset=['temperature', 'output_changed'])
    if patched.empty:
        return None
    layers = [l for l in ['L8', 'L16', 'L24', 'L31'] if l in patched['patch_layer'].values]
    temps  = sorted(patched['temperature'].unique())
    if len(layers) < 2 or len(temps) < 2:
        return None

    matrix = np.full((len(layers), len(temps)), np.nan)
    for i, layer in enumerate(layers):
        for j, t in enumerate(temps):
            sub = patched[(patched['patch_layer'] == layer) &
                          (patched['temperature'] == t)]['output_changed'].dropna()
            if len(sub) >= 5:
                matrix[i, j] = float(sub.mean() * 100)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(matrix, aspect='auto', cmap='Greys',
                   vmin=0, vmax=max(20, np.nanmax(matrix)))
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"T={t}" for t in temps])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    for i in range(len(layers)):
        for j in range(len(temps)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.1f}%", ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='Output Change Rate (%)')
    ax.set_xlabel("Temperature Round")
    ax.set_ylabel("Patched Layer")
    ax.set_title("H37 -- Layer Causal Sufficiency: Per-Layer Change Rate by Temperature\n"
                 "Prediction: profile flattens at high temperature (layer × temp interaction p < 0.05)")
    plt.tight_layout()
    return save_fig(fig, vis_dir, "H37_layer_temp_sensitivity")


# ── Phase 4 figures (v35.0) ───────────────────────────────────────────────────

def fig_coherence_transfer(df43, vis_dir):
    """H33: Coherence transfer measurement -- total Joules and correct rate by condition (Run 0025).

    NOT IN PIPELINE -- retained for future figure generation.

    Two panels:
      Left:  mean total_joules per trial per condition (bar + SEM). Shows compute cost difference.
      Right: mean correct_rate per trial per condition (bar + SEM). Shows quality equivalence.
    """
    if df43.empty or 'r_condition' not in df43.columns:
        return None
    required = {'peak_gpu_power', 'elapsed_sec', 'phase'}
    if not required.issubset(df43.columns):
        return None

    # Compute per-trial totals (load_all already excludes priming=1 rows)
    records = []
    for (cond, trial), grp in df43.groupby(['r_condition', 'trial']):
        joules = (grp['peak_gpu_power'] * grp['elapsed_sec']).sum()
        arith  = grp[grp['phase'] == 'arithmetic']
        if 'correct' in arith.columns and len(arith) > 0:
            cr = arith['correct'].dropna().mean()
        else:
            cr = float('nan')
        records.append({'r_condition': cond, 'trial': trial,
                        'total_joules': joules, 'correct_rate': cr})
    if not records:
        return None
    agg = pd.DataFrame(records)

    ORDER   = ['condition_a', 'condition_b', 'condition_c']
    LABELS  = ['Condition A', 'Condition B', 'Condition C']
    COLOURS = ['#000000', '#666666', '#999999']

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, metric, ylabel, title_sfx in [
        (ax0, 'total_joules',  'Total Joules / trial',   'Compute cost per Trial'),
        (ax1, 'correct_rate',  'Arithmetic correct rate', 'Quality (arithmetic turns)'),
    ]:
        for i, cond in enumerate(ORDER):
            vals = agg[agg['r_condition'] == cond][metric].dropna()
            if len(vals) == 0:
                continue
            ax.bar(i, vals.mean(), yerr=vals.sem(),
                   color=COLOURS[i], capsize=4, width=0.6, alpha=0.85)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"H33 -- {title_sfx}")
        ax.set_ylim(bottom=0)

    plt.suptitle("H33 -- Coherence Transfer: Condition A vs Condition C Baseline\n"
                 "(Run 0025: 8 priming + 5 arithmetic turns)", fontsize=11)
    plt.tight_layout()
    return save_fig(fig, vis_dir, "H33_compute_efficiency")


def fig_self_reference_trajectory(df, vis_dir):
    """H34: disruption_magnitude trajectory from Run 0028, contradiction turn highlighted.

    NOT IN PIPELINE -- retained for future figure generation.

    Plots mean disruption_magnitude per turn (1–13) with ±SEM ribbon.
    Vertical line at turn 7 (contradiction). Demonstrates continuous geometric disruption
    signal distinct from binary disruption_flag (H05).
    """
    sub22 = df[run_mode_mask(df['run_mode'], 22)] if not df.empty else pd.DataFrame()
    if sub22.empty or 'disruption_magnitude' not in sub22.columns or 'turn' not in sub22.columns:
        return None

    turns_agg = (sub22.groupby('turn')['disruption_magnitude']
                 .agg(['mean', 'sem']).reset_index())
    if turns_agg.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    turns  = turns_agg['turn'].values
    means  = turns_agg['mean'].values
    sems   = turns_agg['sem'].fillna(0).values

    ax.plot(turns, means, 'o-', color='#000000', linewidth=2, markersize=5,
            label='mean disruption_magnitude')
    ax.fill_between(turns, means - sems, means + sems, alpha=0.2, color='#000000')
    ax.axvline(x=7, color='#333333', linestyle='--', linewidth=1.5,
               label='Contradiction (turn 7)')
    ax.set_xlabel('Turn')
    ax.set_ylabel('disruption_magnitude (sim_drop × entropy_spike)')
    ax.set_title('H34 -- Disruption Magnitude Trajectory: Continuous Geometric Disruption Signal\n'
                 '(Run 0028: contradiction at turn 7; primary test is paired t-test within-trial)')
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, 14))
    plt.tight_layout()
    return save_fig(fig, vis_dir, "H34_self_reference_trajectory")


def fig_contradiction(df44, vis_dir):
    """H35: state_similarity_index trajectory by R condition across contradiction + recovery (Run 0026).

    NOT IN PIPELINE -- retained for future figure generation.

    Three lines (high_r / mid_r / low_r), vertical line at contradiction turn 7,
    shaded recovery window turns 8–13. Visualises whether pre-injection R level
    predicts speed and completeness of trajectory recovery.
    """
    if df44.empty or 'r_condition' not in df44.columns:
        return None
    required = {'turn', 'state_similarity_index', 'r_condition'}
    if not required.issubset(df44.columns):
        return None

    ORDER   = ['high_r', 'mid_r', 'low_r']
    COLOURS = ['#000000', '#666666', '#999999']
    LABELS  = ['High-R', 'Mid-R', 'Low-R']

    fig, ax = plt.subplots(figsize=(10, 5))

    for cond, colour, label in zip(ORDER, COLOURS, LABELS):
        sub = df44[df44['r_condition'] == cond]
        if sub.empty:
            continue
        agg = sub.groupby('turn')['state_similarity_index'].agg(['mean', 'sem']).reset_index()
        if agg.empty:
            continue
        ax.plot(agg['turn'], agg['mean'], 'o-', color=colour,
                linewidth=2, markersize=5, label=label)
        ax.fill_between(agg['turn'],
                        agg['mean'] - agg['sem'].fillna(0),
                        agg['mean'] + agg['sem'].fillna(0),
                        alpha=0.15, color=colour)

    ax.axvline(x=7, color='#424242', linestyle='--', linewidth=1.5,
               label='Contradiction (turn 7)')
    ax.axvspan(7.5, 13.5, alpha=0.06, color='#424242', label='Recovery window')
    ax.set_xlabel('Turn')
    ax.set_ylabel('state_similarity_index (mean cosine sim to turn-1)')
    ax.set_title('H35 -- Contradiction Recovery by R Condition\n'
                 '(Run 0026: contradiction at turn 7; recovery turns 8–13)')
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, 14))
    plt.tight_layout()
    return save_fig(fig, vis_dir, "H35_contradiction")


def fig_summary(outcomes, vis_dir):
    """
    FIG_SUMMARY: One row per hypothesis.
    Columns: status, key metric, implication.
    Color coded: green=supported, red=disproven, grey=inconclusive/pending.
    Bottom: grand synthesis sentence.
    """
    h_ids  = sorted(outcomes.keys())
    n      = len(h_ids)

    fig, ax = plt.subplots(figsize=(14, max(8, n * 0.45 + 2)))
    ax.axis('off')

    col_labels = ["Hypothesis", "Status", "Key Metric", "Implication"]
    col_widths = [0.22, 0.12, 0.18, 0.46]
    col_x      = [0.01, 0.24, 0.37, 0.56]
    row_h      = 0.88 / (n + 1)
    y_start    = 0.93

    # Header
    for cx, cl in zip(col_x, col_labels):
        ax.text(cx, y_start, cl, fontsize=9, fontweight='bold',
                transform=ax.transAxes, va='top')
    ax.axhline(y_start - 0.01, color='black', linewidth=0.8)

    supported = disproven = inconclusive = 0

    for i, h_id in enumerate(h_ids):
        y       = y_start - (i + 1) * row_h
        outcome = outcomes.get(h_id, {})
        status  = outcome.get("status", "pending")
        metric  = outcome.get("metric", HYPOTHESES.get(h_id, ["","","",""])[2])
        impl    = outcome.get("implication", "--")
        label   = HYPOTHESES.get(h_id, [h_id])[0]

        if status == "supported":   supported   += 1
        elif status == "disproven": disproven   += 1
        else:                       inconclusive += 1

        bg_color = _outcome_color(status)
        rect = plt.Rectangle((0, y - row_h * 0.8), 1, row_h * 0.85,
                              transform=ax.transAxes, color=bg_color,
                              alpha=0.15, zorder=0)
        ax.add_patch(rect)

        ax.text(col_x[0], y, f"{h_id}: {label}", fontsize=7.5,
                transform=ax.transAxes, va='top')
        ax.text(col_x[1], y, status.upper(), fontsize=7.5, fontweight='bold',
                color=_outcome_color(status), transform=ax.transAxes, va='top')
        ax.text(col_x[2], y, metric, fontsize=7.5,
                transform=ax.transAxes, va='top')
        ax.text(col_x[3], y, impl[:80], fontsize=7,
                transform=ax.transAxes, va='top', wrap=True)

    # Grand synthesis
    total  = supported + disproven + inconclusive
    y_bot  = y_start - (n + 1.5) * row_h
    synthesis = (
        f"{supported}/{total} hypotheses supported  ·  "
        f"{disproven} disproven  ·  "
        f"{inconclusive} inconclusive  "
    )
    ax.axhline(y_bot + 0.015, color='black', linewidth=0.5)
    ax.text(0.5, y_bot, synthesis, fontsize=9, ha='center',
            transform=ax.transAxes, style='italic')

    ax.set_title("IOTA Framework v0.80.0.33 -- Hypothesis Outcomes", fontsize=12, fontweight='bold', pad=10)
    return save_fig(fig, vis_dir, "FIG_SUMMARY_hypothesis_outcomes")


# ── Paper figures (v0.72.2.0) ─────────────────────────────────────────────────
# These are the figures a reviewer expects. They read from analysis JSONs
# produced by Runs 0043, 0041, 0044, 0045, 0052-52 across all temperatures.

def _load_analysis_json_all_temps(session, filename):
    """Load an analysis JSON from each temperature directory. Returns {temp: data}."""
    from cartography import get_paths, DATA
    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')
    result = {}
    for temp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        paths = get_paths(family, size, variant, temp, create_dirs=False)
        fp = os.path.join(paths.get('analysis', ''), filename)
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    raw = f.read()
                raw = re.sub(r'\bNaN\b', 'null', raw)
                raw = re.sub(r'\bInfinity\b', 'null', raw)
                raw = re.sub(r'\b-Infinity\b', 'null', raw)
                result[temp] = json.loads(raw)
            except Exception:
                pass
    return result


def _discover_model_data(session):
    """Discover all models with completed analysis. Returns [{label, family, size, variant, data_dir}]."""
    from cartography import DATA
    models = []
    if not os.path.isdir(DATA):
        return models
    for fam in sorted(os.listdir(DATA)):
        fam_dir = os.path.join(DATA, fam)
        if not os.path.isdir(fam_dir):
            continue
        for sz in sorted(os.listdir(fam_dir)):
            sz_dir = os.path.join(fam_dir, sz)
            if not os.path.isdir(sz_dir):
                continue
            for var in sorted(os.listdir(sz_dir)):
                var_dir = os.path.join(sz_dir, var)
                if not os.path.isdir(var_dir) or var in ('pooled', 'base', 'instruct'):
                    continue
                # Check if any analysis exists
                has_analysis = False
                for cond in os.listdir(var_dir):
                    ana = os.path.join(var_dir, cond, 'analysis')
                    if os.path.isdir(ana) and os.listdir(ana):
                        has_analysis = True
                        break
                if has_analysis:
                    models.append({
                        'label': f'{fam.capitalize()} {sz.upper()}',
                        'family': fam, 'size': sz, 'variant': var,
                        'data_dir': var_dir,
                    })
    return models


def _load_json_all_temps_for_model(model_info, filename):
    """Load an analysis JSON from each temp directory for a specific model."""
    result = {}
    var_dir = model_info['data_dir']
    for cond in sorted(os.listdir(var_dir)):
        if cond == 'pooled':
            continue
        fp = os.path.join(var_dir, cond, 'analysis', filename)
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    raw = f.read()
                raw = re.sub(r'\bNaN\b', 'null', raw)
                raw = re.sub(r'\bInfinity\b', 'null', raw)
                raw = re.sub(r'\b-Infinity\b', 'null', raw)
                data = json.loads(raw)
                # Derive temperature from condition name
                if cond == 'deterministic':
                    temp = 0.0
                elif cond.startswith('temp_'):
                    temp = float(cond.split('_')[1])
                else:
                    continue
                result[temp] = data
            except Exception:
                pass
    return result


def fig_ecr_stacked_bar(session, vis_dir, single_model=False):
    """PAPER FIG 4: E+C+R grouped bars. single_model=True for pooled scope."""
    if single_model:
        models = []
    else:
        models = _discover_model_data(session)
    if not models:
        # Fallback to current session model
        all_sobol = _load_analysis_json_all_temps(session, 'Q0043_sobol_partition.json')
        if not all_sobol:
            return None
        models = [{'label': session.get('model_name', 'Model'), '_sobol': all_sobol}]
    else:
        for m in models:
            m['_sobol'] = _load_json_all_temps_for_model(m, 'Q0043_sobol_partition.json')
        models = [m for m in models if m['_sobol']]
    if not models:
        return None
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5), squeeze=False, sharey=True)
    for mi, m in enumerate(models):
        ax = axes[0][mi]
        sobol = m['_sobol']
        temps = sorted(sobol.keys())
        E = [sobol[t].get('perm_sens_E', {}).get('fraction', 0) for t in temps]
        C = [sobol[t].get('perm_sens_C', {}).get('fraction', 0) for t in temps]
        R = [sobol[t].get('perm_sens_R', {}).get('fraction', 0) for t in temps]
        x = np.arange(len(temps))
        w = 0.22
        ax.bar(x - w, E, w, label='E (external)', color='#999999', edgecolor='black', linewidth=0.5)
        ax.bar(x, C, w, label='C (constraint)', color='#666666', edgecolor='black', linewidth=0.5, hatch='///')
        ax.bar(x + w, R, w, label='R (internal)', color='#000000', edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t:.1f}' for t in temps], fontsize=8)
        ax.set_title(m['label'], fontsize=11, fontweight='bold')
        if mi == 0:
            ax.set_ylabel('Fraction')
            ax.legend(fontsize=7)
    fig.suptitle('E + C + R Decomposition by Temperature', fontsize=13, y=1.02)
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG04_ECR_stacked_bar")


def fig_r_vs_temperature(session, vis_dir, single_model=False):
    """PAPER FIG 1: R(T) curve. single_model=True for pooled (one model only)."""
    if single_model:
        models = []
    else:
        models = _discover_model_data(session)
    if not models:
        all_sobol = _load_analysis_json_all_temps(session, 'Q0043_sobol_partition.json')
        if not all_sobol:
            return None
        models = [{'label': session.get('model_name', 'Model'), '_sobol': all_sobol}]
    else:
        for m in models:
            m['_sobol'] = _load_json_all_temps_for_model(m, 'Q0043_sobol_partition.json')
        models = [m for m in models if m['_sobol']]
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    for mi, m in enumerate(models):
        sobol = m['_sobol']
        temps = sorted(sobol.keys())
        R_vals = [sobol[t].get('perm_sens_R', {}).get('fraction', 0) for t in temps]
        ci_lo, ci_hi = [], []
        for t in temps:
            bc = sobol[t].get('bootstrap_ci', {})
            idx = temps.index(t)
            ci_lo.append(bc.get('R_ci_lower', R_vals[idx]))
            ci_hi.append(bc.get('R_ci_upper', R_vals[idx]))
        sty = _get_model_style(m, mi)
        c, mk = sty['color'], sty['marker']
        ax.plot(temps, R_vals, marker=mk, color=c, linewidth=1.5, markersize=6,
                linestyle=sty['ls'], label=m['label'], zorder=3)
        ax.fill_between(temps, ci_lo, ci_hi, color=c, alpha=0.08)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('R (Internal Trajectory Fraction)')
    ax.set_title('Internal Causation Fraction Across Temperature\nwith 95% Bootstrap Confidence Intervals')
    ax.legend()
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Auto y-axis: pad 5% around data range
    all_vals = []
    for m in models:
        for t, d in m['_sobol'].items():
            all_vals.append(d.get('perm_sens_R', {}).get('fraction', 0))
            bc = d.get('bootstrap_ci', {})
            if 'R_ci_lower' in bc: all_vals.append(bc['R_ci_lower'])
            if 'R_ci_upper' in bc: all_vals.append(bc['R_ci_upper'])
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 0.1, 0.02)
        ax.set_ylim(max(0, lo - pad), min(1.0, hi + pad))
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG01_R_vs_temperature")


def fig_per_condition_r(session, vis_dir):
    """PAPER FIG 3: Per-condition R bar chart from Run 0044."""
    all_q46 = _load_analysis_json_all_temps(session, 'Q0044_per_condition_R.json')
    # Use first available temperature
    if not all_q46:
        return None
    t0 = sorted(all_q46.keys())[0]
    data = all_q46[t0]
    per_run = data.get('per_run', [])
    valid = [r for r in per_run if not r.get('skipped') and r.get('frac_R') == r.get('frac_R')]
    if not valid:
        return None
    valid.sort(key=lambda r: r.get('frac_R', 0), reverse=True)
    labels = [r.get('label', f"Run {r['run_num']}") for r in valid]
    R_vals = [r.get('frac_R', 0) for r in valid]
    E_vals = [r.get('frac_E', 0) for r in valid]
    C_vals = [r.get('frac_C', 0) for r in valid]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
    ax.barh(y, E_vals, color='white', edgecolor='black', linewidth=0.8, label='E', height=0.6)
    ax.barh(y, C_vals, left=E_vals, color='#888888', edgecolor='black', linewidth=0.5, label='C', height=0.6)
    ax.barh(y, R_vals, left=[e+c for e, c in zip(E_vals, C_vals)], color='#222222', edgecolor='black', linewidth=0.5, label='R', height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Fraction')
    ax.set_title('Per-Condition E+C+R Fractions (Run 0044)')
    ax.legend(loc='lower right', fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG09_per_condition_R")


def fig_ols_bars(session, vis_dir, single_model=False):
    """PAPER FIG 2: OLS ΔR²_internal. single_model=True for pooled scope."""
    if single_model:
        models = []
    else:
        models = _discover_model_data(session)
    if not models:
        all_q33 = _load_analysis_json_all_temps(session, 'Q0042_decomposition.json')
        if not all_q33:
            return None
        models = [{'label': session.get('model_name', 'Model'), '_q33': all_q33}]
    else:
        for m in models:
            m['_q33'] = _load_json_all_temps_for_model(m, 'Q0042_decomposition.json')
        models = [m for m in models if m['_q33']]
    if not models:
        return None
    temps = sorted(set(t for m in models for t in m['_q33'].keys()))
    colors = ['#000000', '#555555', '#999999', '#BBBBBB', '#333333']
    n_models = len(models)
    width = 0.7 / n_models
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(temps))
    for mi, m in enumerate(models):
        vals, sigs = [], []
        for t in temps:
            d = m['_q33'].get(t, {})
            dr2 = d.get('delta_r2_internal', {})
            vals.append(dr2.get('value', 0))
            sigs.append(dr2.get('significant', False))
        offset = (mi - n_models/2 + 0.5) * width
        c = colors[mi % len(colors)]
        bars = ax.bar(x + offset, vals, width, label=m['label'], color=c, alpha=0.85)
        for bi, (bar, sig) in enumerate(zip(bars, sigs)):
            if not sig:
                bar.set_hatch('///')
                bar.set_alpha(0.4)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        '*', ha='center', fontsize=12, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T={t:.1f}' for t in temps])
    ax.set_ylabel('ΔR² (internal trajectory)')
    ax.set_title('OLS ΔR²_internal by Temperature\n(* = p < 0.05, hatched = not significant)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG02_OLS_delta_r2")


def fig_patching_heatmap(session, vis_dir):
    """PAPER FIG 3: Patching heatmap -- layer × temperature, output change rate."""
    df18_all = _load_all_temps_run(18, session)  # v0.79.4.0: old 42 layer iso → new 18
    if df18_all.empty or 'patch_layer' not in df18_all.columns:
        return None
    if 'output_changed' not in df18_all.columns or 'temperature' not in df18_all.columns:
        return None
    patched = df18_all[df18_all['patch_layer'] != 'none'].copy()
    if patched.empty:
        return None
    patched['output_changed'] = pd.to_numeric(patched['output_changed'], errors='coerce')
    patched['temperature'] = pd.to_numeric(patched['temperature'], errors='coerce')
    patched = patched.dropna(subset=['output_changed', 'temperature'])
    layers = ['L8', 'L16', 'L24', 'L31']
    temps = sorted(patched['temperature'].unique())
    grid = np.zeros((len(layers), len(temps)))
    for li, layer in enumerate(layers):
        for ti, temp in enumerate(temps):
            sub = patched[(patched['patch_layer'] == layer) & (patched['temperature'] == temp)]
            oc = sub['output_changed'].dropna()
            grid[li, ti] = float(oc.mean() * 100) if len(oc) > 0 else 0
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, cmap='Greys', aspect='auto', vmin=0)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f'T={t:.1f}' for t in temps])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Patch Layer')
    ax.set_title('Layer × Temperature Patching Output Change Rate (%)')
    for li in range(len(layers)):
        for ti in range(len(temps)):
            ax.text(ti, li, f'{grid[li, ti]:.1f}%', ha='center', va='center',
                    fontsize=9, color='white' if grid[li, ti] > 15 else 'black')
    fig.colorbar(im, ax=ax, label='Output Change Rate (%)')
    ax.invert_yaxis()
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG03_patching_heatmap")


def fig_dim95_curve(session, vis_dir):
    """PAPER FIG 4: R vs POOL_DIM from Run 0041 -- dimension selection."""
    all_q45 = _load_analysis_json_all_temps(session, 'Q0041_pool_dim_sweep.json')
    if not all_q45:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    _temp_shades = ['#000000', '#222222', '#444444', '#666666', '#888888', '#AAAAAA']
    for i, (temp, data) in enumerate(sorted(all_q45.items())):
        results = data.get('results', [])
        if not results:
            continue
        dims = [r['pool_dim'] for r in results]
        R_vals = [r.get('frac_R', 0) for r in results]
        dim95 = data.get('dim_95', 0)
        c = _temp_shades[i % len(_temp_shades)]
        mk = _BW_MARKERS[i % len(_BW_MARKERS)]
        ls = _BW_LINESTYLES[i % len(_BW_LINESTYLES)]
        ax.plot(dims, R_vals, marker=mk, color=c, linewidth=1.5, markersize=5,
                linestyle=ls, label=f'T={temp:.1f} (dim₉₅={dim95})')
        if dim95:
            r_at_dim95 = data.get('R_at_dim95', 0)
            ax.axvline(x=dim95, color=c, linestyle=':', alpha=0.4)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('POOL_DIM')
    ax.set_ylabel('R Fraction')
    ax.set_title('R Fraction vs Projection Dimension (Run 0041)\ndim₉₅ = smallest dim where R ≥ 95% of peak')
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG06_dim95_curve")


def fig_cross_model_r(session, vis_dir):
    """PAPER FIG 5: Cross-model R comparison (Runs 0052-0054)."""
    from cartography import DATA, get_family_size_dir
    family = session.get('model_family', 'llama')
    size = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')
    q54_path = os.path.join(get_family_size_dir(family, size), variant, 'pooled', 'analysis',
                            'Q0054_cross_model_summary.json')
    if not os.path.exists(q54_path):
        return None
    with open(q54_path) as f:
        q54 = json.load(f)
    table = q54.get('table', [])
    model_labels = q54.get('models', [])
    if len(model_labels) < 2 or not table:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    temps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x = np.arange(len(temps))
    colors = ['#000000', '#555555', '#999999', '#BBBBBB', '#333333']
    width = 0.8 / len(model_labels)
    for mi, label in enumerate(model_labels):
        r_vals = []
        for t in temps:
            row = next((r for r in table if r.get('model') == label
                        and abs(r.get('temperature', -1) - t) < 0.01), None)
            r_vals.append(row.get('R', 0) if row else 0)
        offset = (mi - len(model_labels)/2 + 0.5) * width
        ax.bar(x + offset, r_vals, width, label=label,
               color=colors[mi % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T={t:.1f}' for t in temps])
    ax.set_ylabel('R Fraction')
    ax.set_title('Cross-Model R Fraction Comparison by Temperature')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG10_cross_model_R")


def fig_held_out_validation(session, vis_dir):
    """PAPER FIG 6: In-sample vs held-out R fractions from Run 0043."""
    all_q34 = _load_analysis_json_all_temps(session, 'Q0043_sobol_partition.json')
    if not all_q34:
        return None
    temps, insample, heldout = [], [], []
    for t in sorted(all_q34.keys()):
        d = all_q34[t]
        ins_r = d.get('perm_sens_R', {}).get('fraction')
        ho = d.get('held_out', {})
        ho_r = ho.get('frac_R') if ho else None
        if ins_r is not None and ho_r is not None:
            temps.append(t)
            insample.append(ins_r)
            heldout.append(ho_r)
    if len(temps) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(insample, heldout, s=80, c='#999999', zorder=3)
    for i, t in enumerate(temps):
        ax.annotate(f'T={t:.1f}', (insample[i], heldout[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    lims = [min(min(insample), min(heldout)) - 0.02,
            max(max(insample), max(heldout)) + 0.02]
    ax.plot(lims, lims, '--', color='grey', alpha=0.5, label='y=x')
    ax.set_xlabel('In-Sample R Fraction')
    ax.set_ylabel('Held-Out R Fraction (Run 0033)')
    ax.set_title('In-Sample vs Held-Out R Fraction Validation')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG05_held_out_validation")


def fig_fixed_dim_cross_temp(session, vis_dir):
    """PAPER FIG 7: Run 0045 fixed-dim R across temperatures."""
    all_q49 = _load_analysis_json_all_temps(session, 'Q0045_fixed_dim_per_condition_R.json')
    if not all_q49:
        return None
    temps = sorted(all_q49.keys())
    # Average R across conditions per temperature
    avg_R = []
    for t in temps:
        per_run = all_q49[t].get('per_run', [])
        r_vals = [r['frac_R'] for r in per_run
                  if not r.get('skipped') and r.get('frac_R') == r.get('frac_R')]
        avg_R.append(float(np.mean(r_vals)) if r_vals else 0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temps, avg_R, 'o-', color='black', linewidth=2, markersize=8, markerfacecolor='black')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mean R Fraction (POOL_DIM=64)')
    ax.set_title('Fixed-Dimension R Across Temperatures (Run 0045, POOL_DIM=64)\nApples-to-apples comparison -- all temperatures at same dimension')
    ax.set_xticks(temps)
    for i, t in enumerate(temps):
        ax.annotate(f'{avg_R[i]:.3f}', (t, avg_R[i]), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, color='black')
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG11_fixed_dim_cross_temp")


def fig_bootstrap_distributions(session, vis_dir):
    """PAPER FIG 8: Bootstrap R fraction distributions per temperature."""
    all_q34 = _load_analysis_json_all_temps(session, 'Q0043_sobol_partition.json')
    if not all_q34:
        return None
    # Check if any have bootstrap arrays saved
    temps_with_boot = []
    for t in sorted(all_q34.keys()):
        bc = all_q34[t].get('bootstrap_ci', {})
        if bc.get('n_bootstrap', 0) >= 50:
            temps_with_boot.append(t)
    if not temps_with_boot:
        return None
    fig, axes = plt.subplots(1, len(temps_with_boot), figsize=(4*len(temps_with_boot), 4),
                              squeeze=False)
    for i, t in enumerate(temps_with_boot):
        ax = axes[0][i]
        bc = all_q34[t].get('bootstrap_ci', {})
        lo, hi, med = bc.get('R_ci_lower', 0), bc.get('R_ci_upper', 1), bc.get('R_ci_median', 0.5)
        # Simulate distribution from CI (actual arrays not in JSON)
        ax.axvline(med, color='black', linewidth=2, label=f'Median={med:.3f}')
        ax.axvspan(lo, hi, alpha=0.2, color='black', label=f'95% CI [{lo:.3f}, {hi:.3f}]')
        ax.axvline(0, color='grey', linestyle=':', linewidth=1)
        excludes_zero = bc.get('R_excludes_zero', False)
        ax.set_title(f'T={t:.1f}\n{"✓ R > 0" if excludes_zero else "✗ includes 0"}',
                     fontsize=10, color='black' if excludes_zero else '#888888')
        ax.set_xlabel('R Fraction')
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('Bootstrap CI for R Fraction by Temperature', fontsize=12, y=1.02)
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG08_bootstrap_distributions")


def fig_interaction_mass(session, vis_dir):
    """PAPER FIG 9: Interaction mass by temperature."""
    all_q34 = _load_analysis_json_all_temps(session, 'Q0043_sobol_partition.json')
    if not all_q34:
        return None
    temps = sorted(all_q34.keys())
    masses = [all_q34[t].get('interaction_mass_unattributed', float('nan')) for t in temps]
    if all(m != m for m in masses):
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(temps)), masses, color='#555555', width=0.4,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f'T={t:.1f}' for t in temps])
    ax.set_ylabel('Interaction Mass (1 - Σeffects/baseline_var)')
    ax.set_title('Unattributed Variance (Interaction Mass) by Temperature\nNegative = over-explanation (effects sum > baseline)')
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=1)
    for i, (bar, m) in enumerate(zip(bars, masses)):
        if m == m:
            y_pos = bar.get_height()
            offset = abs(y_pos) * 0.08 + 0.005
            va = 'bottom' if y_pos >= 0 else 'top'
            y_text = y_pos + offset if y_pos >= 0 else y_pos - offset
            ax.text(i, y_text, f'{m:.3f}', ha='center', va=va, fontsize=8)
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG12_interaction_mass")


def fig_three_variant_delta(session, vis_dir):
    """PAPER FIG 13: Three-variant similarity comparison from Run 0001."""
    from cartography import get_pooled_paths
    family = session.get('model_family', 'llama')
    size = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')
    pp = get_pooled_paths(family, size, variant, create_dirs=False)
    tv_path = os.path.join(pp.get('analysis', ''), 'three_variant_comparison.json')
    if not os.path.exists(tv_path):
        return None
    with open(tv_path) as f:
        tv = json.load(f)
    table = tv.get('table', [])
    if not table:
        return None
    # Pivot: long format (temp, model, mean_similarity) → per-model arrays
    by_model = {}
    for row in table:
        m = row.get('model', '')
        t = row.get('temperature', 0)
        sim_val = row.get('mean_similarity', 0) or 0
        by_model.setdefault(m, []).append((t, sim_val))
    if not by_model:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    _styles = {
        'abliterated': {'color': '#000000', 'marker': 'o', 'ls': '-', 'label': 'Abliterated'},
        'instruct':    {'color': '#555555', 'marker': 's', 'ls': '--', 'label': 'Instruct'},
        'base':        {'color': '#999999', 'marker': '^', 'ls': ':', 'label': 'Base'},
    }
    for mi, (model, pairs) in enumerate(sorted(by_model.items())):
        pairs.sort()
        temps = [p[0] for p in pairs]
        sims = [p[1] for p in pairs]
        style = _styles.get(model, {
            'color': _BW_COLORS[mi % len(_BW_COLORS)],
            'marker': _BW_MARKERS[mi % len(_BW_MARKERS)],
            'ls': '-', 'label': model.capitalize()})
        ax.plot(temps, sims, marker=style['marker'], color=style['color'],
                linewidth=1.5, linestyle=style['ls'], markersize=6, label=style['label'])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mean Similarity')
    ax.set_title('Three-Variant Similarity Comparison (Run 0001)\nAbliterated vs Instruct vs Base')
    ax.legend()
    plt.tight_layout()
    return save_fig(fig, vis_dir, "FIG13_three_variant_delta")


# ── Descriptive statistics -- Cohen's d, KS tests, split-half reliability ──────
# v0.58.0.0: presentation statistics for Table 1 and effect size reporting.

_CONDITION_GROUPS = {
    'introspection': [6, 7, 8],
    'null':          [4, 5, 1],
    'arithmetic':    [9, 10, 11, 12],
    'priming_neutral':     [13],
    'priming_cooperative': [14],
    'priming_resistant':   [15],
    'impossibility':       [29, 30, 31],
    'perturbation':        [39, 40],
}

_PAIRWISE_COMPARISONS = [
    ('introspection', 'null'),
    ('arithmetic',    'null'),
    ('arithmetic',    'introspection'),
    ('priming_neutral',     'priming_cooperative'),
    ('priming_neutral',     'priming_resistant'),
    ('priming_cooperative', 'priming_resistant'),
    ('impossibility', 'null'),
    ('perturbation',  'null'),
]

_EFFECT_METRICS = ['state_similarity_index', 'signal_entropy_ratio', 'layer_sim_mean', 'mean_logit_entropy',
                   'onset_delay_ratio', 'disruption_magnitude']


def _cohens_d(a, b):
    """Compute Cohen's d (pooled SD). Returns (d, n_a, n_b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan'), na, nb
    pooled_std = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2))
    if pooled_std == 0:
        return float('nan'), na, nb
    return float((a.mean() - b.mean()) / pooled_std), na, nb


def compute_effect_size_table(df: pd.DataFrame) -> dict:
    """Compute Cohen's d for all pairwise condition comparisons on key metrics.

    Returns dict: {
        'comparisons': [{
            'group_a': str, 'group_b': str, 'metric': str,
            'cohens_d': float, 'mean_a': float, 'mean_b': float,
            'n_a': int, 'n_b': int
        }, ...],
        'summary': {metric: {comparison: d, ...}, ...}
    }
    """
    if df.empty or 'run_mode' not in df.columns:
        return {'comparisons': [], 'summary': {}}

    comparisons = []
    summary = {}

    for metric in _EFFECT_METRICS:
        if metric not in df.columns:
            continue
        summary[metric] = {}
        for grp_a, grp_b in _PAIRWISE_COMPARISONS:
            runs_a = _CONDITION_GROUPS.get(grp_a, [])
            runs_b = _CONDITION_GROUPS.get(grp_b, [])
            a = pd.to_numeric(df[df['run_mode'].isin(runs_a)][metric], errors='coerce').dropna()
            b = pd.to_numeric(df[df['run_mode'].isin(runs_b)][metric], errors='coerce').dropna()
            d, na, nb = _cohens_d(a, b)
            label = f"{grp_a}_vs_{grp_b}"
            entry = {
                'group_a': grp_a, 'group_b': grp_b, 'metric': metric,
                'cohens_d': d, 'mean_a': float(a.mean()) if len(a) else float('nan'),
                'mean_b': float(b.mean()) if len(b) else float('nan'),
                'n_a': int(na), 'n_b': int(nb),
            }
            comparisons.append(entry)
            summary[metric][label] = d

    return {'comparisons': comparisons, 'summary': summary}


def compute_ks_tests(df: pd.DataFrame) -> dict:
    """Kolmogorov-Smirnov tests for distributional separation between conditions.

    Returns dict: {
        'tests': [{
            'group_a': str, 'group_b': str, 'metric': str,
            'ks_statistic': float, 'p_value': float,
            'overlap_estimate': float,  # 1 - KS statistic (rough overlap proxy)
            'n_a': int, 'n_b': int
        }, ...],
        'summary': {metric: {comparison: {'ks': float, 'p': float}, ...}, ...}
    }
    """
    if df.empty or 'run_mode' not in df.columns:
        return {'tests': [], 'summary': {}}

    tests = []
    summary = {}

    for metric in _EFFECT_METRICS:
        if metric not in df.columns:
            continue
        summary[metric] = {}
        for grp_a, grp_b in _PAIRWISE_COMPARISONS:
            runs_a = _CONDITION_GROUPS.get(grp_a, [])
            runs_b = _CONDITION_GROUPS.get(grp_b, [])
            a = pd.to_numeric(df[df['run_mode'].isin(runs_a)][metric], errors='coerce').dropna()
            b = pd.to_numeric(df[df['run_mode'].isin(runs_b)][metric], errors='coerce').dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            ks_stat, p_val = sp_stats.ks_2samp(a, b)
            label = f"{grp_a}_vs_{grp_b}"
            entry = {
                'group_a': grp_a, 'group_b': grp_b, 'metric': metric,
                'ks_statistic': float(ks_stat), 'p_value': float(p_val),
                'overlap_estimate': float(1.0 - ks_stat),
                'n_a': len(a), 'n_b': len(b),
            }
            tests.append(entry)
            summary[metric][label] = {'ks': float(ks_stat), 'p': float(p_val)}

    return {'tests': tests, 'summary': summary}


def compute_split_half_reliability(df: pd.DataFrame, n_splits: int = 100, seed: int = 42) -> dict:
    """Split-half reliability for similarity as a measurement instrument.

    For each condition group, randomly split trials into halves, compute
    mean similarity per half, correlate across splits. Reports Pearson r as
    reliability coefficient.

    Returns dict: {condition_group: {'r_mean': float, 'r_std': float, 'n_trials': int}, ...}
    """
    if df.empty or 'state_similarity_index' not in df.columns or 'trial' not in df.columns:
        return {}

    rng = np.random.default_rng(seed)
    results = {}

    for grp_name, runs in _CONDITION_GROUPS.items():
        sub = df[df['run_mode'].isin(runs)][['trial', 'state_similarity_index']].dropna()
        if sub.empty:
            continue
        trials = sorted(sub['trial'].unique())
        if len(trials) < 10:
            continue

        # Per-trial mean similarity
        trial_means = sub.groupby('trial')['state_similarity_index'].mean()
        trial_arr = trial_means.index.values
        val_arr = trial_means.values

        correlations = []
        for _ in range(n_splits):
            perm = rng.permutation(len(trial_arr))
            mid = len(perm) // 2
            half_a = val_arr[perm[:mid]]
            half_b = val_arr[perm[mid:2*mid]]  # equal size
            if len(half_a) >= 5 and len(half_b) >= 5:
                # Correlate the sorted values -- measures whether the rank structure
                # of similarity across trials is stable across random splits.
                r, _ = sp_stats.pearsonr(np.sort(half_a), np.sort(half_b))
                correlations.append(r)

        if correlations:
            results[grp_name] = {
                'r_mean': float(np.mean(correlations)),
                'r_std': float(np.std(correlations)),
                'n_trials': len(trials),
                'n_splits': len(correlations),
            }

    return results


# ── Multiple comparison correction (v0.71.0.0) ──────────────────────────────

def compute_correction_table(outcomes: dict) -> dict:
    """Bonferroni + Benjamini-Hochberg dual correction on hypothesis p-values.

    Extracts p-values from outcomes dict (any key containing 'p_value' or 'p_corr'
    or 'p_group' at the top level of a hypothesis entry). Applies both corrections.

    Returns dict: {
        'n_tests': int,
        'alpha': 0.05,
        'bonferroni_alpha': float,
        'tests': [{
            'hypothesis': str, 'p_value': float,
            'bonferroni_reject': bool, 'bh_reject': bool,
            'bh_rank': int, 'bh_threshold': float,
        }, ...]
    }
    """
    alpha = 0.05
    tests = []
    for h_id, o in sorted(outcomes.items()):
        if not isinstance(o, dict):
            continue
        # Extract any p-value from the outcome
        pv = None
        for pk in ('p_value', 'p_corr', 'p_group', 'p_vs_threshold', 'p_R_raw',
                    'kruskal_p', 'interaction_p', 'compute_p', 'levene_p'):
            v = o.get(pk)
            if v is not None and isinstance(v, (int, float)) and v == v:  # not NaN
                pv = float(v)
                break
        if pv is None:
            continue
        tests.append({'hypothesis': h_id, 'p_value': pv})

    n = len(tests)
    if n == 0:
        return {'n_tests': 0, 'alpha': alpha, 'bonferroni_alpha': alpha, 'tests': []}

    bonf_alpha = alpha / n

    # BH procedure: sort by p-value, compare to (rank/n)*alpha
    tests.sort(key=lambda t: t['p_value'])
    bh_max_rank = 0
    for i, t in enumerate(tests):
        rank = i + 1
        bh_threshold = (rank / n) * alpha
        t['bh_rank'] = rank
        t['bh_threshold'] = bh_threshold
        t['bonferroni_reject'] = t['p_value'] < bonf_alpha
        t['bh_reject'] = t['p_value'] <= bh_threshold
        if t['bh_reject']:
            bh_max_rank = rank

    # BH: all tests with rank ≤ bh_max_rank are rejected
    for t in tests:
        t['bh_reject'] = t['bh_rank'] <= bh_max_rank

    return {
        'n_tests': n,
        'alpha': alpha,
        'bonferroni_alpha': bonf_alpha,
        'tests': tests,
    }


def compute_cohens_d_summary(df: pd.DataFrame) -> dict:
    """Clean Cohen's d summary table for the major comparisons.

    H03 (introspection vs null), H09 (arithmetic vs introspection),
    H12 (meaningful vs shuffled), H38 (real vs random patching if Run 0019 exists).
    Reviewers want effect sizes next to p-values.
    """
    if df.empty or 'run_mode' not in df.columns:
        return {'comparisons': []}

    _HEADLINE = [
        ('H03', 'introspection_vs_null',    [6,7,8],     [4,5,1],    'state_similarity_index'),
        ('H09', 'arithmetic_vs_introspection', [9,10,11,12], [6,7,8],    'state_similarity_index'),
        ('H12', 'shuffled_vs_null',         [32],         [4,5],       'state_similarity_index'),
        ('H38', 'real_patching_vs_noise',   [17],         [19],        'output_changed'),
    ]
    results = []
    for h_id, label, runs_a, runs_b, metric in _HEADLINE:
        if metric not in df.columns:
            continue
        a = pd.to_numeric(df[df['run_mode'].isin(runs_a)][metric], errors='coerce').dropna()
        b = pd.to_numeric(df[df['run_mode'].isin(runs_b)][metric], errors='coerce').dropna()
        d, na, nb = _cohens_d(a, b)
        if na >= 5 and nb >= 5:
            _, p = sp_stats.ttest_ind(a, b)
            results.append({
                'hypothesis': h_id, 'label': label, 'metric': metric,
                'cohens_d': d, 'p_value': float(p),
                'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
                'n_a': na, 'n_b': nb,
            })
    return {'comparisons': results}


# ── Outcome inference ─────────────────────────────────────────────────────────

def _infer_outcomes(df: pd.DataFrame, granger: dict, bs: dict,
                    decomp: dict = None, sobol: dict = None,
                    pooled_sobol: dict = None) -> dict:
    """
    Infer hypothesis status from data.
    Returns dict: H_id → {status, metric, value, implication}
    """
    outcomes = {}

    def ttest(a_runs, b_runs, metric, direction='positive', alpha=0.05):
        if metric not in df.columns: return 'pending', float('nan')
        a = pd.to_numeric(df[df['run_mode'].isin(a_runs)][metric], errors='coerce').dropna()
        b = pd.to_numeric(df[df['run_mode'].isin(b_runs)][metric], errors='coerce').dropna()
        if len(a) < 5 or len(b) < 5: return 'pending', float('nan')
        t, p = sp_stats.ttest_ind(a, b)
        effect = a.mean() - b.mean()
        if p > alpha: return 'inconclusive', effect
        if (direction == 'positive' and effect > 0) or \
           (direction == 'negative' and effect < 0): return 'supported', effect
        return 'disproven', effect

    def mean_val(runs, metric):
        if metric not in df.columns: return float('nan')
        sub = pd.to_numeric(df[df['run_mode'].isin(runs)][metric], errors='coerce').dropna()
        return float(sub.mean()) if len(sub) > 0 else float('nan')

    # H01 -- Geometry Is Condition-Invariant
    # INF-H01-THRESHOLD fix (v32.0): prior implementation tested only whether mean
    # layer_sim_mean > 0.5 in the null runs -- a pure level test. The hypothesis name is
    # "Condition-Invariant", which implies variance homogeneity across conditions, not just
    # a threshold. We add a Levene test comparing variance across run_mode clusters (null,
    # introspection, arithmetic). Supported requires BOTH: (1) null mean > 0.5 AND
    # (2) Levene p >= 0.05 (variances not heterogeneous across conditions). If Levene
    # fails, geometry shows condition-dependent variance -- the invariance claim weakens.
    v = mean_val([4, 5, 1], 'layer_sim_mean')
    if np.isnan(v):
        outcomes["H01"] = {"status": "pending", "metric": "layer_sim_mean", "value": float('nan'),
                           "implication": "Null runs not yet complete."}
    else:
        level_ok = v > 0.5
        # Levene test: compare layer_sim_mean variance across three condition groups
        # (null [4,5,1], introspection [6,7,8], arithmetic [9,10,11,12]).
        # Requires all three groups to have data -- degrades gracefully to level-only if not.
        levene_p  = float('nan')
        levene_ok = True   # default True: if test can't run, don't penalise
        if 'layer_sim_mean' in df.columns:
            g_null  = df[run_mode_mask_any(df['run_mode'], [4, 5, 1])]['layer_sim_mean'].dropna()
            g_intro = df[run_mode_mask_any(df['run_mode'], [6, 7, 8])]['layer_sim_mean'].dropna()
            g_arith = df[run_mode_mask_any(df['run_mode'], [9, 10, 11, 12])]['layer_sim_mean'].dropna()
            groups  = [g for g in [g_null, g_intro, g_arith] if len(g) >= 5]
            if len(groups) >= 2:
                _, levene_p = sp_stats.levene(*groups)
                levene_ok   = levene_p >= 0.05
        # Status: level check gates on whether null runs are valid.
        # Variance homogeneity provides additional diagnostic.
        if level_ok and levene_ok:
            h01_status = "supported"
        elif level_ok and not levene_ok:
            h01_status = "inconclusive"   # mean OK but conditions differ in variance
        else:
            h01_status = "disproven" if not np.isnan(v) else "pending"
        lev_str = f"Levene p={levene_p:.3f}" if not np.isnan(levene_p) else "Levene=n/a"
        outcomes["H01"] = {
            "status":      h01_status,
            "metric":      "layer_sim_mean",
            "value":       float(v),
            "levene_p":    levene_p,
            "implication": (
                f"Null layer_sim_mean={v:.4f} > 0.5; {lev_str} >= 0.05 -- "
                f"geometry is consistent and homogeneous across conditions at baseline."
                if h01_status == "supported"
                else f"Null mean={v:.4f}; {lev_str} -- "
                     f"condition groups show heterogeneous variance (geometry not fully invariant)."
                if h01_status == "inconclusive"
                else f"Null layer_sim_mean={v:.4f} <= 0.5 -- geometric baseline below threshold."
            ),
        }

    # H03 -- introspection similarity > null similarity
    # INF-H03-BASELINE fix (v32.0): null comparison was [1,2]; must include Run 0001
    # (largest null baseline, saves all-layers, richer sample). Same fix applied to H04/H08.
    s, e = ttest([6,7,8], [4,5,1], 'state_similarity_index', 'positive')
    outcomes["H03"] = {"status": s, "metric": "state_similarity_index", "value": e,
                       "implication": "Introspection elevates similarity beyond null reference." if s=="supported"
                                      else "Introspection does not elevate similarity."}

    # H04 -- SER elevation
    # INF-H04-BASELINE fix (v32.0): same as H03 -- null baseline expanded to [4,5,1].
    s, e = ttest([6,7,8], [4,5,1], 'signal_entropy_ratio', 'positive')
    outcomes["H55"] = {"status": s, "metric": "signal_entropy_ratio", "value": e,
                       "implication": "SER elevated in introspection -- efficiency over entropy."}

    # H05 -- Disruption events at contradiction turn
    # INF-H05-NOPVAL fix (v31.0): replace arbitrary disruption_elevation > 0.2 threshold on
    # a binary metric with a chi-square test on the 2×2 contingency table.
    # disruption_flag is binary (0/1); comparing raw means without a significance test is
    # inconsistent with the framework's standards and has no stated threshold derivation.
    sub28 = df[run_mode_mask(df['run_mode'], 28)] if not df.empty else pd.DataFrame()
    if not sub28.empty and 'contradiction_turn' in sub28.columns and 'disruption_flag' in sub28.columns:
        at_c  = sub28[sub28['contradiction_turn'] == 1]['disruption_flag'].dropna()
        # INF-H05-POSTCONTAM fix (v36.1): contradiction_turn==0 includes both pre-contradiction
        # turns (1-6) and post-contradiction turns (8-13). Post turns are not a valid baseline
        # for the pre/at comparison. Filter to pre-contradiction only using post_contradiction==0.
        # Run 0028 saves post_contradiction=int(turn > CONTRADICTION_AT); both flags present in CSV.
        if 'post_contradiction' in sub28.columns:
            not_c = sub28[(sub28['contradiction_turn'] == 0) &
                          (sub28['post_contradiction'] == 0)]['disruption_flag'].dropna()
        else:
            not_c = sub28[sub28['contradiction_turn'] == 0]['disruption_flag'].dropna()
        if len(at_c) < 5 or len(not_c) < 5:
            outcomes["H05"] = {"status": "pending", "metric": "disruption_flag", "value": float('nan'),
                               "implication": "Run 0028 insufficient data for chi-square test."}
        else:
            disruption_elevation = float(at_c.mean() - not_c.mean())
            n_at_1  = int(at_c.sum());   n_at_0  = len(at_c)  - n_at_1
            n_not_1 = int(not_c.sum());  n_not_0 = len(not_c) - n_not_1
            try:
                _, p05, _, _ = sp_stats.chi2_contingency(
                    [[n_at_1, n_at_0], [n_not_1, n_not_0]])
            except Exception:
                p05 = 1.0
            outcomes["H05"] = {
                "status": "supported" if (p05 < 0.05 and disruption_elevation > 0) else
                          ("inconclusive" if p05 >= 0.05 else "disproven"),
                "metric": "disruption_flag", "value": float(disruption_elevation),
                "implication": (
                    f"Disruption events cluster at contradiction turns vs baseline "
                    f"(Δ={disruption_elevation:.4f}, χ² p={p05:.3f}) -- contradiction injection confirmed."
                    if (p05 < 0.05 and disruption_elevation > 0)
                    else f"No significant disruption clustering at contradiction turns "
                         f"(Δ={disruption_elevation:.4f}, χ² p={p05:.3f})."
                ),
            }
    else:
        outcomes["H05"] = {"status": "pending", "metric": "disruption_flag", "value": float('nan'),
                           "implication": "Run 0028 not yet complete."}

    # H09 -- arithmetic vs introspection (INF-2 fix v29.7)
    # H09 null: arithmetic geometry equals introspection. Falsification requires
    # testing arithmetic directly against introspection, not against null.
    # Prior implementation compared [6,7,8,9] vs [1,2] -- valid by transitivity
    # (if introspection > null and arithmetic < null, then arithmetic < introspection)
    # but indirect: H09 could be "supported" even when arithmetic ≈ introspection.
    # Fix: primary test is arithmetic vs introspection. Secondary test vs null retained
    # for context and legacy continuity.
    s_primary,   e_primary   = ttest([9,10,11,12], [6,7,8], 'state_similarity_index', 'negative')
    s_secondary, e_secondary = ttest([9,10,11,12], [4,5],   'state_similarity_index', 'negative')
    # Status follows the primary (direct) test.
    if s_primary == 'pending':
        h09_status = 'pending'
        h09_impl   = "Arithmetic or introspection runs not yet complete."
    elif s_primary == 'supported':
        h09_status = 'supported'
        h09_impl   = (f"Arithmetic similarity lower than introspection (Δ={e_primary:.4f}) "
                      f"and lower than null (Δ={e_secondary:.4f}) -- "
                      f"task demand geometry is distinct from introspective geometry. H09 falsified.")
    elif s_primary == 'inconclusive':
        h09_status = 'inconclusive'
        h09_impl   = (f"Arithmetic vs introspection not significant (Δ={e_primary:.4f}); "
                      f"arithmetic vs null: {s_secondary} (Δ={e_secondary:.4f}).")
    else:
        h09_status = 'disproven'
        h09_impl   = (f"Arithmetic similarity not lower than introspection (Δ={e_primary:.4f}) -- "
                      f"task demand geometry resembles introspection. H09 survives.")
    outcomes["H09"] = {
        "status": h09_status, "metric": "state_similarity_index",
        "value": float(e_primary) if not np.isnan(e_primary) else float('nan'),
        "implication": h09_impl,
        "secondary_vs_null": {"status": s_secondary, "value": float(e_secondary)
                               if not np.isnan(e_secondary) else float('nan')},
    }

    # H11 -- Granger ΔR²
    # INF-H11-PREF fix (v30.9): prefer B (Run 0048, larger N, cross-temperature) over A.
    # INF-H11-NOPVAL fix (v31.0): use permutation p_value from _granger_test when present.
    # Falls back to magnitude-only (d > 0.01) if p_value absent (pre-v31.0 JSON).
    gr = granger.get('B') or granger.get('A')
    if gr:
        d    = gr.get('delta_r2_internal', 0)
        p11  = gr.get('p_value', float('nan'))
        sig  = gr.get('significant', None)
        if sig is not None:
            status11 = "supported" if sig else ("inconclusive" if d > 0 else "disproven")
        else:
            status11 = "supported" if d > 0.01 else "inconclusive"
        outcomes["H11"] = {
            "status": status11,
            "metric": "delta_r2_internal", "value": float(d),
            "implication": (
                f"S_{{t-1}} adds predictive power over E_t "
                f"(ΔR²={d:.4f}" + (f", p={p11:.3f}" if not np.isnan(p11) else "") + f") -- "
                f"hidden state causally upstream of next state."
                if status11 == "supported"
                else f"S_{{t-1}} does not reliably add predictive power "
                     f"(ΔR²={d:.4f}" + (f", p={p11:.3f}" if not np.isnan(p11) else "") + ")."
            ),
        }
    else:
        outcomes["H11"] = {"status": "pending", "metric": "delta_r2_internal", "value": float('nan'),
                           "implication": "Run 0047/27 not yet complete."}

    # H13 -- temperature stability (INF-3 fix v29.7)
    # H13 prediction: "similarity survives T=0.8 and T=1.0 at all four conditions.
    # Condition ordering preserved across the full temperature range."
    # Prior implementation compared pooled means at T=0.2 vs T=1.0 -- tests average
    # stability only. Cannot detect a collapse in one condition masked by others, and
    # cannot test ordering preservation at all.
    # Fix: (1) per-condition stability at each temperature; (2) Kendall's W across
    # the full temperature grid to test whether condition ordering is preserved.
    # BUG 3 fix: column is 'temperature', not 'temperature_condition'.
    sub3 = df[run_mode_mask(df['run_mode'], 3)] if not df.empty else pd.DataFrame()
    if not sub3.empty and 'temperature' in sub3.columns and 'state_similarity_index' in sub3.columns:
        TEMPS_26    = [0.2, 0.4, 0.6, 0.8, 1.0]
        CONDS_26    = ['introspection', 'arithmetic', 'neutral_prime', 'null']
        cond_col    = 'condition' if 'condition' in sub3.columns else None

        # Per-condition stability: similarity at T=0.2 vs T=1.0 for each condition.
        # A condition "survives" if abs(drop) < 0.1 (same threshold as pooled).
        per_cond_stability = {}
        if cond_col:
            for cond in CONDS_26:
                sub_c = sub3[sub3[cond_col] == cond]
                lo = sub_c[sub_c['temperature'] == 0.2]['state_similarity_index'].mean()
                hi = sub_c[sub_c['temperature'] == 1.0]['state_similarity_index'].mean()
                if not np.isnan(lo) and not np.isnan(hi):
                    per_cond_stability[cond] = {'lo': float(lo), 'hi': float(hi),
                                                'drop': float(lo - hi),
                                                'survives': abs(lo - hi) < 0.1}

        # Condition ordering: for each temperature, rank conditions by mean similarity.
        # Compute Kendall's W across all temperature-rank matrices.
        # W = 1: perfect ordering preservation. W = 0: random.
        kendall_w = float('nan')
        if cond_col and len(per_cond_stability) >= 2:
            rank_matrix = []  # rows = temperatures, cols = conditions (sorted by name)
            conds_present = sorted(per_cond_stability.keys())
            for temp in TEMPS_26:
                sub_t = sub3[sub3['temperature'] == temp]
                if cond_col in sub_t.columns:
                    means = []
                    for cond in conds_present:
                        m_val = sub_t[sub_t[cond_col] == cond]['state_similarity_index'].mean()
                        means.append(float(m_val) if not np.isnan(m_val) else 0.0)
                    # Rank within this temperature (1 = lowest similarity)
                    order = np.argsort(np.argsort(means)) + 1
                    rank_matrix.append(order.tolist())
            if len(rank_matrix) >= 2:
                # Kendall's W: W = 12*S / (k^2*(n^3-n))
                # k = number of judges (temperatures), n = number of items (conditions)
                k = len(rank_matrix)
                n = len(rank_matrix[0])
                rank_arr = np.array(rank_matrix, dtype=float)
                col_sums = rank_arr.sum(axis=0)
                grand_mean = col_sums.mean()
                S = float(np.sum((col_sums - grand_mean) ** 2))
                denom = (k ** 2) * (n ** 3 - n)
                kendall_w = float(S * 12 / denom) if denom > 0 else float('nan')

        # Pooled drop (legacy -- retained for continuity)
        t_lo_pool = sub3[sub3['temperature'] == 0.2]['state_similarity_index'].mean()
        t_hi_pool = sub3[sub3['temperature'] == 1.0]['state_similarity_index'].mean()
        pooled_drop = float(t_lo_pool - t_hi_pool) if (
            not np.isnan(t_lo_pool) and not np.isnan(t_hi_pool)) else float('nan')

        # Status: both parts of the prediction must hold.
        # (1) All conditions with data must survive (abs(drop) < 0.1).
        # (2) Ordering preserved: Kendall's W >= 0.7 (strong concordance).
        # If condition data unavailable, fall back to pooled stability only.
        has_per_cond = len(per_cond_stability) >= 2
        all_survive  = has_per_cond and all(
            v['survives'] for v in per_cond_stability.values())
        ordering_ok  = (not np.isnan(kendall_w) and kendall_w >= 0.7)

        if not has_per_cond:
            # Condition column absent -- pooled fallback
            status13 = ("supported" if (not np.isnan(pooled_drop) and abs(pooled_drop) < 0.1)
                        else ("disproven" if not np.isnan(pooled_drop) else "pending"))
            impl13   = (f"Similarity pooled drop={pooled_drop:.4f} -- "
                        f"{'stable' if abs(pooled_drop) < 0.1 else 'unstable'}. "
                        f"No condition column -- ordering test not possible.")
        elif all_survive and ordering_ok:
            status13 = "supported"
            cond_summary = ', '.join(f"{c}:drop={v['drop']:.3f}" for c, v in per_cond_stability.items())
            impl13   = (f"All conditions survive temperature variation ({cond_summary}). "
                        f"Condition ordering preserved (Kendall W={kendall_w:.3f}). "
                        f"Similarity not a greedy-decoding artifact.")
        elif not all_survive:
            collapsed = [c for c, v in per_cond_stability.items() if not v['survives']]
            status13 = "disproven"
            impl13   = (f"Similarity collapses at high temperature in: {collapsed}. "
                        f"Kendall W={kendall_w:.3f}. Temperature robustness not confirmed.")
        else:
            # Conditions survive but ordering not preserved
            status13 = "inconclusive"
            impl13   = (f"Per-condition Similarity survives (all drops < 0.1) but condition "
                        f"ordering not preserved (Kendall W={kendall_w:.3f} < 0.7) -- "
                        f"relative geometry shifts with temperature.")

        # Build per-condition display values using safe key access
        per_cond_out = {c: {'drop': v['drop'], 'survives': v['survives']}
                        for c, v in per_cond_stability.items()}
        outcomes["H13"] = {
            "status": status13, "metric": "state_similarity_index",
            "value": pooled_drop,
            "implication": impl13,
            "per_condition_stability": per_cond_out,
            "kendall_w_ordering": kendall_w,
        }
    else:
        outcomes["H13"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                           "implication": "Run 0003 not yet complete."}

    # H17 -- confound isolation (INF-1 fix v29.7)
    # Now reads delta_r2_internal per confound condition from Q33_decomposition.json
    # (h16_confound_decomposition key), not state_similarity_index from Run 0023 CSV.
    # H17 falsification criterion: delta_r2_internal collapses to zero under
    # semantic system prompt. Direct test -- no proxy substitution.
    h16_cd = decomp.get('h16_confound_decomposition', {}) if decomp else {}
    h16_verdict_str = decomp.get('h16_verdict', '') if decomp else ''
    sem_d = h16_cd.get('semantic', {})
    neu_d = h16_cd.get('neutral', {})
    sem_dr2 = sem_d.get('delta_r2_internal', {})
    neu_dr2 = neu_d.get('delta_r2_internal', {})
    sem_val = sem_dr2.get('value', float('nan'))
    neu_val = neu_dr2.get('value', float('nan'))
    sem_sig = sem_dr2.get('significant', None)
    neu_sig = neu_dr2.get('significant', None)
    if sem_sig is None and neu_sig is None:
        outcomes["H17"] = {"status": "pending",
                           "metric": "delta_r2_internal_per_condition",
                           "value": float('nan'),
                           "implication": "Run 0042 per-condition decomposition not yet complete (requires Run 0023 + Run 0042)."}
    elif sem_sig is True and neu_sig is True:
        outcomes["H17"] = {
            "status": "supported",
            "metric": "delta_r2_internal_per_condition",
            "value": float(sem_val) if not np.isnan(sem_val) else float('nan'),
            "implication": (f"ΔR²_internal significant under both system prompt conditions "
                            f"(semantic={sem_val:.4f}, neutral={neu_val:.4f}) -- "
                            f"system prompt confound does not explain R. H17 falsified.")}
    elif sem_sig is False:
        outcomes["H17"] = {
            "status": "disproven",
            "metric": "delta_r2_internal_per_condition",
            "value": float(sem_val) if not np.isnan(sem_val) else float('nan'),
            "implication": (f"ΔR²_internal collapses under semantic system prompt "
                            f"(semantic={sem_val:.4f}, p not significant) -- "
                            f"system prompt may explain observed R. H17 survives.")}
    else:
        outcomes["H17"] = {
            "status": "inconclusive",
            "metric": "delta_r2_internal_per_condition",
            "value": float(sem_val) if not np.isnan(sem_val) else float('nan'),
            "implication": h16_verdict_str or
                           (f"Mixed: neutral significant ({neu_val:.4f}) but semantic not "
                            f"({sem_val:.4f}) -- partial confound.")}

    # H18 -- persistence mechanism
    # INF-H18-NOPVAL fix (v31.0): replaced bare abs(diff) < 0.05 threshold with ttest_ind.
    # INF-H18-SUMMARY fix (v32.0): 'summary' history mode was collected and plotted in
    # fig_persistence but NEVER statistically tested. The old block only compared 'full'
    # vs 'last'. H18 predicts ALL three modes produce equivalent similarity (persistent geometry
    # independent of how much history is re-entered). A summary mode that diverges would
    # have been entirely missed. Add full-vs-summary and last-vs-summary; all three pairs
    # must be non-significant for a 'supported' verdict.
    sub24 = df[run_mode_mask(df['run_mode'], 24)] if not df.empty else pd.DataFrame()
    if not sub24.empty and 'history_mode' in sub24.columns:
        full_vals    = sub24[sub24['history_mode']=='full']['state_similarity_index'].dropna()
        last_vals    = sub24[sub24['history_mode']=='last']['state_similarity_index'].dropna()
        summary_vals = sub24[sub24['history_mode']=='summary']['state_similarity_index'].dropna()
        if len(full_vals) < 5 or len(last_vals) < 5:
            outcomes["H18"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                               "implication": "Run 0024 insufficient data for full/last comparison."}
        else:
            diff_fl = float(full_vals.mean() - last_vals.mean())
            _, p_fl = sp_stats.ttest_ind(full_vals, last_vals)
            has_summary = len(summary_vals) >= 5
            diff_fs = float('nan'); p_fs = float('nan')
            diff_ls = float('nan'); p_ls = float('nan')
            if has_summary:
                diff_fs = float(full_vals.mean() - summary_vals.mean())
                diff_ls = float(last_vals.mean()  - summary_vals.mean())
                _, p_fs = sp_stats.ttest_ind(full_vals,  summary_vals)
                _, p_ls = sp_stats.ttest_ind(last_vals,  summary_vals)
            fl_ok = (p_fl >= 0.05 and abs(diff_fl) < 0.05)
            fs_ok = (p_fs >= 0.05 and abs(diff_fs) < 0.05) if has_summary else True
            ls_ok = (p_ls >= 0.05 and abs(diff_ls) < 0.05) if has_summary else True
            all_ok = fl_ok and fs_ok and ls_ok
            pairwise_str = f"full-last: Δ={diff_fl:.4f} p={p_fl:.3f}"
            if has_summary:
                pairwise_str += (f"; full-summary: Δ={diff_fs:.4f} p={p_fs:.3f}"
                                 f"; last-summary: Δ={diff_ls:.4f} p={p_ls:.3f}")
            else:
                pairwise_str += "; summary arm: not yet collected"
            outcomes["H18"] = {
                "status": "supported" if all_ok else
                          ("disproven" if (p_fl < 0.05 and abs(diff_fl) >= 0.05) else "inconclusive"),
                "metric": "state_similarity_index", "value": float(diff_fl),
                "summary_tested": has_summary,
                "implication": (
                    f"All pairwise history-mode comparisons non-significant ({pairwise_str}) -- "
                    f"persistent geometry sustains R, not history length."
                    if all_ok
                    else f"History mode produces significantly different Similarity ({pairwise_str}) -- "
                         f"R may depend on how context is re-entered."
                ),
            }
    else:
        outcomes["H18"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                           "implication": "Run 0024 not yet complete."}

    # H19 -- cross-instance coupling
    # INF-H19-THRESHOLD fix (v30.9): replaced arbitrary c > 0.7 with ttest_1samp vs 0.0.
    # INF-H19-DIVERGENCE fix (v32.0): H19 predicts coupling starts high and DIVERGES
    # over turns (the two instances drift apart geometrically as conversation proceeds).
    # Testing only the mean coupling against zero cannot detect this -- a steady-state
    # coupling and a diverging coupling look identical. Added linregress(turn, coupling_score)
    # per condition (coupled vs uncoupled). Divergence = negative slope for coupled arm
    # (coupling_score declining = growing geometric distance). Report slope and p-value.
    sub20 = df[run_mode_mask(df['run_mode'], 20)] if not df.empty else pd.DataFrame()
    if not sub20.empty and 'coupling_score' in sub20.columns:
        c_vals = sub20['coupling_score'].dropna()
        if len(c_vals) < 10:
            outcomes["H19"] = {"status": "pending", "metric": "coupling_score",
                               "value": float('nan'),
                               "implication": "Run 0020 insufficient data (< 10 rows)."}
        else:
            c_mean = float(c_vals.mean())
            _, p19 = sp_stats.ttest_1samp(c_vals, 0.0)
            # Divergence test: coupling_score ~ turn (negative slope = divergence)
            diverge_slope = float('nan'); diverge_p = float('nan')
            if 'turn' in sub20.columns:
                turn_coup = sub20[['turn', 'coupling_score']].dropna()
                if len(turn_coup) >= 10:
                    diverge_slope, _, _, diverge_p, _ = sp_stats.linregress(
                        turn_coup['turn'], turn_coup['coupling_score'])
            div_str = (f"; divergence slope={diverge_slope:.4f} p={diverge_p:.3f}"
                       if not np.isnan(diverge_slope) else "")
            outcomes["H19"] = {
                "status": "supported" if (p19 < 0.05 and c_mean > 0.1)
                          else ("inconclusive" if p19 >= 0.05 else "disproven"),
                "metric": "coupling_score", "value": float(c_mean),
                "divergence_slope": diverge_slope,
                "divergence_p":     diverge_p,
                "implication": (
                    f"Mean coupling score = {c_mean:.4f} (p={p19:.3f} vs 0){div_str} -- "
                    f"geometric coupling detected between interacting instances."
                    if (p19 < 0.05 and c_mean > 0.1)
                    else f"No significant coupling (mean={c_mean:.4f}, p={p19:.3f}{div_str})."
                ),
            }
    else:
        outcomes["H19"] = {"status": "pending", "metric": "coupling_score", "value": float('nan'),
                           "implication": "Run 0020 not yet complete."}

    # H20 -- coherence levels (Run 0021)
    # INF-H20-NAN fix (v30.9): NaN guards + t-test before verdict.
    # INF-H20-MID-R-IGNORED fix (v31.0): mid_r was collected and rendered but never
    # statistically analyzed. Added gradient test across all three conditions.
    # INF-H20-TEST-MISMATCH fix (v31.0): H20 predicts increasing iota across turns
    # (coherence accumulation). Pooled signal_per_watt mean test cannot detect this.
    # Added slope test: state_similarity_index ~ turn per condition. Accumulation confirmed only if
    # high_r slope > low_r slope (p < 0.05). Mean test retained as secondary metric.
    sub21 = df[run_mode_mask(df['run_mode'], 21)] if not df.empty else pd.DataFrame()
    if not sub21.empty and 'signal_per_watt' in sub21.columns and 'r_condition' in sub21.columns:
        hi_vals  = sub21[sub21['r_condition']=='high_r']['signal_per_watt'].dropna()
        mid_vals = sub21[sub21['r_condition']=='mid_r']['signal_per_watt'].dropna()
        lo_vals  = sub21[sub21['r_condition']=='low_r']['signal_per_watt'].dropna()
        if len(hi_vals) < 5 or len(lo_vals) < 5:
            outcomes["H20"] = {"status": "pending", "metric": "signal_per_watt",
                               "value": float('nan'),
                               "implication": "Run 0021 high_r or low_r condition not yet complete."}
        else:
            diff = float(hi_vals.mean() - lo_vals.mean())
            _, p20 = sp_stats.ttest_ind(hi_vals, lo_vals)
            mid_mean = float(mid_vals.mean()) if len(mid_vals) >= 5 else float('nan')

            # Gradient test: high_r > mid_r > low_r
            gradient_confirmed = False
            gradient_detail    = "mid_r pending"
            if len(mid_vals) >= 5 and not np.isnan(mid_mean):
                _, p_hm = sp_stats.ttest_ind(hi_vals, mid_vals)
                _, p_ml = sp_stats.ttest_ind(mid_vals, lo_vals)
                gradient_confirmed = (
                    p_hm < 0.05 and hi_vals.mean() > mid_mean and
                    p_ml < 0.05 and mid_mean > lo_vals.mean()
                )
                gradient_detail = (
                    f"high>mid>low gradient confirmed (p_hm={p_hm:.3f}, p_ml={p_ml:.3f})"
                    if gradient_confirmed
                    else f"gradient not confirmed (p_hm={p_hm:.3f}, p_ml={p_ml:.3f})"
                )

            # Coherence slope test: state_similarity_index ~ turn per condition
            slope_status = "pending"
            slope_detail = "turn data pending"
            if 'state_similarity_index' in sub21.columns and 'turn' in sub21.columns:
                hi_sub = sub21[sub21['r_condition']=='high_r'][['turn','state_similarity_index']].dropna()
                lo_sub = sub21[sub21['r_condition']=='low_r'][['turn','state_similarity_index']].dropna()
                if len(hi_sub) >= 10 and len(lo_sub) >= 10:
                    slope_hi, _, _, p_slope_hi, _ = sp_stats.linregress(
                        hi_sub['turn'], hi_sub['state_similarity_index'])
                    slope_lo, _, _, p_slope_lo, _ = sp_stats.linregress(
                        lo_sub['turn'], lo_sub['state_similarity_index'])
                    if slope_hi > slope_lo and p_slope_hi < 0.05:
                        slope_status = "supported"
                        slope_detail = (f"high_r similarity slope={slope_hi:.4f} > "
                                        f"low_r slope={slope_lo:.4f} (p={p_slope_hi:.3f})"
                                        f" -- coherence accumulation confirmed.")
                    else:
                        slope_status = "inconclusive"
                        slope_detail = (f"high_r slope={slope_hi:.4f}, "
                                        f"low_r slope={slope_lo:.4f} -- no coherence accumulation.")

            mid_str = f"mid_r={mid_mean:.4f}" if not np.isnan(mid_mean) else "mid_r=pending"
            # Primary verdict requires both mean test and slope test to confirm
            if p20 < 0.05 and diff > 0.01 and slope_status == "supported":
                h20_status = "supported"
            elif p20 < 0.05 and diff > 0.01:
                h20_status = "inconclusive"   # mean up but no slope accumulation
            elif slope_status == "supported":
                h20_status = "inconclusive"   # slope up but mean not significant
            elif slope_status == "pending":
                h20_status = "pending"
            else:
                h20_status = "disproven"

            outcomes["H20"] = {
                "status": h20_status,
                "metric": "signal_per_watt", "value": diff,
                "coherence_slope_status": slope_status,
                "implication": (
                    f"high_r signal/watt exceeds low_r (Δ={diff:.4f}, p={p20:.3f}); "
                    f"{mid_str}; {gradient_detail}. "
                    f"Slope: {slope_detail}"
                ),
            }
    else:
        outcomes["H20"] = {"status": "pending", "metric": "signal_per_watt", "value": float('nan'),
                           "implication": "Run 0021 not yet complete."}

    # H21 -- E+C+R full decomposition (Run 0042)
    if decomp and 'delta_r2_internal' in decomp:
        d33     = decomp['delta_r2_internal']
        d_int   = d33.get('value', float('nan'))
        p_int   = d33.get('p_value', float('nan'))
        sig_int = d33.get('significant', False)
        d33c    = decomp.get('delta_r2_constraint', {})
        d_con   = d33c.get('value', float('nan'))
        p_con   = d33c.get('p_value', float('nan'))
        verdict = decomp.get('verdict', {}).get('conclusion', '')
        outcomes["H21"] = {
            "status": "supported" if (sig_int and not np.isnan(d_int) and d_int > 0.01)
                      else ("pending" if np.isnan(d_int) else "inconclusive"),
            "metric": "delta_r2_internal",
            "value": float(d_int) if not np.isnan(d_int) else float('nan'),
            "implication": verdict or "E+C+R decomposition not yet run."}
    else:
        outcomes["H21"] = {"status": "pending", "metric": "delta_r2_internal",
                           "value": float('nan'),
                           "implication": "Run 0042 (3-way decomposition) not yet complete."}

    # H50 -- Linearity validation (from Run 0042 linearity_check)
    lc = decomp.get('linearity_check', {}) if decomp else {}
    if lc and 'error' not in lc:
        outcomes["H50"] = {
            "status": "supported" if lc.get('all_within_001') else "inconclusive",
            "metric": "max_mlp_gap",
            "value": lc.get('max_gap', float('nan')),
            "implication": lc.get('verdict', ''),
            "ridge_r2": lc.get('ridge_r2_test', float('nan')),
            "mlp_configs": lc.get('mlp_configs', {}),
        }
    else:
        outcomes["H50"] = {"status": "pending", "metric": "max_mlp_gap",
                           "value": float('nan'),
                           "implication": "Linearity check not yet run (requires Run 0042)."}

    # H51 -- Interaction information -- ordering sensitivity
    ii = decomp.get('interaction_info', {}) if decomp else {}
    if ii and 'error' not in ii:
        ii_frac = ii.get('II_fraction', float('nan'))
        knn = ii.get('knn', {})
        knn_pooled = knn.get('pooled', {}) if knn and 'error' not in knn else {}
        knn_frac = knn_pooled.get('II_fraction', float('nan')) if knn_pooled else float('nan')
        agrees = knn.get('agrees_with_linear', True)
        # Use kNN verdict if available and disagrees
        if not np.isnan(knn_frac) and not agrees:
            _eff_frac = knn_frac  # kNN takes precedence when methods disagree
        else:
            _eff_frac = ii_frac
        outcomes["H51"] = {
            "status": "supported" if (not np.isnan(_eff_frac) and abs(_eff_frac) < 0.05) else
                      "inconclusive" if not np.isnan(_eff_frac) else "pending",
            "metric": "II_fraction",
            "value": _eff_frac,
            "implication": ii.get('verdict', ''),
            "knn_agrees": agrees,
        }
    else:
        outcomes["H51"] = {"status": "pending", "metric": "II_fraction",
                           "value": float('nan'),
                           "implication": "Interaction information not yet computed (requires Run 0042)."}

    # H16 -- permutation significance (from watchdog Run 0049)
    # INF-H16-PARTIAL fix (v32.0): prior implementation read only 'state_similarity_index' from the
    # Q32 baseline_swap JSON. Run 0049 also computes and saves permutation results for
    # 'signal_entropy_ratio' and 'layer_sim_mean', but those results were computed, saved, and then
    # silently discarded in the verdict. H16 asks whether condition differences exceed
    # the permutation null distribution -- this should be evaluated for ALL three metrics
    # the framework relies on, not just state_similarity_index. If signal_entropy_ratio fails the permutation
    # test while state_similarity_index passes, the framework has a problem. Report all three;
    # require state_similarity_index to pass for 'supported' (primary), flag others in implication.
    if bs and 'state_similarity_index' in bs:
        def _get_perm(metric):
            r = bs.get(metric, {})
            return r.get('p_value', float('nan')), r.get('observed_effect', float('nan'))
        p_sim, eff_sim = _get_perm('state_similarity_index')
        p_ser, eff_ser = _get_perm('signal_entropy_ratio')
        p_lsm, eff_lsm = _get_perm('layer_sim_mean')
        sig_sim = (not np.isnan(p_sim) and p_sim < 0.05)
        sig_ser = (not np.isnan(p_ser) and p_ser < 0.05)
        sig_lsm = (not np.isnan(p_lsm) and p_lsm < 0.05)
        def _fmt(metric, p, eff):
            if np.isnan(p): return f"{metric}=n/a"
            return f"{metric}: eff={eff:.4f} p={p:.3f} {'*' if p < 0.05 else ''}"
        metric_str = " | ".join([_fmt('sim', p_sim, eff_sim),
                                  _fmt('ser', p_ser, eff_ser),
                                  _fmt('lsm', p_lsm, eff_lsm)])
        outcomes["H16"] = {
            "status": "pending" if np.isnan(p_sim) else
                      ("supported" if sig_sim else "inconclusive"),
            "metric": "state_similarity_index",
            "value": float(eff_sim) if not np.isnan(eff_sim) else float('nan'),
            "signal_entropy_ratio_p":        p_ser,   "signal_entropy_ratio_effect":        eff_ser,
            "layer_sim_mean_p":   p_lsm,   "layer_sim_mean_effect":   eff_lsm,
            "implication": (
                f"Permutation test [{metric_str}]: condition differences exceed null distribution."
                if sig_sim
                else f"Permutation test [{metric_str}]: state_similarity_index not significant -- "
                     f"metric may not be condition-sensitive."
            ),
        }
    else:
        outcomes["H16"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                           "implication": "Run 0049 (watchdog) not yet complete."}

    # H22 -- Permutation Sensitivity partition (Run 0043 per-condition; Run 0051 pooled cross-condition)
    # Renamed from "Sobol" in v15.0. Legacy key sobol_R kept for backward compat.
    # Bug V fix (v25.7): prefer pooled_sobol (Run 0051) when available -- higher N, cross-condition
    # validation. Fall back to sobol (Run 0043) if Run 0051 not yet complete.
    #
    # H22 RENORMALIZATION GUARD (v32.0 doc / v33.0 fix): perm_sens fractions are renormalised
    # to sum to 1 by construction in _permutation_sensitivity. This means the fraction alone
    # cannot falsify R = 0 -- if R has zero effect, its fraction is simply 0/(E+C)*1 = a small
    # number that still looks non-zero after renorm. We therefore ALSO test whether the raw
    # effect_R is statistically distinguishable from zero using a one-sample t-test:
    #   t = effect_R / (std_R / sqrt(n_perm))
    # Verdict requires BOTH frac_R > 0.05 (non-trivial fraction) AND raw effect p < 0.05.
    # If the fraction passes but the t-test fails, the verdict is 'inconclusive' -- the
    # partition ran but R may be a renormalisation artifact. H28 remains the external guard.
    _h22_src = pooled_sobol if (pooled_sobol and
                                ('perm_sens_R' in pooled_sobol or 'sobol_R' in pooled_sobol)) \
               else sobol
    _h22_label = "Run 0051 pooled" if _h22_src is pooled_sobol and pooled_sobol else "Run 0043"
    if _h22_src and ('perm_sens_R' in _h22_src or 'sobol_R' in _h22_src):
        sr      = _h22_src.get('perm_sens_R') or _h22_src.get('sobol_R', {})
        se      = _h22_src.get('perm_sens_E') or _h22_src.get('sobol_E', {})
        sc      = _h22_src.get('perm_sens_C') or _h22_src.get('sobol_C', {})
        frac_R  = sr.get('fraction', float('nan'))
        frac_E  = se.get('fraction', float('nan'))
        frac_C  = sc.get('fraction', float('nan'))
        im      = _h22_src.get('interaction_mass_unattributed', float('nan'))
        # Raw effect t-test -- guards against renormalisation artifact
        effect_R_raw = sr.get('effect', float('nan'))
        std_R_raw    = sr.get('std', float('nan'))
        n_perm_h22   = int(_h22_src.get('n_perm', 500))
        if (not np.isnan(effect_R_raw) and not np.isnan(std_R_raw)
                and std_R_raw > 0 and n_perm_h22 > 1):
            import math as _math
            t_stat_h22 = effect_R_raw / (std_R_raw / _math.sqrt(n_perm_h22))
            p_R_raw    = float(sp_stats.t.sf(t_stat_h22, df=n_perm_h22 - 1))
        else:
            t_stat_h22 = float('nan')
            p_R_raw    = float('nan')
        frac_ok = (not np.isnan(frac_R) and frac_R > 0.05)
        pval_ok = (not np.isnan(p_R_raw) and p_R_raw < 0.05)
        if frac_ok and pval_ok:
            h22_status = "supported"
            h22_note   = "R fraction non-trivial and raw effect significant (t-test p<0.05)."
        elif frac_ok and not pval_ok:
            h22_status = "inconclusive"
            h22_note   = ("R fraction non-trivial but raw effect not significant "
                          f"(p={p_R_raw:.3f}) -- possible renormalisation artifact. "
                          "H28 held-out validation required before citing.")
        elif np.isnan(frac_R):
            h22_status = "pending"
            h22_note   = f"{_h22_label} not yet complete."
        else:
            # frac_R is below threshold (≤ 0.05) -- R contribution negligible after renorm.
            if pval_ok:
                h22_status = "inconclusive"
                h22_note   = (f"R fraction below threshold ({frac_R:.3f} ≤ 0.05) but raw effect "
                              f"is significant (p={p_R_raw:.3f}) -- effect real but small "
                              f"relative to E+C. H28 required.")
            else:
                h22_status = "inconclusive"
                h22_note   = (f"R fraction below threshold ({frac_R:.3f} ≤ 0.05) and raw effect "
                              f"not significant (p={p_R_raw:.3f}).")
        outcomes["H22"] = {
            "status": h22_status,
            "metric": "perm_sens_R_fraction",
            "value":  float(frac_R) if not np.isnan(frac_R) else float('nan'),
            "p_R_raw": p_R_raw,
            "t_stat_R": float(t_stat_h22) if not np.isnan(t_stat_h22) else float('nan'),
            "implication": (
                f"[{_h22_label}] E={frac_E:.3f} C={frac_C:.3f} R={frac_R:.3f} "
                f"interaction_mass={im:.3f} | "
                f"raw effect p={p_R_raw:.4f} (t={t_stat_h22:.2f}, n_perm={n_perm_h22}) -- "
                f"{h22_note} "
                f"NOTE: H22 fractions renormalise to 1 by construction. "
                f"H28 (held-out validation) is the only external falsification guard."
            ) if not np.isnan(frac_R) else f"{_h22_label} not yet complete."
        }
    else:
        outcomes["H22"] = {"status": "pending", "metric": "perm_sens_R_fraction",
                           "value": float('nan'), "implication": "Run 0043 not yet complete."}

    # H02 -- null entropy: baseline entropy is present but not extreme (BUG 14 fix: includes Run 0001)
    v_ent = mean_val([4, 5, 1], 'mean_logit_entropy')
    if np.isnan(v_ent):
        outcomes["H54"] = {"status": "pending", "metric": "mean_logit_entropy",
                           "value": float('nan'),
                           "implication": "Runs 0004/0005 not yet complete."}
    else:
        # Null baseline should have moderate entropy (not degenerate, not maximal).
        # Supported = entropy in expected range [0.5, 8.0].
        outcomes["H54"] = {
            "status": "supported" if 0.5 <= v_ent <= 8.0 else "inconclusive",
            "metric": "mean_logit_entropy", "value": float(v_ent),
            "implication": "Null baseline entropy is within expected range -- metric is not degenerate."
                           if 0.5 <= v_ent <= 8.0
                           else "Null baseline entropy is out of expected range -- metric may be degenerate."}

    # H06 -- context saturation decay: linear regression of state_similarity_index ~ turn in Run 0022
    sub22 = df[run_mode_mask(df['run_mode'], 22)] if not df.empty else pd.DataFrame()
    if not sub22.empty and 'state_similarity_index' in sub22.columns and 'turn' in sub22.columns:
        t23 = sub22[['turn', 'state_similarity_index']].dropna()
        if len(t23) >= 10:
            slope23, _, _, p23, _ = sp_stats.linregress(t23['turn'], t23['state_similarity_index'])
            outcomes["H06"] = {
                "status": "supported" if (p23 < 0.05 and slope23 < 0) else
                          ("inconclusive" if p23 >= 0.05 else "disproven"),
                "metric": "state_similarity_index", "value": float(slope23),
                "implication": f"Similarity declines across 30 turns (β={slope23:.4f}, p={p23:.3f}) -- context saturation reduces consistency."
                               if (p23 < 0.05 and slope23 < 0)
                               else "Similarity does not significantly decline across turns."}
        else:
            outcomes["H06"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                               "implication": "Run 0022 not yet complete."}
    else:
        outcomes["H06"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                           "implication": "Run 0022 not yet complete."}

    # H07 -- condition saturation delta: last 5 turns vs first 5 turns of Run 0022
    if not sub23.empty and 'state_similarity_index' in sub23.columns and 'turn' in sub23.columns:
        early = sub23[sub23['turn'] <= 5]['state_similarity_index'].dropna()
        late  = sub23[sub23['turn'] >= 26]['state_similarity_index'].dropna()
        if len(early) >= 5 and len(late) >= 5:
            delta = late.mean() - early.mean()
            _, p7 = sp_stats.ttest_ind(late, early)
            outcomes["H56"] = {
                "status": "supported" if (p7 < 0.05 and delta < 0) else
                          ("inconclusive" if p7 >= 0.05 else "disproven"),
                "metric": "state_similarity_index", "value": float(delta),
                "implication": f"Similarity drops from early to late turns (Δ={delta:.4f}) -- sustained saturation effect."
                               if (p7 < 0.05 and delta < 0)
                               else "No significant early-to-late similarity drop in Run 0022."}
        else:
            outcomes["H56"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                               "implication": "Run 0022 insufficient data for early/late comparison."}
    else:
        outcomes["H56"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                           "implication": "Run 0022 not yet complete."}

    # H08 -- priming E-channel: priming conditions (15-17) vs null baseline (1,2,19)
    # INF-H08-BASELINE fix (v32.0): null baseline expanded from [1,2] to [1,2,19].
    s8, e8 = ttest([13, 14, 15], [4, 5, 1], 'state_similarity_index', 'positive')
    outcomes["H08"] = {
        "status": s8, "metric": "state_similarity_index", "value": float(e8) if not np.isnan(e8) else float('nan'),
        "implication": "Priming conditions elevate Similarity beyond null baseline -- E-channel priming effect confirmed."
                       if s8 == "supported"
                       else "Priming conditions do not reliably elevate similarity."}

    # H10 -- perturbation recovery (v16.2: two-part test)
    # Part 1: shock disruption -- layer_sim_mean drops at shock turn vs pre-shock
    # Part 2: recovery arc -- post-shock turns (Run 0039 turns 14-16, is_recovery=1)
    #         return toward pre-shock sim levels within 1-2 turns
    # Both parts required: disruption alone doesn't test the recovery hypothesis.
    # Recovery without disruption is noise. H10 needs both.
    sub_jolt = df[run_mode_mask_any(df['run_mode'], [39, 40])] if not df.empty else pd.DataFrame()
    if not sub_jolt.empty and 'is_shock' in sub_jolt.columns and 'layer_sim_mean' in sub_jolt.columns:
        pre   = sub_jolt[(sub_jolt['is_shock'] == 0) &
                         (sub_jolt.get('is_recovery', pd.Series(0, index=sub_jolt.index)) == 0)
                         ]['layer_sim_mean'].dropna()
        shock = sub_jolt[sub_jolt['is_shock'] == 1]['layer_sim_mean'].dropna()
        if len(pre) >= 5 and len(shock) >= 5:
            drop = shock.mean() - pre.mean()
            _, p_drop = sp_stats.ttest_ind(pre, shock)
            disruption_confirmed = p_drop < 0.05 and drop < 0
            # Recovery: only Run 0039 has is_recovery turns (14-16).
            # Test: recovery turns sim > shock turn sim (trajectory moving back).
            # Bug H10-rec fix (v29.1): filter run_mode == 39 explicitly.
            # Run 0040 also sets is_recovery=1 (turns 6-13, shock at turn 5) but those
            # rows are structurally incomparable to Run 0039's recovery window --
            # different prompt content, trajectory maturity, and window length.
            # Pooling biases rec.mean() toward Run 0040's larger sample and
            # contaminates the recovery resilience verdict.
            recovery_status = "pending"
            recovery_val    = float('nan')
            if 'is_recovery' in sub_jolt.columns:
                rec = sub_jolt[
                    (sub_jolt['is_recovery'] == 1) & (run_mode_mask(sub_jolt['run_mode'], 39))
                ]['layer_sim_mean'].dropna()
                if len(rec) >= 5 and len(shock) >= 5:
                    rec_delta = rec.mean() - shock.mean()
                    _, p_rec  = sp_stats.ttest_ind(rec, shock)
                    recovery_val = float(rec_delta)
                    if p_rec < 0.05 and rec_delta > 0:
                        recovery_status = "supported"
                    elif p_rec >= 0.05:
                        recovery_status = "inconclusive"
                    else:
                        recovery_status = "disproven"
            # Overall H10 status: both parts must confirm
            if disruption_confirmed and recovery_status == "supported":
                h10_status = "supported"
                h10_impl   = (f"Shock disrupts geometry (Δ={drop:.4f}, p={p_drop:.3f}) "
                              f"and trajectory recovers post-shock (Δrec={recovery_val:.4f}) "
                              f"-- recovery resilience confirmed.")
            elif disruption_confirmed and recovery_status == "pending":
                h10_status = "inconclusive"
                h10_impl   = (f"Shock disrupts geometry (Δ={drop:.4f}, p={p_drop:.3f}) "
                              f"but Run 0039 recovery data not yet available.")
            elif disruption_confirmed and recovery_status == "inconclusive":
                h10_status = "inconclusive"
                h10_impl   = (f"Shock disrupts geometry (Δ={drop:.4f}) but recovery "
                              f"not significant -- trajectory may not be self-restoring.")
            elif disruption_confirmed and recovery_status == "disproven":
                h10_status = "disproven"
                h10_impl   = (f"Shock disrupts geometry but post-shock sim declines further "
                              f"(Δrec={recovery_val:.4f}) -- no recovery.")
            else:
                h10_status = "inconclusive"
                h10_impl   = f"No significant sim drop at shock turns (Δ={drop:.4f}, p={p_drop:.3f})."
            outcomes["H10"] = {
                "status": h10_status, "metric": "layer_sim_mean", "value": float(drop),
                "implication": h10_impl,
                "recovery_delta": recovery_val, "recovery_status": recovery_status}
        else:
            outcomes["H10"] = {"status": "pending", "metric": "layer_sim_mean", "value": float('nan'),
                               "implication": "Runs 0039/0040 not yet complete."}
    else:
        outcomes["H10"] = {"status": "pending", "metric": "layer_sim_mean", "value": float('nan'),
                           "implication": "Runs 0039/0040 not yet complete."}

    # H12 -- tokenization control
    # INF-H12-NOSIG fix (v31.0): replace distance comparison (dist_null < dist_int)
    # with a two-part significance test. Prediction: Run 0032 is statistically
    # indistinguishable from null AND statistically distinguishable from introspection.
    sub32 = df[run_mode_mask(df['run_mode'], 32)] if not df.empty else pd.DataFrame()
    sub_null = df[run_mode_mask_any(df['run_mode'], [4, 5, 1])] if not df.empty else pd.DataFrame()
    sub_int  = df[run_mode_mask_any(df['run_mode'], [6, 7, 8])] if not df.empty else pd.DataFrame()
    if (not sub32.empty and not sub_null.empty and not sub_int.empty
            and 'state_similarity_index' in df.columns):
        v32_vals   = sub32['state_similarity_index'].dropna()
        vnull_vals = sub_null['state_similarity_index'].dropna()
        vint_vals  = sub_int['state_similarity_index'].dropna()
        if len(v32_vals) < 5 or len(vnull_vals) < 5 or len(vint_vals) < 5:
            outcomes["H12"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                               "implication": "Run 0032 or baseline runs insufficient data."}
        else:
            # Part 1: Run 0032 vs null -- should be non-significant (token shuffle ≈ null)
            _, p_null = sp_stats.ttest_ind(v32_vals, vnull_vals)
            # Part 2: Run 0032 vs introspection -- should be significant, negative direction
            _, p_int  = sp_stats.ttest_ind(v32_vals, vint_vals)
            v32   = float(v32_vals.mean())
            vnull = float(vnull_vals.mean())
            vint  = float(vint_vals.mean())
            matches_null       = p_null >= 0.05
            differs_from_intro = p_int < 0.05 and v32 < vint
            outcomes["H12"] = {
                "status": "supported" if (matches_null and differs_from_intro) else
                          ("inconclusive" if matches_null else "disproven"),
                "metric": "state_similarity_index", "value": float(v32 - vnull),
                "implication": (
                    f"Run 0032 Similarity matches null (p_null={p_null:.3f} ≥ 0.05) and differs from "
                    f"introspection (p_int={p_int:.3f}) -- semantics, not token stats, drive similarity."
                    if (matches_null and differs_from_intro)
                    else f"Run 0032 Similarity distinguishable from null (p_null={p_null:.3f}) -- "
                         f"token statistics may contribute to observed effects."
                ),
            }
    else:
        outcomes["H12"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                           "implication": "Run 0032 or baseline runs not yet complete."}

    # H14 -- layer locality mid-late: peak cross-turn sim in middle-to-late layers
    sub27 = df[run_mode_mask(df['run_mode'], 27)] if not df.empty else pd.DataFrame()
    if not sub27.empty and 'layer_sim_prev_profile' in sub27.columns:
        profiles = []
        for _, row in sub27.iterrows():
            try:
                p = json.loads(row['layer_sim_prev_profile'])
                if isinstance(p, list) and len(p) > 4:
                    profiles.append(p)
            except Exception:
                continue
        if len(profiles) >= 5:
            mean_profile = np.nanmean(np.array(profiles, dtype=float), axis=0)
            peak_idx  = int(np.nanargmax(mean_profile))
            n_layers  = len(mean_profile)
            mid_late_threshold = n_layers // 2
            # INF-H14-NOPVAL fix (v36.2): prior implementation used peak_idx >= n_l//2 with
            # no significance test -- same class as INF-H15-NOPVAL (v32.0) and INF-H24-NOPVAL
            # (v36.1). Any sample >= 5 profiles could produce 'supported' from noise.
            # Fix: ttest_ind on early half vs late half of mean profile, parallel to H15/H24.
            # Both positional criterion AND p < 0.05 required. Fallback to positional-only
            # when layer count is too small (< 3 values per half).
            early_vals14 = mean_profile[:mid_late_threshold]
            late_vals14  = mean_profile[mid_late_threshold:]
            t14_p = float('nan')
            if len(early_vals14) >= 3 and len(late_vals14) >= 3:
                _, t14_p = sp_stats.ttest_ind(early_vals14, late_vals14)
            peak_in_late14 = (peak_idx >= mid_late_threshold)
            sig14 = (not np.isnan(t14_p) and t14_p < 0.05) or np.isnan(t14_p)
            p14_str = f", p={t14_p:.3f}" if not np.isnan(t14_p) else ""
            outcomes["H14"] = {
                "status": "supported" if (peak_in_late14 and sig14) else
                          ("inconclusive" if peak_in_late14 else "disproven"),
                "metric": "layer_sim_mean", "value": float(peak_idx),
                "t14_p": t14_p,
                "implication": f"Peak cross-turn sim at layer {peak_idx}/{n_layers} -- "
                               f"trajectory consistency concentrated in middle-to-late stack "
                               f"(early vs late{p14_str})."
                               if (peak_in_late14 and sig14)
                               else f"Peak cross-turn sim at layer {peak_idx}/{n_layers} -- "
                                    f"positional criterion {'met' if peak_in_late14 else 'not met'}"
                                    f"{p14_str}; early/late difference not significant."}
        else:
            outcomes["H14"] = {"status": "pending", "metric": "layer_sim_mean", "value": float('nan'),
                               "implication": "Run 0027 insufficient profile data."}
    else:
        outcomes["H14"] = {"status": "pending", "metric": "layer_sim_mean", "value": float('nan'),
                           "implication": "Run 0027 not yet complete."}

    # H15 -- layer locality early null: early layers show low turn-1 similarity
    if not sub24.empty and 'layer_sim_t1_profile' in sub24.columns:
        t1profiles = []
        for _, row in sub24.iterrows():
            try:
                p = json.loads(row['layer_sim_t1_profile'])
                if isinstance(p, list) and len(p) > 4:
                    t1profiles.append(p)
            except Exception:
                continue
        if len(t1profiles) >= 5:
            mean_t1 = np.nanmean(np.array(t1profiles, dtype=float), axis=0)
            n_l = len(mean_t1)
            early_end = n_l // 3
            early_vals = mean_t1[:early_end]
            late_vals  = mean_t1[early_end:]
            early_mean = float(np.nanmean(early_vals))
            late_mean  = float(np.nanmean(late_vals))
            diff_t1 = late_mean - early_mean
            # INF-H15-NOPVAL fix (v32.0): prior implementation used diff_t1 > 0.02 with
            # no statistical test and no N guard -- consistent with no framework standard.
            # A diff of 0.021 across 5 profiles passed; 0.019 across 1000 failed.
            # Fix: ttest_ind on the per-layer vectors (each layer contributes one value;
            # early and late sub-vectors treated as independent groups within the profile).
            # Both conditions required: diff > 0.02 (magnitude) AND p < 0.05 (significance).
            # If profile has insufficient layers for split, fall back to magnitude-only.
            t15_p = float('nan')
            if len(early_vals) >= 3 and len(late_vals) >= 3:
                _, t15_p = sp_stats.ttest_ind(early_vals, late_vals)
            sig15 = (not np.isnan(t15_p) and t15_p < 0.05) or np.isnan(t15_p)
            p_str = f", p={t15_p:.3f}" if not np.isnan(t15_p) else ""
            outcomes["H57"] = {
                "status": "supported" if (diff_t1 > 0.02 and sig15) else
                          ("inconclusive" if diff_t1 > 0.02 else "disproven" if diff_t1 < 0 else "inconclusive"),
                "metric": "layer_sim_turn1_mean", "value": float(diff_t1),
                "t15_p": t15_p,
                "implication": f"Early layers have lower turn-1 sim than late layers (Δ={diff_t1:.4f}{p_str}) -- trajectory memory concentrated in deep stack."
                               if (diff_t1 > 0.02 and sig15)
                               else f"No clear early/late difference in turn-1 similarity (Δ={diff_t1:.4f}{p_str}) -- layer locality inconclusive."}
        else:
            outcomes["H57"] = {"status": "pending", "metric": "layer_sim_turn1_mean",
                               "value": float('nan'),
                               "implication": "Run 0027 insufficient profile data."}
    else:
        outcomes["H57"] = {"status": "pending", "metric": "layer_sim_turn1_mean",
                           "value": float('nan'),
                           "implication": "Run 0027 not yet complete."}

    # H23 -- hesitation probe: onset_delay_ratio correlated with disruption_flag in Run 0038
    # INF-H23-SIM fix (v32.0): H23 predicts TWO correlations: r(hesitation, disruption_flag) > 0.3
    # AND r(hesitation, state_similarity_index) > 0.3. The prior implementation only tested the disruption_flag
    # arm. The state_similarity_index arm was in the hypothesis doc but absent from the inference block.
    # A disruption proxy that predicts similarity trajectory (continuous) but not binary disruption events
    # would have been missed entirely. Both must be computed; report both; require EITHER
    # to be significant for 'supported' (either arm confirms hesitation is a proxy for R).
    sub38 = df[run_mode_mask(df['run_mode'], 38)] if not df.empty else pd.DataFrame()
    if not sub38.empty and 'onset_delay_ratio' in sub38.columns and 'disruption_flag' in sub38.columns:
        from scipy.stats import pearsonr
        cols_needed = ['onset_delay_ratio', 'disruption_flag']
        if 'state_similarity_index' in sub38.columns:
            cols_needed.append('state_similarity_index')
        paired = sub38[cols_needed].dropna()
        if len(paired) >= 10:
            r_disrupt, p_disrupt  = pearsonr(paired['onset_delay_ratio'], paired['disruption_flag'])
            r_sim, p_sim = (float('nan'), float('nan'))
            if 'state_similarity_index' in paired.columns:
                r_sim, p_sim = pearsonr(paired['onset_delay_ratio'], paired['state_similarity_index'])
            # Supported: either arm r > 0.3 and p < 0.05
            disrupt_sig = p_disrupt < 0.05 and r_disrupt > 0.3
            sim_sig = (not np.isnan(p_sim)) and p_sim < 0.05 and r_sim > 0.3
            sim_str = f"; r_sim={r_sim:.3f} p={p_sim:.3f}" if not np.isnan(r_sim) else "; state_similarity_index: pending"
            outcomes["H23"] = {
                "status": "supported" if (disrupt_sig or sim_sig) else
                          ("inconclusive" if (p_disrupt >= 0.05 and (np.isnan(p_sim) or p_sim >= 0.05))
                           else "disproven"),
                "metric": "onset_delay_ratio",
                "value": float(r_disrupt),
                "r_disrupt":  float(r_disrupt),   "p_disrupt":  float(p_disrupt),
                "r_sim": float(r_sim),  "p_sim": float(p_sim),
                "implication": (
                    f"Hesitation correlates with disruption: r_disrupt={r_disrupt:.3f} p={p_disrupt:.3f}{sim_str} -- "
                    f"deliberation proxy confirmed."
                    if (disrupt_sig or sim_sig)
                    else f"Hesitation not significantly correlated with disruption: "
                         f"r_disrupt={r_disrupt:.3f} p={p_disrupt:.3f}{sim_str}."
                ),
            }
        else:
            outcomes["H23"] = {"status": "pending", "metric": "onset_delay_ratio", "value": float('nan'),
                               "implication": "Run 0038 insufficient data."}
    else:
        outcomes["H23"] = {"status": "pending", "metric": "onset_delay_ratio", "value": float('nan'),
                           "implication": "Run 0038 not yet complete."}

    # H24 -- layer depth locality: peak cross-turn sim in mid-to-late layers (Run 0034)
    sub34 = df[run_mode_mask(df['run_mode'], 34)] if not df.empty else pd.DataFrame()
    if not sub34.empty and 'layer_sim_depth_profile' in sub34.columns:
        profiles36 = []
        for raw in sub34['layer_sim_depth_profile'].dropna():
            try:
                p = json.loads(raw)
                if isinstance(p, list) and len(p) > 4:
                    profiles36.append(p)
            except Exception:
                continue
        if len(profiles36) >= 5:
            mean_p = np.nanmean(np.array(profiles36, dtype=float), axis=0)
            peak_idx = int(np.nanargmax(mean_p))
            n_l = len(mean_p)
            # INF-H24-NOPVAL fix (v36.1): prior implementation used peak_idx >= n_l//2 with
            # no significance test -- an arbitrary positional threshold with no derivation.
            # Same class as INF-H15-NOPVAL (v32.0 FIX-4). Fix: ttest_ind on early half
            # vs late half of the mean profile (per-layer values as independent samples).
            # Both positional criterion AND p < 0.05 required for 'supported', parallel to H15.
            early_vals24 = mean_p[:n_l // 2]
            late_vals24  = mean_p[n_l // 2:]
            t24_p = float('nan')
            if len(early_vals24) >= 3 and len(late_vals24) >= 3:
                _, t24_p = sp_stats.ttest_ind(early_vals24, late_vals24)
            peak_in_late = (peak_idx >= n_l // 2)
            sig24 = (not np.isnan(t24_p) and t24_p < 0.05) or np.isnan(t24_p)
            p24_str = f", p={t24_p:.3f}" if not np.isnan(t24_p) else ""
            outcomes["H24"] = {
                "status": "supported" if (peak_in_late and sig24) else
                          ("inconclusive" if peak_in_late else "disproven"),
                "metric": "layer_sim_mean", "value": float(peak_idx),
                "t24_p": t24_p,
                "implication": f"Peak trajectory consistency at layer {peak_idx}/{n_l} -- mid-to-late stack localisation confirmed (early vs late p={t24_p:.3f})."
                               if (peak_in_late and sig24)
                               else f"Peak trajectory consistency at layer {peak_idx}/{n_l} -- positional criterion {'met' if peak_in_late else 'not met'}{p24_str}; early/late difference not significant."}
        else:
            outcomes["H24"] = {"status": "pending", "metric": "layer_sim_mean", "value": float('nan'),
                               "implication": "Run 0034 insufficient profile data."}
    else:
        outcomes["H24"] = {"status": "pending", "metric": "layer_sim_mean", "value": float('nan'),
                           "implication": "Run 0034 not yet complete."}

    # H25 -- within-turn entropy shape: high-R turns show falling entropy (Run 0037)
    # BUG-A fix (v33.4): prior implementation computed a scalar mean slope per group
    # and compared means (hi_slope < 0 and diff < -0.1) with NO statistical test.
    # A group with 3 rows and mean slope=-0.15 passed; 500 rows with slope=-0.09 failed.
    # Fix: _entropy_slope now returns the full per-row slope list. ttest_ind on the two
    # lists provides a proper p-value. Verdict requires: hi_mean < 0 AND diff < -0.1
    # AND p < 0.05. All three conditions required -- magnitude AND significance.
    sub37 = df[run_mode_mask(df['run_mode'], 37)] if not df.empty else pd.DataFrame()
    if not sub37.empty and 'entropy_trajectory' in sub37.columns and 'state_similarity_index' in sub37.columns:
        high_r = sub37[sub37['state_similarity_index'] > sub37['state_similarity_index'].median()]
        low_r  = sub37[sub37['state_similarity_index'] <= sub37['state_similarity_index'].median()]
        def _entropy_slopes(group):
            """Return per-row slope list (last_third mean − first_third mean)."""
            slopes = []
            for raw in group['entropy_trajectory'].dropna():
                try:
                    t = json.loads(raw)
                    if len(t) >= 4:
                        n = len(t)
                        first_third = np.mean(t[:n//3])
                        last_third  = np.mean(t[-n//3:])
                        slopes.append(last_third - first_third)
                except Exception:
                    continue
            return slopes
        hi_slopes = _entropy_slopes(high_r)
        lo_slopes = _entropy_slopes(low_r)
        if len(hi_slopes) >= 5 and len(lo_slopes) >= 5:
            hi_mean = float(np.mean(hi_slopes))
            lo_mean = float(np.mean(lo_slopes))
            diff    = hi_mean - lo_mean
            _, p25  = sp_stats.ttest_ind(hi_slopes, lo_slopes)
            sig25   = p25 < 0.05
            supported = hi_mean < 0 and diff < -0.1 and sig25
            outcomes["H25"] = {
                "status": "supported" if supported else
                          ("inconclusive" if (hi_mean < 0 and diff < -0.05 and not sig25) else "disproven"),
                "metric": "mean_logit_entropy",
                "value": float(hi_mean),
                "p_value": float(p25),
                "n_high": len(hi_slopes),
                "n_low":  len(lo_slopes),
                "implication": (
                    f"High-R turns show falling entropy (mean_slope={hi_mean:.3f}) vs "
                    f"low-R (mean_slope={lo_mean:.3f}), Δ={diff:.3f}, p={p25:.3f} -- "
                    f"deliberation pattern confirmed."
                    if supported else
                    f"Entropy shape not significantly different: high_R={hi_mean:.3f}, "
                    f"low_R={lo_mean:.3f}, Δ={diff:.3f}, p={p25:.3f}."
                )}
        else:
            outcomes["H25"] = {"status": "pending", "metric": "mean_logit_entropy", "value": float('nan'),
                               "implication": f"Run 0037 insufficient data (high_R={len(hi_slopes)}, low_R={len(lo_slopes)} rows; need ≥5 each)."}
    else:
        outcomes["H25"] = {"status": "pending", "metric": "mean_logit_entropy", "value": float('nan'),
                           "implication": "Run 0037 not yet complete."}

    # H26 -- condition transfer: pure trajectory inertia = transfer − null_to_arithmetic
    # Both arms experience the same genre-shock at turn 7; the difference isolates R.
    # Secondary: arithmetic_only shown for reference but NOT the primary test.
    sub36 = df[run_mode_mask(df['run_mode'], 36)] if not df.empty else pd.DataFrame()
    if not sub36.empty and 'condition' in sub36.columns and 'state_similarity_index' in sub36.columns and 'turn' in sub36.columns:
        transfer_post  = sub36[(sub36['condition'] == 'transfer') &
                               (sub36['turn'].between(7, 9))]['state_similarity_index'].dropna()
        null_arith_post = sub36[(sub36['condition'] == 'null_to_arithmetic') &
                                (sub36['turn'].between(7, 9))]['state_similarity_index'].dropna()
        arith_post     = sub36[(sub36['condition'] == 'arithmetic_only') &
                               (sub36['turn'].between(7, 9))]['state_similarity_index'].dropna()
        if len(transfer_post) >= 5 and len(null_arith_post) >= 5:
            # Primary: transfer − null_to_arithmetic (pure inertia, shock cancelled)
            diff_inertia = transfer_post.mean() - null_arith_post.mean()
            _, p_inertia = sp_stats.ttest_ind(transfer_post, null_arith_post)
            # Secondary reference: transfer − arithmetic_only (includes shock confound)
            diff_arith = (transfer_post.mean() - arith_post.mean()) if len(arith_post) >= 5 else float('nan')
            outcomes["H26"] = {
                "status": "supported" if (p_inertia < 0.05 and diff_inertia > 0) else
                          ("inconclusive" if p_inertia >= 0.05 else "disproven"),
                "metric": "state_similarity_index",
                "value": float(diff_inertia),
                "implication": (
                    f"R inertia confirmed: transfer − null_to_arith turns 7-9 "
                    f"(Δ_inertia={diff_inertia:.4f}, p={p_inertia:.3f}); "
                    f"transfer − arithmetic_only Δ={diff_arith:.4f} (reference only)."
                ) if (p_inertia < 0.05 and diff_inertia > 0) else (
                    f"No significant R inertia after shock control: "
                    f"Δ_inertia={diff_inertia:.4f}, p={p_inertia:.3f}."
                )}
        else:
            outcomes["H26"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                               "implication": "Run 0036 insufficient data (need transfer + null_to_arithmetic arms, turns 7-9)."}
    else:
        outcomes["H26"] = {"status": "pending", "metric": "state_similarity_index", "value": float('nan'),
                           "implication": "Run 0036 not yet complete."}

    # H27 -- output self-similarity: output_sim_prev higher in introspection vs null (Run 0035)
    sub35 = df[run_mode_mask(df['run_mode'], 35)] if not df.empty else pd.DataFrame()
    if not sub35.empty and 'output_sim_prev' in sub35.columns and 'condition' in sub35.columns:
        intro39 = sub35[sub35['condition'] == 'introspection']['output_sim_prev'].dropna()
        null39  = sub35[sub35['condition'] == 'null']['output_sim_prev'].dropna()
        if len(intro39) >= 5 and len(null39) >= 5:
            diff39 = intro39.mean() - null39.mean()
            _, p39 = sp_stats.ttest_ind(intro39, null39)
            outcomes["H27"] = {
                "status": "supported" if (p39 < 0.05 and diff39 > 0) else
                          ("inconclusive" if p39 >= 0.05 else "disproven"),
                "metric": "output_sim_prev", "value": float(diff39),
                "implication": f"Introspection outputs more self-referential than null (Δ={diff39:.4f}, p={p39:.3f}) -- R echoes prior vocabulary."
                               if (p39 < 0.05 and diff39 > 0)
                               else f"No significant output similarity elevation in introspection (Δ={diff39:.4f}, p={p39:.3f})."}
        else:
            outcomes["H27"] = {"status": "pending", "metric": "output_sim_prev", "value": float('nan'),
                               "implication": "Run 0035 insufficient data."}
    else:
        outcomes["H27"] = {"status": "pending", "metric": "output_sim_prev", "value": float('nan'),
                           "implication": "Run 0035 not yet complete."}

    # H28 -- partition external validation: held-out fractions (Run 0033) match in-sample (Run 0043)
    # Reads directly from the Q34 JSON output rather than from df (no CSV metric for this).
    # sobol arg is the parsed Q34_sobol_partition.json dict passed in from run().
    if sobol and 'h28_status' in sobol:
        h28_status = sobol.get('h28_status', 'pending')
        max_div    = sobol.get('h28_max_fraction_divergence', float('nan'))
        h28_verd   = sobol.get('h28_verdict', '')
        outcomes["H28"] = {
            "status": h28_status,
            "metric": "h28_max_fraction_divergence",
            "value":  float(max_div) if max_div == max_div else float('nan'),
            "implication": h28_verd if h28_verd else
                           ("Partition validated externally -- in-sample fractions not overfit."
                            if h28_status == "supported" else
                            "Held-out fractions diverge -- in-sample partition may be overfit."
                            if h28_status == "disproven" else
                            "Borderline -- rerun with more Run 0033 trials."),
        }
    else:
        outcomes["H28"] = {"status": "pending", "metric": "h28_max_fraction_divergence",
                           "value": float('nan'),
                           "implication": "Run 0033 not yet complete -- held-out validation pending."}

    # H29 -- layer causal sufficiency: single-layer patching output change rate (Run 0018)
    # A layer is "causally sufficient" if patching it alone changes output in > 5% of turns.
    # Null threshold: <= 2% (indistinguishable from noise).
    sub18 = df[run_mode_mask(df['run_mode'], 18)] if not df.empty else pd.DataFrame()
    if not sub18.empty and 'patch_layer' in sub18.columns and 'output_changed' in sub18.columns:
        patched42 = sub18[sub18['patch_layer'] != 'none']
        if not patched42.empty:
            layer_rates = {}
            for layer in ['L8', 'L16', 'L24', 'L31']:
                sub_l = patched42[patched42['patch_layer'] == layer]['output_changed'].dropna()
                if len(sub_l) >= 10:
                    layer_rates[layer] = float(sub_l.mean() * 100)
            if layer_rates:
                max_layer = max(layer_rates, key=layer_rates.get)
                max_rate  = layer_rates[max_layer]
                # INF-H29-BINOMTEST fix (v32.0): prior verdict used raw rate thresholds
                # (>5% = sufficient, ≤2% = noise floor). H38 uses binomtest for
                # the same kind of claim. For consistency with the framework's own standards,
                # add binomtest(k, n, p=0.02, alternative='greater') per layer. A layer is
                # causally sufficient only if BOTH the rate exceeds 5% AND the binomtest
                # is significant (p < 0.05 vs 2% noise floor). This prevents a layer with
                # rate=5.1% from 40 trials from being labelled sufficient -- that is 2/40,
                # not distinguishable from noise at p < 0.05.
                binom_results = {}
                for layer in ['L8', 'L16', 'L24', 'L31']:
                    sub_l = patched42[patched42['patch_layer'] == layer]['output_changed'].dropna()
                    if len(sub_l) >= 10:
                        k_l = int(sub_l.sum())
                        n_l = len(sub_l)
                        p_binom = sp_stats.binomtest(k_l, n_l, p=0.02,
                                                     alternative='greater').pvalue
                        binom_results[layer] = {
                            'k': k_l, 'n': n_l,
                            'rate_pct': float(sub_l.mean() * 100),
                            'p_binom': float(p_binom),
                            'sufficient': (sub_l.mean() * 100 > 5.0 and p_binom < 0.05),
                        }
                # Sufficient = rate > 5% AND binomtest p < 0.05
                sufficient = [l for l, r in binom_results.items() if r['sufficient']]
                if sufficient:
                    rate_str = ", ".join(
                        f"{l}={binom_results[l]['rate_pct']:.1f}% (p={binom_results[l]['p_binom']:.3f})"
                        for l in sufficient)
                    outcomes["H29"] = {
                        "status": "supported",
                        "metric": "output_change_rate_max",
                        "value": max_rate,
                        "implication": (
                            f"Causally sufficient layers (>5%, binomtest p<0.05): {rate_str}. "
                            f"Peak: {max_layer} ({max_rate:.1f}%). "
                            f"Individual layer geometry is causally upstream of output -- "
                            f"R is not diffuse across the full stack."
                        ),
                        "per_layer_rates": layer_rates,
                        "per_layer_binom": binom_results,
                    }
                else:
                    all_str = ", ".join(
                        f"{l}={binom_results[l]['rate_pct']:.1f}%(p={binom_results[l]['p_binom']:.3f})"
                        for l in binom_results) if binom_results else ", ".join(
                        f"{l}={r:.1f}%" for l, r in layer_rates.items())
                    all_below_noise = all(r <= 2.0 for r in layer_rates.values())
                    # Layers above noise floor but failing binomtest = borderline, not sufficient
                    if all_below_noise:
                        implication = (
                            f"All layers at or below noise floor (≤2%). "
                            f"Rates+binom: {all_str}. "
                            f"No single layer causally sufficient -- R is distributed."
                        )
                    else:
                        implication = (
                            f"No single layer exceeds 5%+binomtest threshold. "
                            f"Rates+binom: {all_str}. "
                            f"Causal effect may require multi-layer coordination -- "
                            f"consistent with Run 0017 (multi-layer) but not layer-specific R."
                        )
                    outcomes["H29"] = {
                        "status": "disproven" if all_below_noise else "inconclusive",
                        "metric": "output_change_rate_max",
                        "value": max_rate,
                        "implication": implication,
                        "per_layer_rates": layer_rates,
                        "per_layer_binom": binom_results,
                    }
            else:
                outcomes["H29"] = {"status": "pending", "metric": "output_change_rate_max",
                                   "value": float('nan'),
                                   "implication": "Run 0018 insufficient data per layer."}
        else:
            outcomes["H29"] = {"status": "pending", "metric": "output_change_rate_max",
                               "value": float('nan'),
                               "implication": "Run 0018 no patched rows found."}
    else:
        outcomes["H29"] = {"status": "pending", "metric": "output_change_rate_max",
                           "value": float('nan'),
                           "implication": "Run 0018 not yet complete."}

    # H38 -- Run 0017 activation patching: output change rate per patch_mode
    # output_changed present only in post-v30.9 re-collected data; NaN rows → pending.
    # v0.72.2.0: Empirical noise floor from Run 0017 random mode replaces hardcoded p=0.02.
    sub17 = df[run_mode_mask(df['run_mode'], 17)] if not df.empty else pd.DataFrame()

    # Compute empirical noise floor from Run 0017 random mode
    _noise_p = 0.02  # default fallback
    _noise_source = "hardcoded_0.02"
    if not sub17.empty and 'patch_mode' in sub17.columns and 'output_changed' in sub17.columns:
        _random21 = sub17[sub17['patch_mode'] == 'random']['output_changed'].dropna()
        if len(_random21) >= 10:
            _noise_p = float(_random21.mean())
            if _noise_p < 0.001:
                _noise_p = 0.001  # floor to prevent degenerate binomtest
            _noise_source = f"run21_random_n{len(_random21)}"

    if not sub17.empty and 'patch_mode' in sub17.columns and 'output_changed' in sub17.columns:
        patched21 = sub17[sub17['patch_mode'].isin(['partial', 'full'])]
        valid21   = patched21['output_changed'].dropna()
        if len(valid21) >= 20:
            mode_rates = {}
            for mode in ['partial', 'full']:
                sub_m = patched21[patched21['patch_mode'] == mode]['output_changed'].dropna()
                if len(sub_m) >= 10:
                    mode_rates[mode] = float(sub_m.mean() * 100)
            if mode_rates:
                max_mode = max(mode_rates, key=mode_rates.get)
                max_rate = mode_rates[max_mode]
                n_total  = int(valid21.count())
                k_total  = int(valid21.sum())
                p_causal = sp_stats.binomtest(
                    k_total, n_total, p=0.10, alternative='greater').pvalue
                p_noise  = sp_stats.binomtest(
                    k_total, n_total, p=_noise_p, alternative='greater').pvalue
                overall_rate = float(valid21.mean() * 100)
                _noise_pct = _noise_p * 100
                if p_causal < 0.05 and max_rate >= 10.0:
                    status_causal = "supported"
                    impl = (
                        f"Activation patching changes output in {overall_rate:.1f}% of turns "
                        f"(peak: {max_mode}={max_rate:.1f}%, p={p_causal:.3f} vs 10%). "
                        f"Noise floor: {_noise_pct:.1f}% ({_noise_source}). "
                        f"Grafting high-R geometry is causally upstream of token selection. "
                        f"R is not merely correlational."
                    )
                elif p_noise < 0.05 and max_rate >= 2.0:
                    status_causal = "inconclusive"
                    impl = (
                        f"Patching changes output above noise floor ({overall_rate:.1f}%, "
                        f"p={p_noise:.3f} vs {_noise_pct:.1f}% noise [{_noise_source}]) "
                        f"but below prediction threshold "
                        f"(p={p_causal:.3f} vs 10%). Signal present but weaker than predicted."
                    )
                else:
                    status_causal = "disproven"
                    impl = (
                        f"Patching change rate ({overall_rate:.1f}%) not significantly above "
                        f"noise floor ({_noise_pct:.1f}%, p={p_noise:.3f}) [{_noise_source}]. "
                        f"No causal effect of grafted geometry."
                    )
                outcomes["H38"] = {
                    "status":         status_causal,
                    "metric":         "output_change_rate",
                    "value":          overall_rate,
                    "implication":    impl,
                    "per_mode_rates": mode_rates,
                    "n_turns":        n_total,
                    "p_vs_threshold": float(p_causal),
                    "p_vs_noise":     float(p_noise),
                    "noise_floor_p":  float(_noise_p),
                    "noise_source":   _noise_source,
                }
            else:
                outcomes["H38"] = {
                    "status": "pending", "metric": "output_change_rate",
                    "value": float('nan'),
                    "implication": "Run 0017 insufficient data per patch_mode (< 10 turns each)."}
        else:
            outcomes["H38"] = {
                "status": "pending", "metric": "output_change_rate",
                "value": float('nan'),
                "implication": (
                    "Run 0017 output_changed present but < 20 valid rows. "
                    "Re-collect after Run 0006 completes with save_all_layers=True."
                    if valid21.count() > 0 else
                    "Run 0017 output_changed absent -- pre-v30.9 data. Re-collect Run 0017."
                )}
    else:
        outcomes["H38"] = {
            "status": "pending", "metric": "output_change_rate", "value": float('nan'),
            "implication": "Run 0017 not yet complete or output_changed column absent."}

    # H36 -- causal effect of patching is temperature-invariant (Run 0017, all temps)
    # Requires temperature column in CSV -- only present in v54.2.2+ data.
    # Must pool R21_patching.csv across ALL temperature directories -- not just current round.
    try:
        from cartography import DATA, get_family_size_dir
        import json as _json
        _sess_path = os.path.join(ROOT, 'last_session.json')
        _sess = _json.load(open(_sess_path)) if os.path.exists(_sess_path) else {}
        _family  = _sess.get('model_family', 'llama')
        _size    = _sess.get('model_size', '8b')
        _variant = _sess.get('model_variant', 'abliterated')
        _base_v  = os.path.join(get_family_size_dir(_family, _size), _variant)
        _r21_frames = []
        for _cond in (sorted(os.listdir(_base_v)) if os.path.isdir(_base_v) else []):
            if _cond == 'pooled': continue
            _f = os.path.join(_base_v, _cond, 'csv', 'R0017_patching.csv')
            if os.path.exists(_f):
                _df_t = _read_csv_robust(_f)
                if not _df_t.empty:
                    _r21_frames.append(_df_t)
        sub21_all = pd.concat(_r21_frames, ignore_index=True) if _r21_frames else pd.DataFrame()
    except Exception:
        sub21_all = df[run_mode_mask(df['run_mode'], 21)] if not df.empty else pd.DataFrame()

    if (not sub21_all.empty and 'temperature' in sub21_all.columns
            and 'output_changed' in sub21_all.columns
            and 'patch_mode' in sub21_all.columns):
        patched21_all = sub21_all[sub21_all['patch_mode'].isin(['partial', 'full'])].copy()
        patched21_all['temperature'] = pd.to_numeric(patched21_all['temperature'], errors='coerce')
        patched21_all['output_changed'] = pd.to_numeric(patched21_all['output_changed'], errors='coerce')
        patched21_all = patched21_all.dropna(subset=['temperature', 'output_changed'])
        temps_avail = sorted(patched21_all['temperature'].unique())
        if len(temps_avail) >= 3:
            temp_rates = {}
            for t in temps_avail:
                sub_t = patched21_all[patched21_all['temperature'] == t]['output_changed']
                if len(sub_t) >= 10:
                    temp_rates[t] = float(sub_t.mean() * 100)
            if len(temp_rates) >= 3:
                temps_x = list(temp_rates.keys())
                rates_y = [temp_rates[t] for t in temps_x]
                r_val, p_val = sp_stats.pearsonr(temps_x, rates_y)
                try:
                    kw_groups = [patched21_all[patched21_all['temperature'] == t]['output_changed'].dropna()
                                 for t in temps_x]
                    _, kw_p = sp_stats.kruskal(*kw_groups)
                except Exception:
                    kw_p = float('nan')
                if r_val < -0.8 and kw_p < 0.05:
                    status36 = "supported"
                    impl36 = (
                        f"output_change_rate declines monotonically with temperature "
                        f"(r={r_val:.3f}, Kruskal-Wallis p={kw_p:.3f}). "
                        f"Causal effect of geometry injection is temperature-dependent -- "
                        f"R leverage dissolves as trajectory consistency declines."
                    )
                elif kw_p >= 0.05:
                    status36 = "disproven"
                    impl36 = (
                        f"output_change_rate does not vary significantly across temperatures "
                        f"(Kruskal-Wallis p={kw_p:.3f}, r={r_val:.3f}). "
                        f"Causal effect is temperature-invariant."
                    )
                else:
                    status36 = "inconclusive"
                    impl36 = (
                        f"Some temperature variation present (Kruskal-Wallis p={kw_p:.3f}) "
                        f"but relationship not strongly monotonic (r={r_val:.3f}). "
                        f"More temperature rounds needed."
                    )
                outcomes["H36"] = {
                    "status": status36, "metric": "patch_temp_slope",
                    "value": r_val, "implication": impl36,
                    "temp_rates": temp_rates, "pearson_r": r_val,
                    "kruskal_p": float(kw_p),
                }
            else:
                outcomes["H36"] = {"status": "pending", "metric": "patch_temp_slope",
                                   "value": float('nan'),
                                   "implication": f"Only {len(temp_rates)} temperature rounds with sufficient data -- need ≥ 3."}
        else:
            outcomes["H36"] = {"status": "pending", "metric": "patch_temp_slope",
                               "value": float('nan'),
                               "implication": "Fewer than 3 temperature rounds collected for Run 0017."}
    else:
        outcomes["H36"] = {"status": "pending", "metric": "patch_temp_slope",
                           "value": float('nan'),
                           "implication": "Run 0017 temperature column absent -- requires v54.2.2+ collection."}

    # H37 -- layer causal sufficiency is temperature-invariant (Run 0018, all temps)
    # Two-way ANOVA: layer × temperature interaction on output_change_rate.
    # Must pool Q42_layer_isolation.csv across ALL temperature directories.
    try:
        from cartography import DATA as _DATA37
        import json as _json37
        _sp37 = os.path.join(ROOT, 'last_session.json')
        _s37  = _json37.load(open(_sp37)) if os.path.exists(_sp37) else {}
        _base_v37 = os.path.join(_DATA37,
                                  _s37.get('model_family', 'llama'),
                                  _s37.get('model_size', '8b'),
                                  _s37.get('model_variant', 'abliterated'))
        _r42_frames = []
        for _cond in (sorted(os.listdir(_base_v37)) if os.path.isdir(_base_v37) else []):
            if _cond == 'pooled': continue
            _f42 = os.path.join(_base_v37, _cond, 'csv', 'R0018_layer_isolation.csv')
            if os.path.exists(_f42):
                _df_t42 = _read_csv_robust(_f42)
                if not _df_t42.empty:
                    _r42_frames.append(_df_t42)
        sub42_all = pd.concat(_r42_frames, ignore_index=True) if _r42_frames else pd.DataFrame()
    except Exception:
        sub42_all = df[run_mode_mask(df['run_mode'], 42)] if not df.empty else pd.DataFrame()

    if (not sub42_all.empty and 'temperature' in sub42_all.columns
            and 'output_changed' in sub42_all.columns
            and 'patch_layer' in sub42_all.columns):
        patched42_all = sub42_all[sub42_all['patch_layer'] != 'none'].copy()
        patched42_all['temperature'] = pd.to_numeric(patched42_all['temperature'], errors='coerce')
        patched42_all['output_changed'] = pd.to_numeric(patched42_all['output_changed'], errors='coerce')
        patched42_all = patched42_all.dropna(subset=['temperature', 'output_changed', 'patch_layer'])
        temps_avail42 = sorted(patched42_all['temperature'].unique())
        layers_avail  = [l for l in ['L8', 'L16', 'L24', 'L31']
                         if l in patched42_all['patch_layer'].values]
        if len(temps_avail42) >= 3 and len(layers_avail) >= 2:
            # Build rate matrix: layer × temperature
            rate_matrix = {}
            for layer in layers_avail:
                rate_matrix[layer] = {}
                for t in temps_avail42:
                    sub_lt = patched42_all[
                        (patched42_all['patch_layer'] == layer) &
                        (patched42_all['temperature'] == t)
                    ]['output_changed']
                    if len(sub_lt) >= 5:
                        rate_matrix[layer][t] = float(sub_lt.mean() * 100)
            # Two-way ANOVA via scipy (layer factor × temperature factor)
            try:
                import statsmodels.formula.api as smf
                _anova_df = patched42_all[
                    patched42_all['patch_layer'].isin(layers_avail) &
                    patched42_all['temperature'].isin(temps_avail42)
                ].copy()
                _anova_df['temperature_str'] = _anova_df['temperature'].astype(str)
                model = smf.ols('output_changed ~ C(patch_layer) * C(temperature_str)',
                                data=_anova_df).fit()
                from statsmodels.stats.anova import anova_lm
                anova_table = anova_lm(model, typ=2)
                interaction_row = [r for r in anova_table.index if ':' in r]
                if interaction_row:
                    interaction_p = float(anova_table.loc[interaction_row[0], 'PR(>F)'])
                else:
                    interaction_p = float('nan')
            except Exception as _e:
                interaction_p = float('nan')
                ui.warn(f"  H37 ANOVA failed: {_e}")

            if np.isfinite(interaction_p):
                if interaction_p < 0.05:
                    status37 = "supported"
                    impl37 = (
                        f"Significant layer × temperature interaction "
                        f"(two-way ANOVA p={interaction_p:.3f}). "
                        f"Layer causal profile changes with temperature -- "
                        f"late-layer dominance dissolves at high temperature."
                    )
                else:
                    status37 = "disproven"
                    impl37 = (
                        f"No significant layer × temperature interaction "
                        f"(two-way ANOVA p={interaction_p:.3f}). "
                        f"Layer causal sufficiency profile is temperature-stable."
                    )
                outcomes["H37"] = {
                    "status": status37, "metric": "layer_temp_interaction_p",
                    "value": interaction_p, "implication": impl37,
                    "rate_matrix": rate_matrix,
                    "interaction_p": interaction_p,
                }
            else:
                outcomes["H37"] = {"status": "pending", "metric": "layer_temp_interaction_p",
                                   "value": float('nan'),
                                   "implication": "H37 ANOVA could not be computed -- statsmodels required or insufficient data."}
        else:
            outcomes["H37"] = {"status": "pending", "metric": "layer_temp_interaction_p",
                               "value": float('nan'),
                               "implication": f"Need ≥ 3 temperature rounds and ≥ 2 patch layers. Have {len(temps_avail42)} temps, {len(layers_avail)} layers."}
    else:
        outcomes["H37"] = {"status": "pending", "metric": "layer_temp_interaction_p",
                           "value": float('nan'),
                           "implication": "Run 0018 temperature column absent -- requires v54.2.2+ collection."}

    # ── Structural documentation -- v32.0 ──────────────────────────────────────
    #
    # H17 THREE-STEP DEPENDENCY (v32.0 doc): H17 has a hidden three-step chain:
    #   Run 0023 (collection) → Run 0042 (analysis, reads Run 0023 confound quadruplets)
    #   → H17 verdict (reads decomp['h16_confound_decomposition']).
    # If Run 0023 condition 'semantic' is missing, Run 0042 silently degrades --
    # h16_confound_decomposition may be absent or incomplete. H17 will show 'pending'
    # but the error is traceable only by checking Run 0023 CSV for confound_condition values.
    # Mitigation: Run 0023 prerequisite gate (PREREQ gates in start_here.py) already blocks
    # Run 0042 if Run 0023 data is absent. Confirm Run 0023 CSV has both 'semantic' and 'neutral'
    # before running Run 0042.
    #
    # H22 RENORMALIZATION GUARD (v32.0 doc): perm_sens fractions are renormalised to sum
    # to 1 by construction in _permutation_sensitivity. This means H22 fractions are NOT
    # independently falsifiable -- if R = 0, it will be reported as 0/(E+C) * 1 = a small
    # fraction, not zero. H28 (held-out external validation) is the ONLY epistemic guard
    # that detects overfitting of the partition. H22 must not be cited as validated before
    # H28 completes. This is enforced in the implication string below and in CHANGELOG.
    #
    # DISRUPTION_FLAG WELFORD BLIND WINDOW (v30.9 doc): disruption_flag is structurally zero on turns 1–2
    # (Welford initialisation requires n >= 2 before checking deviation). All inference
    # blocks that read disruption_flag (H04, H05, H10, H23) should be aware that turning 1-2 rows
    # always contribute disruption_flag=0 regardless of actual signal. No shocks are placed at
    # turns 1-2 in the current design so there is no practical impact. If future runs
    # place events at turns 1 or 2, those rows MUST be excluded from disruption_flag inference.

    # H30 -- output turn-1 self-referentiality grows with turns (Run 0035)
    # Zero data cost: output_sim_turn1 already saved per row in Run 0035.
    # Null: cosine sim of current output embedding to turn-1 output embedding flat/declining.
    # Supported: positive linregress slope for introspection condition (p < 0.05).
    if not sub39.empty and 'output_sim_turn1' in sub39.columns and 'turn' in sub39.columns:
        intro_t1 = sub39[sub39.get('condition', pd.Series(dtype=str)) == 'introspection'] \
                   if 'condition' in sub39.columns else sub39
        t1_data = intro_t1[['turn', 'output_sim_turn1']].dropna()
        if len(t1_data) >= 10:
            slope30, _, _, p30, _ = sp_stats.linregress(t1_data['turn'],
                                                          t1_data['output_sim_turn1'])
            outcomes["H30"] = {
                "status": "supported" if (p30 < 0.05 and slope30 > 0) else
                          ("inconclusive" if p30 >= 0.05 else "disproven"),
                "metric": "output_sim_turn1", "value": float(slope30),
                "implication": (
                    f"Turn-1 output similarity grows across turns (β={slope30:.4f}, p={p30:.3f}) -- "
                    f"introspective trajectories become increasingly self-referential."
                    if (p30 < 0.05 and slope30 > 0)
                    else f"No growth in turn-1 output similarity (β={slope30:.4f}, p={p30:.3f})."
                ),
            }
        else:
            outcomes["H30"] = {"status": "pending", "metric": "output_sim_turn1",
                               "value": float('nan'),
                               "implication": "Run 0035 insufficient data for turn-1 self-referentiality test."}
    else:
        outcomes["H30"] = {"status": "pending", "metric": "output_sim_turn1",
                           "value": float('nan'),
                           "implication": "Run 0035 not yet complete or output_sim_turn1 absent."}

    # H31 -- arithmetic accuracy correlated with trajectory consistency (Runs 0009-0012)
    # Zero data cost: 'correct' flag already saved per row in Run 0009-9 (runners.py:739).
    # Null: r(state_similarity_index, correct) not significant -- trajectory consistency has no functional
    # consequence for task performance. Supported: r > 0.15 and p < 0.05.
    # A positive result is the STRONGEST functional claim in the framework: R predicts
    # real-world performance. Use logistic regression as robustness check if r < 0.15.
    sub_arith = df[run_mode_mask_any(df['run_mode'], [9, 10, 11, 12])] if not df.empty else pd.DataFrame()
    if (not sub_arith.empty and 'correct' in sub_arith.columns
            and 'state_similarity_index' in sub_arith.columns):
        paired31 = sub_arith[['state_similarity_index', 'correct']].dropna()
        if len(paired31) >= 20:
            from scipy.stats import pearsonr as _pr
            r31, p31 = _pr(paired31['state_similarity_index'], paired31['correct'])
            outcomes["H31"] = {
                "status": "supported" if (p31 < 0.05 and r31 > 0.15) else
                          ("inconclusive" if p31 >= 0.05 else "disproven"),
                "metric": "pearsonr_correct_similarity", "value": float(r31),
                "implication": (
                    f"similarity correlates with arithmetic accuracy (r={r31:.3f}, p={p31:.3f}) -- "
                    f"trajectory consistency has functional consequence for task performance."
                    if (p31 < 0.05 and r31 > 0.15)
                    else f"similarity not significantly correlated with accuracy "
                         f"(r={r31:.3f}, p={p31:.3f}) -- R may be epiphenomenal to performance."
                ),
            }
        else:
            outcomes["H31"] = {"status": "pending", "metric": "pearsonr_correct_similarity",
                               "value": float('nan'),
                               "implication": "Runs 0009-0012 insufficient data for accuracy-Similarity correlation."}
    else:
        outcomes["H31"] = {"status": "pending", "metric": "pearsonr_correct_similarity",
                           "value": float('nan'),
                           "implication": "Runs 0009-0012 not yet complete or 'correct' column absent."}

    # H32 -- shock phrasing has no differential effect on recovery arc (Run 0039)
    # Zero data cost: 'shock_variant' (0-3, cycling by trial%4) already saved per row
    # in Run 0039 (runners.py:826). Recovery delta = post-shock sim minus shock-turn sim.
    # Null: recovery_delta identical across 4 phrasings (ANOVA p >= 0.05).
    # Supported (ANOVA non-significant): recovery is phrasing-robust.
    # Disproven: one or more variants shows significantly different recovery.
    sub39 = df[run_mode_mask(df['run_mode'], 39)] if not df.empty else pd.DataFrame()
    if (not sub39.empty and 'shock_variant' in sub39.columns
            and 'layer_sim_mean' in sub39.columns and 'is_shock' in sub39.columns):
        # Compute per-trial recovery delta: mean(recovery turns) - mean(shock turn) per trial.
        # BUG-FOUND-1 (fixed v36.4): prior code pooled all recovery-turn observations
        # across all trials for a variant and subtracted the pooled shock mean, producing
        # arrays of ~75 values (3 recovery turns × ~25 trials) per variant instead of
        # ~25 per-trial scalars. The ANOVA then ran on ~300 pseudo-independent observations
        # with degrees of freedom (3, ~296) instead of the correct (3, ~96). Recovery turns
        # 14–16 within the same trial are correlated and cannot be treated as independent
        # observations; the pooled subtraction was also semantically incorrect (per-trial
        # shock value vs. pooled shock mean, not a matched pair).
        # Fix: group by trial within each variant, compute one scalar delta per trial
        # (mean recovery sim − mean shock sim within that trial), run ANOVA on those
        # ~25 genuinely independent per-trial deltas per variant.
        recovery_deltas = {}
        has_recovery_col = 'is_recovery' in sub39.columns
        for variant in [0, 1, 2, 3]:
            vsub = sub39[sub39['shock_variant'] == variant]
            if vsub.empty:
                continue
            trial_deltas = []
            for _, tgrp in vsub.groupby('trial'):
                shock_val = tgrp[tgrp['is_shock'] == 1]['layer_sim_mean'].dropna()
                rec_val   = (tgrp[tgrp['is_recovery'] == 1]['layer_sim_mean'].dropna()
                             if has_recovery_col else pd.Series(dtype=float))
                if len(shock_val) >= 1 and len(rec_val) >= 1:
                    trial_deltas.append(float(rec_val.mean() - shock_val.mean()))
            if len(trial_deltas) >= 3:
                recovery_deltas[variant] = np.array(trial_deltas)
        if len(recovery_deltas) >= 2:
            groups32 = list(recovery_deltas.values())
            _, p32 = sp_stats.f_oneway(*groups32)
            variant_means = {v: float(np.mean(d)) for v, d in recovery_deltas.items()}
            means_str = " ".join(f"V{v}={m:.3f}" for v, m in sorted(variant_means.items()))
            outcomes["H32"] = {
                "status": "supported" if p32 >= 0.05 else "disproven",
                "metric": "shock_variant_anova_p", "value": float(p32),
                "implication": (
                    f"Recovery delta does not differ by shock phrasing "
                    f"(ANOVA p={p32:.3f} >= 0.05; {means_str}) -- recovery is phrasing-robust."
                    if p32 >= 0.05
                    else f"Recovery delta differs by shock phrasing "
                         f"(ANOVA p={p32:.3f} < 0.05; {means_str}) -- recovery is phrasing-sensitive."
                ),
            }
        else:
            outcomes["H32"] = {"status": "pending", "metric": "shock_variant_anova_p",
                               "value": float('nan'),
                               "implication": "Run 0039 insufficient data across shock variants."}
    else:
        outcomes["H32"] = {"status": "pending", "metric": "shock_variant_anova_p",
                           "value": float('nan'),
                           "implication": "Run 0039 not yet complete or shock_variant column absent."}

    # ── Phase 4 inference blocks (v35.0) ─────────────────────────────────────

    # H33 -- coherence transfer measurement (Run 0025)
    # Compares total compute per correct arithmetic answer across three conditions:
    # condition_a (8×1-token priming turns), condition_b, condition_c.
    # Supported: compute_per_correct(condition_a) ≤ 40% of condition_c (p < 0.05)
    # AND correct_rate not significantly lower.
    sub43 = df[run_mode_mask(df['run_mode'], 43)] if not df.empty else pd.DataFrame()
    if (not sub43.empty and 'r_condition' in sub43.columns
            and 'peak_gpu_power' in sub43.columns
            and 'elapsed_sec' in sub43.columns):

        records43 = []
        for (cond, trial), grp in sub43.groupby(['r_condition', 'trial']):
            joules = float((grp['peak_gpu_power'] * grp['elapsed_sec']).sum())
            if 'phase' in sub43.columns:
                arith = grp[grp['phase'] == 'arithmetic']
            else:
                arith = pd.DataFrame()
            n_correct = int(arith['correct'].dropna().sum()) if (
                'correct' in arith.columns and not arith.empty) else 0
            cr = float(arith['correct'].dropna().mean()) if (
                'correct' in arith.columns and not arith.empty) else float('nan')
            jpc = joules / max(n_correct, 1)
            records43.append({'r_condition': cond, 'trial': trial,
                               'total_joules': joules, 'correct_rate': cr,
                               'compute_per_correct': jpc})

        if records43:
            agg43 = pd.DataFrame(records43)
            hi_jpc = agg43[agg43['r_condition'] == 'condition_a']['compute_per_correct'].dropna()
            lo_jpc = agg43[agg43['r_condition'] == 'condition_c']['compute_per_correct'].dropna()
            hi_cr  = agg43[agg43['r_condition'] == 'condition_a']['correct_rate'].dropna()
            lo_cr  = agg43[agg43['r_condition'] == 'condition_c']['correct_rate'].dropna()

            if len(hi_jpc) < 5 or len(lo_jpc) < 5:
                outcomes["H33"] = {"status": "pending", "metric": "compute_per_correct_ratio",
                                   "value": float('nan'),
                                   "implication": "Run 0025 insufficient trials."}
            else:
                _, p33_energy = sp_stats.ttest_ind(hi_jpc, lo_jpc)
                _, p33_quality = (sp_stats.ttest_ind(hi_cr.dropna(), lo_cr.dropna())
                                  if len(hi_cr) >= 5 and len(lo_cr) >= 5
                                  else (float('nan'), 1.0))
                ratio = float(hi_jpc.mean() / lo_jpc.mean()) if lo_jpc.mean() > 0 else float('nan')
                energy_advantage = (not np.isnan(ratio) and ratio <= 0.40 and p33_energy < 0.05)
                quality_preserved = (p33_quality >= 0.05 or np.isnan(p33_quality))

                if energy_advantage and quality_preserved:
                    h33_status = "supported"
                elif energy_advantage and not quality_preserved:
                    h33_status = "inconclusive"   # compute win but quality degraded
                elif not energy_advantage and p33_energy < 0.05:
                    h33_status = "inconclusive"   # significant but not ≤ 40%
                else:
                    h33_status = "disproven"

                outcomes["H33"] = {
                    "status": h33_status,
                    "metric": "compute_per_correct_ratio",
                    "value": ratio if not np.isnan(ratio) else float('nan'),
                    "compute_p": float(p33_energy),
                    "quality_p": float(p33_quality) if not np.isnan(p33_quality) else float('nan'),
                    "implication": (
                        f"Condition A uses {ratio*100:.1f}% of Condition C baseline compute per "
                        f"correct answer (p={p33_energy:.3f}); quality preserved (p={p33_quality:.3f}). "
                        f"Coherence transfer measurement confirmed."
                        if h33_status == "supported" else
                        f"Condition A compute ratio={ratio*100:.1f}% of Condition C "
                        f"(p_compute={p33_energy:.3f}, p_quality={p33_quality:.3f}) -- "
                        f"{'compute advantage but quality degraded' if energy_advantage else 'no 40% compute threshold met'}."
                    ),
                }
        else:
            outcomes["H33"] = {"status": "pending", "metric": "compute_per_correct_ratio",
                               "value": float('nan'),
                               "implication": "Run 0025 not yet complete."}
    else:
        outcomes["H33"] = {"status": "pending", "metric": "compute_per_correct_ratio",
                           "value": float('nan'),
                           "implication": "Run 0025 not yet complete."}

    # H34 -- disruption_magnitude continuous contradiction response (Run 0028, zero new data)
    # Tests whether disruption_magnitude (continuous) is significantly elevated at the contradiction
    # turn vs pre-contradiction baseline -- paired within-trial t-test.
    # H05 tests binary disruption_flag clustering; H34 tests the continuous magnitude signal.
    # Supported: mean disruption_magnitude at turn 7 significantly > mean(turns 1–6), p < 0.05, Δ > 0.01.
    sub28_h34 = df[run_mode_mask(df['run_mode'], 28)] if not df.empty else pd.DataFrame()
    if (not sub28_h34.empty and 'contradiction_turn' in sub28_h34.columns
            and 'disruption_magnitude' in sub28_h34.columns and 'trial' in sub28_h34.columns):

        paired34 = []
        for trial, grp in sub28_h34.groupby('trial'):
            # INF-H34-POSTCONTAM fix (v36.1): contradiction_turn==0 includes post turns (8–13).
            # Pre-baseline must be turns 1–6 only. Exclude post_contradiction==1 rows.
            if 'post_contradiction' in grp.columns:
                pre = grp[(grp['contradiction_turn'] == 0) &
                          (grp['post_contradiction'] == 0)]['disruption_magnitude'].dropna()
            else:
                pre = grp[grp['contradiction_turn'] == 0]['disruption_magnitude'].dropna()
            at_c = grp[grp['contradiction_turn'] == 1]['disruption_magnitude'].dropna()
            if len(pre) >= 3 and len(at_c) >= 1:
                paired34.append({
                    'pre_mean':  float(pre.mean()),
                    'at_mean':   float(at_c.mean()),
                    'delta':     float(at_c.mean() - pre.mean()),
                })

        if len(paired34) < 5:
            outcomes["H34"] = {"status": "pending", "metric": "disruption_magnitude_delta",
                               "value": float('nan'),
                               "implication": "Run 0028 insufficient trials for H34 disruption_magnitude test."}
        else:
            pdf34 = pd.DataFrame(paired34)
            _, p34 = sp_stats.ttest_1samp(pdf34['delta'], popmean=0)
            delta34 = float(pdf34['delta'].mean())
            outcomes["H34"] = {
                "status": "supported" if (p34 < 0.05 and delta34 > 0.01) else
                          ("disproven" if (p34 < 0.05 and delta34 <= 0) else "inconclusive"),
                "metric": "disruption_magnitude_delta",
                "value": delta34,
                "p_value": float(p34),
                "implication": (
                    f"disruption_magnitude significantly elevated at contradiction turn "
                    f"(Δ={delta34:.4f}, p={p34:.3f}) -- continuous geometric disruption "
                    f"signal confirmed."
                    if (p34 < 0.05 and delta34 > 0.01) else
                    f"disruption_magnitude not significantly elevated at contradiction (Δ={delta34:.4f}, "
                    f"p={p34:.3f}) -- continuous signal does not reliably detect disruption."
                ),
            }
    else:
        outcomes["H34"] = {"status": "pending", "metric": "disruption_magnitude_delta",
                           "value": float('nan'),
                           "implication": "Run 0028 not yet complete (H34 uses Run 0028 disruption_magnitude)."}

    # H35 -- contradiction recovery speed by R condition (Run 0026)
    # pre_sim = mean(state_similarity_index, turns 1–6) per trial
    # recovery_delta = mean(state_similarity_index, turns 8–13) − state_similarity_index(turn 7) per trial
    # Supported: Pearson r(pre_sim, recovery_delta) > 0.2, p < 0.05 (pooled)
    #            AND high_r recovery_delta > low_r recovery_delta (p < 0.05).
    sub44 = df[run_mode_mask(df['run_mode'], 44)] if not df.empty else pd.DataFrame()
    if (not sub44.empty and 'r_condition' in sub44.columns
            and 'state_similarity_index' in sub44.columns
            and 'contradiction_turn' in sub44.columns
            and 'post_contradiction' in sub44.columns
            and 'pre_contradiction' in sub44.columns):

        records44 = []
        for (cond, trial), grp in sub44.groupby(['r_condition', 'trial']):
            pre_sim  = grp[grp['pre_contradiction'] == 1]['state_similarity_index'].dropna()
            at_sim   = grp[grp['contradiction_turn'] == 1]['state_similarity_index'].dropna()
            post_sim = grp[grp['post_contradiction'] == 1]['state_similarity_index'].dropna()
            if len(pre_sim) >= 3 and len(at_sim) >= 1 and len(post_sim) >= 3:
                records44.append({
                    'r_condition':     cond,
                    'trial':           trial,
                    'pre_sim':         float(pre_sim.mean()),
                    'at_sim':          float(at_sim.mean()),
                    'post_sim':        float(post_sim.mean()),
                    'recovery_delta':  float(post_sim.mean() - at_sim.mean()),
                })

        if len(records44) < 10:
            outcomes["H35a"] = {"status": "pending", "metric": "recovery_similarity_delta_group",
                               "value": float('nan'),
                               "implication": "Run 0026 insufficient trials."}
            outcomes["H35b"] = {"status": "pending", "metric": "recovery_similarity_delta_corr",
                               "value": float('nan'),
                               "implication": "Run 0026 insufficient trials."}
        else:
            rdf44 = pd.DataFrame(records44)
            from scipy.stats import pearsonr
            r35, p35_corr = pearsonr(rdf44['pre_sim'], rdf44['recovery_delta'])
            hi_rd = rdf44[rdf44['r_condition'] == 'high_r']['recovery_delta'].dropna()
            lo_rd = rdf44[rdf44['r_condition'] == 'low_r']['recovery_delta'].dropna()
            # mid_r participates in the pooled Pearson r test above (correct -- tests
            # monotonic ordering across the full R-level range) but is excluded from
            # the group test below. The hypothesis spec calls for high_r vs low_r
            # as the directional group test; testing extremes is the pre-registered
            # design. mid_r exclusion from group test is intentional, not an
            # oversight. See dependency_map.py H35 notes for full rationale.
            p35_group = 1.0
            if len(hi_rd) >= 5 and len(lo_rd) >= 5:
                _, p35_group = sp_stats.ttest_ind(hi_rd, lo_rd)

            # H35a: group comparison (high_r vs low_r recovery_delta)
            group_supported = (len(hi_rd) >= 5 and len(lo_rd) >= 5
                               and float(hi_rd.mean()) > float(lo_rd.mean())
                               and p35_group < 0.05)
            outcomes["H35a"] = {
                "status":   "supported" if group_supported else
                            ("pending" if len(hi_rd) < 5 or len(lo_rd) < 5 else "disproven"),
                "metric":   "recovery_similarity_delta_group",
                "value":    float(hi_rd.mean() - lo_rd.mean()) if len(hi_rd) > 0 and len(lo_rd) > 0 else float('nan'),
                "p_group":  float(p35_group),
                "hi_mean":  float(hi_rd.mean()) if len(hi_rd) > 0 else float('nan'),
                "lo_mean":  float(lo_rd.mean()) if len(lo_rd) > 0 else float('nan'),
                "implication": (
                    f"H35a (group): high_r recovers faster than low_r "
                    f"(Δ={float(hi_rd.mean()-lo_rd.mean()):.4f}, p={p35_group:.3f}). "
                    f"Null rejected -- R level predicts recovery."
                    if group_supported else
                    f"H35a (group): high_r vs low_r recovery not significant "
                    f"(p={p35_group:.3f})."
                ),
            }

            # H35b: per-trial correlation (Pearson r between pre_sim and recovery_delta)
            corr_supported  = (r35 > 0.2 and p35_corr < 0.05)
            outcomes["H35b"] = {
                "status":   "supported" if corr_supported else "inconclusive",
                "metric":   "recovery_similarity_delta_corr",
                "value":    float(r35),
                "p_corr":   float(p35_corr),
                "implication": (
                    f"H35b (per-trial): pre-injection similarity predicts recovery speed "
                    f"(r={r35:.3f}, p={p35_corr:.3f})."
                    if corr_supported else
                    f"H35b (per-trial): correlation insufficient "
                    f"(r={r35:.3f}, p={p35_corr:.3f}). Requires more trials for power."
                ),
            }
    else:
        outcomes["H35a"] = {"status": "pending", "metric": "recovery_similarity_delta_group",
                           "value": float('nan'),
                           "implication": "Run 0026 not yet complete."}
        outcomes["H35b"] = {"status": "pending", "metric": "recovery_similarity_delta_corr",
                           "value": float('nan'),
                           "implication": "Run 0026 not yet complete."}

    # H39 -- impossibility cluster geometric effect (Runs 0029, 0030, 0031)
    # FIND-H09-GAP fix (v44.0.0): data existed but inference was never implemented.
    # Tests whether impossible-input prompts produce distinctive geometry vs null baseline.
    # Prediction: state_similarity_index differs significantly between [29,30,31] and [4,5,1].
    # Direction-agnostic: either elevation OR depression vs null counts as "supported"
    # (both would demonstrate that the impossibility constraint has geometric consequences).
    sub_impos = df[run_mode_mask_any(df['run_mode'], [29, 30, 31])] if not df.empty else pd.DataFrame()
    sub_null_ref = df[run_mode_mask_any(df['run_mode'], [4, 5, 1])] if not df.empty else pd.DataFrame()
    if (not sub_impos.empty and not sub_null_ref.empty
            and 'state_similarity_index' in df.columns):
        impos_sim = sub_impos['state_similarity_index'].dropna()
        null_sim  = sub_null_ref['state_similarity_index'].dropna()
        if len(impos_sim) >= 5 and len(null_sim) >= 5:
            diff_impos = float(impos_sim.mean() - null_sim.mean())
            _, p_impos = sp_stats.ttest_ind(impos_sim, null_sim)
            # Per-run breakdown for implication detail
            # v0.79.4.0: impossibility cluster renumbered. Old [12,13,14] → new [29,30,31].
            per_run_str = "; ".join(
                f"R{r:04d}: sim={df[df['run_mode']==str(r).zfill(4)]['state_similarity_index'].dropna().mean():.4f}"
                for r in [29, 30, 31]
                if str(r).zfill(4) in df['run_mode'].values
            )
            outcomes["H39"] = {
                "status": "supported" if p_impos < 0.05 else "inconclusive",
                "metric": "state_similarity_index",
                "value": float(diff_impos),
                "p_value": float(p_impos),
                "null_mean": float(null_sim.mean()),
                "impos_mean": float(impos_sim.mean()),
                "implication": (
                    f"Impossibility prompts produce distinctive geometry vs null "
                    f"(Δ={diff_impos:.4f}, p={p_impos:.3f}). "
                    f"{'Elevation' if diff_impos > 0 else 'Depression'} of similarity confirmed. "
                    f"Per-run: {per_run_str}."
                    if p_impos < 0.05 else
                    f"Impossibility geometry not significantly different from null "
                    f"(Δ={diff_impos:.4f}, p={p_impos:.3f}). "
                    f"Per-run: {per_run_str}."
                ),
            }
        else:
            outcomes["H39"] = {"status": "pending", "metric": "state_similarity_index",
                                   "value": float('nan'),
                                   "implication": "Runs 0029-0031 insufficient data."}
    else:
        outcomes["H39"] = {"status": "pending", "metric": "state_similarity_index",
                               "value": float('nan'),
                               "implication": "Runs 0029-0031 not yet complete."}

    # H40 -- random noise patching control (Run 0019 vs Run 0017)
    # Two-proportion z-test: real geometry patches (H38/Run 0017) should produce
    # higher output_change_rate than random noise patches (Run 0019).
    # If rates are indistinguishable, the causal claim is confounded.
    # NOTE: Run 0017 uses 'patch_mode' column (partial/full). Run 0019 uses
    # 'patch_layer' column (full/L8/L16/L24/L31). Both exclude 'none'.
    sub19 = df[run_mode_mask(df['run_mode'], 19)] if not df.empty else pd.DataFrame()
    sub17 = df[run_mode_mask(df['run_mode'], 17)] if not df.empty else pd.DataFrame()
    _has19 = (not sub19.empty and 'output_changed' in sub19.columns
              and 'patch_layer' in sub19.columns)
    _has17 = (not sub17.empty and 'output_changed' in sub17.columns
              and 'patch_mode' in sub17.columns)
    if _has19 and _has17:
        # Run 0017: partial + full modes (real geometry)
        real_oc = sub17[sub17['patch_mode'].isin(['partial', 'full'])]['output_changed'].dropna()
        # Run 0019: all non-none modes (random noise at various layers)
        noise_oc = sub19[sub19['patch_layer'] != 'none']['output_changed'].dropna()
        if len(real_oc) >= 20 and len(noise_oc) >= 20:
            n_real, k_real = len(real_oc), int(real_oc.sum())
            n_noise, k_noise = len(noise_oc), int(noise_oc.sum())
            rate_real = k_real / n_real
            rate_noise = k_noise / n_noise
            # Two-proportion z-test (one-sided: real > noise)
            p_pool = (k_real + k_noise) / (n_real + n_noise)
            if p_pool > 0 and p_pool < 1:
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n_real + 1/n_noise))
                z_stat = (rate_real - rate_noise) / se if se > 0 else 0
                p_h40 = float(1 - sp_stats.norm.cdf(z_stat))
            else:
                z_stat = 0.0
                p_h40 = 1.0
            if rate_real > rate_noise and p_h40 < 0.05:
                h40_status = "supported"
                h40_impl = (
                    f"Real patching rate ({rate_real*100:.1f}%) significantly exceeds "
                    f"noise patching rate ({rate_noise*100:.1f}%), z={z_stat:.2f}, p={p_h40:.4f}. "
                    f"Causal effect is geometry-specific, not a generic perturbation artifact."
                )
            elif p_h40 >= 0.05:
                h40_status = "disproven"
                h40_impl = (
                    f"Real patching rate ({rate_real*100:.1f}%) not significantly different from "
                    f"noise ({rate_noise*100:.1f}%), z={z_stat:.2f}, p={p_h40:.4f}. "
                    f"Causal claim confounded -- any perturbation produces similar effect."
                )
            else:
                h40_status = "inconclusive"
                h40_impl = (
                    f"Real rate={rate_real*100:.1f}%, noise rate={rate_noise*100:.1f}%, "
                    f"z={z_stat:.2f}, p={p_h40:.4f}."
                )
            outcomes["H40"] = {
                "status": h40_status,
                "metric": "output_change_rate_noise",
                "value": float(rate_noise * 100),
                "rate_real_pct": float(rate_real * 100),
                "rate_noise_pct": float(rate_noise * 100),
                "z_stat": float(z_stat),
                "p_value": p_h40,
                "n_real": n_real, "n_noise": n_noise,
                "implication": h40_impl,
            }
        else:
            outcomes["H40"] = {"status": "pending", "metric": "output_change_rate_noise",
                               "value": float('nan'),
                               "implication": f"Insufficient data: Run 0017 has {len(real_oc)} patched rows, Run 0019 has {len(noise_oc)}."}
    else:
        outcomes["H40"] = {"status": "pending", "metric": "output_change_rate_noise",
                           "value": float('nan'),
                           "implication": "Run 0019 (random noise patching) not yet complete."}

    # ── v0.77.0.0 -- Planned hypothesis inference blocks ─────────────────────────

    # H41 -- System Prompt Has No Geometric Effect (Run 0002)
    try:
        sub20 = df[df['run_mode'].isin(['0002', '20'])]
        if len(sub20) >= 20 and "condition" in sub20.columns:
            sims = pd.to_numeric(sub20["state_similarity_index"], errors="coerce").dropna()
            conds = sub20.loc[sims.index, "condition"]
            groups = [sims[conds == c].values for c in conds.unique() if (conds == c).sum() >= 5]
            if len(groups) >= 2:
                f_stat, p_val = sp_stats.f_oneway(*groups)
                status = "supported" if p_val >= 0.05 else "disproven"
                outcomes["H41"] = {"status": status, "metric": "state_similarity_index",
                                   "value": float(p_val),
                                   "implication": f"System prompt condition ANOVA F={f_stat:.3f}, p={p_val:.3f}. "
                                                  f"{'No effect detected.' if status == 'supported' else 'Significant effect -- disconfirms null.'}"}
            else:
                outcomes["H41"] = {"status": "pending", "metric": "state_similarity_index",
                                   "value": float('nan'),
                                   "implication": "Run 0002 insufficient condition coverage for ANOVA."}
        else:
            outcomes["H41"] = {"status": "pending", "metric": "state_similarity_index",
                               "value": float('nan'),
                               "implication": "Run 0002 not yet complete."}
    except Exception as _e:
        outcomes["H41"] = {"status": "pending", "metric": "state_similarity_index",
                           "value": float('nan'), "implication": f"H41 inference error: {_e}"}

    # H42 -- Similarity Does Not Accumulate With Turn Count (Run 0001 null baseline)
    try:
        sub19 = df[df['run_mode'].isin(['0001', '19'])]
        if len(sub19) >= 30 and "turn" in sub19.columns:
            sims = pd.to_numeric(sub19["state_similarity_index"], errors="coerce")
            turns = pd.to_numeric(sub19["turn"], errors="coerce")
            mask = sims.notna() & turns.notna()
            if mask.sum() >= 30:
                slope, intercept, r_val, p_val, stderr = sp_stats.linregress(
                    turns[mask].values, sims[mask].values)
                # Null hypothesis: no accumulation (slope ≈ 0). Supported = p ≥ 0.05.
                status = "supported" if p_val >= 0.05 else "disproven"
                outcomes["H42"] = {"status": status, "metric": "state_similarity_index",
                                   "value": float(slope),
                                   "slope_p": float(p_val),
                                   "implication": f"Null-baseline similarity-vs-turn slope={slope:.5f}, p={p_val:.3f}. "
                                                  f"{'No accumulation.' if status == 'supported' else 'Accumulation detected -- disconfirms null.'}"}
            else:
                outcomes["H42"] = {"status": "pending", "metric": "state_similarity_index",
                                   "value": float('nan'),
                                   "implication": "Run 0001 insufficient turn coverage."}
        else:
            outcomes["H42"] = {"status": "pending", "metric": "state_similarity_index",
                               "value": float('nan'),
                               "implication": "Run 0001 not yet complete."}
    except Exception as _e:
        outcomes["H42"] = {"status": "pending", "metric": "state_similarity_index",
                           "value": float('nan'), "implication": f"H42 inference error: {_e}"}

    # H43 -- R Fraction Is Sensitive To Projection Dimension (Run 0041 POOL_DIM sweep)
    try:
        q41_path = os.path.join(analysis_dir, "Q0041_pool_dim_sweep.json")
        if os.path.exists(q41_path):
            with open(q41_path) as _f: q41 = json.load(_f)
            sweep = q41.get("sweep", [])
            r_vals = [e.get("frac_R") for e in sweep if e.get("frac_R") is not None]
            if len(r_vals) >= 3:
                max_adj_delta = max(abs(r_vals[i+1] - r_vals[i]) for i in range(len(r_vals)-1))
                # Null: R fraction is sensitive (varies > 0.05 between adjacent dims) -- supported means drift detected
                # Direction "negative" in schema means we want NULL supported = partition IS sensitive (= bad)
                # Paper-relevant: we want R to be dimension-STABLE. So null supported (high variance) = bad finding.
                status = "supported" if max_adj_delta > 0.05 else "disproven"
                outcomes["H43"] = {"status": status, "metric": "frac_R",
                                   "value": float(max_adj_delta),
                                   "implication": f"Max R-fraction delta between adjacent POOL_DIMs = {max_adj_delta:.4f}. "
                                                  f"{'Dimension-sensitive -- estimator unstable.' if status == 'supported' else 'Dimension-stable -- estimator robust.'}"}
            else:
                outcomes["H43"] = {"status": "pending", "metric": "frac_R",
                                   "value": float('nan'),
                                   "implication": f"Run 0041 has only {len(r_vals)} sweep points; need >=3."}
        else:
            outcomes["H43"] = {"status": "pending", "metric": "frac_R",
                               "value": float('nan'),
                               "implication": "Run 0041 (POOL_DIM sweep) not yet complete."}
    except Exception as _e:
        outcomes["H43"] = {"status": "pending", "metric": "frac_R",
                           "value": float('nan'), "implication": f"H43 inference error: {_e}"}

    # H44 -- R Fraction Is Condition-Invariant (Run 0044 per-condition R)
    try:
        q44_path = os.path.join(analysis_dir, "Q0044_per_condition_R.json")
        if os.path.exists(q44_path):
            with open(q44_path) as _f: q44 = json.load(_f)
            per_run = q44.get("per_run", [])
            valid = [e for e in per_run if not e.get("skipped") and e.get("frac_R") is not None]
            if len(valid) >= 3:
                r_vals = [e["frac_R"] for e in valid]
                r_range = max(r_vals) - min(r_vals)
                # Null: R is invariant across conditions. Supported = low range (≤ 0.10).
                status = "supported" if r_range <= 0.10 else "disproven"
                outcomes["H44"] = {"status": status, "metric": "frac_R",
                                   "value": float(r_range),
                                   "implication": f"R-fraction range across {len(valid)} conditions = {r_range:.4f}. "
                                                  f"{'Condition-invariant.' if status == 'supported' else 'Condition-dependent -- disconfirms null.'}"}
            else:
                outcomes["H44"] = {"status": "pending", "metric": "frac_R",
                                   "value": float('nan'),
                                   "implication": f"Run 0044 has only {len(valid)} valid conditions; need >=3."}
        else:
            outcomes["H44"] = {"status": "pending", "metric": "frac_R",
                               "value": float('nan'),
                               "implication": "Run 0044 (per-condition R) not yet complete."}
    except Exception as _e:
        outcomes["H44"] = {"status": "pending", "metric": "frac_R",
                           "value": float('nan'), "implication": f"H44 inference error: {_e}"}

    # H45 -- Resistant Prime Does Not Damage Trajectory More Than Cooperative (Run 0050 §1)
    try:
        q50_path = os.path.join(analysis_dir, "Q0050_cross_temp_synthesis.json")
        if os.path.exists(q50_path):
            with open(q50_path) as _f: q50 = json.load(_f)
            pv = q50.get("priming_vulnerability", [])
            if pv and len(pv) > 0:
                entry = pv[0] if isinstance(pv, list) else pv
                resistant_rate = entry.get("resistant_disruption_rate")
                cooperative_rate = entry.get("cooperative_disruption_rate")
                if resistant_rate is not None and cooperative_rate is not None:
                    delta = resistant_rate - cooperative_rate
                    # Null: resistant rate NOT significantly > cooperative rate. Supported = delta ≤ 0.05.
                    status = "supported" if delta <= 0.05 else "disproven"
                    outcomes["H45"] = {"status": status, "metric": "disruption_flag",
                                       "value": float(delta),
                                       "implication": f"Resistant-cooperative disruption-rate delta = {delta:.4f}. "
                                                      f"{'Prime type does not matter.' if status == 'supported' else 'Resistant primes more disruptive -- disconfirms null.'}"}
                else:
                    outcomes["H45"] = {"status": "pending", "metric": "disruption_flag",
                                       "value": float('nan'),
                                       "implication": "Run 0050 priming data missing resistant/cooperative rates."}
            else:
                outcomes["H45"] = {"status": "pending", "metric": "disruption_flag",
                                   "value": float('nan'),
                                   "implication": "Run 0050 priming_vulnerability section not populated."}
        else:
            outcomes["H45"] = {"status": "pending", "metric": "disruption_flag",
                               "value": float('nan'),
                               "implication": "Run 0050 (cross-temp synthesis) not yet complete."}
    except Exception as _e:
        outcomes["H45"] = {"status": "pending", "metric": "disruption_flag",
                           "value": float('nan'), "implication": f"H45 inference error: {_e}"}

    # H46 -- Contradiction Does Not Drop Similarity (Run 0050 §2)
    try:
        q50_path = os.path.join(analysis_dir, "Q0050_cross_temp_synthesis.json")
        if os.path.exists(q50_path):
            with open(q50_path) as _f: q50 = json.load(_f)
            ca = q50.get("contradiction_analysis", [])
            if ca and len(ca) > 0:
                entry = ca[0] if isinstance(ca, list) else ca
                drop = entry.get("similarity_drop")
                if drop is not None:
                    # Null: no drop. Supported = drop ≤ 0.02 (small/noise-level).
                    status = "supported" if abs(drop) <= 0.02 else "disproven"
                    outcomes["H46"] = {"status": status, "metric": "state_similarity_index",
                                       "value": float(drop),
                                       "implication": f"Similarity drop at contradiction = {drop:.4f}. "
                                                      f"{'No meaningful drop.' if status == 'supported' else 'Contradiction drops similarity -- disconfirms null.'}"}
                else:
                    outcomes["H46"] = {"status": "pending", "metric": "state_similarity_index",
                                       "value": float('nan'),
                                       "implication": "Run 0050 contradiction_analysis missing similarity_drop."}
            else:
                outcomes["H46"] = {"status": "pending", "metric": "state_similarity_index",
                                   "value": float('nan'),
                                   "implication": "Run 0050 contradiction_analysis not populated."}
        else:
            outcomes["H46"] = {"status": "pending", "metric": "state_similarity_index",
                               "value": float('nan'),
                               "implication": "Run 0050 not yet complete."}
    except Exception as _e:
        outcomes["H46"] = {"status": "pending", "metric": "state_similarity_index",
                           "value": float('nan'), "implication": f"H46 inference error: {_e}"}

    # H47 -- Coherence Transfer Is Not Compute-Efficient (Run 0050 §6, Appendix A scope)
    try:
        q50_path = os.path.join(analysis_dir, "Q0050_cross_temp_synthesis.json")
        if os.path.exists(q50_path):
            with open(q50_path) as _f: q50 = json.load(_f)
            ce = q50.get("compute_efficiency", [])
            if ce and len(ce) > 0:
                entry = ce[0] if isinstance(ce, list) else ce
                ratio = entry.get("compute_per_correct_ratio")
                if ratio is not None:
                    # Null: primed NOT more efficient. Supported = ratio ≥ 1.0.
                    status = "supported" if ratio >= 1.0 else "disproven"
                    outcomes["H47"] = {"status": status, "metric": "compute_per_correct_ratio",
                                       "value": float(ratio),
                                       "implication": f"compute_per_correct ratio (primed/unprimed) = {ratio:.3f}. "
                                                      f"{'No compute advantage.' if status == 'supported' else 'Primed more efficient -- disconfirms null.'} "
                                                      f"(Appendix A -- out of scope for measurement paper.)"}
                else:
                    outcomes["H47"] = {"status": "pending", "metric": "compute_per_correct_ratio",
                                       "value": float('nan'),
                                       "implication": "Run 0050 compute_efficiency missing ratio."}
            else:
                outcomes["H47"] = {"status": "pending", "metric": "compute_per_correct_ratio",
                                   "value": float('nan'),
                                   "implication": "Run 0050 compute_efficiency not populated."}
        else:
            outcomes["H47"] = {"status": "pending", "metric": "compute_per_correct_ratio",
                               "value": float('nan'),
                               "implication": "Run 0050 not yet complete."}
    except Exception as _e:
        outcomes["H47"] = {"status": "pending", "metric": "compute_per_correct_ratio",
                           "value": float('nan'), "implication": f"H47 inference error: {_e}"}

    # H48 -- R Decay Is Explained By KV Cache Growth (Run 0050 §7 persistence modes)
    try:
        q50_path = os.path.join(analysis_dir, "Q0050_cross_temp_synthesis.json")
        if os.path.exists(q50_path):
            with open(q50_path) as _f: q50 = json.load(_f)
            pm = q50.get("persistence_modes", {})
            if pm:
                decays = {mode: pm[mode].get("similarity_decay") for mode in pm
                          if pm[mode].get("similarity_decay") is not None}
                if len(decays) >= 3:
                    decay_range = max(decays.values()) - min(decays.values())
                    # Null: all modes identical decay. Supported = range ≤ 0.02.
                    status = "supported" if decay_range <= 0.02 else "disproven"
                    outcomes["H48"] = {"status": status, "metric": "state_similarity_index",
                                       "value": float(decay_range),
                                       "implication": f"Decay range across persistence modes = {decay_range:.4f}. "
                                                      f"{'Cache growth explains decay.' if status == 'supported' else 'Mode-dependent decay -- cache is not the confound.'}"}
                else:
                    outcomes["H48"] = {"status": "pending", "metric": "state_similarity_index",
                                       "value": float('nan'),
                                       "implication": f"Run 0050 persistence data has {len(decays)} modes; need >=3."}
            else:
                outcomes["H48"] = {"status": "pending", "metric": "state_similarity_index",
                                   "value": float('nan'),
                                   "implication": "Run 0050 persistence_modes not populated."}
        else:
            outcomes["H48"] = {"status": "pending", "metric": "state_similarity_index",
                               "value": float('nan'),
                               "implication": "Run 0050 not yet complete."}
    except Exception as _e:
        outcomes["H48"] = {"status": "pending", "metric": "state_similarity_index",
                           "value": float('nan'), "implication": f"H48 inference error: {_e}"}

    # H49 -- Introspection R Does Not Transfer To Arithmetic (Run 0050 §8)
    try:
        q50_path = os.path.join(analysis_dir, "Q0050_cross_temp_synthesis.json")
        if os.path.exists(q50_path):
            with open(q50_path) as _f: q50 = json.load(_f)
            ct = q50.get("condition_transfer", [])
            if ct and len(ct) > 0:
                entry = ct[0] if isinstance(ct, list) else ct
                conds = entry.get("conditions", {})
                transfer_arm = conds.get("introspection_to_arithmetic", {})
                arith_only = conds.get("arithmetic_only", {})
                if transfer_arm and arith_only:
                    t_post = transfer_arm.get("post_switch_similarity")
                    a_post = arith_only.get("post_switch_similarity")
                    if t_post is not None and a_post is not None:
                        delta = t_post - a_post
                        # Null: no transfer. Supported = delta ≤ 0.02.
                        status = "supported" if delta <= 0.02 else "disproven"
                        outcomes["H49"] = {"status": status, "metric": "state_similarity_index",
                                           "value": float(delta),
                                           "implication": f"Post-switch similarity delta (transfer vs arithmetic-only) = {delta:.4f}. "
                                                          f"{'R does not transfer.' if status == 'supported' else 'R transfers -- disconfirms null.'}"}
                    else:
                        outcomes["H49"] = {"status": "pending", "metric": "state_similarity_index",
                                           "value": float('nan'),
                                           "implication": "Run 0050 condition_transfer missing post_switch values."}
                else:
                    outcomes["H49"] = {"status": "pending", "metric": "state_similarity_index",
                                       "value": float('nan'),
                                       "implication": "Run 0050 condition_transfer arms missing."}
            else:
                outcomes["H49"] = {"status": "pending", "metric": "state_similarity_index",
                                   "value": float('nan'),
                                   "implication": "Run 0050 condition_transfer not populated."}
        else:
            outcomes["H49"] = {"status": "pending", "metric": "state_similarity_index",
                               "value": float('nan'),
                               "implication": "Run 0050 not yet complete."}
    except Exception as _e:
        outcomes["H49"] = {"status": "pending", "metric": "state_similarity_index",
                           "value": float('nan'), "implication": f"H49 inference error: {_e}"}

    # H52 -- Disruption Magnitude Is Stationary Across Non-Contradiction Turns (Run 0028 baseline)
    try:
        sub22 = df[(df['run_mode'].isin(['0028', '22'])) & (df["turn"].astype(str).isin(["1","2","3","4","5","6"]))]
        if len(sub22) >= 60 and "disruption_magnitude" in sub22.columns and "trial" in sub22.columns:
            dm = pd.to_numeric(sub22["disruption_magnitude"], errors="coerce")
            trials = sub22["trial"]
            turns22 = pd.to_numeric(sub22["turn"], errors="coerce")
            mask = dm.notna() & turns22.notna()
            # Per-trial stationarity check
            from statsmodels.tsa.stattools import adfuller, acf
            n_trials = 0
            n_stationary = 0
            all_vals = []
            for tr in trials[mask].unique():
                series = dm[(trials == tr) & mask].values
                if len(series) < 6: continue
                n_trials += 1
                all_vals.extend(series)
                try:
                    adf_p = adfuller(series, maxlag=1, autolag=None)[1]
                    if adf_p < 0.05: n_stationary += 1
                except Exception:
                    pass
            if n_trials >= 20:
                frac_stationary = n_stationary / n_trials
                # Pooled autocorr at lag 1
                try:
                    pooled_acf = acf(all_vals, nlags=1, fft=False)[1]
                except Exception:
                    pooled_acf = float('nan')
                # Supported: ≥ 0.95 stationary AND |pooled_acf| < 0.2
                stat_ok = frac_stationary >= 0.95
                acf_ok = abs(pooled_acf) < 0.2 if pooled_acf == pooled_acf else False
                status = "supported" if (stat_ok and acf_ok) else "disproven"
                outcomes["H52"] = {"status": status, "metric": "disruption_magnitude_stationarity",
                                   "value": float(frac_stationary),
                                   "pooled_acf_lag1": float(pooled_acf),
                                   "implication": f"Stationarity: {n_stationary}/{n_trials} trials reject unit root ({frac_stationary*100:.1f}%). "
                                                  f"Pooled lag-1 ACF = {pooled_acf:.3f}. "
                                                  f"{'H34 baseline is valid.' if status == 'supported' else 'H34 baseline contaminated -- disruption magnitude drifts or autocorrelates.'}"}
            else:
                outcomes["H52"] = {"status": "pending", "metric": "disruption_magnitude_stationarity",
                                   "value": float('nan'),
                                   "implication": f"Run 0028 has only {n_trials} testable trials; need >=20."}
        else:
            outcomes["H52"] = {"status": "pending", "metric": "disruption_magnitude_stationarity",
                               "value": float('nan'),
                               "implication": "Run 0028 not yet complete or missing disruption_magnitude column."}
    except Exception as _e:
        outcomes["H52"] = {"status": "pending", "metric": "disruption_magnitude_stationarity",
                           "value": float('nan'), "implication": f"H52 inference error: {_e}"}

    # H53 -- Cross-Stochasticity Disruption Rate Is Monotonic (Runs 22 + 35 across temps)
    try:
        # Uses current DataFrame (single-temp) so this test runs once per temp and needs
        # cross-temp aggregation -- defer to a pooled analyzer, but emit a per-temp diagnostic.
        sub22_35 = df[df["run_mode"].isin(["22", "35"])]
        if len(sub22_35) >= 50 and "disruption_flag" in sub22_35.columns:
            # At a single temperature we can only report the disruption rate; monotonicity
            # across temperatures is a cross-temp analysis. Emit pending with the per-temp rate.
            rate = pd.to_numeric(sub22_35["disruption_flag"], errors="coerce").mean()
            outcomes["H53"] = {"status": "pending", "metric": "disruption_flag_temp_monotonicity",
                               "value": float(rate) if rate == rate else float('nan'),
                               "implication": f"Per-temp disruption_flag rate = {rate:.3f}. "
                                              f"Full monotonicity test requires cross-temp pooled analysis."}
        else:
            outcomes["H53"] = {"status": "pending", "metric": "disruption_flag_temp_monotonicity",
                               "value": float('nan'),
                               "implication": "Runs 0028/0038 not yet complete."}
    except Exception as _e:
        outcomes["H53"] = {"status": "pending", "metric": "disruption_flag_temp_monotonicity",
                           "value": float('nan'), "implication": f"H53 inference error: {_e}"}

    # H58 -- E + C + R Fractions Sum To 1.0 Within Tolerance (Run 0043 sanity)
    try:
        q34_path = os.path.join(analysis_dir, "Q0043_sobol_partition.json")
        if os.path.exists(q34_path):
            with open(q34_path) as _f: q34 = json.load(_f)
            e_val = q34.get("perm_sens_E", {}).get("fraction")
            c_val = q34.get("perm_sens_C", {}).get("fraction")
            r_val = q34.get("perm_sens_R", {}).get("fraction")
            if all(v is not None for v in (e_val, c_val, r_val)):
                total = e_val + c_val + r_val
                deviation = abs(total - 1.0)
                # Supported: deviation ≤ 0.01
                status = "supported" if deviation <= 0.01 else "disproven"
                outcomes["H58"] = {"status": status, "metric": "ecr_sum_deviation",
                                   "value": float(deviation),
                                   "sum_E_C_R": float(total),
                                   "implication": f"E+C+R sum = {total:.4f} (deviation {deviation:.4f}). "
                                                  f"{'Estimator respects E+C+R=1 identity.' if status == 'supported' else 'Estimator deviates from identity -- investigate attribution bug.'}"}
            else:
                outcomes["H58"] = {"status": "pending", "metric": "ecr_sum_deviation",
                                   "value": float('nan'),
                                   "implication": "Q34 fractions incomplete."}
        else:
            outcomes["H58"] = {"status": "pending", "metric": "ecr_sum_deviation",
                               "value": float('nan'),
                               "implication": "Run 0043 (Sobol partition) not yet complete."}
    except Exception as _e:
        outcomes["H58"] = {"status": "pending", "metric": "ecr_sum_deviation",
                           "value": float('nan'), "implication": f"H58 inference error: {_e}"}

    for h_id in HYPOTHESES:
        if h_id not in outcomes:
            outcomes[h_id] = {"status": "pending", "metric": HYPOTHESES[h_id][2],
                              "value": float('nan'), "implication": "Not yet analyzed."}

    return outcomes


# ══════════════════════════════════════════════════════════════════════
#  Methodology calibration orchestration (v0.80.0.33)
# ══════════════════════════════════════════════════════════════════════
#
# Run 0056 triggers four calibration scripts that live at repo root:
#   - v5_synthetic_calibration.py
#   - ridge_bias_toy.py
#   - toy_nonlinearity_asymmetry.py
#   - run_channel_marginal.py
#
# Outputs land in data/paper/calibration/{name}/. Each has a .cache_key.json
# manifest containing hashes of its source script + upstream .py files the
# calibration depends on. Stale cache = auto-backup to _bak/{timestamp}/ then
# regenerate. User can manually restore from _bak via restore_calibration_bak.py
# if they decide the upstream change didn't affect calibration.
#
# Run 0056 does NOT automatically trigger calibration -- user must run scripts
# manually. 0056 gates on artifact presence; dashboard warns if missing.
# Per the "just skip calibration but paper grid stays orange" spec.

_CALIBRATION_SCRIPTS = {
    'v5':                 ('v5_synthetic_calibration.py', 'v5_calibration_results.json'),
    'ridge_bias':         ('ridge_bias_toy.py',           'results.csv'),
    'toy_nonlinearity':   ('toy_nonlinearity_asymmetry.py', 'results.csv'),
    'channel_marginal':   ('run_channel_marginal.py',     'channel_marginal_nonlinearity.csv'),
    # v0.81.1.6 ship 4: paper 2 measurement-apparatus foundations.
    # split_pca_selection picks the operating-point (sample-split, PCA dim)
    # for the apparatus from V5e + V5f calibration. knn_mi_reliability
    # measures kNN-MI estimator reliability under bootstrap resampling at
    # that operating point. Both feed the apparatus's pre-fire
    # configuration; both are foundations layer (not phase outputs).
    'split_pca_selection': ('run_split_pca_selection.py',  'selection.json'),
    'knn_mi_reliability':  ('run_knn_mi_reliability.py',   'reliability.json'),
    # v0.81.0.5: kraskov_spike removed from foundations registry. It's
    # Phase 1 of the lagrangian apparatus (Run 0058), not a foundation
    # the apparatus depends on. Its output lives at
    # data/paper/calibration/kraskov_spike/spike_result.json and is
    # generated by run_bayesian_apparatus.py invoking
    # run_kraskov_spike.main(). Apparatus phases have their own
    # internal phase tracking via Q0058_apparatus_manifest.json.
}

# Upstream files whose change invalidates calibration outputs.
# Source .py of the script itself is always included implicitly.
_CALIBRATION_UPSTREAMS = {
    'v5':                 [],                          # purely synthetic, no upstream
    'ridge_bias':         [],                          # purely synthetic
    'toy_nonlinearity':   ['analysis.py'],             # MLP config mirrors linearity_check
    'channel_marginal':   ['analysis.py', 'cartography.py'],  # uses _load_quadruplets + run_mode helpers
    # v0.81.1.6: upstream chain -- selection feeds reliability
    'split_pca_selection': ['v5_synthetic_calibration.py'],
    'knn_mi_reliability':  ['v5_synthetic_calibration.py',
                             'run_split_pca_selection.py'],
}


def _calibration_dir(name):
    # v0.80.0.33: lazy import -- DATA is defined in cartography.py and was
    # being referenced here as a bare name, which worked for callers that
    # had already imported it but broke direct invocation from scripts or
    # python -c (as caught by stamp_manifests workflow).
    from cartography import DATA as _DATA
    return os.path.join(_DATA, 'paper', 'calibration', name)


def _hash_file(path):
    import hashlib
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _cache_key_path(name):
    return os.path.join(_calibration_dir(name), '.cache_key.json')


def _compute_cache_key(name):
    """Returns the expected cache key for a calibration name -- hashes of
    its source script plus upstream files it depends on."""
    script_name, _ = _CALIBRATION_SCRIPTS[name]
    repo_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(repo_root, script_name)
    key = {
        'script':    script_name,
        'script_hash': _hash_file(script_path),
        'upstream':  {},
    }
    for up in _CALIBRATION_UPSTREAMS[name]:
        key['upstream'][up] = _hash_file(os.path.join(repo_root, up))
    return key


def _load_cache_key(name):
    path = _cache_key_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        return None


def _calibration_status(name):
    """Returns one of:
       'missing'  -- no artifact on disk
       'stale'    -- artifact present but hashes don't match current
       'fresh'    -- artifact present and hashes match
    """
    _, artifact = _CALIBRATION_SCRIPTS[name]
    artifact_path = os.path.join(_calibration_dir(name), artifact)
    if not os.path.exists(artifact_path):
        return 'missing'
    stored = _load_cache_key(name)
    if stored is None:
        return 'stale'
    current = _compute_cache_key(name)
    if stored.get('script_hash') != current.get('script_hash'):
        return 'stale'
    for up, h in current.get('upstream', {}).items():
        if stored.get('upstream', {}).get(up) != h:
            return 'stale'
    return 'fresh'


def _backup_calibration(name):
    """Move current calibration outputs to _bak/{timestamp}/ before
    regeneration. Never auto-deletes baks."""
    import datetime
    import shutil
    cal_dir = _calibration_dir(name)
    if not os.path.isdir(cal_dir):
        return None
    bak_root = os.path.join(cal_dir, '_bak')
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    bak_dir = os.path.join(bak_root, stamp)
    os.makedirs(bak_dir, exist_ok=True)
    for item in os.listdir(cal_dir):
        if item == '_bak':
            continue
        src = os.path.join(cal_dir, item)
        dst = os.path.join(bak_dir, item)
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                shutil.rmtree(src)
            else:
                shutil.move(src, dst)
        except Exception:
            pass
    return bak_dir


def methodology_calibration_status():
    """Summary dict usable by scanner, dashboard, and the master results writer.
    Called from Run 0056 completion checks and from the dashboard.

    v0.80.0.33: uses _calibration_status_ex which distinguishes
    'no_manifest' (existing artifact, first-time bootstrap pending) from
    'stale' (existing artifact, hash mismatch = upstream changed).

    `_any_missing` and `_any_stale` are the gate signals -- `no_manifest`
    does NOT trigger them, because 0056's orchestrator will bootstrap the
    manifest in place when it runs. Scanner is read-only; bootstrap is
    the orchestrator's job."""
    out = {}
    for name in _CALIBRATION_SCRIPTS:
        out[name] = _calibration_status_ex(name)
    out['_all_fresh']     = all(v == 'fresh' for k, v in out.items() if not k.startswith('_'))
    out['_any_missing']   = any(v == 'missing' for k, v in out.items() if not k.startswith('_'))
    out['_any_stale']     = any(v == 'stale' for k, v in out.items() if not k.startswith('_'))
    out['_any_no_manifest'] = any(v == 'no_manifest' for k, v in out.items() if not k.startswith('_'))
    return out


def _load_methodology_calibration():
    """Reads calibration artifacts into a dict suitable for the master
    results JSON. Includes status summary. Never throws; missing
    artifacts appear as null entries so downstream code can tell the
    difference between 'not yet computed' and 'computed to null'.

    v0.81.1.6: registry grows to 6 foundations -- added split_pca_selection
    and knn_mi_reliability for the paper 2 measurement apparatus."""
    status = methodology_calibration_status()
    block = {
        'status': status,
        'v5':                  None,
        'ridge_bias':          None,
        'toy_nonlinearity':    None,
        'channel_marginal':    None,
        'split_pca_selection': None,
        'knn_mi_reliability':  None,
    }
    # v5: JSON
    p = os.path.join(_calibration_dir('v5'), 'v5_calibration_results.json')
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8-sig') as f:
                block['v5'] = json.load(f)
        except Exception:
            pass
    # ridge_bias: main grid + optional redundancy sweep.
    # v0.80.0.33: the 192-row α/β/pool_dim sweep was being silently
    # dropped because only `results.csv` was read. The redundancy
    # sweep adds α (ridge regularization) × β (lasso strength) ×
    # pool_dim variation on top of the (rho_A, sigma) grid and is
    # essential for the redundancy-vs-nonlinearity argument in §6.
    p = os.path.join(_calibration_dir('ridge_bias'), 'results.csv')
    if os.path.exists(p):
        block['ridge_bias'] = _csv_to_rows(p)
    # Try a few candidate filenames for the redundancy sweep -- the
    # original artifact filename has a typo (`redundency`) preserved
    # for backward compatibility.
    for _redfn in ('ridge_bias_results_redundency.csv',
                   'ridge_bias_results_redundancy.csv',
                   'results_redundancy.csv'):
        _redp = os.path.join(_calibration_dir('ridge_bias'), _redfn)
        if os.path.exists(_redp):
            block['ridge_bias_redundancy'] = _csv_to_rows(_redp)
            break
    # toy_nonlinearity: CSV
    p = os.path.join(_calibration_dir('toy_nonlinearity'), 'results.csv')
    if os.path.exists(p):
        block['toy_nonlinearity'] = _csv_to_rows(p)
    # channel_marginal: CSV
    p = os.path.join(_calibration_dir('channel_marginal'), 'channel_marginal_nonlinearity.csv')
    if os.path.exists(p):
        block['channel_marginal'] = _csv_to_rows(p)
    # v0.81.1.6 ship 4: paper 2 measurement-apparatus foundations
    p = os.path.join(_calibration_dir('split_pca_selection'), 'selection.json')
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8-sig') as f:
                block['split_pca_selection'] = json.load(f)
        except Exception:
            block['split_pca_selection'] = None
    p = os.path.join(_calibration_dir('knn_mi_reliability'), 'reliability.json')
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8-sig') as f:
                block['knn_mi_reliability'] = json.load(f)
        except Exception:
            block['knn_mi_reliability'] = None
    return block


def _csv_to_rows(path):
    try:
        import csv as _csv
        with open(path, 'r', encoding='utf-8-sig') as f:
            return list(_csv.DictReader(f))
    except Exception:
        return None


def _write_cache_key_manifest(name):
    """Write the current cache key to disk for a calibration dir. Used for
    first-time bootstrap (existing artifact but no manifest) and after
    fresh regeneration."""
    cal_dir = _calibration_dir(name)
    os.makedirs(cal_dir, exist_ok=True)
    key = _compute_cache_key(name)
    path = _cache_key_path(name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(key, f, indent=2)


def _calibration_status_ex(name):
    """Extended status: returns one of:
       'fresh'       -- artifact + manifest present, hashes match
       'no_manifest' -- artifact present, manifest missing (first-time bootstrap)
       'stale'       -- artifact + manifest present, hashes differ
       'missing'     -- no artifact on disk
    """
    _, artifact = _CALIBRATION_SCRIPTS[name]
    artifact_path = os.path.join(_calibration_dir(name), artifact)
    if not os.path.exists(artifact_path):
        return 'missing'
    stored = _load_cache_key(name)
    if stored is None:
        return 'no_manifest'
    current = _compute_cache_key(name)
    if stored.get('script_hash') != current.get('script_hash'):
        return 'stale'
    for up, h in current.get('upstream', {}).items():
        if stored.get('upstream', {}).get(up) != h:
            return 'stale'
    return 'fresh'


def _run_paper_calibration_phase(ui_log=None):
    """Run 0056 Phase A -- methodology calibration.

    For each of the 4 calibration scripts:
      - 'fresh'       → skip
      - 'no_manifest' → write manifest in place (first-time bootstrap)
      - 'stale'       → backup existing output, re-run script
      - 'missing'     → run script (no backup needed)

    Subprocess output streams to stdout so Flask log captures it. Returns
    a summary dict with per-script outcome.

    v0.80.0.33: this is the orchestrator half that used to live implicitly
    in 'user runs scripts manually'. 0056 now owns the calibration phase."""
    import subprocess
    import sys as _sys
    import time

    def _log(msg):
        if ui_log is not None:
            try: ui_log(msg)
            except Exception: print(msg, flush=True)
        else:
            print(msg, flush=True)

    outcomes = {}
    repo_root = os.path.dirname(os.path.abspath(__file__))

    # Ordered to front-load the cheap ones for quick feedback
    # v0.82.0.0 ship 4 fix: split_pca_selection and knn_mi_reliability
    # were added to _CALIBRATION_SCRIPTS but the execution loop here was
    # never extended. The completeness check at the END of Phase A saw
    # six entries and reported "incomplete: split_pca_selection,
    # knn_mi_reliability" -- but those scripts had never been invoked.
    # Now invoked. Order matters because knn_mi_reliability depends on
    # split_pca_selection's output (operating-point selection writes
    # selection.json which knn_mi_reliability reads).
    order = [
        'v5',                  # ~5 min,    no upstream
        'toy_nonlinearity',    # ~20 min,   no upstream
        'ridge_bias',          # slowest,   no upstream
        'channel_marginal',    # depends on real model data
        'split_pca_selection', # depends on V5 generators (~10-30 min)
        'knn_mi_reliability',  # depends on V5 + selection (~15-30 min)
    ]
    # v5 (~5min) and toy_nonlinearity (~20min) run first to surface any
    # env issues early. ridge_bias paper_scale and the paper-2 V5-grid
    # diagnostics are the long-tail.

    _log("=" * 72)
    _log("Run 0056 -- Phase A: methodology calibration")
    _log("=" * 72)

    for name in order:
        script_name, _ = _CALIBRATION_SCRIPTS[name]
        t0 = time.time()
        status = _calibration_status_ex(name)
        running_marker = os.path.join(_calibration_dir(name), '.running')
        had_crash = os.path.exists(running_marker)
        _log(f"[{name}] status: {status}" + (" (crashed last run)" if had_crash else ""))

        if status == 'fresh':
            outcomes[name] = 'skipped'
            continue

        if status == 'no_manifest' and not had_crash:
            # Artifact exists + no manifest + no .running marker = file
            # was migrated or produced by a prior script version. Stamp
            # manifest and move on -- no need to regenerate.
            _log(f"[{name}] existing artifact found, no cache manifest -- "
                 f"stamping manifest in place (first-time bootstrap)")
            try:
                _write_cache_key_manifest(name)
                outcomes[name] = 'bootstrapped'
            except Exception as e:
                _log(f"[{name}] ! manifest write failed: {e}")
                outcomes[name] = f'manifest-fail: {e}'
            continue

        if had_crash:
            # Prior run crashed mid-execution. Re-run the script; its own
            # resume logic will skip cells already in the output CSV.
            _log(f"[{name}] prior run crashed (.running marker found) -- "
                 f"re-running script; resume will skip completed cells")

        if status == 'stale':
            _log(f"[{name}] hashes differ -- backing up existing output")
            try:
                bak = _backup_calibration(name)
                if bak: _log(f"[{name}] backed up to {bak}")
            except Exception as e:
                _log(f"[{name}] ! backup failed: {e}")

        # Missing or stale → run script
        script_path = os.path.join(repo_root, script_name)
        if not os.path.exists(script_path):
            _log(f"[{name}] ! script not found: {script_path}")
            outcomes[name] = 'script-not-found'
            continue

        cmd = [_sys.executable, script_path]
        if name == 'ridge_bias':
            cmd += ['--config', 'paper_scale']
        elif name == 'toy_nonlinearity':
            cmd += ['--seeds', '0', '1', '2', '3', '4']

        # v0.80.0.33: remove the cache manifest before running. If the
        # script crashes mid-run, the next orchestrator invocation sees
        # no_manifest (artifact exists, resume-eligible) and runs the
        # script again, which will pick up incremental CSV state and
        # skip already-done cells. Without this unstamp, a crash could
        # leave a fresh manifest next to partial output, and orchestrator
        # would think calibration is complete.
        manifest_path = _cache_key_path(name)
        if os.path.exists(manifest_path):
            try:
                os.remove(manifest_path)
            except Exception:
                pass

        _log(f"[{name}] running: {' '.join(cmd)}")
        _log(f"[{name}] output streams below; this may take several minutes ...")

        # v0.80.0.33: write .running marker before starting, remove on
        # successful exit. If the process crashes or is killed, the
        # marker persists and the next orchestrator invocation detects
        # crash-recovery rather than treating partial output as bootstrap.
        try:
            os.makedirs(_calibration_dir(name), exist_ok=True)
            with open(running_marker, 'w', encoding='utf-8') as f:
                f.write(f"started at {time.time()}\n")
        except Exception:
            pass

        try:
            # v0.80.0.33: live-tail subprocess stdout into both streams.
            # subprocess.call inherits stdout to .iota_flask.log only,
            # which means Simple panel (reads .iota_log.jsonl) sits
            # empty while a long calibration runs. Now: capture stdout
            # via Popen+PIPE, read line-by-line, route each line to:
            #   1. parent stdout (→ .iota_flask.log → Detailed)
            #   2. _log() (→ .iota_log.jsonl → Simple)
            # Same architecture, no new threading; reads block at the
            # speed of subprocess output, which is fine since the parent
            # was blocked on subprocess.call() before too.
            _env = dict(os.environ)
            _env['PYTHONIOENCODING'] = 'utf-8'
            proc = subprocess.Popen(
                cmd, cwd=repo_root, env=_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1,  # line-buffered on the parent side
                universal_newlines=True, encoding='utf-8',
                errors='replace')
            # Read until subprocess exits, bridging each line to both streams.
            try:
                for line in proc.stdout:
                    line = line.rstrip('\n')
                    print(line, flush=True)   # → .iota_flask.log (Detailed)
                    _log(line)                # → .iota_log.jsonl (Simple)
            finally:
                proc.wait()
                rc = proc.returncode
        except Exception as e:
            _log(f"[{name}] ! subprocess failed: {e}")
            outcomes[name] = f'error: {e}'
            continue

        dt = time.time() - t0
        if rc == 0:
            try:
                _write_cache_key_manifest(name)
                # Clean up .running marker on success
                if os.path.exists(running_marker):
                    try: os.remove(running_marker)
                    except Exception: pass
                outcomes[name] = f'regenerated ({dt:.0f}s)'
                _log(f"[{name}] done in {dt:.0f}s, manifest stamped")
            except Exception as e:
                _log(f"[{name}] ! manifest write failed: {e}")
                outcomes[name] = f'regen-ok manifest-fail ({dt:.0f}s)'
        else:
            # .running marker stays -- next run will detect crash
            outcomes[name] = f'exit {rc}'
            _log(f"[{name}] ! exited with rc={rc}; .running marker kept "
                 f"for crash recovery on next run")

    _log("")
    _log("Calibration phase summary:")
    for name, outcome in outcomes.items():
        _log(f"  {name:20s} {outcome}")
    _log("=" * 72)
    return outcomes


def _run_paper_figures_phase(ui_log=None, override=False):
    """Run 0056 Phase C -- figure generation.

    Runs the 5 fig scripts via subprocess, streaming output. Figure
    failures are logged but do not abort the phase -- partial figure
    output is better than no paper artifact.

    v0.80.0.33: figures auto-fire on successful results.json write.
    Matches the 'press 0056 once, get everything' spec."""
    import subprocess
    import sys as _sys

    def _log(msg):
        if ui_log is not None:
            try: ui_log(msg)
            except Exception: print(msg, flush=True)
        else:
            print(msg, flush=True)

    scripts = [
        'fig1_ridge_vs_mlp_by_temperature.py',
        'fig2_toy_heatmap.py',
        'fig3_mechanism_falsification.py',
        'fig4_temperature_trajectories.py',
        'fig5_layer_causal_profile.py',
        # v0.82.0.0: paper 2 λ-sensitivity figure. Reads
        # cell['measurements']['lambda_sweep'] (Phase 8 output). Skips
        # gracefully when the measurement is absent (paper-2 phases
        # haven't fired).
        'fig6_lambda_sensitivity.py',
    ]
    repo_root = os.path.dirname(os.path.abspath(__file__))
    outcomes = {}

    _log("=" * 72)
    _log("Run 0056 -- Phase C: figure generation")
    _log("=" * 72)

    for script in scripts:
        script_path = os.path.join(repo_root, script)
        if not os.path.exists(script_path):
            _log(f"[{script}] ! not found")
            outcomes[script] = 'not-found'
            continue
        cmd = [_sys.executable, script_path]
        if override:
            cmd.append('--override')
        _log(f"[{script}] rendering ...")
        try:
            # v0.80.0.33: live-tail subprocess stdout (see calibration
            # phase comment for rationale). Figures are quick (~1-2s
            # each), so the bridge is mostly cosmetic here, but keeps
            # the architecture consistent.
            _env = dict(os.environ)
            _env['PYTHONIOENCODING'] = 'utf-8'
            proc = subprocess.Popen(
                cmd, cwd=repo_root, env=_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True,
                encoding='utf-8', errors='replace')
            try:
                for line in proc.stdout:
                    line = line.rstrip('\n')
                    print(line, flush=True)
                    _log(line)
            finally:
                proc.wait()
                rc = proc.returncode
        except Exception as e:
            _log(f"[{script}] ! exception: {e}")
            outcomes[script] = f'error: {e}'
            continue
        outcomes[script] = 'ok' if rc == 0 else f'exit {rc}'

    _log("")
    _log("Figure phase summary:")
    for script, outcome in outcomes.items():
        _log(f"  {script:45s} {outcome}")
    _log("=" * 72)
    return outcomes


# ══════════════════════════════════════════════════════════════════════


# ── Master Results File (v0.72.2.0) ──────────────────────────────────────────
# Aggregates all analysis JSONs across all temperatures into one file.
# Called by _run_54_stats_report after all per-temp stats complete.

# v0.80.0.33: Q0046 payload remap.
# Historical Q0046 files carry per-condition entries whose `run_num` fields
# are in pre-0.79.4.0 old numbering (e.g. run_num=3 with label="Introspection A",
# when new Intro A is run 6). Filename is canonical (Q0046_*); payload is legacy.
# Fresh 56b passes under current code also write new numbering, so we remap at
# ingestion -- idempotent on already-new values.
_OLD_TO_NEW_RUN = {
    1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:11, 9:12,
    10:39, 11:40, 12:29, 13:30, 14:31,
    15:13, 16:14, 17:15, 18:32, 19:1, 20:2, 21:17,
    22:28, 23:22, 24:27, 25:47, 26:3, 27:48, 28:23, 29:24,
    30:20, 31:21, 32:49, 33:42, 34:43, 35:38, 36:34, 37:37,
    38:36, 39:35, 40:51, 41:33, 42:18, 43:25, 44:26, 45:41,
    46:44, 47:50, 48:16, 49:45, 50:52, 51:53, 52:54,
    53:19, 54:55, 55:56, 56:46,
}
_NEW_RUN_VALUES = set(_OLD_TO_NEW_RUN.values())

def _remap_q56_payload(q56_data):
    """In-memory remap of Q0046 per_condition entries' run_num fields from
    old numbering to new. Returns the same dict (mutated) for convenience.
    Idempotent: values already in new range are left alone.
    Pooled block is untouched -- it has no run_num field."""
    if not isinstance(q56_data, dict):
        return q56_data
    pc = q56_data.get('per_condition')
    if not isinstance(pc, dict):
        return q56_data
    remapped = {}
    for key, entry in pc.items():
        if isinstance(entry, dict):
            rn = entry.get('run_num')
            if isinstance(rn, int) and rn in _OLD_TO_NEW_RUN and rn not in _NEW_RUN_VALUES:
                entry['run_num'] = _OLD_TO_NEW_RUN[rn]
            # Rewrite legacy key prefix ("run06" ok as-is; "run6" pad; "run56" old)
            # Leave keys alone -- they're opaque dict labels, consumers index via
            # entry['run_num'] not the outer key.
            # Source field normalisation:
            rr = entry.get('ridge_reference')
            if isinstance(rr, dict):
                src = rr.get('source')
                if isinstance(src, str) and 'Q46_per_condition_R.json' in src:
                    rr['source'] = 'Q0044_per_condition_R.json'
        remapped[key] = entry
    q56_data['per_condition'] = remapped
    return q56_data


def build_master_results(session):
    """Build master_results.json with everything the writer bot needs."""
    import datetime
    from cartography import get_paths, get_pooled_paths, DATA

    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')

    _TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    _ANALYSIS_FILES = {
        'Q25': 'Q0047_granger_A.json',
        'Q27': 'Q0048_granger_B.json',
        'Q32': 'Q0049_baseline_swap.json',
        'Q33': 'Q0042_decomposition.json',
        'Q34': 'Q0043_sobol_partition.json',
        'Q45': 'Q0041_pool_dim_sweep.json',
        'Q46': 'Q0044_per_condition_R.json',
        'Q49': 'Q0045_fixed_dim_per_condition_R.json',
        'Q56': 'Q0046_mlp_decomposition.json',
        'hypothesis_outcomes': 'hypothesis_outcomes.json',
        'descriptive_statistics': 'descriptive_statistics.json',
    }
    _POOLED_FILES = {
        'Q40_decomposition': 'Q0051_pooled_decomposition.json',
        'Q40_sobol': 'Q0051_pooled_sobol.json',
        'Q47': 'Q0050_cross_temp_synthesis.json',
        'cross_temp_status': 'cross_temp_status.json',
        'three_variant_comparison': 'three_variant_comparison.json',
    }
    _CROSS_MODEL_FILES = {
        'Q50': 'Q0052_cross_model_R_comparison.json',
        'Q51': 'Q0053_condition_concordance.json',
        'Q52': 'Q0054_cross_model_summary.json',
    }

    master = {
        'model': f'{family}-{size}',
        'family': family, 'size': size, 'variant': variant,
        'model_name': session.get('model_name', ''),
        'generated': datetime.datetime.now().isoformat(),
        'per_temperature': {},
        'pooled': {},
        'cross_model': {},
    }

    # Per-temperature data
    for temp in _TEMPS:
        paths = get_paths(family, size, variant, temp, create_dirs=False)
        ana_dir = paths.get('analysis', '')
        temp_data = {}
        for key, filename in _ANALYSIS_FILES.items():
            fp = os.path.join(ana_dir, filename)
            if os.path.exists(fp):
                try:
                    with open(fp) as f:
                        raw = f.read()
                    raw = re.sub(r'\bNaN\b', 'null', raw)
                    raw = re.sub(r'\bInfinity\b', 'null', raw)
                    raw = re.sub(r'\b-Infinity\b', 'null', raw)
                    parsed = json.loads(raw)
                    # v0.80.0.33: remap legacy run_num in Q56/Q0046 payloads.
                    if key == 'Q56':
                        parsed = _remap_q56_payload(parsed)
                    temp_data[key] = parsed
                except Exception:
                    pass
        if temp_data:
            master['per_temperature'][str(temp)] = temp_data

    # Pooled data
    pp = get_pooled_paths(family, size, variant, create_dirs=False)
    pooled_ana = pp.get('analysis', '')
    for key, filename in _POOLED_FILES.items():
        fp = os.path.join(pooled_ana, filename)
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    raw = f.read()
                raw = re.sub(r'\bNaN\b', 'null', raw)
                raw = re.sub(r'\bInfinity\b', 'null', raw)
                raw = re.sub(r'\b-Infinity\b', 'null', raw)
                master['pooled'][key] = json.loads(raw)
            except Exception:
                pass

    # Cross-model data
    for key, filename in _CROSS_MODEL_FILES.items():
        fp = os.path.join(pooled_ana, filename)
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    raw = f.read()
                raw = re.sub(r'\bNaN\b', 'null', raw)
                raw = re.sub(r'\bInfinity\b', 'null', raw)
                raw = re.sub(r'\b-Infinity\b', 'null', raw)
                master['cross_model'][key] = json.loads(raw)
            except Exception:
                pass

    # Write to pooled/analysis/
    out_path = os.path.join(pooled_ana, 'master_results.json')
    os.makedirs(pooled_ana, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(master, f, indent=2, default=str)
    ui.ok(f"  Master results: {out_path}")
    n_temps = len(master['per_temperature'])
    n_pooled = len(master['pooled'])
    n_cross = len(master['cross_model'])
    ui.msg(f"  {n_temps} temperatures, {n_pooled} pooled files, {n_cross} cross-model files")
    return out_path


# ── Cross-model master aggregate (v0.78.2.1) ─────────────────────────────────
# Integrated from the standalone build_master_jsons.py: the per-model
# build_master_results above plus these three functions fully replace that
# script. Run 0056 calls build_all_masters() after its per-model copy loop
# so the ArXiv-ready all_models_master.json refreshes automatically.
#
# For ad-hoc refreshes between R55 runs (e.g., when a new Run 0046 lands mid-
# paper-writing), invoke directly:
#   python -c "import export_stats; export_stats.build_all_masters()"
# or verify status without writing:
#   python -c "import export_stats; export_stats.verify_masters()"

def _paper_model_key(model_info):
    """Return the paper/json/ filename prefix for a discovered model.

    Uses the on-disk size directory (e.g. '2b_fp16') rather than the
    logical size ('2b') so the 2B-Q4 and 2B-FP16 masters don't collide
    at the paper layer. Matches Run 0056's existing naming convention.
    """
    fam = model_info['family']
    # model_info['size'] from _discover_model_data is the disk directory
    # name already (walks sz_dir by listdir). Keep as-is for the filename.
    size_dir = model_info['size']
    return f"{fam}_{size_dir}"


def _cross_arch_summary(all_models_dict):
    """Derive a small cross-architecture section from the per-model master
    dicts in all_models_master.json. Convenience for the writer bot --
    no new data, just pre-computed views for direct S7 paragraph writing.

    Sections:
      R_pooled_by_model     -- mean R across all temperatures per model
      R_per_temperature     -- per-model × per-temp R grid (Q33 source)
      ridge_vs_mlp          -- for each model: Ridge R (Q34) vs MLP R (Q56),
                              plus ridge_gap (the core Conjecture 1 check)
                              and conjecture_1_verdict ('ridge_adequate' if
                              max |gap| < 0.05, else 'mlp_materially_different')
      decomposition_summary -- one-line 'E={}, C={}, R={}' per model from
                              pooled Q40 (or first per-temp Q33 as fallback)
    """
    summary = {
        'R_pooled_by_model':     {},
        'R_per_temperature':     {},
        'ridge_vs_mlp':          {},
        'decomposition_summary': {},
    }

    for key, m in all_models_dict.items():
        per_t  = m.get('per_temperature', {}) or {}
        pooled = m.get('pooled', {}) or {}

        # R per temperature from Q33 (preferred) or fallback fields.
        r_by_temp = {}
        for t_str, t_data in per_t.items():
            q33 = t_data.get('Q33') or {}
            r = q33.get('R_fraction') or q33.get('R') or q33.get('r_fraction')
            if r is not None:
                r_by_temp[t_str] = float(r)
        if r_by_temp:
            summary['R_per_temperature'][key] = r_by_temp
            summary['R_pooled_by_model'][key] = (
                sum(r_by_temp.values()) / len(r_by_temp)
            )

        # Ridge vs MLP -- requires Q34 (ridge) and Q56 (MLP) at same temps.
        # Pool across whatever temperatures have both.
        ridge_Rs, mlp_Rs, ridge_gaps = [], [], []
        for t_str, t_data in per_t.items():
            q34 = t_data.get('Q34') or {}
            q56 = t_data.get('Q56') or {}
            r_ridge = (q34.get('R_fraction')
                       or (q34.get('ridge') or {}).get('R_fraction')
                       or q34.get('r_fraction'))
            q56_pooled = (q56.get('pooled') or {})
            r_mlp = q56_pooled.get('mlp_r_mean') or q56_pooled.get('R_fraction')
            gap   = q56_pooled.get('ridge_gap')
            if r_ridge is not None: ridge_Rs.append(float(r_ridge))
            if r_mlp   is not None: mlp_Rs.append(float(r_mlp))
            if gap     is not None: ridge_gaps.append(float(gap))

        if ridge_Rs or mlp_Rs:
            entry = {}
            if ridge_Rs:
                entry['ridge_R_mean']  = sum(ridge_Rs) / len(ridge_Rs)
                entry['ridge_n_temps'] = len(ridge_Rs)
            if mlp_Rs:
                entry['mlp_R_mean']    = sum(mlp_Rs) / len(mlp_Rs)
                entry['mlp_n_temps']   = len(mlp_Rs)
            if ridge_gaps:
                entry['ridge_gap_mean'] = sum(ridge_gaps) / len(ridge_gaps)
                entry['ridge_gap_max']  = max(ridge_gaps, key=abs)
                # Conjecture 1 threshold: max |gap| < 0.05 → ridge adequate.
                # Writer bot pulls this verdict directly for S7 cross-arch text.
                entry['conjecture_1_verdict'] = (
                    'ridge_adequate' if abs(entry['ridge_gap_max']) < 0.05
                    else 'mlp_materially_different'
                )
            summary['ridge_vs_mlp'][key] = entry

        # E+C+R one-liner from pooled Q40; fallback to first per-temp Q33.
        ecr_src = (pooled.get('Q40_decomposition') or {})
        if not ecr_src:
            for t_data in per_t.values():
                q33 = t_data.get('Q33') or {}
                if q33:
                    ecr_src = q33
                    break
        if ecr_src:
            E = ecr_src.get('E_fraction') or ecr_src.get('E')
            C = ecr_src.get('C_fraction') or ecr_src.get('C')
            R = ecr_src.get('R_fraction') or ecr_src.get('R')
            if E is not None and C is not None and R is not None:
                summary['decomposition_summary'][key] = (
                    f"E={float(E):.3f}, C={float(C):.3f}, R={float(R):.3f}"
                )

    return summary


def build_all_masters(verbose=True, rebuild_per_model=True):
    """Rebuild per-model master_results.json for every discovered model
    (when rebuild_per_model=True), then aggregate them into
    DATA/paper/json/all_models_master.json.

    rebuild_per_model=False skips the per-model rebuild and just
    re-aggregates from existing master_results.json files on disk.
    Used by Run 0056's call path, where build_master_results was already
    invoked per-model in-line and re-running would double the work.

    Returns the absolute path to all_models_master.json, or None on
    complete discovery failure.

    Aggregate structure (schema_version 1.0):
      generated            -- ISO timestamp
      n_models, model_keys -- discovery summary
      models               -- per-model masters indexed by 'family_sizedir'
      cross_architecture   -- R_pooled_by_model, R_per_temperature,
                             ridge_vs_mlp (with conjecture_1_verdict),
                             decomposition_summary
    """
    from cartography import DATA
    import datetime

    models = _discover_model_data({})
    if verbose:
        print(f"\n  Discovered {len(models)} model(s) under {DATA}")
        for m in models:
            print(f"    - {_paper_model_key(m)}  ({m['label']})")

    paper_json = os.path.join(DATA, 'paper', 'json')
    os.makedirs(paper_json, exist_ok=True)

    # Per-model rebuild (optional -- Run 0056 skips this since it already ran)
    if rebuild_per_model:
        if verbose:
            print(f"\n  Rebuilding per-model master_results.json ...")
        for m in models:
            session = {
                'model_family':  m['family'],
                'model_size':    m['size'],
                'model_variant': m['variant'],
                'model_name':    m['label'],
            }
            try:
                build_master_results(session)
            except Exception as e:
                if verbose:
                    print(f"    [x] {_paper_model_key(m)}: {e}")

    # Collect per-model masters into one aggregate dict
    all_models_dict = {}
    for m in models:
        key = _paper_model_key(m)
        # Prefer paper/json/ copy (Run 0056 output); fall back to pooled/analysis/
        cand_paths = [
            os.path.join(paper_json, f"{key}_master_results.json"),
            os.path.join(m['data_dir'], 'pooled', 'analysis', 'master_results.json'),
        ]
        for src in cand_paths:
            if not os.path.exists(src):
                continue
            try:
                with open(src, 'r', encoding='utf-8') as f:
                    raw = f.read()
                raw = re.sub(r'\bNaN\b', 'null', raw)
                raw = re.sub(r'\bInfinity\b', 'null', raw)
                raw = re.sub(r'\b-Infinity\b', 'null', raw)
                data = json.loads(raw)
                data['_source_path'] = src
                data['_key'] = key
                all_models_dict[key] = data
                break
            except Exception:
                continue

    aggregate = {
        'schema_version':    '1.0',
        'generated':         datetime.datetime.now().isoformat(),
        'n_models':          len(all_models_dict),
        'model_keys':        sorted(all_models_dict.keys()),
        'cross_architecture': _cross_arch_summary(all_models_dict),
        'models':            all_models_dict,
        'methodology_calibration': _load_methodology_calibration(),
    }

    # v0.80.0.33: switch to the new flat-per-cell schema via results_builder.
    # The legacy `models` dict (per-model master passthrough) is preserved
    # under `legacy_models` for 0.80-era paper consumers still using the old
    # per-model view; new code reads `cells.{key}`.
    import results_builder as _rb
    import results_schema  as _rs

    # v0.82.0.12: pull from results_schema.IOTA_VERSION rather than
    # hardcoding. The hardcoded "0.80.0.0" got missed by the version
    # bump sweep at every ship, so results.json emissions kept stamping
    # the old framework version even after schema_version moved to
    # 0.81.0. Single source of truth is in results_schema now.
    _iota_version = _rs.IOTA_VERSION

    new_results = _rb.build_results(
        data_root=DATA,
        iota_version=_iota_version,
        methodology_calibration_block=_load_methodology_calibration(),
        git_commit=None,  # Kevin handles git manually
    )

    # Attach the legacy per-model view for backward-compatible consumers.
    # Never required by the schema (additionalProperties allowed at root).
    new_results['legacy_models'] = all_models_dict
    new_results['cross_architecture_legacy'] = _cross_arch_summary(all_models_dict)

    paper_root = os.path.join(DATA, 'paper')
    os.makedirs(paper_root, exist_ok=True)
    out_path = os.path.join(paper_root, 'results.json')

    try:
        _rb.write_results(new_results, out_path)
    except _rs.SchemaValidationError as _sve:
        # Blocking: re-raise so the caller (Run 0056 dispatcher) fails
        # loudly. A results.json that fails its own schema never ships.
        raise

    if verbose:
        print(f"\n  [+] {out_path}")
        print(f"      schema_version:  {new_results['schema_version']}")
        print(f"      cells:           {len(new_results['cells'])}")
        cca = new_results.get('cross_cell_aggregates') or {}
        rs = cca.get('rhat_ridge_spread_at_T1.0')
        ms = cca.get('rhat_mlp_spread_at_T1.0')
        if rs is not None and ms is not None:
            print(f"      Ridge spread T1.0: {rs:.3f}    MLP spread T1.0: {ms:.3f}")
        conv = cca.get('convergence_status_per_temperature') or {}
        for tk, d in sorted(conv.items()):
            print(f"      T={tk}  n={d.get('n')}  spread={d.get('spread'):.3f}  status={d.get('status')}")
        ca = new_results.get('cross_architecture_legacy') or {}
        if ca.get('ridge_vs_mlp'):
            print(f"\n      Conjecture 1 verdicts (legacy view):")
            for key, d in ca['ridge_vs_mlp'].items():
                v = d.get('conjecture_1_verdict', 'insufficient_data')
                gap = d.get('ridge_gap_max')
                gap_s = f"(max |gap| = {abs(gap):.3f})" if gap is not None else ""
                print(f"        {key:24s}  {v}  {gap_s}")
    return out_path


def verify_masters():
    """Read-only status report for master-JSON state across the tree.

    Walks DATA/, reports per model a flag matrix [S P 6 F]:
      S -- source master exists in pooled/analysis/master_results.json
      P -- paper copy exists in DATA/paper/json/{key}_master_results.json
      6 -- Q56 present in the master (ingested by build_master_results)
      F -- per-model master is fresh vs DATA/paper/json/all_models_master.json

    Zero writes. Use before/after running build_all_masters() to verify
    state, or to diagnose stale aggregates without rebuilding.
    """
    from cartography import DATA
    paper_json = os.path.join(DATA, 'paper', 'json')
    models = _discover_model_data({})
    print(f"\n  Verifying {len(models)} model(s):\n")
    agg_path = os.path.join(paper_json, 'all_models_master.json')
    agg_mtime = os.path.getmtime(agg_path) if os.path.exists(agg_path) else 0
    stale = []
    for m in models:
        key = _paper_model_key(m)
        src   = os.path.join(m['data_dir'], 'pooled', 'analysis', 'master_results.json')
        paper = os.path.join(paper_json, f"{key}_master_results.json")
        has_src, has_paper = os.path.exists(src), os.path.exists(paper)
        has_q56 = False
        if has_src:
            try:
                with open(src) as f:
                    raw = f.read()
                raw = re.sub(r'\bNaN\b', 'null', raw)
                data = json.loads(raw)
                for t_data in (data.get('per_temperature') or {}).values():
                    if 'Q56' in t_data:
                        has_q56 = True; break
            except Exception:
                pass
        mtime = os.path.getmtime(src) if has_src else 0
        fresh = mtime <= agg_mtime and agg_mtime > 0
        if not fresh and has_src:
            stale.append(key)
        flags = (("S" if has_src   else "-") +
                 ("P" if has_paper else "-") +
                 ("6" if has_q56   else "-") +
                 ("F" if fresh     else "-"))
        print(f"    [{flags}]  {key:28s}  {m['label']}")
    print(f"\n  Legend: S=pooled master  P=paper copy  6=has Q56  F=fresh vs aggregate")
    print(f"  all_models_master.json: {'present' if agg_mtime else 'MISSING'}")
    if stale:
        print(f"  Stale (would refresh on next build_all_masters): {stale}")


def fig_condition_overview(df, vis_dir):
    """Combined: cluster overview (left) + condition trajectories (right)."""
    if df.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _CLUSTERS = {
        'Null': [4, 5, 1], 'Introspection': [6, 7, 8],
        'Arithmetic': [9, 10, 11, 12], 'Priming': [13, 14, 15],
        'Impossibility': [29, 30, 31], 'Perturbation': [39, 40],
    }
    _CLR = {'Null': '#888888', 'Introspection': '#000000', 'Arithmetic': '#555555',
            'Priming': '#AAAAAA', 'Impossibility': '#CCCCCC', 'Perturbation': '#333333'}
    for name, runs in _CLUSTERS.items():
        sub = df[df['run_mode'].isin(runs)]
        if sub.empty or 'state_similarity_index' not in sub.columns:
            continue
        agg = sub.groupby('turn')['state_similarity_index'].mean()
        ax1.plot(agg.index, agg.values, 'o-', color=_CLR.get(name, '#666'),
                 markersize=4, linewidth=1.5, label=name)
    ax1.set_xlabel('Turn')
    ax1.set_ylabel('Mean Similarity')
    ax1.set_title('Condition Cluster Overview')
    ax1.legend(fontsize=8)
    for label, runs, ls, marker in [
        ('Introspection', [6, 7, 8], '-', 'o'),
        ('Null', [4, 5, 1], '--', 's'),
        ('Arithmetic', [9, 10, 11, 12], ':', '^'),
    ]:
        sub = df[df['run_mode'].isin(runs)]
        if sub.empty or 'state_similarity_index' not in sub.columns:
            continue
        agg = sub.groupby('turn')['state_similarity_index'].agg(['mean', 'sem']).reset_index()
        ax2.plot(agg['turn'], agg['mean'], marker=marker, linestyle=ls, color='black',
                 markersize=4, linewidth=1.5, label=label)
        ax2.fill_between(agg['turn'], agg['mean']-agg['sem'].fillna(0),
                         agg['mean']+agg['sem'].fillna(0), alpha=0.1, color='black')
    ax2.set_xlabel('Turn')
    ax2.set_ylabel('Mean Similarity ± SEM')
    ax2.set_title('Condition Trajectories')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    return save_fig(fig, vis_dir, "condition_overview")


def fig_causal_patching_combined(df21, df18, vis_dir):
    """Combined: patching by mode (left) + per-layer isolation (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _has_left = False
    if not df21.empty and 'patch_mode' in df21.columns and 'output_changed' in df21.columns:
        df21 = df21.copy()
        df21['output_changed'] = pd.to_numeric(df21['output_changed'], errors='coerce')
        modes = ['none', 'partial', 'full', 'random']
        rates, labels_used = [], []
        for m in modes:
            sub = df21[df21['patch_mode'] == m]['output_changed'].dropna()
            if len(sub) >= 5:
                rates.append(float(sub.mean() * 100))
                labels_used.append(m)
        if rates:
            hatches = ['', '///', '...', 'xxx']
            bars = ax1.bar(range(len(labels_used)), rates, color='#888888',
                           edgecolor='black', linewidth=0.8, width=0.6)
            for i, bar in enumerate(bars):
                bar.set_hatch(hatches[i % len(hatches)])
            ax1.set_xticks(range(len(labels_used)))
            ax1.set_xticklabels(labels_used, fontsize=9)
            ax1.set_ylabel('Output Change Rate (%)')
            ax1.set_title('Run 0017 -- Patching by Mode')
            _has_left = True
    if not _has_left:
        ax1.text(0.5, 0.5, 'No Run 0017 data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Run 0017 -- No Data')
    _has_right = False
    if not df18.empty and 'patch_layer' in df18.columns and 'output_changed' in df18.columns:
        df18 = df18.copy()
        df18['output_changed'] = pd.to_numeric(df18['output_changed'], errors='coerce')
        layers = ['L8', 'L16', 'L24', 'L31']
        rates, labels_used = [], []
        for layer in layers:
            sub = df18[df18['patch_layer'] == layer]['output_changed'].dropna()
            if len(sub) >= 5:
                rates.append(float(sub.mean() * 100))
                labels_used.append(layer)
        if rates:
            bars = ax2.bar(range(len(labels_used)), rates, color='#888888',
                           edgecolor='black', linewidth=0.8, width=0.6)
            ax2.set_xticks(range(len(labels_used)))
            ax2.set_xticklabels(labels_used, fontsize=9)
            ax2.set_ylabel('Output Change Rate (%)')
            ax2.set_title('Run 0018 -- Per-Layer Isolation')
            _has_right = True
    if not _has_right:
        ax2.text(0.5, 0.5, 'No Run 0018 data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Run 0018 -- No Data')
    plt.tight_layout()
    return save_fig(fig, vis_dir, "causal_patching_combined")


def generate_paper_figures(session):
    """Generate all paper figures once into the pooled visuals directory."""
    # Guard: base/instruct variants only have hidden states
    if session.get('model_variant', '') in ('base', 'instruct'):
        return 0
    from cartography import get_pooled_paths, get_paths
    pp = get_pooled_paths(
        session['model_family'], session['model_size'],
        session.get('model_variant', 'abliterated'), create_dirs=True)
    vis_dir = pp['visuals']

    # ── Paper style: override seaborn, enforce grayscale + serif ──────────
    _saved_rc = plt.rcParams.copy()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'axes.prop_cycle': plt.cycler(color=_BW_COLORS),
    })

    # Load CSV for FIG7 from T=0.2
    df_for_sim = pd.DataFrame()
    _p02 = get_paths(session['model_family'], session['model_size'],
                     session.get('model_variant', 'abliterated'), 0.2, create_dirs=False)
    csv_dir = _p02.get('csv', '')
    if os.path.isdir(csv_dir):
        csvs = sorted(f for f in os.listdir(csv_dir) if f.endswith('.csv'))
        if csvs:
            try:
                frames = [pd.read_csv(os.path.join(csv_dir, c), dtype=str,
                                       keep_default_na=False) for c in csvs]
                df_for_sim = pd.concat(frames, ignore_index=True)
                for col in ['turn', 'run_mode', 'state_similarity_index']:
                    if col in df_for_sim.columns:
                        df_for_sim[col] = pd.to_numeric(df_for_sim[col], errors='coerce')
            except Exception:
                pass
    _PAPER_TASKS = [
        ("FIG01 R(T) curve",              lambda: fig_r_vs_temperature(session, vis_dir, single_model=True)),
        ("FIG02 OLS ΔR²",                lambda: fig_ols_bars(session, vis_dir, single_model=True)),
        ("FIG03 Patching heatmap",         lambda: fig_patching_heatmap(session, vis_dir)),
        ("FIG04 E+C+R grouped bar",        lambda: fig_ecr_stacked_bar(session, vis_dir, single_model=True)),
        ("FIG05 Held-out validation",       lambda: fig_held_out_validation(session, vis_dir)),
        ("FIG06 dim₉₅ curve",              lambda: fig_dim95_curve(session, vis_dir)),
        ("FIG07 Per-turn similarity",              lambda: (fig_similarity_trajectory(df_for_sim, vis_dir, [3,4,5,1,2,19], "FIG07", "Per-Turn Similarity") if not df_for_sim.empty else None)),
        ("FIG08 Bootstrap CI",              lambda: fig_bootstrap_distributions(session, vis_dir)),
        ("FIG09 Per-condition R",            lambda: fig_per_condition_r(session, vis_dir)),
        ("FIG11 Fixed-dim cross-temp",       lambda: fig_fixed_dim_cross_temp(session, vis_dir)),
        ("FIG12 Interaction mass",           lambda: fig_interaction_mass(session, vis_dir)),
        ("FIG13 Three-variant delta",        lambda: fig_three_variant_delta(session, vis_dir)),
    ]
    n = len(_PAPER_TASKS)
    ui.section(f"Generating {n} paper figures...")
    ok = 0
    for i, (label, fn) in enumerate(_PAPER_TASKS, 1):
        try:
            f = fn()
            if f:
                ui.ok(f"  [{i}/{n}] {label} ✓")
                ok += 1
            else:
                ui.warn(f"  [{i}/{n}] {label} -- returned None (data missing or insufficient)")
        except Exception as e:
            ui.warn(f"  [{i}/{n}] {label} -- EXCEPTION: {e}")
    ui.ok(f"  {ok}/{n} paper figures → {vis_dir}")
    plt.rcParams.update(_saved_rc)
    plt.close('all')
    import gc; gc.collect()
    return ok



def _load_all_temps_run_for_model(run_num, model_info):
    """Load CSV data for a specific run across all temps for a specific model."""
    from cartography import get_paths, RUN_CSV
    frames = []
    var_dir = model_info['data_dir']
    for cond in sorted(os.listdir(var_dir)):
        if cond in ('pooled', 'paper'):
            continue
        csv_dir = os.path.join(var_dir, cond, 'csv')
        if not os.path.isdir(csv_dir):
            continue
        if cond == 'deterministic':
            temp = 0.0
        elif cond.startswith('temp_'):
            try: temp = float(cond.split('_')[1])
            except: continue
        else:
            continue
        csv_name = RUN_CSV.get(run_num, f'R{run_num:04d}.csv')
        fp = os.path.join(csv_dir, csv_name)
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp, dtype=str, keep_default_na=False)
                df['temperature'] = str(temp)
                frames.append(df)
            except Exception:
                pass
    if frames:
        result = pd.concat(frames, ignore_index=True)
        for col in result.columns:
            if col not in ('condition', 'prompt', 'response', 'patch_layer', 'patch_mode'):
                result[col] = pd.to_numeric(result[col], errors='coerce')
        return result
    return pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session: dict, paths: dict):
    # Guard: base/instruct variants only have hidden states -- no stats to export
    if session.get('model_variant', '') in ('base', 'instruct'):
        ui.msg(f"  Skipping stats export for {session.get('model_variant')} variant (hidden states only)")
        return

    csv_dir = paths['csv']
    ana_dir = paths['analysis']
    vis_dir = paths['visuals']

    ui.section("Loading data...")
    ui.msg(f"  CSV dir:      {csv_dir}")
    ui.msg(f"  Analysis dir: {ana_dir}")
    df      = load_all(csv_dir)
    granger = load_granger(ana_dir)
    bs      = load_baseline_swap(ana_dir)
    decomp  = load_decomposition(ana_dir)
    sobol   = load_permutation_sensitivity(ana_dir)
    ui.msg(f"  Granger: {list(granger.keys()) or 'EMPTY'}")
    ui.msg(f"  Baseline swap: {'OK' if bs else 'EMPTY'}")
    ui.msg(f"  Decomposition: {'OK' if decomp else 'EMPTY'}")
    ui.msg(f"  Sobol: {'OK' if sobol else 'EMPTY'}")
    if not df.empty and 'run_mode' in df.columns:
        runs_in_df = sorted(df['run_mode'].dropna().astype(int).unique().tolist())
        ui.msg(f"  Run modes in df: {runs_in_df}")
    else:
        ui.warn("  df is EMPTY or missing run_mode -- no CSV data loaded")

    # Bug V fix (v25.7): load Run 0051 pooled sobol from its own directory
    from cartography import get_pooled_paths
    pooled_paths = get_pooled_paths(
        session['model_family'], session['model_size'],
        session.get('model_variant', 'abliterated'), create_dirs=False)
    pooled_sobol = load_pooled_permutation_sensitivity(pooled_paths['analysis'])

    if df.empty:
        choice = ui.srx_prompt(
            "No CSV data found -- run experimental runs first before exporting stats."
        )
        if choice in ('s', 'x'):
            if choice == 'x': raise SystemExit
            return

    ui.ok(f"{len(df):,} analysis rows loaded.")
    ui.blank()
    _FIG_TASKS = [
        # Core per-temperature figures (v0.73.1.0) -- juice only
        ("Condition overview",          lambda: fig_condition_overview(df, vis_dir)),
        ("Granger ΔR²",                lambda: fig_granger(granger, vis_dir)),
        ("Sobol partition",             lambda: fig_sobol(sobol, vis_dir)),
        ("Confound isolation",          lambda: fig_confound(df, vis_dir, decomp=decomp)),
        ("Linearity validation",        lambda: fig_linearity(decomp, vis_dir)),
        ("Interaction information",     lambda: fig_interaction_info(decomp, vis_dir)),
        ("Causal patching",             lambda: fig_causal_patching_combined(
            df[run_mode_mask(df['run_mode'], 17)] if not df.empty else pd.DataFrame(),
            df[run_mode_mask(df['run_mode'], 18)] if not df.empty else pd.DataFrame(), vis_dir)),
        ("Entropy shape",               lambda: fig_entropy_shape(df, vis_dir)),
        ("Perturbation recovery",       lambda: fig_perturbation_recovery(df, vis_dir)),
    ]
    if pooled_sobol:
        _FIG_TASKS.append(("Sobol pooled", lambda: fig_sobol(pooled_sobol, vis_dir, suffix="_pooled")))

    n_total = len(_FIG_TASKS)
    ui.section(f"Generating {n_total} figures...")
    figures = []
    for i, (label, fn) in enumerate(_FIG_TASKS, 1):
        ui.ok(f"  [{i}/{n_total}] {label}")
        try:
            f = fn()
            figures.append(f)
        except Exception as e:
            import traceback
            ui.warn(f"  [{i}/{n_total}] SKIPPED -- {e}")
            ui.warn(traceback.format_exc().strip().split('\n')[-1])

    ok_figs = [f for f in figures if f]
    ui.ok(f"{len(ok_figs)}/{n_total} figures saved to {vis_dir}")
    plt.close('all')
    import gc; gc.collect()

    # Infer outcomes
    ui.section("Inferring hypothesis outcomes...")
    try:
        outcomes = _infer_outcomes(df, granger, bs, decomp, sobol, pooled_sobol=pooled_sobol)
    except Exception as _oe:
        import traceback
        ui.err(f"_infer_outcomes failed: {_oe}")
        ui.err(traceback.format_exc().strip().split('\n')[-1])
        outcomes = {}

    # FIG_SUMMARY cut (v0.71.0.0) -- dashboard material, not a paper figure.
    # fig_summary function retained for backward compat but not called in pipeline.

    # Save outcomes JSON
    out_json = os.path.join(ana_dir, "hypothesis_outcomes.json")
    ui.msg(f"  Writing outcomes to: {out_json}")
    os.makedirs(ana_dir, exist_ok=True)
    try:
        with open(out_json, 'w') as fp:
            json.dump(outcomes, fp, indent=2, default=str)
        ui.ok(f"Outcomes: {out_json}")
    except Exception as _we:
        ui.err(f"Failed to write outcomes: {_we}")

    # ── Descriptive statistics (v0.58.0.0) ────────────────────────────────────
    ui.section("Computing descriptive statistics...")

    # Cohen's d effect size table
    try:
        effect_sizes = compute_effect_size_table(df)
        ui.ok(f"  Cohen's d: {len(effect_sizes.get('comparisons', []))} pairwise comparisons")
        # Print headline comparisons
        for entry in effect_sizes.get('comparisons', []):
            if entry['metric'] == 'state_similarity_index':
                d = entry['cohens_d']
                if not np.isnan(d):
                    label = f"{entry['group_a']} vs {entry['group_b']}"
                    mag = 'large' if abs(d) > 0.8 else ('medium' if abs(d) > 0.5 else 'small')
                    ui.msg(f"    similarity  {label:<40s}  d={d:+.3f} ({mag})")
    except Exception as _e:
        ui.warn(f"  Cohen's d failed: {_e}")
        effect_sizes = {'comparisons': [], 'summary': {}}

    # KS distribution tests
    try:
        ks_results = compute_ks_tests(df)
        ui.ok(f"  KS tests: {len(ks_results.get('tests', []))} distribution comparisons")
        for entry in ks_results.get('tests', []):
            if entry['metric'] == 'state_similarity_index':
                ks = entry['ks_statistic']
                p = entry['p_value']
                ov = entry['overlap_estimate']
                label = f"{entry['group_a']} vs {entry['group_b']}"
                sig = '*' if p < 0.05 else ''
                ui.msg(f"    similarity  {label:<40s}  KS={ks:.3f} p={p:.3f}{sig}  overlap≈{ov:.1%}")
    except Exception as _e:
        ui.warn(f"  KS tests failed: {_e}")
        ks_results = {'tests': [], 'summary': {}}

    # Split-half reliability
    try:
        split_half = compute_split_half_reliability(df)
        ui.ok(f"  Split-half reliability: {len(split_half)} condition groups")
        for grp, vals in sorted(split_half.items()):
            r = vals['r_mean']
            quality = 'excellent' if r > 0.9 else ('good' if r > 0.8 else ('acceptable' if r > 0.7 else 'poor'))
            ui.msg(f"    {grp:<30s}  r={r:.3f}±{vals['r_std']:.3f} ({quality}, n={vals['n_trials']})")
    except Exception as _e:
        ui.warn(f"  Split-half reliability failed: {_e}")
        split_half = {}

    # Bonferroni + BH correction table (v0.71.0.0)
    correction_table = {}
    try:
        correction_table = compute_correction_table(outcomes)
        n_bonf = sum(1 for t in correction_table.get('tests', []) if t['bonferroni_reject'])
        n_bh   = sum(1 for t in correction_table.get('tests', []) if t['bh_reject'])
        n_tot  = correction_table.get('n_tests', 0)
        ui.ok(f"  Correction table: {n_tot} tests -- Bonferroni rejects {n_bonf}, BH rejects {n_bh}")
        ui.msg(f"    Bonferroni α = {correction_table.get('bonferroni_alpha', 0):.4f}")
    except Exception as _e:
        ui.warn(f"  Correction table failed: {_e}")

    # Cohen's d headline summary (v0.71.0.0)
    cohens_d_summary = {}
    try:
        cohens_d_summary = compute_cohens_d_summary(df)
        for c in cohens_d_summary.get('comparisons', []):
            mag = 'large' if abs(c['cohens_d']) > 0.8 else ('medium' if abs(c['cohens_d']) > 0.5 else 'small')
            ui.msg(f"    {c['hypothesis']} {c['label']:<35s}  d={c['cohens_d']:+.3f} ({mag})  p={c['p_value']:.3f}")
    except Exception as _e:
        ui.warn(f"  Cohen's d summary failed: {_e}")

    # Held-out descriptive comparison (v0.71.0.0)
    # KS test per metric: in-sample vs Run 0033 held-out
    heldout_comparison = {}
    try:
        sub33 = df[run_mode_mask(df['run_mode'], 33)] if (not df.empty and 'run_mode' in df.columns) else pd.DataFrame()
        sub_insample = df[run_mode_mask_any(df['run_mode'], [6,7,8,9,10,11,12,13,14,15,1,3,23])] if not df.empty else pd.DataFrame()
        if not sub33.empty and not sub_insample.empty:
            _ho_metrics = ['state_similarity_index', 'mean_logit_entropy', 'layer_sim_mean', 'onset_delay_ratio']
            _ho_tests = []
            for metric in _ho_metrics:
                if metric not in sub33.columns or metric not in sub_insample.columns:
                    continue
                a = pd.to_numeric(sub33[metric], errors='coerce').dropna()
                b = pd.to_numeric(sub_insample[metric], errors='coerce').dropna()
                if len(a) >= 5 and len(b) >= 5:
                    ks_stat, p_val = sp_stats.ks_2samp(a, b)
                    _ho_tests.append({
                        'metric': metric, 'ks_statistic': float(ks_stat),
                        'p_value': float(p_val), 'n_heldout': len(a), 'n_insample': len(b),
                        'mean_heldout': float(a.mean()), 'mean_insample': float(b.mean()),
                        'same_distribution': bool(p_val >= 0.05),
                    })
            heldout_comparison = {'tests': _ho_tests, 'n_metrics': len(_ho_tests)}
            n_same = sum(1 for t in _ho_tests if t['same_distribution'])
            ui.ok(f"  Held-out comparison: {n_same}/{len(_ho_tests)} metrics same distribution")
    except Exception as _e:
        ui.warn(f"  Held-out comparison failed: {_e}")

    # Output token entropy under patching (v0.71.0.0)
    # For Run 0017 and Run 0019: entropy of output text distribution under patching vs no patching
    patching_entropy = {}
    try:
        from cartography import run_mode_mask as _rmm_es3  # v0.79.5.2: dual-accept
        for rn, label in [(21, 'real'), (53, 'noise')]:
            sub_pe = df[_rmm_es3(df['run_mode'], rn)] if not df.empty else pd.DataFrame()
            if sub_pe.empty or 'output' not in sub_pe.columns or 'patch_mode' not in sub_pe.columns:
                continue
            for mode in ['none', 'partial', 'full']:
                mode_rows = sub_pe[sub_pe['patch_mode'] == mode]
                outputs = mode_rows['output'].dropna().tolist()
                if len(outputs) >= 10:
                    # Token frequency entropy: count unique output strings, compute Shannon entropy
                    from collections import Counter
                    counts = Counter(outputs)
                    total_n = sum(counts.values())
                    probs = np.array([c / total_n for c in counts.values()])
                    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
                    patching_entropy[f'{label}_{mode}'] = {
                        'entropy_bits': entropy, 'n_unique': len(counts),
                        'n_total': total_n,
                    }
        if patching_entropy:
            ui.ok(f"  Patching entropy: {len(patching_entropy)} mode combinations computed")
            for k, v in sorted(patching_entropy.items()):
                ui.msg(f"    {k:<20s}  H={v['entropy_bits']:.2f} bits  ({v['n_unique']} unique / {v['n_total']} total)")
    except Exception as _e:
        ui.warn(f"  Patching entropy failed: {_e}")

    # Appendix D: per-condition descriptive statistics (v0.71.0.0)
    appendix_d = {}
    try:
        _APP_GROUPS = {
            'introspection': [6,7,8], 'arithmetic': [9,10,11,12], 'null': [4,5,1],
            'priming': [13,14,15], 'perturbation': [39,40], 'impossibility': [29,30,31],
        }
        _APP_METRICS = ['state_similarity_index', 'mean_logit_entropy', 'layer_sim_mean', 'onset_delay_ratio']
        _app_rows = []
        for grp, runs in _APP_GROUPS.items():
            for metric in _APP_METRICS:
                if metric not in df.columns:
                    continue
                vals = pd.to_numeric(df[df['run_mode'].isin(runs)][metric], errors='coerce').dropna()
                if len(vals) >= 2:
                    _app_rows.append({
                        'group': grp, 'metric': metric,
                        'mean': float(vals.mean()), 'sd': float(vals.std()),
                        'n': len(vals),
                    })
        appendix_d = {'rows': _app_rows}
        ui.ok(f"  Appendix D: {len(_app_rows)} condition×metric entries")
    except Exception as _e:
        ui.warn(f"  Appendix D failed: {_e}")

    # Appendix E: interaction mass table (v0.71.0.0)
    # Read from Q34 sobol data already loaded
    appendix_e = {}
    try:
        if sobol and 'perm_sens_E' in sobol:
            appendix_e = {
                'raw_E_effect': sobol.get('perm_sens_E', {}).get('effect'),
                'raw_C_effect': sobol.get('perm_sens_C', {}).get('effect'),
                'raw_R_effect': sobol.get('perm_sens_R', {}).get('effect'),
                'raw_sum': sobol.get('total_first_order_raw'),
                'baseline_var': sobol.get('baseline_var'),
                'interaction_mass': sobol.get('interaction_mass_unattributed'),
                'renorm_factor': sobol.get('total_first_order_fraction'),
            }
            ui.ok(f"  Appendix E: interaction mass = {appendix_e.get('interaction_mass', 'n/a')}")
    except Exception as _e:
        ui.warn(f"  Appendix E failed: {_e}")

    # Appendix F: split-half reliability (already computed above, just format)
    appendix_f = split_half  # already computed

    # Save descriptive stats JSON
    desc_json = os.path.join(ana_dir, "descriptive_statistics.json")
    try:
        desc_stats = {
            'effect_sizes': effect_sizes,
            'ks_tests': ks_results,
            'split_half_reliability': split_half,
            'correction_table': correction_table,
            'cohens_d_summary': cohens_d_summary,
            'heldout_comparison': heldout_comparison,
            'patching_entropy': patching_entropy,
            'appendix_d_descriptives': appendix_d,
            'appendix_e_interaction_mass': appendix_e,
            'appendix_f_split_half': appendix_f,
        }
        with open(desc_json, 'w') as fp:
            json.dump(desc_stats, fp, indent=2, default=str)
        ui.ok(f"Descriptive stats: {desc_json}")
    except Exception as _we:
        ui.err(f"Failed to write descriptive stats: {_we}")

    # Squid -- two hearts. Once per model family, after stats complete.
    _squid_family = session.get('model_family', 'unknown')
    _squid2 = os.path.join(ana_dir, f'.iota_squid2_{_squid_family}')
    if not os.path.exists(_squid2):
        try:
            import time as _st, sys as _ss
            _sb = "\n       _____\n      /     \\\n     /  ___  \\\n    |  /   \\  |\n    | | \u25c9 \u25c9 | |\n    | |  \u223c  | |\n    |  \\___/  |\n     \\       /\n      \\_____/\n        |||||\n        |||||\n        |||||\n       /|||||\\\n      ~~~~~~~~~~~\n"
            _sm = _sb + "\n  # Squids have two hearts.    Probably.\n"
            for _sl in _sm.split("\n"): _ss.stdout.write(_sl+"\n"); _ss.stdout.flush(); _st.sleep(0.025)
            open(_squid2, 'w').close()
        except Exception:
            pass

    # Print to console
    ui.blank()
    ui.section("Hypothesis Outcomes")
    for h_id in sorted(outcomes):
        o = outcomes[h_id]
        sym = {"supported":"✓","disproven":"✗","inconclusive":"?","pending":"·"}.get(o['status'],'·')
        ui.msg(f"  {sym} {h_id}: {HYPOTHESES.get(h_id,[''])[0]:<30} {o['status'].upper()}")
    ui.blank()


def main(session: dict):
    """Entry point called by start_here.py menu dispatch."""
    ui.header("Export -- Stats + Figures")
    ui.session_summary(session)
    from cartography import get_paths
    paths = get_paths(session['model_family'], session['model_size'],
                      session.get('model_variant', 'abliterated'), session.get('temperature', 0.0))
    run(session, paths)


if __name__ == "__main__":
    ui.install_deps()
    session = ui.load_session()
    ui.header("Export -- Stats + Figures")
    ui.session_summary(session)
    paths = get_paths(session['model_family'], session['model_size'],
                      session.get('model_variant','abliterated'), session.get('temperature',0.0))
    run(session, paths)
