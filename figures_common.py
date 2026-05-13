"""
figures_common.py -- shared plumbing for the 5 paper figures.

Provides:
  setup_nature_rcparams()   -- matplotlib config per paper_figure_specs.md
  OKABE_ITO                 -- colorblind-safe palette
  COLORS                    -- paper-specific color assignments
  load_results()            -- read data/paper/results.json + its hash
  save_figure(fig, name, meta) -- writes fig.svg + fig.pdf + fig.meta.json
                                 to data/paper/. Sidecar carries the
                                 results_json_hash + source_fields so
                                 later readers can verify provenance.
  MM_TO_INCHES              -- conversion constant
  check_calibration_or_exit(override=False) -- block figure generation
                              when calibration is missing, unless override
                              flag is set (with loud warnings).
"""

import datetime
import hashlib
import json
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
PAPER_ROOT = os.path.join(DATA, 'paper')
RESULTS_JSON = os.path.join(PAPER_ROOT, 'results.json')


MM_TO_INCHES = 1.0 / 25.4


OKABE_ITO = {
    'black':          '#000000',
    'orange':         '#E69F00',
    'sky_blue':       '#56B4E9',
    'bluish_green':   '#009E73',
    'yellow':         '#F0E442',
    'blue':           '#0072B2',
    'vermillion':     '#D55E00',
    'reddish_purple': '#CC79A7',
}

# Paper-specific semantic assignments. Consistency across figures so a
# reader who sees "blue = Ridge" in fig1 sees the same in fig3/fig4.
COLORS = {
    'ridge':         OKABE_ITO['blue'],           # estimator, bar + line
    'mlp':           OKABE_ITO['vermillion'],     # estimator, bar + line

    # Architecture assignments (for fig4/fig5 per-arch lines)
    'llama_8b':      OKABE_ITO['black'],
    'gemma_9b':      OKABE_ITO['sky_blue'],
    'gemma_2b_q4':   OKABE_ITO['bluish_green'],
    'gemma_2b_fp16': OKABE_ITO['orange'],

    # Heatmap diverging (fig2)
    'diverging':     'RdBu_r',
}


def setup_nature_rcparams():
    import matplotlib as mpl
    # Font
    # Universal font selection -- detect what's available, prefer in order.
    # v0.80.0.34: keeps only fonts matplotlib has indexed on this machine,
    # so no 'font not found' warnings on any platform.
    import matplotlib.font_manager as _fm
    _available = {f.name for f in _fm.fontManager.ttflist}
    _preferred = ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans L', 'DejaVu Sans']
    _family = [f for f in _preferred if f in _available] or ['DejaVu Sans']
    mpl.rcParams['font.family']     = _family
    mpl.rcParams['font.size']       = 7
    mpl.rcParams['axes.labelsize']  = 7
    mpl.rcParams['axes.titlesize']  = 7
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['figure.titlesize']= 8
    # Editable text in PDF/SVG
    mpl.rcParams['pdf.fonttype']    = 42
    mpl.rcParams['ps.fonttype']     = 42
    mpl.rcParams['svg.fonttype']    = 'none'
    # Lines and ticks
    mpl.rcParams['axes.linewidth']  = 0.6
    mpl.rcParams['xtick.major.width'] = 0.6
    mpl.rcParams['ytick.major.width'] = 0.6
    mpl.rcParams['xtick.major.size']  = 2.5
    mpl.rcParams['ytick.major.size']  = 2.5
    mpl.rcParams['xtick.direction']   = 'out'
    mpl.rcParams['ytick.direction']   = 'out'
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['axes.spines.right'] = False


def file_sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load_results():
    """Read results.json. Returns (dict, hash). Raises FileNotFoundError
    if the file isn't there."""
    if not os.path.exists(RESULTS_JSON):
        raise FileNotFoundError(
            f"results.json not found at {RESULTS_JSON}. "
            "Run Run 0056 (paper assembly) to generate.")
    import re as _re
    with open(RESULTS_JSON, 'r', encoding='utf-8-sig') as f:
        raw = f.read()
    # Handle "inf" / "-inf" strings we wrote as NaN-safe markers
    parsed = json.loads(raw)
    return parsed, file_sha256(RESULTS_JSON)


def save_figure(fig, name, source_fields, cell_filter=None, caption_draft='',
                iota_version=None, results_hash=None):
    """Write fig.svg + fig.pdf + fig.meta.json to data/paper/.

    name: e.g. 'fig1_ridge_vs_mlp_by_temperature' -- no extension
    source_fields: list of dotted paths from results.json the figure reads
    cell_filter: optional dict describing any row/cell filter applied
    caption_draft: first-pass caption text for the paper
    iota_version: framework version stamped into the sidecar. If None,
                  pulled from results_schema.IOTA_VERSION (single source
                  of truth). v0.82.0.12: prior default was hardcoded
                  "0.80.0.0" which the bump sweep missed at every ship.
    results_hash: sha256 of results.json at render time
    """
    if iota_version is None:
        try:
            import results_schema as _rs
            iota_version = _rs.IOTA_VERSION
        except Exception:
            iota_version = 'unknown'
    os.makedirs(PAPER_ROOT, exist_ok=True)
    svg_path = os.path.join(PAPER_ROOT, f'{name}.svg')
    pdf_path = os.path.join(PAPER_ROOT, f'{name}.pdf')
    meta_path = os.path.join(PAPER_ROOT, f'{name}.meta.json')

    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)

    meta = {
        'figure_name':       name,
        'source_fields':     source_fields,
        'cell_filter':       cell_filter,
        'caption_draft':     caption_draft,
        'results_json_hash': results_hash,
        'generated_at':      datetime.datetime.now().isoformat(),
        'iota_version':      iota_version,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return {'svg': svg_path, 'pdf': pdf_path, 'meta': meta_path}


def cell_rhat_ridge(measurements):
    """Resolve the cell's Ridge R-hat with fallback to MLP-minus-gap.

    v0.80.0.30: critical correctness fix. The earlier helper attempted
    `rhat_mlp_reference - mlp_vs_ridge_gap_R_pooled`, but the gap is
    defined as `pooled_mean_R - ridge_R` (analysis.py:3360 -- `gap[c] =
    mean[c] - ridge_fracs[c]` where `mean` is the pooled mean across
    architectures). Subtracting the gap from a different MLP value
    (the reference architecture's R, not the pooled mean) produces an
    offset Ridge value that's off by exactly `reference_R -
    pooled_mean_R`. For Run 0046's typical multi-architecture sweeps
    that offset is several percent -- meaningful enough to corrupt the
    Ridge bars in fig1 and the cross-cell Ridge spread in
    cross_cell_aggregates. The pooled mean is the only correct
    denominator. No other fallback is mathematically valid.

    Returns None when neither the direct R0043 value nor the
    pooled-mean-minus-gap form is available.
    """
    if not isinstance(measurements, dict):
        return None
    direct = measurements.get('rhat_ridge_insample')
    if isinstance(direct, (int, float)) and not (
            isinstance(direct, float) and direct != direct):  # NaN check
        return direct
    pm  = measurements.get('rhat_mlp_pooled_mean')
    gap = measurements.get('mlp_vs_ridge_gap_R_pooled')
    if (isinstance(pm, (int, float)) and isinstance(gap, (int, float))
            and not (isinstance(pm, float) and pm != pm)
            and not (isinstance(gap, float) and gap != gap)):
        return pm - gap
    return None


def cell_rhat_mlp(measurements, prefer_pooled=False):
    """Resolve the cell's MLP R-hat.

    By default returns `rhat_mlp_reference` (the designated reference
    architecture's R, typically '256') with fallback to
    `rhat_mlp_pooled_mean`. Set `prefer_pooled=True` when the caller
    needs the pooled-mean specifically -- for paired-bar plots where
    the visible MLP-vs-Ridge gap should equal the stored gap value,
    or for cross-cell aggregates that compare against the gap.
    Reference and pooled differ by a few percent in typical 0046
    output.
    """
    if not isinstance(measurements, dict):
        return None
    if prefer_pooled:
        pm = measurements.get('rhat_mlp_pooled_mean')
        if isinstance(pm, (int, float)) and not (isinstance(pm, float) and pm != pm):
            return pm
        ref = measurements.get('rhat_mlp_reference')
        return ref if isinstance(ref, (int, float)) else None
    ref = measurements.get('rhat_mlp_reference')
    if isinstance(ref, (int, float)) and not (isinstance(ref, float) and ref != ref):
        return ref
    pm = measurements.get('rhat_mlp_pooled_mean')
    return pm if isinstance(pm, (int, float)) else None


def check_calibration_or_exit(override=False):
    """If any calibration artifact is missing/stale, block figure generation
    unless override=True. Override prints three loud warnings first so no
    one can claim accident."""
    try:
        import export_stats as _es
        st = _es.methodology_calibration_status()
    except Exception:
        st = None
    if st is None:
        print("! could not read calibration status; proceeding", file=sys.stderr)
        return

    if st.get('_any_missing') or st.get('_any_stale'):
        missing = [k for k, v in st.items() if not k.startswith('_') and v == 'missing']
        stale   = [k for k, v in st.items() if not k.startswith('_') and v == 'stale']
        msg = []
        if missing: msg.append(f"missing: {', '.join(missing)}")
        if stale:   msg.append(f"stale: {', '.join(stale)}")
        if not override:
            print("=" * 70, file=sys.stderr)
            print("CALIBRATION INCOMPLETE -- figure generation BLOCKED", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            print(" | ".join(msg), file=sys.stderr)
            print("", file=sys.stderr)
            print("Run these first:", file=sys.stderr)
            print("  python v5_synthetic_calibration.py", file=sys.stderr)
            print("  python ridge_bias_toy.py --config paper_scale", file=sys.stderr)
            print("  python toy_nonlinearity_asymmetry.py --seeds 0 1 2 3 4", file=sys.stderr)
            print("  python run_channel_marginal.py", file=sys.stderr)
            print("", file=sys.stderr)
            print("Or re-run this script with --override to generate figures", file=sys.stderr)
            print("anyway. Figures produced under override are NOT valid", file=sys.stderr)
            print("for paper submission -- the methodology evidence they", file=sys.stderr)
            print("depend on has not been regenerated.", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            sys.exit(1)
        else:
            # Three loud warnings
            for i in range(3):
                print("!" * 70, file=sys.stderr)
                print(f"!!! OVERRIDE #{i+1}/3: calibration incomplete; figures NOT "
                      "valid for paper submission", file=sys.stderr)
                print(f"!!! {' | '.join(msg)}", file=sys.stderr)
                print("!" * 70, file=sys.stderr)
