"""
IOTA FRAMEWORK — CARTOGRAPHY
==============================
Single source of truth for all file paths.
Every script imports this. Nothing writes outside data/.

Structure:
  {root}/data/{family}/{size}_{quant}/{variant}/{condition}/
    csv/
    hidden_states/
    analysis/
    visuals/

  {root}/data/{family}/{size}_{quant}/
    queue.json          ← per-model job queue and completion state (v30.0)
    calibration_*.json  ← power calibration per model

  {root}/
    .iota_env.json      ← machine environment: GPU, VRAM, venv path (v30.0)

Root = directory containing start_here.py (auto-detected). Flat layout — no subfolders.
Run from anywhere. Same structure every time.

D is for cartography. You know where you are by where you've been.

v30.0: added queue helpers (load_queue, save_queue, scan_run_completion),
       added env file helpers (load_env, get_vram_gb), added get_family_size_dir().
       All existing APIs unchanged.
"""

import os
import re
import glob
import json
import numpy as np

VALID_TEMPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


# ── Run ID helpers (v0.79.4.0) ────────────────────────────────────────────────
# v0.79.4.0 promotes run identifiers to 4-digit zero-padded strings as the
# primary key everywhere — in-memory dicts, CSV run_mode column, on-disk
# filenames, JSON keys, UI labels, paper section cross-refs.
#
# Accept both int and string at every public entry point for transitional
# robustness: code that still passes `run_num=33` (integer) gets normalised
# to `"0033"` at the boundary. The internal canonical form is always the
# 4-digit string.

def run_id_pad(x) -> str:
    """Normalise any run identifier (int or string) to the canonical
    4-digit zero-padded string form.
    
    Examples:
      3       -> "0003"
      "3"     -> "0003"
      "0003"  -> "0003"
      "R0003" -> "0003"   (R/Q prefix stripped)
      "Q0033" -> "0033"
      33      -> "0033"
    
    Returns the string unchanged if it's already a valid 4-digit run_id.
    Raises ValueError for anything else.
    """
    if isinstance(x, int):
        if x < 0 or x > 9999:
            raise ValueError(f"Run integer out of range: {x}")
        return f"{x:04d}"
    if isinstance(x, str):
        s = x.strip().upper()
        # Strip historical R/Q phase prefix if present.
        if s and s[0] in ('R', 'Q'):
            s = s[1:]
        # Must now be all digits.
        if s.isdigit() and len(s) <= 4:
            return s.zfill(4)
        raise ValueError(f"Cannot parse as run_id: {x!r}")
    raise ValueError(f"run_id_pad: expected int or str, got {type(x).__name__}")


def run_id_to_int(run_id) -> int:
    """Inverse of run_id_pad — strip padding/prefix, return integer.
    Accepts int passthrough."""
    if isinstance(run_id, int):
        return run_id
    return int(run_id_pad(run_id))


def run_mode_matches(cell_value, run_num) -> bool:
    """Test whether a single CSV run_mode cell value matches a run identifier.
    Canonical 4-digit form only ("0003"). Legacy integer-string acceptance
    dropped in v0.79.6.4 after migration completion verified.

    v0.79.5.0."""
    if cell_value is None:
        return False
    s = str(cell_value).strip()
    if not s:
        return False
    try:
        rid = run_id_pad(run_num)
    except ValueError:
        return False
    return s == rid


def run_mode_mask(run_mode_series, run_num):
    """Pandas-friendly: return a boolean Series matching the canonical
    4-digit run_mode. One-call helper for the dozens of filter sites
    across analysis/export_stats/runners.

    v0.79.5.0."""
    try:
        rid = run_id_pad(run_num)
    except ValueError:
        return run_mode_series.astype(str) == ''  # no matches
    return run_mode_series.astype(str) == rid


def run_mode_mask_any(run_mode_series, run_nums):
    """Multi-run variant of run_mode_mask — returns boolean Series matching
    ANY of the given run identifiers (iterable) in canonical 4-digit form.

    Replaces `df['run_mode'].isin([3,4,5])` patterns that broke on migrated
    data (run_mode is a string column; int comparisons always returned
    False under pandas-native string-vs-int).

    v0.79.5.0."""
    values = []
    for rn in run_nums:
        try:
            values.append(run_id_pad(rn))
        except ValueError:
            continue
    if not values:
        return run_mode_series.astype(str) == ''
    return run_mode_series.astype(str).isin(values)


class DualKeyRunDict(dict):
    """Dict that accepts both int and 4-digit string run-ID keys.
    
    The canonical internal storage is 4-digit string ("0003"); any int key
    passed to __getitem__, __setitem__, __contains__, get(), or pop() is
    transparently normalised via run_id_pad(). String keys pass through
    unchanged (with the same normalisation, so "3" becomes "0003" too).
    
    Purpose: lets the v0.79.4.0 run-ID migration flip the canonical key
    format to string without breaking the 100+ `RUN_MAP["0042"]` style access
    sites across runners.py / analysis.py / export_flask.py / scanner.py /
    export_stats.py. Each of those sites continues to work; the dict
    handles the int→string conversion at lookup time.
    
    Iteration yields keys in their stored (4-digit string) form — writers
    that care about the string form (e.g., filename builders) already use
    run_id_pad internally. Writers that pass the key back as an integer
    (legacy code) can use run_id_to_int() at the boundary.
    """

    @staticmethod
    def _norm(k):
        if isinstance(k, int):
            return run_id_pad(k)
        if isinstance(k, str):
            try:
                return run_id_pad(k)
            except ValueError:
                return k
        return k

    def __getitem__(self, k):
        return super().__getitem__(self._norm(k))

    def __setitem__(self, k, v):
        super().__setitem__(self._norm(k), v)

    def __contains__(self, k):
        return super().__contains__(self._norm(k))

    def get(self, k, default=None):
        return super().get(self._norm(k), default)

    def pop(self, k, *args):
        return super().pop(self._norm(k), *args)


class DualKeyRunSet(set):
    """Set variant of DualKeyRunDict — accepts int or 4-digit string
    members transparently. Membership checks (`r in S`) normalise at
    the boundary. Canonical storage is 4-digit strings.
    
    Used for _GEN_RUNS, _ANALYSIS_RUNS, _ET_RECOVERY_RUNS, TEMP_INDEP_RUNS,
    etc. — sets of run IDs that need to accept both legacy integer and
    new 4-digit string lookups during the transition."""

    def __init__(self, iterable=()):
        super().__init__(DualKeyRunDict._norm(x) for x in iterable)

    def __contains__(self, x):
        return super().__contains__(DualKeyRunDict._norm(x))

    def add(self, x):
        super().add(DualKeyRunDict._norm(x))

    def discard(self, x):
        super().discard(DualKeyRunDict._norm(x))

    def remove(self, x):
        super().remove(DualKeyRunDict._norm(x))


# ── Run → CSV filename map ─────────────────────────────────────────────────────
# Single source of truth. Used by dedup, flask, start_here, runners_core.
# v0.79.4.0: RUNS RENUMBERED TO EXECUTION-ORDER POSITIONS (1-56). Keys are
# 4-digit strings. Filename values follow new R/Q convention (R=1-21, Q=22+).
# The "was" comments reference the pre-renumber run IDs to aid migration.
RUN_CSV = DualKeyRunDict({
    "0001": "R0001_null.csv",                   # was R19 (null+hidden)
    "0002": "R0002_robustness.csv",             # was R20
    "0003": "R0003_temperature_grid.csv",       # was Q26
    "0004": "R0004_null.csv",                   # was R01
    "0005": "R0005_null.csv",                   # was R02
    "0006": "R0006_introspection.csv",          # was R03
    "0007": "R0007_introspection.csv",          # was R04
    "0008": "R0008_introspection.csv",          # was R05
    "0009": "R0009_math.csv",                   # was R06
    "0010": "R0010_math.csv",                   # was R07
    "0011": "R0011_math.csv",                   # was R08
    "0012": "R0012_math.csv",                   # was R09
    "0013": "R0013_framing.csv",                # was R15
    "0014": "R0014_framing.csv",                # was R16
    "0015": "R0015_framing.csv",                # was R17
    "0016": "R0016_et_recovery.csv",            # v0.79.5.0: Run 16 promoted to first-class MC run with 13 source_run conditions
    "0017": "R0017_patching.csv",               # was R21
    "0018": "R0018_layer_isolation.csv",        # was Q42
    "0019": "R0019_random_patching.csv",        # was R53
    "0020": "R0020_instances.csv",              # was Q30
    "0021": "R0021_conditions.csv",             # was Q31
    "0022": "Q0022_saturation.csv",             # was Q23
    "0023": "Q0023_introspection.csv",          # was Q28 (C_t confound)
    "0024": "Q0024_persistence.csv",            # was Q29
    "0025": "Q0025_lengths.csv",                # was Q43
    "0026": "Q0026_recovery.csv",               # was Q44
    "0027": "Q0027_layers.csv",                 # was Q24
    "0028": "Q0028_contradiction.csv",          # was Q22 (self-reference)
    "0029": "Q0029_limit.csv",                  # was R12
    "0030": "Q0030_limit.csv",                  # was R13
    "0031": "Q0031_limit.csv",                  # was R14
    "0032": "Q0032_tokenization.csv",           # was R18
    "0033": "Q0033_validation.csv",             # was Q41
    "0034": "Q0034_layer_depth.csv",            # was Q36
    "0035": "Q0035_output_similarity.csv",      # was Q39
    "0036": "Q0036_condition_transfer.csv",     # was Q38
    "0037": "Q0037_entropy_shape.csv",          # was Q37
    "0038": "Q0038_hesitation.csv",             # was Q35
    "0039": "Q0039_jolt.csv",                   # was R10
    "0040": "Q0040_jolt.csv",                   # was R11
    "0041": None,                               # was Q45 — POOL_DIM sweep (no GPU)
    "0042": None,                               # was Q33 — decomposition (no GPU)
    "0043": None,                               # was Q34 — Permutation Sensitivity (no GPU)
    "0044": None,                               # was Q46 — per-condition R (no GPU)
    "0045": None,                               # was Q49 — fixed-dim per-cond R (no GPU)
    "0046": None,                               # was Q56 — MLP decomposition (no GPU)
    "0047": None,                               # was Q25 — Granger A (no GPU)
    "0048": None,                               # was Q27 — Granger B (no GPU)
    "0049": None,                               # was Q32 — baseline swap (no GPU)
    "0050": None,                               # was Q47 — cross-temp synthesis (no GPU)
    "0051": None,                               # was Q40 — pooled decomposition (no GPU)
    "0052": None,                               # was Q50 — cross-model R compare (no GPU)
    "0053": None,                               # was Q51 — condition concordance (no GPU)
    "0054": None,                               # was Q52 — cross-model paper summary (no GPU)
    "0055": None,                               # was R54 — stats export (no GPU)
    "0056": None,                               # v0.80.0.38 (was R55→paper assembly): now CALIBRATION ONLY (V5b synthetic, channel marginal, methodology_calibration_block)
    "0057": None,                               # v0.80.0.38 (new): function-class sensitivity (Ridge vs MLP vs RF on §5.4 cross-cell pattern; no GPU)
    "0058": None,                               # v0.80.0.51 (new): lagrangian apparatus — kraskov anchor, solver, V5d threshold, aggregator (no GPU)
    "0059": None,                               # v0.80.0.51 (was 0058): stats export + paper assembly (results.json + figures)
})

ANALYSIS_JSON = DualKeyRunDict({
    "0041": "Q0041_pool_dim_sweep.json",             # was Q45
    "0042": "Q0042_decomposition.json",              # was Q33
    "0043": "Q0043_sobol_partition.json",            # was Q34
    "0044": "Q0044_per_condition_R.json",            # was Q46
    "0045": "Q0045_fixed_dim_per_condition_R.json",  # was Q49
    "0046": "Q0046_mlp_decomposition.json",          # was Q56
    "0047": "Q0047_granger_A.json",                  # was Q25
    "0048": "Q0048_granger_B.json",                  # was Q27
    "0049": "Q0049_baseline_swap.json",              # was Q32
    "0050": "Q0050_cross_temp_synthesis.json",       # was Q47
    "0051": "Q0051_pooled_sobol.json",               # was Q40
    "0052": "Q0052_cross_model_summary.json",        # was Q50 / Q52 (aliased)
    "0053": "Q0053_condition_concordance.json",      # was Q51
    "0054": "Q0054_cross_model_summary.json",        # was Q52 proper
    "0055": "Q0055_stats_report.json",               # was R54
    "0056": "Q0056_calibration_manifest.json",       # v0.80.0.38: Run 0056 is calibration-only; manifest summarizes which calibration scripts ran fresh
    "0057": "Q0057_function_class_sensitivity.json", # v0.80.0.38: function-class sensitivity (RF beyond Ridge/MLP)
    "0058": "Q0058_apparatus_manifest.json",         # v0.80.0.51 (new): lagrangian apparatus manifest (kraskov anchor, solver, threshold, aggregator)
    "0059": "results.json",                          # v0.80.0.51 (was 0058): paper assembly
})


def _find_root():
    """Walk up from this file until we find the iota root (contains start_here.py)."""
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(candidate, "start_here.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.dirname(os.path.abspath(__file__))


ROOT     = _find_root()
DATA     = os.path.join(ROOT, "data")
ENV_FILE = os.path.join(ROOT, ".iota_env.json")


# ─────────────────────────────────────────────
# ACTIVE QUANTIZATION  (v0.75.4.0)
# ─────────────────────────────────────────────
# Set once at session start. All path functions use this to build
# {size}_{quant} directory names. Prevents Q4/FP16 data collision.

_ACTIVE_QUANT = '4bit'


def set_active_quant(quant):
    """Set the active quantization for all path construction.
    Called at session start from start_here.py and export_flask.py."""
    global _ACTIVE_QUANT
    _ACTIVE_QUANT = quant or '4bit'


def _size_dir(size, quant=None):
    """Build the size directory name: {size}_{quant}.
    Skips append if size already has a quant suffix (from disk scan)."""
    q = quant or _ACTIVE_QUANT
    for qv in ('4bit', '8bit', 'fp16', 'fp32'):
        if size.endswith('_' + qv):
            return size
    return f"{size}_{q}"


def logical_size(size_dir_name):
    """Strip quant suffix from a directory name to get the logical model size.
    '2b_4bit' → '2b', '9b_fp16' → '9b', '8b' → '8b'."""
    for qv in ('4bit', '8bit', 'fp16', 'fp32'):
        if size_dir_name.endswith('_' + qv):
            return size_dir_name[:-(len(qv) + 1)]
    return size_dir_name


def quant_from_dir(size_dir_name):
    """Extract quant from a directory name. '2b_4bit' → '4bit', '8b' → None."""
    for qv in ('4bit', '8bit', 'fp16', 'fp32'):
        if size_dir_name.endswith('_' + qv):
            return qv
    return None


# ─────────────────────────────────────────────
# ENVIRONMENT FILE  (v30.0)
# ─────────────────────────────────────────────

def load_env() -> dict:
    """Return .iota_env.json contents or empty dict if not present."""
    if os.path.exists(ENV_FILE):
        try:
            return json.load(open(ENV_FILE))
        except Exception:
            pass
    return {}


def get_vram_gb() -> float:
    """Return total VRAM in GB from env file. 0.0 if not recorded."""
    return float(load_env().get("vram_gb", 0.0))


# ─────────────────────────────────────────────
# QUEUE FILE  (v30.0)
# ─────────────────────────────────────────────

def get_family_size_dir(family: str, size: str) -> str:
    """Return path to the family/size_quant directory (shared across variants)."""
    return os.path.join(DATA, family, _size_dir(size))


def get_queue_path(family: str, size: str) -> str:
    """Return path to queue.json for a given model family/size."""
    return os.path.join(get_family_size_dir(family, size), "queue.json")




def scan_run_completion(family: str, size: str, variant: str,
                        temperature: float, n_trials: int = 100) -> dict:
    """
    Scan CSV files to determine actual completion state per run.

    Returns {run_num: 'complete' | 'partial' | 'missing'} for runs 0004-0026.
    Uses the same max(trial)+1 >= n_trials check as _scan_runs in start_here.py
    and get_next_trial in orchestration_core.py.

    LIMITATION: analysis runs (25, 27, 32-34, 40) produce JSON not CSV; this
    function marks them 'missing' unless a CSV happens to exist. Multi-condition
    runs (26, 28-31, 37-39, 41) are checked via simple max(trial)+1 which may
    over-report 'complete' if one condition finishes before others. For accurate
    completion status use _scan_runs() in start_here.py (which handles both
    cases correctly). This function is retained for external callers.

    v33.3: extended from runs 0004-0028 to 1-42 (was silently missing all Phase 2/3).
    v36.1: extended from runs 0004-0018 to 1-44 (Runs 0025/0026 added in v35.0, scan not updated).
    """
    import pandas as pd
    paths  = get_paths(family, size, variant, temperature, create_dirs=False)
    result = {}
    for run_num in range(1, 57):  # v0.79.4.0: widened to include Run 0016 (E_t meta) + 49-56
        rid = run_id_pad(run_num)
        pattern = os.path.join(paths["csv"], f"run{rid}_*.csv")
        matches = glob.glob(pattern)
        if not matches:
            # Also try R/Q prefix naming
            pfx      = "R" if run_num <= 21 else "Q"
            pattern2 = os.path.join(paths["csv"], f"{pfx}{rid}_*.csv")
            matches  = glob.glob(pattern2)
        if not matches:
            result[rid] = "missing"
            continue
        try:
            dfs = []
            for m in matches:
                try:
                    dfs.append(pd.read_csv(
                        m, dtype=str, keep_default_na=False,
                        usecols=lambda c: c in ('run_mode', 'trial', 'priming')
                    ))
                except Exception:
                    pass
            if not dfs:
                result[rid] = "missing"
                continue
            df = pd.concat(dfs, ignore_index=True)
            df = df.replace("NA", float('nan'))
            if 'priming' in df.columns:
                df = df[df['priming'].astype(str) != '1']
            if 'run_mode' in df.columns:
                # v0.79.4.0: CSV run_mode column now carries 4-digit string.
                # Accept both 4-digit string AND legacy integer string for
                # transitional read-compat before data migration completes.
                df = df[df['run_mode'].astype(str).isin([rid, str(run_num)])]
            if df.empty or 'trial' not in df.columns:
                result[rid] = "missing"
            elif int(df['trial'].max()) + 1 >= n_trials:
                result[rid] = "complete"
            else:
                result[rid] = "partial"
        except Exception:
            result[rid] = "missing"
    return result


# ─────────────────────────────────────────────
# CONDITION / POOLED PATHS
# ─────────────────────────────────────────────

def condition_name(temperature: float) -> str:
    """Map a temperature float to its directory name.
    T=0.0 -> 'deterministic' (historical — greedy decode has no
    temperature semantically). T>0.0 -> 'temp_{T}' with one-decimal
    rounding, matching VALID_TEMPS precision. All path-building goes
    through this — single source of truth."""
    if temperature <= 0.0:
        return "deterministic"
    return f"temp_{round(float(temperature), 1)}"


def get_paper_paths(create_dirs: bool = True) -> dict:
    """Return paths for the global paper directory at base/paper/.
    Cross-model figures and summary JSONs live here — not per-model.

    v0.80.0.32: visuals/ and json/ subdirectories no longer auto-created.
    The new pipeline writes figures + results.json directly to base/paper/
    (the root); subdirs were vestigial holdovers from the pre-0.80
    paper-assembly path that copied per-model JSONs into json/ and
    rendered legacy figures into visuals/. Callers that explicitly
    expect those paths still receive them in the return dict (so existing
    code doesn't crash on KeyError); they just no longer get pre-created.
    Calibration subdirectory under base/paper/calibration/ is unaffected
    — it's created on demand by the calibration scripts themselves.
    """
    base = os.path.join(DATA, 'paper')
    paths = {
        "root":    ROOT,
        "data":    DATA,
        "base":    base,
        "visuals": os.path.join(base, "visuals"),  # legacy key — not created
        "json":    os.path.join(base, "json"),     # legacy key — not created
    }
    if create_dirs:
        os.makedirs(base, exist_ok=True)
    return paths


def get_pooled_paths(family: str, size: str, variant: str,
                     create_dirs: bool = True) -> dict:
    """Return paths for the pooled analysis directory.
    Lives at base/{family}/{size_quant}/{variant}/pooled/ — outside any temperature
    condition directory so it is never confused with per-round data.
    Used by Run 0051 (pooled cross-directory decomposition).
    """
    base  = os.path.join(DATA, family, _size_dir(size), variant, 'pooled')
    paths = {
        "root":      ROOT,
        "data":      DATA,
        "condition": "pooled",
        "base":      base,
        "analysis":  os.path.join(base, "analysis"),
        "visuals":   os.path.join(base, "visuals"),
    }
    if create_dirs:
        for d in [paths["analysis"], paths["visuals"]]:
            os.makedirs(d, exist_ok=True)
    return paths


def get_paths(family: str, size: str, variant: str, temperature: float,
              create_dirs: bool = True) -> dict:
    """Return the canonical path dict for one (family, size, variant, temp) cell.

    This is the one function every runner, analyzer, and dashboard
    endpoint goes through to locate data. Hard-coded path joins
    elsewhere are a bug.

    Returns dict with keys: root, data, condition, base, csv, hidden,
    analysis, visuals, calibration, temp_indep_csv, temp_indep_hidden,
    label.

    v0.76.1.0: temp_indep_* point at deterministic/ regardless of
    session temperature. Runs in TEMP_INDEP_RUNS (1, 2, 3) always
    read/write those paths — saves ~5x disk on those three runs which
    would otherwise duplicate identical data across all 6 temp dirs.

    create_dirs=True (default) makes all directories that don't exist.
    Pass False for read-only / discovery scans where missing dirs are
    meaningful signal.
    """
    cond  = condition_name(temperature)
    sd    = _size_dir(size)
    base  = os.path.join(DATA, family, sd, variant, cond)
    # v0.76.1.0: temperature-independent runs (R0001, R0002, R0003) always read/write
    # the deterministic directory. Adds temp_indep_csv and temp_indep_hidden
    # keys that point at deterministic/ regardless of session temperature.
    # Saves ~5× disk space on R0001/R0002/R0003 files.
    det_base = os.path.join(DATA, family, sd, variant, 'deterministic')
    paths = {
        "root":             ROOT,
        "data":             DATA,
        "condition":        cond,
        "base":             base,
        "csv":              os.path.join(base, "csv"),
        "hidden":           os.path.join(base, "hidden_states"),
        "analysis":         os.path.join(base, "analysis"),
        "visuals":          os.path.join(base, "visuals"),
        "calibration":      os.path.join(DATA, family, sd, variant),
        "temp_indep_csv":    os.path.join(det_base, "csv"),
        "temp_indep_hidden": os.path.join(det_base, "hidden_states"),
        "label":            describe(family, size, variant, temperature),
    }
    if create_dirs:
        for d in [paths["csv"], paths["hidden"], paths["analysis"], paths["visuals"],
                  paths["temp_indep_csv"], paths["temp_indep_hidden"]]:
            os.makedirs(d, exist_ok=True)
    return paths


# v0.76.1.0: Runs whose data is temperature-independent. Writes always go to
# paths['temp_indep_csv'] / paths['temp_indep_hidden'] (deterministic dir).
# Reads also check there. Eliminates 5× duplication across temperature rounds.
TEMP_INDEP_RUNS = DualKeyRunSet({1, 2, 3})  # was {19, 20, 26}





def sanitize(name: str) -> str:
    """Sanitize a string for filenames — replaces anything outside
    [A-Za-z0-9._-] with underscore. Used for model names (which
    contain /) before embedding them in .npy / .csv filenames."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name)


def describe(family, size, variant, temperature) -> str:
    """Human-readable 'family/size/variant/condition' label. Used in
    UI headers, log messages, error text. Not a path — for display."""
    return f"{family}/{size}/{variant}/{condition_name(temperature)}"



def run_prefix(run_num) -> str:
    """Return 'R' for proof runs (0001-0021) or 'Q' for quantify runs
    (0022-0056). Historical naming convention retained for human
    readability of on-disk files. Run 0017 (activation patching) is the
    final proof run before the convention switches.
    
    v0.79.4.0: accepts either the integer run number (legacy callers) or
    the 4-digit string run_id (preferred). Internal threshold is numeric
    so both paths reach the same answer.
    """
    n = run_id_to_int(run_num) if isinstance(run_num, str) else run_num
    return "R" if n <= 21 else "Q"


# ─────────────────────────────────────────────
# EMBEDDING / HIDDEN STATE I/O
# ─────────────────────────────────────────────

def embedding_path(hidden_dir: str, run_num, model_name: str,
                   trial: int, turn: int, kind: str = 'E') -> str:
    """Build the canonical filename path for an embedding/constraint vector.

    Four kinds of files live in hidden_states/:

      kind='E'       -> R{NNNN}_{model}_trial{TTTT}_turn{TT}_emb.npy
      kind='E_base'  -> R{NNNN}_{model}_trial{TTTT}_turn{TT}_et_base.npy
      kind='C'       -> R{NNNN}_{model}_trial{TTTT}_constraint.npy
      kind= (other)  -> R{NNNN}_{model}_trial{TTTT}_turn{TT}.npy

    v0.79.4.0: run_num identifier is now 4 digits. Accepts either integer
    (legacy) or 4-digit string (preferred). Pre-0.79.4 on-disk files using
    R{NN} must be migrated via migrate_run_ids.py.
    """
    mn  = sanitize(model_name)
    pfx = run_prefix(run_num)
    rid = run_id_pad(run_num)
    if kind == 'E_base':
        fname = f"{pfx}{rid}_{mn}_trial{trial:04d}_turn{turn:02d}_et_base.npy"
    elif kind == 'E':
        fname = f"{pfx}{rid}_{mn}_trial{trial:04d}_turn{turn:02d}_emb.npy"
    elif kind == 'C':
        fname = f"{pfx}{rid}_{mn}_trial{trial:04d}_constraint.npy"
    else:
        fname = f"{pfx}{rid}_{mn}_trial{trial:04d}_turn{turn:02d}.npy"
    return os.path.join(hidden_dir, fname)


def save_embedding(vec, hidden_dir: str, run_num, model_name: str,
                   trial: int, turn: int, kind: str = 'E'):
    """np.save an embedding vector to its canonical path. Casts to float32."""
    path = embedding_path(hidden_dir, run_num, model_name, trial, turn, kind)
    np.save(path, vec.astype(np.float32))


def load_embedding(hidden_dir: str, run_num, model_name: str,
                   trial: int, turn: int, kind: str = 'E'):
    """np.load an embedding from its canonical path, or None if absent."""
    path = embedding_path(hidden_dir, run_num, model_name, trial, turn, kind)
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            return None
    return None


def load_hidden_states(run_num, model_name: str, hidden_dir: str,
                       trial: int = None, turn: int = None, all_layers=False):
    """Load final-layer S_t hidden state(s) by glob pattern.

    File naming: R{NNNN}_{model}_trial{TTTT}_turn{TT}.npy (v0.79.4.0).
    Accepts int or string run_num."""
    mn      = sanitize(model_name)
    pfx     = run_prefix(run_num)
    rid     = run_id_pad(run_num)
    pattern = f"{pfx}{rid}_{mn}_trial"
    if trial is not None:
        pattern += f"{trial:04d}_turn"
        pattern += f"{turn:02d}" if turn is not None else "*"
    else:
        pattern += "*_turn*"
    if all_layers:
        pattern += "_alllayers"
    pattern += ".npy"

    files = sorted(glob.glob(os.path.join(hidden_dir, pattern)))
    if not files:
        return [] if trial is None else None
    if trial is not None and turn is not None:
        try:
            arr = np.load(files[0])
            if all_layers and arr.ndim == 2:
                # Return full layer stack as list of per-layer vectors.
                # Previously returned only arr[-1] — a scalar-shaped failure
                # that corrupted graft_patching patch_vecs (Bug M, v23.3).
                return [arr[i] for i in range(arr.shape[0])]
            return arr[-1] if arr.ndim == 2 else arr
        except Exception:
            return None
    vecs = []
    for f in files:
        try:
            arr = np.load(f)
            vecs.append(arr[-1] if arr.ndim == 2 else arr)
        except Exception:
            continue
    return vecs


def find_existing_trials(csv_path_str: str, run_num: int) -> int:
    """Return max(trial)+1 for completed rows of run_num in the CSV."""
    import csv as _csv, io as _io
    if not os.path.exists(csv_path_str):
        return 0
    try:
        with open(csv_path_str, 'rb') as fh:
            raw = fh.read()
        content = raw.decode('utf-8', errors='replace')
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        rows = list(_csv.reader(_io.StringIO(content)))
        if len(rows) < 2:
            return 0
        header = rows[0]
        try:
            rm_idx    = header.index('run_mode')
            pr_idx    = header.index('priming')
            trial_idx = header.index('trial')
        except ValueError:
            return 0
        max_trial = -1
        for row in rows[1:]:
            if len(row) <= max(rm_idx, pr_idx, trial_idx):
                continue
            if row[pr_idx] == '1':
                continue
            if not run_mode_matches(row[rm_idx], run_num):
                continue
            try:
                t = int(row[trial_idx])
                if t > max_trial:
                    max_trial = t
            except (ValueError, TypeError):
                pass
        return max_trial + 1 if max_trial >= 0 else 0
    except Exception:
        return 0
