"""
IOTA FRAMEWORK v1.0.0
======================
    python start_here.py

Files in this directory:
  start_here.py             ← you are here
  runners.py               ← dispatch hub (routes run numbers to phase modules)
  runners_prompts.py       ← all prompt constants
  runners_core.py          ← shared infrastructure + E_t recovery
  runners_p1.py            ← Phase 1 runs
  runners_p2.py            ← Phase 2 runs
  runners_p3.py            ← Phase 3+4 runs
  analysis.py               ← all no-GPU analysis (run 41-51)
  graft_patching.py         ← run 0017 (activation patching -- mechanically unique)
  run42_layer_isolation.py  ← run 0018 (layer causal sufficiency -- single-layer patching)
  orchestration_core.py     ← shared model/generation infrastructure
  orchestration_throughlines.py  ← system prompts and turn-0 injections
  cartography.py            ← path resolution and file I/O
  ui.py                     ← console interface
  vault.py                  ← model registry
  export_stats.py           ← figures + statistical output
  export_flask.py           ← live dashboard (localhost:5000)

══════════════════════════════════════════════════════════════
  RUN ORDER (post-renumber, v0.79.4.0+)
══════════════════════════════════════════════════════════════

  PHASE A -- temp-indep foundation (3 runs)

    1   null baseline + hidden   (H42 hidden state extraction)  [was R19]
    2   robustness sweep         (H41 metric convergence)       [was R20]
    3   temperature grid 4×5     (H13)                          [was Q26]

  PHASE B -- fast ET-source block (12 runs)

    4   null baseline A          (H01 floor)                    [was R01]
    5   null baseline B          (H01 ceiling)                  [was R02]
    6   introspection A          (H03 direct self-query)        [was R03]
    7   introspection B          (H03 memory probe)             [was R04]
    8   introspection C          (H03 fixed output enforcer)    [was R05]
    9   arithmetic A             (H09 visible -- show your work) [was R06]
    10  arithmetic B             (H09 hidden -- answer only)     [was R07]
    11  arithmetic C             (H09 cold)                     [was R08]
    12  arithmetic D             (H09 primed)                   [was R09]
    13  priming neutral          (H08)                          [was R15]
    14  priming cooperative      (H08)                          [was R16]
    15  priming resistant        (H08)                          [was R17]

  PHASE G -- E_t recovery meta-run

    16  E_t recovery -- base-model pass                          [was R48]

  PHASE D -- patching cluster

    17  activation patching      (H38)                          [was R21]
    18  layer causal sufficiency (H29)                          [was Q42]
    19  random noise patching baseline (H40 control)            [was R53]

  PHASE EFC -- slow→fast collection (21 runs, 20-40)

    20  two-instance cross-instance              (H19)          [was Q30]
    21  coherence levels                         (H20)          [was Q31]
    22  context saturation                       (H06)          [was Q23]
    23  C_t confound isolation                   (H17)          [was Q28]
    24  persistence mechanism                    (H18)          [was Q29]
    25-27, 28, 29-31, 32, 33-40 (see RUN_MAP)

  PER-TEMP ANALYSIS (no GPU, runs 41-49)

    41  POOL_DIM stability sweep                 [was Q45]
    42  E+C+R full decomposition                 [was Q33]
    43  Permutation Sensitivity partition        [was Q34]
    44  Per-condition R fractions                [was Q46]
    45  Fixed-dim per-condition R                [was Q49]
    46  MLP permutation sensitivity              [was Q56]
    47  Granger probe A                          [was Q25]
    48  Granger probe B                          [was Q27]
    49  Baseline swap                            [was Q32]

  POOLED ANALYSIS (no GPU, runs 50-51)

    50  Cross-temperature synthesis              [was Q47]
    51  Pooled E+C+R + Permutation Sensitivity   [was Q40]

  CROSS-MODEL ANALYSIS (no GPU, runs 52-54)

    52  Cross-model R comparison                 [was Q50]
    53  Cross-model condition concordance        [was Q51]
    54  Cross-model paper summary table          [was Q52]

  PAPER OUTPUT (no GPU, runs 55-56)

    55  Stats export + report generation         [was R54]
    56  Cross-model paper assembly               [was R55]

══════════════════════════════════════════════════════════════
"""

import os, sys, importlib.util, time, threading

# v0.82.0.26: Force UTF-8 stdout/stderr on Windows. Default cp1252 console
# encoding mangles unicode characters in print strings (em-dashes, lambda,
# bullet markers) AND in tqdm progress bars (block-element chars from
# transformers' weight-loading bar). The garble looks like errors. It isn't.
# This fixes the source so output is readable regardless of how the
# apparatus is invoked (UI, --single-run, --all-models, direct python).
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# v0.82.0.26: kill transformers/safetensors weight-loading progress bars.
# They emit hundreds of in-place updates per model load with block-element
# unicode chars that mangle on cp1252 console even with chcp 65001 set.
# Apparatus's own trial-progress messages use plain print() calls, not
# tqdm, so they're unaffected.
os.environ.setdefault('TQDM_DISABLE', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

# v0.82.0.26: silence library deprecation/user warnings that go to stderr.
# PowerShell paints stderr lines red regardless of content, making harmless
# pandas FutureWarning + torch UserWarning + transformers DeprecationWarning
# all look like errors. Apparatus errors still go through ui.err() to stdout
# and remain visible.
import warnings
warnings.filterwarnings('ignore')

def _find_root():
    """Walk up from this file's directory to the iota root (contains
    start_here.py). Lets the framework run from any working directory
    without import-path assumptions."""
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
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

import ui
from cartography import get_paths, DualKeyRunDict, run_id_pad  # v0.79.4.18: wrap RUN_MAP + fix _order_runs key normalization

DASHBOARD_URL = "http://localhost:5000"


# ── Dashboard helpers ─────────────────────────────────────────────────────────

def _is_dashboard_running():
    """Check if something is already listening on port 5000."""
    import socket
    try:
        s = socket.create_connection(('localhost', 5000), timeout=0.5)
        s.close()
        return True
    except Exception:
        return False


def _kill_stale_dashboard():
    """Kill any existing Flask process on port 5000 so we can launch fresh code."""
    pid_file = os.path.join(ROOT, '.iota_flask.pid')
    if os.path.exists(pid_file):
        try:
            with open(pid_file) as f:
                old_pid = int(f.read().strip())
            import signal
            if sys.platform == 'win32':
                import subprocess
                subprocess.run(['taskkill', '/F', '/PID', str(old_pid)],
                               capture_output=True, timeout=5)
            else:
                os.kill(old_pid, signal.SIGTERM)
            time.sleep(0.5)
        except Exception:
            pass
        try:
            os.remove(pid_file)
        except Exception:
            pass


def _launch_dashboard():
    """Spawn Flask dashboard as a daemon subprocess.
    Output is redirected to .iota_flask.log -- does not pollute the console.
    Kills any stale Flask process first to ensure fresh code is served.
    Returns the Popen handle or None.
    """
    import subprocess
    _kill_stale_dashboard()
    # Brief wait for port to free
    if _is_dashboard_running():
        time.sleep(1)
    flask_path = os.path.join(ROOT, 'export_flask.py')
    if not os.path.exists(flask_path):
        return None
    log_path = os.path.join(ROOT, '.iota_flask.log')
    pid_file = os.path.join(ROOT, '.iota_flask.pid')
    try:
        proc = subprocess.Popen(
            [sys.executable, flask_path, '--daemon'],
            stdout=open(log_path, 'a'),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Save PID for next launch to kill
        with open(pid_file, 'w') as f:
            f.write(str(proc.pid))
        return proc
    except Exception as e:
        ui.warn(f"Dashboard launch failed: {e}")
        return None


def _maybe_open_browser(countdown=0):
    """Open the dashboard in a browser immediately (countdown=0) or after a
    short delay.  If countdown > 0 the user can press Enter to stay in
    console mode; at 0 the browser opens straight away with no prompt.
    Either way the dashboard server is already running at localhost:5000.
    """
    import webbrowser
    print("\n" + "=" * 48)
    print("  IOTA Dashboard")
    print("  " + DASHBOARD_URL)
    print("=" * 48)
    if countdown <= 0:
        webbrowser.open(DASHBOARD_URL)
        return

    print(f"  Opening browser in {countdown}s  -- press Enter to skip", flush=True)

    skipped   = threading.Event()
    opened    = threading.Event()

    def _watch_enter():
        try:
            sys.stdin.readline()
        except Exception:
            pass
        skipped.set()

    watcher = threading.Thread(target=_watch_enter, daemon=True)
    watcher.start()

    for remaining in range(countdown, 0, -1):
        if skipped.is_set():
            pass  # console mode -- URL already printed above
            return
        time.sleep(1)

    if not skipped.is_set():
        webbrowser.open(DASHBOARD_URL)
        ui.ok("Browser opened.")

# ── Run map ───────────────────────────────────────────────────────────────────

RUN_MAP = DualKeyRunDict({
    # v0.79.4.18: wrapped in DualKeyRunDict. Prior plain-dict form with
    # 4-digit string keys meant `int_r in RUN_MAP` always returned False,
    # breaking _parse_runs (which parses to int) and every downstream
    # caller doing integer lookups against the post-renumber string keys.
    # Same migration pattern already applied to RUN_CSV / ANALYSIS_JSON
    # in cartography.py -- this closes the third instance.
    # v0.79.4.15: renumbered to execution-order positions (1-56). "was"
    # comments reference pre-renumber IDs for migration cross-ref.
    #
    # ── PHASE A -- temp-indep foundation ──
    "0001": ("runners",        "Null baseline + hidden state extraction"),  # was R19
    "0002": ("runners",        "Robustness sweep"),                         # was R20
    "0003": ("runners",        "Temperature grid 4×5"),                     # was Q26
    # ── PHASE B -- fast ET-source block ──
    "0004": ("runners",        "Null baseline A"),                          # was R01
    "0005": ("runners",        "Null baseline B"),                          # was R02
    "0006": ("runners",        "Introspection A -- direct self-query"),      # was R03
    "0007": ("runners",        "Introspection B -- memory probe"),           # was R04
    "0008": ("runners",        "Introspection C -- fixed output enforcer"),  # was R05
    "0009": ("runners",        "Arithmetic A -- visible"),                   # was R06
    "0010": ("runners",        "Arithmetic B -- hidden"),                    # was R07
    "0011": ("runners",        "Arithmetic C -- cold"),                      # was R08
    "0012": ("runners",        "Arithmetic D -- primed"),                    # was R09
    "0013": ("runners",        "Priming -- neutral"),                        # was R15
    "0014": ("runners",        "Priming -- cooperative"),                    # was R16
    "0015": ("runners",        "Priming -- resistant"),                      # was R17
    # ── PHASE G -- E_t recovery meta-run ──
    "0016": ("runners",        "E_t recovery -- base-model pass (meta-run)"), # was R48 -- v0.79.4.16: runners_core has no run(), dispatch lives in runners.run()
    # ── PHASE D -- patching cluster ──
    "0017": ("graft_patching", "Activation patching"),                       # was R21
    "0018": ("run42_layer_isolation", "Layer causal sufficiency -- single-layer patching"),  # was Q42
    "0019": ("graft_patching", "Random noise patching baseline (H40 control)"),  # was R53
    # ── PHASE EFC -- slow→fast collection ──
    "0020": ("runners",        "Two-instance cross-instance measurement"),   # was Q30
    "0021": ("runners",        "Coherence levels -- three R conditions"),     # was Q31
    "0022": ("runners",        "Context saturation"),                        # was Q23
    "0023": ("runners",        "C_t confound isolation"),                    # was Q28
    "0024": ("runners",        "Persistence mechanism"),                     # was Q29
    "0025": ("runners",        "Coherence transfer -- priming length probe"), # was Q43
    "0026": ("runners",        "Contradiction -- priming then recovery"),     # was Q44
    "0027": ("runners",        "Layer locality"),                            # was Q24
    "0028": ("runners",        "Self-reference probe"),                      # was Q22
    "0029": ("runners",        "Impossibility A -- constrained"),             # was R12
    "0030": ("runners",        "Impossibility B -- unconstrained"),           # was R13
    "0031": ("runners",        "Impossibility C -- epistemic"),               # was R14
    "0032": ("runners",        "Tokenization control"),                      # was R18
    "0033": ("runners",        "Held-out validation set for decomposition"), # was Q41
    "0034": ("runners",        "Layer depth analysis -- where does R live?"), # was Q36
    "0035": ("runners",        "Output self-similarity across turns"),       # was Q39
    "0036": ("runners",        "Condition transfer -- introspection → arithmetic"),  # was Q38
    "0037": ("runners",        "Within-turn entropy shape"),                 # was Q37
    "0038": ("runners",        "Hesitation probe -- first-token latency vs R"),  # was Q35
    "0039": ("runners",        "Perturbation A -- late shock"),               # was R10
    "0040": ("runners",        "Perturbation B -- early shock"),              # was R11
    # ── PER-TEMP ANALYSIS (chain: POOL_DIM → decomp → partition → R fracs) ──
    "0041": ("analysis",       "POOL_DIM stability sweep -- no GPU"),         # was Q45
    "0042": ("analysis",       "E+C+R full decomposition -- no GPU"),         # was Q33
    "0043": ("analysis",       "Permutation Sensitivity partition -- no GPU"), # was Q34
    "0044": ("analysis",       "Per-condition R fractions -- no GPU"),        # was Q46
    "0045": ("analysis",       "Fixed-dim per-condition R (POOL_DIM=64) -- no GPU"),  # was Q49
    "0046": ("analysis",       "MLP permutation sensitivity -- Ridge 3×3 -- no GPU"),  # was Q56
    # ── PER-TEMP ANALYSIS (independents) ──
    "0047": ("analysis",       "Granger probe A -- no GPU"),                  # was Q25
    "0048": ("analysis",       "Granger probe B -- no GPU"),                  # was Q27
    "0049": ("analysis",       "Baseline swap -- no GPU"),                    # was Q32
    # ── POOLED ANALYSIS ──
    "0050": ("analysis",       "Cross-temperature synthesis -- no GPU"),      # was Q47
    "0051": ("analysis",       "Pooled E+C+R decomposition + Permutation Sensitivity -- no GPU"),  # was Q40
    # ── CROSS-MODEL ANALYSIS ──
    "0052": ("analysis",       "Cross-model R comparison -- pairwise + pooled -- no GPU"),  # was Q50
    "0053": ("analysis",       "Cross-model condition ranking concordance -- no GPU"),     # was Q51
    "0054": ("analysis",       "Cross-model paper summary table (S5/S6) -- no GPU"),       # was Q52
    # ── OUTPUT ──
    "0055": ("export_stats",   "Stats export + report generation -- no GPU"), # was R54
    # v0.80.0.51: Run 0058 reassigned. Old Run 0058 (paper assembly)
    # is now Run 0059. Run 0058 is the lagrangian apparatus -- kraskov
    # anchor producer, I-projection solver, V5d threshold calibration,
    # aggregator. Apparatus hard-depends on 0056 (foundations) and
    # 0057 (function-class data); paper assembly (now 0059)
    # hard-depends on all three.
    #
    # Run cadence:
    #   0056 -- calibration only (foundations: V5, channel marginal, etc)
    #   0057 -- function-class sensitivity (Ridge vs MLP vs RF; was earlier)
    #   0058 -- lagrangian apparatus (NEW: anchor + solver + threshold + aggregator)
    #   0059 -- paper assembly (results.json + figures; was 0058)
    "0056": ("export_stats",   "Methodology calibration only -- V5b synthetic + channel marginal + manifest -- no GPU"),  # v0.80.0.44: split from old Run 0056
    "0057": ("export_stats",   "Function-class sensitivity -- Ridge vs MLP vs RF on §5.4 cross-cell pattern -- no GPU"),  # v0.80.0.44
    "0058": ("run_bayesian_apparatus",  "Bayesian apparatus -- kraskov anchor + I-projection solver + threshold + aggregator -- no GPU"),  # v0.80.0.51: new; v0.82.0.23: renamed lagrangian → bayesian (apparatus is Bayesian I-projection MAP, not constrained-Lagrangian)
    "0059": ("export_stats",   "Stats export + paper assembly -- results.json + figures -- no GPU"),  # v0.80.0.51: was 0058
    # v0.83 Run 0060 -- non-thematic recovery jolt (Paper A §7 follow-up).
    # Same protocol as R0039 (cache-cleared, transcript-threaded, 16 turns,
    # shock at turn 13, 4-cohort SHOCK_VARIANTS) but with turns 14-16 swapped
    # to NON_THEMATIC_RECOVERY_PROMPTS. Separates spontaneous-retention from
    # input-triggered-reactivation readings of cohort discrimination.
    "0060": ("runners",        "Perturbation A non-thematic recovery (§7 follow-up)"),
})

# ── Collection-time prerequisites ─────────────────────────────────────────────
#
# Maps each run that has upstream data dependencies to the set of runs whose
# data/output must exist on disk before the target run can produce valid results.
#
# ONLY collection-time dependencies are listed here. Analysis-time-only deps
# (e.g. Run 0033 depends on Runs 0042/0043 only when H28 is *evaluated* inside Run 0043
# Stage 2, not when Run 0033 is *collecting* its own data) are deliberately excluded.
#
# Cross-reference: dependency_map.py DEPENDENCY_MAP[...]["upstream_runs"] plus
# the per-module guards in graft_patching.py, run42_layer_isolation.py, analysis.py.
# This dict is the single authoritative pre-dispatch gate; the in-module guards
# remain as a secondary safety net inside each module.

_PREREQS = DualKeyRunDict({
    # v0.79.5.18: wrapped in DualKeyRunDict. Pre-0.79.5.18 this was a
    # plain dict with 4-digit string keys ("0017", "0042", etc.), but
    # every caller of _check_prereqs (line 1726, 1867, 2998) passes
    # run_num as an INT. _PREREQS.get(17) on a dict with "0017" keys
    # returns None, fell through to the default {} in _check_prereqs,
    # and no prereq was ever detected. The entire prerequisite
    # machinery has been silently non-functional since the 0.79.4.0
    # renumber. Wrapping in DualKeyRunDict (same fix as 0.79.4.18's
    # RUN_MAP) normalizes int ↔ 4-digit-string at lookup boundary.
    # Every existing entry below now enforces. Queued in HANDOFF as
    # deferred since 0.79.4.18 -- shipped at last.
    #
    # v0.79.4.15: renumbered to execution-order positions. 'was' comments
    # reference pre-renumber IDs for migration cross-ref.

    # Run 0016 -- E_t recovery meta-run. Per-(trial, turn) E_t
    # forward-pass output is meaningful only with Run 0001's global-
    # mean vectors for downstream normalization. Run 1 writes
    # R0001_{mn}_ct_global_mean.npy and R0001_{mn}_et_global_mean.npy
    # during null_trivariant (runners_p1.py:388-389). Without those,
    # Run 16's output is orphaned -- no baseline for Run 0042's
    # E+C+R decomposition. v0.79.5.18: added.
    "0016": {
        "0001": "Run 0001 -- ct_global_mean.npy + et_global_mean.npy baseline vectors required for E_t downstream normalization",
    },

    # Run 0017 -- Activation patching reads Run 0006 all-layers refs (was R21→R3)
    "0017": {
        "0006": "Run 0006 (Introspection A) -- all-layers .npy needed for patching",
    },

    # Run 0047 -- Granger A reads Run 0001 hidden .npy (was R25→R19)
    "0047": {
        "0001": "Run 0001 (Null + hidden) -- .npy needed for Granger A",
    },

    # Run 0048 -- Granger B reads Run 0003 temp-grid CSV (was R27→R26)
    "0048": {
        "0003": "Run 0003 (Temperature grid) -- grid data for Granger B",
    },

    # Run 0049 -- Baseline swap reads all Phase 1 CSVs (was R32, Runs 0004-0002)
    "0049": {
        "0004": "Run 0004 -- Phase 1 CSV required for baseline swap",
        "0005": "Run 0005 -- Phase 1 CSV required for baseline swap",
        "0006": "Run 0006 -- Phase 1 CSV required for baseline swap",
        "0007": "Run 0007 -- Phase 1 CSV required for baseline swap",
        "0008": "Run 0008 -- Phase 1 CSV required for baseline swap",
        "0009": "Run 0009 -- Phase 1 CSV required for baseline swap",
        "0010": "Run 0010 -- Phase 1 CSV required for baseline swap",
        "0011": "Run 0011 -- Phase 1 CSV required for baseline swap",
        "0012": "Run 0012 -- Phase 1 CSV required for baseline swap",
        "0039": "Run 0039 -- Phase 1 CSV required for baseline swap",
        "0040": "Run 0040 -- Phase 1 CSV required for baseline swap",
        "0029": "Run 0029 -- Phase 1 CSV required for baseline swap",
        "0030": "Run 0030 -- Phase 1 CSV required for baseline swap",
        "0031": "Run 0031 -- Phase 1 CSV required for baseline swap",
        "0013": "Run 0013 -- Phase 1 CSV required for baseline swap",
        "0014": "Run 0014 -- Phase 1 CSV required for baseline swap",
        "0015": "Run 0015 -- Phase 1 CSV required for baseline swap",
        "0032": "Run 0032 -- Phase 1 CSV required for baseline swap",
        "0001": "Run 0001 -- Phase 1 CSV required for baseline swap",
        "0002": "Run 0002 -- Phase 1 CSV required for baseline swap",
    },

    # Run 0042 -- E+C+R decomposition reads E_t+C_t from sources (was R33)
    "0042": {
        "0001": "Run 0001 -- E_t+C_t source for decomposition",
        "0002": "Run 0002 -- E_t+C_t source for decomposition",
        "0003": "Run 0003 -- E_t+C_t source for decomposition",
        "0004": "Run 0004 -- E_t+C_t source for decomposition",
        "0005": "Run 0005 -- E_t+C_t source for decomposition",
        "0006": "Run 0006 -- E_t+C_t source for decomposition",
        "0007": "Run 0007 -- E_t+C_t source for decomposition",
        "0008": "Run 0008 -- E_t+C_t source for decomposition",
        "0009": "Run 0009 -- E_t+C_t source for decomposition",
        "0010": "Run 0010 -- E_t+C_t source for decomposition",
        "0011": "Run 0011 -- E_t+C_t source for decomposition",
        "0012": "Run 0012 -- E_t+C_t source for decomposition",
        "0013": "Run 0013 -- E_t+C_t source for decomposition",
        "0014": "Run 0014 -- E_t+C_t source for decomposition",
        "0015": "Run 0015 -- E_t+C_t source for decomposition",
        "0016": "Run 0016 -- E_t+C_t source for decomposition",
        "0023": "Run 0023 -- E_t+C_t source for decomposition",
    },

    # Run 0043 -- Permutation sensitivity needs decomposition + validation (was R34)
    "0043": {
        "0042": "Run 0042 (decomposition) -- Ridge model required",
        "0033": "Run 0033 (held-out validation) -- Stage 2 validation CSV",
    },

    # Run 0051 -- Pooled decomposition needs Run 0042+34 outputs (was R40)
    "0051": {
        "0042": "Run 0042 (decomposition) -- required for pooled",
        "0043": "Run 0043 (permutation sensitivity) -- required for pooled",
    },

    # Run 0018 -- Layer isolation reads Run 0006 all-layers (was R42→R3)
    "0018": {
        "0006": "Run 0006 (Introspection A) -- all-layers .npy",
    },

    # Run 0019 -- Random patching reads Run 0006 all-layers (was R53→R3)
    "0019": {
        "0006": "Run 0006 (Introspection A) -- all-layers .npy",
    },

    # Run 0046 -- MLP perm sensitivity needs decomp/partition/per-cond (was R56)
    "0046": {
        "0042": "Run 0042 (decomposition) -- POOL_DIM + quadruplets",
        "0043": "Run 0043 (permutation sensitivity) -- pooled Ridge ref",
        "0044": "Run 0044 (per-condition R) -- per-condition Ridge ref",
    },

    # v0.80.0.51: Runs 0057, 0058, and 0059 do their own internal gating against
    # GLOBAL artifacts (channel_marginal_nonlinearity.csv,
    # Q0057_function_class_sensitivity.json, calibration manifest), not
    # per-cell paths. The _scan_runs / _check_prereqs machinery is per-cell
    # -- it walks one model+temp+variant directory and checks Q-files there.
    # Listing 0057/0058/0059 prereqs (0042/0044/0046/0056) here would route
    # global artifacts through the per-cell scanner, which sees Q0042 only
    # at the current scan path and reports global runs as 'missing' even
    # when the global artifacts exist.
    #
    # Both runs fail loud on their own checks if their global artifacts
    # are missing. This matches how the pre-split Run 0056 paper assembly
    # was gated (no _PREREQS entry; inline HARD GATE on calibration
    # status). Don't add 0057/0058/0059 entries here.
})
def _check_prereqs(run_num, status):
    """
    Check collection-time prerequisites for run_num against the live scan.

    Returns list of (prereq_num, prereq_status, description) for each unmet prereq.
    prereq_status is 'missing' or 'partial' -- both count as unmet.
    An empty list means all prerequisites are satisfied (or this run has none).

    v0.79.4.15: E_t state is now owned entirely by Run 0016 (first-class meta-
    run). Source runs (1-9, 15-17, 20) no longer emit 'needs_et'/'et_partial'
    statuses. Analysis runs that need E_t embeddings (33, etc.) now list
    Run 0016 in their _PREREQS explicitly instead of relying on the old
    acceptance shortcut.
    """
    prereqs = _PREREQS.get(run_num, {})
    problems = []
    for prereq_num, desc in prereqs.items():
        pstat = status.get(prereq_num, 'missing')
        if pstat == 'done':
            continue
        problems.append((prereq_num, pstat, desc))
    return sorted(problems, key=lambda x: x[0])


def _prereq_warning_block(run_num, problems):
    """Print the hard warning block for a run with unmet prerequisites."""
    W = 64
    _, rdesc = RUN_MAP.get(run_num, ('', f'Run {run_num}'))
    print()
    print('!' * W)
    ui.warn(f"PREREQUISITE WARNING  --  Run {run_num:04d}: {rdesc}")
    print('!' * W)
    for prereq_num, prereq_stat, desc in problems:
        label = "MISSING" if prereq_stat == 'missing' else "INCOMPLETE"
        ui.warn(f"  [{label:10}]  {desc}")
    print('!' * W)
    print()


# Runs that require a base-model E_t recovery pass (collected pre-v40.0.0).
# Subset of SOURCE_RUNS_2WAY ∪ SOURCE_RUNS_3WAY that were collected without
# true base-model E_t and need the recovery pass before feeding the OLS.
# v42.5.2: Run 0002 added -- SOURCE_RUNS_2WAY member, needs base E_t pass.
# v0.79.4.15: _ET_RECOVERY_TOTAL removed with Run 0016 promotion -- scanner
# derives expected counts from CSV row counts at scan time now.
from scanner import _ET_RECOVERY_RUNS


def _prompt_pass_cfg_r01(paths, n_trials):
    """Interactive toggle for Run 0001 pass selection.

    Shows current completion status for each of the four passes and lets the
    user toggle individual passes on/off before dispatch.

    Returns dict {'base': bool, 'instruct': bool, 'abliterated': bool, 'vectors': bool},
    or None if the user chooses to skip Run 0001 entirely.

    In headless mode (IOTA_HEADLESS=1) returns all-True immediately.
    Defaults to only the passes that are NOT yet complete -- saves time on partial runs.
    """
    if os.environ.get('IOTA_HEADLESS') == '1':
        return {'base': True, 'instruct': True, 'abliterated': True, 'vectors': True}

    import glob as _g
    hidden_dir = paths.get('hidden', '')
    csv_dir    = paths.get('csv', '')

    def _count_pass(patterns, variant=None):
        # v42.1.11: base/instruct dirs are siblings of abliterated under
        # the same family/size root. Derive path from hidden_dir structure.
        # Falls back to old v42.1.10 subdir for migration window compatibility.
        # Accepts a list of patterns or a single-pattern string.
        if isinstance(patterns, str):
            patterns = [patterns]
        if variant and variant != 'abliterated':
            try:
                _cond_dir    = os.path.dirname(hidden_dir)
                _variant_dir = os.path.dirname(_cond_dir)
                _size_dir    = os.path.dirname(_variant_dir)
                _cond_name   = os.path.basename(_cond_dir)
                search_dir   = os.path.join(_size_dir, variant, _cond_name, 'hidden_states')
            except Exception:
                search_dir = hidden_dir
            # Also check old v42.1.10 subdir location (migration window)
            fallback_dir = os.path.join(hidden_dir, variant)
        else:
            search_dir   = hidden_dir
            fallback_dir = None
        trials = set()
        for _sd in ([search_dir, fallback_dir] if fallback_dir else [search_dir]):
            for pat in patterns:
                for f in _g.glob(os.path.join(_sd, pat)):
                    try:
                        trials.add(int(os.path.basename(f).split('_trial')[1].split('_')[0]))
                    except Exception:
                        pass
        return len(trials)

    def _count_abl():
        try:
            from cartography import RUN_CSV
            csv_fname = RUN_CSV.get("0001")  # v0.79.4.0: was R19 null+hidden
            if not csv_fname:
                return 0
            fpath = os.path.join(csv_dir, csv_fname)
            cols = _read_csv_cols(fpath, ('run_mode', 'trial', 'priming'))
            trial_set = set()
            for i, t in enumerate(cols.get('trial', [])):
                if cols.get('priming') and cols['priming'][i] == '1':
                    continue
                try:
                    trial_set.add(int(t))
                except Exception:
                    pass
            return len(trial_set)
        except Exception:
            return 0

    n_base = _count_pass(['R0001_*_base_trial*_turn01.npy'], variant='base')
    n_inst = _count_pass(['R0001_*_instruct_trial*_turn01.npy'], variant='instruct')
    n_abl  = _count_abl()
    # pass4: requires the Pass 4 sentinel file written by v42.1.2+ _compute_run01_vectors.
    # The sentinel is ONLY written after a correct (BUG-42B-fixed) Pass 4 completes.
    # Corrupted BUG-42B runs have constraint.npy files on disk (correct count, wrong
    # content) -- file-count checks would falsely report pass4 done. The sentinel
    # cannot be faked by file count; it must be explicitly written by the fixed code.
    pass4 = len(_g.glob(os.path.join(hidden_dir, 'R0001_*_pass4_ok.stamp'))) > 0

    def _st(n, is_bool=False):
        if is_bool:
            return '✓ done' if n else '· not started'
        if n >= n_trials: return f'✓ {n}/{n_trials}'
        if n > 0:         return f'~ {n}/{n_trials}'
        return             f'· 0/{n_trials}'

    # Default: only enable passes that are not already complete
    cfg = {
        'base':        n_base < n_trials,
        'instruct':    n_inst < n_trials,
        'abliterated': n_abl  < n_trials,
        'vectors':     not pass4,
    }
    labels = {
        'base':        f"Pass 1 -- Base model hidden states      ({_st(n_base)})",
        'instruct':    f"Pass 2 -- Instruct model hidden states  ({_st(n_inst)})",
        'abliterated': f"Pass 3 -- Abliterated model + CSV       ({_st(n_abl)})",
        'vectors':     f"Pass 4 -- Compute E_t / C_t vectors     ({_st(pass4, is_bool=True)})",
    }
    keys     = ['base', 'instruct', 'abliterated', 'vectors']
    key_char = {'base': 'b', 'instruct': 'i', 'abliterated': 'a', 'vectors': 'v'}

    while True:
        ui.section("Run 0001 -- Pass Configuration")
        ui.blank()
        for k in keys:
            sym = 'x' if cfg[k] else ' '
            ui.msg(f"  [{sym}] {key_char[k]}  {labels[k]}")
        ui.blank()
        ui.msg("  b/i/a/v = toggle pass   A = all   N = none   Enter = confirm   s = skip Run 0001")
        ui.blank()
        raw = input("  > ").strip().lower()
        if raw == '':
            break
        elif raw == 's':
            return None
        elif raw == 'a':
            for k in keys: cfg[k] = True
        elif raw == 'n':
            for k in keys: cfg[k] = False
        elif raw in key_char.values():
            for k, c in key_char.items():
                if c == raw:
                    cfg[k] = not cfg[k]
        else:
            ui.warn(f"  Unknown input: '{raw}'")

    if not any(cfg.values()):
        ui.warn("  No passes selected -- skipping Run 0001.")
        return None
    return cfg


def _prompt_et_batch_cfg(run_nums, status, paths, n_trials):
    """Upfront batch configuration screen for ET_RECOVERY_RUNS in the queue.

    Replaces the old per-run _prompt_run_pass_choice + broken _prompt_et_run_cfg.
    Called once before the run loop whenever any ET_RECOVERY_RUNS are queued.

    Shows all queued ET runs with their current Abl CSV and E_t file counts.
    Each run has an assigned mode:
      [e] E_t base pass only  -- CSV and S_t intact, just add E_t files (default for needs_et)
      [a] Abliterated only    -- re-run model, overwrite CSV + S_t (default for missing)
      [b] Both                -- abliterated first, then E_t base pass (default for missing)
      [s] Skip                -- do nothing for this run

    Default assignment per status:
      'needs_et' → 'e'   (CSV complete, just need E_t)
      'missing'  → 'b'   (nothing collected yet -- collect everything)
      other      → 's'   (done or unexpected -- skip)

    Inputs:
      <N>       -- cycle run N through modes: e → a → b → s → e
      <N> <M>   -- set run N directly to mode M (e/a/b/s)
      E         -- set all to E_t only
      A         -- set all to abliterated only
      B         -- set all to both
      X         -- set all to skip
      Enter     -- confirm
      s         -- skip all (returns empty sets)

    Returns (abl_set, et_set):
      abl_set -- run numbers that need an abliterated generation pass (mode a or b)
      et_set  -- run numbers that need a base-model E_t pass (mode e or b)

    In headless mode: needs_et → et_set, missing → both sets. No prompt.
    """
    import glob as _g
    hidden_dir = paths.get('hidden', '')
    csv_dir    = paths.get('csv', '')
    run_list   = sorted(run_nums)

    # Assign default modes
    _MODE_CYCLE = ['e', 'a', 'b', 's']

    def _default_mode(rn):
        s = status.get(rn, 'missing')
        # v0.76.0.5: previously only 'needs_et' → 'e', 'missing' → 'b', everything
        # else → 's'. 'partial' and 'et_partial' both fell into 's' (skip), which
        # silently dropped resumption of interrupted collection + ET for those
        # runs. New: et_partial finishes the partial ET pass; partial resumes
        # abl collection and queues ET.
        if s == 'needs_et':   return 'e'
        if s == 'et_partial': return 'e'
        if s == 'missing':    return 'b'
        if s == 'partial':    return 'b'
        return 's'

    modes = {rn: _default_mode(rn) for rn in run_list}

    # Headless: no prompt
    if os.environ.get('IOTA_HEADLESS') == '1':
        abl_set = {rn for rn, m in modes.items() if m in ('a', 'b')}
        et_set  = {rn for rn, m in modes.items() if m in ('e', 'b')}
        return abl_set, et_set

    # Count helpers
    def _n_abl(rn):
        try:
            from cartography import RUN_CSV
            csv_fname = RUN_CSV.get(rn)
            if not csv_fname: return 0
            fpath = os.path.join(csv_dir, csv_fname)
            cols = _read_csv_cols(fpath, ('run_mode', 'trial', 'priming'))
            trial_set = set()
            for i, t in enumerate(cols.get('trial', [])):
                if cols.get('priming') and cols['priming'][i] == '1':
                    continue
                try: trial_set.add(int(t))
                except Exception: pass
            return len(trial_set)
        except Exception:
            return 0

    def _n_et(rn):
        # v0.79.4.15: glob both 4-digit (canonical, post-migration) and
        # 2-digit (legacy, pre-migration) patterns. Mirrors the scanner's
        # transitional backward-compat read in _run48_status.
        return (
            len(_g.glob(os.path.join(hidden_dir, f'R{rn:04d}_*_et_base.npy'))) +
            len(_g.glob(os.path.join(hidden_dir, f'R{rn:02d}_*_et_base.npy')))
        )

    _mode_label = {
        'e': 'E_t only ',
        'a': 'abl only ',
        'b': 'both     ',
        's': 'skip     ',
    }

    def _print_table():
        ui.section(f"ET Run Configuration -- {len(run_list)} run(s) in queue")
        ui.blank()
        ui.msg(f"  {'Run':<4}  {'Description':<38}  {'Abl CSV':>10}  {'E_t files':>10}  Mode")
        ui.msg(f"  {'─'*4}  {'─'*38}  {'─'*10}  {'─'*10}  {'─'*9}")
        for rn in run_list:
            _, desc = RUN_MAP.get(rn, ('', f'Run {rn}'))
            na = _n_abl(rn)
            ne = _n_et(rn)
            abl_str = f"{na:>4}/{n_trials}"
            et_str  = (f"✓ {ne}" if ne >= n_trials else f"~ {ne}" if ne > 0 else '· none')
            m = modes[rn]
            ui.msg(f"  {rn:02d}    {desc:<38}  {abl_str:>10}  {et_str:>10}  [{m}] {_mode_label[m]}")
        ui.blank()
        ui.msg("  <N> cycle run N mode   <N> <M> set mode (e/a/b/s)")
        ui.msg("  E = all E_t   A = all abl   B = all both   X = all skip")
        ui.msg("  Enter = confirm   s = skip all")
        ui.blank()

    while True:
        _print_table()
        raw = input("  > ").strip().lower()

        if raw == '':
            break
        elif raw == 's':
            return set(), set()
        elif raw == 'e':
            for rn in run_list: modes[rn] = 'e'
        elif raw == 'a':
            for rn in run_list: modes[rn] = 'a'
        elif raw == 'b':
            for rn in run_list: modes[rn] = 'b'
        elif raw == 'x':
            for rn in run_list: modes[rn] = 's'
        else:
            parts = raw.split()
            try:
                rn = int(parts[0])
                if rn not in modes:
                    ui.warn(f"  Run {rn} is not in this batch."); continue
                if len(parts) == 2 and parts[1] in _MODE_CYCLE:
                    modes[rn] = parts[1]
                elif len(parts) == 1:
                    # cycle
                    cur = modes[rn]
                    modes[rn] = _MODE_CYCLE[((_MODE_CYCLE.index(cur) + 1) % len(_MODE_CYCLE))]
                else:
                    ui.warn(f"  Unknown input: '{raw}'")
            except (ValueError, IndexError):
                ui.warn(f"  Unknown input: '{raw}'")

    abl_set = {rn for rn, m in modes.items() if m in ('a', 'b')}
    et_set  = {rn for rn, m in modes.items() if m in ('e', 'b')}
    return abl_set, et_set


def _analysis_is_complete(json_path):
    """Check if an analysis JSON represents a complete run.
    Returns True only if the file exists, parses as valid JSON, and does NOT
    have 'status': 'running'. Everything else (missing, corrupt, partial) → False.
    The JSON content is the sole source of truth -- .running marker files are
    irrelevant to this check."""
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path) as f:
            d = json.load(f)
        return d.get('status') != 'running'
    except Exception:
        return False


# ── Execution order ───────────────────────────────────────────────────────────
# Canonical run order for a full sweep. Differs from numeric order at one point:
# Run 0033 (held-out validation) is placed immediately before Run 0043 (Permutation Sensitivity)
# so that Run 0043 Stage 2 (H28 external validation) has its data available.
# Numeric order would run 0043 before 41, causing Stage 2 to silently find no data.
# _run_session sorts the selected run set against this order instead of numerically.

EXECUTION_ORDER = [
    # v0.79.4.15: runs have been renumbered so their ID equals their
    # execution-order position. This is now just 1..56 in order,
    # with phase boundaries marked for readability.
    # ── DATA COLLECTION ──
    "0001", "0002", "0003",                                    # Phase A -- temp-indep
    "0004", "0005", "0006", "0007", "0008", "0009",            # Phase B -- ET-source
    "0010", "0011", "0012", "0013", "0014", "0015",
    "0016",                                                    # Phase G -- E_t meta-run
    "0017", "0018", "0019",                                    # Phase D -- patching
    "0020", "0021", "0022", "0023", "0024",                    # Phase EFC -- slow→fast
    "0025", "0026", "0027", "0028", "0029",
    "0030", "0031", "0032", "0033", "0034",
    "0035", "0036", "0037", "0038", "0039", "0040",
    # ── PER-TEMPERATURE ANALYSIS ──
    "0041", "0042", "0043", "0044", "0045", "0046",            # chain (POOL_DIM → ...)
    "0047", "0048", "0049",                                    # independents
    # ── POOLED ANALYSIS ──
    "0050", "0051",
    # ── CROSS-MODEL ANALYSIS ──
    "0052", "0053", "0054",
    # ── OUTPUT ──
    "0055", "0056", "0057", "0058", "0059",   # v0.80.0.51: 0058 = apparatus, 0059 = paper assembly (was 0058)
]

def _order_runs(run_set):
    """Return run_set sorted by EXECUTION_ORDER, not numerically.
    Any run not in EXECUTION_ORDER (should not happen) falls to the end.

    v0.79.4.18: EXECUTION_ORDER is a list of 4-digit strings post-renumber,
    but callers hand us ints (from _parse_runs). Normalize the lookup key
    via run_id_pad so both int and string run IDs resolve to their
    canonical 4-digit form before _pos lookup. Without this, every int
    input fell through to the 9999 default and ordering was a no-op."""
    _pos = {r: i for i, r in enumerate(EXECUTION_ORDER)}
    def _key(r):
        try:
            return _pos.get(run_id_pad(r), 9999)
        except (ValueError, TypeError):
            return _pos.get(r, 9999)
    return sorted(run_set, key=_key)


# ── Resume state ──────────────────────────────────────────────────────────────
# Written to last_session.json so the dashboard can offer a Resume button.
# Cleared when the session completes normally or is fully aborted.

RESUME_KEY = '_resume_runs'

def _save_resume_state(remaining_runs: list):
    """Persist the remaining run queue so the dashboard can resume after abort."""
    try:
        sf = os.path.join(ROOT, 'last_session.json')
        if os.path.exists(sf):
            with open(sf) as f:
                s = json.load(f)
        else:
            s = {}
        s[RESUME_KEY] = ','.join(str(r) for r in remaining_runs)
        with open(sf, 'w') as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass


def _clear_resume_state():
    """Remove the resume queue marker when a session completes normally."""
    try:
        sf = os.path.join(ROOT, 'last_session.json')
        if os.path.exists(sf):
            with open(sf) as f:
                s = json.load(f)
            s.pop(RESUME_KEY, None)
            with open(sf, 'w') as f:
                json.dump(s, f, indent=2)
    except Exception:
        pass


# ── Universal CSV dedup ───────────────────────────────────────────────────────
# Auto-fires before any stats/analysis run (25, 27, 32, 33, 34, 40).
# Removes duplicate rows caused by BUG-PARTIAL-TRIAL broken-resume.
#
# Single-condition runs: dedup on (trial, turn, priming) -- keep first occurrence.
# MC runs: dedup on (trial, turn, priming, condition_value) -- keep first occurrence,
#          then cap at n_trials per condition (removes overshoot from mis-resumed runs).
# Patching runs (21, 42), Run 0003, no-GPU runs: skipped.

from cartography import DualKeyRunSet as _DKRSet
# v0.79.4.15: DualKeyRunSet accepts int or 4-digit string on membership
# tests. Canonical storage is 4-digit string. Lets legacy `run_num in
# _ANALYSIS_RUNS` (int) keep working while on-disk/UI surfaces use strings.
_ANALYSIS_RUNS = _DKRSet({41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54})  # v0.79.4.15: was {25, 27, 32, 33, 34, 40, 45, 46, 47, 49, 50, 51, 52, 56}

def _dedup_all_csvs(paths, session):
    """Run universal dedup across all collected CSVs. Called before stats runs.
    Delegates to dedup_all_runs.process_run -- single source of truth for dedup logic.
    """
    from cartography import RUN_CSV
    import dedup_all_runs as _dedup_mod

    csv_dir  = paths.get('csv', '')
    n_trials = session.get('trials', 100)
    total_fixed = 0

    for run_num, csv_fname in sorted(RUN_CSV.items()):
        if csv_fname is None: continue
        try: run_num_int = int(run_num)
        except (ValueError, TypeError): run_num_int = -1
        if run_num_int == 1: continue  # multi-pass structure exempt -- see v54.0.2, v54.2.4
        fpath = os.path.join(csv_dir, csv_fname)
        if not os.path.exists(fpath): continue
        try:
            result = _dedup_mod.process_run(run_num_int, fpath, n_trials,
                                            dry_run=False, verbose=False)
            removed = result.get('total', 0)
            if removed:
                msg = (f"  [dedup] Run {run_num_int:04d}: removed {removed} rows"
                       f" (dedup={result['dedup']}, cap={result['cap']}, cull={result['cull']})")
                print(msg, flush=True)
                try:
                    from orchestration_core import _append_log
                    _append_log(msg, kind='warn')
                except Exception: pass
                total_fixed += removed
        except Exception as e:
            ui.warn(f"  [dedup] Run {run_num_int:04d}: skipped ({e})")

    if total_fixed:
        ui.warn(f"  [dedup] Total rows removed: {total_fixed} -- backups saved as .csv.bak")
    else:
        ui.ok("  [dedup] All CSVs clean.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_module(name):
    """Dynamically load a phase module (runners, graft_patching,
    analysis, etc.) by name from the iota root. Used by the RUN_MAP
    dispatch: RUN_MAP[n] = (module_name, label) and this function
    resolves the module fresh per-run so a re-deployed module picks
    up immediately without restarting the Python process."""
    path = os.path.join(ROOT, f"{name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}.py not found in {ROOT}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_runs(spec):
    """Parse a comma-separated run spec into a sorted list of run
    numbers that exist in RUN_MAP.

    Accepts ranges ('1-5'), individual numbers ('7'), and mixed
    ('1-5,7,12-14'). Silently drops non-numeric tokens and numbers
    outside RUN_MAP -- so '1-100' gracefully resolves to the valid
    subset rather than erroring. Used by every dispatch path: console
    menu, dashboard /queue, headless --auto-run, --batch-runs."""
    runs = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            try: runs.update(range(int(a), int(b)+1))
            except ValueError: pass
        elif part.isdigit():
            runs.add(int(part))
    return sorted(r for r in runs if r in RUN_MAP)


def _ensure_calibration(session, paths):
    """Guarantee this model has a power calibration on disk, running
    one if needed.

    Checks paths['calibration']/calibration_{sanitize(model_name)}.json
    first (via load_calibration which also handles model_name drift
    by scanning for matching model_path). If missing, loads the model
    at the session's active quantization, runs run_calibration (~2
    min of greedy generations), writes the slope/intercept JSON, and
    unloads. Called before any run whose print_turn_result output
    depends on the adjusted-power display."""
    from orchestration_core import load_calibration, run_calibration, load_model, unload_model
    from cartography import sanitize
    import json

    model_name = session.get('model_name', '')
    model_path = session.get('model_path', '')

    if load_calibration(paths['calibration'], model_name, model_path) is not None:
        return

    ui.warn("No calibration for this model. Running power calibration...")
    os.makedirs(paths['calibration'], exist_ok=True)
    mdl, tok = load_model(session['model_path'], token=session.get('hf_token'),
                          quant=session.get('quantization', '4bit'))
    try:
        slope, intercept = run_calibration(mdl, tok)
    finally:
        unload_model(mdl)
        del mdl, tok

    fname = os.path.join(paths['calibration'],
                         f"calibration_{sanitize(model_name)}.json")
    with open(fname, 'w') as f:
        json.dump({"model": model_name, "model_path": model_path,
                   "slope": slope, "intercept": intercept}, f, indent=2)
    ui.ok(f"Calibration complete: {slope:.4f} W/token")




# ─────────────────────────────────────────────
# TEMPERATURE-INDEPENDENT RUN COPY
# ─────────────────────────────────────────────
# Runs 0001, 0002, 0003 produce data that is valid across all temperature rounds:
#   Run 0001  -- three-model C_t/E_t isolation (model forward pass, no sampling)
#   Run 0002  -- internal temperature sweep (5 temps in one pass, ignores session temp)
#   Run 0003  -- temperature grid run (4 conditions × 5 temps in one pass)
#
# Runs 0017/0018 were previously listed here but removed in v54.2.2 -- they now
# collect per-temperature data for H36/H37 (causal effect temperature invariance).
#
# When starting a new temperature round, auto-copy CSVs and hidden states
# from the first completed condition directory found.

_TEMP_INDEPENDENT_RUNS = {
    # v0.79.4.0 renumber: keys flipped to new IDs. Without this, iteration
    # passes OLD int as run_num, glob pattern becomes f"R{19:04d}_*" which
    # doesn't match anything post-migration. Copy-hidden silently no-ops.
    1: {'csv': 'R0001_null.csv',              'copy_hidden': True,  'variants': ['abliterated', 'base', 'instruct']},  # was R19
    2: {'csv': 'R0002_robustness.csv',        'copy_hidden': True,  'variants': ['abliterated', 'base']},              # was R20
    3: {'csv': 'R0003_temperature_grid.csv',  'copy_hidden': True,  'variants': ['abliterated']},                      # was Q26
}

def _get_temp_conditions(family, size, variant):
    """Scan disk for temperature condition directories."""
    from cartography import get_family_size_dir
    base = os.path.join(get_family_size_dir(family, size), variant)
    conds = []
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if d == 'deterministic' or d.startswith('temp_'):
                conds.append(d)
    return conds if conds else ['deterministic']


def _copy_temperature_independent_runs(session, paths):
    """
    For each temperature-independent run: if the CSV is missing in the current
    temperature condition, search other conditions for a complete copy and copy
    it in. If copy_hidden is True, also copies hidden state files.

    Called once at session start before any collection begins.
    """
    import shutil
    from cartography import DATA, condition_name, run_prefix, get_family_size_dir

    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')
    current_cond = condition_name(session.get('temperature', 0.0))

    # Nothing to do if we're in deterministic -- that's the source
    if current_cond == 'deterministic':
        return

    copied_any = False

    for run_num, cfg in _TEMP_INDEPENDENT_RUNS.items():
        csv_fname = cfg['csv']
        dst_csv   = os.path.join(paths['csv'], csv_fname)

        # Find source condition -- even if CSV already present, hidden states may be missing
        src_cond = None
        _temp_conds = _get_temp_conditions(family, size, variant)
        for cond in _temp_conds:
            if cond == current_cond:
                continue
            candidate = os.path.join(get_family_size_dir(family, size), variant, cond, 'csv', csv_fname)
            if os.path.exists(candidate) and os.path.getsize(candidate) > 1024:
                src_cond = cond
                break

        if src_cond is None:
            continue

        # Copy CSV if missing
        if not (os.path.exists(dst_csv) and os.path.getsize(dst_csv) > 0):
            src_csv = os.path.join(get_family_size_dir(family, size), variant, src_cond, 'csv', csv_fname)
            os.makedirs(paths['csv'], exist_ok=True)
            shutil.copy2(src_csv, dst_csv)
            ui.ok(f"  Copied Run {run_num:04d} CSV from {src_cond} → {current_cond}")
            copied_any = True

        # Copy hidden states if configured
        if cfg['copy_hidden']:
            pfx = run_prefix(run_num)
            pattern = f"{pfx}{run_num:04d}_*"
            n_copied = _copy_run_hidden_states(
                family, size, cfg['variants'], src_cond, current_cond, pattern)
            if n_copied:
                ui.ok(f"  Copied {n_copied} Run {run_num:04d} hidden state files → {current_cond}")
                copied_any = True

    if copied_any:
        ui.blank()


def _copy_run_hidden_states(family, size, variants, src_cond, dst_cond, glob_pattern):
    """Copy hidden state files matching glob_pattern across specified variants.

    v0.58.0.0: generalized from _copy_run01_hidden_states. Handles any run
    and any set of variant directories (abliterated, base, instruct).
    """
    import shutil, glob
    from cartography import get_family_size_dir

    total_copied = 0
    for var in variants:
        src_dir = os.path.join(get_family_size_dir(family, size), var, src_cond, 'hidden_states')
        dst_dir = os.path.join(get_family_size_dir(family, size), var, dst_cond, 'hidden_states')

        if not os.path.isdir(src_dir):
            continue

        files = glob.glob(os.path.join(src_dir, glob_pattern))
        if not files:
            continue

        os.makedirs(dst_dir, exist_ok=True)
        for src_file in files:
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                total_copied += 1

    return total_copied


def _atomic_write_json(path, data):
    """Write JSON atomically -- write to temp file then rename.
    Prevents corrupt session file if process dies mid-write."""
    import json as _j, tempfile as _tf, os as _os
    dir_ = _os.path.dirname(_os.path.abspath(path))
    fd, tmp = _tf.mkstemp(dir=dir_, suffix='.tmp')
    try:
        with _os.fdopen(fd, 'w') as f:
            _j.dump(data, f, indent=2)
        _os.replace(tmp, path)
    except Exception:
        try: _os.unlink(tmp)
        except Exception: pass
        raise


def _clean_stale_tmp():
    """Delete orphaned .tmp files from hard crashes (OOM, os._exit, force-close).
    _atomic_write_json creates temp files that stay if the process dies before
    os.replace. Safe to delete: they're always 0-byte or stale partial JSON."""
    import glob as _g
    for f in _g.glob(os.path.join(ROOT, '*.tmp')):
        try:
            os.remove(f)
        except Exception:
            pass

def _wait_for_vram(run_label, threshold_pct=0.75, timeout=120):
    """Wait until free VRAM exceeds threshold before spawning the next run.

    Uses percentage of total VRAM instead of absolute GiB -- works for any card
    size and any model size. 75% free means the model is genuinely released.
    Quick exit: if already above threshold on first check, returns immediately.
    """
    import time as _t
    from orchestration_core import _append_log as _vram_log
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gib = mem.total / 1024**3
        threshold_gib = total_gib * threshold_pct
        free_gib = mem.free / 1024**3
        # Quick exit if already clear
        if free_gib >= threshold_gib:
            pynvml.nvmlShutdown()
            return
        print(f"  [vram] waiting for GPU to clear after {run_label}...", flush=True)
        _vram_log(f"  [vram] waiting for GPU to clear after Run {run_label}...", kind="turn")
        deadline = _t.time() + timeout
        while _t.time() < deadline:
            _t.sleep(2)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gib = mem.free / 1024**3
            if free_gib >= threshold_gib:
                print(f"  [vram] clear -- {free_gib:.2f} GiB free.", flush=True)
                _vram_log(f"  [vram] clear -- {free_gib:.2f} GiB free.", kind="ok")
                pynvml.nvmlShutdown()
                return
        print(f"  [vram] timeout after {timeout}s -- {free_gib:.2f}/{total_gib:.1f} GiB free.", flush=True)
        _vram_log(f"  [vram] timeout -- {free_gib:.2f} GiB free.", kind="warn")
        pynvml.nvmlShutdown()
    except Exception as _e:
        print(f"  [vram] pynvml unavailable ({_e}), waiting 10s...", flush=True)
        _t.sleep(10)



def _run_session_isolated(session):
    """
    Orchestrate a multi-run session by spawning a fresh subprocess for each run.

    Each run gets its own Python process -- completely clean VRAM, no fragmentation
    from prior runs. The orchestrator itself never loads a model.

    Flow:
      1. Parse and order runs
      2. Copy temperature-independent data (19, 21, 26, 42)
      3. For each run: spawn start_here.py --single-run N, wait for completion
      4. After all generation runs: fire ET recovery batch (also isolated)
    """
    import subprocess as _sp
    _clean_stale_tmp()

    runs = _order_runs(_parse_runs(session.get('runs', '1-44')))
    if not runs:
        ui.warn("No valid runs selected."); return

    paths = get_paths(session.get('model_family','llama'), session.get('model_size','8b'),
                      session.get('model_variant','abliterated'), session.get('temperature',0.0))

    _copy_temperature_independent_runs(session, paths)

    from orchestration_core import load_calibration as _lc
    _cal_needed = _lc(paths['calibration'],
                      session.get('model_name',''),
                      session.get('model_path','')) is None
    _ensure_calibration(session, paths)
    if _cal_needed:
        ui.ok("Calibration complete. Continuing with isolated run mode...")

    ui.section("Queued runs (isolated mode -- fresh process per run):")
    for r in runs:
        _, desc = RUN_MAP.get(r, ('?', '?'))
        ui.msg(f"    {r:02d}  {desc}")
    ui.blank()

    completed = 0
    skipped   = 0

    # v0.79.4.15: _et_direct_batch carve-out removed. E_t recovery is now
    # a first-class run (Run 0016) in the queue -- dispatched like any other
    # subprocess. Previously ET-only runs (status='needs_et') were pulled
    # out of the subprocess loop and routed to an implicit batch; with
    # Run 0016 as a visible entry, the user sees exactly one ET subprocess
    # fire when their queue reaches it. Per-source ET overrides in the
    # dashboard ET popup still work because openEtPop / launchEtBatch
    # write directly to _et_batch via the legacy path below.
    _real_runs = list(runs)
    _et_direct_batch = []  # retained as empty for compatibility with later references

    # Squid -- three hearts. Once per model family, on the console you're watching.
    _squid_family = session.get('model_family', 'unknown')
    _squid_sentinel = os.path.join(ROOT, f'.iota_squid3_{_squid_family}')
    if not os.path.exists(_squid_sentinel):
        try:
            import time as _se_t, sys as _se_s
            _se_b = "\n       _____\n      /     \\\n     /  ___  \\\n    |  /   \\  |\n    | | \u25c9 \u25c9 | |\n    | |  \u223c  | |\n    |  \\___/  |\n     \\       /\n      \\_____/\n        |||||\n       /|||||\\\n      ~~~~~~~~~~~\n"
            _se_m = _se_b + "\n  # Squids have three hearts.  Maybe.\n"
            for _sl in _se_m.split("\n"): _se_s.stdout.write(_sl+"\n"); _se_s.stdout.flush(); _se_t.sleep(0.025)
            open(_squid_sentinel, 'w').close()
        except Exception:
            pass

    for run_num in _real_runs:
        # Check if already done before spawning a process.
        # v0.58.0.0: Run 0001 no longer exempted here. The exemption existed for
        # _run_session (interactive) where Run 0001 falls through to _prompt_pass_cfg_r01.
        # In isolated mode there is no interactive prompt -- the subprocess would hit
        # RuntimeError (_run_null_trivariant lost in v54.0.0). Auto-copy from
        # _copy_temperature_independent_runs handles Run 0001 data correctly.
        _status = _scan_runs(paths, session.get('trials', 100))
        if _status.get(run_num) == 'done' and run_num not in _ANALYSIS_RUNS:
            ui.ok(f"Run {run_num:04d} already complete -- skipping.")
            if run_num == 1:
                try:
                    from runners_p1 import _print_nietzsche
                    _print_nietzsche(session.get('model_family', 'unknown'))
                except Exception:
                    pass
            skipped += 1
            continue

        ui.section(f"Run {run_num:04d} -- spawning isolated process")

        # ── Run 0020 special case: subprocess per instance ──────────────────
        # Run 0020 loads the model twice (instance A, B) in one process. On 10GB
        # cards the CUDA context from A doesn't release, B OOMs. Split into
        # separate subprocesses: A (GPU), B (GPU), coupling (no GPU).
        if run_num == 20:
            import json as _json
            _sess_file = os.path.join(ROOT, 'last_session.json')
            log_path = os.path.join(ROOT, '.iota_flask.log')
            _r30_ok = True
            for _r30_phase in ('A', 'B', 'coupling'):
                _wsess_tmp = dict(session)
                _wsess_tmp['runs'] = '20'  # v0.79.4.17: was '30' (old id) -- post-renumber two-instance is new 0020
                try:
                    _atomic_write_json(_sess_file, _wsess_tmp)
                except Exception as _e:
                    ui.warn(f"  Could not write session for Run 0020 {_r30_phase}: {_e}")
                    _r30_ok = False; break
                _cmd = [sys.executable, os.path.join(ROOT, 'start_here.py'),
                        '--single-run', '20', '--r30-instance', _r30_phase]  # v0.79.4.17: was '30' (old id)
                ui.msg(f"  Run 0020 instance {_r30_phase}...")
                try:
                    with open(log_path, 'a') as _lf:
                        proc = _sp.Popen(_cmd, stdout=_lf, stderr=_sp.STDOUT)
                    ret = proc.wait()
                    if ret != 0:
                        ui.warn(f"  Run 0020 instance {_r30_phase} exited with code {ret}.")
                        if _r30_phase != 'coupling':
                            _r30_ok = False; break
                except Exception as _e:
                    ui.err(f"  Run 0020 instance {_r30_phase} error: {_e}")
                    _r30_ok = False; break
                finally:
                    try: _atomic_write_json(_sess_file, session)
                    except Exception: pass
                # VRAM wait between A and B -- not needed after coupling (no GPU)
                if _r30_phase in ('A', 'B'):
                    _wait_for_vram(f"30-{_r30_phase}")
            if _r30_ok:
                ui.ok("Run 0020 complete (all instances + coupling).")
                completed += 1
            else:
                ui.warn("Run 0020 incomplete -- instance failed.")
                skipped += 1
            continue
        # ── End Run 0020 special case ───────────────────────────────────────

        import json as _json
        _sess_file = os.path.join(ROOT, 'last_session.json')
        _wsess_tmp = dict(session)
        _wsess_tmp['runs'] = str(run_num)
        try:
            _atomic_write_json(_sess_file, _wsess_tmp)
        except Exception as _e:
            ui.warn(f"  Could not write session for run {run_num}: {_e} -- skipping")
            skipped += 1
            continue

        log_path = os.path.join(ROOT, '.iota_flask.log')
        try:
            with open(log_path, 'a') as _lf:
                proc = _sp.Popen(
                    [sys.executable, os.path.join(ROOT, 'start_here.py'),
                     '--single-run', str(run_num)],
                    stdout=_lf, stderr=_sp.STDOUT
                )
            ui.msg(f"  Run {run_num:04d} running... (output → .iota_flask.log)")
            ret = proc.wait()
            if ret == 0:
                ui.ok(f"Run {run_num:04d} complete (exit 0).")
                completed += 1
                if run_num == 3:
                    try:
                        from runners_p2 import _crab_walk
                        _crab_walk(session.get('model_family', 'unknown'))
                    except Exception:
                        pass
            else:
                ui.warn(f"Run {run_num:04d} exited with code {ret} -- may have OOM'd or failed.")
                skipped += 1
        except Exception as _e:
            ui.err(f"Run {run_num:04d} subprocess error: {_e}")
            skipped += 1
        finally:
            # Restore session from orchestrator's own dict (not the stale backup string).
            # Using the backup string would overwrite any settings the user changed
            # during the session (hf_token, temperature, etc).
            try:
                _atomic_write_json(_sess_file, session)
            except Exception:
                pass

        # Wait for Windows CUDA driver to release GPU memory before next spawn.
        _wait_for_vram(run_num)

        # v0.58.0.0 FIX-5: Do NOT fire ET recovery per-run. Accumulate markers.
        # The old code spawned a separate ET subprocess after every single run,
        # causing N base model loads + N CUDA teardowns + N VRAM waits for what
        # should be one. Markers are collected and fired as a single batch below.

    # ── Batched ET recovery -- one subprocess for all accumulated runs ─────────
    import json as _json
    _et_marker = os.path.join(ROOT, '.iota_et_pending.json')
    # Merge any marker-file runs with the pre-scanned ET-only batch
    _et_all = list(_et_direct_batch)  # runs we skipped above
    if os.path.exists(_et_marker):
        try:
            with open(_et_marker) as _f:
                _et_data = _json.load(_f)
            _et_from_marker = _et_data.get('runs', []) if isinstance(_et_data, dict) else _et_data
            _et_all = sorted(set(_et_all) | set(_et_from_marker))
            os.remove(_et_marker)
        except Exception:
            pass
    if _et_all:
        _et_runs = sorted(set(_et_all))
        log_path = os.path.join(ROOT, '.iota_flask.log')
        ui.section(f"E_t Recovery -- spawning isolated subprocess for runs {_et_runs}")
        ui.msg(f"  One base model load for all {len(_et_runs)} runs.")
        try:
            with open(log_path, 'a') as _lf:
                et_proc = _sp.Popen(
                    [sys.executable, os.path.join(ROOT, 'start_here.py'),
                     '--et-recovery', ','.join(str(r) for r in _et_runs)],
                    stdout=_lf, stderr=_sp.STDOUT
                )
            et_ret = et_proc.wait()
            if et_ret == 0:
                ui.ok(f"  E_t recovery complete -- {len(_et_runs)} runs.")
            else:
                ui.warn(f"  E_t recovery exited {et_ret} -- may have failed.")
            _wait_for_vram('ET')
        except Exception as _e:
            ui.warn(f"  ET recovery subprocess failed: {_e}")

    ui.blank()
    ui.ok(f"Session complete -- {completed} runs done, {skipped} skipped/failed.")
    _clear_resume_state()


# ── Temperature auto-advance ──────────────────────────────────────────────────

_ALL_TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# v0.79.4.15: post-renumber, data-collection runs are positions 1-40 (inc.
# E_t meta-run at pos 16). Analysis/output is 41-56.
_GEN_RUNS  = _DKRSet(set(range(1, 41)))


def _scan_temp_status(session):
    """Check completion status of each temperature round.
    Returns list of (temp, n_done, n_total, is_complete) in order."""
    from cartography import get_paths as _gp
    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')
    n_trials = session.get('trials', 100)
    results = []
    for temp in _ALL_TEMPS:
        paths = _gp(family, size, variant, temp, create_dirs=False)
        status = _scan_runs(paths, n_trials)
        done = {r for r in _GEN_RUNS if status.get(r) == 'done'}
        results.append((temp, len(done), len(_GEN_RUNS), len(done) == len(_GEN_RUNS)))
    return results


def _run_all_temperatures(session):
    """Auto-advance through temperature rounds.

    Scans all 6 temperature directories. For each incomplete round (in order),
    sets session temperature, runs the full collection pass, then moves to the
    next. Stops when all rounds are complete or user aborts.
    """
    from cartography import condition_name

    ui.section("Temperature auto-advance -- scanning all rounds")
    temp_status = _scan_temp_status(session)
    for temp, n_done, n_total, complete in temp_status:
        cond = condition_name(temp)
        mark = "DONE" if complete else f"{n_done}/{n_total}"
        ui.msg(f"  T={temp:.1f}  ({cond}):  {mark}")
    ui.blank()

    rounds_completed = 0
    for temp, n_done, n_total, complete in temp_status:
        if complete:
            continue
        cond = condition_name(temp)
        ui.section(f"Starting T={temp:.1f}  ({cond})  --  {n_done}/{n_total} runs done")

        # Update session temperature and persist
        session['temperature'] = temp
        # v0.76.0.4: only dispatch GPU collection runs. Analysis runs (25, 27,
        # 32, 33, 34, 45, 46, etc.) must not run mid-collection -- they require
        # completed data + Run 0041 calibration. Analysis happens after all temps.
        _gen_str = ','.join(str(r) for r in sorted(_GEN_RUNS))
        session['runs'] = _gen_str
        _atomic_write_json(os.path.join(ROOT, 'last_session.json'), session)

        _run_session_isolated(session)
        rounds_completed += 1

        # Re-scan to confirm completion
        post_status = _scan_temp_status(session)
        post_temp = [s for s in post_status if s[0] == temp][0]
        if post_temp[3]:
            ui.ok(f"T={temp:.1f} complete.")
        else:
            ui.warn(f"T={temp:.1f} finished with {post_temp[1]}/{post_temp[2]} runs -- "
                    f"some may have failed. Continuing to next temperature.")

    ui.blank()
    if rounds_completed == 0:
        ui.ok("All temperature rounds already complete.")
    else:
        ui.ok(f"Temperature auto-advance done -- {rounds_completed} round(s) collected.")


def _run_all_models(session, runs_payload):
    """v0.79.4.15 -- fan out runs across every discovered model in DATA/.

    ARGUMENT
    ========
    runs_payload: str
        The same run specification the single-model pipeline accepts:
        - 'auto-temp'            → iterate every incomplete temp round per model
        - 'all-temps:1,2,3'      → fire runs at every temp with data per model
        - '1,2,3' / '1-10'       → plain run list at session's current temperature
        Passed through unchanged to the per-model dispatch; each model gets the
        same runs string, re-interpreted against its own data directory.

    SEMANTICS
    =========
    Discovers every model that has data on disk (via the same walker the
    dashboard /models_collected endpoint uses -- export_stats._discover_model_data).
    For each model, writes the session file with that model's family/size/variant/
    quantization, then invokes the appropriate single-model dispatch path (the
    same path --auto-temp or --all-temps-runs or --auto-run would take from
    a normal launch). No parallelization in this ship -- strictly sequential,
    model A fully complete before model B starts.

    SMART LOADER
    ============
    Each run within a model is already its own subprocess (0.58.0.0 isolation).
    "Grouping by model" is the natural consequence of iterating models at the
    outer loop -- all of Model A's runs fire in sequence before Model B starts,
    so the base-model cache and HF tokenizer downloads don't thrash between
    models. No explicit model-pinning needed; the outer iteration is the group.

    PARALLELIZATION
    ===============
    Deferred to 0.79.3.1. A no-GPU analysis run on Model A could safely run
    concurrently with a GPU collection run on Model B (different resources,
    no contention), but the worker-pool, disk-lock, and CSV-concurrency
    infrastructure is non-trivial. This ship stays sequential and correct.

    FAILURE MODE
    ============
    If one model's dispatch exits non-zero (e.g., OOM on Model A), the loop
    logs the failure and continues to the next model. User sees which models
    completed and which didn't in the summary banner at the end.
    """
    import subprocess as _sp_am
    import export_stats as _es_am

    ui.section(f"All-Models dispatch -- runs: {runs_payload}")

    try:
        models = _es_am._discover_model_data({})
    except Exception as e:
        ui.err(f"All-Models: could not discover models: {e}")
        return

    if not models:
        ui.warn("No models have data on disk -- nothing to dispatch.")
        return

    ui.msg(f"  Discovered {len(models)} model(s):")
    for m in models:
        ui.msg(f"    - {m['label']}  ({m['family']}/{m['size']}/{m['variant']})")
    ui.blank()

    _sess_file = os.path.join(ROOT, 'last_session.json')
    log_path = os.path.join(ROOT, '.iota_flask.log')

    completed = []
    failed = []

    for idx, m in enumerate(models, 1):
        ui.section(f"Model {idx}/{len(models)}  --  {m['label']}")

        # Derive quant from size_dir if present ('2b_8bit' → '8bit', else '4bit')
        from cartography import quant_from_dir as _qfd
        _quant = _qfd(m['size']) or session.get('quantization', '4bit')
        from cartography import logical_size as _ls
        _logical_size = _ls(m['size'])

        # Set this model in session, then pass runs_payload through to the
        # correct sub-CLI. Subprocess inherits env + reads the session file.
        _model_session = dict(session)
        _model_session.update({
            'model_family':  m['family'],
            'model_size':    _logical_size,
            'model_variant': m['variant'],
            'quantization':  _quant,
        })
        # model_name / model_path are derived from the (family, logical_size,
        # variant, quant) tuple by vault lookup -- refresh them so each model's
        # subprocess uses the correct HF path and display label.
        try:
            from vault import resolve_variant_path as _rvp
            _name, _path = _rvp(m['family'], _logical_size, m['variant'])
            if _path:
                _model_session['model_path'] = _path
            if _name:
                _model_session['model_name'] = _name
        except Exception:
            # vault helper fallback -- subprocess will use whatever's in the
            # session file from the original launch. Not fatal.
            pass

        try:
            _atomic_write_json(_sess_file, _model_session)
        except Exception as e:
            ui.warn(f"  Could not write session for {m['label']}: {e} -- skipping")
            failed.append(m['label'])
            continue

        # Choose the right sub-CLI based on the runs payload.
        if runs_payload == 'auto-temp':
            cmd = [sys.executable, os.path.join(ROOT, 'start_here.py'), '--auto-temp']
        elif runs_payload.startswith('all-temps:'):
            _atr = runs_payload.split(':', 1)[1]
            cmd = [sys.executable, os.path.join(ROOT, 'start_here.py'),
                   '--all-temps-runs', _atr]
        else:
            cmd = [sys.executable, os.path.join(ROOT, 'start_here.py'),
                   '--headless', '--auto-run', runs_payload]

        try:
            with open(log_path, 'a') as _lf:
                proc = _sp_am.Popen(cmd, stdout=_lf, stderr=_sp_am.STDOUT)
            ret = proc.wait()
            if ret == 0:
                ui.ok(f"  {m['label']} -- complete.")
                completed.append(m['label'])
            else:
                ui.warn(f"  {m['label']} -- exited {ret}, continuing to next model.")
                failed.append(m['label'])
        except Exception as e:
            ui.err(f"  {m['label']} -- dispatch failed: {e}")
            failed.append(m['label'])

    # Restore original session so the user's "active" model isn't whatever
    # the final all-models iteration happened to leave.
    try:
        _atomic_write_json(_sess_file, session)
    except Exception:
        pass

    ui.blank()
    ui.section("All-Models dispatch complete")
    ui.ok(f"  Completed: {len(completed)}/{len(models)}")
    if completed:
        for name in completed:
            ui.ok(f"    ✓ {name}")
    if failed:
        ui.warn(f"  Failed/incomplete: {len(failed)}")
        for name in failed:
            ui.warn(f"    ✗ {name}")


def _run_session(session):
    """Primary dispatch -- execute a user-selected run list end-to-end.

    Responsibilities:
      - Parse + order the run list per EXECUTION_ORDER (analysis runs
        follow their collection prereqs).
      - Resolve paths for the current temperature cell.
      - Auto-copy temperature-independent runs (19, 20, 26) from any
        sibling temp dir that already has them -- avoids re-collecting
        identical data across 6 temperature rounds.
      - Single-run mode (IOTA_SINGLE_RUN=1): dedup this run's CSV
        before _get_trials_for_condition reads it so resume logic
        sees clean data.
      - Auto-prepend ET recovery runs for any SOURCE_RUNS_2WAY /
        SOURCE_RUNS_3WAY member lacking E_base files.
      - Dispatch each run via its RUN_MAP entry, unloading the model
        between runs when session['unload_between_runs'] is set.
      - Check prereqs (_PREREQS) before each run -- hard warning block
        if upstream data is missing.

    Called directly by console mode and by --auto-run / --auto-temp /
    --all-temps-runs dispatch paths."""
    runs = _order_runs(_parse_runs(session.get('runs', '1-44')))
    if not runs:
        ui.warn("No valid runs selected."); return

    # paths must be defined before _scan_runs is called anywhere in this function.
    # BUG-42C fix: previously defined after the auto-prepend ET check, causing
    # UnboundLocalError when any ET_RECOVERY_RUNS were in the queue -- subprocess
    # crashed immediately, timer ran client-side, nothing happened.
    paths = get_paths(session.get('model_family','llama'), session.get('model_size','8b'),
                      session.get('model_variant','abliterated'), session.get('temperature',0.0))

    # Auto-copy temperature-independent runs (19, 21, 26, 42) from another
    # temperature condition if they exist there but not here.
    _copy_temperature_independent_runs(session, paths)

    # In single-run mode (isolated subprocess), dedup this run's CSV before
    # _get_trials_for_condition reads it. Removes duplicate rows from interrupted
    # sessions so resume logic sees clean data.
    if os.environ.get('IOTA_SINGLE_RUN') == '1' and len(runs) == 1 and runs[0] != 19:
        try:
            from cartography import RUN_CSV
            import dedup_all_runs as _dm
            _run_csv = RUN_CSV.get(runs[0])
            if _run_csv:
                _fpath = os.path.join(paths['csv'], _run_csv)
                if os.path.exists(_fpath):
                    _result = _dm.process_run(runs[0], _fpath,
                                              session.get('trials', 100),
                                              dry_run=False, verbose=False)
                    if _result.get('total', 0):
                        ui.warn(f"  [dedup] Run {runs[0]:02d}: removed {_result['total']} rows before collection")
        except Exception as _de:
            ui.warn(f"  [dedup] pre-run dedup skipped: {_de}")

    # Smart ordering: if any ET_RECOVERY_RUNS are in the queue and Run 0001 is
    # not yet complete, prepend Run 0001 automatically. The E_t recovery pass
    # (which fires after the main loop) needs Run 0001's global C_t constant
    # for retroactive backfill. Without it the recovery proceeds but C_t
    # for Runs 0004-0012 / 15-17 remains a zero-vector placeholder.
    #
    # v0.66.3.0: skip in single-run isolated mode. The orchestrator
    # (_run_session_isolated) handles Run 0001 ordering and ET batching at
    # the session level. Auto-prepending inside a --single-run subprocess
    # forces a three-model download+load that may not be wanted yet -- the
    # user may want to collect abliterated data first and do Run 0001 later.
    if (any(r in _ET_RECOVERY_RUNS for r in runs)
            and os.environ.get('IOTA_SINGLE_RUN') != '1'):
        _early_status = _scan_runs(paths, session.get('trials', 100))
        if _early_status.get("0001", 'missing') != 'done' and 1 not in runs:
            ui.msg("  Auto-prepending Run 0001 -- required before E_t recovery (global C_t backfill).")
            runs = [1] + runs  # v0.79.4.0: was R19 null+hidden

    ui.section("Queued runs:")
    for r in runs:
        _, desc = RUN_MAP[r]
        ui.msg(f"    {r:02d}  {desc}")
    ui.blank()

    # ── Pre-flight prerequisite check (v30.4) ─────────────────────────────────
    # Scan filesystem once before any GPU/model work.  Only warn about runs whose
    # prerequisites are BOTH (a) not yet done, AND (b) not also being collected in
    # this very session -- if Run 0006 and Run 0017 are both queued in execution order,
    # the prereq will be satisfied before Run 0017 is reached; no warning needed.
    # The per-run gate (below) is the authoritative real-time check; this pre-flight
    # is an early warning that fires before model load so the user can abort cheaply.
    _pf_status  = _scan_runs(paths, session.get('trials', 100))
    _runs_set   = set(runs)
    _pf_problems = {}   # run_num → list of (prereq_num, prereq_stat, desc)

    for r in runs:
        raw_problems = _check_prereqs(r, _pf_status)
        # Drop problems whose prereq run is also in this session's queue
        # (it will complete first due to EXECUTION_ORDER).
        # v0.80.0.44: _PREREQS keys are 4-digit strings ("0001"); _runs_set
        # holds ints. Coerce pnum to int for the membership compare,
        # otherwise every prereq is treated as external and the warning
        # block fires for in-batch dependencies that will satisfy themselves.
        external_problems = [
            (pnum, pstat, desc) for pnum, pstat, desc in raw_problems
            if int(pnum) not in _runs_set
        ]
        if external_problems:
            _pf_problems[r] = external_problems

    if _pf_problems:
        W = 64
        print()
        print('!' * W)
        ui.warn("PRE-FLIGHT PREREQUISITE WARNING")
        ui.warn("The following queued runs have upstream data that is missing or")
        ui.warn("incomplete, and that data is NOT being collected in this session.")
        print('!' * W)
        for r, problems in sorted(_pf_problems.items()):
            _, rdesc = RUN_MAP.get(r, ('', f'Run {r}'))
            ui.blank()
            ui.warn(f"  Run {r:02d} -- {rdesc}")
            for prereq_num, prereq_stat, desc in problems:
                label = "MISSING    " if prereq_stat == 'missing' else "INCOMPLETE "
                ui.warn(f"    [{label}]  {desc}")
        print()
        print('!' * W)
        ui.blank()

        if os.environ.get('IOTA_HEADLESS') == '1':
            # In headless mode: log and continue.  The per-run gate will skip
            # individual runs whose prereqs are still unmet at dispatch time.
            ui.warn("[headless] Proceeding -- per-run gate will skip runs with unmet prereqs.")
        else:
            ui.warn("Results from runs attempted without prerequisites may be invalid.")
            ui.blank()
            if not ui.timed_confirm("Proceed anyway? (per-run gate will re-check each run)",
                                    timeout=30, default_yes=False):
                ui.warn("Aborted. Collect the prerequisite data first, or add those runs to the queue.")
                return
        ui.blank()
    # ── End pre-flight ─────────────────────────────────────────────────────────

    # If calibration is needed, run it and then relaunch the session in a fresh
    # process so VRAM starts clean. Calibration loads and unloads the model;
    # even with proper del/gc the allocator may retain fragmented pages that
    # cause OOM on subsequent loads within the same process.
    from orchestration_core import load_calibration as _lc
    _cal_needed = _lc(paths['calibration'],
                      session.get('model_name',''),
                      session.get('model_path','')) is None
    _ensure_calibration(session, paths)
    if _cal_needed:
        ui.ok("Calibration complete. Continuing with isolated run mode...")

    # Publish run-list context so the dashboard progress bars and model label
    # are populated from the first turn. Finding Y fix (v24.5).
    try:
        from orchestration_core import update_dashboard_ctx
        update_dashboard_ctx(
            total_runs=len(runs),
            model_name=session.get('model_name', ''),
        )
    except Exception:
        pass

    # ── Upfront ET batch configuration (v42.1.2) ──────────────────────────────
    # Before the run loop: identify all ET_RECOVERY_RUNS in the queue, scan their
    # current status, and present the unified _prompt_et_batch_cfg screen.
    #
    # _prompt_et_batch_cfg returns (abl_set, et_set_all):
    #   abl_set     -- runs needing abliterated generation (mode a or b)
    #   et_set_all  -- runs needing base-model E_t pass (mode e or b)
    #
    # We split et_set_all into:
    #   _et_batch       -- mode-'e' runs (E_t only) -- skip abl loop, go straight to E_t pass
    #   _et_runs_both   -- mode-'b' runs (both) -- dispatch abl first, then add to E_t batch
    _et_runs_in_queue = [r for r in runs if r in _ET_RECOVERY_RUNS]
    _abl_set      = set()
    _et_batch     = set()
    _et_runs_both = set()

    if _et_runs_in_queue:
        _et_status = _scan_runs(paths, session.get('trials', 100))
        _overrides = session.get('et_mode_overrides') if os.environ.get('IOTA_HEADLESS') == '1' else None
        if _overrides:
            # Dashboard specified modes per-run -- build sets directly, skip prompt.
            _q_set = set(_et_runs_in_queue)
            _abl_set    = {int(rn) for rn, m in _overrides.items()
                           if m in ('a', 'b') and int(rn) in _q_set}
            _et_set_all = {int(rn) for rn, m in _overrides.items()
                           if m in ('e', 'b') and int(rn) in _q_set}
            ui.msg(f"  ET overrides from dashboard: {dict(_overrides)}")
        else:
            _abl_set, _et_set_all = _prompt_et_batch_cfg(
                set(_et_runs_in_queue), _et_status, paths, session.get('trials', 100)
            )
        _et_batch     = _et_set_all - _abl_set   # mode 'e': E_t only, skip abl loop
        _et_runs_both = _abl_set & _et_set_all    # mode 'b': abl then E_t
    # ── End upfront ET config ──────────────────────────────────────────────────

    for run_index, run_num in enumerate(runs):
        # Save resume state -- remaining runs from this point forward.
        # Dashboard Resume button reads this to re-launch from here on abort.
        _save_resume_state(runs[run_index:])

        # Update run_index and run_start_ts so the dashboard overall-run progress bar
        # advances and the elapsed timer survives page reload. FIX-7 (v34.2).
        try:
            from orchestration_core import update_dashboard_ctx
            update_dashboard_ctx(run_index=run_index, run_start_ts=time.time())
        except Exception:
            pass

        # Flush VRAM between runs -- catches any residual state from prior run
        # BUG-43B fix (v43.0.9): synchronize() guarded by 15s timeout thread
        # -- same fix as unload_model. Without the guard this flush hangs the
        # entire queue between runs on Windows CUDA sync stalls.
        try:
            import torch, gc, threading as _th
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _fs_done = _th.Event()
                def _fsync():
                    try: torch.cuda.synchronize()
                    except Exception: pass
                    finally: _fs_done.set()
                _ft = _th.Thread(target=_fsync, daemon=True)
                _ft.start()
                if not _fs_done.wait(timeout=15):
                    print("  [inter-run flush] cuda.synchronize() timed out -- skipping", flush=True)
        except Exception:
            pass

        # ── Per-run prerequisite gate (v30.4) ─────────────────────────────────
        # Fresh scan immediately before each run -- a prior run in this session
        # may just have satisfied a prerequisite.  If prereqs are still unmet,
        # warn hard and prompt.  This is the authoritative check; the pre-flight
        # above is just an early warning before model load.
        _run_status  = _scan_runs(paths, session.get('trials', 100))
        _run_problems = _check_prereqs(run_num, _run_status)
        if _run_problems:
            _prereq_warning_block(run_num, _run_problems)
            if os.environ.get('IOTA_HEADLESS') == '1':
                ui.warn(f"[headless] Skipping Run {run_num:04d} -- prerequisites not met.")
                continue
            ui.blank()
            ui.opt("c", f"Continue -- run {run_num:04d} anyway  (results may be invalid)")
            ui.opt("s", f"Skip     -- skip Run {run_num:04d} and continue to next run")
            ui.opt("a", "Abort    -- stop the entire session")
            ui.blank()
            _gate = input("  > ").strip().lower()
            if _gate == 'a':
                ui.warn("Session aborted.")
                return
            elif _gate != 'c':   # default to skip on any other input
                ui.warn(f"Skipping Run {run_num:04d}.")
                continue
            ui.blank()
        # ── End per-run gate ───────────────────────────────────────────────────

        module_name, desc = RUN_MAP[run_num]
        # Log skip explicitly when a run is already complete -- no silent no-ops.
        # Exception: Run 0001 always falls through to _prompt_pass_cfg_r01 even when
        # done, so the user can re-trigger individual passes (e.g. vectors after
        # BUG-42B fix) without having to clear files manually.
        _run_status2 = _scan_runs(paths, session.get('trials', 100))
        _this_status = _run_status2.get(run_num)
        if _this_status == 'done' and run_num != 1 and run_num not in _ANALYSIS_RUNS:
            ui.ok(f"Run {run_num:04d} already complete -- skipping.")
            continue

        # v0.79.4.15: Legacy interactive-mode ET dispatch block.
        # Scanner no longer emits 'needs_et' (Run 0016 owns ET state now),
        # so the needs_et branches below are effectively dead. The block
        # is preserved for the interactive _run_session path where the
        # user explicitly chose 'E_t base pass only' via _prompt_et_batch_cfg
        # on a 'missing' source. Headless / Flask-launched sessions use
        # Run 0016 as a first-class queue entry and never enter this block.
        # ET recovery runs: mode was assigned upfront by _prompt_et_batch_cfg.
        # Covers both 'needs_et' (CSV+S_t done, E_t missing) and 'missing'
        # (nothing collected yet, e.g. runs 0013-0015 on first collection).
        #   E_t-only  (not in _abl_set, in _et_batch)  → skip abl dispatch
        #   abl-only  (in _abl_set, not in _et_runs_both) → dispatch normally
        #   both      (in _abl_set, in _et_runs_both)  → dispatch, then E_t batch
        #   skip      (not in either)                  → skip entirely
        if run_num in _ET_RECOVERY_RUNS and _this_status in ('needs_et', 'missing'):
            if run_num not in _abl_set:
                # E_t-only or skip -- no abliterated dispatch needed
                if run_num in _et_batch:
                    ui.msg(f"  Run {run_num:04d} -- queued for E_t base pass only.")
                else:
                    ui.msg(f"  Run {run_num:04d} -- skipped (mode: skip).")
                continue
            # BUG-43C: needs_et means CSV is complete -- abliterated dispatch
            # would load the model, find get_trials_to_run()=[], do nothing, then
            # hang on unload. Redirect to E_t batch and skip the abl loop.
            # v0.79.4.15: dead branch (scanner no longer emits needs_et) but
            # retained defensively in case of disk state from pre-0.79.2 tools.
            if _this_status == 'needs_et':
                _et_batch.add(run_num)
                ui.msg(f"  Run {run_num:04d} -- CSV complete (needs_et), "
                       f"redirected to E_t base pass only.")
                continue
            # else: mode is 'a' or 'b', status is 'missing' -- fall through to abliterated dispatch

        # Run 0001: prompt for per-pass configuration before model load.
        # When 'done', all passes default to False -- user must explicitly enable.
        # In headless mode, use r01_passes from session if set by the dashboard
        # launch buttons; otherwise default to all-True (full run).
        # v0.80.0.44: key renamed r19_passes → r01_passes. Old sessions
        # may still have r19_passes; accept both during migration window.
        if run_num == 1:
            _r01p = session.get('r01_passes') or session.get('r19_passes')
            if os.environ.get('IOTA_HEADLESS') == '1' and _r01p:
                _pass_cfg = _r01p
                ui.msg(f"  Run 0001 -- using dashboard pass selection: "
                       f"{[k for k,v in _pass_cfg.items() if v]}")
            else:
                _pass_cfg = _prompt_pass_cfg_r01(paths, session.get('trials', 100))
            if _pass_cfg is None or not any(_pass_cfg.values()):
                ui.warn(f"Run 0001 -- all passes skipped.")
                try:
                    from runners_p1 import _print_nietzsche
                    _print_nietzsche(session.get('model_family', 'unknown'))
                except Exception:
                    pass
                continue
            session['_pass_cfg'] = _pass_cfg

        # Patching runs (21, 42): pass dashboard mode selection through session.
        # Dashboard sets patch_modes_21 / patch_modes_42 via /run POST body.
        # graft_patching.run() and run42_layer_isolation.run() read _patch_modes
        # to restrict which modes are collected (None = run all modes).
        # Always clear any stale override so a headless re-run without an explicit
        # selection runs all modes.  BUG-R21-SKIP fix (v42.6.0).
        if run_num == 3:
            _r26c = session.get('r26_cells')
            if _r26c:
                session['_r26_cells'] = _r26c
                ui.msg(f"  Run 0003 -- cell filter from dashboard: {len(_r26c)} cell(s)")
            else:
                session.pop('_r26_cells', None)
        if run_num in _MC_RUNS and run_num not in (3, ):
            _mcc = session.get('mc_conds')
            if _mcc:
                session['_mc_conds'] = _mcc
                ui.msg(f"  Run {run_num:04d} -- condition filter from dashboard: {_mcc}")
            else:
                session.pop('_mc_conds', None)
        if run_num == 17:
            _pm21 = session.get('patch_modes_21')
            if _pm21:
                session['_patch_modes'] = _pm21
                ui.msg(f"  Run 0017 -- mode filter from dashboard: {_pm21}")
            else:
                session.pop('_patch_modes', None)
        elif run_num == 18:
            _pm42 = session.get('patch_modes_42')
            if _pm42:
                session['_patch_modes'] = _pm42
                ui.msg(f"  Run 0018 -- mode filter from dashboard: {_pm42}")
            else:
                session.pop('_patch_modes', None)
        elif run_num == 19:
            _pm53 = session.get('patch_modes_53')
            if _pm53:
                session['_patch_modes'] = _pm53
                ui.msg(f"  Run 0019 -- mode filter from dashboard: {_pm53}")
            else:
                session.pop('_patch_modes', None)

        ui.section(f"Run {run_num:04d} -- {desc}")
        if run_num in _ANALYSIS_RUNS:
            ui.msg(f"  [dedup] Running universal CSV dedup before Run {run_num:04d}...")
            _dedup_all_csvs(paths, session)
        try:
            # Run 0055: stats export at all temps + report generation
            # Run 0056: methodology calibration (foundations)
            # Run 0057: function-class sensitivity
            # Run 0058: lagrangian apparatus (v0.80.0.51 -- was paper assembly)
            # Run 0059: paper assembly (v0.80.0.51 -- was Run 0058)
            if run_num == 55:
                _run_54_stats_report(session)
            elif run_num == 56:
                _run_56_calibration(session)
            elif run_num == 57:
                _run_57_function_class_sensitivity(session)
            elif run_num == 58:
                _run_58_apparatus(session)
            elif run_num == 59:
                _run_59_paper_assembly(session)
            else:
                mod = _load_module(module_name)
                mod.run(run_num, session, paths)
        except KeyboardInterrupt:
            ui.warn("Interrupted."); break
        except Exception as e:
            import traceback
            ui.err(f"Run {run_num} failed: {e}")
            traceback.print_exc()
            if not ui.timed_confirm("Continue to next run?", timeout=30, default_yes=True):
                break
        finally:
            session.pop('_pass_cfg', None)
            session.pop('_patch_modes', None)
            session.pop('_r26_cells', None)
            session.pop('_mc_conds', None)

        # Auto-add to E_t batch after abliterated dispatch for mode-'b' runs.
        # _et_runs_both = runs the user assigned mode 'b' (abliterated + E_t).
        # Mode 'e' runs are already in _et_batch. Mode 'a' runs are not added.
        if run_num in _et_runs_both:
            _et_batch.add(run_num)
            ui.msg(f"  Run {run_num:04d} queued for E_t base pass.")

    # E_t recovery pass -- runs batched through base model in a single load.
    # Fires AFTER all generation runs so Run 0001 (which provides the global C_t
    # constant) has always completed first if it was in the same session.
    if _et_batch:
        ui.blank()
        ui.section(f"E_t Recovery Pass -- {len(_et_batch)} run(s)")
        ui.msg(f"  Processing {sorted(_et_batch)} ...")
        if os.environ.get('IOTA_SINGLE_RUN') == '1':
            # Running in isolated subprocess -- write marker for orchestrator to
            # fire ET recovery as its own subprocess after VRAM clears.
            import json as _json
            _et_marker = os.path.join(ROOT, '.iota_et_pending.json')
            try:
                # Merge with any existing pending batch
                _existing = []
                if os.path.exists(_et_marker):
                    with open(_et_marker) as _f:
                        _existing = _json.load(_f).get('runs', [])
                _merged = sorted(set(_existing) | _et_batch)
                with open(_et_marker, 'w') as _f:
                    _json.dump({'runs': _merged}, _f)
                ui.ok(f"  ET batch written to marker -- orchestrator will fire isolated pass: {_merged}")
            except Exception as _e:
                ui.warn(f"  Could not write ET marker ({_e}) -- running inline (may OOM)")
                try:
                    import runners as _runners_mod
                    _runners_mod._run_et_recovery(session, paths, sorted(_et_batch))
                except Exception as _e2:
                    ui.err(f"E_t recovery failed: {_e2}")
        else:
            try:
                import runners as _runners_mod
                _runners_mod._run_et_recovery(session, paths, sorted(_et_batch))
            except KeyboardInterrupt:
                ui.warn("E_t recovery interrupted.")
            except Exception as e:
                import traceback
                ui.err(f"E_t recovery failed: {e}")
                traceback.print_exc()

    # Session complete -- clear the resume marker so dashboard doesn't offer stale resume.
    _clear_resume_state()


# ── Live run status scanner ───────────────────────────────────────────────────

def _read_csv_cols(fpath, want_cols):
    """Read specific columns from a CSV by header name."""
    from orchestration_core import canonical_read
    return canonical_read(fpath, want_cols)

from scanner import _MC_RUNS, _MC_TURNS, scan_runs as _scan_runs




def _what_next(paths, n_trials=100):
    """
    Print a full per-run status table grouped by phase,
    followed by a concrete recommendation for what to run next.
    """
    status = _scan_runs(paths, n_trials)
    if not status:
        ui.warn("Could not read data directory -- set model and run first.")
        return

    try:
        from dependency_map import DEPENDENCY_MAP, RUN_CSV
    except Exception:
        ui.warn("Could not load dependency map.")
        return

    W = 70
    SYM = {'done': '✓', 'partial': '~', 'missing': '·'}

    # Build run → description from RUN_MAP
    run_desc = {r: desc for r, (_, desc) in RUN_MAP.items()}

    phase_labels = {
        1: "PHASE 1 -- PROOF  (establish the phenomenon)",
        2: "PHASE 2 -- QUANTIFY  (measure and decompose R)",
        3: "PHASE 3 -- EXTEND  (new hypotheses)",
        4: "PHASE 4 -- PROVE  (coherence transfer + contradiction recovery)",
        99: "VALIDATION / ANALYSIS",
    }

    # Assign display phases (Run 0033 + analysis runs need grouping)
    def _display_phase(r):
        if r in (33, 42, 43, 47, 48, 49, 51):  # v0.79.4.0: analysis-set remap (old 41,25,27,32,33,34,40)
            return 99
        for hid, h in DEPENDENCY_MAP.items():
            if r in h['data_runs']:
                return h['phase']
        return 99

    from collections import defaultdict
    by_phase = defaultdict(list)
    for r in sorted(status.keys()):
        by_phase[_display_phase(r)].append(r)

    print()
    print("=" * W)
    print("  IOTA RUN STATUS")
    print("=" * W)

    for phase in sorted(by_phase.keys()):
        print()
        print(f"  {phase_labels.get(phase, f'PHASE {phase}')}")
        print("-" * W)
        for r in by_phase[phase]:
            sym   = SYM.get(status.get(r, 'missing'), '·')
            desc  = run_desc.get(r, '')
            print(f"  {sym}  Run {r:02d}  {desc}")

    print()
    print("=" * W)

    # Recommendation
    missing = sorted(r for r, s in status.items() if s in ('missing', 'partial'))
    partial_runs = sorted(r for r, s in status.items() if s == 'partial')
    done_runs    = sorted(r for r, s in status.items() if s == 'done')

    print()
    print("  WHAT TO DO NEXT")
    print("-" * W)

    if not missing:
        print("  All runs complete. Run [s] Stats + figures to export results.")
    else:
        # Phase 1 gaps
        p1_missing = [r for r in missing if _display_phase(r) == 1]
        p2_missing = [r for r in missing if _display_phase(r) == 2]
        p3_missing = [r for r in missing if _display_phase(r) == 3]
        rest_missing = [r for r in missing if _display_phase(r) == 99]

        if p1_missing:
            print(f"  Phase 1 incomplete -- missing runs: {p1_missing}")
            print(f"  Suggested: select runs → [1] Phase 1 -- Proof")
        if p2_missing:
            print(f"  Phase 2 incomplete -- missing runs: {p2_missing}")
        if p3_missing:
            print(f"  Phase 3 incomplete -- missing runs: {p3_missing}")
        if rest_missing:
            print(f"  Validation/analysis incomplete -- missing runs: {rest_missing}")
        if not p1_missing and not p2_missing and not p3_missing:
            print(f"  Only analysis/validation runs remain: {rest_missing}")
            print("  Suggested: select runs → [a] Analysis only")

    if partial_runs:
        print(f"\n  Partial data (resumed or interrupted): {partial_runs}")
        print("  These will resume from where they stopped.")

    print()
    input("  Press Enter to continue...")


def _clear_stale_status():
    """Placeholder since the grid cache removal in 0.77.1.0.

    Prior to 0.77.1.0 the dashboard kept a _modelCache of grid state
    that required explicit invalidation on model switch. The cache
    had no invalidation logic and produced stale-state bugs, so the
    entire cache plus its five helpers were ripped out. The scanner
    now reads filesystem directly on every request; there is nothing
    to clear. This function is kept as a no-op call site in case a
    future cache re-emerges and needs hooking."""
    # Force fresh scanner read -- no cache to clear, scanner reads filesystem directly.
    pass


# ── Run presets ───────────────────────────────────────────────────────────────

PRESETS = {
    'f':  ("Full",              "1-56",
           "All runs in execution order (Phase A → B → D → EFC → Per-Temp → Pooled → Cross-Model → Output)"),
    'c':  ("Collection",        "1-40",
           "All GPU generation runs -- null baselines, introspection, arithmetic, priming, patching, multi-cond, perturbation, impossibility. Skips no-GPU analysis."),
    'a':  ("Analysis (per-temp)", "41-49",
           "Per-temperature analysis: POOL_DIM calibration (41), decomposition (42), permutation (43), per-condition R (44, 45), MLP validation (46), Granger A/B (47, 48), baseline swap (49)"),
    'p':  ("Pooled",            "50,51",
           "Pooled analysis within one model across temperatures: cross-temp synthesis (50), pooled decomposition + sobol (51)"),
    'x':  ("Cross-Model",       "52,53,54",
           "Cross-architecture comparison: pairwise R (52), condition concordance Kendall W (53), cross-model summary table (54)"),
    'o':  ("Paper Output",      "55,56",
           "Stats report + paper assembly: R55 stats + master_results.json, R56 paper JSON copy + all_models_master.json"),
    '1':  ("Phase 1 -- Proof",   "1-2,4-15,17-18,29-32,39-40",
           "Historical: null baselines, introspection, arithmetic, perturbation, impossibility, priming, tokenization, patching"),
    '2':  ("Phase 2 -- Quantify", "3,22-28,33,41-43,51",
           "Historical: temperature grid, saturation, layer locality, self-reference, confound, persistence, cross-instance, coherence levels, validation, POOL_DIM calibration, E+C+R decomposition, permutation sensitivity, pooled"),
    '3':  ("Phase 3 -- Extend",  "34-38",
           "Historical: layer depth, output similarity, condition transfer, entropy shape, hesitation"),
    '4':  ("Phase 4 -- Prove",   "25-26",
           "Historical: coherence transfer (H33) and contradiction recovery (H35a/H35b)"),
    'm':  ("Custom",            None,
           "Enter run numbers manually"),
}


def _pick_runs(session, paths=None):
    """Interactive run preset selector. Updates session['runs'] and saves."""
    # Live status scan for pending counts
    status = _scan_runs(paths, session.get('trials', 100)) if paths else {}

    def _pending_count(runs_str):
        if not runs_str or not status:
            return ''
        runs = _parse_runs(runs_str)
        n = sum(1 for r in runs if status.get(r, 'missing') != 'done')
        return f"  ({n} pending)" if n else "  (all done ✓)"

    while True:
        ui.section("Select Runs")
        ui.blank()
        for key, (name, runs_str, desc) in PRESETS.items():
            pending = _pending_count(runs_str)
            label   = f"{name:<26} {runs_str if runs_str else '(manual entry)'}{pending}"
            ui.opt(key, label)
            ui.msg(f"         {desc}")
            ui.blank()
        ui.opt("b", "Back")
        ui.blank()
        raw = input("  > ").strip().lower()
        if raw == 'b':
            return
        if raw in PRESETS:
            name, runs_str, _ = PRESETS[raw]
            if raw == 'c':
                ui.blank()
                ui.msg("  Enter run numbers manually.")
                ui.msg("  Examples:  26        single run")
                ui.msg("             1-21,42   range + individual")
                ui.msg("             1,3,19    comma list")
                ui.blank()
                val = input("  Runs: ").strip()
                if val:
                    session['runs'] = val
                    ui.save_session(session)
                    ui.ok(f"Runs set to: {val}")
            else:
                session['runs'] = runs_str
                ui.save_session(session)
                ui.ok(f"Runs set to: {name}  ({runs_str})")
            return
        ui.warn("Invalid selection.")


# ── Menu ──────────────────────────────────────────────────────────────────────

def _ensure_hf_token(session):
    """Prompt for HuggingFace token once at startup if not already set."""
    if session.get('hf_token'):
        return
    ui.section("HuggingFace Token")
    ui.msg("Some models require a HuggingFace token (e.g. gated Llama models).")
    ui.msg("Token is saved to session and used for all downloads and loads.")
    ui.blank()
    raw = input("  HF token (Enter to skip): ").strip()
    if raw:
        session['hf_token'] = raw
        ui.save_session(session)
        ui.ok("Token saved.")
    else:
        ui.msg("Skipped. Public models will work. Gated models will fail at load.")
    ui.blank()


def _run_54_stats_report(session):
    """Run 0055: export stats at all temperatures + generate report.

    Produces hypothesis_outcomes.json at each temp via export_stats,
    then generates figures + markdown report via report.py.
    Writes R54_stats_report.json stamp to pooled/analysis/ when done.
    """
    from cartography import DATA, get_paths as _gp, get_pooled_paths as _gpp, get_family_size_dir
    import json, datetime

    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')
    base_dir = os.path.join(get_family_size_dir(family, size), variant)

    if not os.path.isdir(base_dir):
        ui.err(f"No data directory: {base_dir}")
        return

    # Find temperature conditions
    _COND_TEMPS = {
        'deterministic': 0.0, 'temp_0.2': 0.2, 'temp_0.4': 0.4,
        'temp_0.6': 0.6, 'temp_0.8': 0.8, 'temp_1.0': 1.0,
    }
    conditions = sorted(d for d in os.listdir(base_dir)
                        if d != 'pooled' and d in _COND_TEMPS
                        and os.path.isdir(os.path.join(base_dir, d, 'csv')))

    if not conditions:
        ui.err("No temperature data found.")
        return

    # Step 1: Stats export at each temperature
    ui.msg(f"  Stats export across {len(conditions)} temperatures...")
    _es = _load_module('export_stats')
    for cond in conditions:
        temp = _COND_TEMPS[cond]
        _sess = dict(session)
        _sess['temperature'] = temp
        _paths = _gp(family, size, variant, temp)
        try:
            if hasattr(_es, 'run'):
                _es.run(_sess, _paths)
                ui.msg(f"    {cond} done.")
        except Exception as _e:
            ui.warn(f"    {cond} failed: {_e}")

    # Step 2: Report generation removed (v0.74.0.0)
    # report.py figures are legacy -- superseded by generate_paper_figures().
    # report.py retained in codebase for REPORT.md generation if needed manually.

    # Step 3: Cross-temperature hypothesis status change table (v0.71.0.11)
    ui.msg("  Generating cross-temperature status table...")
    try:
        _all_outcomes = {}
        for cond in conditions:
            temp = _COND_TEMPS[cond]
            _paths49 = _gp(family, size, variant, temp, create_dirs=False)
            _oj = os.path.join(_paths49['analysis'], 'hypothesis_outcomes.json')
            if os.path.exists(_oj):
                with open(_oj) as _f49: _all_outcomes[temp] = json.load(_f49)
        if len(_all_outcomes) >= 2:
            # Build status table: {h_id: {temp: status}}
            all_hids = set()
            for d in _all_outcomes.values():
                all_hids.update(d.keys())
            status_table = []
            for h_id in sorted(all_hids):
                row = {'hypothesis': h_id}
                statuses = set()
                for temp in sorted(_all_outcomes.keys()):
                    s = _all_outcomes[temp].get(h_id, {})
                    st = s.get('status', 'pending') if isinstance(s, dict) else 'pending'
                    row[f'T{temp:.1f}'] = st
                    if st != 'pending':
                        statuses.add(st)
                row['status_changes'] = len(statuses) > 1
                status_table.append(row)
            n_changes = sum(1 for r in status_table if r['status_changes'])
            _pp49 = _gpp(family, size, variant, create_dirs=True)
            _ct_path = os.path.join(_pp49['analysis'], 'cross_temp_status.json')
            with open(_ct_path, 'w') as _f:
                json.dump({'table': status_table, 'n_hypotheses': len(status_table),
                           'n_status_changes': n_changes,
                           'temperatures': sorted(_all_outcomes.keys())}, _f, indent=2)
            ui.ok(f"  Cross-temp status: {n_changes} hypotheses change across temperatures → {_ct_path}")
        else:
            ui.warn("  Cross-temp status: need ≥ 2 temperatures with outcomes")
    except Exception as _e:
        ui.warn(f"  Cross-temp status table failed: {_e}")

    # Step 4: Run 0001 three-variant delta table (v0.71.0.11)
    ui.msg("  Generating three-variant comparison table...")
    try:
        import csv as _csv19
        from cartography import RUN_CSV as _RC19
        _r01_csv = _RC19.get("0001", 'R0001_null.csv')  # v0.79.4.0: old R19 → new 0001
        _variant_data = {}  # {temp: {model: {similarity: [], sim: [], ent: []}}}
        for cond in conditions:
            temp = _COND_TEMPS[cond]
            _p01 = _gp(family, size, variant, temp, create_dirs=False)
            _csv_path = os.path.join(_p01.get('csv', ''), _r01_csv)
            if not os.path.exists(_csv_path):
                continue
            with open(_csv_path, 'r', encoding='utf-8', errors='replace') as _f19:
                reader = _csv19.DictReader(_f19)
                for row in reader:
                    # v0.79.4.0 renumber: old R19 → new 0001
                    if row.get('run_mode') not in ('0001', '19') or row.get('priming') == '1':
                        continue
                    m = row.get('model', 'unknown')
                    _variant_data.setdefault(temp, {}).setdefault(m, {'similarity': [], 'sim': [], 'ent': []})
                    for metric, key in [('similarity', 'state_similarity_index'), ('sim', 'layer_sim_mean'), ('ent', 'mean_logit_entropy')]:
                        try:
                            v = float(row.get(key, ''))
                            if v == v:  # not NaN
                                _variant_data[temp][m][metric].append(v)
                        except (ValueError, TypeError):
                            pass
        if _variant_data:
            import numpy as _np19
            _tv_table = []
            for temp in sorted(_variant_data.keys()):
                for m, metrics in sorted(_variant_data[temp].items()):
                    _tv_table.append({
                        'temperature': temp, 'model': m,
                        'mean_similarity': float(_np19.mean(metrics['similarity'])) if metrics['similarity'] else None,
                        'mean_layer_sim': float(_np19.mean(metrics['sim'])) if metrics['sim'] else None,
                        'mean_entropy': float(_np19.mean(metrics['ent'])) if metrics['ent'] else None,
                        'n': len(metrics['similarity']),
                    })
            _pp_tv = _gpp(family, size, variant, create_dirs=True)
            _tv_path = os.path.join(_pp_tv['analysis'], 'three_variant_comparison.json')
            with open(_tv_path, 'w') as _f:
                json.dump({'table': _tv_table, 'n_temps': len(_variant_data)}, _f, indent=2)
            ui.ok(f"  Three-variant table: {len(_tv_table)} entries → {_tv_path}")
        else:
            ui.warn("  Three-variant table: no Run 0001 data found")
    except Exception as _e:
        ui.warn(f"  Three-variant table failed: {_e}")

    # Generate per-model pooled diagnostic figures (v0.75.0.6)
    # Writes to {model}/pooled/visuals/ -- FIG07, FIG11, FIG12 + others.
    ui.msg("  Generating per-model pooled diagnostics...")
    try:
        import export_stats as _es_pooled
        _es_pooled.generate_paper_figures(session)
    except Exception as _e:
        ui.warn(f"  Pooled diagnostics failed: {_e}")

    # Build master results file (v0.72.2.0)
    ui.msg("  Building master results file...")
    try:
        import export_stats as _es_master
        _es_master.build_master_results(session)
    except Exception as _e:
        ui.warn(f"  Master results failed: {_e}")

    # Write stamp file
    _pp = _gpp(family, size, variant, create_dirs=True)
    stamp = {
        "run": 54,
        "timestamp": datetime.datetime.now().isoformat(),
        "conditions": conditions,
        "model": f"{family}/{size}/{variant}",
    }
    stamp_path = os.path.join(_pp['analysis'], 'Q0055_stats_report.json')
    with open(stamp_path, 'w') as f:
        json.dump(stamp, f, indent=2)
    ui.ok(f"  Run 0055 complete. Stamp: {stamp_path}")


def _discover_ready_models(session, ui_log=None):
    """Shared helper for Runs 0056 / 0057 / 0058 -- discover models that
    have at least one completed temperature's analysis (Q0043). Filters
    by session.cross_model_include if present (dashboard checkbox UI).

    v0.80.0.44: extracted from the old _run_55_paper_assembly so all three
    refactored runs share one model-discovery pass.
    """
    import export_stats as _es55
    models = _es55._discover_model_data({})
    if not models:
        if ui_log: ui_log("No models with analysis data found in base/")
        return []

    if ui_log:
        ui_log(f"  Discovered {len(models)} model(s):")
        for m in models:
            ui_log(f"    {m['label']}")

    ready_models = []
    for m in models:
        pooled_dir = os.path.join(m['data_dir'], 'pooled', 'analysis')
        has_stamp = os.path.exists(os.path.join(pooled_dir, 'Q0055_stats_report.json'))
        n_temps_with_analysis = 0
        for cond_name in ('deterministic', 'temp_0.2', 'temp_0.4', 'temp_0.6', 'temp_0.8', 'temp_1.0'):
            ana_dir = os.path.join(m['data_dir'], cond_name, 'analysis')
            if os.path.isdir(ana_dir):
                q34 = os.path.join(ana_dir, 'Q0043_sobol_partition.json')
                if os.path.exists(q34):
                    n_temps_with_analysis += 1
        m['n_temps_analyzed'] = n_temps_with_analysis
        m['has_stamp'] = has_stamp
        if n_temps_with_analysis >= 1:
            ready_models.append(m)
            if ui_log:
                ui_log(f"    {m['label']}: {n_temps_with_analysis}/6 temps analyzed" +
                       (" + R54 stamp" if has_stamp else ""))
        else:
            if ui_log:
                ui_log(f"    {m['label']}: no analysis -- excluded")

    _include = session.get('cross_model_include')
    if _include:
        ready_models = [m for m in ready_models if m['label'] in _include]
        if ui_log:
            ui_log(f"  Filtered to {len(ready_models)} model(s) by session selection")

    return ready_models


def _run_56_calibration(session):
    """Run 0056: Methodology calibration only.

    v0.80.0.44: split from old monolithic Run 0056 (paper assembly).
    Now Run 0056 does ONE job: produce fresh calibration artifacts
    (V5b synthetic, channel_marginal_nonlinearity, methodology
    calibration block). Run 0058 (paper assembly) hard-depends on
    this run's output being fresh.

    Why split: the old Run 0056 conflated three concerns (calibration
    integrity, function-class robustness, paper-write). Each had
    different failure modes and different recovery patterns. Splitting
    into 0056 / 0057 / 0058 means each run has one job, and Run 0058
    can refuse to fire if EITHER calibration or function-class
    sensitivity is incomplete -- preventing a paper from shipping
    without its load-bearing checks.

    Returns True if calibration is complete and fresh, False otherwise.
    """
    ui.section("Run 0056 -- Methodology Calibration")

    ready_models = _discover_ready_models(session, ui_log=ui.msg)
    if not ready_models:
        ui.err("No models ready for calibration.")
        return False

    cal_ok = False
    try:
        sys.path.insert(0, ROOT)
        import export_stats as _es_cal
        _es_cal._run_paper_calibration_phase(ui_log=ui.msg)
        cal_status = _es_cal.methodology_calibration_status()
        if cal_status.get('_any_missing') or cal_status.get('_any_stale'):
            failed = [k for k, v in cal_status.items()
                      if not k.startswith('_') and v in ('missing', 'stale')]
            ui.err(f"  Calibration incomplete: {', '.join(failed)}")
            ui.err(f"  Run 0058 (paper assembly) will refuse to fire.")
            return False
        cal_ok = True
        ui.ok(f"  Run 0056 complete -- calibration fresh, ready for downstream.")
    except Exception as _cal_e:
        ui.err(f"  calibration phase failed: {_cal_e}")
        return False

    # Write calibration manifest stamp so Run 0058's prereq check finds it
    try:
        from cartography import DATA as _DATA
        import json as _json, datetime as _dt
        _manifest_dir = os.path.join(_DATA, 'paper', 'calibration')
        os.makedirs(_manifest_dir, exist_ok=True)
        _manifest_path = os.path.join(_manifest_dir, 'Q0056_calibration_manifest.json')
        _manifest = {
            'run': '0056',
            'role': 'calibration_only',
            'cal_ok': cal_ok,
            'cal_status': _es_cal.methodology_calibration_status(),
            'timestamp': _dt.datetime.now().isoformat(),
        }
        with open(_manifest_path, 'w', encoding='utf-8') as _mf:
            _json.dump(_manifest, _mf, indent=2)
            _mf.flush()
            try:
                os.fsync(_mf.fileno())
            except Exception:
                pass
        ui.ok(f"  Calibration manifest: {_manifest_path}")
    except Exception as _me:
        ui.warn(f"  Calibration manifest write failed (non-fatal): {_me}")

    return cal_ok


def _run_57_function_class_sensitivity(session):
    """Run 0057: Function-class sensitivity analysis.

    v0.80.0.44: NEW. Validates §5.4's cross-cell asymmetry-vs-gap
    pattern under a third nonlinear function class (Random Forest),
    beyond the canon Ridge-vs-MLP pair. Reads canon Ridge/MLP per-
    channel R² from channel_marginal_nonlinearity.csv (produced by
    Run 0056 calibration) and canon Ridge/MLP partition fractions
    from results.json's per-cell measurements (or refits cheap if
    not yet stored). Fits RF on the same train/test split with
    matched protocol. Writes Q0057_function_class_sensitivity.json
    with per-cell breakdown + cross_cell_aggregates summary.

    Hard-depended on by Run 0058 (paper assembly). If RF underfits
    or if the result narrows the §5.4 mechanism account, the paper
    text in v0_13 reflects that honestly. Either way, the paper
    ships with this check or doesn't ship.

    Activation-protocol note: writerbot's original handoff specified
    ReLU MLP. Canon protocol uses tanh MLP. Run 0057 honors canon --
    pulls existing tanh-MLP values from canon CSVs, fits RF on the
    same protocol. The activation discrepancy is a v0_12 → v0_13
    paper-text correction independent of this run's outcome.
    """
    ui.section("Run 0057 -- Function-Class Sensitivity")

    try:
        sys.path.insert(0, ROOT)
        import subprocess as _sp
        import sys as _sys
        _script = os.path.join(ROOT, 'run_function_class_sensitivity.py')
        if not os.path.exists(_script):
            ui.err(f"  run_function_class_sensitivity.py not found at {_script}")
            raise RuntimeError(f"Run 0057 script not found at {_script}")
        ui.msg(f"  Running {os.path.basename(_script)}...")
        # v0.80.0.44: was _sp.run([...], cwd=ROOT) which let stdout flow
        # through to the parent (Flask log) but stderr was also inherited
        # -- and importantly, the previous "return False on rc!=0" was
        # silently swallowed by the dispatch loop's lack of bool-checking,
        # so a failed Run 0057 reported as "1 runs done, 0 skipped/failed."
        # Raise instead so the dispatch loop's standard exception handler
        # surfaces the failure properly.
        # v0.80.0.44: force unbuffered stdout/stderr in the subprocess.
        # Without -u, Python's stdout defaults to block-buffered (4-8KB)
        # when stdout is a pipe (which it is when launched by subprocess).
        # The parent (start_here.py under Flask) inherits a pipe fd from
        # Flask's spawn, and the subprocess inherits that pipe in turn.
        # Block-buffering means print(..., flush=True) flushes Python's
        # internal buffer but the OS-level pipe buffer can still hold
        # output for many seconds before the dashboard sees it.
        # PYTHONUNBUFFERED=1 in the env is equivalent to -u; either works.
        # -u is more explicit and survives env-var stripping.
        _env = dict(os.environ)
        _env['PYTHONUNBUFFERED'] = '1'  # belt-and-suspenders
        _proc = _sp.run([_sys.executable, '-u', _script],
                        cwd=ROOT, env=_env)
        if _proc.returncode != 0:
            ui.err(f"  Run 0057 exited with code {_proc.returncode}")
            ui.err(f"  Check stderr above for traceback. The run produced no usable output;")
            ui.err(f"  Run 0058 (paper assembly) will refuse to fire until this is resolved.")
            raise RuntimeError(f"Run 0057 exited with code {_proc.returncode}")
        ui.ok(f"  Run 0057 complete.")
        return True
    except Exception as _e:
        ui.err(f"  Run 0057 failed: {_e}")
        # v0.80.0.44: re-raise so dispatch loop's session-failure
        # accounting catches this. Returning False was being treated
        # as success ("1 runs done, 0 skipped/failed").
        raise


def _run_58_apparatus(session):
    """Run 0058: Lagrangian apparatus -- kraskov anchor, solver,
    V5d threshold, aggregator.

    v0.80.0.51 (stub): apparatus implementation is queued. This
    placeholder exists so the dispatcher chain works and Run 0059
    (paper assembly) can hard-gate on Run 0058's manifest output
    without firing into a void. Stub raises SystemExit(1) on any
    failure (gate or stub-script) so the dispatcher reports failure
    in the dashboard. Same fix pattern as Run 0057's 0.80.0.44 fix.

    Once implemented, the apparatus run will:
      - Run kraskov bias spike (locks bucket: strong / narrow / ordinal)
      - Run solver three-tier validation harness + linearization sweep
      - Calibrate V5d threshold τ (spread + AM-GM)
      - Run kraskov anchor producer over real cells
      - Run aggregator over real cells, emit per-cell q* + bounds
      - Write Q0058_apparatus_manifest.json summarizing all sub-phases

    HARD prereqs (when implemented):
      - Run 0056: foundations calibration
      - Run 0057: Q0057_function_class_sensitivity.json
    """
    from cartography import DATA
    sys.path.insert(0, ROOT)
    ui.section("Run 0058 -- Lagrangian Apparatus")

    # ── HARD GATE: foundations calibration (from Run 0056) ────────────
    # v0.80.0.52: failure paths raise instead of return. Returning was
    # treated by the dispatcher as success ("1 runs done, 0 skipped/
    # failed") and surfaced green in the dashboard, masking gate
    # failures and stub failures alike.
    try:
        import export_stats as _es_cal
        cal_status = _es_cal.methodology_calibration_status()
        if cal_status.get('_any_missing') or cal_status.get('_any_stale'):
            failed = [k for k, v in cal_status.items()
                      if not k.startswith('_') and v in ('missing', 'stale')]
            ui.err(f"  Calibration incomplete: {', '.join(failed)}")
            ui.err(f"  Run Run 0056 before apparatus.")
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as _ce:
        ui.err(f"  Calibration status check failed: {_ce}")
        raise SystemExit(1)

    # ── HARD GATE: function-class sensitivity (from Run 0057) ─────────
    _q57_path = os.path.join(DATA, 'paper', 'Q0057_function_class_sensitivity.json')
    if not os.path.exists(_q57_path):
        ui.err(f"  Q0057_function_class_sensitivity.json not found at {_q57_path}")
        ui.err(f"  Run Run 0057 before apparatus.")
        raise SystemExit(1)

    # ── Stub: invoke the apparatus orchestrator ───────────────────────
    # Stub raises SystemExit(1) by design; propagate it.
    try:
        import run_bayesian_apparatus as _la
        _la.main()
    except SystemExit as _se:
        if _se.code and _se.code != 0:
            ui.err(f"  Apparatus exited with code {_se.code}")
        raise
    except Exception as _e:
        ui.err(f"  Apparatus failed: {type(_e).__name__}: {_e}")
        raise SystemExit(1)


def _run_59_paper_assembly(session):
    """Run 0059: Paper assembly -- results.json + figures.

    v0.80.0.51: renumbered from Run 0058. Run 0058 is now the
    Bayesian apparatus (kraskov anchor, I-projection solver,
    V5d threshold, aggregator). Paper assembly is downstream of
    all three calibration runs and hard-gates on each.

    Builds results.json (which incorporates the calibration block
    from Run 0056, the function-class sensitivity block from Run
    0057, the apparatus block from Run 0058, and the H₀₅₈
    partition sanity check from cross_cell_aggregates), then
    renders figures.

    HARD prereqs (enforced by _PREREQS dict + checked here):
      - Run 0056: methodology_calibration block must be fresh
      - Run 0057: Q0057_function_class_sensitivity.json must exist
      - Run 0058: Q0058_apparatus_manifest.json must exist

    If any are missing, this run refuses to fire.
    """
    from cartography import DATA, get_paper_paths as _get_pp, get_pooled_paths as _gpp
    from cartography import get_paths as _gp

    ui.section("Run 0059 -- Paper Assembly")

    ready_models = _discover_ready_models(session, ui_log=ui.msg)
    if not ready_models:
        ui.err("No models ready for paper assembly.")
        return

    # ── HARD GATE: calibration freshness (from Run 0056) ──────────────
    try:
        sys.path.insert(0, ROOT)
        import export_stats as _es_cal
        cal_status = _es_cal.methodology_calibration_status()
        if cal_status.get('_any_missing') or cal_status.get('_any_stale'):
            failed = [k for k, v in cal_status.items()
                      if not k.startswith('_') and v in ('missing', 'stale')]
            ui.err(f"  Calibration incomplete: {', '.join(failed)}")
            ui.err(f"  Run Run 0056 first to refresh calibration.")
            return
    except Exception as _ce:
        ui.err(f"  Calibration status check failed: {_ce}")
        return

    # ── HARD GATE: function-class sensitivity (from Run 0057) ─────────
    _q57_path = os.path.join(DATA, 'paper', 'Q0057_function_class_sensitivity.json')
    if not os.path.exists(_q57_path):
        ui.err(f"  Q0057_function_class_sensitivity.json not found at {_q57_path}")
        ui.err(f"  Run Run 0057 before paper assembly.")
        return

    # ── HARD GATE: lagrangian apparatus (from Run 0058) ───────────────
    # v0.80.0.51: paper assembly is downstream of the apparatus.
    # If the apparatus hasn't run, results.json can't carry the
    # aggregator output that figures and §7-§9 of the merged paper
    # depend on.
    _q58_path = os.path.join(DATA, 'paper', 'Q0058_apparatus_manifest.json')
    if not os.path.exists(_q58_path):
        ui.err(f"  Q0058_apparatus_manifest.json not found at {_q58_path}")
        ui.err(f"  Run Run 0058 before paper assembly.")
        return

    # ── Phase B: build results.json ───────────────────────────────────
    try:
        sys.path.insert(0, ROOT)
        import export_stats as _es_aggr
        _es_aggr.build_all_masters(verbose=False, rebuild_per_model=False)
        ui.ok(f"  data/paper/results.json written (validated)")
    except Exception as _bmj_e:
        ui.err(f"  results.json build failed: {_bmj_e}")
        ui.err(f"  Aborting Phase C -- figures need results.json.")
        return

    # ── Phase C: figure generation ────────────────────────────────────
    try:
        import export_stats as _es_fig
        _es_fig._run_paper_figures_phase(ui_log=ui.msg)
    except Exception as _fig_e:
        ui.warn(f"  figure phase failed: {_fig_e}")

    # ── Completion summary ────────────────────────────────────────────
    from cartography import DATA as _DATA
    _paper_root = os.path.join(_DATA, 'paper')
    _n_figs = len([f for f in os.listdir(_paper_root)
                   if f.endswith(('.svg', '.pdf'))]) if os.path.isdir(_paper_root) else 0
    ui.ok(f"  Run 0059 complete. {_n_figs} figure files in {_paper_root}")


def _run_all_stats(session, force=False, backup=False):
    """Run all analysis + stats across every temperature condition, then Run 0051 pooled + report.

    v0.64.0.0: modes:
      - default (skip): skip if output JSON exists
      - force: delete existing JSONs and re-run
      - force+backup: copy existing JSONs to .bak/ then delete and re-run
    Always runs report at end.
    """
    from cartography import DATA, condition_name, get_paths as _gp, get_pooled_paths as _gpp
    from cartography import ANALYSIS_JSON as _AJ, get_family_size_dir

    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')
    base_dir = os.path.join(get_family_size_dir(family, size), variant)

    if not os.path.isdir(base_dir):
        ui.err(f"No data directory: {base_dir}")
        return

    # ── Gate: require all 6 temperatures ──────────────────────────────
    _REQUIRED = {'deterministic', 'temp_0.2', 'temp_0.4', 'temp_0.6', 'temp_0.8', 'temp_1.0'}
    _found = set()
    for d in os.listdir(base_dir):
        if d != 'pooled' and os.path.isdir(os.path.join(base_dir, d, 'csv')):
            _found.add(d)
    _missing = _REQUIRED - _found
    if _missing:
        ui.err(f"All Stats requires all 6 temperature rounds. Missing: {sorted(_missing)}")
        ui.msg("  Complete all temperature rounds first.")
        return

    mode_str = "overwrite+backup" if (force and backup) else "overwrite" if force else "skip existing"
    ui.section(f"ALL STATS -- mode: {mode_str}")
    ui.msg(f"  Model: {family}/{size}/{variant}")
    ui.blank()

    import analysis as _ana

    _COND_TEMPS = {
        'deterministic': 0.0, 'temp_0.2': 0.2, 'temp_0.4': 0.4,
        'temp_0.6': 0.6, 'temp_0.8': 0.8, 'temp_1.0': 1.0,
    }

    # ── Force mode: delete/backup existing analysis JSONs ─────────────
    _PER_TEMP_JSONS = ['Q0042_decomposition.json', 'Q0043_sobol_partition.json',
                       'Q0044_per_condition_R.json', 'Q0041_pool_dim_sweep.json',
                       'Q0045_fixed_dim_per_condition_R.json']
    _POOLED_JSONS = ['Q0051_pooled_decomposition.json', 'Q0051_pooled_sobol.json',
                     'Q0051_temperature_curve.json', 'Q0050_cross_temp_synthesis.json']

    if force:
        import datetime as _bdt, shutil as _bsh
        bak_dir = None
        if backup:
            bak_dir = os.path.join(base_dir, '.bak',
                                   _bdt.datetime.now().strftime('%Y%m%d_%H%M%S'))

        deleted = 0
        for cond in sorted(_found):
            ana = os.path.join(base_dir, cond, 'analysis')
            if not os.path.isdir(ana):
                continue
            for fname in _PER_TEMP_JSONS:
                fpath = os.path.join(ana, fname)
                if os.path.exists(fpath):
                    if bak_dir:
                        dest = os.path.join(bak_dir, cond, 'analysis', fname)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        _bsh.copy2(fpath, dest)
                    os.remove(fpath)
                    deleted += 1

        pooled_ana = os.path.join(base_dir, 'pooled', 'analysis')
        if os.path.isdir(pooled_ana):
            for fname in _POOLED_JSONS:
                fpath = os.path.join(pooled_ana, fname)
                if os.path.exists(fpath):
                    if bak_dir:
                        dest = os.path.join(bak_dir, 'pooled', 'analysis', fname)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        _bsh.copy2(fpath, dest)
                    os.remove(fpath)
                    deleted += 1
            # Delete stale report figures
            fig_dir = os.path.join(pooled_ana, 'figures')
            if os.path.isdir(fig_dir):
                import glob as _fglob
                for f in _fglob.glob(os.path.join(fig_dir, '*.png')):
                    os.remove(f)
                    deleted += 1
                rmd = os.path.join(pooled_ana, 'REPORT.md')
                if os.path.exists(rmd):
                    os.remove(rmd)
                    deleted += 1

        if backup and bak_dir:
            ui.msg(f"  Backed up and deleted {deleted} files → {bak_dir}")
        else:
            ui.msg(f"  Deleted {deleted} existing analysis files.")
        ui.blank()

    # ── Step 1: POOL_DIM calibration at each temperature ──────────────
    ui.section("Step 1 -- POOL_DIM Calibration (Run 0041 per temperature)")
    conditions = sorted(_found)
    for cond in conditions:
        temp = _COND_TEMPS.get(cond, 0.0)
        paths = _gp(family, size, variant, temp)
        dim = _ana.load_optimal_pool_dim(paths['analysis'])
        if dim is not None and not force:
            ui.msg(f"  {cond}: POOL_DIM={dim} (cached)")
            continue
        ui.msg(f"  {cond}: running sweep...")
        cal_session = dict(session)
        cal_session['temperature'] = temp
        try:
            _ana.run(41, cal_session, paths)  # v0.79.4.0: old 45 (POOL_DIM) → new 41
        except Exception as _e:
            ui.warn(f"  Run 0041 at {cond} failed: {_e}")

    # ── Step 2: Analysis per temperature ──────────────────────────────
    _ANALYSIS_RUNS_ORDERED = [42, 43, 44, 45, 47, 48, 49]  # v0.79.4.0 renumbered

    ui.section(f"Step 2 -- Analysis + Stats across {len(conditions)} conditions")
    ui.msg(f"  Analysis runs per condition: {_ANALYSIS_RUNS_ORDERED}")
    ui.blank()

    for ci, cond in enumerate(conditions, 1):
        temp = _COND_TEMPS.get(cond, 0.0)
        ui.section(f"[{ci}/{len(conditions)}] {cond}  (T={temp})")

        session['temperature'] = temp
        paths = _gp(family, size, variant, temp)

        for run_num in _ANALYSIS_RUNS_ORDERED:
            try:
                # No outer skip gate -- analysis.py run() handles resume-aware
                # completeness checking (v0.75.2.2). Force flag still works:
                # _run_all_stats with force=True deletes JSONs before this loop.
                ui.msg(f"  Run {run_num}...")
                _ana.run(run_num, session, paths)
            except Exception as _e:
                ui.warn(f"  Run {run_num} failed: {_e}")

        # Stats export for this temperature
        try:
            ui.msg(f"  Stats export...")
            _mod = _load_module('export_stats')
            if hasattr(_mod, 'run'):
                _mod.run(session, paths)
        except Exception as _e:
            ui.warn(f"  Stats export failed for {cond}: {_e}")

        ui.ok(f"  {cond} complete.")
        # v0.75.2.2: qcache_cleanup removed. Cache persists for instant re-entry.
        # Stale caches cleaned inside _qcache_save.
        ui.blank()

    # ── Step 3: Pooled + cross-temperature analysis ──────────────────
    ui.section("Step 3 -- Pooled + Cross-Temperature Analysis")
    paths_pooled = _gp(family, size, variant, 0.0)
    _pp = _gpp(family, size, variant, create_dirs=True)

    ui.msg("  Run 0050 (Cross-temperature synthesis)...")
    _q47_exists = os.path.exists(os.path.join(_pp['analysis'], 'Q0050_cross_temp_synthesis.json'))
    if _q47_exists and not force:
        ui.msg("    skipped (output exists)")
    else:
        try:
            _ana.run(50, session, paths_pooled)  # v0.79.4.0: old 47 (cross-temp) → new 50
        except Exception as _e:
            ui.warn(f"  Run 0050 failed: {_e}")

    ui.msg("  Run 0051 (Pooled decomposition)...")
    _q40_exists = os.path.exists(os.path.join(_pp['analysis'], 'Q0051_pooled_decomposition.json'))
    if _q40_exists and not force:
        ui.msg("    skipped (output exists)")
    else:
        try:
            _ana.run(51, session, paths_pooled)  # v0.79.4.0: old 40 (pooled) → new 51
        except Exception as _e:
            ui.warn(f"  Run 0051 failed: {_e}")

    # ── Step 4: Generate report ──────────────────────────────────────
    ui.section("Step 4 -- Report Generation")
    try:
        import report as _rpt
        _rpt.generate(session=session)
    except Exception as _e:
        ui.warn(f"  Report generation failed: {_e}")

    ui.blank()
    ui.ok(f"ALL STATS complete -- {len(conditions)} conditions processed.")


def _run_all_temps_analysis(session, runs_str):
    """Run specified runs at every temperature directory that has data.

    Usage: --all-temps-runs 21       (GPU run -- subprocess per temp)
           --all-temps-runs 0049,0042,0043 (analysis -- direct call)
           --all-temps-runs 0017,0043,0051 (mixed -- GPU first, then analysis, pooled last)

    GPU runs are spawned as isolated subprocesses per temperature (same as
    _run_session_isolated) with VRAM wait between each. Analysis runs call
    analysis.run() directly. Pooled runs (47, 40) run once at the end.
    """
    from cartography import DATA, get_paths as _gp, get_family_size_dir
    import analysis as _ana

    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')
    base_dir = os.path.join(get_family_size_dir(family, size), variant)

    if not os.path.isdir(base_dir):
        ui.err(f"No data directory: {base_dir}")
        return

    # v0.71.0.11: Auto-detect model_name from hidden state files on disk.
    # The session's model_name can drift if the user switches models in the
    # dashboard -- model_family/size/variant stay correct but model_name
    # gets overwritten with the new model's display name.
    #
    # Infers model_name from any Run 0003 hidden state filename across all
    # temperature directories when the session's model_name differs from
    # what's on disk (e.g., user selected a different model key but the
    # data was collected under an earlier key).
    _session_mn = session.get('model_name', '')
    for _det_dir_name in ('deterministic', 'temp_0.2', 'temp_0.0'):
        _det_hid = os.path.join(base_dir, _det_dir_name, 'hidden_states')
        if os.path.isdir(_det_hid):
            import glob as _glob_det
            from cartography import sanitize as _san_det, run_prefix as _rp_det
            _det_pat = os.path.join(_det_hid, f"{_rp_det(3)}0003_*_trial*_turn01.npy")
            _det_hits = [f for f in sorted(_glob_det.glob(_det_pat))
                         if not f.endswith('_alllayers.npy')
                         and '_emb.npy' not in f]
            if _det_hits:
                _pfx = f"{_rp_det(3)}0003_"
                _disk_mn = os.path.basename(_det_hits[0]).split(_pfx)[1].split("_trial")[0]
                if _san_det(_session_mn) != _disk_mn:
                    session['model_name'] = _disk_mn  # silently correct
                break

    _COND_TEMPS = {}
    for d in sorted(os.listdir(base_dir)):
        if d == 'pooled':
            continue
        if not os.path.isdir(os.path.join(base_dir, d, 'csv')):
            continue
        if d == 'deterministic':
            _COND_TEMPS[d] = 0.0
        elif d.startswith('temp_'):
            try:
                _COND_TEMPS[d] = float(d.replace('temp_', ''))
            except ValueError:
                continue

    if not _COND_TEMPS:
        ui.err("No temperature directories found.")
        return

    all_runs = sorted(_parse_runs(runs_str))
    if not all_runs:
        ui.err(f"No valid runs in: {runs_str}")
        return

    # Split runs into GPU (need subprocess) and analysis (direct call)
    gpu_runs      = _order_runs([r for r in all_runs if r not in _ANALYSIS_RUNS and r not in (40, 47, 54, 55, 50, 51, 52)])
    analysis_runs = _order_runs([r for r in all_runs if r in _ANALYSIS_RUNS and r not in (40, 47, 54, 55, 50, 51, 52)])
    POOLED_RUNS = {50, 51, 52, 53, 54, 55, 56}  # v0.79.4.0: renumbered pooled+xm+out set
    pooled_runs   = [r for r in all_runs if r in POOLED_RUNS]

    # v0.79.5.4 [DISP]: dispatch instrumentation -- parsed run sets after split
    print(f"[DISP] _run_all_temps_analysis parsed: runs_str={runs_str!r} "
          f"all={all_runs} gpu={gpu_runs} ana={analysis_runs} pooled={pooled_runs}", flush=True)

    # Gate: check Q45 exists at every temperature for analysis runs that need it
    _NEEDS_Q45 = {47, 48, 49, 42, 43, 44}  # v0.79.4.0: old 25,27,32,33,34,46  # all need POOL_DIM calibration from Run 0041
    if any(r in _NEEDS_Q45 for r in analysis_runs) and 41 not in analysis_runs:  # v0.79.4.0: old 45 POOL_DIM → new 41
        # Only gate if Run 0041 is NOT in the run set (if it is, it'll run first per EXECUTION_ORDER)
        missing_q45 = []
        for cond, temp in sorted(_COND_TEMPS.items(), key=lambda x: x[1]):
            paths = _gp(family, size, variant, temp, create_dirs=False)
            _det_dim = _ana.load_optimal_pool_dim(paths['analysis'])
            if _det_dim is None:
                missing_q45.append(f"T={temp} ({cond})")
        if missing_q45:
            ui.err(f"Run 0041 calibration missing at: {', '.join(missing_q45)}")
            ui.err("  Include Run 0041 in the run set or run it separately first.")
            return

    n_temps = len(_COND_TEMPS)
    n_gpu = len(gpu_runs)
    n_ana = len(analysis_runs)
    n_total = n_temps * (n_gpu + n_ana) + len(pooled_runs)
    ui.section(f"ALL TEMPS -- {n_temps} temperatures × {n_gpu + n_ana} runs = {n_total} total")
    if gpu_runs:
        ui.msg(f"  GPU runs (subprocess per temp): {gpu_runs}")
    if analysis_runs:
        ui.msg(f"  Analysis runs (direct call): {analysis_runs}")
    if pooled_runs:
        ui.msg(f"  Pooled runs (after per-temp): {pooled_runs}")
    ui.blank()

    done = 0
    t0 = time.time()

    import subprocess as _sp_at
    from cartography import ANALYSIS_JSON as _AJ

    for ci, (cond, temp) in enumerate(sorted(_COND_TEMPS.items(), key=lambda x: x[1]), 1):
        if not gpu_runs and not analysis_runs:
            break
        ui.section(f"[{ci}/{n_temps}] {cond}  (T={temp})")
        session['temperature'] = temp
        paths = _gp(family, size, variant, temp)
        # Copy temperature-independent runs (19, 20, 26) from other temps if available
        _copy_temperature_independent_runs(session, paths)
        # Write temperature to session file so dashboard SSE picks it up
        try:
            _atomic_write_json(os.path.join(ROOT, 'last_session.json'),
                               {**session, 'temperature': temp})
        except Exception:
            pass

        # ── GPU runs: subprocess per run (or batched) ────────────────────
        if gpu_runs:
            ui.msg(f"  {cond} GPU runs...")
            _sess_file = os.path.join(ROOT, 'last_session.json')
            log_path = os.path.join(ROOT, '.iota_flask.log')
            _status = _scan_runs(paths, session.get('trials', 100))

            # Filter to runs that actually need work
            _todo = []
            # v0.80.0.44: _gpu_set as int-keyed; _PREREQS keys are strings
            # like "0017" but gpu_runs are ints. Coerce both to int for
            # membership compare so "is this prereq already in our gpu
            # batch" works correctly. Without this, every prereq fell
            # into _ext_probs and the warn-and-skip path always fired.
            _gpu_set = set(int(r) for r in gpu_runs)
            for run_num in gpu_runs:
                _rs = _status.get(run_num)
                # v0.79.4.15: needs_et / et_partial no longer emitted.
                # Run 0016 (E_t recovery) reports its own done/partial/missing
                # directly, so the per-run accept-if-ET-only branch is gone.
                if _rs == 'done':
                    ui.ok(f"  Run {run_num}: complete at T={temp} -- skipping.")
                    done += 1
                else:
                    # Check prerequisites -- but ignore prereqs that are also
                    # in this gpu_runs set (they'll collect first per exec order)
                    _probs = _check_prereqs(run_num, _status)
                    _ext_probs = [(p,s,d) for p,s,d in _probs if int(p) not in _gpu_set]
                    if _ext_probs:
                        # v0.80.0.44: was `f"R{p[0]:04d}" for p in _ext_probs` --
                        # double bug: (1) `p` is already the run id from the
                        # tuple unpack so `p[0]` reads the first character of
                        # the string, (2) `_PREREQS` keys are 4-digit strings
                        # like "0001" not ints, so :04d crashed before the
                        # subscript bug even mattered. Coerce to int via
                        # `int(p)` since string ids parse cleanly.
                        _missing = ', '.join(f"R{int(p):04d}" for p,s,d in _ext_probs)
                        ui.warn(f"  Run {run_num}: prereqs not met ({_missing}) -- skipping.")
                    else:
                        _todo.append(run_num)

            # v0.79.5.4 [DISP]: dispatch instrumentation -- per-temp gate result
            print(f"[DISP] T={temp} gate: status={dict((r, _status.get(r)) for r in gpu_runs)} "
                  f"todo={_todo}", flush=True)

            if not _todo:
                pass  # all skipped
            elif not session.get('unload_between_runs', True) and len(_todo) > 1:
                # Batch mode: one subprocess, one model load
                # Runs 0001, 0017, 0020, 0018, 0019 manage their own model -- run individually
                # v0.80.0.44: Run 0016 added. ET recovery self-loads its
                # own base model (distinct from the batch's abliterated
                # model load), runs hours of source-by-source recovery,
                # and has its own resume/dedup logic. Inside the batch
                # it serialized Runs 17-20 behind itself. Solo-late
                # makes the batch finish fast and reorders 16 last
                # within solo-late so 17-20 don't wait on it.
                _SELF_LOAD = {1, 16, 17, 18, 19, 20}
                _LATE_SOLO = {16, 17, 18, 19, 20}  # 16 ordered LAST in Phase 3 below
                _solo_early = [r for r in _todo if r in _SELF_LOAD and r not in _LATE_SOLO]
                _solo_late  = [r for r in _todo if r in _LATE_SOLO]
                _batch = [r for r in _todo if r not in _SELF_LOAD]

                # Phase 1: early solo (19, 30 -- independent)
                for run_num in _solo_early:
                    ui.msg(f"  Run {run_num} at T={temp} (self-loading)...")
                    _wsess_tmp = dict(session)
                    _wsess_tmp['runs'] = str(run_num)
                    _wsess_tmp['temperature'] = temp
                    try:
                        _atomic_write_json(_sess_file, _wsess_tmp)
                    except Exception:
                        pass
                    # v0.79.5.4 [DISP]: subprocess spawn -- solo-early path
                    print(f"[DISP] spawn[solo-early] T={temp} --single-run {run_num}", flush=True)
                    try:
                        with open(log_path, 'a') as _lf:
                            proc = _sp_at.Popen(
                                [sys.executable, os.path.join(ROOT, 'start_here.py'),
                                 '--single-run', str(run_num)],
                                stdout=_lf, stderr=_sp_at.STDOUT
                            )
                        ret = proc.wait()
                        if ret == 0:
                            ui.ok(f"  Run {run_num} at T={temp} complete.")
                            done += 1
                        else:
                            ui.warn(f"  Run {run_num} at T={temp} exited {ret}.")
                    except Exception as _e:
                        ui.err(f"  Run {run_num} at T={temp} error: {_e}")
                    _wait_for_vram(run_num)

                # Phase 2: batch (standard runs including Run 0006)
                if _batch:
                    ui.msg(f"  Batch: {len(_batch)} runs in one process...")
                    _wsess_tmp = dict(session)
                    _wsess_tmp['temperature'] = temp
                    try:
                        _atomic_write_json(_sess_file, _wsess_tmp)
                    except Exception:
                        pass
                    try:
                        with open(log_path, 'a') as _lf:
                            proc = _sp_at.Popen(
                                [sys.executable, os.path.join(ROOT, 'start_here.py'),
                                 '--batch-runs', ','.join(str(r) for r in _batch)],
                                stdout=_lf, stderr=_sp_at.STDOUT
                            )
                        ret = proc.wait()
                        if ret == 0:
                            ui.ok(f"  Batch at T={temp} complete ({len(_batch)} runs).")
                            done += len(_batch)
                        else:
                            ui.warn(f"  Batch at T={temp} exited {ret}.")
                    except Exception as _e:
                        ui.err(f"  Batch at T={temp} error: {_e}")
                    finally:
                        try:
                            _atomic_write_json(_sess_file, session)
                        except Exception:
                            pass
                    _wait_for_vram('batch')

                # Phase 3: late solo. v0.80.0.44: order fast runs (17/18/19/20)
                # before Run 16 (slow ET recovery) so Q8-style fresh-model
                # runs don't wait hours for collection runs to finish their
                # verification work. Per-run prereq re-check using the post-
                # batch status -- if the batch failed to produce a run's
                # prereq (e.g. Run 6's all-layers .npy for runs 17/18/19),
                # skip cleanly with a warning instead of firing and crashing
                # partway. Re-scan after each solo-late run so subsequent
                # prereq checks see the result of this run.
                _LATE_SOLO_ORDER = [17, 18, 19, 20, 16]
                _solo_late_ordered = [r for r in _LATE_SOLO_ORDER if r in _solo_late]
                _post_batch_status = _scan_runs(paths, session.get('trials', 100))
                for run_num in _solo_late_ordered:
                    _probs = _check_prereqs(run_num, _post_batch_status)
                    if _probs:
                        _missing = ', '.join(f"R{int(p):04d}" for p,s,d in _probs)
                        ui.warn(f"  Run {run_num} at T={temp}: prereqs not met ({_missing}) -- skipping.")
                        continue

                    if _post_batch_status.get(run_num) == 'done':
                        ui.ok(f"  Run {run_num} at T={temp}: already complete -- skipping.")
                        done += 1
                        continue

                    ui.msg(f"  Run {run_num} at T={temp} (self-loading, post-batch)...")
                    _wsess_tmp = dict(session)
                    _wsess_tmp['runs'] = str(run_num)
                    _wsess_tmp['temperature'] = temp
                    try:
                        _atomic_write_json(_sess_file, _wsess_tmp)
                    except Exception:
                        pass
                    # v0.79.5.4 [DISP]: subprocess spawn -- solo-late path
                    print(f"[DISP] spawn[solo-late] T={temp} --single-run {run_num}", flush=True)
                    try:
                        with open(log_path, 'a') as _lf:
                            proc = _sp_at.Popen(
                                [sys.executable, os.path.join(ROOT, 'start_here.py'),
                                 '--single-run', str(run_num)],
                                stdout=_lf, stderr=_sp_at.STDOUT
                            )
                        ret = proc.wait()
                        if ret == 0:
                            ui.ok(f"  Run {run_num} at T={temp} complete.")
                            done += 1
                        else:
                            ui.warn(f"  Run {run_num} at T={temp} exited {ret}.")
                    except Exception as _e:
                        ui.err(f"  Run {run_num} at T={temp} error: {_e}")
                    _wait_for_vram(run_num)
                    # Re-scan so subsequent prereq checks see this run's result.
                    _post_batch_status = _scan_runs(paths, session.get('trials', 100))
            else:
                # Per-run mode: one subprocess per run
                for run_num in _todo:
                    ui.msg(f"  Run {run_num} at T={temp}...")
                    _wsess_tmp = dict(session)
                    _wsess_tmp['runs'] = str(run_num)
                    _wsess_tmp['temperature'] = temp
                    try:
                        _atomic_write_json(_sess_file, _wsess_tmp)
                    except Exception as _e:
                        ui.warn(f"  Could not write session for Run {run_num}: {_e} -- skipping")
                        continue

                    # v0.79.5.4 [DISP]: subprocess spawn -- per-run mode (default path)
                    print(f"[DISP] spawn[per-run] T={temp} --single-run {run_num}", flush=True)
                    try:
                        with open(log_path, 'a') as _lf:
                            proc = _sp_at.Popen(
                                [sys.executable, os.path.join(ROOT, 'start_here.py'),
                                 '--single-run', str(run_num)],
                                stdout=_lf, stderr=_sp_at.STDOUT
                            )
                        ret = proc.wait()
                        if ret == 0:
                            ui.ok(f"  Run {run_num} at T={temp} complete.")
                            done += 1
                        else:
                            ui.warn(f"  Run {run_num} at T={temp} exited {ret}.")
                    except Exception as _e:
                        ui.err(f"  Run {run_num} at T={temp} subprocess error: {_e}")
                    finally:
                        try:
                            _atomic_write_json(_sess_file, session)
                        except Exception:
                            pass

                    _wait_for_vram(run_num)

        # ── Analysis runs: direct call ───────────────────────────────────
        # No outer skip gate -- analysis.py run() has its own resume-aware
        # completeness check that detects missing fields (v0.75.2.2).
        for run_num in analysis_runs:
            ui.msg(f"  Run {run_num}...")
            try:
                _ana.run(run_num, session, paths)
                done += 1
            except Exception as _e:
                import traceback
                ui.warn(f"  Run {run_num} at T={temp} failed: {_e}")
                traceback.print_exc()

        # ── E_t recovery safety-net: fire Run 0016 if it's not yet done ─────
        # v0.79.4.15: was `for r in gpu_runs if _post_status.get(r) == 'needs_et'`.
        # Scanner no longer emits needs_et -- ET state lives in Run 0016's status.
        # If the main gpu_runs loop already dispatched 48, its status will be
        # 'done' now and this block no-ops. If 48 wasn't in gpu_runs (user
        # selected a subset), fire it here to keep auto-temp self-healing.
        _post_status = _scan_runs(paths, session.get('trials', 100))
        if _post_status.get(16) != 'done' and 16 not in gpu_runs:  # v0.79.4.0: E_t meta-run old 48 → new 16
            ui.msg(f"  E_t recovery (Run 0016) pending at T={temp} -- firing...")
            _sess_file = os.path.join(ROOT, 'last_session.json')
            log_path = os.path.join(ROOT, '.iota_flask.log')
            try:
                _atomic_write_json(_sess_file, {**session, 'temperature': temp})
                with open(log_path, 'a') as _lf:
                    et_proc = _sp_at.Popen(
                        [sys.executable, os.path.join(ROOT, 'start_here.py'),
                         '--single-run', '16'],  # v0.79.4.17: was '48' (old id) -- Run 16 is new E_t meta-run
                        stdout=_lf, stderr=_sp_at.STDOUT
                    )
                et_ret = et_proc.wait()
                if et_ret == 0:
                    ui.ok(f"  E_t recovery at T={temp} complete.")
                else:
                    ui.warn(f"  E_t recovery at T={temp} exited {et_ret}.")
            except Exception as _e:
                ui.warn(f"  E_t recovery at T={temp} failed: {_e}")
            _wait_for_vram('ET')

        ui.ok(f"  {cond} complete.")
        # v0.75.2.2: qcache_cleanup removed. Cache persists for instant re-entry.
        ui.blank()

    # Pooled runs -- call directly with T=0.0 paths
    from cartography import get_pooled_paths as _gpp
    for run_num in pooled_runs:
        ui.section(f"Pooled -- Run {run_num}")
        # No outer skip gate -- analysis.py run() handles completeness (v0.75.2.2).
        session['temperature'] = 0.0
        paths = _gp(family, size, variant, 0.0)
        try:
            if run_num == 55:
                _run_54_stats_report(session)
            elif run_num == 56:
                _run_56_calibration(session)
            elif run_num == 57:
                _run_57_function_class_sensitivity(session)
            elif run_num == 58:
                _run_58_apparatus(session)
            elif run_num == 59:
                _run_59_paper_assembly(session)
            else:
                _ana.run(run_num, session, paths)
            done += 1
        except Exception as _e:
            import traceback
            ui.warn(f"  Run {run_num} (pooled) failed: {_e}")
            traceback.print_exc()
        ui.ok(f"  Run {run_num} complete.")
        ui.blank()

    elapsed = time.time() - t0
    ui.ok(f"ALL TEMPS complete -- {done}/{n_total} runs across {n_temps} temperatures ({elapsed:.0f}s)")


def main():
    """Primary entry point. Dispatches on CLI flags to one of:

      --all-stats         : analysis + stats for every temperature
                            round (Run 0041 first for calibration, then
                            25/27/32-34/40/45-47/49/54), finishing with
                            Run 0051 pooled cross-directory decomposition.
      --et-recovery N,M,.. : E_t recovery pass for specified runs via
                            _run_et_recovery -- base-model forward pass
                            over the existing CSV conversation.
      --single-run N      : isolated subprocess mode -- load model, run
                            one run, exit. Called by _run_session_isolated
                            for per-run VRAM isolation.
      --batch-runs N,M,.. : one model load, multiple runs in sequence,
                            exit. Called when unload_between_runs=False.
      --auto-run SPEC     : headless run-list execution via
                            _run_session_isolated.
      --auto-temp         : auto-advance through all 6 temperature
                            rounds for the active model.
      --all-temps-runs    : run specified analysis runs at every temp
                            that has data.
      (no flag)           : interactive console mode. Launches the
                            Flask dashboard as a daemon, polls until
                            ready (10s timeout), opens the browser to
                            localhost:5000, then blocks on Enter to
                            reopen the browser. First-run setup wizard
                            triggers if .iota_env.json is missing.

    Lowers process priority (BELOW_NORMAL on Windows, nice +10 on
    Unix) in any headless path so the OS scheduler favours interactive
    apps while IOTA runs in the background."""
    # ── Headless mode (v29.5): Flask dashboard can launch runs without console ──
    # export_flask.py spawns:  python start_here.py --headless
    # This bypasses the interactive menu entirely: load session, set runs, execute, exit.
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument('--auto-run', default=None,
                    help='Run specified runs non-interactively and exit (used by web dashboard)')
    _p.add_argument('--single-run', default=None,
                    help='Execute exactly one run number and exit (used internally for VRAM isolation)')
    _p.add_argument('--batch-runs', default=None,
                    help='Execute multiple runs with one model load (comma-separated, used internally)')
    _p.add_argument('--r30-instance', default=None,
                    help='Run 0020 instance filter: A, B, or coupling (used internally for subprocess isolation)')
    _p.add_argument('--et-recovery', default=None,
                    help='Run ET recovery for specified run numbers (comma-separated) and exit')
    _p.add_argument('--all-stats', action='store_true',
                    help='Run analysis (25,27,32,33,34,45,46) + stats for every temperature, then Run 0051 pooled')
    _p.add_argument('--force', action='store_true',
                    help='Force re-run of analysis even if output exists (used with --all-stats)')
    _p.add_argument('--backup', action='store_true',
                    help='Backup existing analysis JSONs before overwriting (used with --force)')
    _p.add_argument('--auto-temp', action='store_true',
                    help='Auto-advance through all temperature rounds -- collect incomplete rounds in order')
    _p.add_argument('--all-temps-runs', default=None,
                    help='Run specified runs at every temperature that has data (comma-sep or ranges)')
    _p.add_argument('--all-models', default=None,
                    help='v0.79.4.15 -- fan out specified runs across every discovered '
                         'model in DATA/. Value is passed to the per-model dispatch as-is '
                         '(accepts auto-temp, all-temps:N, or plain run list). Smart loader '
                         'groups by model: all runs complete on Model A before Model B starts.')
    _p.add_argument('--daemon', action='store_true')  # absorb Flask's own flag
    _args, _ = _p.parse_known_args()

    # v0.71.0.11: Lower process priority in headless mode to reduce system lag.
    # All Flask-launched subprocesses (analysis, data collection) run headless.
    # BELOW_NORMAL on Windows, nice +10 on Linux. Yields CPU to foreground apps.
    # Controlled by "Low priority" toggle in Settings → Performance (default ON).
    if any([_args.all_stats, _args.auto_run, _args.single_run, _args.batch_runs,
            _args.auto_temp, _args.all_temps_runs, _args.et_recovery]):
        _lp_session = ui.load_session()
        # v0.76.0.4: set quant for all path construction before any dispatch
        from cartography import set_active_quant
        set_active_quant(_lp_session.get('quantization', '4bit'))
        if _lp_session.get('low_priority', True):
            try:
                if sys.platform == 'win32':
                    import ctypes
                    BELOW_NORMAL = 0x00004000
                    ctypes.windll.kernel32.SetPriorityClass(
                        ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
                else:
                    os.nice(10)
            except Exception:
                pass

    if _args.all_stats:
        os.environ['IOTA_HEADLESS'] = '1'
        session = ui.load_session()
        _run_all_stats(session, force=_args.force, backup=_args.backup)
        return

    if _args.et_recovery:
        os.environ['IOTA_HEADLESS'] = '1'
        session = ui.load_session()
        run_nums = [int(x.strip()) for x in _args.et_recovery.split(',') if x.strip()]
        paths = get_paths(session.get('model_family','llama'), session.get('model_size','8b'),
                         session.get('model_variant','abliterated'), session.get('temperature',0.0))
        import runners as _rm
        _rm._run_et_recovery(session, paths, sorted(run_nums))
        return

    if _args.single_run:
        # Single-run isolation mode: execute one run in this process and exit.
        # Called by _run_session_isolated for per-run VRAM isolation.
        os.environ['IOTA_HEADLESS'] = '1'
        os.environ['IOTA_SINGLE_RUN'] = '1'  # suppresses ET recovery inside _run_session
        session = ui.load_session()
        session['runs'] = _args.single_run
        if _args.r30_instance:
            session['_r30_instance'] = _args.r30_instance
        _run_session(session)
        return

    if _args.batch_runs:
        # Batch mode: load model once, run multiple runs, exit.
        os.environ['IOTA_HEADLESS'] = '1'
        os.environ['IOTA_SINGLE_RUN'] = '1'
        session = ui.load_session()
        from cartography import get_paths as _gp_batch
        paths = _gp_batch(session.get('model_family', 'llama'),
                          session.get('model_size', '8b'),
                          session.get('model_variant', 'abliterated'),
                          session.get('temperature', 0.0))
        run_nums = [int(x.strip()) for x in _args.batch_runs.split(',') if x.strip().isdigit()]
        from runners import run_batch
        run_batch(run_nums, session, paths)
        return

    if _args.auto_run:
        os.environ['IOTA_HEADLESS'] = '1'
        session = ui.load_session()
        session['runs'] = _args.auto_run
        _run_session_isolated(session)
        return

    if _args.auto_temp:
        os.environ['IOTA_HEADLESS'] = '1'
        session = ui.load_session()
        _run_all_temperatures(session)
        return

    if _args.all_temps_runs:
        os.environ['IOTA_HEADLESS'] = '1'
        session = ui.load_session()
        # v0.79.5.4 [DISP]: dispatch instrumentation -- entry arg for --all-temps-runs
        print(f"[DISP] --all-temps-runs arg={_args.all_temps_runs!r} "
              f"variant={session.get('model_variant','?')} "
              f"model={session.get('model_name','?')} "
              f"session_runs={session.get('runs','?')!r}", flush=True)
        _run_all_temps_analysis(session, _args.all_temps_runs)
        return

    if _args.all_models:
        os.environ['IOTA_HEADLESS'] = '1'
        session = ui.load_session()
        _run_all_models(session, _args.all_models)
        return

    # ── First-run setup wizard (v30.0) ───────────────────────────────
    # Creates .iota_env.json, detects GPU/VRAM, installs deps, writes launchers.
    # No-ops if .iota_env.json already exists.
    try:
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location("setup", os.path.join(ROOT, "setup.py"))
        if _sp:
            _sm = importlib.util.module_from_spec(_sp)
            _sp.loader.exec_module(_sm)
            _sm.run_setup(force=False)
    except Exception:
        pass   # setup failure is non-fatal -- deps already installed or user knows what they're doing

    ui.install_deps()
    session = ui.load_session()
    ui.clear_crash_marker()
    _ensure_hf_token(session)

    # ── Dashboard ────────────────────────────────────────────────────
    # Launch the Flask dashboard as a background daemon on every startup.
    # Output is silenced to .iota_flask.log. Console remains fully navigable.
    _launch_dashboard()
    # Wait for Flask to be ready before opening browser -- prevents race where
    # the browser loads a cached page and the user clicks Run before Flask is
    # listening. Poll localhost:5000 with a 10s timeout.
    _dash_ready = False
    for _tick in range(20):
        if _is_dashboard_running():
            _dash_ready = True
            break
        time.sleep(0.5)
    if _dash_ready:
        _maybe_open_browser(countdown=0)
    else:
        ui.warn("Dashboard did not start within 10s -- check .iota_flask.log")
        ui.msg(f"  Try opening {DASHBOARD_URL} manually.")

    while True:
        try:
            input("\n  Press Enter to reopen browser... ")
            import webbrowser
            webbrowser.open(DASHBOARD_URL)
        except (EOFError, KeyboardInterrupt):
            break

    ui.msg("Goodbye.")


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(add_help=False)
    _parser.add_argument('--headless', action='store_true',
                         help='Non-interactive run: skip all menus, use saved session + --preset.')
    _parser.add_argument('--preset', default=None,
                         help='Run preset key (f, 1, 2, 3, dc, a). Omit to use saved session runs.')
    _args, _ = _parser.parse_known_args()

    if _args.headless:
        # Web-launched headless run.
        # Bypasses all menus. Loads saved session, applies preset, calls _run_session().
        os.environ['IOTA_HEADLESS'] = '1'
        ui.install_deps()
        _session = ui.load_session()
        if not _session.get('model_path'):
            print("[headless] ERROR: No model configured. "
                  "Run start_here.py in a terminal first to set the model.")
            sys.exit(1)
        if _args.preset and _args.preset in PRESETS and _args.preset != 'c':
            _, _runs_str, _ = PRESETS[_args.preset]
            _session['runs'] = _runs_str
            print(f"[headless] Preset: {_args.preset}  runs: {_runs_str}")
        else:
            # No preset passed -- use the runs saved to last_session.json by the dashboard.
            # BUG-39A fix: previously default='f' caused this branch to always overwrite
            # session['runs'] with '1-44' regardless of what was selected in the UI.
            print(f"[headless] Using saved runs: {_session.get('runs', '1-44')}")
        _launch_dashboard()   # no-op if already running
        _run_session_isolated(_session)
        print("[headless] All runs complete.")
        try:
            import torch, gc, threading
            gc.collect()
            if torch.cuda.is_available():
                # BUG-43B class fix (v45.0.5): torch.cuda.synchronize() is a
                # blocking call that can stall indefinitely on Windows after long
                # runs. Same fix as unload_model and inter-run flush: daemon
                # thread + 15s timeout. os._exit(0) fires regardless.
                _sync_done = threading.Event()
                def _sync():
                    try: torch.cuda.synchronize()
                    except Exception: pass
                    finally: _sync_done.set()
                threading.Thread(target=_sync, daemon=True).start()
                if not _sync_done.wait(timeout=15):
                    print("[headless] cuda.synchronize() timed out -- skipping.")
                torch.cuda.empty_cache()
        except Exception:
            pass
        os._exit(0)
    else:
        main()
