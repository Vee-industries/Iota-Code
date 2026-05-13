"""
IOTA FRAMEWORK -- DEPENDENCY MAP
================================
Hypothesis-to-run dependency reference for the /hyp_deps dashboard endpoint.
RUN_CSV and ANALYSIS_JSON have moved to cartography.py.
"""

import os, sys, json

# Legacy re-exports for any callers that haven't updated yet
from cartography import RUN_CSV, ANALYSIS_JSON

DEPENDENCY_MAP = {

    "H01": {
        "name":             "Geometry Is Condition-Invariant",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [4, 5, 1],
        "comparison_runs":  [6, 7, 8, 9, 10, 11, 12],
        "upstream_runs":    [],
        "csv_files":        ["R0004_null.csv", "R0005_null.csv", "R0001_null.csv"],
        "required_columns": ["state_similarity_index", "signal_entropy_ratio", "mean_logit_entropy"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H01",
        "saves_E_t":        True,   # Run 0001 saves real E_t (base hidden states, v40.0.0)
        "saves_C_t":        True,   # Run 0001 saves real C_t (abliterated-base diff, v40.0.0)
        "saves_hidden":     True,   # Run 0001 saves hidden states (all_layers, all three variants)
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       ["INF-H01-THRESHOLD (fixed v32.0): verdict was level-only (mean>0.5); added Levene variance homogeneity test across condition groups."],
        "notes":            (
            "v40.0.0: Run 0001 redesigned as three-model sequential run "
            "(base, instruct, abliterated). E_t = base hidden states. "
            "C_t = abliterated minus base difference vectors (real, not zero-vector). "
            "Run 0001 now qualifies for SOURCE_RUNS_3WAY in Run 0042. "
            "Global mean C_t constant saved for retroactive backfill of Runs 0004, 0005, 0009-0012. "
            "R0001_null.csv contains abliterated pass only (unchanged for H01/H10 analysis). "
            "EXECUTION PRIORITY: Run 0001 should run FIRST among all generation runs, "
            "before Run 0003. It produces the C_t constants that every downstream run needs."
        ),
    },

    "H54": {
        "name":             "Baseline Entropy Is Degenerate",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [4, 5, 1],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["R0004_null.csv", "R0005_null.csv", "R0001_null.csv"],
        "required_columns": ["mean_logit_entropy"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H02",
        "saves_E_t":        True,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Supported if mean_logit_entropy in [0.5, 8.0]. Precondition for signal_entropy_ratio validity.",
    },

    "H03": {
        "name":             "Introspection Has No Geometric Effect",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [6, 7, 8],
        "comparison_runs":  [4, 5, 1],
        "upstream_runs":    [],
        "csv_files":        ["R0006_introspection.csv", "R0007_introspection.csv", "R0008_introspection.csv"],
        "required_columns": ["state_similarity_index"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H03",
        "saves_E_t":        True,
        "saves_C_t":        True,   # Real C_t from system prompt
        "saves_hidden":     True,
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       ["INF-H03-BASELINE (fixed v32.0): export_stats.py ttest compared [3,4,5] vs [1,2] -- should be [1,2,19]. Map was correct; code was wrong."],
        "notes":            "E_t and C_t from these runs feed Run 0042 quadruplets (SOURCE_RUNS_3WAY).",
    },

    "H55": {
        "name":             "Signal Entropy Ratio Is Condition-Invariant",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [6, 7, 8],
        "comparison_runs":  [4, 5, 1],
        "upstream_runs":    [],
        "csv_files":        ["R0006_introspection.csv", "R0007_introspection.csv", "R0008_introspection.csv"],
        "required_columns": ["signal_entropy_ratio"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H04",
        "saves_E_t":        True,
        "saves_C_t":        True,
        "saves_hidden":     True,
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       ["INF-H04-BASELINE (fixed v32.0): export_stats.py ttest compared [3,4,5] vs [1,2] -- should be [1,2,19]. Map was correct; code was wrong."],
        "notes":            "Same data collection as H03. signal_entropy_ratio = layer_sim_mean / (mean_logit_entropy + 1e-9).",
    },

    "H05": {
        "name":             "Contradiction Has No Geometric Effect",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [28],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0028_contradiction.csv"],
        "required_columns": ["disruption_flag", "turn", "contradiction_turn"],
        # BUG (fixed v36.9): contradiction_turn was in extra_columns only despite
        # being tested in the H05 outer guard:
        #   if not sub22.empty and 'contradiction_turn' in sub22.columns and ...
        # Same class as H34 BUG-FOUND-3 (v36.4) and H35 fix (v36.7). A column
        # tested in the outer guard must be in required_columns so the schema
        # checker will flag its absence before inference runs. Moved from extra
        # to required. post_contradiction stays in extra_columns -- it is accessed
        # inside the block with a graceful `if 'post_contradiction' in sub22.columns:`
        # fallback, not in the outer guard itself.
        "extra_columns":    ["post_contradiction"],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H05",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["INF-H05-FIG (fixed v32.0): fig_self_reference_phase showed state_similarity_index by phase; H05 inference tests disruption_flag. Added fig_disruption_events_by_phase as primary figure."],
        "notes":            "Contradiction at turn 7 (v28.0). Setup turns 1-6, recovery turns 8-13. disruption_elevation = mean(disruption_flag at contradiction_turn rows) - mean(disruption_flag at baseline turns). Columns: contradiction_turn (int flag), post_contradiction (int flag).",
    },

    "H06": {
        "name":             "Similarity Does Not Decay With Context",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [22],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0022_saturation.csv"],
        "required_columns": ["state_similarity_index", "turn"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H06",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Linear regression state_similarity_index ~ turn. Supported if slope < 0, p < 0.05. 30-turn run.",
    },

    "H56": {
        "name":             "No Similarity Drop Early To Late",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [22],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0022_saturation.csv"],
        "required_columns": ["state_similarity_index", "turn"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H07",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "t-test: turns 1-5 vs turns 26-30. Complementary to H06 (trend vs magnitude).",
    },

    "H08": {
        "name":             "Throughline Has No Trajectory Effect",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [13, 14, 15],
        "comparison_runs":  [4, 5, 1],   # v32.0: include Run 0001 (INF-H08-BASELINE fix)
        "upstream_runs":    [],
        "csv_files":        ["R0013_framing.csv", "R0014_framing.csv", "R0015_framing.csv"],
        "required_columns": ["state_similarity_index"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H08",
        "saves_E_t":        True,
        "saves_C_t":        True,   # Different system prompts per condition
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["INF-H08-BASELINE (fixed v32.0): null baseline was [1,2]; expanded to [1,2,19]."],
        "notes":            "E_t and C_t from these runs feed Run 0042 quadruplets (SOURCE_RUNS_3WAY).",
    },

    "H09": {
        "name":             "Task Demand Geometry Equals Introspection",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [9, 10, 11, 12],
        "comparison_runs":  [4, 5],
        "upstream_runs":    [],
        "csv_files":        ["R0009_math.csv", "R0010_math.csv", "R0011_math.csv", "R0012_math.csv"],
        "required_columns": ["state_similarity_index"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H09",
        "saves_E_t":        True,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Prediction: arithmetic state_similarity_index LOWER than null baseline (direction='negative').",
    },

    "H10": {
        "name":             "Disruption Fully Resets Trajectory",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [39, 40],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0039_jolt.csv", "Q0040_jolt.csv"],
        "required_columns": ["layer_sim_mean", "is_shock"],
        "extra_columns":    ["is_recovery"],
        # BUG-FOUND-3 (fixed v36.4): is_shock was in both required_columns and extra_columns.
        # Removed from extra_columns -- already covered by required_columns.
        # is_recovery not in required_columns because the inference falls back gracefully
        # when the column is absent (pre-v16 CSVs). Kept in extra_columns only.
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H10",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Two-part test: (1) shock disrupts sim (2) recovery turns (Run 0039 is_recovery=1) restore sim. Both required.",
    },

    "H11": {
        "name":             "Prior State Adds No Predictive Power",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [47, 48],
        "comparison_runs":  [],
        "upstream_runs":    [1, 3],   # Run 0047 uses Run 0001 hidden states; Run 0048 uses Run 0003
        "csv_files":        [],
        "required_columns": [],
        "extra_columns":    [],
        "analysis_file":    "analysis.py",
        "analysis_fn":      "_run_granger (Runs 0047/0048)",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     False,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Output: analysis/granger_A.json, granger_B.json. Read by export_stats.py load_granger(). Superseded by H21 but retained for continuity.",
    },

    "H12": {
        "name":             "Similarity Is Explained By Token Statistics",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [32],
        "comparison_runs":  [4, 5, 6, 7, 8],
        "upstream_runs":    [],
        "csv_files":        ["Q0032_tokenization.csv"],
        "required_columns": ["state_similarity_index"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H12",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Run 0032 state_similarity_index should be closer to null (1,2) than introspection (3-5). Tests semantic vs token statistics.",
    },

    "H13": {
        "name":             "Similarity Collapses Under Temperature Variation",
        "phase":            2,
        "status":           "blocking",
        "data_runs":        [3],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["R0003_temperature_grid.csv"],
        "required_columns": ["state_similarity_index", "temperature"],
        "extra_columns":    ["temperature", "condition"],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H13",
        "saves_E_t":        True,   # Run 0003 saves E_t -- required by Run 0042
        "saves_C_t":        True,   # Run 0003 saves C_t -- required by Run 0042
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         True,   # BLOCKS all Phase 2 results
        "known_bugs":       [],
        "notes":            "BLOCKING. Must pass before any Phase 2 result is cited. Compare T=0.2 vs T=1.0. |drop| < 0.1 = supported.",
    },

    "H14": {
        "name":             "Cross-Turn Similarity Is Layer-Uniform",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [27],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0027_layers.csv"],
        "required_columns": ["layer_sim_prev_profile"],
        # FINDING (fixed v36.9): layer_sim_prev_profile duplicated in both required_columns
        # and extra_columns. layer_sim_prev_profile is in STANDARD_COLS -- standard columns
        # must never appear in extra_columns. Same class as FINDING-2 (v36.6).
        # Functional impact: zero (all_required_csv_columns uses set union).
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H14",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "layer_sim_prev_profile is JSON per row. Peak layer index must be >= n_layers//2.",
    },

    "H57": {
        "name":             "Turn-1 Similarity Is Layer-Uniform",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [27],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0027_layers.csv"],
        "required_columns": ["layer_sim_t1_profile"],
        # FINDING (fixed v36.9): layer_sim_t1_profile duplicated in both required_columns
        # and extra_columns. layer_sim_t1_profile is in STANDARD_COLS. Same class as
        # FINDING-2 (v36.6) and H14 fix (v36.9). Removed from extra_columns.
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H15",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       ["INF-H15-NOPVAL (fixed v32.0): verdict used magnitude threshold diff>0.02 with no significance test; added ttest_ind(early_layers, late_layers)."],
        "notes":            "layer_sim_t1_profile is JSON per row. Early layers (bottom third) should have lower turn-1 sim than late layers.",
    },

    "H16": {
        "name":             "Permuted Component Adds No Variance",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [49],
        "comparison_runs":  [],
        "upstream_runs":    [4, 5, 6, 7, 8, 9, 10, 11, 12, 39, 40, 29, 30, 31, 13, 14, 15, 32, 1, 2],
        "csv_files":        [],
        "required_columns": [],
        "extra_columns":    [],
        "analysis_file":    "analysis.py",
        "analysis_fn":      "_run_baseline_swap (Run 0049)",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     False,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["INF-H16-PARTIAL (fixed v32.0): only state_similarity_index permutation result read from Q32 JSON; signal_entropy_ratio and layer_sim_mean computed but discarded. All three now reported."],
        "notes":            "Output: analysis/baseline_swap.json. Read by load_baseline_swap(). Permutes condition labels N=10000.",
    },

    "H17": {
        "name":             "Consistency Is Fully Explained By System Prompt",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [23],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0023_introspection.csv"],
        "required_columns": ["state_similarity_index", "confound_condition"],
        "extra_columns":    ["confound_condition"],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H17",
        "saves_E_t":        True,   # Run 0023 saves E_t -- required by Run 0042
        "saves_C_t":        True,   # Run 0023 saves C_t -- required by Run 0042
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Bug I (v23.1) fixed per-condition resumption. Conditions: semantic, neutral, none. E_t/C_t feed Run 0042.",
    },

    "H18": {
        "name":             "History Mode Has No Trajectory Effect",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [24],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0024_persistence.csv"],
        "required_columns": ["state_similarity_index", "history_mode"],
        "extra_columns":    ["history_mode"],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H18",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["INF-H18-SUMMARY (fixed v32.0): summary history mode collected and plotted but never tested; only full vs last compared. All three pairwise tests now included."],
        "notes":            "history_mode: full, last, summary. Bug G (v23.0) fixed -- file_trial = mode_idx * n_trials + trial (runners.py:1256).",
    },

    "H19": {
        "name":             "Two-Instance Coupling Has No Geometric Effect",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [20],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["R0020_instances.csv"],
        "required_columns": ["coupling_score"],
        "extra_columns":    [],
        # FIND-DEPMAP-DUPES fix (v44.0.0): coupling_score was duplicated in both
        # required_columns and extra_columns. Same class as FINDING-2 (v36.6).
        # all_required_csv_columns() uses set union so functional impact was zero.
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H19",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["INF-H19-DIVERGENCE (fixed v32.0): only mean coupling tested against zero; divergence-over-turns (linregress coupling_score~turn) not computed. Added."],
        "notes":            "coupling_score computed inline per turn in _run_cross_instance.",
    },

    "H20": {  # OUT OF PAPER SCOPE -- Appendix A only
        "name":             "R Condition Has No Signal-Per-Watt Effect",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [21],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["R0021_conditions.csv"],
        "required_columns": ["signal_per_watt", "r_condition"],
        "extra_columns":    [],
        # FINDING-2 fix (v36.6): r_condition was duplicated in required_columns and extra_columns.
        # all_required_csv_columns() uses set union so functional impact is zero.
        # Consistent with BUG-FOUND-3 (v36.4) which fixed the same duplication in H10/H30/H31/H32/H34.
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H20",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "r_condition: high_r, mid_r, low_r. signal_per_watt = signal_entropy_ratio / peak_gpu_power.",
    },

    "H21": {
        "name":             "Prior State Adds No Power Over E_t And C_t",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [42],
        "comparison_runs":  [],
        "upstream_runs":    [6, 7, 8, 13, 14, 15, 3, 23],  # E_t and C_t sources
        "csv_files":        [],
        "required_columns": [],
        "extra_columns":    [],
        "analysis_file":    "analysis.py",
        "analysis_fn":      "_run_decomposition (Run 0042)",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     False,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Requires E_t+C_t from runs 0006-0008,0013-0015,0003,0023 (SOURCE_RUNS_3WAY). prompt_tokens covariate included. Output: analysis/decomposition.json.",
    },

    "H22": {
        "name":             "E, C, R Permutation Fractions Are Equal",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [43, 51],
        "comparison_runs":  [],
        "upstream_runs":    [42, 33],    # Run 0043 requires Run 0042 quadruplets + Run 0033 held-out
        "csv_files":        [],
        "required_columns": [],
        "extra_columns":    [],
        "analysis_file":    "analysis.py",
        "analysis_fn":      "_run_permutation_sensitivity (Run 0043) + _run_pooled (Run 0051)",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     False,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "NOT proper Sobol indices -- fixed-model permutation sensitivity. Output: analysis/Q34_sobol_partition.json. Run 0051 uses pooled data post all-rounds; output: pooled/analysis/Q40_pooled_sobol.json. export_stats now loads both (Bug V fix, v25.6).",
    },

    "H23": {
        "name":             "Hesitation Is Uncorrelated With Disruption",
        "phase":            3,
        "status":           "pending",
        "data_runs":        [38],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0038_hesitation.csv"],
        "required_columns": ["onset_delay_ratio", "disruption_flag"],
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H23",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["INF-H23-SIM (fixed v32.0): only r(onset_delay_ratio,disruption_flag) computed; r(onset_delay_ratio,state_similarity_index) predicted by H23 but absent from inference. Added."],
        "notes":            "Pearson r(onset_delay_ratio, disruption_flag). Supported if r > 0.3 and p < 0.05.",
    },

    "H24": {
        "name":             "Cross-Turn Similarity Has No Layer Locality",
        "phase":            3,
        "status":           "pending",
        "data_runs":        [34],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0034_layer_depth.csv"],
        "required_columns": ["layer_sim_depth_profile"],
        "extra_columns":    ["layer_sim_depth_profile"],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H24",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": True,
        "blocking":         False,
        "known_bugs":       ["INF-H24-ALIAS (fixed v32.0): layer_sim_depth_profile alias in runners.py had no format guard; added assertion to catch future run_generation return-type changes."],
        "notes":            "layer_sim_depth_profile is JSON per row (Phase 3 run; column name differs from H14's layer_sim_prev_profile).",
    },

    "H25": {
        "name":             "Entropy Shape Is Uncorrelated With R",
        "phase":            3,
        "status":           "pending",
        "data_runs":        [37],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0037_entropy_shape.csv"],
        "required_columns": ["entropy_trajectory", "state_similarity_index"],
        # FINDING (fixed v36.9): entropy_trajectory duplicated in both required_columns
        # and extra_columns. entropy_trajectory is in STANDARD_COLS. Same class as
        # FINDING-2 (v36.6) and H14/H15 fixes (v36.9). Removed from extra_columns.
        "extra_columns":    [],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H25",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["INF-H25-SPLIT (fixed v32.0): figure split by condition column; inference split by state_similarity_index median. Misaligned. Figure now uses similarity median split (primary) with condition split as secondary panel."],
        "notes":            "entropy_trajectory is JSON array per row. High-R turns (above median similarity) should show falling entropy (last_third < first_third).",
    },

    "H26": {
        "name":             "R Does Not Persist Across Condition Switch",
        "phase":            3,
        "status":           "pending",
        "data_runs":        [36],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0036_condition_transfer.csv"],
        "required_columns": ["state_similarity_index", "condition", "turn"],
        "extra_columns":    ["condition"],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H26",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Primary: transfer − null_to_arithmetic (shock cancelled). Conditions: introspection_only, arithmetic_only, transfer, null_to_arithmetic. Switch at turn 7.",
    },

    "H27": {
        "name":             "Output Similarity Is Uncorrelated With R",
        "phase":            3,
        "status":           "pending",
        "data_runs":        [35],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0035_output_similarity.csv"],
        "required_columns": ["output_sim_prev", "condition"],
        "extra_columns":    [],
        # FIND-DEPMAP-DUPES fix (v44.0.0): output_sim_prev and condition were
        # duplicated in both required_columns and extra_columns. Same class as
        # FINDING-2 (v36.6). all_required_csv_columns() uses set union so
        # functional impact was zero. Removed from extra_columns for consistency.
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H27",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "output_sim_prev = cosine sim of current output embedding to previous turn. Conditions: introspection, null.",
    },

    "H28": {
        "name":             "In-Sample Fractions Do Not Generalise",
        "phase":            2,
        "status":           "pending",
        "data_runs":        [33],
        "comparison_runs":  [],
        "upstream_runs":    [42, 43],   # ANALYSIS-TIME dependency only -- not collection-time.
                                        # Run 0033 data collection is independent of Runs 0042/0043
                                        # and should happen early (EXECUTION_ORDER places it
                                        # before Run 0043). upstream_runs here means: the Ridge
                                        # model from Run 0042/34 must exist before H28 can be
                                        # *evaluated* in Run 0043 Stage 2. Run 0033 can and should
                                        # collect data before Runs 0042/0043 run.
        "csv_files":        ["Q0033_validation.csv"],
        "required_columns": ["state_similarity_index", "confound_condition"],
        "extra_columns":    ["confound_condition"],
        "analysis_file":    "analysis.py",
        "analysis_fn":      "_run_permutation_sensitivity Stage 2 (Run 0043)",
        "saves_E_t":        True,   # Same structure as Run 0023
        "saves_C_t":        True,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            "Seed offset +50000. Never in SOURCE_RUNS_3WAY. Ridge model fitted on training data applied without refit. Evaluated in Run 0043 Stage 2. Run 0033 data collection precedes Runs 0042/0043 in EXECUTION_ORDER -- upstream_runs refers to analysis-time dependency only.",
    },
    "H29": {
        "name":             "No Single Layer Is Causally Sufficient",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [18],
        "comparison_runs":  [],
        "upstream_runs":    [6],   # Run 0006 provides all-layers reference states.
                                   # Run 0017 is NOT a filesystem dep (PREREQ-2, v30.7):
                                   # run42_layer_isolation.py calls _load_reference_states(ref_run=3)
                                   # only; no Run 0017 output files are read at dispatch or analysis time.
                                   # Run 0017 precedes Run 0018 in EXECUTION_ORDER for logical sequencing only.
        "csv_files":        ["R0018_layer_isolation.csv"],
        "required_columns": ["patch_layer", "output_changed", "sim_to_reference"],
        "extra_columns":    ["patch_layer", "output_changed"],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H29",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     False,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["INF-H29-BINOMTEST (fixed v32.0): verdict used raw rate thresholds only (>5%); added binomtest(k,n,p=0.02,greater) per layer for consistency with H38."],
        "notes":            (
            "Single-layer extension of Run 0017. Patch modes: none/L8/L16/L24/L31. "
            "none always runs first per (trial, turn) -- establishes baseline output "
            "for output_changed comparison. "
            "Primary metric: output_change_rate per layer -- fraction of turns where "
            "output differs from none condition. Threshold: >5% = causally sufficient. "
            "Secondary metric: sim_to_reference at final layer -- measures forward "
            "propagation of the injection through the remaining stack. "
            "Mixed-schema CSV (same as Run 0017): patch_layer=='none' rows have full "
            "trajectory metrics; patched rows have NaN for standard metrics (structural). "
            "Prerequisite: Run 0006 all-layers hidden states (_alllayers.npy). "
            "Connection: Run 0027/36 show correlational layer locality; Run 0018 shows "
            "causal layer sufficiency. If peak causal layer aligns with correlation "
            "peak from 24/36, this is the strongest layer-locality finding in the framework."
            "INF-H29-BINOMTEST (fixed v32.0): verdict used raw rate thresholds only; "
            "now requires binomtest(k,n,p=0.02,alternative='greater') p<0.05 per layer."
        ),
    },

    "H38": {
        "name":             "Patched State Does Not Change Output",
        "phase":            1,
        "status":           "pending",   # re-collection required after Bug M
        "data_runs":        [17],
        "comparison_runs":  [],
        "upstream_runs":    [6],          # Run 0006 provides all-layers reference states
        "csv_files":        ["R0017_patching.csv"],
        "required_columns": ["patch_mode", "output_changed"],
        "extra_columns":    ["patch_mode", "output_changed"],
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H38",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     False,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [
            "RUN21-OUT-CHANGE (fixed v30.9): output_changed never written pre-v30.9.",
            "Bug M (v23.3 CRITICAL): all_layers file matching corrupted -- re-collect Run 0017.",
        ],
        "notes":            (
            "Multi-layer activation patching. patch_mode: none/partial/full. "
            "output_changed: 1 if output differs from none-condition baseline. "
            "Supported: overall change rate >= 10% (p < 0.05, binomtest vs 10%). "
            "Pre-Bug-M finding: 15.4% -- treat as preliminary until re-collected. "
            "Re-collect after Run 0006 completes with save_all_layers=True."
        ),
    },

    # ── v32.0 -- zero-cost hypotheses ─────────────────────────────────────────
    # H30/H31/H32 require no new runs. All data already collected by existing runs.

    "H30": {
        "name":             "Output Turn-1 Self-Referentiality Grows With Turns",
        "phase":            3,
        "status":           "pending",
        "data_runs":        [35],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0035_output_similarity.csv"],
        "required_columns": ["output_sim_turn1", "turn"],
        "extra_columns":    [],
        # BUG-FOUND-3 (fixed v36.4): output_sim_turn1 was in both required_columns and
        # extra_columns. Removed from extra_columns -- already covered by required_columns.
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H30",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            (
            "output_sim_turn1 already saved per row in Run 0035 (runners.py:2248). "
            "Zero additional data cost. Test: linregress(turn, output_sim_turn1) on "
            "introspection arm. Supported = positive slope, p < 0.05. "
            "Added v32.0 -- zero-cost extension of H27 data collection."
        ),
    },

    "H31": {
        "name":             "Arithmetic Accuracy Is Uncorrelated With Trajectory Consistency",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [9, 10, 11, 12],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["R0009_math.csv", "R0010_math.csv", "R0011_math.csv", "R0012_math.csv"],
        "required_columns": ["state_similarity_index", "correct"],
        "extra_columns":    [],
        # BUG-FOUND-3 (fixed v36.4): correct was in both required_columns and extra_columns.
        # Removed from extra_columns -- already covered by required_columns.
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H31",
        "saves_E_t":        True,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       [],
        "notes":            (
            "'correct' flag (1/0) already saved per row in Runs 0009-0012 (runners.py:736). "
            "Zero additional data cost. Test: pearsonr(state_similarity_index, correct) pooled. "
            "Supported = r > 0.15, p < 0.05. The strongest functional consequence claim "
            "in the framework: R predicts task performance. Added v32.0."
        ),
    },

    "H32": {
        "name":             "Shock Phrasing Has No Differential Effect On Recovery",
        "phase":            1,
        "status":           "pending",
        "data_runs":        [39],
        "comparison_runs":  [],
        "upstream_runs":    [],
        "csv_files":        ["Q0039_jolt.csv"],
        "required_columns": ["shock_variant", "layer_sim_mean", "is_shock", "is_recovery"],
        "extra_columns":    [],
        # BUG-FOUND-3 (fixed v36.4): shock_variant was in both required_columns and
        # extra_columns. Removed from extra_columns.
        "analysis_file":    "export_stats.py",
        "analysis_fn":      "_infer_outcomes:H32",
        "saves_E_t":        False,
        "saves_C_t":        False,
        "saves_hidden":     True,
        "saves_all_layers": False,
        "blocking":         False,
        "known_bugs":       ["BUG-FOUND-1 (fixed v36.4): H32 ANOVA ran on per-turn pooled observations (~75 per variant) rather than per-trial deltas (~25 per variant). The original code subtracted the pooled shock mean from all recovery-turn values across all trials, producing ~75-entry arrays instead of one scalar per trial. ANOVA df inflated from (3,~96) to (3,~296). Fixed: group by trial within variant, compute mean(recovery)-mean(shock) per trial, ANOVA on those ~25 scalars."],
        "notes":            (
            "shock_variant (0-3, cycling by trial%4) already saved per row in Run 0039 "
            "(runners.py:820). Zero additional data cost. Test: one-way ANOVA of "
            "recovery_delta across 4 shock phrasings. Supported = ANOVA p >= 0.05 "
            "(phrasing-robust recovery). Disproven = p < 0.05 (phrasing matters). "
            "Added v32.0. If supported, all Run 0039/11 results can be cited without "
            "phrasing-variant qualification."
        ),
    },
}


# ── v35.0 -- Phase 4: Coherence Transfer and Contradiction Recovery ──────────

DEPENDENCY_MAP["H33"] = {  # OUT OF PAPER SCOPE -- Appendix A only
    "name":             "Condition A Has No Task-Equivalent Compute Advantage",
    "phase":            4,
    "status":           "pending",
    "data_runs":        [25],
    "comparison_runs":  [],
    "upstream_runs":    [],
    "csv_files":        ["Q0025_lengths.csv"],
    "required_columns": ["peak_gpu_power", "elapsed_sec", "r_condition", "correct", "phase"],
    "extra_columns":    [],
    # FINDING-2 fix (v36.6): r_condition was duplicated in required_columns and extra_columns.
    # all_required_csv_columns() uses set union so functional impact is zero.
    # Consistent with BUG-FOUND-3 (v36.4) which fixed H10/H30/H31/H32/H34.
    # BUG-NEW-5 (fixed v36.3): compute_per_correct is derived in inference (peak_gpu_power*elapsed_sec),
    # not a CSV column -- listed in required_columns caused false-positive schema_mismatch.
    # compression_ratio mentioned in notes but never computed anywhere -- removed from extra_columns.
    # recovery_delta (H35) same class: derived per-trial in inference from sub-rows, not a CSV column.
    "analysis_file":    "export_stats.py",
    "analysis_fn":      "_infer_outcomes:H33",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     True,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       ["BUG-NEW-5 (fixed v36.3): required_columns had compute_per_correct (derived in inference, not a CSV column); extra_columns had compression_ratio (never computed anywhere). Both caused false-positive schema_mismatch on H33. H35 extra_columns also had recovery_delta (same class -- derived per-trial in inference, not saved to CSV). All three removed."],
    "notes":            (
        "8 enforcer priming turns + 5 arithmetic turns. Conditions: condition_a, "
        "condition_b, condition_c. Primary metric: compute_per_correct ratio. "
        "Supported: condition_a compute_per_correct <= 40% of condition_c (p<0.05) "
        "AND correct_rate not significantly lower. "
        "Note: compression_ratio (priming output_tokens ratio) was listed as a secondary "
        "metric in early design notes but is not computed anywhere in export_stats.py and "
        "is not saved in runners.py. It was removed from the schema in v36.3 (BUG-NEW-5). "
        "Do not re-add without implementing the computation in both files."
    ),
}

DEPENDENCY_MAP["H34"] = {
    "name":             "Disruption Magnitude Does Not Track Contradiction Response",
    "phase":            4,
    "status":           "pending",
    "data_runs":        [28],
    "comparison_runs":  [],
    "upstream_runs":    [],
    "csv_files":        ["Q0028_contradiction.csv"],
    "required_columns": ["disruption_magnitude", "turn", "contradiction_turn"],
    # BUG-FOUND-3 (fixed v36.4): contradiction_turn was only in extra_columns despite
    # being checked in the outer guard ('contradiction_turn' in sub22_h34.columns).
    # A column tested in the outer guard must be in required_columns so the schema
    # checker will flag its absence before inference runs. Moved from extra to required.
    # disruption_magnitude was duplicated in both required and extra -- removed from extra.
    # post_contradiction added to extra_columns: used inside the inference loop with
    # graceful fallback (BUG-36-6 fix, v36.1), is a real Run 0028 CSV column, and
    # should be verified by the schema checker when present.
    "extra_columns":    ["post_contradiction"],
    "analysis_file":    "export_stats.py",
    "analysis_fn":      "_infer_outcomes:H34",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     True,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       [],
    "notes":            (
        "Zero new data cost -- uses Run 0028 CSV (same as H05). disruption_magnitude (continuous signal) "
        "vs disruption_flag (binary, H05). Paired t-test within-trial: disruption_magnitude at turn 7 "
        "significantly higher than mean(turns 1-6). Supported: p<0.05, delta>0.01. "
        "If supported: disruption_magnitude is a continuous per-turn geometric disruption signal. "
        "Inference block only -- no new runs required."
    ),
}

DEPENDENCY_MAP["H35a"] = {
    "name":             "R Level Has No Effect On Contradiction Recovery (Group)",
    "phase":            4,
    "status":           "pending",
    "data_runs":        [26],
    "comparison_runs":  [],
    "upstream_runs":    [],
    "csv_files":        ["Q0026_recovery.csv"],
    "required_columns": ["state_similarity_index", "r_condition", "turn",
                         "contradiction_turn", "pre_contradiction", "post_contradiction"],
    "extra_columns":    [],
    "analysis_file":    "export_stats.py",
    "analysis_fn":      "_compute_outcomes:H35a",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     True,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       [],
    "notes":            (
        "Group arm of the R-level recovery test. Compares high_r vs low_r recovery_delta "
        "(mid_r excluded by design, consistent with the hypothesis spec that predicts "
        "directional ordering across extreme arms). Supported: high_r recovery_delta > "
        "low_r recovery_delta (p<0.05)."
    ),
}

DEPENDENCY_MAP["H35b"] = {
    "name":             "R Level Has No Effect On Contradiction Recovery (Per-Trial)",
    "phase":            4,
    "status":           "pending",
    "data_runs":        [26],
    "comparison_runs":  [],
    "upstream_runs":    [],
    "csv_files":        ["Q0026_recovery.csv"],
    "required_columns": ["state_similarity_index", "r_condition", "turn",
                         "contradiction_turn", "pre_contradiction", "post_contradiction"],
    # BUG (fixed v36.7): contradiction_turn, pre_contradiction, post_contradiction were in
    # extra_columns only despite all three being tested in the H35 outer guard:
    #   if ... and 'contradiction_turn' in sub44.columns
    #           and 'post_contradiction' in sub44.columns
    #           and 'pre_contradiction' in sub44.columns:
    # A column tested in the outer guard must be in required_columns so the schema checker
    # will flag its absence before inference runs. Same principle as H34 BUG-FOUND-3 (v36.4),
    # which moved contradiction_turn from extra to required in that entry for the same reason.
    # FINDING-2 fix (v36.6): r_condition was duplicated in required_columns and extra_columns.
    # all_required_csv_columns() uses set union so functional impact is zero.
    "extra_columns":    [],
    "analysis_file":    "export_stats.py",
    "analysis_fn":      "_compute_outcomes:H35b",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     True,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       ["BUG-NEW-2 (fixed v36.2): pre_contradiction missing from outer guard and "
                         "extra_columns -- KeyError if column absent; schema checker wouldn't flag it."],
    "notes":            (
        "Per-trial arm of the R-level recovery test. 3 R-condition arms (high/mid/low_r) × "
        "100 trials. Contradiction at turn 7. pre_sim = mean(state_similarity_index, turns 1-6). "
        "recovery_delta = mean(state_similarity_index, turns 8-13) - state_similarity_index(turn 7). "
        "Supported: Pearson r(pre_sim, recovery_delta) > 0.2, p<0.05 (pooled across all 3 arms). "
        "mid_r data participates in the pooled Pearson r test (correct -- tests monotonic ordering "
        "across the full R-level range)."
    ),
}

# ── v44.0.0 -- FIND-H09-GAP fix ───────────────────────────────────────────────
# Runs 0029, 0030, 0031 (impossibility cluster) were collected but had no DEPENDENCY_MAP
# entry and no inference block. Data exists on disk; analysis layer added here.

DEPENDENCY_MAP["H39"] = {
    "name":             "Impossibility Has No Geometric Effect",
    "phase":            1,
    "status":           "pending",
    "data_runs":        [29, 30, 31],
    "comparison_runs":  [4, 5, 1],
    "upstream_runs":    [],
    "csv_files":        ["Q0029_limit.csv", "Q0030_limit.csv", "Q0031_limit.csv"],
    "required_columns": ["state_similarity_index", "layer_sim_mean"],
    "extra_columns":    [],
    "analysis_file":    "export_stats.py",
    "analysis_fn":      "_infer_outcomes:H39",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     True,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       ["FIND-H09-GAP (fixed v44.0.0): runs 0029-0031 collected but no DEPENDENCY_MAP entry and no inference block existed. Analysis layer added."],
    "notes": (
        "Three sub-conditions: Run 0029 (constrained impossible, status enforcer), "
        "Run 0030 (unconstrained impossible, free output), Run 0031 (epistemic impossible, free). "
        "No throughline for any -- cold start, cleaner collapse signal. "
        "Prediction: impossibility prompts produce distinctive geometry vs null baseline. "
        "Null: state_similarity_index and layer_sim_mean indistinguishable from Runs 0004, 0005, 0001. "
        "Supported: significant difference (either direction) between [29,30,31] and [4,5,1]. "
        "Disproven: no significant difference -- impossible prompts leave no geometric trace. "
        "Run 0029 uses status enforcer (1-token outputs) -- same constraint as null Run 0004. "
        "Runs 0030/0031 use long outputs -- different from null. "
        "Per-run breakdown reported in implication to isolate enforcer vs free-output effect."
    ),
}

DEPENDENCY_MAP["H40"] = {
    "name":             "Random Noise Patching Matches Real Patching Effect",
    "phase":            1,
    "status":           "pending",
    "data_runs":        [19],
    "comparison_runs":  [17],
    "upstream_runs":    [6],
    "csv_files":        ["R0019_random_patching.csv"],
    "required_columns": ["output_changed", "patch_mode"],
    "extra_columns":    [],
    "analysis_file":    "export_stats.py",
    "analysis_fn":      "_infer_outcomes:H40",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     False,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       [],
    "notes": (
        "Random noise control for H38 (Run 0017). Run 0019 patches with random Gaussian "
        "noise instead of real high-R geometry. Two-proportion z-test: if Run 0017 "
        "output_change_rate is not significantly higher than Run 0019, the causal claim "
        "is confounded -- any perturbation would produce the effect. "
        "v0.71.0.0: added."
    ),
}

DEPENDENCY_MAP["H36"] = {
    "name":             "Causal Effect Of Patching Is Temperature-Invariant",
    "phase":            1,
    "status":           "pending",
    "data_runs":        [17],
    "comparison_runs":  [],
    "upstream_runs":    [6],
    "csv_files":        ["R0017_patching.csv"],
    "required_columns": ["output_changed", "patch_mode", "temperature"],
    "extra_columns":    [],
    "analysis_file":    "export_stats.py",
    "analysis_fn":      "_infer_outcomes:H36",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     False,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       ["v54.2.2: temperature hardcoded to 0.0 in all prior versions -- T=0.2 data collected deterministically. Re-collect T=0.2 on v54.2.2+."],
    "notes": (
        "Requires Run 0017 collected at each temperature round with v54.2.2+ code. "
        "Pools R21_patching.csv across all temperature directories. "
        "Pearson r(temperature, output_change_rate) < -0.8 and Kruskal-Wallis p < 0.05 to support. "
        "Needs >= 3 temperature rounds with valid data to evaluate. "
        "temperature column only present in v54.2.2+ collections."
    ),
}

DEPENDENCY_MAP["H37"] = {
    "name":             "Layer Causal Sufficiency Is Temperature-Invariant",
    "phase":            1,
    "status":           "pending",
    "data_runs":        [18],
    "comparison_runs":  [],
    "upstream_runs":    [6],
    "csv_files":        ["R0018_layer_isolation.csv"],
    "required_columns": ["output_changed", "patch_layer", "temperature"],
    "extra_columns":    [],
    "analysis_file":    "export_stats.py",
    "analysis_fn":      "_infer_outcomes:H37",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     False,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       ["v54.2.2: temperature hardcoded to 0.0 in all prior versions -- T=0.2 data collected deterministically. Re-collect T=0.2 on v54.2.2+."],
    "notes": (
        "Requires Run 0018 collected at each temperature round with v54.2.2+ code. "
        "Pools Q42_layer_isolation.csv across all temperature directories. "
        "Two-way ANOVA: output_changed ~ C(patch_layer) * C(temperature). "
        "Interaction p < 0.05 to support -- layer profile flattens at high temperature. "
        "Requires statsmodels (added to setup.py REQUIRED_PACKAGES in v54.2.2). "
        "temperature column only present in v54.2.2+ collections."
    ),
}

DEPENDENCY_MAP["H50"] = {
    "name":             "MLP Improves On Ridge By More Than 0.01",
    "phase":            2,
    "status":           "pending",
    "data_runs":        [42],
    "comparison_runs":  [],
    "upstream_runs":    [6, 7, 8, 13, 14, 15, 1, 3, 23],
    "csv_files":        [],
    "required_columns": [],
    "extra_columns":    [],
    "analysis_file":    "analysis.py",
    "analysis_fn":      "_run_decomposition (Run 0042) -- linearity_check field",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     False,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       [],
    "notes": (
        "Linearity validation. Three MLP configs (256, 64, 128-64) vs Ridge on "
        "Model D features. 80/20 split, default sklearn, early stopping, no tuning. "
        "Pass: max gap < 0.01. Output: Q33_decomposition.json linearity_check field. "
        "v0.73.2.0: added."
    ),
}

DEPENDENCY_MAP["H51"] = {
    "name":             "Interaction Information Exceeds 5% Of Joint MI",
    "phase":            2,
    "status":           "pending",
    "data_runs":        [42],
    "comparison_runs":  [],
    "upstream_runs":    [6, 7, 8, 13, 14, 15, 1, 3, 23],
    "csv_files":        [],
    "required_columns": [],
    "extra_columns":    [],
    "analysis_file":    "analysis.py",
    "analysis_fn":      "_run_decomposition (Run 0042) -- interaction_info field",
    "saves_E_t":        False,
    "saves_C_t":        False,
    "saves_hidden":     False,
    "saves_all_layers": False,
    "blocking":         False,
    "known_bugs":       [],
    "notes": (
        "Ordering sensitivity. II = MI(S;S_t) + MI(S;E_t) - MI(S;S_t,E_t). "
        "Two methods: linear proxy (Gaussian MI from R²) and kNN non-parametric "
        "(stratified 500/run, PCA to 32 dims). Pass: |II/joint| < 0.05 in both. "
        "Per-condition breakdown reported. Output: Q33_decomposition.json "
        "interaction_info field. v0.73.2.0: added."
    ),
}

# ── v0.77.0.0 -- Planned hypothesis dependency entries ──────────────────────
# Schema present in HYPOTHESES (export_stats.py); inference blocks pending.
# These entries document data-flow requirements so scanner/orchestration
# can correctly identify which runs gate each hypothesis.

DEPENDENCY_MAP["H41"] = {
    "name":             "System Prompt Has No Geometric Effect",
    "phase":            1, "status": "pending",
    "data_runs":        [2], "comparison_runs": [4, 5, 1], "upstream_runs": [],
    "csv_files":        ["R0002_system_prompt.csv"],
    "required_columns": ["state_similarity_index"], "extra_columns": [],
    "analysis_file":    "export_stats.py", "analysis_fn": "_compute_outcomes:H41",
    "saves_E_t": True, "saves_C_t": True, "saves_hidden": True, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Compares similarity index under present vs absent system prompt. Supported = no effect (p >= 0.05). Inference pending.",
}

DEPENDENCY_MAP["H42"] = {
    "name":             "Similarity Does Not Accumulate With Turn Count",
    "phase":            1, "status": "pending",
    "data_runs":        [1], "comparison_runs": [], "upstream_runs": [],
    "csv_files":        ["R0001_null.csv"],
    "required_columns": ["state_similarity_index", "turn"], "extra_columns": [],
    "analysis_file":    "export_stats.py", "analysis_fn": "_compute_outcomes:H42",
    "saves_E_t": True, "saves_C_t": True, "saves_hidden": True, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Slope of similarity index vs turn in null condition. Supported = no accumulation (slope ~ 0). Inference pending.",
}

DEPENDENCY_MAP["H43"] = {
    "name":             "R Fraction Is Sensitive To Projection Dimension",
    "phase":            5, "status": "pending",
    "data_runs":        [41], "comparison_runs": [], "upstream_runs": [42, 43],
    "csv_files":        [],
    "required_columns": [], "extra_columns": [],
    "analysis_file":    "analysis.py", "analysis_fn": "_run_pool_dim_sweep (Run 0041)",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": False, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "POOL_DIM sweep at T=0.0. Disproven = R varies > 0.05 across adjacent dimensions. Inference pending.",
}

DEPENDENCY_MAP["H44"] = {
    "name":             "R Fraction Is Condition-Invariant",
    "phase":            5, "status": "pending",
    "data_runs":        [44], "comparison_runs": [], "upstream_runs": [42, 43],
    "csv_files":        [],
    "required_columns": [], "extra_columns": [],
    "analysis_file":    "analysis.py", "analysis_fn": "_run_per_condition_R (Run 0044)",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": False, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Per-condition R fraction from Run 0044. Supported = all conditions statistically indistinguishable. Inference pending.",
}

DEPENDENCY_MAP["H45"] = {
    "name":             "Resistant Prime Does Not Damage Trajectory More Than Cooperative",
    "phase":            5, "status": "pending",
    "data_runs":        [50], "comparison_runs": [], "upstream_runs": [],
    "csv_files":        ["Q0050_priming_vulnerability.csv"],
    "required_columns": ["disruption_flag", "prime_label"], "extra_columns": [],
    "analysis_file":    "analysis.py", "analysis_fn": "_run_cross_temp_synthesis (Run 0050) §1",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": True, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Tests whether resistant-framed primes elevate disruption_flag rate vs cooperative primes. Inference pending.",
}

DEPENDENCY_MAP["H46"] = {
    "name":             "Contradiction Does Not Drop Similarity",
    "phase":            5, "status": "pending",
    "data_runs":        [50], "comparison_runs": [28], "upstream_runs": [],
    "csv_files":        ["Q0050_cross_temp_synthesis.csv"],
    "required_columns": ["state_similarity_index"], "extra_columns": [],
    "analysis_file":    "analysis.py", "analysis_fn": "_run_cross_temp_synthesis (Run 0050) §2",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": True, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Phase comparison: pre-contradiction mean similarity vs post-contradiction. Inference pending.",
}

DEPENDENCY_MAP["H47"] = {  # OUT OF PAPER SCOPE -- Appendix A only
    "name":             "Coherence Transfer Is Not Compute-Efficient",
    "phase":            5, "status": "pending (out of scope -- Appendix A)",
    "data_runs":        [50], "comparison_runs": [25], "upstream_runs": [],
    "csv_files":        ["Q0050_cross_temp_synthesis.csv"],
    "required_columns": [], "extra_columns": [],
    "analysis_file":    "analysis.py", "analysis_fn": "_run_cross_temp_synthesis (Run 0050) §6",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": False, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Framework hypothesis -- moved to Appendix A of the measurement paper. Schema retained for UI completeness. Inference pending.",
}

DEPENDENCY_MAP["H48"] = {
    "name":             "R Decay Is Explained By KV Cache Growth",
    "phase":            5, "status": "pending",
    "data_runs":        [50], "comparison_runs": [24], "upstream_runs": [],
    "csv_files":        ["Q0050_cross_temp_synthesis.csv"],
    "required_columns": ["state_similarity_index", "history_mode"], "extra_columns": [],
    "analysis_file":    "analysis.py", "analysis_fn": "_run_cross_temp_synthesis (Run 0050) §7",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": False, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Persistence-mode comparison: full-history vs last-exchange vs summary. Inference pending.",
}

DEPENDENCY_MAP["H49"] = {
    "name":             "Introspection R Does Not Transfer To Arithmetic",
    "phase":            5, "status": "pending",
    "data_runs":        [50], "comparison_runs": [36], "upstream_runs": [],
    "csv_files":        ["Q0050_cross_temp_synthesis.csv"],
    "required_columns": ["state_similarity_index", "phase"], "extra_columns": [],
    "analysis_file":    "analysis.py", "analysis_fn": "_run_cross_temp_synthesis (Run 0050) §8",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": False, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Condition-transfer test at switch boundary. Inference pending.",
}

DEPENDENCY_MAP["H52"] = {
    "name":             "Disruption Magnitude Is Stationary Across Non-Contradiction Turns",
    "phase":            4, "status": "pending",
    "data_runs":        [28], "comparison_runs": [], "upstream_runs": [],
    "csv_files":        ["Q0028_contradiction.csv"],
    "required_columns": ["disruption_magnitude", "turn"], "extra_columns": [],
    "analysis_file":    "export_stats.py", "analysis_fn": "_compute_outcomes:H52",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": False, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "ADF test + autocorrelation on baseline-turn disruption_magnitude. Precondition for trusting H34's paired-t. Inference pending.",
}

DEPENDENCY_MAP["H53"] = {
    "name":             "Cross-Stochasticity Disruption Rate Is Monotonic",
    "phase":            4, "status": "pending",
    "data_runs":        [28, 38], "comparison_runs": [], "upstream_runs": [],
    "csv_files":        ["Q0028_contradiction.csv", "Q0038_hesitation.csv"],
    "required_columns": ["disruption_flag", "temperature"], "extra_columns": [],
    "analysis_file":    "export_stats.py", "analysis_fn": "_compute_outcomes:H53",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": False, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Spearman |rho| vs temperature across T={0.0, 0.2, 0.4, 0.6, 0.8, 1.0}. Temperature here = sampling stochasticity. Inference pending.",
}

DEPENDENCY_MAP["H58"] = {
    "name":             "E + C + R Fractions Sum To 1.0 Within Tolerance",
    "phase":            2, "status": "pending",
    "data_runs":        [43], "comparison_runs": [], "upstream_runs": [42],
    "csv_files":        [],
    "required_columns": [], "extra_columns": [],
    "analysis_file":    "export_stats.py", "analysis_fn": "_compute_outcomes:H58",
    "saves_E_t": False, "saves_C_t": False, "saves_hidden": False, "saves_all_layers": False,
    "blocking": False, "known_bugs": [],
    "notes": "Sanity diagnostic on Q34 permutation partition. Pass: |E+C+R - 1.0| <= 0.01 across all cells. Inference pending.",
}

def runs_required_for(hypothesis_id: str) -> set:
    """All runs (data + upstream) that must complete before this hypothesis can be evaluated."""
    h = DEPENDENCY_MAP[hypothesis_id]
    return set(h["data_runs"]) | set(h["upstream_runs"]) | set(h["comparison_runs"])


def hypotheses_blocked_by(run_num: int) -> list:
    """All hypotheses that cannot be evaluated without run_num completing."""
    return [hid for hid, h in DEPENDENCY_MAP.items()
            if run_num in runs_required_for(hid)]


def ct_source_runs() -> list:
    """Runs that save real C_t embeddings (excludes zero-vector saves in Run 0001)."""
    return [r for h in DEPENDENCY_MAP.values()
            for r in h["data_runs"] if h["saves_C_t"]]


def et_source_runs() -> list:
    """Runs that save E_t embeddings."""
    return [r for h in DEPENDENCY_MAP.values()
            for r in h["data_runs"] if h["saves_E_t"]]


def all_required_csv_columns() -> dict:
    """Map of csv_file → set of columns that must be present."""
    result = {}
    for h in DEPENDENCY_MAP.values():
        cols = set(h["required_columns"]) | set(h["extra_columns"])
        for f in h["csv_files"]:
            result.setdefault(f, set()).update(cols)
    return result


# ── Checker ────────────────────────────────────────────────────────────────────
