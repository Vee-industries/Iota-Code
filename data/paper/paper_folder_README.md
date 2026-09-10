# data/paper/

The released paper-facing data: everything the three papers quote a number from, plus the
figures. About 12 MB, no hidden-state arrays. Tracked in git since v1.0.1 so a fresh clone can
check every number in the papers without collecting anything (`python reproduce.py --only verify`).

Label convention for the papers: A = technical note on R (2026a), B = apparatus (2026b),
C = One iota (2026c).

## Layout

```
data/paper/
├── results8th.json                     # THE released results object (24 cells; B §6, C §4 and App. C read it)
├── Q0057_function_class_sensitivity.json   # Run 0057: four-class permutation shares per cell (Ridge/MLP/RF/RKHS)
├── Q0058_apparatus_manifest.json       # Run 0058 phase manifest
├── calibration/                        # every calibration artifact (see below)
├── behavioral_trial_response/          # Run 0061 cohort-discrimination analysis on Runs 0039, 0060, 0062 (C §3)
├── reviewer_response/, reviewer_three_response/, v26_response/   # per-review follow-up pulls the papers cite
├── verification/v17_1_paper_vs_disk_audit.json   # paper-vs-disk audit record
├── appendix_inputs/appendix_source_data 4pm.json  # B appendices: apparatus config, V5 constructions, validation gates
├── data_integrity/                     # collection integrity checks and the temperature-seed bug reproduction
├── statsbot_inputs/                    # pulls used for the statistics review
├── recollection_status_*.json          # May 2026 re-collection logs
├── fig1..fig6 .{svg,pdf} + .meta.json  # paper figures with provenance sidecars (results hash at render time)
└── paper_figure_specs.md               # figure content spec
```

## results8th.json versus results.json

`results8th.json` is the frozen results object the papers cite (generated 2026-05-08, iota 0.82.0.23;
its `anchored_shares` equal the 2026-05-17 Run 0058 aggregator output to 5e-7). The pipeline writes
`results.json` when Run 0059 runs; `reproduce.py --only verify` compares a rebuilt `results.json`
against `results8th.json` cell by cell. Do not overwrite `results8th.json`.

## calibration/

| folder or file | produced by | used in |
|---|---|---|
| `v5/v5_calibration_results.json` | `v5_synthetic_calibration.py` | B §5 (V5a-V5f) |
| `v5/v5_lambda_calibration.json`, `v5/lambda_calibration_outcome.json` | `_run_v5_lambda_calibration.py`, `_finalize_lambda_calibration.py` | B §5.8 |
| `v5/v5g_bridge_calibration.json`, `v5/v5g_d128_reproduction_2026-09-09.txt` | `_run_v5g_mid_dim_bridge.py --n-steps 8000` | B §5.9 |
| `v5/h4_threshold_derivation.json` | `_derive_h4_threshold.py` | B §5.7, App. E (tau_H4 = 0.72) |
| `v5d_threshold/calibration.json` | Run 0058 phase 3 | B §5.4 |
| `split_pca_selection/`, `knn_mi_reliability/` | `run_split_pca_selection.py`, `run_knn_mi_reliability.py` | B §4.1, App. A |
| `kraskov_spike/`, `lagrangian_solver/`, `kraskov_anchor/`, `apparatus_aggregator/` | Run 0058 phases 1, 2, 4, 5 | B §3.3-3.4, §6 (headline layer) |
| `function_class_p2/`, `kraskov_anchor_p2/`, `apparatus_p2/`, `stacking_baseline_p2/`, `bootstrap_variance_p2/` (n_bootstrap 12 on the six subset cells), `geometric_diagnostics_p2/`, `estimator_joint_R2_p2/`, `hull_diagnostic_p2/` | Run 0058 phases 6-11 | B §3.4-3.6, §6.2-6.7 (the "paper-2 layer") |
| `channel_marginal/`, `ridge_bias/`, `toy_nonlinearity/`, `Q0056_calibration_manifest.json` | Run 0056 and the toy scripts | B §7.1, Fig. 2 |
| `gauss_ii_fleet.json` | `_gauss_ii_fleet.py fleet` | B §2.1, §5.4 (redundancy on de-duplicated rows) |
| `multistep_R_fleet.json` | `_multistep_R.py` | B §6.9 (persistence R_k) |
| `kv_interpolation_gemma_2b_4bit_abliterated_deterministic.json` | `_kv_interpolation_test.py` | B §3.1 |
| `drop_history_probe_gemma_2b_q4_T0.json` | `_drop_history_probe.py` | C §3.4 |

Backups (`*.bak*`, `_bak/`) and pull checkpoints (`_*checkpoint*.json`) are kept on the authoring
machine and excluded from git.

## Provenance

Each `fig{N}.meta.json` records the SHA-256 of the results file at render time. The v1.0.1
changelog entry summarizes what changed in this folder during the 2026-09 consistency pass that
produced the current papers.
