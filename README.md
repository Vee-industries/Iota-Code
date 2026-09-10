# IOTA

**A measurement apparatus for transformer hidden states.**

Information-theoretic Operational Triangulation Apparatus. Measures how much
of a transformer's next hidden state comes from its prior internal state
versus its current input. Calibrated against synthetic systems with known
ground truth, so the numbers you get on real models have documented
uncertainty bounds.

Built for a single 10 GB consumer GPU. Runs on Windows or Linux. Pick a
model, push a button.

---

## What it does

IOTA decomposes next-state predictability in a transformer's hidden states
into three channels:

- **E** — current input
- **C** — training constraint (instruction-tuning / RLHF / abliteration delta)
- **R** — prior hidden state, holding input fixed

The decomposition `E + C + R = 1` is exact by construction. The apparatus
estimates the three shares using four function-class families (Ridge, MLP,
RKHS, Random Forest), anchored against a kNN mutual-information estimator,
aggregated on the probability simplex.

It's calibrated. The apparatus's behavior is characterized on a six-system
synthetic suite (V5a–V5f) with known ground truth, so when you read a number
off the apparatus on a real model, you know how trustworthy it is in that
regime.

It's reproducible. Every collection and analysis run is dispatched the same
way, `python start_here.py --single-run N`; the calibration scripts and the
cohort test are standalone; `python reproduce.py` sequences all of it. The
full pipeline from raw model weights to paper figures is one repository.

---

## Quickstart

```bash
# 1. First-time setup (installs PyTorch with CUDA, transformers, etc.)
python setup.py

# 2. Open the dashboard
./iota.sh             # Linux / macOS
iota.bat              # Windows
                      # → http://localhost:5000

# 3. Or run a single experiment from the CLI
python start_here.py --single-run 39    # Run 0039 jolt (shock-injection cohort test)
python start_here.py --single-run 1     # Run 0001 null isolation (C_t / E_t substrate)
python start_here.py --single-run 3     # Run 0003 temperature grid
```

After any data-collection run, verify the data is sane:

```bash
python post_collection_integrity_check.py --run 39
```

Expected output: `Verdict: PASS`. The check is a cross-temperature md5
audit that catches the kind of silent bug that produces "data that looks
right but isn't" — for example, a seed scheme that doesn't differentiate
across temperatures (see `CHANGELOG.md` v1.0.0 entry for the case study).

---

## Hardware

- **GPU:** one consumer card, ≥10 GB VRAM. Tested on RTX 3080 10 GB.
  Smaller cards (8 GB) need smaller models (≤3B) and 4-bit quantization.
- **Disk:** 50–100 GB for a 24-cell fleet (4 model configs × 6 temperatures
  × 100 trials × 13–16 turns of hidden states).
- **CPU + RAM:** any modern Python 3.10+ machine.

If you have less than 8 GB of VRAM, the framework still works — you'll just
be restricted to smaller models.

---

## What's in the box

| Layer | Files | What it does |
|---|---|---|
| **Entry point** | `start_here.py` | Interactive UI + `--single-run N` dispatch |
| **Collection** | `runners.py`, `runners_p1/p2/p3.py`, `runners_core.py` | The 40 generation runs |
| **Inference** | `orchestration_core.py` | Model loading, hidden-state extraction, sampling |
| **Models** | `vault.py` | Model catalog: 26+ verified base/instruct/abliterated triplets |
| **Analysis** | `analysis.py`, `decomposition_p2.py`, `function_class_p2.py` | The chain-rule decomposition + four-class triangulation |
| **Calibration** | `v5_synthetic_calibration.py`, `run_v5d_threshold_calibration.py` | The V5a–V5f synthetic suite |
| **Aggregation** | `bayesian_solver.py`, `run_bayesian_apparatus.py` | I-projection on the simplex |
| **Diagnostics** | `bootstrap_variance_p2.py`, `geometric_diagnostics_p2.py` | Bootstrap, hull, joint-R² |
| **Output** | `export_stats.py`, `report.py`, `results_builder.py`, `fig*.py` | Paper figures + tables |
| **Dashboard** | `export_flask.py` | Live web UI, sparklines, pause/resume, run grid |
| **Utility** | `post_collection_integrity_check.py` | Post-collection data-integrity audit |

---

## Run map

The framework has 62 runs across three phases.

- **Phase A (1–3):** temperature-independent foundation (null isolation,
  robustness sweep, temperature grid).
- **Phase B (4–15):** fast E_t-source block (null baselines, introspection,
  arithmetic, priming).
- **Phase G (16):** E_t recovery meta-run.
- **Phase D (17–19):** patching cluster (activation patching, layer causal
  sufficiency, noise patching baseline).
- **Phase EFC (20–40):** slow→fast collection (cross-instance, coherence,
  saturation, jolt, impossibility, contradiction, etc.).
- **Per-temp analysis (41–49):** decomposition, Granger probes, baseline
  swap.
- **Pooled analysis (50–51):** cross-temperature synthesis.
- **Cross-model (52–54):** comparison + concordance + paper summary.
- **Paper output (55–56, 59):** stats export, methodology calibration, and
  paper assembly (results.json + figures).
- **Apparatus orchestration (57–58):** function-class sensitivity + the full
  Bayesian apparatus (kraskov anchor → I-projection → threshold → aggregator
  → paper-2 measurement layer).
- **Cohort follow-ups (60, 62) and the cohort test (61):** non-thematic and
  enforcer-off jolt collections; Run 0061 is the standalone analysis script
  `_run_run0061_jolt_carryover.py --run 39|60|62`.

Full hypothesis-to-run map is in `start_here.py`'s docstring and on the
dashboard's Hypotheses tab.

---

## Reproducing the papers, end to end

One command runs the whole pipeline unattended and resumes wherever it stopped:

```bash
python reproduce.py --plan      # see the stages
python reproduce.py             # run everything (collection is 1-2 days per model on a 10 GB card)
python reproduce.py --skip-collection   # analysis, calibration, results, probes, verify (CPU, a few hours)
```

Stages, in order: `env` (packages, CUDA, session file, unit tests) -> `collect` (Runs 0001, 0003, 0002 at T=0; runs 4-15, 23, 39, 60 and the E_t recovery Run 0016 at each of six temperatures; 62, 17, 18 at T=0; four model configurations in size order) -> `analysis` (`--all-stats`, runs 41-54 per model) -> `calibration` (operating point, kNN-MI reliability, V5 suite, Runs 0056-0058, lambda calibration, V5g bridge, tau_H4) -> `behavioral` (Run 0061 on Runs 0039/0060/0062) -> `results` (Run 0059: results.json + figures) -> `probes` (the 2026-09-10 measurements: Gaussian redundancy, multi-step persistence, KV interpolation, drop-history) -> `verify` (rebuilt results against the released `data/paper/results8th.json` and the numbers the papers quote).

Every stage is idempotent: the framework's own resume logic skips completed trials, the calibration scripts skip existing outputs, and `data/paper/reproduce_status.json` records finished stages. GPU stages wait for the card to drain and the 9B model runs with `IOTA_VRAM_FRACTION=0.93` (the 0.85 default fails at load on 10 GB). Set `HF_TOKEN` in the environment if a model on your list is gated. Log: `.iota_reproduce.log`.

Which runs each paper needs:

| Paper | Substrate runs |
|---|---|
| A: Technical note on R | none (theory) |
| B: IOTA apparatus | 1, 2, 3, 4-15, 16, 17, 18, 23, 41-59; probes |
| C: One iota | 39, 60, 62, 61; drop-history probe |

The released `data/paper/` folder (12 MB) is what the papers cite; a fresh clone can check every quoted number against it without collecting anything (`python reproduce.py --only verify`).

---

## Verifying it works

There's a unit-test suite that should pass before you commit time to
collection:

```bash
python -m unittest discover -s tests
```

Expected: `OK (skipped=2)`. Two tests are skipped with documented rationale
(stale fixtures, not hidden bugs).


---

## When something goes wrong

The framework has a built-in dashboard at `http://localhost:5000`. It shows
live console output, run-status grid, hypothesis scoreboard, dependency map,
and the current data-integrity state. Most diagnostic information surfaces
there before you'd see it in the console.

The `post_collection_integrity_check.py` utility catches the most common
class of silent failure (cross-temperature data duplication caused by
sampler or seed bugs). Run it after every collection batch.

If a run interrupts midway, the framework's resume logic (`get_trials_to_run`
in `runners_core.py`) detects which trials are complete and restarts from
the next missing trial. Trial completeness is gap-aware: if trial 47 was
interrupted at turn 8 of 13, its partial CSV rows are stripped and trial 47
re-runs from scratch on resume.

---

## Architecture

Each run is a thin cartridge: a `turn_fn` and a list of prompts, passed to
`_standard_trial_loop` in `runners_core.py`. The trial loop handles model
state, seed scheme, CSV writing, hidden-state extraction, and resume logic.

Path resolution is a single function, `get_paths(family, size, variant,
temperature)` in `cartography.py`. Every file write goes through it. No
hardcoded paths anywhere.

File naming follows a fixed convention: `R0001_*` for proof runs (0001–0021),
`Q0022_*` and up for quantify/analysis runs, with 4-digit zero-padded
run IDs and trial/turn suffixes. The convention is enforced by `run_prefix`
and `run_id_pad` in `cartography.py`.

Data layout:
```
data/{family}/{size}/{variant}/
    deterministic/      # T=0.0 + temp-independent runs (R0001, R0002, R0003)
    temp_0.2/
    temp_0.4/
    temp_0.6/
    temp_0.8/
    temp_1.0/
        csv/            # per-run CSVs
        hidden_states/  # per-(trial, turn) .npy files
        analysis/       # per-temperature analysis outputs
        visuals/        # per-temperature figure outputs
```

Cross-temperature pooled output lives at `data/paper/`.

---

## Citation

If IOTA is part of your work, please cite the companion paper(s):

```bibtex
@misc{vaillancourt2026rnote,
  title  = {Technical note on R: formal properties of the prior-state
            predictive share in transformer-like systems},
  author = {Vaillancourt, Kevin},
  year   = {2026},
  note   = {Submitted to Information and Inference; arXiv preprint},
}

@misc{vaillancourt2026iota,
  title  = {IOTA: a calibrated apparatus for chain-rule decomposition of
            predictive information in transformer hidden states},
  author = {Vaillancourt, Kevin},
  year   = {2026},
  note   = {Submitted to TMLR; arXiv preprint},
}

@misc{vaillancourt2026oneiota,
  title  = {One iota: hidden-state carryover in transformer language
            models is measurable},
  author = {Vaillancourt, Kevin},
  year   = {2026},
  note   = {arXiv preprint},
}
```

See `CITATION.cff` for a machine-readable citation file.

---

## License

**PolyForm Noncommercial License 1.0.0.** See `LICENSE`.

Personal use, research, education, hobby projects, government, and
charitable / non-profit use are explicitly permitted. Commercial use
requires a separate license. Contact the author if you want to use IOTA
commercially.

---

## Author

**Kevin Vaillancourt** — independent researcher. kvsudbury@gmail.com

This is single-author work. Issues, questions, and pull requests are
welcome at [github.com/Vee-industries/Iota-Code](https://github.com/Vee-industries/Iota-Code).

---

## Changelog

See `CHANGELOG.md` for the full version history. Most recent change
(v1.0.1, 2026-09-10): `reproduce.py` one-command pipeline; Run 0023
collected on all fleet cells; `data/paper/` tracked in the repo; four new
measurement scripts (Gaussian redundancy, multi-step persistence, KV
interpolation, drop-history) and the tau_H4 derivation.
