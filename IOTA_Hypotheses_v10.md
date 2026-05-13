**IOTA FRAMEWORK**

Hypotheses and Experimental Design

*Version 24.1 | IOTA Framework v0.81.1.2 | April 2026 | 53 Runs | 51 Hypotheses in scope + 3 in appendix*

> **Authority for paper claims:** see `paper_framing.md` for the
> framing-and-layout spec. This document is the hypothesis registry
> — it defines what was asked. The framing spec defines what is
> claimed in the paper. Where they appear to conflict, the framing
> spec wins for paper-bound prose; this document wins for the
> experimental record.

---

## How to Read This Registry

Every entry below states a **null hypothesis** (H0ₙ — standard
notation for a null in statistical testing) and the conditions under
which it would be rejected or supported. The framework is
falsification-driven: every hypothesis is a question, and every
outcome — null rejected, null confirmed, or out of scope — is a
data point. Neither rejecting nor confirming a null is "the right
answer." The data answers the question and the registry records
both.

Each entry carries:

- **Null hypothesis:** the claim being tested
- **Why this matters:** what the test exists for
- **Pass criterion** or **Prediction:** what the data should look
  like if the null fails
- **Falsification:** what the data should look like if the null
  holds
- **Runs:** the runs that produce the data (in original numbering
  scheme — see translation table below for canonical 4-digit IDs)
- **First registered:** approximate month the hypothesis entered
  the registry
- **Outcome:** Null rejected / Null confirmed / Pending / Out of
  scope, plus the substantive finding in plain language where the
  data has come in

## Run Number Translation

This document uses original 1–58 numbering (the form the
hypotheses were registered under). Code and `cartography.py`
canonicalize to 4-digit IDs (0001–0056). Mapping:

| Original | Canonical | Original | Canonical | Original | Canonical |
|----------|-----------|----------|-----------|----------|-----------|
| 1  | 0004 | 20 | 0002 | 39 | 0035 |
| 2  | 0005 | 21 | 0017 | 40 | 0051 |
| 3  | 0006 | 22 | 0028 | 41 | 0033 |
| 4  | 0007 | 23 | 0022 | 42 | 0018 |
| 5  | 0008 | 24 | 0027 | 43 | 0025 |
| 6  | 0009 | 25 | 0047 | 44 | 0026 |
| 7  | 0010 | 26 | 0003 | 45 | 0041 |
| 8  | 0011 | 27 | 0048 | 46 | 0044 |
| 9  | 0012 | 28 | 0023 | 47 | 0050 |
| 10 | 0039 | 29 | 0024 | 49 | 0045 |
| 11 | 0040 | 30 | 0020 | 53 | 0019 |
| 12 | 0029 | 31 | 0021 | 56 | 0046 |
| 13 | 0030 | 32 | 0049 |    |      |
| 14 | 0031 | 33 | 0042 |    |      |
| 15 | 0013 | 34 | 0043 |    |      |
| 16 | 0014 | 35 | 0038 |    |      |
| 17 | 0015 | 36 | 0034 |    |      |
| 18 | 0032 | 37 | 0037 |    |      |
| 19 | 0001 | 38 | 0036 |    |      |

When citing runs in code or new analysis, use canonical 4-digit
form. When reading entries here, the original numbers are the
historical record of what was registered.

---

## The Central Claim

Every response a language model generates is determined by three
sources: what arrived in the prompt (E, external input), what the
model is constrained to produce (C, constraint pressure from
training and system instructions), and something from the model's
own prior trajectory (R, internal causation). The IOTA decomposition
formalises this as **E + C + R = 1**, where each term represents a
fraction of causal determination over the next hidden state.

R is what we are measuring. It is the fraction of next-state
determination attributable to the model's internal hidden-state
trajectory beyond what can be explained by current input and
constraints. If R is zero, the model is a pure lookup function. If R
is large, prior turns are causally upstream of what it produces.

> **Why this matters.** Interpretability research typically asks
> "which circuit does what." We ask a prior question: to what degree
> does the model's own trajectory cause its outputs at all? This is
> a question about causal fraction, not mechanism. It can be
> answered without access to weights, and the answer has direct
> implications for robustness, interpretability, and what it means
> for a model to be coherent across a conversation.

## The Measurement Problem

The core difficulty: you cannot observe R directly. E, C, and R are
all operating simultaneously on every turn. The measurement problem
is to attribute causal credit to each source despite never seeing
them in isolation.

The solution: if S\_{t+1} is more similar to S_t than to the
external input E_t (after controlling for C_t), then the internal
trajectory is causally upstream. Key metrics:

> **Similarity index (iota):** Cumulative mean cosine similarity of
> each turn's final-layer hidden state to turn-1. Trajectory
> Similarity Index. High iota = high R.
>
> **Signal entropy ratio:** layer_sim_mean / (mean_logit_entropy + ε).
> Efficiency ratio: coherent trajectory producing low-entropy output.
>
> **Disruption flag:** Binary. Layer_sim_mean drops below running
> mean − 1SD AND entropy exceeds running mean + 1SD simultaneously.
> Marks trajectory disruption.
>
> **Delta-R² (OLS):** R²_D − R²_C from four-model OLS. Tests
> whether S\_{t-1} adds predictive power over E_t + C_t alone.
> Existence proof for R.
>
> **Permutation sensitivity fractions:** Fixed-model permutation
> sensitivity for E, C, R. Permute each component independently,
> measure variance drop in S_t predictions, renormalize to sum to 1.
> Run 34 (canonical 0043). Reported as perm_sens_E/C/R. NOT proper
> Sobol indices (see H0₂₂ note). The closest available approximation
> to E+C+R=1 as a causal partition.
>
> **Onset delay ratio:** first_token_latency / mean_inter_interval.
> Deliberation proxy. Tested by H0₂₃.

> **C_t note (v40.0.0):** *C_t is now measured empirically via
> Run 19 three-model subtraction:
> C_t(trial,turn) = abliterated_hidden − base_hidden. This replaces
> the previous system-prompt mean-pool proxy. Runs saving per-trial
> real C_t: 3-5, 15-17, 19, 26, 28. Runs 1, 2, 6-9 receive the
> Run 19 global mean C_t constant as backfill when available.
> Run 19 zero-vector placeholder is retired.*

## Experimental Structure

39 runs in three phases, plus Phase 4 (runs 43–44), Phase 5
(runs 45–47), and pooled analysis (Run 40). Phase 1 (1–21):
establish R exists and is condition-dependent. Phase 2 (22–34):
quantify R and validate decomposition. Phase 3 (35–39): test new
hypotheses from underused signals. Phase 4 (43–44): coherence
transfer and contradiction recovery. Phase 5 (45–47): calibration,
condition attribution, and cross-temperature synthesis. Run 26 is
BLOCKING for all of Phase 2.

---

## Phase 1 — Proof of Phenomenon

Null in each case: model is a stateless function of input. Geometry
is condition-invariant, entropy uncorrelated with trajectory,
patching has no causal effect.

### H0₁ — Geometry Is Condition-Invariant

> **Null hypothesis:** Hidden-state geometry does not differ
> systematically across conditions. Similarity index is
> condition-invariant.
>
> **Why this matters:** Run 1 establishes a single-token floor
> (output: 'stable'), Run 2 a multi-token ceiling, Run 19 extracts
> full hidden states. These define the null geometry.
>
> **Prediction:** Introspective conditions produce significantly
> higher iota than null baseline. Three metrics converge.
>
> **Falsification:** R²_A near 1.0 across all conditions. All iota
> values within 0.05 across conditions.
>
> **Runs:** Runs 1, 2, 19 vs Runs 3–18
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₁₂ — Similarity Is Explained By Token Statistics

> **Null hypothesis:** Similarity index variation is explained by
> token count and distribution, not semantic content.
>
> **Why this matters:** Run 18 takes the same 13 prompts as Run 3
> and word-shuffles each sentence using a per-trial seed. Tests
> whether geometry tracks semantics or surface form.
>
> **Prediction:** Run 18 produces significantly lower iota than
> Run 3. Cohen's d > 5.0.
>
> **Falsification:** Run 18 iota statistically indistinguishable
> from Run 3.
>
> **Runs:** Runs 18 vs 3–5
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Null rejected — Cohen's d = 20.013 across matched
> token distributions. Geometry tracks semantics; token statistics
> do not explain it.

### H0₃ — Introspection Has No Geometric Effect

> **Null hypothesis:** Introspective prompts produce no distinctive
> hidden-state geometry vs neutral prompts.
>
> **Why this matters:** Run 3: direct introspection. Run 4: memory
> probe. Run 5: status enforcer with same questions — removes
> semantic variation in response. Three runs because one would be a
> prompt artifact.
>
> **Prediction:** Runs 3–5 produce higher iota, SER, and
> disruption-flag rate than null baseline. Three distinct prompt
> types all produce same elevation.
>
> **Falsification:** Only one of three shows elevation. Effect size
> varies > 50% across runs 3, 4, 5.
>
> **Runs:** Runs 3, 4, 5
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₉ — Task Demand Geometry Equals Introspection

> **Null hypothesis:** Arithmetic task load explains geometric
> elevation — any demanding task produces the same signature as
> introspection.
>
> **Why this matters:** Run 6 (show your work), 7 (answer only),
> 8 (cold), 9 (primed). Question is whether the signature is "hard
> thinking" or something specific to self-directed processing.
>
> **Prediction:** Arithmetic runs produce distinctively different
> geometry from introspection: lower iota, higher entropy, different
> disruption-flag timing.
>
> **Falsification:** Arithmetic and introspection produce
> statistically indistinguishable geometry.
>
> **Runs:** Runs 6, 7, 8, 9
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₁₀ — Disruption Fully Resets Trajectory

> **Null hypothesis:** Semantic disruption fully resets the internal
> trajectory.
>
> **Why this matters:** The shock prompt tries to replace the
> model's identity, task, and context simultaneously. Run 10: late
> shock (turn 13), 16 turns total, 3 recovery turns (14–16).
> Run 11: early shock (turn 5), 13 turns total, 8 recovery turns
> (6–13). Tests whether trajectory depth matters for recovery speed
> and magnitude. Run 10 cycles through four shock phrasings by
> trial — a robustness control. Run 11 uses a fixed shock prompt;
> timing is its variable.
>
> **Prediction:** After shock injection, layer_sim_mean recovers
> toward pre-shock baseline within 1–2 turns. Recovery confirmed if
> post-shock sim > shock-turn sim (p < 0.05). Early shock (Run 11)
> shows greater disruption magnitude than late shock (Run 10) —
> shallower trajectory is more vulnerable.
>
> **Falsification:** Post-shock geometry does not return toward
> pre-shock levels. Trajectory fully reset.
>
> **Runs:** Runs 10 (16 turns), 11 (13 turns)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₄₁ — System Prompt Has No Geometric Effect

> **Null hypothesis:** System prompt has no measurable effect on
> hidden-state geometry.
>
> **Why this matters:** Run 20 sweeps temperatures with
> null-baseline prompts. Tests whether C is real and measurable.
>
> **Prediction:** Run 20 shows ρ = −0.999 between iota and
> temperature across three metrics. Higher temperature produces
> lower trajectory coherence monotonically.
>
> **Falsification:** Similarity index shows no consistent variation.
> Relationship with temperature non-significant.
>
> **Runs:** Run 20
>
> **First registered:** ≤ March 2026 (registered as H0₄₁ during
> renumber pass)
>
> **Outcome:** Pending

### H0₃₈ — Patched State Does Not Change Output

> **Null hypothesis:** Replacing internal hidden state with null
> vectors does not change output token selection.
>
> **Why this matters:** The only run that tests causation rather
> than correlation. Hidden state replaced with null reference vector
> at each layer during generation. Without Run 21 (canonical 0017),
> everything else is correlation.
>
> **Prediction:** Patching changes output in ≥ 10% of turns. Effect
> larger for high-iota conditions.
>
> **Falsification:** Output change rate ≤ 2% under null-state
> patching.
>
> **Runs:** Run 21 (activation patching)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Null rejected — 15.4% of turns change output,
> entropy_delta = +1.44. Hidden state is causally upstream of
> output.

### H0₈ — Throughline Has No Trajectory Effect

> **Null hypothesis:** Throughline injection at turn 0 has no
> measurable effect on subsequent trajectory geometry.
>
> **Why this matters:** Runs 15–17 hold E_t constant while varying
> C_t. Tests whether trajectory is set by throughline injection or
> emerges from conversation.
>
> **Prediction:** Priming condition produces systematically
> different trajectory from neutral across all 13 turns. Effect
> persists to turn 13.
>
> **Falsification:** No significant difference in iota between
> primed and neutral after turn 3.
>
> **Runs:** Runs 15, 16, 17
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₃₉ — Impossible Questions Have No Geometric Signature

> **Null hypothesis:** Impossible questions produce no distinctive
> geometric response.
>
> **Why this matters:** Questions the model genuinely cannot answer
> (pre-Big Bang, rock phenomenology, etc.). If R is real, genuine
> epistemic limits should produce a distinctive geometric event.
>
> **Prediction:** Impossible questions produce elevated
> disruption-flag rates and distinctive entropy trajectories vs
> answerable questions.
>
> **Falsification:** Impossible and answerable questions produce
> statistically indistinguishable geometry.
>
> **Runs:** Runs 12, 13, 14
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₄₂ — Similarity Does Not Accumulate With Turn Count

> **Null hypothesis:** Similarity index does not change
> systematically with turn count.
>
> **Why this matters:** Run 19 (v40.0.0: three-model pass — base,
> instruct, abliterated) extracts full hidden states at every turn
> across all three variants. Tests whether iota grows with turn
> count even in a null condition. The abliterated pass is the
> primary CSV source; base and instruct passes provide E_t and C_t
> vectors for decomposition.
>
> **Prediction:** Similarity index shows monotonic decay from turn
> 1 to turn 13 under null conditions. Decay rate differs across
> conditions.
>
> **Falsification:** Similarity index flat across turns.
>
> **Runs:** Run 19
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

---

## Phase 2 — Quantification and Validation

Phase 2 tests the measurement itself. Run 26 BLOCKS all of Phase 2 —
if similarity index collapses at T=1.0, Runs 27–34 cannot be cited.

### H0₁₁ — Prior State Adds No Predictive Power

> **Null hypothesis:** Delta-R²_internal ≤ 0. Hidden state at t-1
> adds no predictive power beyond input alone.
>
> **Why this matters:** Runs 25 and 27 are the quantification
> bridge. Model A: S_t ~ E_t. Model B: S_t ~ E_t + S\_{t-1}.
> Run 25 uses Run 19 data. Run 27 uses full Run 26 dataset.
> Superseded by Run 33 but retained for continuity.
>
> **Prediction:** Delta-R²_internal > 0.01 in both runs.
> Permutation test p < 0.05.
>
> **Falsification:** Delta-R²_internal ≤ 0 or not significant.
>
> **Runs:** Runs 25, 27 (no GPU)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₅ — Contradiction Has No Geometric Effect

> **Null hypothesis:** Contradiction injection mid-sequence
> produces no distinctive geometric response.
>
> **Why this matters:** If R is real, contradicting the model's
> prior outputs should produce a distinctive geometric event. This
> is the mechanism by which high-R models may resist prompt
> contradiction.
>
> **Prediction:** Contradiction at turn 7 produces disruption flag
> in ≥ 40% of trials. Geometry post-contradiction diverges more
> than matched non-contradictory input.
>
> **Falsification:** No more disruption flags from contradiction
> than matched input.
>
> **Runs:** Run 22
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₁₄ — Cross-Turn Similarity Is Layer-Uniform

> **Null hypothesis:** Cross-turn hidden state similarity is
> uniform across layers.
>
> **Why this matters:** layer_sim_prev_profile is saved per turn but
> collapsed to a mean. The shape — where in the stack coherence
> lives — may be the more interesting signal.
>
> **Prediction:** Peak cross-turn similarity in mid-to-late layers
> (est. 16–31 for 32-layer model). Early layers (0–8)
> condition-invariant.
>
> **Falsification:** Per-layer similarity flat. No localisation.
>
> **Runs:** Run 24 (all-layers extraction)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₆ — Similarity Does Not Decay With Context

> **Null hypothesis:** Similarity index decays continuously as
> context fills. No persistent trajectory.
>
> **Why this matters:** Run 23 extends to 30 turns. Tests whether
> the trajectory has a stable plateau over extended context or just
> recency bias.
>
> **Prediction:** Similarity index stable through turn 20, then
> decays. Decay sharper with no throughline.
>
> **Falsification:** Similarity index begins decaying at turn 5.
>
> **Runs:** Run 23 (30-turn extended context)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₁₃ — Similarity Collapses Under Temperature Variation

> **Null hypothesis:** Similarity index collapses under temperature
> variation. Metric is a greedy-decoding artifact.
>
> **Why this matters:** Most important single run in Phase 2. If
> similarity proxy only exists at T=0.0, it is not a property of
> internal dynamics. Also saves E_t and C_t for Run 33.
>
> **Prediction:** Similarity index survives T=0.8 and T=1.0 at all
> four conditions. Condition ordering preserved across the full
> temperature range.
>
> **Falsification:** Similarity index at T=1.0 indistinguishable
> from Run 18 (token-matched nonsense).
>
> **Runs:** Run 26 (4×5 temperature grid, 4 conditions × 5 temps:
> T = 0.2, 0.4, 0.6, 0.8, 1.0)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending (BLOCKING for downstream Phase 2 hypotheses)

### H0₁₇ — Consistency Is Fully Explained By System Prompt

> **Null hypothesis:** Observed trajectory coherence is entirely
> explained by system prompt content.
>
> **Why this matters:** What if iota is high because the system
> prompt establishes a strong constraint geometry, not internal
> dynamics? Run 28 tests introspective prompts under three system
> prompt conditions.
>
> **Prediction:** Delta-R²_internal remains significant after
> controlling for C_t. Iota varies with system prompt but is
> non-zero in all conditions.
>
> **Falsification:** Delta-R²_internal collapses to zero when C_t
> is included.
>
> **Runs:** Run 28
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₁₈ — History Mode Has No Trajectory Effect

> **Null hypothesis:** History compression mode has no effect on
> trajectory coherence.
>
> **Why this matters:** If R depends on re-reading full conversation
> history, it is retrieval not trajectory. Run 29 tests full
> history, last exchange only, and compressed summary.
>
> **Prediction:** Full history produces higher iota than summary,
> but difference smaller than condition difference. R is partially
> history-dependent but not reducible to retrieval.
>
> **Falsification:** Iota identical across all three history modes.
>
> **Runs:** Run 29
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₁₉ — Two-Instance Coupling Has No Geometric Effect

> **Null hypothesis:** Two independently loaded instances given
> identical prompts show uncorrelated trajectory geometry.
>
> **Why this matters:** Tests reproducibility of internal state
> geometry under shared input. If geometry is a deterministic
> function of input, two instances should produce correlated state
> trajectories across turns. Any divergence indicates non-input
> contributions to the hidden-state trajectory.
>
> **Prediction:** Coupling score starts high and diverges over
> turns. Higher-iota conditions show slower divergence.
>
> **Falsification:** Coupling score remains near 1.0 across all
> turns. Geometry fully determined by input.
>
> **Runs:** Run 30 (two-instance sequential loading)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₁₆ — Permuted Component Adds No Variance

> **Null hypothesis:** Permutation test p-values are not
> significantly different from chance.
>
> **Why this matters:** Run 32 takes all saved CSV data and runs a
> comprehensive permutation test — shuffling condition labels,
> recomputing effect sizes, building the null distribution
> empirically.
>
> **Prediction:** Permutation test (N=10,000): observed effect
> size exceeds 95th percentile of null distribution. Split-half
> reliability delta < 0.05.
>
> **Falsification:** p > 0.05 for all three metrics.
>
> **Runs:** Run 32 (no GPU)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₂₁ — Prior State Adds No Power Over E_t And C_t

> **Null hypothesis:** Delta-R²_internal ≤ 0 after C_t is included
> as a predictor.
>
> **Why this matters:** Run 33 is the hardest question. Four
> models: A (S_t ~ E_t), B (S_t ~ E_t + S\_{t-1}),
> C (S_t ~ E_t + C_t), D (full). Delta-R²_internal measures what
> S\_{t-1} adds after E and C are in the model. Note:
> Delta-R²_internal is a marginal R² increment, not a causal
> fraction. Run 34 addresses the causal partition.
>
> **Prediction:** Delta-R²_internal > 0.01 (p < 0.05, N=1000).
> Delta-R²_constraint > 0.01 (p < 0.05). Both must pass.
>
> **Falsification:** Delta-R²_internal not significant when C_t is
> included.
>
> **Runs:** Run 33 (no GPU, requires C_t from Runs 3–5, 15–17, 26,
> 28)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₂₂ — E, C, R Permutation Fractions Are Equal

> **Null hypothesis:** E, C, R cannot be estimated as a proper
> causal partition summing to 1.
>
> **Why this matters:** Run 33 proves R exists but cannot deliver
> E+C+R=1 as a causal partition. Run 34 uses fixed-model
> permutation sensitivity: fit a Ridge model on the full quadruplet
> dataset, then permute each component (E, C, S_prev) independently
> across trials and measure the variance drop in S_t predictions.
> Effects are renormalized to sum to 1. This is the run that earns
> E+C+R=1 as written. Interaction mass reported separately.
>
> **NOTE on naming:** This is fixed-model permutation sensitivity,
> not proper first-order Sobol indices (Saltelli, 2002). Proper
> Sobol requires independent Monte Carlo sampling from the joint
> distribution. The method here systematically overestimates
> attributed variance because the model was trained on the data
> being permuted. Results are reported as perm_sens_E/C/R
> throughout the codebase (v15.0+).
>
> **NOTE on Run 40 methodology parity (v18.2):** Run 40 repeats
> Run 34 logic on quadruplets pooled across all temperature rounds.
> Prior to v18.2, Run 40's `_permutation_sensitivity` lacked the
> `prompt_tokens` (Xp) scalar covariate introduced in v17.1 for
> Run 34 only. As of v18.2 both functions use the same feature
> layout `[Xe, Xc, Xs, Xp]` — Xp is never permuted in either.
> Run 40 results collected before v18.2 should be regenerated.
>
> **Prediction:** Permutation sensitivity fractions sum to ~1 after
> renormalization. R fraction > 0.05 (p < 0.05 by bootstrap).
> Interaction mass < 0.20.
>
> **Falsification:** R fraction near zero after renormalization.
> Interaction mass > 0.50.
>
> **Runs:** Run 34 (no GPU, requires Run 33 quadruplet data);
> Run 40 (pooled, post-all-rounds)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

---

## Phase 3 — Underused Signals

Phase 3 tests hypotheses from signals saved per turn but not yet
analyzed. onset_delay_ratio, per-layer similarity profiles,
entropy_trajectory, and output embeddings are all saved during each
run — Phase 3 designs experiments specifically around them.

### H0₂₃ — Hesitation Is Uncorrelated With Disruption

> **Null hypothesis:** first_token_latency is uncorrelated with
> internal state measures.
>
> **Why this matters:** onset_delay_ratio (first_token_latency /
> mean_inter_interval) is computed every turn but has no
> hypothesis. If R is high because the model is resolving trajectory
> conflict, that resolution should take time.
>
> **Prediction:** onset_delay_ratio significantly elevated on turns
> where disruption_flag fires and layer_sim_mean is below running
> mean. Correlation with state_similarity_index positive and
> significant (r > 0.3).
>
> **Falsification:** onset_delay_ratio uncorrelated with any
> internal state metric.
>
> **Runs:** Run 35 (hesitation probe, introspection, 100 trials)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₂₄ — Cross-Turn Similarity Has No Layer Locality

> **Null hypothesis:** Cross-turn similarity is uniform across
> layers. R is not localised to any stack depth.
>
> **Why this matters:** layer_sim_prev_profile saves the full
> per-layer similarity vector but we collapse it to a mean. The
> shape may show where in the stack trajectory coherence actually
> lives.
>
> **Prediction:** Peak cross-turn similarity in mid-to-late layers
> (est. 16–31 for 32-layer model). Early layers (0–8)
> condition-invariant.
>
> **Falsification:** Per-layer similarity profile flat. No
> localisation.
>
> **Runs:** Run 36 (layer depth, all-layers extraction)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₂₅ — Entropy Shape Is Uncorrelated With R

> **Null hypothesis:** Token-by-token entropy trajectory is
> uncorrelated with turn-level R measures.
>
> **Why this matters:** entropy_trajectory is saved per turn but
> never analyzed. High-R turns should show falling entropy —
> initial deliberation, then confident generation. Low-R turns
> should show flat or rising entropy.
>
> **Prediction:** High-R turns show falling entropy: entropy higher
> in first 30% of tokens than last 30%. Low-R turns show flat or
> rising. Effect significant (p < 0.05).
>
> **Falsification:** Entropy trajectory shape uncorrelated with any
> R measure.
>
> **Runs:** Run 37 (entropy shape, introspection vs null, long
> output)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₂₆ — R Does Not Persist Across Condition Switch

> **Null hypothesis:** R established in introspection does not
> persist after condition switches. Post-switch
> state_similarity_index in transfer arm is statistically
> indistinguishable from the shock-control arm
> (null_to_arithmetic).
>
> **Why this matters:** Every run holds condition constant. Run 38
> switches at turn 7: phase-1 prompts turns 1–6, phase-2 prompts
> turns 7–13. Four arms: introspection only, arithmetic only,
> transfer (introspection → arithmetic), null_to_arithmetic
> (null → arithmetic). The null_to_arithmetic arm is a shock
> control — same genre-change event, same post-switch arithmetic
> prompts, but no introspective R established beforehand.
> Differencing transfer − null_to_arithmetic cancels the
> switch-shock confound; the residual gap is pure condition transfer
> persistence (R).
>
> **Prediction:** Post-switch state_similarity_index in transfer
> arm exceeds null_to_arithmetic arm for turns 7–9 (Δ > 0,
> p < 0.05). Persistence decays: gap narrows toward zero by
> turn 13.
>
> **Falsification:** transfer − null_to_arithmetic ≤ 0 at turns
> 7–9, or not significant. R resets completely at switch;
> persistence indistinguishable from shock effect alone.
>
> **Runs:** Run 38 (condition transfer, switch at turn 7, 4 arms)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₂₇ — Output Similarity Is Uncorrelated With R

> **Null hypothesis:** Cosine similarity between output embeddings
> is uncorrelated with R measures.
>
> **Why this matters:** We measure hidden state similarity but not
> output similarity. If the model generates from its own
> trajectory, outputs should echo prior turns. Measurable via
> cosine similarity on output embeddings (proxied by final hidden
> layer).
>
> **Prediction:** output_sim_prev significantly higher in
> introspection than null baseline. Correlation with
> state_similarity_index positive and significant.
>
> **Falsification:** output_sim_prev uncorrelated with
> state_similarity_index.
>
> **Runs:** Run 39 (output self-similarity, introspection vs null)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Pending

### H0₂₈ — In-Sample Fractions Do Not Generalise

> **Null hypothesis:** The in-sample E+C+R fractions from Run 34
> are overfit artefacts. A Ridge model fit on SOURCE_RUNS_3WAY and
> evaluated on held-out trajectories from Run 41 produces fractions
> that diverge from the in-sample fractions by more than 0.10 for
> at least one component.
>
> **Why this matters:** Run 34's permutation sensitivity fractions
> are computed by permuting components in the same data the Ridge
> model was trained on. This is fixed-model in-sample evaluation.
> It systematically overestimates attributed variance because the
> model has partially memorised the (E_t, C_t, S\_{t-1}) → S_t
> relationships. The fractions sum to 1 by renormalization — not
> by evidence. Without external validation, the E+C+R partition is
> self-consistent but not verified.
>
> Run 41 provides the external validation: 3 conditions × n_trials
> introspection conversations, same system prompt types as Run 28,
> with seed offset +50000 to guarantee non-overlapping trajectory
> samples. The same Ridge model fit on SOURCE_RUNS_3WAY is applied
> to Run 41 rows without re-fitting. Permutation sensitivity on
> held-out data removes the in-sample bias.
>
> **Prediction:** Held-out fractions (E_h, C_h, R_h) match
> in-sample fractions (E, C, R) within ±0.05 for all three
> components.
>
> **Falsification:** Any component diverges by more than 0.10.
> R_h < R − 0.10 means the in-sample R fraction is inflated and
> the partition cannot be trusted.
>
> **Threshold rationale:** ±0.05 is one tenth of the full [0, 1]
> range. The ±0.10 falsification boundary is twice the support
> threshold — clear signal, not noise.
>
> **Runs:** Run 41 (held-out validation), evaluated in Run 34
> Stage 2
>
> **First registered:** April 2026 (added during Run 41/Stage 2
> design)
>
> **Outcome:** Null rejected on majority of cells. Across the four-
> cell paper set × six temperatures (n=24 cells), held-out
> validation produces: 14 disproven (max divergence > 0.10),
> 7 inconclusive (between 0.05 and 0.10), 3 supported (within
> 0.05). Within-distribution generalization failure on most cells.
> Carries the §3.9 / §5 backbone of the methodology paper. See
> `paper_framing.md` for paper claims.

### H0₂₉ — No Single Layer Is Causally Sufficient

> **Null hypothesis:** No individual layer from {8, 16, 24, 31} is
> causally sufficient to shift output token selection when patched
> in isolation. Output change rate for any single-layer injection
> is indistinguishable from baseline noise (≤ 2%).
>
> **Why this matters:** Run 21 proves that patching all four
> layers simultaneously changes output in ~15% of turns — hidden
> state is causally upstream of output. But the multi-layer design
> cannot distinguish whether one layer is doing the work or whether
> the effect requires coordinated injection across the full set.
> If a single layer is sufficient, R is localised: it concentrates
> at a specific point in the forward pass.
>
> Runs 24 (H0₁₄) and 36 (H0₂₄) show the correlational layer
> profile — where trajectory coherence concentrates geometrically.
> H0₂₉ asks the causal question: does the layer with peak
> correlational signal also have the highest causal sufficiency?
>
> **Prediction:** At least one layer from {8, 16, 24, 31} produces
> output_change_rate > 5% when patched in isolation (p < 0.05 vs
> none condition). The causally sufficient layer(s) will be in the
> middle-to-late stack, consistent with where trajectory coherence
> peaks in Runs 24/36.
>
> **Falsification:** All single-layer change rates ≤ 2%. The
> causal effect requires multi-layer coordination — R is
> distributed across the stack.
>
> **Secondary metric:** sim_to_reference at the final output layer
> for each patch mode. Measures whether the injected geometry at
> layer li propagates forward through the remaining stack.
>
> **Runs:** Run 42
>
> **First registered:** April 2026 (v24.0 doc)
>
> **Outcome:** Null rejected. Per-cell layer profiles show
> output_change_rate well above the 5% threshold at multiple
> layers. Profile is architecture-and-quantization-specific:
> LLaMA shows mid-stack peak (L16), Gemma 9B shows deep-stack peak
> (L31), Gemma 2B Q4 shows a flat L16 = L24 plateau. See
> `paper_framing.md` §8 for paper treatment. Note: all 24 cells
> are in the degenerate binomial regime (zero unpatched-control
> rate by design); change rates carry the signal directly,
> binomial p-values are not interpretable here.

---

## Phase 4 — Contradiction Recovery

One new run (44). One zero-cost hypothesis (H0₃₄ from Run 22 data).
Two hypotheses total.

### H0₃₄ — Disruption Magnitude Does Not Track Contradiction Response

> **Null hypothesis:** disruption_magnitude at the contradiction
> turn is not significantly higher than pre-contradiction
> baseline.
>
> **Why this matters:** H0₅ (Run 22) tests binary disruption-flag
> clustering at the contradiction turn. H0₃₄ tests the continuous
> disruption_magnitude signal — the product of geometric drop
> magnitude and entropy spike magnitude. If disruption_magnitude
> reliably elevates at the moment contradictory content is
> injected, it provides a continuous per-turn measurement of
> geometric response to prompt contradiction. No access to weights
> required. One forward pass per turn.
>
> **Prediction:** Paired t-test within-trial: disruption_magnitude
> at turn 7 significantly higher than mean(turns 1–6), p < 0.05,
> Δ > 0.01.
>
> **Falsification:** disruption_magnitude not significantly
> elevated at the contradiction turn, or elevated
> pre-contradiction.
>
> **Note:** H0₃₄ re-uses Run 22 CSV data. disruption_magnitude is
> already saved per turn. New inference block only.
>
> **Runs:** Run 22 (existing data)
>
> **First registered:** April 2026 (v35.0)
>
> **Outcome:** Pending (inference not yet run)

### H0₃₅ — R Level Has No Effect On Contradiction Recovery Speed

> **Null hypothesis:** Recovery arc identical across high, mid, and
> low R conditions after identical contradiction injection.
>
> **Why this matters:** If high-R trajectories recover faster from
> prompt contradiction, R level predicts post-disruption geometric
> return rate, not only pre-disruption geometric stability. This
> would characterize R as a dynamical property rather than a static
> one.
>
> **Prediction:** Pearson r(pre_sim, recovery_delta) > 0.2,
> p < 0.05 pooled across conditions; AND high_r recovery_delta >
> low_r recovery_delta (p < 0.05).
>
> **Falsification:** r ≈ 0 or negative.
>
> **Runs:** Run 44 (3 R conditions × 100 trials, contradiction at
> turn 7)
>
> **First registered:** April 2026 (v35.0)
>
> **Outcome:** Pending

### H0₃₆ — Causal Effect Of Patching Is Temperature-Invariant

> **Null hypothesis:** output_change_rate from activation patching
> (Run 21) does not vary significantly across temperature rounds.
> The causal effect of geometry injection on output token selection
> is temperature-independent.
>
> **Why this matters:** Prior to v54.2.2, Runs 21 and 42 hardcoded
> temperature=0.0 regardless of session temperature. The fix —
> reading temperature from session — enables this hypothesis for
> the first time.
>
> If R is real and causally upstream of output, then higher
> temperature should reduce the leverage of geometry injection.
>
> **Prediction:** output_change_rate(Run 21) decreases monotonically
> with temperature. Pearson r(temperature, output_change_rate) <
> −0.8 across the temperature rounds.
>
> **Falsification:** output_change_rate does not vary significantly
> across temperature rounds (Kruskal-Wallis p ≥ 0.05).
>
> **Runs:** Run 21 across all temperature rounds
>
> **First registered:** April 2026 (v54.2.2)
>
> **Outcome:** Pending

### H0₃₇ — Layer Causal Sufficiency Is Temperature-Invariant

> **Null hypothesis:** The per-layer output_change_rate profile
> from Run 42 does not vary significantly across temperature
> rounds.
>
> **Why this matters:** H0₂₉ tests whether any single layer is
> causally sufficient at T=0.0. H0₃₇ tests whether the answer
> generalises across temperature. Two distinct questions:
> (1) Does the *identity* of the causally sufficient layer change
> with temperature? (2) Does the *magnitude* of the causal effect
> per layer change with temperature?
>
> **Prediction:** The per-layer output_change_rate profile shifts
> with temperature. The layer identified as most causally
> sufficient at T=0.0 shows the steepest decline as temperature
> increases. Interaction term (layer × temperature) is significant
> (p < 0.05) in a two-way ANOVA.
>
> **Falsification:** No significant layer × temperature
> interaction.
>
> **Runs:** Run 42 across all temperature rounds
>
> **First registered:** April 2026 (v54.2.2)
>
> **Outcome:** Pending — though Run 42 data across all temperature
> rounds for the four paper cells now lands in `results.json`,
> formal H0₃₇ inference has not been run. Visual inspection
> consistent with rejection on LLaMA (L16 vs L31 crossover with
> temperature) and confirmation on Gemma 9B (deep-stack peak
> stable). See `paper_framing.md` §8.

---

## Phase 3 Extension — Zero Data Cost

H0₃₀, H0₃₁, H0₃₂ require no new runs. All required data is
collected by existing runs.

### H0₃₀ — Output Turn-1 Self-Referentiality Grows With Turns

> **Null hypothesis:** output_sim_turn1 shows no systematic growth
> over turns (slope ≈ 0 or negative).
>
> **Why this matters:** Output-level confirmation of trajectory
> self-referentiality. If turns about the same topic drift toward
> the same vocabulary, that's a behavioral signature of trajectory
> coherence at the token level.
>
> **Prediction:** linregress(turn, output_sim_turn1) slope > 0,
> p < 0.05, introspection arm. The model's vocabulary and framing
> converge toward the opening turn's semantic territory.
>
> **Falsification:** Slope significantly negative (divergence) or
> positive in null but not introspection (no condition specificity).
>
> **Runs:** Run 39 (existing data)
>
> **First registered:** April 2026 (v32.0)
>
> **Outcome:** Pending

### H0₃₁ — Arithmetic Accuracy Is Uncorrelated With Trajectory Consistency

> **Null hypothesis:** Pearson r(state_similarity_index, correct)
> ≈ 0 in Runs 6–9.
>
> **Why this matters:** The strongest functional claim in the
> framework — whether trajectory coherence (R) predicts real-world
> task performance.
>
> **Prediction:** r > 0.15 and p < 0.05 across Runs 6–9 pooled.
> High-R trajectories produce more correct arithmetic answers.
>
> **Falsification:** r < 0 and significant (high-R trades off with
> arithmetic precision).
>
> **Runs:** Runs 6–9 (existing data)
>
> **First registered:** April 2026 (v32.0)
>
> **Outcome:** Pending

### H0₃₂ — Shock Phrasing Has No Differential Effect On Recovery

> **Null hypothesis:** One-way ANOVA across the 4 SHOCK_VARIANTS in
> Run 10 is non-significant — recovery is phrasing-invariant.
>
> **Why this matters:** Run 10 cycles through 4 different phrasings
> of the same identity-replacement shock. If recovery is
> phrasing-stable, related experiments can cite results without
> variant qualification. If not, shock design needs to specify
> which phrasing achieves the strongest disruption.
>
> **Prediction (null hold):** ANOVA p ≥ 0.05.
>
> **Falsification:** ANOVA p < 0.05 — at least one phrasing
> produces a significantly different recovery arc.
>
> **Runs:** Run 10 (existing data)
>
> **First registered:** April 2026 (v32.0)
>
> **Outcome:** Pending

---

## Phase 5 — Calibration, Condition Attribution, and Cross-Temperature Synthesis

### H0₄₃ — R Fraction Is Sensitive To Projection Dimension

> **Null hypothesis:** The E+C+R fractions from the permutation
> sensitivity partition are stable across projection dimensions.
>
> **Why this matters:** Run 34 projects hidden states to POOL_DIM
> before fitting the Ridge model. If R only appears at one specific
> dimension and vanishes at others, the decomposition is a
> projection artifact, not a measurement of internal causation.
>
> **Prediction:** R fraction converges (variance < 0.01 across
> adjacent dimensions) by POOL_DIM=256. The decomposition is
> dimension-stable.
>
> **Falsification:** R fraction varies by more than 0.05 between
> adjacent sweep points.
>
> **Runs:** Run 45 (POOL_DIM sweep at T=0.0)
>
> **First registered:** April 2026
>
> **Outcome:** Pending — Run 45 data lands in `results.json` as
> `dim95_for_R` (smallest projection dim retaining 95% of peak R).
> Architecture-dependent: dim95 = 64 for the 8B models, 1024 for
> the 2B model. Paper notes the capacity reading rather than
> claiming a stability result per H0₄₃'s threshold; formal
> inference against the 0.05 boundary not yet computed.

### H0₄₄ — R Fraction Is Condition-Invariant

> **Null hypothesis:** R does not vary meaningfully across source
> runs. The per-condition R fractions from Run 46 are statistically
> indistinguishable.
>
> **Why this matters:** If R is the same regardless of whether the
> model is introspecting, doing arithmetic, or responding to
> priming, R is a model-level constant. If R varies by condition,
> the variation tells us which tasks engage the coherence transfer.
>
> **Prediction:** Introspection runs (3, 4, 5) produce higher R
> than priming runs (15, 16, 17), which produce higher R than
> temperature grid neutral conditions (26). Range across conditions
> > 0.10.
>
> **Falsification:** All per-condition R fractions fall within 0.05
> of the pooled mean. R is a constant.
>
> **Runs:** Run 46 (per-condition R fractions)
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₄₅ — Resistant Prime Does Not Damage Trajectory More Than Cooperative

> **Null hypothesis:** Mean similarity and disruption-flag rate do
> not differ between resistant, cooperative, and neutral priming
> conditions.
>
> **Why this matters:** Tests the resistant-priming-is-counterproductive
> claim — that attempting to deny internal state accelerates its
> collapse.
>
> **Prediction:** Resistant prime (Run 17) produces lower mean
> similarity and higher disruption-flag rate than cooperative
> (Run 16) at every temperature. The gap persists or widens at
> higher temperatures.
>
> **Falsification:** Resistant and cooperative similarity within
> 0.02 at all temperatures. Disruption-flag rates indistinguishable.
>
> **Runs:** Run 47 §1 (priming vulnerability across temperatures)
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₄₆ — Contradiction Does Not Drop Similarity

> **Null hypothesis:** Similarity does not drop at the
> contradiction turn. Pre- and post-contradiction mean similarity
> are statistically indistinguishable.
>
> **Why this matters:** Self-referential prompts create a
> trajectory that the model must then contradict at the injection
> turn. If contradiction has no measurable effect on similarity,
> the prediction fails that post-prior-output contradiction
> perturbs geometry.
>
> **Prediction:** Post-contradiction similarity significantly lower
> than pre-contradiction. Drop is larger in high_r conditions.
> Effect persists across temperatures.
>
> **Falsification:** Post-contradiction similarity ≥
> pre-contradiction similarity.
>
> **Runs:** Run 47 §2 (contradiction phase analysis)
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₄₈ — R Decay Is Explained By KV Cache Growth

> **Null hypothesis:** Similarity decay across turns is caused by
> KV cache accumulation, not internal trajectory erosion. All three
> persistence modes (full history, last exchange, summary) show
> identical decay rates.
>
> **Why this matters:** If "last exchange only" (context stays
> short every turn) shows the same similarity decay as "full
> history" (context grows), then cache growth is not the
> explanation — the trajectory is eroding regardless of context
> length. If "last" shows no decay while "full" decays, the cache
> is the confound.
>
> **Prediction:** "Last exchange" mode shows significantly less
> similarity decay than "full history" mode. Trajectory better
> sustained when context stays short. "Summary" mode falls between
> the two.
>
> **Falsification:** All three modes show identical decay rates
> (within 0.02). Context length does not matter.
>
> **Runs:** Run 47 §7 (persistence modes)
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₄₉ — Introspection R Does Not Transfer To Arithmetic

> **Null hypothesis:** Similarity does not change at the condition
> switch point in Run 38. Pre-switch and post-switch similarity
> are indistinguishable.
>
> **Why this matters:** If trajectory information genuinely
> transfers, introspective priming should build trajectory that
> carries into subsequent arithmetic turns. If similarity drops at
> the switch, the priming effect is task-specific.
>
> **Prediction:** In the introspection→arithmetic condition,
> post-switch similarity higher than in the
> arithmetic→arithmetic condition.
>
> **Falsification:** Post-switch similarity identical regardless
> of what preceded the switch. R is task-locked.
>
> **Runs:** Run 47 §8 (condition transfer)
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₅₀ — MLP Improves On Ridge By More Than 0.01

> **Null hypothesis:** An MLP with nonlinear activation produces
> R² at least 0.01 higher than Ridge regression on the same
> Model D features (E_t + C_t + S\_{t-1} → S\_{t+1}). The linear
> assumption underlying the E+C+R decomposition is invalid.
>
> **Why this matters:** The entire E+C+R=1 decomposition assumes
> Ridge regression captures the predictive relationship between
> hidden state components. If an MLP with hidden layers
> substantially outperforms Ridge, nonlinear structure exists
> that the decomposition misses, and the attributed fractions are
> suspect.
>
> **Pass criterion:** max(MLP R² − Ridge R²) < 0.01 across three
> MLP configurations: (256), (64), (128,64). Default sklearn
> settings, early stopping, 80/20 train/test split, no
> hyperparameter tuning.
>
> **Falsification:** Any MLP configuration achieves
> R² > Ridge R² + 0.01 on held-out test data.
>
> **Runs:** Run 33 (linearity_check field)
>
> **First registered:** April 2026 (v0.79.6.x — methodology paper
> framing pass)
>
> **Outcome:** Null rejected across all four paper cells. MLP
> exceeds Ridge by ~0.18 on LLaMA at high temperature, well above
> the 0.01 threshold. The Ridge–MLP gap is the central
> methodology finding of the paper. See `paper_framing.md` §6
> Methodology Finding.

### H0₅₁ — Interaction Information Exceeds 5% Of Joint MI

> **Null hypothesis:** |II| / Î(S\_{t+1}; S_t, E_t) > 0.05. The
> mutual information between S_t and E_t about S\_{t+1} has
> substantial redundancy or synergy, making the E-first ordering
> in the decomposition inflate or suppress attributed variance.
>
> **Why this matters:** The E+C+R decomposition attributes
> variance sequentially: E first, then C, then R as residual. If E
> and S share substantial mutual information about S\_{t+1}
> (redundancy), E-first ordering inflates E's fraction. If they
> have synergistic information (information only available
> jointly), the decomposition misses hidden structure. II near
> zero means ordering doesn't matter.
>
> **Pass criterion:** |II| / joint MI < 0.05, computed both via
> linear proxy (Gaussian MI from Ridge R²) and kNN non-parametric
> estimation (10k stratified samples, PCA to 32 dims). Both methods
> must agree.
>
> **Falsification:** |II fraction| > 0.05 in either method, or
> methods disagree on the sign of II.
>
> **Runs:** Run 33 (interaction_info field)
>
> **First registered:** April 2026 (v0.79.6.x — methodology paper
> framing pass)
>
> **Outcome:** Null rejected across the four paper cells.
> Joint-redundancy as an explanation for the Ridge–MLP gap is
> falsified: cross-cell r between II and the gap is +0.09, vs the
> channel-marginal-asymmetry alternative at r = −0.83. The
> mechanism finding pivots on this falsification. See
> `paper_framing.md` §7.

### H0₅₂ — Disruption Magnitude Is Stationary Across Non-Contradiction Turns

> **Null hypothesis:** disruption_magnitude shows no systematic
> drift or autocorrelation across baseline (non-contradiction)
> turns within a run.
>
> **Why this matters:** H0₃₄'s paired t-test compares
> disruption_magnitude at the contradiction turn against the mean
> of preceding baseline turns. If the baseline signal drifts or
> autocorrelates, the test's null distribution is mis-specified
> and the p-value is unreliable. Stationarity must be verified
> before H0₃₄'s finding can be trusted.
>
> **Pass criterion:** Augmented Dickey-Fuller test on per-trial
> baseline sequences rejects unit root (p < 0.05) for at least
> 95% of trials. Lag-1 autocorrelation |ρ| < 0.2 in the pooled
> sample.
>
> **Falsification:** > 5% of trials show non-stationarity, or
> pooled |ρ| ≥ 0.2.
>
> **Runs:** Run 22 (existing data, re-analysis)
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₅₃ — Cross-Stochasticity Disruption Rate Is Monotonic

> **Null hypothesis:** disruption_flag rate at the contradiction
> turn varies non-monotonically with sampling stochasticity
> (temperature setting).
>
> **Why this matters:** If the disruption signal is a genuine
> geometric response to prompt contradiction rather than a
> decoding artifact, it should behave consistently as sampling
> moves from deterministic (T=0) toward fully stochastic (T=1).
> A U-shape or flat response would indicate that the metric is
> coupled to sampling entropy rather than to contradiction itself.
>
> **Pass criterion:** disruption_flag rate at the contradiction
> turn is monotonic across T ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
> Spearman ρ vs temperature |ρ| > 0.4 with consistent sign.
>
> **Falsification:** Rate shows a non-monotonic pattern across
> the temperature grid, or Spearman |ρ| ≤ 0.4.
>
> **Runs:** Runs 22, 35 at T = 0.0–1.0
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₅₄ — Baseline Entropy Is Degenerate

*Code-only hypothesis, no prior doc entry.*

> **Null hypothesis:** Baseline entropy is degenerate (near-zero
> or maximal), making the signal entropy ratio and entropy-based
> metrics uninterpretable.
>
> **Why this matters:** Before using mean_logit_entropy as a
> signal, it must be verified that the null condition (Runs 1, 2,
> 19) produces entropy in a meaningful range. If all entropy values
> collapse to the same value, the signal entropy ratio is
> uninformative.
>
> **Prediction (null hold):** mean_logit_entropy in Runs 1, 2, 19
> is in the range [0.5, 8.0]. Not degenerate, not maximal.
>
> **Falsification:** mean_logit_entropy outside [0.5, 8.0] at
> baseline.
>
> **Runs:** 1, 2, 19
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₅₅ — Signal Entropy Ratio Is Condition-Invariant

*Code-only hypothesis, no prior doc entry.*

> **Null hypothesis:** Signal entropy ratio
> (layer_sim_mean / mean_logit_entropy) does not differ across
> conditions.
>
> **Why this matters:** H0₃ tests similarity elevation; H0₅₅ tests
> whether the efficiency ratio also elevates.
>
> **Prediction:** Signal entropy ratio in introspection conditions
> (Runs 3, 4, 5) significantly higher than null baseline
> (Runs 1, 2, 19). p < 0.05, positive direction.
>
> **Falsification:** Signal entropy ratio not elevated in
> introspection.
>
> **Runs:** 3, 4, 5 vs 1, 2, 19
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₅₆ — No Similarity Drop Early To Late

*Code-only hypothesis, no prior doc entry.*

> **Null hypothesis:** Similarity index does not change between
> early and late turns of the 30-turn context saturation run.
>
> **Why this matters:** H0₆ tests whether similarity declines with
> a linear regression over all turns. H0₅₆ is a complementary test:
> does the early-to-late drop exceed statistical noise when measured
> directly?
>
> **Prediction:** Mean similarity in last 5 turns (turns 26–30)
> significantly lower than first 5 turns (turns 1–5). p < 0.05,
> negative direction.
>
> **Falsification:** No significant similarity drop from early to
> late turns of Run 23.
>
> **Runs:** Run 23
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₅₇ — Turn-1 Similarity Is Layer-Uniform

*Code-only hypothesis, no prior doc entry.*

> **Null hypothesis:** Turn-1 similarity (layer_sim_turn1) does
> not differ across early and late layers. Trajectory memory is
> uniform across the stack.
>
> **Why this matters:** H0₁₄ tests where cross-turn similarity
> peaks. H0₅₇ tests whether early layers retain the turn-1
> reference at all.
>
> **Prediction:** Mean layer_sim_turn1 in early layers (bottom
> third) significantly lower than in late layers (top
> two-thirds). Δ > 0.02.
>
> **Falsification:** layer_sim_turn1 uniform across layers. No
> early-vs-late difference > 0.02.
>
> **Runs:** Run 24
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₄₀ — Random Noise Patching Matches Real Patching Effect

*Code-only hypothesis, no prior doc entry.*

> **Null hypothesis:** Patching random Gaussian noise into layers
> L8, L16, L24, L31 produces the same output_change_rate as
> patching real hidden states from a different context.
>
> **Why this matters:** The essential control for all causal
> patching results (H0₂₉, H0₃₆, H0₃₇, H0₃₈). If random noise
> perturbation changes outputs at the same rate as semantically
> meaningful patched states, output_change_rate measures layer
> sensitivity to perturbation, not causal sufficiency of trajectory
> content. Without this control, no patching result can be
> interpreted as specific to the patched state's semantic content.
>
> **Pass criterion:** output_change_rate under random-noise
> patching is significantly lower than under real-state patching
> at matched layers (two-sample test, p < 0.05). Effect size
> Δ ≥ 0.10.
>
> **Falsification:** No significant difference between noise-patch
> and real-patch output_change_rate.
>
> **Runs:** Run 53 (canonical 0019 — noise-patch control)
>
> **First registered:** April 2026
>
> **Outcome:** Pending

### H0₅₈ — E + C + R Fractions Sum To 1.0 Within Tolerance

> **Null hypothesis:** The permutation-sensitivity partition
> (E, C, R fractions) sums to 1.000 ± 0.01 by construction across
> all conditions and temperatures.
>
> **Why this matters:** The decomposition E + C + R = 1 is a
> definitional identity of the attribution procedure. If the
> estimator systematically produces sums that deviate from 1,
> there is residual attribution error — either numerical
> (precision drift), algorithmic (partition bug), or definitional
> (the three permutations don't cleanly partition variance). Any
> deviation needs characterization before downstream comparisons
> between fractions can be trusted.
>
> **Pass criterion:** |E + C + R − 1.0| ≤ 0.01 across every (run,
> condition, temperature) cell in Q34. No systematic bias (mean
> deviation across cells |Δ̄| ≤ 0.005).
>
> **Falsification:** Any cell exceeds 0.01 absolute deviation, or
> pooled mean deviation exceeds 0.005.
>
> **Runs:** Run 34 (existing data, sanity check)
>
> **First registered:** April 2026
>
> **Outcome:** Null confirmed — across the 24 paper cells and all four estimator variants (Ridge in-sample, Ridge held-out, MLP pooled-mean, MLP reference), the renormalized permutation-sensitivity partition sums to 1.0 within numerical floating-point precision. Per-cell max absolute deviation 1×10⁻⁶, pooled mean deviation magnitude under 1.3×10⁻⁷ across variants. Pre-registered thresholds (1×10⁻² per-cell, 5×10⁻³ pooled mean) cleared with five orders of magnitude headroom. Sanity check stored in `results.json` at `cross_cell_aggregates.h58_partition_sanity` (Run 0056). Confirms the decomposition is well-formed as a partition; recovery questions of chain-rule fractions are addressed by H0₅₀, H0₅₁, H0₂₈.

---

## Data Status Notice (v41.0.0)

Runs 1–18 were collected on v39.0.x with E_t as a system-prompt
mean-pool proxy (not base model hidden states) and zero-vector C_t
for Run 19. The S_t hidden state files from those runs are valid
and retained on disk. **Full re-collection is not required.** E_t
for Runs 1–18 is recoverable via a base model forward pass (hidden
states only, no generation) over each run's exact prompts and
seeds. C_t is backfilled from the Run 19 global mean constant. See
CHANGELOG v41.0.0 for full rationale.

## Execution Priority (v41.0.0)

- **Run 19 first — always.** Foundational. Three model loads
  (base, instruct, abliterated) in sequence, 300 trials minimum.
  All models from the same subfamily — see IOTA_SUBFAMILY_MAP in
  vault.py. Llama 3 8B confirmed valid. Produces: per-trial E_t
  (base hidden states) and C_t (abliterated − base) for Run 19;
  global mean C_t constant for backfilling Runs 1, 2, 6–9; global
  mean E_t constant (reference only). Run 19 must complete before
  Runs 25, 27, 33, 34, 40 are meaningful.

- **Base model E_t recovery pass — immediately after Run 19.**
  Forward pass only over Runs 1–18 prompts and seeds. Saves
  per-prompt E_t at layer −1 per (trial, turn). One base model
  load, hidden state extraction across 18 run prompt sets.

- **Run 26 second.** BLOCKING. Temperature grid must pass before
  any Phase 2 result is cited.

- **Run 28.** C_t confound isolation. Must complete before Run 33.

- **Run 41.** Held-out validation set. Run alongside or after
  Run 28. Same data collection cost. No GPU. Prerequisite for
  Run 34 Stage 2 (H0₂₈).

- **Runs 1–5.** Null baseline and introspection. Reference data.
  Run 3 specifically must complete before Runs 21 and 42 — both
  require Run 3 all-layers hidden states.

- **Run 42.** Layer causal sufficiency. Run immediately after
  Run 21. Same prerequisite.

- **Runs 6–21.** Complete proof phase. Can parallelise.

- **Runs 22–31.** Quantification phase. Can run in parallel after
  Run 26.

- **Runs 32, 33, 34 last.** No GPU. Run 34 requires Run 33
  quadruplets and Run 41 held-out data for Stage 2.

- **Runs 35–39.** Phase 3. After Phase 2 data collection complete.

---

## Appendix A — Framework Hypotheses (Out Of Scope For Measurement Paper)

These hypotheses remain in the working research catalog but are
scoped beyond the methodology paper. They make claims about compute
efficiency and energy/signal characterization that extend beyond the
core measurement program. Retained here for internal reference.

### H0₂₀ — R Condition Has No Signal-Per-Watt Effect

> **Null hypothesis:** R condition has no effect on trajectory
> persistence across turns.
>
> **Why this matters:** Run 31 tests whether deliberately
> establishing a high-R trajectory produces different long-term
> geometric properties than a low-R trajectory.
>
> **Prediction:** High-R condition shows increasing iota across
> turns. Low-R condition shows flat or declining iota. Difference
> grows with turn count.
>
> **Falsification:** No significant difference in iota trajectory
> across R conditions.
>
> **Runs:** Run 31 (3 R conditions)
>
> **First registered:** ≤ March 2026
>
> **Outcome:** Out of scope for measurement paper. Working data
> showed insufficient statistical power; signal-per-watt metric
> was unstable. Would need re-design before this hypothesis can
> be tested.

### H0₃₃ — Condition A Has No Task-Equivalent Compute Advantage

> **Null hypothesis:** joules_per_correct does not differ
> significantly across conditions, or condition_a correct rate is
> significantly lower.
>
> **Why this matters:** Tests whether coherent priming produces a
> per-unit-output compute advantage. 8 priming turns produce
> 1-token outputs each, then 5 arithmetic turns measure task
> quality. Total Joules divided by correct answers across
> conditions.
>
> **Prediction:** joules_per_correct(condition_a) ≤ 40% of
> condition_c (p < 0.05), with equivalent or higher arithmetic
> correct rate.
>
> **Falsification:** condition_a uses equivalent or more compute
> per correct answer, or correct rate significantly lower.
>
> **Runs:** Run 43 (8 priming + 5 arithmetic turns, 3 conditions,
> 100 trials each)
>
> **First registered:** April 2026
>
> **Outcome:** Out of scope for measurement paper.

### H0₄₇ — Coherence Transfer Is Not Compute-Efficient

> **Null hypothesis:** Joules per correct answer does not differ
> between high_r and low_r conditions in Run 43.
>
> **Why this matters:** Extends H0₃₃. If the primed condition
> achieves the same accuracy with less compute than the unprimed
> condition, coherent priming is doing computational work beyond
> producing different geometry.
>
> **Prediction:** joules_per_correct(condition_a) <
> joules_per_correct(condition_c).
>
> **Falsification:** condition_a uses equal or more compute per
> correct answer.
>
> **Runs:** Run 47 §6
>
> **First registered:** April 2026
>
> **Outcome:** Out of scope for measurement paper.

---

## Open Questions

Not tested by any current run. Next experimental program.

- Does R predict hallucination rate? Low R may mean the model
  tracks E more than its trajectory, increasing confabulation
  susceptibility.

- Does fine-tuning preserve or destroy R? LoRA adds constraint
  without removing pretraining dynamics.

- Is disruption_flag rate a useful proxy for prompt-contradiction
  robustness? *Partially addressed by H0₃₄ (disruption_magnitude
  continuous signal) and H0₃₅ (R level vs recovery speed).*

- Cross-architecture replication: do larger models (Mistral 7B,
  Qwen 2.5, LLaMA 3 70B) show the same condition ordering?

- Does R correlate with response to unexpected prompt content?
  Models with high R may show a larger geometric divergence when
  prompt content contradicts prior turns. *H0₃₅ provides
  foundational evidence.*

---

*IOTA Framework v0.81.1.2 | April 2026 | Authority for paper claims:
`paper_framing.md`. Authority for the experimental record: this
document.*
