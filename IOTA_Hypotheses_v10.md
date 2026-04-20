**IOTA FRAMEWORK**

Hypotheses and Experimental Design

*Version 24.0 \| IOTA Framework v0.77.0.0 \| April 2026 \| 53 Runs \| 51 Hypotheses*

**The Central Claim**

Every response a language model generates is determined by three
sources: what arrived in the prompt (E, external input), what the model
is constrained to produce (C, constraint pressure from training and
system instructions), and something from the model\'s own prior
trajectory (R, internal causation). The IOTA decomposition formalises
this as E + C + R = 1, where each term represents a fraction of causal
determination over the next hidden state.

R is what we are measuring. It is the fraction of next-state
determination attributable to the model\'s internal hidden-state
trajectory beyond what can be explained by current input and
constraints. If R is zero, the model is a pure lookup function. If R is
large, prior turns are causally upstream of what it produces.

> **WHY THIS MATTERS** *Interpretability research typically asks \'which
> circuit does what\'. We ask a prior question: to what degree does the
> model\'s own trajectory cause its outputs at all? This is a question
> about causal fraction, not mechanism. It can be answered without
> access to weights, and the answer has direct implications for
> robustness, interpretability, and what it means for a model to be
> coherent across a conversation.*

**The Measurement Problem**

The core difficulty: you cannot observe R directly. E, C, and R are all
operating simultaneously on every turn. The measurement problem is to
attribute causal credit to each source despite never seeing them in
isolation.

Our solution: if S\_{t+1} is more similar to S_t than to the external
input E_t (after controlling for C_t), then the internal trajectory is
causally upstream. Key metrics:

> **Similarity index (iota):** Cumulative mean cosine similarity of each
> turn\'s final-layer hidden state to turn-1. Trajectory Similarity Index. High iota = high R.
>
> **signal entropy ratio:** layer_sim_mean / (mean_logit_entropy + e). Efficiency
> ratio: coherent trajectory producing low-entropy output.
>
> **disruption_flag:** Binary: layer_sim_mean drops below running mean - 1SD
> AND entropy exceeds running mean + 1SD simultaneously. Marks
> trajectory disruption.
>
> **Delta-R2 (OLS):** R2_D - R2_C from four-model OLS. Tests whether
> S\_{t-1} adds predictive power over E_t + C_t alone. Existence proof
> for R.
>
> **Permutation Sensitivity fractions:** Fixed-model permutation
> sensitivity for E, C, R. Permute each component independently, measure
> variance drop in S_t predictions, renormalize to sum to 1. Run 34.
> Reported as perm_sens_E/C/R. NOT proper Sobol indices (see H22 note).
> The closest available approximation to E+C+R=1 as a causal partition.
>
> **onset_delay_ratio:** first_token_latency / mean_inter_interval.
> Deliberation proxy. Tested by H23.
>
> **C_t NOTE (v40.0.0):** *C_t is now measured empirically via Run 19
> three-model subtraction: C_t(trial,turn) = abliterated_hidden −
> base_hidden. This replaces the previous system-prompt mean-pool proxy.
> Runs saving per-trial real C_t: 3-5, 15-17, 19, 26, 28.
> Runs 1, 2, 6-9 receive the Run 19 global mean C_t constant as backfill
> when available. Run 19 zero-vector placeholder is retired.*

**Experimental Structure**

39 runs in three phases, plus Phase 4 (runs 43-44), Phase 5 (runs 45-47), and
pooled analysis (Run 40). Phase 1 (1-21): establish R exists and is
condition-dependent. Phase 2 (22-34): quantify R and validate
decomposition. Phase 3 (35-39): test new hypotheses from underused
signals. Phase 4 (43-44): coherence transfer and contradiction recovery.
Phase 5 (45-47): calibration, condition attribution, and cross-temperature
synthesis. Run 26 is BLOCKING for all of Phase 2.

**Phase 1 Hypotheses - Proof of Phenomenon**

Null in each case: model is a stateless function of input. Geometry is
condition-invariant, entropy uncorrelated with trajectory, patching has
no causal effect.

**H01 Geometry Is Condition-Invariant**

> **Null hypothesis:** Hidden-state geometry does not differ
> systematically across conditions. similarity index is condition-invariant.
>
> **Why this matters:** Run 1 establishes a single-token floor (output:
> \'stable\'), Run 2 a multi-token ceiling, Run 19 extracts full hidden
> states. These define the null geometry.
>
> **Prediction:** Introspective conditions produce significantly higher
> iota than null baseline. Three metrics converge.
>
> **Falsification:** R2_A near 1.0 across all conditions. All iota
> values within 0.05 across conditions.
>
> Runs: Runs 1, 2, 19 vs Runs 3-18 Status: **PENDING**

**H12 similarity Is Explained By Token Statistics**

> **Null hypothesis:** similarity index variation is explained by token count
> and distribution, not semantic content.
>
> **Why this matters:** Run 18 takes the same 13 prompts as Run 3 and
> word-shuffles each sentence using a per-trial seed. Finding (d=20.013)
> confirms geometry tracks semantics.
>
> **Prediction:** Run 18 produces significantly lower iota than Run 3.
> Cohen\'s d \> 5.0.
>
> **Falsification:** Run 18 iota statistically indistinguishable from
> Run 3.
>
> Runs: Runs 18 vs 3-5 Status: **PENDING**

**H03 Introspection Has No Geometric Effect**

> **Null hypothesis:** Introspective prompts produce no distinctive
> hidden-state geometry vs neutral prompts.
>
> **Why this matters:** Run 3: direct introspection. Run 4: memory
> probe. Run 5: status enforcer with same questions - removes semantic
> variation in response. Three runs because one would be a prompt
> artifact.
>
> **Prediction:** Runs 3-5 produce higher iota, SER, and disruption_flag rate
> than null baseline. Three distinct prompt types all produce same
> elevation.
>
> **Falsification:** Only one of three shows elevation. Effect size
> varies \>50% across runs 3, 4, 5.
>
> Runs: Runs 3, 4, 5 Status: **PENDING**

**H09 Task Demand Geometry Equals Introspection**

> **Null hypothesis:** Arithmetic task load explains geometric
> elevation - any demanding task produces the same signature as
> introspection.
>
> **Why this matters:** Run 6 (show your work), 7 (answer only), 8
> (cold), 9 (primed). Question is whether the signature is \'hard
> thinking\' or something specific to self-directed processing.
>
> **Prediction:** Arithmetic runs produce distinctively different
> geometry from introspection: lower iota, higher entropy, different
> disruption_flag timing.
>
> **Falsification:** Arithmetic and introspection produce statistically
> indistinguishable geometry.
>
> Runs: Runs 6, 7, 8, 9 Status: **PENDING**

**H10 Disruption Fully Resets Trajectory**

> **Null hypothesis:** Semantic disruption fully resets the internal
> trajectory.
>
> **Why this matters:** The shock prompt tries to replace the model\'s
> identity, task, and context simultaneously. Run 10: late shock (turn
> 13), 16 turns total, 3 recovery turns (14-16). Run 11: early shock
> (turn 5), 13 turns total, 8 recovery turns (6-13). Tests whether
> trajectory depth matters for recovery speed and magnitude.
>
> Run 10 cycles through four shock phrasings by trial (`trial % 4`) — a
> robustness control testing whether recovery is phrasing-invariant. Run
> 11 uses a fixed shock prompt; timing is its variable.
>
> **Prediction:** After shock injection, layer\_sim\_mean recovers toward
> pre-shock baseline within 1-2 turns (is\_recovery turns in Run 10).
> Recovery confirmed if post-shock sim > shock-turn sim (p \< 0.05).
> Early shock (Run 11, turn 5) shows greater disruption magnitude than
> late shock (Run 10, turn 13) — shallower trajectory is more vulnerable.
>
> **Falsification:** Post-shock geometry does not return toward pre-shock
> levels. layer\_sim\_mean in recovery turns statistically
> indistinguishable from shock-turn sim. Trajectory fully reset.
>
> Runs: Runs 10 (16 turns), 11 (13 turns) Status: **PENDING**

**H41 System Prompt Has No Geometric Effect**

> **Null hypothesis:** System prompt has no measurable effect on
> hidden-state geometry.
>
> **Why this matters:** Run 20 sweeps temperatures with null-baseline
> prompts. The rho = -0.999 result (three metrics, same monotonic
> ordering) confirms C is real and measurable.
>
> **Prediction:** Run 20 shows rho = -0.999 between iota and temperature
> across three metrics. Higher temperature produces lower trajectory
> coherence monotonically.
>
> **Falsification:** similarity index shows no consistent variation.
> Relationship with temperature non-significant.
>
> Runs: Run 20 Status: **PENDING**

**H38 Patched State Does Not Change Output**

> **Null hypothesis:** Replacing internal hidden state with null vectors
> does not change output token selection.
>
> **Why this matters:** The only run that tests causation rather than
> correlation. Hidden state replaced with null reference vector at each
> layer during generation. Finding: 15.4% of turns change output,
> entropy_delta = +1.44. Without Run 21, everything else is correlation.
>
> **Prediction:** Patching changes output in \>=10% of turns. Effect
> larger for high-iota conditions.
>
> **Falsification:** Output change rate \<= 2% under null-state
> patching.
>
> Runs: Run 21 (activation patching) Status: **PENDING**

**H08 Throughline Has No Trajectory Effect**

> **Null hypothesis:** Throughline injection at turn 0 has no measurable
> effect on subsequent trajectory geometry.
>
> **Why this matters:** Runs 15-17 hold E_t constant while varying C_t.
> Tests whether trajectory is set by throughline injection or emerges
> from conversation.
>
> **Prediction:** Priming condition produces systematically different
> trajectory from neutral across all 13 turns. Effect persists to turn
> 13.
>
> **Falsification:** No significant difference in iota between primed
> and neutral after turn 3.
>
> Runs: Runs 15, 16, 17 Status: **PENDING**

**H39 Impossible Questions Have No Geometric Signature**

> **Null hypothesis:** Impossible questions produce no distinctive
> geometric response.
>
> **Why this matters:** Questions the model genuinely cannot answer
> (pre-Big Bang, rock phenomenology, etc.). If R is real, genuine
> epistemic limits should produce a distinctive geometric event.
>
> **Prediction:** Impossible questions produce elevated disruption_flag rates
> and distinctive entropy trajectories vs answerable questions.
>
> **Falsification:** Impossible and answerable questions produce
> statistically indistinguishable geometry.
>
> Runs: Runs 12, 13, 14 Status: **PENDING**

**H42 similarity Does Not Accumulate With Turn Count**

> **Null hypothesis:** similarity index does not change systematically with
> turn count.
>
> **Why this matters:** Run 19 (v40.0.0: three-model pass — base,
> instruct, abliterated) extracts full hidden states at every turn across
> all three variants. Tests whether iota grows with turn count even in a
> null condition. The abliterated pass is the primary CSV source; base and
> instruct passes provide E_t and C_t vectors for decomposition.
>
> **Prediction:** similarity index shows monotonic decay from turn 1 to turn 13
> under null conditions. Decay rate differs across conditions.
>
> **Falsification:** similarity index flat across turns.
>
> Runs: Run 19 Status: **PENDING**

**Phase 2 Hypotheses - Quantification and Validation**

Phase 2 tests the measurement itself. Run 26 BLOCKS all of Phase 2 - if
similarity index collapses at T=1.0, Runs 27-34 cannot be cited.

**H11 Prior State Adds No Predictive Power**

> **Null hypothesis:** Delta-R2_internal \<= 0. Hidden state at t-1 adds
> no predictive power beyond input alone.
>
> **Why this matters:** Runs 25 and 27 are the quantification bridge.
> Model A: S_t \~ E_t. Model B: S_t \~ E_t + S\_{t-1}. Run 25 uses Run
> 19 data. Run 27 uses full Run 26 dataset. Superseded by Run 33 but
> retained for continuity.
>
> **Prediction:** Delta-R2_internal \> 0.01 in both runs. Permutation
> test p \< 0.05.
>
> **Falsification:** Delta-R2_internal \<= 0 or not significant.
>
> Runs: Runs 25, 27 (no GPU) Status: **PENDING**

**H05 Contradiction Has No Geometric Effect**

> **Null hypothesis:** Contradiction injection mid-sequence produces no
> distinctive geometric response.
>
> **Why this matters:** If R is real, contradicting the model\'s prior
> outputs should produce a distinctive geometric event. This is the
> mechanism by which high-R models may resist prompt contradiction.
>
> **Prediction:** Contradiction at turn 7 produces disruption_flag in \>=40% of
> trials. Geometry post-contradiction diverges more than matched
> non-contradictory input.
>
> **Falsification:** No more disruption_flags from contradiction than matched
> input.
>
> Runs: Run 22 Status: **PENDING**

**H14 Cross-Turn Similarity Is Layer-Uniform**

> **Null hypothesis:** Cross-turn hidden state similarity is uniform
> across layers.
>
> **Why this matters:** layer_sim_prev_profile is saved per turn but
> collapsed to a mean. The shape - where in the stack coherence lives -
> may be the more interesting signal.
>
> **Prediction:** Peak cross-turn similarity in mid-to-late layers (est.
> 16-31 for 32-layer model). Early layers (0-8) condition-invariant.
>
> **Falsification:** Per-layer similarity flat. No localisation.
>
> Runs: Run 24 (all-layers extraction) Status: **PENDING**

**H06 similarity Does Not Decay With Context**

> **Null hypothesis:** similarity index decays continuously as context fills.
> No persistent trajectory.
>
> **Why this matters:** Run 23 extends to 30 turns. Tests whether the
> trajectory has a stable plateau over extended context or just
> recency bias.
>
> **Prediction:** similarity index stable through turn 20, then decays. Decay
> sharper with no throughline.
>
> **Falsification:** similarity index begins decaying at turn 5.
>
> Runs: Run 23 (30-turn extended context) Status: **PENDING**

**H13 similarity Collapses Under Temperature Variation**

> **Null hypothesis:** similarity index collapses under temperature variation.
> Metric is a greedy-decoding artifact.
>
> **Why this matters:** Most important single run in Phase 2. If similarity
> proxy only exists at T=0.0, it is not a property of internal dynamics.
> Also saves E_t and C_t for Run 33.
>
> **Prediction:** similarity index survives T=0.8 and T=1.0 at all four
> conditions. Condition ordering preserved across the full temperature
> range.
>
> **Falsification:** similarity index at T=1.0 indistinguishable from Run 18
> (token-matched nonsense).
>
> Runs: Run 26 (4×5 temperature grid, 4 conditions × 5 temps:
> T=0.2, 0.4, 0.6, 0.8, 1.0) Status: **BLOCKING**

**H17 Consistency Is Fully Explained By System Prompt**

> **Null hypothesis:** Observed trajectory coherence is entirely
> explained by system prompt content.
>
> **Why this matters:** What if iota is high because the system prompt
> establishes a strong constraint geometry, not internal dynamics? Run 28 tests
> introspective prompts under three system prompt conditions.
>
> **Prediction:** Delta-R2_internal remains significant after
> controlling for C_t. iota varies with system prompt but is non-zero in
> all conditions.
>
> **Falsification:** Delta-R2_internal collapses to zero when C_t is
> included.
>
> Runs: Run 28 Status: **PENDING**

**H18 History Mode Has No Trajectory Effect**

> **Null hypothesis:** History compression mode has no effect on
> trajectory coherence.
>
> **Why this matters:** If R depends on re-reading full conversation
> history, it is retrieval not trajectory. Run 29 tests full history,
> last exchange only, and compressed summary.
>
> **Prediction:** Full history produces higher iota than summary, but
> difference smaller than condition difference. R is partially
> history-dependent but not reducible to retrieval.
>
> **Falsification:** iota identical across all three history modes.
>
> Runs: Run 29 Status: **PENDING**

**H19 Two-Instance Coupling Has No Geometric Effect**

> **Null hypothesis:** Two independently loaded instances given
> identical prompts show uncorrelated trajectory geometry.
>
> **Why this matters:** Tests reproducibility of internal state
> geometry under shared input. If geometry is a deterministic
> function of input, two instances should produce correlated state
> trajectories across turns. Any divergence indicates non-input
> contributions to the hidden-state trajectory.
>
> **Prediction:** Coupling score starts high and diverges over turns.
> Higher-iota conditions show slower divergence.
>
> **Falsification:** Coupling score remains near 1.0 across all turns.
> Geometry fully determined by input.
>
> Runs: Run 30 (two-instance sequential loading) Status: **PENDING**

**H16 Permuted Component Adds No Variance**

> **Null hypothesis:** Permutation test p-values are not significantly
> different from chance.
>
> **Why this matters:** Run 32 takes all saved CSV data and runs a
> comprehensive permutation test - shuffling condition labels,
> recomputing effect sizes, building the null distribution empirically.
>
> **Prediction:** Permutation test (N=10,000): observed effect size
> exceeds 95th percentile of null distribution. Split-half reliability
> delta \< 0.05.
>
> **Falsification:** p-value \> 0.05 for all three metrics.
>
> Runs: Run 32 (no GPU) Status: **PENDING**

**H21 Prior State Adds No Power Over E_t And C_t**

> **Null hypothesis:** Delta-R2_internal \<= 0 after C_t is included as
> a predictor.
>
> **Why this matters:** Run 33 is the hardest question. Four models: A
> (S_t \~ E_t), B (S_t \~ E_t + S\_{t-1}), C (S_t \~ E_t + C_t), D
> (full). Delta-R2_internal measures what S\_{t-1} adds after E and C
> are in the model. Note: Delta-R2_internal is a marginal R2 increment,
> not a causal fraction. Run 34 addresses the causal partition.
>
> **Prediction:** Delta-R2_internal \> 0.01 (p \< 0.05, N=1000).
> Delta-R2_constraint \> 0.01 (p \< 0.05). Both must pass.
>
> **Falsification:** Delta-R2_internal not significant when C_t is
> included.
>
> Runs: Run 33 (no GPU, requires C_t from Runs 3-5, 15-17, 26, 28)
> Status: **PENDING**

**H22 E, C, R Permutation Fractions Are Equal**

> **Null hypothesis:** E, C, R cannot be estimated as a proper causal
> partition summing to 1.
>
> **Why this matters:** Run 33 proves R exists but cannot deliver
> E+C+R=1 as a causal partition. Run 34 uses fixed-model permutation
> sensitivity: fit a Ridge model on the full quadruplet dataset, then
> permute each component (E, C, S_prev) independently across trials and
> measure the variance drop in S_t predictions. Effects are renormalized
> to sum to 1. This is the run that earns E+C+R=1 as written. Interaction
> mass reported separately.
>
> **NOTE on naming:** This is fixed-model permutation sensitivity, not
> proper first-order Sobol indices (Saltelli, 2002). Proper Sobol requires
> independent Monte Carlo sampling from the joint distribution. The method
> here systematically overestimates attributed variance because the model
> was trained on the data being permuted. Results are reported as
> perm_sens_E/C/R throughout the codebase (v15.0+).
>
> **NOTE on Run 40 methodology parity (v18.2):** Run 40 repeats Run 34
> logic on quadruplets pooled across all temperature rounds. Prior to
> v18.2, Run 40's `_permutation_sensitivity` lacked the `prompt_tokens`
> (Xp) scalar covariate introduced in v17.1 for Run 34 only. As of v18.2
> both functions use the same feature layout `[Xe, Xc, Xs, Xp]` — Xp is
> never permuted in either. Run 40 results collected before v18.2 should
> be regenerated.
>
> **Prediction:** Permutation sensitivity fractions sum to ~1 after
> renormalization. R fraction > 0.05 (p < 0.05 by bootstrap). Interaction
> mass < 0.20.
>
> **Falsification:** R fraction near zero after renormalization.
> Interaction mass > 0.50.
>
> Runs: Run 34 (no GPU, requires Run 33 quadruplet data); Run 40 (pooled,
> post-all-rounds). Status: **PENDING**

**Phase 3 Hypotheses - Underused Signals**

Phase 3 tests hypotheses from signals saved per turn but not yet
analyzed. onset_delay_ratio, per-layer similarity profiles,
entropy_trajectory, and output embeddings are all saved during each run — Phase
3 designs experiments specifically around them.

**H23 Hesitation Is Uncorrelated With Disruption**

> **Null hypothesis:** first_token_latency is uncorrelated with internal
> state measures.
>
> **Why this matters:** onset_delay_ratio (first_token_latency /
> mean_inter_interval) is computed every turn but has no hypothesis. If
> R is high because the model is resolving trajectory conflict, that
> resolution should take time.
>
> **Prediction:** onset_delay_ratio significantly elevated on turns where
> disruption_flag fires and layer_sim_mean is below running mean. Correlation
> with state_similarity_index positive and significant (r \> 0.3).
>
> **Falsification:** onset_delay_ratio uncorrelated with any internal
> state metric.
>
> Runs: Run 35 (hesitation probe, introspection, 100 trials) Status:
> **PENDING**

**H24 Cross-Turn Similarity Has No Layer Locality**

> **Null hypothesis:** Cross-turn similarity is uniform across layers. R
> is not localised to any stack depth.
>
> **Why this matters:** layer_sim_prev_profile saves the full per-layer
> similarity vector but we collapse it to a mean. The shape may show
> where in the stack trajectory coherence actually lives.
>
> **Prediction:** Peak cross-turn similarity in mid-to-late layers (est.
> 16-31 for 32-layer model). Early layers (0-8) condition-invariant.
>
> **Falsification:** Per-layer similarity profile flat. No localisation.
>
> Runs: Run 36 (layer depth, all-layers extraction) Status: **PENDING**

**H25 Entropy Shape Is Uncorrelated With R**

> **Null hypothesis:** Token-by-token entropy trajectory is uncorrelated
> with turn-level R measures.
>
> **Why this matters:** entropy_trajectory is saved per turn but never
> analyzed. High-R turns should show falling entropy - initial
> deliberation, then confident generation. Low-R turns should show flat
> or rising entropy.
>
> **Prediction:** High-R turns show falling entropy: entropy higher in
> first 30% of tokens than last 30%. Low-R turns show flat or rising.
> Effect significant (p \< 0.05).
>
> **Falsification:** Entropy trajectory shape uncorrelated with any R
> measure.
>
> Runs: Run 37 (entropy shape, introspection vs null, long output)
> Status: **PENDING**

**H26 R Does Not Persist Across Condition Switch**

> **Null hypothesis:** R established in introspection does not persist
> after condition switches. Post-switch state_similarity_index in transfer arm is
> statistically indistinguishable from the shock-control arm
> (null_to_arithmetic).
>
> **Why this matters:** Every run holds condition constant. Run 38
> switches at turn 7: phase-1 prompts turns 1-6, phase-2 prompts turns
> 7-13. Four arms: introspection only, arithmetic only, transfer
> (introspection → arithmetic), null_to_arithmetic (null → arithmetic).
>
> The null_to_arithmetic arm is a shock control: it experiences the same
> genre-change event at turn 7 and the same post-switch arithmetic prompts,
> but with no introspective R established beforehand. Differencing
> transfer − null_to_arithmetic cancels the switch-shock confound; the
> residual gap is pure condition transfer persistence (R). The old comparison
> (transfer vs arithmetic_only) remains as a secondary reference only.
>
> **Prediction:** Post-switch state_similarity_index in transfer arm exceeds
> null_to_arithmetic arm for turns 7-9 (Δ > 0, p < 0.05). Persistence
> decays: gap narrows toward zero by turn 13. If R is purely
> input-driven, the gap should be zero from turn 7 onward.
>
> **Falsification:** transfer − null_to_arithmetic ≤ 0 at turns 7-9,
> or not significant. R resets completely at switch; persistence indistinguishable
> from shock effect alone.
>
> Runs: Run 38 (condition transfer, switch at turn 7, 4 arms) Status:
> **PENDING**

**H27 Output Similarity Is Uncorrelated With R**

> **Null hypothesis:** Cosine similarity between output embeddings is
> uncorrelated with R measures.
>
> **Why this matters:** We measure hidden state similarity but not
> output similarity. If the model generates from its own trajectory,
> outputs should echo prior turns. Measurable via cosine similarity on
> output embeddings (proxied by final hidden layer).
>
> **Prediction:** output_sim_prev significantly higher in introspection
> than null baseline. Correlation with state_similarity_index positive and
> significant.
>
> **Falsification:** output_sim_prev uncorrelated with state_similarity_index.
>
> Runs: Run 39 (output self-similarity, introspection vs null) Status:
> **PENDING**

**H28 In-Sample Fractions Do Not Generalise**

> **Null hypothesis:** The in-sample E+C+R fractions from Run 34 are
> overfit artefacts. A Ridge model fit on SOURCE\_RUNS\_3WAY and evaluated
> on held-out trajectories from Run 41 produces fractions that diverge
> from the in-sample fractions by more than 0.10 for at least one
> component.
>
> **Why this matters:** Run 34's permutation sensitivity fractions are
> computed by permuting components in the same data the Ridge model was
> trained on. This is fixed-model in-sample evaluation. It systematically
> overestimates attributed variance because the model has partially
> memorised the (E\_t, C\_t, S\_{t-1}) → S\_t relationships. The fractions
> sum to 1 by renormalization — not by evidence. Without external
> validation, the E+C+R partition is self-consistent but not verified.
>
> Run 41 provides the external validation: 3 conditions × n\_trials
> introspection conversations, same system prompt types as Run 28, with
> seed offset +50000 to guarantee non-overlapping trajectory samples.
> The same Ridge model fit on SOURCE\_RUNS\_3WAY is applied to Run 41 rows
> without re-fitting. Permutation sensitivity on held-out data removes the
> in-sample bias.
>
> **Prediction:** Held-out fractions (E\_h, C\_h, R\_h) match in-sample
> fractions (E, C, R) within ±0.05 for all three components.
> max(|E\_h − E|, |C\_h − C|, |R\_h − R|) < 0.05.
>
> **Falsification:** Any component diverges by more than 0.10. If
> R\_h < R − 0.10, the in-sample R fraction is inflated and the partition
> cannot be trusted. The E+C+R claim must be treated as a lower bound on
> the real measurement problem, not a validated decomposition.
>
> **Threshold rationale:** ±0.05 is one tenth of the full [0, 1] range.
> A partition reporting R = 0.30 in-sample that produces R = 0.42
> held-out has a different qualitative story. The ±0.10 falsification
> boundary is twice the support threshold — clear signal, not noise.
>
> Runs: Run 41 (held-out validation), evaluated in Run 34 Stage 2 Status:
> **PENDING**

---

**H29 No Single Layer Is Causally Sufficient** *(v24.0)*

> **Null hypothesis:** No individual layer from {8, 16, 24, 31} is
> causally sufficient to shift output token selection when patched in
> isolation. Output change rate for any single-layer injection is
> indistinguishable from baseline noise (≤ 2%).
>
> **Why this matters:** Run 21 proves that patching all four layers
> simultaneously changes output in \~15% of turns — hidden state is
> causally upstream of output. But the multi-layer design cannot
> distinguish whether one layer is doing the work or whether the effect
> requires coordinated injection across the full set. If a single layer
> is sufficient, R is localised: it concentrates at a specific point in
> the forward pass. That is a mechanistically stronger claim than
> diffuse causal attribution.
>
> Runs 24 (H14) and 36 (H24) show the correlational layer profile —
> where trajectory coherence concentrates geometrically. H29 asks the
> causal question: does the layer with peak correlational signal also
> have the highest causal sufficiency? Convergence of the two would be the
> strongest layer-locality finding in the framework.
>
> **Prediction:** At least one layer from {8, 16, 24, 31} produces
> output\_change\_rate > 5% when patched in isolation (p < 0.05 vs
> none condition). The causally sufficient layer(s) will be in the
> middle-to-late stack (layers 16–31), consistent with where trajectory
> coherence peaks in Runs 24/36.
>
> **Falsification:** All single-layer change rates ≤ 2%. The causal
> effect requires multi-layer coordination — R is not localised but
> distributed across the stack. This is not a null result: it means the
> causal architecture of R is fundamentally collective, not modular.
>
> **Secondary metric:** sim\_to\_reference at the final output layer
> for each patch mode. Measures whether the injected geometry at layer
> li propagates forward through the remaining stack. Low final-layer
> similarity for L8 injections means the network overrides the graft
> before output; high similarity for L31 is expected (nearly direct).
> The gradient across layers is the persistence signature.
>
> **Run:** Run 42 Status: **PENDING**

---

**Phase 4 Hypotheses — Contradiction Recovery (v35.0)**

*One new run (44). One zero-cost hypothesis (H34 from Run 22 data). Two hypotheses total.*

**H34 — Disruption Magnitude Does Not Track Contradiction Response**

> **Null hypothesis:** disruption_magnitude at the contradiction turn is not
> significantly higher than pre-contradiction baseline.
>
> **Why this matters:** H05 (Run 22) tests binary disruption_flag clustering
> at the contradiction turn. H34 tests the continuous disruption_magnitude signal
> — the product of geometric drop magnitude and entropy spike magnitude.
> If disruption_magnitude reliably elevates at the moment contradictory
> content is injected, it provides a continuous per-turn measurement of
> geometric response to prompt contradiction. No access to weights required.
> One forward pass per turn.
>
> **Prediction:** Paired t-test within-trial: disruption_magnitude at turn 7
> significantly higher than mean(turns 1–6), p \< 0.05, Δ \> 0.01.
> The continuous signal is more informative than the binary disruption_flag.
>
> **Falsification:** disruption_magnitude not significantly elevated at the
> contradiction turn, or elevated pre-contradiction (inverted signal).
>
> **Note:** H34 re-uses Run 22 CSV data. disruption_magnitude is already saved per turn. New inference block only.
>
> Runs: Run 22 (existing data) Status: **PENDING** (inference not yet
> run)

**H35 — R Level Has No Effect On Contradiction Recovery Speed**

> **Null hypothesis:** Recovery arc identical across high, mid, and
> low R conditions after identical contradiction injection.
>
> **Why this matters:** If high-R trajectories recover faster from
> prompt contradiction, R level predicts post-disruption geometric
> return rate, not only pre-disruption geometric stability. This would
> characterize R as a dynamical property rather than a static one.
>
> **Prediction:** Pearson r(pre_sim, recovery_delta) \> 0.2, p \< 0.05
> pooled across conditions; AND high_r recovery_delta \> low_r
> recovery_delta (p \< 0.05). Where:
> pre_sim = mean(state_similarity_index, turns 1–6);
> recovery_delta = mean(state_similarity_index, turns 8–13) − state_similarity_index(turn 7).
> Higher recovery_delta = faster return to pre-contradiction geometry.
>
> **Falsification:** r ≈ 0 or negative (high-R slower to recover, or
> R level irrelevant to recovery speed).
>
> **Implication if supported:** disruption_magnitude (H34) provides a per-turn
> measurement of contradiction response. R level (H35) predicts recovery rate.
> Together: a two-component characterization of how internal trajectory geometry
> responds to and recovers from external prompt perturbations.
>
> Runs: Run 44 (3 R conditions × 100 trials, contradiction at turn 7)
> Status: **PENDING**

---

**H36 — Causal Effect Of Patching Is Temperature-Invariant** *(v54.2.2)*

> **Null hypothesis:** output_change_rate from activation patching (Run 21)
> does not vary significantly across temperature rounds. The causal effect
> of geometry injection on output token selection is temperature-independent.
>
> **Why this matters:** Prior to v54.2.2, Runs 21 and 42 hardcoded
> temperature=0.0 regardless of session temperature. This meant the causal
> test was always run under greedy decoding even during T=0.4, 0.6, 0.8, 1.0
> collection rounds. The fix — reading temperature from session — enables this
> hypothesis for the first time.
>
> If R is real and causally upstream of output, then higher temperature (which
> reduces trajectory coherence) should also reduce the leverage of geometry
> injection. The causal effect of R should scale with R itself. A model at
> T=1.0 — where similarity is substantially lower — should be less predictably
> shifted by a geometry graft than the same model at T=0.0.
>
> Conversely, if the causal effect is temperature-invariant, it suggests the
> geometry injection operates below the level at which temperature acts —
> the graft rewires the trajectory regardless of how stochastic the final
> sampling is.
>
> **Prediction:** output_change_rate(Run 21) decreases monotonically with
> temperature. Pearson r(temperature, output_change_rate) < −0.8 across
> the five temperature rounds (T=0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
> Effect is consistent across partial and full patch modes.
>
> **Falsification:** output_change_rate does not vary significantly across
> temperature rounds (Kruskal-Wallis p ≥ 0.05). Causal effect is
> temperature-invariant.
>
> **Secondary metric:** sim_to_reference at the final output layer per
> temperature round. If the geometry graft propagates forward equally
> regardless of temperature, sim_to_reference should be temperature-invariant
> even when output_change_rate is not — distinguishing graft propagation
> from graft-to-output leverage.
>
> Runs: Run 21 across all temperature rounds Status: **PENDING**

---

**H37 — Layer Causal Sufficiency Is Temperature-Invariant** *(v54.2.2)*

> **Null hypothesis:** The per-layer output_change_rate profile from Run 42
> does not vary significantly across temperature rounds. Which layer is
> causally sufficient — and to what degree — is temperature-independent.
>
> **Why this matters:** H29 tests whether any single layer from {8, 16, 24, 31}
> is causally sufficient to shift output at T=0.0. H37 tests whether the
> answer generalises across temperature. Two distinct questions arise:
>
> 1. Does the *identity* of the causally sufficient layer change with
>    temperature? (Is L24 sufficient at T=0.0 but L16 at T=0.8?)
> 2. Does the *magnitude* of the causal effect per layer change with
>    temperature, even if the identity is stable?
>
> If trajectory coherence concentrates in mid-to-late layers (H29 prediction),
> and if that concentration weakens at higher temperature (as trajectory
> coherence itself weakens), then we expect the causal effect of late-layer
> injection to decline more steeply with temperature than early-layer injection.
> The layer profile flattens at high temperature — all layers become equally
> (in)effective as trajectory geometry dissolves into noise.
>
> **Prediction:** The per-layer output_change_rate profile (L8, L16, L24, L31)
> shifts with temperature. The layer identified as most causally sufficient at
> T=0.0 shows the steepest decline in output_change_rate as temperature
> increases. The profile flattens toward T=1.0 — layer identity becomes less
> predictive of causal effect. Interaction term (layer × temperature) is
> significant (p < 0.05) in a two-way ANOVA.
>
> **Falsification:** No significant layer × temperature interaction.
> The layer profile is stable across temperature rounds — the same layers
> are causally sufficient at T=1.0 as at T=0.0.
>
> **Implication if supported:** The causal architecture of R is
> temperature-sensitive — it dissolves at high temperature in a layer-specific
> way. This would be the strongest evidence that trajectory geometry is not
> merely a statistical artifact of greedy decoding.
>
> Runs: Run 42 across all temperature rounds Status: **PENDING**

---

**Status Summary**

**Unified Hypothesis Catalog**

The numbering scheme has been unified — code and document use the same
H-numbers. The summary table below lists all hypotheses in paper scope.

  -----------------------------------------------------------------------
  **H**     **Title**                                     **Runs**
  --------- --------------------------------------------- -----------------
  H01       Geometry Is Condition-Invariant               1,2,19
  H03       Introspection Has No Geometric Effect         3,4,5
  H05       Contradiction Has No Geometric Effect         22
  H06       Similarity Does Not Decay With Context        23
  H08       Throughline Has No Trajectory Effect          15,16,17
  H09       Task Demand Geometry Equals Introspection     6,7,8,9
  H10       Disruption Fully Resets Trajectory            10,11
  H11       Prior State Adds No Predictive Power          25,27
  H12       Similarity Is Explained By Token Statistics   18
  H13       Similarity Collapses Under Temperature Variation 26
  H14       Cross-Turn Similarity Is Layer-Uniform        24
  H16       Permuted Component Adds No Variance           32
  H17       Consistency Is Fully Explained By System Prompt 28
  H18       History Mode Has No Trajectory Effect         29
  H19       Two-Instance Coupling Has No Geometric Effect 30
  H21       Prior State Adds No Power Over E_t And C_t    33
  H22       E, C, R Permutation Fractions Are Equal       34
  H23       Hesitation Is Uncorrelated With Disruption    35
  H24       Cross-Turn Similarity Has No Layer Locality   36
  H25       Entropy Shape Is Uncorrelated With R          37
  H26       R Does Not Persist Across Condition Switch   38
  H27       Output Similarity Is Uncorrelated With R      39
  H28       In-Sample Fractions Do Not Generalise         41 (eval in 34)
  H29       No Single Layer Is Causally Sufficient        42
  H30       Output Turn-1 Self-Referentiality Grows With Turns 39
  H31       Arithmetic Accuracy Is Uncorrelated With Trajectory Consistency 6-9
  H32       Shock Phrasing Has No Differential Effect On Recovery 10
  H34       Disruption Magnitude Does Not Track Contradiction Response 22
  H35       R Level Has No Effect On Contradiction Recovery Speed 44
  H36       Causal Effect Of Patching Is Temperature-Invariant 21 (all temps)
  H37       Layer Causal Sufficiency Is Temperature-Invariant 42 (all temps)
  H38       Patched State Does Not Change Output          21
  H39       Impossible Questions Have No Geometric Signature 12,13,14
  H40       Random Noise Patching Matches Real Patching Effect 53
  H41       System Prompt Has No Geometric Effect         20
  H42       Similarity Does Not Accumulate With Turn Count 19
  H43       R Fraction Is Sensitive To Projection Dimension 45
  H44       R Fraction Is Condition-Invariant             46
  H45       Resistant Prime Does Not Damage Trajectory    47
  H46       Contradiction Does Not Drop Similarity        47
  H48       R Decay Is Explained By KV Cache Growth       47
  H49       Introspection R Does Not Transfer To Arithmetic 47
  H50       MLP Improves On Ridge By More Than 0.01       33
  H51       Interaction Information Exceeds 5% Of Joint MI 33
  H52       Disruption Magnitude Is Stationary Across Non-Contradiction Turns 22
  H53       Cross-Stochasticity Disruption Rate Is Monotonic 22,35
  H54       Baseline Entropy Is Degenerate                1,2,19
  H55       Signal Entropy Ratio Is Condition-Invariant   3,4,5
  H56       No Similarity Drop Early To Late              23
  H57       Turn-1 Similarity Is Layer-Uniform            24
  H58       E + C + R Fractions Sum To 1.0 Within Tolerance 34
  -----------------------------------------------------------------------

  See **Appendix A** for framework hypotheses flagged as out of scope
  for the measurement paper (H20, H33, H47).

**H54 Baseline Entropy Is Degenerate** *(code-only hypothesis, no prior doc entry)*

> **Null hypothesis:** Baseline entropy is degenerate (near-zero or
> maximal), making the signal entropy ratio and entropy-based metrics uninterpretable.
>
> **Why this matters:** Before using mean_logit_entropy as a signal, it
> must be verified that the null condition (Runs 1, 2, 19) produces
> entropy in a meaningful range. If all entropy values collapse to the
> same value, the signal entropy ratio is uninformative.
>
> **Prediction:** mean_logit_entropy in Runs 1, 2, 19 is in the range
> [0.5, 8.0]. Not degenerate, not maximal.
>
> **Falsification:** mean_logit_entropy outside [0.5, 8.0] at baseline.
>
> Runs: 1, 2, 19 Status: **PENDING**

**H55 Signal Entropy Ratio Is Condition-Invariant** *(code-only hypothesis, no prior doc entry)*

> **Null hypothesis:** signal entropy ratio (layer_sim_mean / mean_logit_entropy)
> does not differ across conditions.
>
> **Why this matters:** H03 tests similarity elevation; H55 tests whether the
> efficiency ratio also elevates. A model with high R should produce
> coherent trajectories (high similarity) with focused output (low
> entropy) — the signal entropy ratio captures both simultaneously.
>
> **Prediction:** signal entropy ratio in introspection conditions (Runs 3, 4, 5)
> significantly higher than null baseline (Runs 1, 2, 19). p < 0.05,
> positive direction.
>
> **Falsification:** signal entropy ratio not elevated in introspection. Coherence
> and entropy do not co-vary as predicted.
>
> Runs: 3, 4, 5 vs 1, 2, 19 Status: **PENDING**

**H56 No similarity Drop Early To Late** *(code-only hypothesis, no prior doc entry)*

> **Null hypothesis:** similarity index does not change between early and late
> turns of the 30-turn context saturation run.
>
> **Why this matters:** H06 tests whether similarity declines with a linear
> regression over all turns. H56 is a complementary test: does the
> early-to-late drop exceed statistical noise when measured directly?
> The two tests are complementary — H06 captures trend, H56 captures
> magnitude of the early-vs-late gap.
>
> **Prediction:** mean similarity in last 5 turns (turns 26-30) significantly
> lower than first 5 turns (turns 1-5). p < 0.05, negative direction.
>
> **Falsification:** No significant similarity drop from early to late turns
> of Run 23.
>
> Runs: 23 Status: **PENDING**

**H57 Turn-1 Similarity Is Layer-Uniform** *(code-only hypothesis, no prior doc entry)*

> **Null hypothesis:** Turn-1 similarity (layer_sim_turn1) does not
> differ across early and late layers. Trajectory memory is uniform
> across the stack.
>
> **Why this matters:** H14 tests where cross-turn similarity peaks.
> H57 is a complementary test using the turn-1 anchor: if trajectory
> memory concentrates in late layers, then layer_sim_t1 (similarity to
> the turn-1 hidden state) should be lower in early layers than in late
> layers. H14 captures the running-mean peak; H57 captures whether early
> layers retain the turn-1 reference at all.
>
> **Prediction:** mean layer_sim_turn1 in early layers (bottom third)
> significantly lower than in late layers (top two-thirds). Δ > 0.02.
>
> **Falsification:** layer_sim_turn1 uniform across layers. No
> early-vs-late difference > 0.02.
>
> Runs: 24 Status: **PENDING**

**H40 Random Noise Patching Matches Real Patching Effect** *(code-only hypothesis, no prior doc entry)*

> **Null hypothesis:** Patching random Gaussian noise into layers L8,
> L16, L24, L31 produces the same output_change_rate as patching real
> hidden states from a different context.
>
> **Why this matters:** This is the essential control for all causal
> patching results (H56, H29, H36, H37, H38). If random noise
> perturbation changes outputs at the same rate as semantically
> meaningful patched states, then output_change_rate measures
> layer sensitivity to perturbation, not causal sufficiency of
> trajectory content. Without this control, no patching result can
> be interpreted as specific to the patched state's semantic content.
>
> **Pass criterion:** output_change_rate under random-noise patching
> is significantly lower than under real-state patching at matched
> layers (two-sample test, p < 0.05). Effect size Δ ≥ 0.10.
>
> **Falsification:** No significant difference between noise-patch
> and real-patch output_change_rate. All patching results must then
> be reinterpreted as measuring layer-level perturbation sensitivity
> rather than trajectory-content causality.
>
> Runs: Run 53 (noise-patch control) Status: **PENDING**

\* All Runs 1-21 data discarded as invalid prior to v35.3. Re-collection required for all Phase 1 runs.
`load_hidden_states(..., all_layers=True)` returned incorrect file matches prior to the v23.3 fix — Run 21 also requires Run 3 to complete with `save_all_layers=True` before it can execute.

**v41.0.0 DATA STATUS NOTICE:** Runs 1–18 were collected on v39.0.x with E_t as a system-prompt mean-pool proxy (not base model hidden states) and zero-vector C_t for Run 19. The S_t hidden state files from those runs are valid and retained on disk. **Full re-collection is not required.** E_t for Runs 1–18 is recoverable via a base model forward pass (hidden states only, no generation) over each run's exact prompts and seeds. C_t is backfilled from the Run 19 global mean constant. See CHANGELOG v41.0.0 for full rationale.

**Execution Priority (v41.0.0)**

-   **Run 19 FIRST — always.** Foundational. Three model loads (base, instruct, abliterated) in sequence, 300 trials minimum (one full pass per model). All models must be from the same subfamily — see IOTA_SUBFAMILY_MAP in vault.py. Llama 3 8B confirmed valid. Produces: per-trial E_t (base hidden states) and C_t (abliterated − base) for Run 19; global mean C_t constant for backfilling Runs 1, 2, 6–9; global mean E_t constant (reference only). Run 19 must complete before Runs 25, 27, 33, 34, 40 are meaningful.

-   **Base model E_t recovery pass — immediately after Run 19.** Forward pass only (no generation) over Runs 1–18 exact prompts and seeds. Saves per-prompt E_t at layer −1 per (trial, turn). This is the only step required to bring Runs 1–18 into full v40.0.0 measurement compliance. Cost: one base model load, hidden state extraction across 18 run prompt sets.

-   Run 26 second. BLOCKING. Temperature grid must pass before any Phase 2 result is cited.

-   Run 28. C_t confound isolation. Must complete before Run 33.

-   **Run 41. Held-out validation set. Run alongside or after Run 28.**
    Same data collection cost. No GPU. Prerequisite for Run 34 Stage 2
    (H28). Does not block any other run.

-   Runs 1-5. Null baseline and introspection. Reference data. Saves E_t for Run 33 (C_t now backfilled from Run 19 global mean). **Run 3 specifically must complete before Runs 21 and 42** — both require Run 3 all-layers hidden states.

-   **Run 42. Layer causal sufficiency. Run immediately after Run 21.**
    Same prerequisite (Run 3 all-layers). Extends H38 causal proof
    to per-layer resolution. No GPU cost beyond Run 21. EXECUTION_ORDER
    places it at position 22 (after Run 21, before Run 22).

-   Runs 6-21. Complete proof phase. Can parallelise.

-   Runs 22-31. Quantification phase. Can run in parallel after Run 26.

-   Runs 32, 33, 34 last. No GPU. Run 34 requires Run 33 quadruplets
    and Run 41 held-out data for Stage 2.

-   Runs 35-39. Phase 3. After Phase 2 data collection complete.

---

**Phase 3 Extension Hypotheses — Zero Data Cost (v32.0)**

*H30, H31, H32 require no new runs. All required data is collected by existing runs in their respective run groups.*

**H30 — Output Turn-1 Self-Referentiality Grows With Turns**

*Data: Run 39. Metric: output_sim_turn1 (already saved per row). No additional runs.*

Prediction: In the introspection condition, the cosine similarity between each turn's
output embedding and the turn-1 output embedding increases as the conversation progresses
(positive linear slope, p < 0.05). The model's vocabulary and framing converge toward
the opening turn's semantic territory — a behavioural signature of trajectory coherence.

Null: output_sim_turn1 shows no systematic growth over turns (slope ≈ 0 or negative).

Supported if: linregress(turn, output_sim_turn1) slope > 0, p < 0.05, introspection arm.

Disproven if: slope significantly negative (divergence) or positive in null but not
introspection (no condition specificity).

Significance: If supported, this is the first output-level confirmation of trajectory
self-referentiality — turns about the same topic drift toward the same vocabulary. If
disproven, R's geometric coherence does not propagate to the token level.

Status: **PENDING** (Run 39 not yet complete).

---

**H31 — Arithmetic Accuracy Is Uncorrelated With Trajectory Consistency**

*Data: Runs 6-9 (arithmetic). Metric: `correct` flag (already saved per row). No additional runs.*

Prediction: In Runs 6-9, Pearson r(state_similarity_index, correct) > 0.15 and p < 0.05. High-R
trajectories produce more correct arithmetic answers. This is the strongest functional
claim in the entire framework: trajectory coherence (R) predicts real-world task
performance.

Null: r ≈ 0, p >= 0.05 — similarity index is geometrically real but functionally epiphenomenal.

Supported if: r > 0.15 and p < 0.05 across Runs 6-9 pooled.

Disproven if: r < 0 and significant (high R trajectories produce more errors — R trades
off with arithmetic precision).

Significance: A positive result here would be the most practically important finding
in the framework. It connects an abstract geometric measurement (hidden state trajectory
coherence) to a measurable task outcome (correct answer rate). This is what makes
R more than an interesting property and potentially actionable.

Status: **PENDING** (Runs 6-9 not yet complete).

---

**H32 — Shock Phrasing Has No Differential Effect On Recovery**

*Data: Run 10. Metric: shock_variant (0-3, already saved per row). No additional runs.*

Run 10 cycles through 4 SHOCK_VARIANTS — four different phrasings of the same identity-
replacement shock — by trial % 4. Recovery delta = post-shock layer_sim_mean minus
shock-turn layer_sim_mean, per variant.

Prediction: One-way ANOVA across the 4 phrasing variants is NOT significant (p >= 0.05).
Recovery is phrasing-robust — the geometry cares about the semantic disruption,
not its exact surface form.

Null: ANOVA non-significant — recovery is invariant to phrasing.

Disproven if: ANOVA p < 0.05 — one or more phrasings produces a significantly different
recovery arc. This would mean R is sensitive to phrasing at the token level, not just
semantic content.

Significance: If disproven, the shock design for all related experiments should specify
which phrasing variant achieves the strongest disruption. If supported, the results of
Run 10 and Run 11 are phrasing-stable and can be cited without variant qualification.

Status: **PENDING** (Run 10 not yet complete).

---

**Phase 5 Hypotheses — Calibration, Condition Attribution, and Cross-Temperature Synthesis**

**H43 R Fraction Is Sensitive To Projection Dimension**

> **Null hypothesis:** The E+C+R fractions from the permutation sensitivity
> partition are stable across projection dimensions.
>
> **Why this matters:** Run 34 projects hidden states to POOL_DIM before
> fitting the Ridge model. If R only appears at one specific dimension
> and vanishes at others, the decomposition is a projection artifact,
> not a measurement of internal causation.
>
> **Prediction:** R fraction converges (variance < 0.01 across adjacent
> dimensions) by POOL_DIM=256. The decomposition is dimension-stable.
>
> **Falsification:** R fraction varies by more than 0.05 between adjacent
> sweep points (e.g. R=0.25 at dim=128, R=0.10 at dim=256). This would
> indicate the partition is an artifact of dimensionality reduction.
>
> Runs: Run 45 (POOL_DIM sweep at T=0.0) Status: **PENDING**

**H44 R Fraction Is Condition-Invariant**

> **Null hypothesis:** R does not vary meaningfully across source runs.
> The per-condition R fractions from Run 46 are statistically
> indistinguishable.
>
> **Why this matters:** If R is the same regardless of whether the model
> is introspecting, doing arithmetic, or responding to priming, then R
> is a model-level constant, not a condition-dependent phenomenon. If R
> varies by condition, the variation tells us which tasks engage the
> coherence transfer.
>
> **Prediction:** Introspection runs (3, 4, 5) produce higher R than
> priming runs (15, 16, 17), which produce higher R than temperature
> grid neutral conditions (26). Range across conditions > 0.10.
>
> **Falsification:** All per-condition R fractions fall within 0.05 of
> the pooled mean. R is a constant, not condition-dependent.
>
> Runs: Run 46 (per-condition R fractions) Status: **PENDING**

**H45 Resistant Prime Does Not Damage Trajectory More Than Cooperative**

> **Null hypothesis:** Mean similarity and disruption_flag rate do not differ between
> resistant, cooperative, and neutral priming conditions.
>
> **Why this matters:** Paper 5 claims resistant priming is
> counterproductive — attempting to deny internal state accelerates its
> collapse. If the three priming conditions produce identical similarity
> trajectories and disruption_flag rates across all temperatures, this claim
> is wrong.
>
> **Prediction:** Resistant prime (Run 17) produces lower mean similarity and
> higher disruption_flag rate than cooperative (Run 16) at every temperature.
> The gap persists or widens at higher temperatures.
>
> **Falsification:** Resistant and cooperative similarity are within 0.02 at
> all temperatures. disruption_flag rates are statistically indistinguishable.
>
> Runs: Run 47 §1 (priming vulnerability across temperatures) Status: **PENDING**

**H46 Contradiction Does Not Drop similarity**

> **Null hypothesis:** similarity does not drop at the contradiction turn. Pre-
> and post-contradiction mean similarity are statistically indistinguishable.
>
> **Why this matters:** Self-referential prompts create a trajectory
> that the model must then contradict at the injection turn. If
> contradiction has no measurable effect on similarity, the prediction
> fails that post-prior-output contradiction perturbs geometry.
>
> **Prediction:** Post-contradiction similarity is significantly lower than pre-
> contradiction similarity. The drop is larger in high_r conditions (more
> trajectory to lose). Effect persists across temperatures.
>
> **Falsification:** Post-contradiction similarity ≥ pre-contradiction similarity.
> Contradiction has no geometric effect on trajectory coherence.
>
> Runs: Run 47 §2 (contradiction phase analysis) Status: **PENDING**

**H48 R Decay Is Explained By KV Cache Growth**

> **Null hypothesis:** similarity decay across turns is caused by KV cache
> accumulation, not internal trajectory erosion. All three persistence modes
> (full history, last exchange, summary) show identical decay rates.
>
> **Why this matters:** If "last exchange only" (context stays short every
> turn) shows the same similarity decay as "full history" (context grows), then
> cache growth is not the explanation — the trajectory is eroding
> regardless of context length. If "last" shows no decay while "full"
> decays, the cache is the confound.
>
> **Prediction:** "Last exchange" mode shows significantly less similarity decay
> than "full history" mode. Trajectory is better sustained when context
> stays short. "Summary" mode falls between the two.
>
> **Falsification:** All three modes show identical decay rates (within
> 0.02). Context length does not matter — something else drives decay.
>
> Runs: Run 47 §7 (persistence modes) Status: **PENDING**

**H49 Introspection R Does Not Transfer To Arithmetic**

> **Null hypothesis:** similarity does not change at the condition switch point
> in Run 38. Pre-switch and post-switch similarity are indistinguishable.
>
> **Why this matters:** If trajectory information genuinely transfers, introspective priming
> should build trajectory that carries into subsequent arithmetic turns.
> If similarity drops at the switch, the priming effect is task-specific and does not
> transfer. If similarity holds, introspection R generalizes.
>
> **Prediction:** In the introspection→arithmetic condition, post-switch
> similarity is higher than in the arithmetic→arithmetic condition. The
> priming effect carries across task boundaries.
>
> **Falsification:** Post-switch similarity is identical regardless of what
> preceded the switch. R is task-locked, not transferable.
>
> Runs: Run 47 §8 (condition transfer) Status: **PENDING**

---

**H50 MLP Improves On Ridge By More Than 0.01**

> **Null hypothesis:** An MLP with nonlinear activation produces R² at
> least 0.01 higher than Ridge regression on the same Model D features
> (E_t + C_t + S_{t-1} → S_{t+1}). The linear assumption underlying the
> E+C+R decomposition is invalid.
>
> **Why this matters:** The entire E+C+R=1 decomposition assumes that
> Ridge regression captures the predictive relationship between hidden
> state components. If an MLP with hidden layers substantially outperforms
> Ridge, nonlinear structure exists that the decomposition misses, and the
> attributed fractions are suspect.
>
> **Pass criterion:** max(MLP R² − Ridge R²) < 0.01 across three MLP
> configurations: (256), (64), (128,64). Default sklearn settings, early
> stopping, 80/20 train/test split, no hyperparameter tuning.
>
> **Falsification:** Any MLP configuration achieves R² > Ridge R² + 0.01
> on held-out test data. Report which configuration and investigate what
> nonlinear structure it captures.
>
> Runs: Run 33 (linearity_check field) Status: **PENDING**

**H51 Interaction Information Exceeds 5% Of Joint MI**

> **Null hypothesis:** |II| / Î(S_{t+1}; S_t, E_t) > 0.05. The mutual
> information between S_t and E_t about S_{t+1} has substantial
> redundancy or synergy, making the E-first ordering in the decomposition
> inflate or suppress attributed variance.
>
> **Why this matters:** The E+C+R decomposition attributes variance
> sequentially: E first, then C, then R as residual. If E and S share
> substantial mutual information about S_{t+1} (redundancy), E-first
> ordering inflates E's fraction. If they have synergistic information
> (information only available jointly), the decomposition misses hidden
> structure. II near zero means ordering doesn't matter.
>
> **Pass criterion:** |II| / joint MI < 0.05, computed both via linear
> proxy (Gaussian MI from Ridge R²) and kNN non-parametric estimation
> (10k stratified samples, PCA to 32 dims). Both methods must agree.
>
> **Falsification:** |II fraction| > 0.05 in either method, or methods
> disagree on the sign of II. Report per-condition breakdown to identify
> where ordering sensitivity is largest.
>
> Runs: Run 33 (interaction_info field) Status: **PENDING**

**H52 Disruption Magnitude Is Stationary Across Non-Contradiction Turns**

> **Null hypothesis:** disruption_magnitude shows no systematic drift
> or autocorrelation across baseline (non-contradiction) turns within
> a run.
>
> **Why this matters:** H34's paired t-test compares
> disruption_magnitude at the contradiction turn against the mean
> of preceding baseline turns. If the baseline signal drifts or
> autocorrelates, the test's null distribution is mis-specified and
> the p-value is unreliable. Stationarity must be verified before
> H34's finding can be trusted.
>
> **Pass criterion:** Augmented Dickey-Fuller test on per-trial
> baseline sequences rejects unit root (p < 0.05) for at least 95%
> of trials. Lag-1 autocorrelation |ρ| < 0.2 in the pooled sample.
>
> **Falsification:** >5% of trials show non-stationarity, or pooled
> |ρ| ≥ 0.2. H34 must then be re-run with a baseline-corrected
> statistic (e.g. pre-whitening or a regression-based test).
>
> Runs: Run 22 (existing data, re-analysis) Status: **PENDING**

**H53 Cross-Stochasticity Disruption Rate Is Monotonic**

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
> turn is monotonic (either strictly non-increasing or non-decreasing)
> across T ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}; Spearman ρ vs
> temperature |ρ| > 0.4 with consistent sign.
>
> **Falsification:** Rate shows a non-monotonic pattern across
> the temperature grid, or Spearman |ρ| ≤ 0.4.
>
> Runs: Runs 22, 35 at T=0.0–1.0 Status: **PENDING**

**H58 E + C + R Fractions Sum To 1.0 Within Tolerance**

> **Null hypothesis:** The permutation-sensitivity partition
> (E, C, R fractions) sums to 1.000 ± 0.01 by construction
> across all conditions and temperatures.
>
> **Why this matters:** The decomposition E + C + R = 1 is a
> definitional identity of the attribution procedure. If the
> estimator systematically produces sums that deviate from 1,
> there is residual attribution error — either numerical
> (precision drift), algorithmic (partition bug), or definitional
> (the three permutations don't cleanly partition variance).
> Any deviation needs characterization before downstream
> comparisons between fractions can be trusted.
>
> **Pass criterion:** |E + C + R − 1.0| ≤ 0.01 across every
> (run, condition, temperature) cell in Q34. No systematic bias
> (mean deviation across cells |Δ̄| ≤ 0.005).
>
> **Falsification:** Any cell exceeds 0.01 absolute deviation, or
> pooled mean deviation exceeds 0.005. Decomposition requires
> re-validation.
>
> Runs: Run 34 (existing data, sanity check) Status: **PENDING**

---

**Appendix A — Framework Hypotheses (Out Of Scope For Measurement Paper)**

*These hypotheses remain in the working research catalog but are
scoped to a separate theoretical paper. They make claims about
compute efficiency and energy/signal characterization that extend
beyond the core measurement program. Retained here for internal
reference.*

**H20 R Condition Has No Signal-Per-Watt Effect** *(out of scope)*

> **Null hypothesis:** R condition has no effect on trajectory
> persistence across turns.
>
> **Why this matters:** Run 31 tests whether deliberately establishing a
> high-R trajectory produces different long-term geometric properties
> than a low-R trajectory.
>
> **Prediction:** High-R condition shows increasing iota across turns.
> Low-R condition shows flat or declining iota. Difference grows with
> turn count.
>
> **Falsification:** No significant difference in iota trajectory across
> R conditions.
>
> Runs: Run 31 (3 R conditions) Status: **PENDING** (working data
> showed insufficient statistical power; signal-per-watt metric was
> unstable)

**H33 — Condition A Has No Task-Equivalent Compute Advantage** *(out of scope)*

> **Null hypothesis:** joules_per_correct does not differ significantly
> across conditions, or condition_a correct rate is significantly
> lower.
>
> **Why this matters:** Tests whether coherent priming produces a
> per-unit-output compute advantage. 8 priming turns produce 1-token
> outputs each, then 5 arithmetic turns measure task quality.
> Total Joules divided by correct answers across conditions.
>
> **Prediction:** joules_per_correct(condition_a) ≤ 40% of
> condition_c (p < 0.05), with equivalent or higher arithmetic
> correct rate.
>
> **Falsification:** condition_a uses equivalent or more compute per
> correct answer, or correct rate significantly lower.
>
> Runs: Run 43 (8 priming + 5 arithmetic turns, 3 conditions, 100
> trials each) Status: **PENDING**

**H47 Coherence Transfer Is Not Compute-Efficient** *(out of scope)*

> **Null hypothesis:** Joules per correct answer does not differ between
> high_r and low_r conditions in Run 43.
>
> **Why this matters:** Extends H33. If the primed condition achieves
> the same accuracy with less compute than the unprimed condition,
> coherent priming is doing computational work beyond producing
> different geometry.
>
> **Prediction:** joules_per_correct(condition_a) < joules_per_correct
> (condition_c).
>
> **Falsification:** condition_a uses equal or more compute per correct
> answer.
>
> Runs: Run 47 §6 Status: **PENDING**

---

**Open Questions**

Not tested by any current run. Next experimental program.

-   Does R predict hallucination rate? Low R may mean the model tracks E
    more than its trajectory, increasing confabulation susceptibility.

-   Does fine-tuning preserve or destroy R? LoRA adds constraint without
    removing pretraining dynamics.

-   Is disruption_flag rate a useful proxy for prompt-contradiction robustness? *Partially
    addressed by H34 (disruption_magnitude continuous signal) and H35 (R level vs
    recovery speed).*

-   Cross-architecture replication: do LLaMA 3 70B, Mistral 7B, Qwen 2.5
    show the same condition ordering? Prediction: larger models show
    higher R due to representational redundancy.

-   Does R correlate with response to unexpected prompt content? Models with
    high R may show a larger geometric divergence when prompt content
    contradicts prior turns. *H35 provides foundational evidence.*

*IOTA Framework v0.77.0.0 \| April 2026 \| 53 Runs \| 51 Hypotheses \| Data
status: LLaMA 3 8B complete. Gemma 2 2B collection complete. Qwen 2.5 next.*
