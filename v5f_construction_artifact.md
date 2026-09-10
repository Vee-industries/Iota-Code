# V5f construction artifact

Companion record for the V5f doubly-conditional calibration system. Referenced
from the apparatus metadata as `v5f_construction_artifact_path` and from the
apparatus paper's Appendix B. Every value below is reproduced from
`data/paper/calibration/v5/v5_calibration_results.json`
(`v5f_doubly_conditional.result`); nothing here is restated from memory.

Generator: `v5_synthetic_calibration.py` (`_v5f_generate`, `_v5f_tune_noise`,
`_v5f_compute_mi_proxies`, `estimate_R_3channel`).

## 1. Construction

Three-channel doubly-conditional system, `E, C -> S -> Y`:

```
E, C ~ N(0, 1)          independent
S    = E + C + eps      eps ~ N(0, sigma_S^2)
Y    = sin(S) + eta     eta ~ N(0, sigma_Y^2)
```

The system exists to place a known-truth point in the doubly-conditional
regime: both upstream channels feed a single intermediate state, and the
observable is a nonlinear function of that state alone. Conditional on `S`,
`Y` is independent of `E` and `C`. That structure is what makes the analytic
truth below exact rather than estimated.

Fixed parameters:

| Parameter | Value |
|---|---|
| `sigma_S` | 0.78209375 |
| `sigma_Y` | 0.547765625 |
| `n_steps` (final runs) | 20000 |
| `n_repeats` (seeds) | 3 |
| `tune_n_steps` | 10000 |

## 2. Tuning loop

`sigma_S` and `sigma_Y` are not chosen a priori. They are fit by binary search
so the system's mutual-information magnitudes land inside the band spanned by
the empirical transformer cells; a calibration point with MI magnitudes far
from the empirical regime would not test the apparatus where it operates.

Targets (nats): `I(Y; E, C) = 0.30`, `I(Y; S) = 0.45`.

Two phases, 11 recorded iterations total, on a 10000-step tuning sample:

- **Phase 1** searches `sigma_S` with `sigma_Y` held at 0.3, driving
  `I(Y; E, C)` toward its target. The search brackets from `sigma_S = 2.5005`
  (`mi_ec` collapsing to 0.0) through 1.25075 (0.0855) and 0.625875 (0.4232),
  converging on 0.78209375.
- **Phase 2** fixes `sigma_S` and searches `sigma_Y`, driving `I(Y; S)` toward
  its target. Final iteration: `sigma_Y = 0.547765625`, `mi_ec = 0.20486`,
  `mi_s = 0.46411`.

Tolerance is +/-10 percent per phase with a cap of 12 iterations per phase.
The full iteration trace is preserved as `result.tuning_log`.

**Convergence is one-sided on `I(Y; E, C)`.** The achieved 0.205 nats sits
below the 0.30 target and outside the +/-10 percent tolerance; `I(Y; S)`
converged cleanly (0.464 against 0.45). The `E, C -> S` path is bottlenecked
by `sigma_S`, which Phase 2 holds fixed, so no `sigma_Y` restores the upstream
target. This is a property of the construction, not a search failure, and it
is why the achieved values rather than the targets are the ones that matter
downstream.

## 3. Analytic truth

Because `Y` depends on `E` and `C` only through `S`, conditioning on `S`
absorbs the upstream channels and the prior-state share has a closed form in
terms of two mutual informations:

```
R_truth = (I(Y; S) - I(Y; E, C)) / I(Y; S)
```

Evaluated on the **final-size** runs (n = 20000, three seeds), not on the
tuning sample:

| Quantity | Value |
|---|---|
| `I(Y; E, C)` (proxy mean, nats) | 0.20393535 |
| `I(Y; S)` (proxy mean, nats) | 0.48717469 |
| `R_truth` (mean) | 0.5812973 |
| `R_truth` (std across seeds) | 0.0069754 |
| `R_truth` per seed | 0.5911506, 0.5759603, 0.5767812 |

**The tuning-sample and final-size MI values differ, and the truth uses the
final-size pair.** Substituting the tuning-log values (0.20486, 0.46411)
into the formula gives 0.5586, not 0.5813. Any reproduction that quotes the
tuning-log "achieved" numbers alongside `R_truth` will appear inconsistent;
the two come from different sample sizes by design.

MI values are Gaussian-MI approximations computed from MLP r^2
(`_v5f_compute_mi_proxies`), so `R_truth` is a high-confidence reference
rather than an exact analytic constant. The formula is exact; its inputs are
estimated.

## 4. Apparatus readings

Four-class attribution on V5f, mean shares across seeds, with joint fit
quality:

| Class | E | C | S | joint R^2 |
|---|---|---|---|---|
| Ridge | 0.00129 | 0.00103 | 0.99768 | 0.2385 |
| MLP | 0.00390 | 0.00613 | 0.98997 | 0.6166 |
| RKHS | 0.07673 | 0.09345 | 0.82982 | 0.5114 |
| Random Forest | 0.00058 | 0.00106 | 0.99836 | 0.5953 |

Three of four classes collapse the upstream channels to near-zero and place
more than 0.99 of the share on `S`, against an analytic truth of 0.581. Only
RKHS recovers non-trivial upstream shares. V5f is therefore the suite's
argument for four-class triangulation: no single class recovers reliably, and
the disagreement between classes is itself the diagnostic signal.

## 5. Limitations

- `R_truth` rests on MI proxies, not closed-form constants. Treat 0.581 as a
  high-confidence reference with the stated seed spread, not as exact.
- `I(Y; E, C)` did not reach its tuning target (§2). V5f sits near, not
  inside, the intended empirical MI band on the upstream channel.
- Three seeds. The 0.00698 standard deviation is an estimate from a small
  sample.
- One point, not a sweep. V5f fixes a single position in the doubly-conditional
  regime; it does not trace apparatus behavior across that regime.
- `S` is one-dimensional here, while the empirical cells are 64 to 1024
  dimensional before reduction. Dimensional transfer is addressed by V5g, not
  by V5f.

## 6. Reproduction

```bash
python v5_synthetic_calibration.py      # V5f runs as part of the suite
```

Seeded throughout; `random_state = 42` where estimators are involved. The
tuning loop is deterministic given the seed, so `sigma_S` and `sigma_Y`
reproduce exactly. Output is written to
`data/paper/calibration/v5/v5_calibration_results.json` under
`v5f_doubly_conditional`.

To verify the numbers in this document without re-running the suite, read the
values directly from that file:

```python
import json
d = json.load(open("data/paper/calibration/v5/v5_calibration_results.json"))
r = d["v5f_doubly_conditional"]["result"]
print(r["construction"], r["analytic_truth"])
```
