## CURRENT STATE — 0.79.4.9 (April 2026)

**Versioning:** `0.MAJOR.MINOR.PATCH`. Leading `0.` = pre-release.

---

### 0.79.4.9 — April 2026  (HOTFIX — local variable name sweep)
`analysis.py`, `export_stats.py`, `CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.8: VARIABLE NAMES REFLECT NEW IDS (D1 CATEGORY).**

Kevin directive: "everything" — including cosmetic-only references
that don't affect runtime correctness but still embed OLD IDs.
Category D1 deferred across prior ships, finally swept here.

**121 variable-name replacements across 2 files:**

Each local variable whose name embedded the OLD run-id was renamed to
the NEW id. Values were already correct; only the NAMES were stale.
No cross-file coupling — these are all scope-local variables. Zero
semantic change to program behavior.

**analysis.py (70 renames):**

| Old name        | New name        | Represents                          |
|-----------------|-----------------|-------------------------------------|
| `r34_file`      | `r43_file`      | Sobol partition file (perm sens)    |
| `d34`, `_f34`   | `d43`, `_f43`   | dict + file handle for above        |
| `r33_file`      | `r42_file`      | Decomposition file                  |
| `d33`, `_f33`   | `d42`, `_f42`   | dict + file handle for above        |
| `q34_path`, `q34` | `q43_path`, `q43` | Sobol path + dict             |
| `q33_path`, `q33` | `q42_path`, `q42` | Decomposition path + dict     |
| `q46_path`, `q46` | `q44_path`, `q44` | Per-condition R path + dict   |
| `r21_csv`       | `r17_csv`       | Activation patching CSV path        |
| `df21`          | `df17`          | DataFrame for above                 |
| `r42_csv`       | `r18_csv`       | Layer isolation CSV path            |
| `df42`          | `df18`          | DataFrame for above                 |

**export_stats.py (51 renames):**

| Old name        | New name        | Represents                          |
|-----------------|-----------------|-------------------------------------|
| `q52_path`, `q52` | `q54_path`, `q54` | Cross-model paper summary       |
| `q45_path`, `q45` | `q41_path`, `q41` | POOL_DIM sweep output           |
| `q46_path`, `q46` | `q44_path`, `q44` | Per-condition R output          |
| `q47_path`, `q47` | `q50_path`, `q50` | Cross-temp synthesis output     |

**LEFT IN PLACE (variable names correctly reflect new IDs):**

- `sub22`, `sub26`, `sub29`, `sub30`, `sub31`, `sub23`, etc. in
  export_stats figure functions — these filter on the new run_mode,
  and the variable name matches (e.g. `sub23 = df[run_mode_mask(...,
  23)]` where 23 = new confound isolation). Self-consistent.
- `r23_entry`, `r23_rows`, `r23_sub`, `run23_correction` — all in
  `analysis.py` confound-isolation section, renamed in 0.79.4.3 to
  match new ID 23.
- `df18`, `df18_all` in export_stats — renamed in 0.79.4.7 to new 18
  (layer isolation).

**METHODOLOGY:**

Used `re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, content)`
to enforce word boundaries. Avoids partial matches (e.g. `r21_csv`
doesn't match inside longer identifiers). Applied per-file, atomic.

Risk of collision: a new name might already exist as a different
variable. Verified post-sweep: `q44_path` now exists in both
analysis.py and export_stats.py, but both independently point to
`Q0044_per_condition_R.json` (the new per-cond R file). No semantic
conflict — just two files that happen to use the same standard name.

**VERIFICATION:**
- All 30 `.py` files compile.
- 56/56 dispatches clean, 0 prereq errors.
- Spot-check: `r17_csv = os.path.join(csv_dir, 'R0017_patching.csv')`
  — variable name matches filename new ID.
- Spot-check: `q43_path = os.path.join(ana, 'Q0043_sobol_partition.json')`
  — variable name matches filename new ID.

**STATUS — AFTER 0.79.4.9:**

Reference-type enumeration completed:

```
A1.  Dispatch comparisons                    [0.79.4.1]
A2.  Module constants (RUN_NUM=N)            [0.79.4.1]
A3.  Kwarg defaults                          [0.79.4.1]
A4.  Kwarg call args                         [0.79.4.2]
A5.  Set/list literals                       [0.79.4.2]
A6.  List literals                           [0.79.4.2]
A7.  'in' membership tests                   [0.79.4.1]
A8.  Dict literal KEYS (pass 1)              [0.79.4.4]
A9.  Dict literal KEYS (pass 2)              [0.79.4.5]
A10. Hardcoded ints in runner bodies         [0.79.4.7]
A11. Orchestration-level sets                [0.79.4.8]
B1.  Filename literals (quoted)              [0.79.4.0]
B2.  F-string patterns                       [pre-existing clean]
B3.  Label strings (f"Run {n:04d}")          [pre-existing clean]
B4/5/6. Prose in comments/docs/strings       [0.79.4.3]
B7.  F-string filename templates             [0.79.4.7]
C1.  DualKey-wrapped dispatch dicts          [0.79.4.0]
C2.  Raw dict.get(OLD_INT)                   [0.79.4.3]
D1.  Variable names (r21_csv, q33_path)      [0.79.4.9] ← this ship
D2.  CSV run_mode == 'N' raw-string filters  [0.79.4.3]
D3.  r['run_num'] == N runtime filters       [0.79.4.3 + 4.6 + 4.7]
E1.  JS frontend constants                   [0.79.4.2]
E2.  JS arrays (EXEC_ORDER, PHASES)          [0.79.4.8]
E3.  JS button handlers (rdpLaunchStd)       [0.79.4.8]
E4.  JS sessionStorage keys                  [0.79.4.8]
F1.  CSV run_mode data                       [migrate_run_ids.py]
G1.  JSON content data                       [migrate_run_ids.py]
J1.  Backend↔JS response key contract        [0.79.4.6]
```

All categories swept. Remaining surface area for run-id bugs is
near-zero.

**Deployment:** install over 0.79.4.8. Restart Flask.
No user-facing changes from this ship — pure-rename.

---

### 0.79.4.8 — April 2026  (HOTFIX — orchestration sets + JS arrays + session-key refs)
`start_here.py`, `scanner.py`, `runners.py`, `export_flask.py`,
`export_stats.py`, `report.py`, `CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.7: ORCHESTRATION-LEVEL OLD-ID SETS + JS ARRAYS + SESSION KEYS.**

After 0.79.4.7 caught runner-body hardcoded ints, a broader scan found
ORCHESTRATION-LEVEL bugs — sets and dispatch tables that govern how
multiple runs coordinate. These live in `start_here.py`, `scanner.py`,
and the JS frontend.

**BUG-ORCHESTRATION (15 targeted fixes across 5 files):**

*start_here.py (9 fixes):*
- `csv_fname = RUN_CSV.get(19)` — OLD 19 silently returned random-
  patching filename via DualKey shim. Flipped to `get("0001")`.
- `runs = [19] + runs` — injecting OLD 19 at head of run queue for
  an ET-recovery dependency. Flipped to `[1]`.
- `if r in (41, 25, 27, 32, 33, 34, 40)` — mixed test for analysis
  runs. Every integer was OLD; flipped to new-ID tuple
  `(33, 42, 43, 47, 48, 49, 51)`.
- `POOLED_RUNS = {40, 47, 54, 55, 50, 51, 52}` — meant to be the
  terminal-analysis runs (pooled + cross-model + output). Was ALL OLD.
  Flipped to new `{50, 51, 52, 53, 54, 55, 56}`.
- `_NEEDS_Q45 = {25, 27, 32, 33, 34, 46}` — runs that need POOL_DIM
  calibration. OLD → new `{47, 48, 49, 42, 43, 44}`.
- `45 not in analysis_runs` (paired with above) → `41 not in
  analysis_runs` (new POOL_DIM ID).
- `_SELF_LOAD = {19, 21, 30, 42, 53}` — runs that load their own
  model (vs. getting it from batch). All OLD. → `{1, 17, 18, 19, 20}`.
- `_LATE_SOLO = {21, 30, 42, 53}` — subset that runs AFTER batch. OLD
  → `{17, 18, 19, 20}`.
- `_post_status.get(48)` — E_t meta status check. OLD 48 → new 16.

*runners.py (1 fix):*
- `_SELF_LOAD = {19, 21, 30, 42, 53}` — same set name as start_here,
  duplicated here. Same fix.

*scanner.py (2 fixes):*
- `_CROSS_MODEL_RUNS = {50, 51, 52}` — OLD {50,51,52} = cross-model
  outputs, but in NEW those are {pooled, pooled, cross-model}. Flipped
  to new `{52, 53, 54}` (actual cross-model runs).
- `_POOLED_PATH_RUNS = {40, 47, 54}` — runs whose output lives in the
  pooled/ directory. OLD → new `{51, 50, 55}`.

*export_flask.py (1 fix):*
- `if 53 in status` — checking random-patching run status. OLD 53 →
  new 19.

*export_stats.py (1 fix):*
- `for label, run_num in [("A", 47), ("B", 27)]` — Granger probes.
  "A" was already new (47 = Granger A), but "B" was still OLD 27.
  Flipped "B" → 48 (new Granger B).

*report.py (1 fix):*
- `_read_csv_rows(csv_path, run_mode=42)` — reads the layer-isolation
  CSV. OLD 42 → new 18.

**JS ARRAYS (4 replaced):**

- `EXEC_ORDER` — was a 56-element array with ALL OLD IDs in old
  execution order. After renumber, new IDs ARE the execution order,
  so array is now clean `[1..56]` sequence.
- `PHASES` (5 groups with embedded ID lists) — full rewrite to new
  IDs. Data Collection = 1-40, Analysis = 41-49, Pooled = 50-51,
  Cross-Model = 52-54, Paper = 55-56.
- `ALL` — 56-element set. Already 1..56 (was correct except for
  missing 48, which was an oversight caught here).
- `_COLLECT` (JS set of collection-eligible runs) — OLD IDs, rewritten
  to contiguous new `{1..40}`.
- `_ANA` / `_ANALYSIS_RUN_SET` — OLD {25,27,32,33,34,40,45,46,47,49,56}
  → new `{41..51}` (analysis chain + independents + pooled).
- `_POOLED_RUNS` (JS) — OLD `{40,47,50,51,52,54,55}` → new
  `{50..56}` (pooled + cross-model + output).

**JS DASHBOARD BUTTON HANDLERS (4 fixes):**

- `rdpLaunchStd(19)` → `rdpLaunchStd(1)` (Run 0001 launch button)
- `rdpLaunchStd(26)` → `rdpLaunchStd(3)` (Run 0003 temp-grid launch)
- `isR53 = (n===53)` → `isR19 = (n===19)` (random-patching check)
- `isR21 = (d.patch_col===...)` → `isR17 = (...)` (activation patching
  check — variable name changed to reflect new ID)

**JS SESSION-STORAGE KEYS (3 renamed):**

- `patch_modes_21` → `patch_modes_17`
- `patch_modes_42` → `patch_modes_18`
- `patch_modes_53` → `patch_modes_19`
- `r26_cells` → `r3_cells`

These are browser localStorage/sessionStorage keys. Rename means users'
previously-saved patch-mode selections won't auto-persist across the
0.79.4.8 upgrade — one-time UX cost for consistency with new IDs.
Users can re-select their patch modes on first post-upgrade session.

**BUG-TERNARY-SYNTAX (caught mid-sed, fixed).**

My sed to rename `isR53` → `isR19` corrupted a JS ternary:
```js
isR19??'patch_modes_19':'patch_modes_18'  // broken — ?? is nullish coalescing
```
Correct form:
```js
isR19?'patch_modes_19':'patch_modes_18'   // ternary
```
sed can't distinguish. Caught manually before push. Line rewritten with
explicit parens + correct ternary.

**_RC_POPUP_RUNS FIX.**

`export_flask._RC_POPUP_RUNS = {10, 11, 22}` — runs where the RDP
popup re-checks CSV header for `disruption_flag` column. OLD
{10,11,22} = perturbation A/B + self-reference. Flipped to new
{39, 40, 28}.

**VERIFICATION:**
- All 30 `.py` files compile.
- 56/56 dispatches, 0 prereq errors.
- `_TEMP_INDEP_RUNS`, `_MC_RUNS`, `_ET_RECOVERY_RUNS` all load with
  correct new-ID keys.
- JS EXEC_ORDER is clean 1..56.
- No residual `patch_modes_21/42/53` or `r26_cells` refs.

**STILL TO HUNT NEXT:**
- Variable names in analysis.py (`q33_path`, `r42_csv`, etc.) —
  cosmetic; values correct. Deferred.
- Any obscure string literals in module docstrings not yet caught.

**Deployment:** install over 0.79.4.7. Restart Flask.
Hard-refresh dashboard (Ctrl+Shift+R) to clear JS cache.
Users may need to re-select patch modes (sessionStorage keys renamed).

---

### 0.79.4.7 — April 2026  (HOTFIX — hardcoded run_num args in collection runners)
`analysis.py`, `runners_p1.py`, `runners_p2.py`, `runners_p3.py`,
`CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.6: HARDCODED INT RUN-IDS PASSED INTO SAVERS/GENERATORS.**

This is the most pervasive class yet — inside every collection runner
function, the run's OWN id is hardcoded as an integer literal that
gets passed into `save_npy(...)`, `save_embedding(...)`,
`run_generation(...)`, `print_turn_result(...)`, `_get_trials_for_condition(...)`,
and `_log_trial_start/end(...)`. These integers become part of filenames
(via `save_npy` building `{R,Q}{num:04d}_*.npy`), CSV row tags (via
`print_turn_result` and append_csv paths), and dashboard run-IDs. Every
single hardcoded int was the OLD id.

**Why the earlier sweeps missed this class:**
- Dispatch sweep (0.79.4.1) only touched `if/elif run_num == N:`
  patterns. It didn't touch function BODIES where the int is a literal
  arg to another function call.
- Filename-literal sweep (0.79.4.0) only touched quoted strings like
  `"R33_foo.csv"`. Integer args to `save_npy(..., 33, ...)` aren't
  strings.
- Dict-literal sweep (0.79.4.4+4.5) only touched `{N: value}` patterns.
  Positional args aren't dict literals.

**The bug's silent failure mode:**

Say `runners.py` dispatch calls `_run_temperature_grid()` when new
`run_num == 3`. Inside, the old code has:
```python
save_npy(_lh, paths['hidden'], 26, session['model_name'], _gt, _ti)
print_turn_result(26, trial, turn_idx, ...)
```
26 is the OLD id. `save_npy` builds filename from run_num=26 as
`Q0026_*.npy`. But cartography says `RUN_CSV["0003"] = R0003_temperature_grid.csv`
and the paired hidden-state files should be `R0003_*.npy`. Analysis
downstream looks for `R0003_*.npy`, doesn't find them, silently treats
the run as no-data.

Same class in EVERY collection runner.

**Sweep methodology (idempotency-safe):**

Built a cross-reference sweep that:
1. Parses `runners.py` dispatch table: `{func_name: new_run_num}`
2. For each runner file, walks every `def _run_*(...)` function
3. Inside each function body, finds every `save_npy/save_embedding/run_generation/print_turn_result/_log_trial/...` call
4. Extracts the hardcoded int
5. Compares against the expected new-id from dispatch table

Where integer didn't match, remapped via `OLD_TO_NEW`.

**Results — 458 lines touched across 3 runner files:**

| File           | Lines changed |
|----------------|---------------|
| runners_p1.py  | 385  (includes prior sweep accumulation) |
| runners_p2.py  | 29   |
| runners_p3.py  | 44   |

Touched calls: `save_npy`, `save_embedding`, `run_generation`,
`print_turn_result`, `_log_trial_start`, `_log_trial_end`,
`_get_trials_for_condition`. Each instance had its integer arg flipped
to the correct new id. String literals like `"Q28"` (for trial-log
tags) flipped to `"Q0023"` with R/Q prefix recomputed per the ≤21 rule
(old Q28 confound isolation = new 23 = Q0023, same Q).

**BUG-DOUBLE-REMAP caught during verification.**

The sweep was aggressive — any int in OLD_TO_NEW keys got remapped.
But I had manually fixed `runners_p3.py` validation run (41→33) in
an earlier step of the same turn. The sweep saw `33` (which is also
an OLD id — 33→42) and double-remapped it to 42.

Caught by cross-referencing runner bodies against dispatch table
post-sweep. Five sites in `_run_validation_set` reverted from 42 back
to 33: `save_npy`, `save_embedding`, `run_generation` (×2),
`print_turn_result`, `_get_trials_for_condition`.

Lesson: aggressive int-remap is unsafe when the source values include
already-migrated ones. Going forward, runs a dispatch-table audit AFTER
the sweep to catch double-remaps.

**GRANGER MATRIX FIX (from earlier this turn).**

`analysis.py:308` — `_build_granger_matrix(hidden_dir, 19, model_name)`
passed OLD 19. Granger looks for hidden-state files indexed by run_num
in glob patterns. 19 → new 1; `[19, 26]` → new `[1, 3]`. Without this,
Granger analyses (new Runs 47, 48) read from random-patching and
random-wrong-run files silently.

**E_T RECOVERY PASS FIX (from earlier this turn).**

`runners_p1.py` _run_et_recovery function had 5 hardcoded 19s:
- 2× `_se(hidden, hidden_dir, 19, ...)` saver calls
- 3× f-string filename patterns: `f"R19_{mn}_ct_global_mean.npy"`,
  `f"R19_{mn}_et_global_mean.npy"`, `f"R19_{mn}_pass4_ok.stamp"`

The f-strings evaded my earlier literal sweep because that regex
required `R\d{2}_\w+\.(csv|json)` — these are `.npy` and `.stamp` with
interpolated `{mn}`. Flipped to `R0001_{mn}_*`.

**Verification:**
- All 30 `.py` files compile.
- Dispatch↔runner-body cross-reference audit: 0 mismatches.
- Spot-checked function bodies: each save_npy/run_generation int-arg
  matches its dispatch-assigned new run_num.
- Validation run (new 33) correctly saves `R0033_*` files and tags
  trial logs with "Q0033".
- Granger A (new 47) correctly reads from Run 0001 (was R19) hidden
  states. Granger B (new 48) reads from [Run 0001, Run 0003].

**STILL OPEN:**
- Possible analogous bugs in analysis-side file reads (non-runner
  modules). Not systematically swept yet.
- D1 (variable names) — still deferred.

**Deployment:** install over 0.79.4.6. Restart Flask.

---

### 0.79.4.6 — April 2026  (HOTFIX — dashboard backend/JS key mismatches)
`analysis.py`, `export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.5: RESIDUAL OLD-ID REFERENCES IN RUNTIME STATE.**

Systematic inspection continued after 0.79.4.5. Found runtime-state
references that survived every prior sweep because they're neither
dispatch comparisons, dict literals, nor prose — they're payload
KEYS in the JSON response contract between backend and frontend.

**BUG-LOAD-RUN28 (function name + 1 caller stale).**

`analysis._load_run28_condition_map` — loads the Run 0023 (was R28)
confound condition map from CSV. Function name preserved old ID, even
though the code inside correctly reads `Q0023_introspection.csv`. Call
site at line 1199 matched the stale name.

Renamed both def and call to `_load_run23_condition_map`. Cosmetic
per-module, but matches Kevin's "everything" directive.

**BUG-ROWS-BASE (runtime filter, silent wrong-exclusion).**

`analysis.py:1201`:
```python
rows_base = [r for r in rows_3way if r['run_num'] != 28]
```

This filter EXCLUDES the confound-isolation run so `rows_base` can
serve as "everything except Run 0023" for the H16 decomposition. But
runtime data has `run_num = 23` (new ID), not 28. The filter excluded
NOTHING — rows_base included Run 0023 rows, polluting the H16 base
decomposition with confound-condition rows.

Subtle failure: the analysis still runs, produces numbers, but the
numbers are contaminated. The decomposition is quietly biased.

Fixed: `!= 28` → `!= 23` (matches the paired `rows_23` filter one
line above that was corrected in 0.79.4.3).

**BUG-BACKEND-FRONTEND-KEY-MISMATCH (dashboard — Run 0003 grid broken).**

The `/run_detail` endpoint emits different JSON shapes for different
run types. For the temperature-grid run (was R26, now 0003), the
backend emits these keys:
- `run26_grid` / `run26_temps` / `run26_conds`
- `run26_cells_done` / `run26_cells_total`
- plus JS module-level `_TEMPS_26` / `_CONDS_26`

The JS consumer at line ~5330 reads the same `run26_*` keys. After
0.79.4.5 I renamed `run26_cells_done` on the BACKEND side (to `run3_*`)
but missed the JS consumer — half-rename broke the dashboard grid
silently. User-facing symptom: click Run 0003, grid panel renders
empty.

Fix (this ship): rename all 7 key names on BOTH backend emit side and
JS consume side. Now consistent `run3_*` naming:
- `run26_grid`        → `run3_grid`
- `run26_temps`       → `run3_temps`
- `run26_conds`       → `run3_conds`
- `run26_cells_done`  → `run3_cells_done`
- `run26_cells_total` → `run3_cells_total`
- `_TEMPS_26`         → `_TEMPS_3`
- `_CONDS_26`         → `_CONDS_3`

**BUG-RUN19-KEY (dashboard — Run 0001 triple-variant detail broken).**

Same class. Backend emits:
```python
'run19': {'abliterated': n_abl, 'base': n_base, 'instruct': n_inst, ...}
```
when reporting status for the null-trivariant run. JS consumes `d.run19`
at 3 sites. Renamed everywhere to `run1` (new canonical ID).

**STALE DOCSTRINGS — `export_flask.py` endpoints.**

Two endpoint docstrings had mixed old/new IDs:
- `/report`:       "Run 0051/46/47 outputs"   → "Run 0051/0046/0050 outputs"
- `/report_status`: "Run 0051 or 47 JSON"     → "Run 0051 or 0050 JSON"

(46 and 47 were BARE 2-digit — ambiguous between "old 46 = per-condition
R = new 44" and "new 46 = MLP decomposition". Context: "report outputs"
refers to pooled + cross-temp synthesis. Old 46 = per-cond R, old 47 =
cross-temp. Old→new → {44, 50}. So docstring was mostly-new-IDs-in-
4-digit-form except those two stragglers in 2-digit. Flipped to canonical
4-digit new.)

**MULTI-CONDITION COMMENT (`export_flask.py:1985`).**

`# ── Multi-condition runs (Run 0002, 28, 29, etc.) ──`. The 2-digit
`28, 29` were OLD IDs. New MC set includes new 2, 23, 24 (the three
the comment meant). Flipped comment to `(Run 0002, 0023, 0024, etc.)`.

**VERIFICATION METHODOLOGY:**

This ship caught a 4th reference category I hadn't explicitly tracked:

```
J1.  Backend JSON response KEYS (emitted to frontend JS)          [0.79.4.6]
J2.  Frontend JS module constants named after run IDs              [0.79.4.6]
J3.  Backend→frontend contract: same key used on both sides       [0.79.4.6]
```

Category J is insidious because:
- Single-sided changes (rename backend only or JS only) look fine on
  compile — both files still parse — but create runtime contract
  mismatches that only show up when the user clicks the specific UI
  element.
- No Python-level type checking would catch it.
- Search tools need to grep BOTH `.py` (backend) and the embedded JS
  block (which is in triple-quoted strings inside `.py`).

Going forward, all renames of backend response keys must be
searched-for in the frontend JS text as well.

**FULL STATUS — POST 0.79.4.6:**

Categories at this point, fully swept:
```
A1. Dispatch comparisons                             [0.79.4.1]
A2. Module constants (RUN_NUM=N)                     [0.79.4.1]
A3. Kwarg defaults                                   [0.79.4.1]
A4. Kwarg call args                                  [0.79.4.2]
A5. Set/list literal IDs                             [0.79.4.2]
A6. List literal IDs                                 [0.79.4.2]
A7. 'in' membership tests                            [0.79.4.1]
A8. Dict literal KEYS (pass 1)                       [0.79.4.4]
A9. Dict literal KEYS (pass 2: scanner/runners_core) [0.79.4.5]
B1. Filename literals (quoted)                       [0.79.4.0]
B2. F-string patterns                                [pre-existing clean]
B3/4/5/6. Prose in comments/docs/strings             [0.79.4.3]
C1. DualKey-wrapped dispatch dicts                   [0.79.4.0]
C2. Raw dict.get(OLD_INT)                            [0.79.4.3]
D2. CSV run_mode == 'N' raw-string filters           [0.79.4.3]
D3. r['run_num'] == N runtime dict filters           [0.79.4.3+4.6]
E1. JS frontend constants (standalone)               [0.79.4.2]
J1/2/3. Backend↔frontend key contract                [0.79.4.6]
F1. CSV run_mode data                                [migrate_run_ids]
G1. JSON content data                                [migrate_run_ids]
```

Still unswept:
- `D1. Variable names (r21_csv, q33_path)` — cosmetic only. Deferred.
- Any edge cases in `runners_p2.py`, `runners_p3.py`, `vault.py`,
  `ui.py`, `setup.py` not yet deeply inspected.

**Verification:**
- All 30 `.py` files compile.
- 56/56 dispatches clean.
- Backend→JS key audit: no `run19_*`, `run20_*`, `run26_*`, `run28_*`,
  etc. — all flipped to new-canonical names.

**Deployment:** install over 0.79.4.5. Restart Flask. Hard-refresh
dashboard (Ctrl+Shift+R) — JS cache will have the old `run26_*` /
`run19` names in browser.

---

### 0.79.4.5 — April 2026  (HOTFIX — second int-keyed dict pass)
`runners_core.py`, `runners_p1.py`, `scanner.py`, `analysis.py`,
`run42_layer_isolation.py`, `export_flask.py`, `migrate_data_keys.py`,
`CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.4: ADDITIONAL INT-KEYED DICTS + TOOLTIP + DOC STALE.**

After 0.79.4.4 caught seven int-keyed dicts, a second systematic pass
found five more that were also landmines. The full enumeration
methodology from 0.79.4.4 was extended to cover scanner + runners_core.

**BUG-ET-SYSTEM-PROMPTS (catastrophic, would KeyError at module load).**

`runners_core._ET_SYSTEM_PROMPTS` was keyed by OLD ids `{1,2,3,4,5,6,7,8,9,
15,16,17,20}`. Values referenced `FRAMING_SYSTEM_PROMPTS[15/16/17]` —
but 0.79.4.4's prior fix had flipped FRAMING's keys to `{13,14,15}`.

**Boot-time failure:** `FRAMING_SYSTEM_PROMPTS[16]` raises KeyError the
moment any module imports `runners_core`. That's `start_here.py`,
`runners.py`, every one of the `runners_p*.py` files — effectively
every collection entrypoint would have crashed on import in 0.79.4.4.

This bug was introduced BY the 0.79.4.4 fix and would have bricked
every collection run. Caught in the systematic follow-up sweep before
any deployment could hit it.

Fix: `_ET_SYSTEM_PROMPTS` dict rebuilt with new IDs matching the 13
actual ET-recovery source runs (`{1, 2, 4-15}`). FRAMING lookups now
use `[13/14/15]` (the new priming keys). Each entry annotated with
its old-ID origin for migration cross-ref.

**BUG-MC-RUNS (scanner — dashboard completion calculations wrong).**

`scanner._MC_RUNS` and `_MC_TURNS` were keyed by old IDs `{20,28,29,30,
31,37,38,39,41,43,44}` — the 11 multi-condition runs with partition
columns. Scanner uses these dicts to compute per-run completion
percentages for the dashboard (e.g. "Run 0030: 60/75 trials — 80%").

Post-renumber, runtime passes new IDs. Lookup `_MC_RUNS[20]` returns
the OLD Run 20's config (`'temperature_condition'` with 5 values), but
new 20 is Two-Instance which uses `'instance'` column. Every MC run
shows wrong progress bar / wrong condition list / wrong expected count.

Fix: bijective remap of all 11 keys:
```
old 20 (robustness)         → new  2
old 28 (confound isolation) → new 23
old 29 (persistence)        → new 24
old 30 (two-instance)       → new 20
old 31 (coherence levels)   → new 21
old 37 (entropy shape)      → new 37  (fixed point)
old 38 (condition transfer) → new 36
old 39 (output similarity)  → new 35
old 41 (held-out valid.)    → new 33
old 43 (priming length)     → new 25
old 44 (contradiction)      → new 26
```

Applied to both `_MC_RUNS` and `_MC_TURNS` dicts. Each entry annotated.

**BUG-IMPOSSIBLE-PROMPTS (runtime dispatch → wrong prompts).**

`runners_p1._IMPOSSIBLE_PROMPTS` was keyed by old `{12, 13, 14}` — the
impossibility cluster. Runtime passes new `{29, 30, 31}`. Lookup
`_IMPOSSIBLE_PROMPTS[29]` raises KeyError mid-collection. Same class
of landmine as the prior fmt/FRAMING fixes.

Fix: keys flipped to new `{29, 30, 31}` (IMPOSSIBLE_CONSTRAINED /
UNCONSTRAINED / EPISTEMIC_IMPOSSIBLE respectively).

**BUG-REQUIRED-FIELDS (analysis resume-gate — silent false-negative).**

`analysis.REQUIRED_FIELDS` was keyed by old `{33, 34, 40, 56}`. This
dict is the completeness gate — when scanner resumes mid-analysis, it
checks that the previously-written JSON has the required field list
for this run before marking "done". Old keys → wrong field lists
checked → gate either false-positives (missing field, passes check)
or false-negatives (wrong dict entry used for new run).

Fix: keys flipped to new `{42, 43, 51, 46}` via the bijection.

**STALE DOCSTRING — `run42_layer_isolation.py:49`.**

Module-level header said `RUN_NUM: 42  CSV prefix: Q (run_num > 17)
Output: Q42_layer_isolation.csv`. Everything stale:
- `RUN_NUM: 42` — should be 18 (fixed in code at 0.79.4.1, doc lagged)
- `CSV prefix: Q` — new 18 is ≤ 21, should be R
- `Output: Q42_layer_isolation.csv` — actual filename is now
  `R0018_layer_isolation.csv`

Fix: header rewritten with new values + note on prior numbering.

**HTML TOOLTIP — E_t Recovery button.**

`export_flask.py:3138` dashboard button tooltip said "E_t Recovery —
run base-model pass to extract E_t embeddings for Runs 1–9, 15–17".
These are OLD IDs. New equivalents are contiguous — Runs 0004–0015 (old
1–9 → new 4–12, old 15–17 → new 13–15).

Fix: tooltip updated to `Runs 0004–0015`.

**LEGACY FLAG — `migrate_data_keys.py`.**

0.76→0.77 data migration utility. References OLD (pre-renumber)
filenames. Added prominent header note: run this BEFORE
`migrate_run_ids.py` if dealing with 0.76-era data. Post-renumber data
won't match these old keys; running in wrong order is a no-op.

**WHAT THIS HOTFIX DOES NOT TOUCH:**
- Variable names (`r21_csv`, `r34_file`, `q33_path`) — still deferred,
  cosmetic only, values correct.
- CHANGELOG historical entries — preserved per Kevin directive.
- `copy_hidden_r20_r26.py` filename — keeping since Kevin has script
  path muscle-memory. Contents already fixed in 0.79.4.3 with dual-
  prefix fallback.

**Verification:**
- All 30 `.py` files compile.
- `_ET_SYSTEM_PROMPTS` loads without KeyError.
- `FRAMING_SYSTEM_PROMPTS` + `_IMPOSSIBLE_PROMPTS` + `_TEMP_INDEPENDENT_RUNS`
  all have new-ID keys verified at runtime.
- `_MC_RUNS` / `_MC_TURNS` keys remapped, all 11 entries present.
- 56/56 dispatches, 0 prereq errors (unchanged — structural checks
  invariant).

**Deployment:** install over 0.79.4.4. Restart Flask. Any instances of
0.79.4.4 that were launched (none expected, since module import would
have crashed) should be stopped and replaced.

---

### 0.79.4.4 — April 2026  (HOTFIX — int-keyed dict sweep)
`start_here.py`, `analysis.py`, `export_flask.py`, `export_stats.py`,
`runners_p1.py`, `runners_prompts.py`, `CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.3: INT-KEYED DICT LITERALS MISSED BY PRIOR SWEEPS.**

Every prior sweep hit comparisons and `.get(N)` calls. This hotfix
catches **dict LITERALS** with old integer run IDs as keys. These are
defined at module load time and consumed by runtime code that passes
NEW run_num. Lookup `d[new_id]` against an OLD-keyed dict either
silently returns None (if `.get()`) or raises KeyError, and even when
the OLD key happens to match (because every new int 1-56 is also an old
int 1-56), it returns the WRONG value for a different run.

**BUG-TEMP-INDEP (catastrophic, silent).**

`start_here._TEMP_INDEPENDENT_RUNS` was keyed by OLD IDs `{19, 20, 26}`
(Phase A temp-indep runs in old numbering). `_copy_temperature_independent_runs`
iterates `.items()` and passes the key as `run_num` into
`run_prefix(run_num)` + `f"{pfx}{run_num:04d}_*"` to build a glob
pattern for copying hidden-state files across temperatures.

Post-migration, files on disk are `R0001_*`, `R0002_*`, `R0003_*` (new
canonical). But iteration yields `run_num = 19`, so glob becomes
`R0019_*`, which either matches nothing (if migration complete) or
matches random-patching files (new run 0019). Copy silently no-ops
for every non-deterministic temperature.

Concrete failure mode: Kevin starts collecting at T=0.2. `_copy_temperature_independent_runs`
runs, reports "0 files copied" silently. `_load_quadruplets` at T=0.2
finds no S_t files for Runs 0001/0002/0003 → zero-row contribution to
the OLS decomposition. Silent biased fit for every temp > 0.0.

Fix: keys flipped to new `{1, 2, 3}`. Comments reference prior IDs
(`# was R19`, `# was R20`, `# was Q26`) for migration cross-ref.

**BUG-PATCHING-CONFIG (dashboard-facing).**

`export_flask._PATCHING_RUNS` was keyed by old `{21, 42, 53}`. Dashboard
calls `_PATCHING_RUNS[run_num]` where `run_num` is now new 17/18/19 for
the patching cluster. Old-keyed dict returns `KeyError`; `if run_num in
_PATCHING_RUNS` returns False; patching dashboard panels render empty
or error.

Fix: keys flipped to new `{17, 18, 19}`. Comment notes old→new mapping
on each key.

**BUG-PROMPT-DISPATCH (runtime dispatch → wrong prompts).**

Two prompt-template dicts keyed by OLD run IDs:

- `runners_p1.fmt = {6, 7, 8, 9}` — arithmetic prompt format strings.
  Runtime passes new `run_mode` in range 9-12 (arithmetic A-D). Lookup
  `fmt[9]` returns the OLD arithmetic-verbal template (old 9 →
  "{q}"). Lookup `fmt[11]` raises KeyError, crashing mid-collection.
  When it "works" it feeds the WRONG prompt to the model. Every
  arithmetic trial silently runs against the wrong framing.

  Fix: keys flipped to new `{9, 10, 11, 12}`. Also: `run_mode == 9`
  comparison (3 sites) flipped to `run_mode == 12` (new arithmetic-D
  uses long output).

- `runners_prompts.FRAMING_SYSTEM_PROMPTS = {15, 16, 17}` — priming
  framing prompts (neutral / cooperative / resistant). Runtime passes
  new 13/14/15. Same silent-wrong-prompt issue.

  Fix: keys flipped to new `{13, 14, 15}`.

**BUG-LABELS (cosmetic display, but misleading).**

Three display-label dicts with OLD keys:

- `analysis._LABELS` (9 entries) — per-run printout labels.
- `analysis._LABELS56` (9 entries) — same pattern in second location;
  renamed to `_LABELS46` (56→46 reflects Run 56 → new 46). All
  references updated via sed.
- `export_stats._COND_LABELS` (6 entries) — paper figure labels for
  per-condition similarity plots.

All three keyed by OLD `{3,4,5,15,16,17,19,26,28}` or subset. Runtime
passes new IDs, `.get(rn, f"Run {rn}")` falls through to the default,
which STILL prints correctly (with the new ID) so the bug was invisible
— but the human-readable label was lost. Figures and tables in the
paper would have shown "Run 0006" instead of "Run 0006 — Introspection
A (direct)".

Fix: keys flipped to new IDs via explicit OLD→NEW mapping. 24 dict-key
remaps across 3 files.

**VERIFICATION METHODOLOGY (systematic, not random this time):**

Kevin asked for systematic enumeration, not random find-fix. The
enumeration categories are:

```
A1.  Dispatch comparisons   (if run_num == N)           [0.79.4.1 fixed 113]
A2.  Module constants       (RUN_NUM = N)                [0.79.4.1 fixed 2]
A3.  Kwarg defaults         (def f(run_label=N))         [0.79.4.1 fixed 2]
A4.  Kwarg call args        (f(ref_run=N))               [0.79.4.2 fixed 3]
A5.  Set literal IDs        (_GEN_RUNS = {1,...,N})      [0.79.4.2 fixed 2]
A6.  List literal IDs       (SOURCE_RUNS = [N,...])      [0.79.4.2 fixed 3]
A7.  'in' membership tests                                [0.79.4.1 embedded]
A8.  Dict literal KEYS      ({N: ...})                    [0.79.4.4 fixed 7]  ← this ship
B1.  Filename literals      ("RNN_foo.csv")              [0.79.4.0 fixed 218]
B2.  F-string patterns      (f"R{run_num:04d}_...")      [pre-existing]
B3.  Label strings          (f"Run {n:04d}")             [pre-existing]
B4/5/6 Prose in comments/docs                             [0.79.4.3 fixed 862]
C1.  Dispatch dicts (DualKeyShim-wrapped)                 [0.79.4.0 shim]
C2.  Raw dict.get(OLD_INT)  (silent wrong-read)           [0.79.4.3 fixed 9]
D1.  Variable names r21_csv (cosmetic)                    [DEFERRED]
D2.  run_mode CSV filters == 'N' (raw string)             [0.79.4.3 fixed 8]
D3.  r['run_num'] == N runtime dict filters               [0.79.4.3 fixed 4]
E1.  JS frontend constants                                [0.79.4.2 fixed 2]
F1.  CSV run_mode column (in data)                        [migrate_run_ids.py]
G1.  JSON content (in data)                               [migrate_run_ids.py]
```

**STILL OPEN / NOT TRACKED:**
- D1 (variable names) — cosmetic only, values are correct.
- Any edge-case dict literals in files not yet sampled.
- Unit tests — there are none in this codebase.

**Verification:**
- All 30 `.py` files compile.
- 56/56 dispatches, 0 prereq errors.
- `_TEMP_INDEPENDENT_RUNS` keys verified `[1, 2, 3]` at runtime.
- `fmt` dict lookup simulated: `fmt[9]` → "Solve step by step: {q}"
  (arithmetic-A slow prompt, correct for new run 9 = old run 6).
- `_PATCHING_RUNS[17]` → patch_mode config (correct for activation
  patching, was old 21).

**Deployment:** install over 0.79.4.3. Restart Flask. Backfill any
temperature conditions that were collected with broken _TEMP_INDEP by
running `python copy_hidden_r20_r26.py --apply` (script has been
renumbered in 0.79.4.3 with legacy-filename fallback).

---

### 0.79.4.3 — April 2026  (HOTFIX — systematic sweep)
`analysis.py`, `cartography.py`, `copy_hidden_r20_r26.py`, `export_flask.py`,
`export_stats.py`, `start_here.py`, plus all renumbered code from earlier hotfixes.
`CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.0/4.1/4.2: SYSTEMATIC SWEEP.**

After two turns of "find and fix" hotfixes, switched to explicit reference-
type enumeration. Found three more load-bearing bug classes that DualKey
shim was silently masking.

**Bug class #1 — raw-string run_mode filters (8 fixes):**

`r.get('run_mode', '') == '44'` style CSV filters comparing strings to
OLD integer-string forms. DualKey doesn't intercept string→string
comparisons. Pre-migration rows had `'44'` (old), post-migration have
`'0026'` (new). Filter misses entirely.

Fixed to dual-accept both forms via `in ('new_4d', 'old_int')`:
- `analysis.py` (5 sites: runs previously filtered as 19, 29, 31, 38, 43, 44)
- `export_stats.py` (3 df-column comparisons: runs 19, 20, 22)

**Bug class #2 — `.get(OLD_INT, ...)` on renumbered dicts (9 fixes):**

`RUN_CSV.get(19, '')` silently returns `RUN_CSV["0019"]` = random-patching
filename (via DualKey shim), when the code WANTS new 1 (null+hidden).
DualKey transparently padded the int → wrong string post-renumber.

Fixed to new IDs:
- `RUN_CSV.get(44)` → `get("0026")` (contradiction recovery)
- `RUN_CSV.get(31)` → `get("0021")` (coherence levels)
- `RUN_CSV.get(43)` → `get("0025")` (priming length)
- `RUN_CSV.get(29)` → `get("0024")` (persistence)
- `RUN_CSV.get(38)` → `get("0036")` (condition transfer)
- `_RC19.get(19)` → `get("0001")` (null+hidden)
- `status.get(19)` / `status.get(48)` — remapped with legacy fallback
- `RUN_MAP[33]` → `RUN_MAP["0042"]` in docstring example

**Bug class #3 — integer comparisons on `r['run_num']` (4 fixes):**

`r28_entry = next(r for r in per_run_results if r.get('run_num') == 28)` —
`per_run_results` is populated at runtime with NEW IDs (since dispatch
is renumbered), so filtering by OLD integer 28 finds nothing. Old 28
(C_t confound) is new 23.

Fixed in analysis.py:
- `r28_entry` / `r28_rows` / `rows_28` / `n_run28_rows` / `run28_correction`
  → all renamed to `r23_*` / `run23_*` with `== 23` comparisons
- `rows_28_cond` → `rows_23_cond` (Run 0023 confound path)

**copy_hidden_r20_r26.py — updated in place:**

COPY_SPEC flipped: old keys 20, 26 → new 2, 3. Filename prefixes updated
to R0002_* / R0003_* (both R-prefix now — new 3 is ≤ 21 so follows
R/Q rule). Added `legacy_prefix` fallback so the script still finds old
R20_*, Q26_* files if run before `migrate_run_ids.py`. Usage docstring
references old IDs in parens for clarity: "Runs 0002 and 0003 (was
R20, Q26)".

**export_stats integer-list iteration:**

`for r in [12, 13, 14]` iterated old impossibility-cluster IDs. New IDs
are 29, 30, 31. Fixed in H39 per-run breakdown loop. Also flipped the
comparison to zfill(4) on the iteration variable so it matches the
canonical 4-digit CSV column value.

**Prose sweeps (829 substitutions total across ships .1/.2/.3):**

- 751 first prose pass (comments/docstrings/ui.section/error messages)
- 32 colon-followed pattern catch ("Run 19:" style)
- 50 range + list prose ("Runs 1-18", "Runs 33→34→46", "Runs 15/16/17")
- 29 lowercase + singular ("run 29", "run 30")

Regex tuned to only match 1-2 digit numbers (already-4-digit doesn't
match), so prose sweep is idempotent and safe to re-run.

**DELIBERATELY SKIPPED:**
- Local variable names (`r21_csv`, `r34_file`, `q33_path`) — cosmetic
  only; VALUES are correct, names are dead references. Would require
  scope-aware renaming to do safely. Deferred.
- CHANGELOG.md — historical record, preserves old IDs in its prose
  per Kevin's direction.

**Verification after all three hotfixes:**
- All 30 `.py` files compile.
- 56/56 dispatches clean, 0 prereq errors.
- 55/55 hypothesis references resolve cleanly.
- No residual `== '19'` / `== '33'` / etc. raw-string filters.
- No residual `RUN_CSV.get(N)` with N in old-ID range ≠ new-ID.
- No `r\.get\('run_num'\) == OLD_N` patterns remaining.
- `copy_hidden_r20_r26.py` compiles; dual-prefix fallback works.

**Deployment:** install over 0.79.4.2. Restart Flask.

---

### 0.79.4.2 — April 2026  (HOTFIX)
`analysis.py`, `graft_patching.py`, `run42_layer_isolation.py`,
`export_flask.py`, `dependency_map.py`, `CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.0/0.79.4.1: INTEGER LIST REMAPS.**

0.79.4.1 caught the `if run_num == N` dispatch branches but missed
several classes of hardcoded integer run-ID references that use
data-structure literal syntax.

**Critical fixes:**

*SOURCE_RUNS lists (analysis.py).* These drive `_load_quadruplets`
which is the primary data path for every decomposition/permutation
analysis. Old IDs would look for hidden-state files that no longer
exist at those IDs.
- `SOURCE_RUNS_3WAY = [3,4,5,15,16,17,19,26,28]` → `[6,7,8,13,14,15,1,3,23]`
- `SOURCE_RUNS_2WAY = [1,2,6,7,8,9,20]` → `[4,5,9,10,11,12,2]`
- `SOURCE_RUNS_VALIDATION = [41]` → `[33]`

*ref_run patching calls.* graft_patching and run42_layer_isolation
both call `_load_reference_states(hidden_dir, ref_run=3, ...)`. Old 3
(Introspection A, all-layers ref source) is new 6. Without this fix,
patching and layer-isolation runs would fail to find their reference
hidden-state files.
- `graft_patching.py` (2 call sites): `ref_run=3` → `ref_run=6`
- `run42_layer_isolation.py` (1 call site): same.

*Dependency map (dependency_map.py).* 165 list-key substitutions —
every `data_runs`, `comparison_runs`, `upstream_runs` list of old
integer run IDs remapped via the OLD_TO_NEW bijection. The paper
pipeline and hypothesis-tracking UI both read this map for run
membership tests.

*Secondary _GEN_RUNS in export_flask (2 duplicate sites).* Replaced
hardcoded old-integer set literal with `set(range(1, 41))` —
post-renumber, data-collection runs are contiguous 1-40.

*Frontend ET_RUNS JS list (export_flask.py:5757).* Dashboard JS
const `ET_RUNS=[1,2,3,4,5,6,7,8,9,15,16,17,20]` → `[1,4,5,6,7,8,9,10,11,12,13,14,15]`.
Drives the ET-mode config UI.

*`is_et_run` hardcoded set.* In Flask `/run_detail` response (line
2158): `{1,2,3,4,5,6,7,8,9,15,16,17,20}` → `{1,4,5,6,7,8,9,10,11,12,13,14,15}`.
Used by the per-run popup to show the ET-mode column.

**What still might be lurking:**

- Comments and docstrings referencing old run numbers by name (Run 19,
  Run 33, etc.) — cosmetic, not load-bearing. Not swept.
- Variable names like `r21_csv`, `r34_file` — local variables. Values
  correctly point at new canonical filenames. Names cosmetic only.
- UI label text in console `ui.section(...)` calls — some reference
  old names. Cosmetic.

**Verification:**
- All 30 `.py` files compile.
- SOURCE_RUNS_3WAY/2WAY — spot-checked against the run descriptions in
  RUN_MAP to confirm semantic mapping is correct (e.g. new 3 is the
  temp-grid run that was old 26, correctly in the 3WAY set).
- `ref_run=6` → RUN_MAP["0006"] = Introspection A. Correct (was R3).
- Frontend ET_RUNS list — matches `scanner._ET_RECOVERY_RUNS` new IDs.

**Deployment:** install over 0.79.4.1. Restart Flask.

---

### 0.79.4.1 — April 2026  (HOTFIX)
`runners.py`, `runners_p1.py`, `graft_patching.py`, `run42_layer_isolation.py`,
`analysis.py`, `start_here.py`, `scanner.py`, `export_stats.py`, `export_flask.py`,
`CHANGELOG.md`, `HANDOFF.md`

**HOTFIX FOR 0.79.4.0: DISPATCH BRANCH SWEEP.**

0.79.4.0 renumbered the data structures but left every `if run_num == N` /
`elif run_num == N` dispatch branch pointing at OLD integer run numbers.
Every runner call would have executed the wrong function. This hotfix
closes that gap.

**113 dispatch substitutions across 9 files:**

| File                         | Subs |
|------------------------------|------|
| `runners.py`                 | 51   |
| `analysis.py`                | 17   |
| `start_here.py`              | 15   |
| `scanner.py`                 | 7    |
| `export_stats.py`            | 7    |
| `runners_p1.py`              | 6    |
| `export_flask.py`            | 5    |
| `graft_patching.py`          | 3    |
| `run42_layer_isolation.py`   | 2    |

All `run_num`, `run_mode`, `RUN_NUM` integer comparisons remapped via
the OLD_TO_NEW bijection:
- `if run_num == 21` → `if run_num == 17` (activation patching)
- `elif run_num == 42` → `elif run_num == 18` (layer isolation)
- `if run_num == 48` → `if run_num == 16` (E_t meta-run)
- `elif run_num == 28` → `elif run_num == 23` (confound isolation)
- ...etc.

**Module constants also fixed:**
- `graft_patching.RUN_NUM = 21` → `17`
- `run42_layer_isolation.RUN_NUM = 42` → `18`

**Dataframe column filters (patterns the regex deliberately excluded
because they use a different semantic — column value comparison):**
- `export_stats.py` causal-patching figure call — swapped from
  `df[df['run_mode']==21]` to `df[run_mode_mask(df['run_mode'], 17)]`
  (and `42` → `18`). Uses the dual-accept mask helper so both old
  integer-string rows and new 4-digit canonical rows match correctly.

**Kwarg defaults also remapped:**
- `analysis._run_per_condition_r(run_label=46, ...)` → `run_label=44`
  (new 44 = per-condition R; old 46 is now MLP decomposition).
- Call site `run_label=49` → `run_label=45` (new 45 = fixed-dim
  per-condition R).

**Sweep methodology:**
Used a targeted regex with negative lookbehind to exclude quoted
column-lookup patterns (`df['run_mode']==N`) from the `run_num == N`
sweep. Those needed separate treatment because `run_mode` column values
are strings post-migration, not ints.

**Verification:**
- All 30 `.py` files compile.
- Dispatch branches spot-checked: `elif run_num == 23: _p2._run_confound`
  correctly fires the confound function for the run that was old Q28 and
  is now new 0023.
- Module constants: `RUN_NUM = 17` in graft_patching (was 21), `RUN_NUM =
  18` in run42_layer_isolation (was 42).

**Note on what this ship does NOT catch:**
Some dispatch paths use the `RUN_MAP[run_num][0]` module-name lookup to
decide which function to call. Those still work because `RUN_MAP` is
keyed by new IDs via `DualKeyRunDict`, so `RUN_MAP[17]` returns the
activation-patching entry. The sweep only needed to catch direct
`run_num == N` branches in dispatch code.

**Deployment:** install over 0.79.4.0. Restart Flask.

---

### 0.79.4.0 — April 2026
`cartography.py`, `start_here.py`, `scanner.py`, `export_flask.py`, `analysis.py`,
`export_stats.py`, `dependency_map.py`, `cleanup_analysis.py`, `purge_stale_analysis.py`,
`report.py`, `runners_p1.py`, `runners_p2.py`, `runners_p3.py`, `migrate_data_keys.py`,
`read_r45.py`, `graft_patching.py`, `migrate_run_ids.py`,
`CHANGELOG.md`, `HANDOFF.md`

**RUNS RENUMBERED TO EXECUTION-ORDER POSITIONS.**

Fourth functional ship of the 0.79.x arc. Every run is renumbered so
its ID equals its position in EXECUTION_ORDER. Bijective remap of
56 runs — not a zero-pad. Old Run 19 becomes new 0001 (because it
runs first). Old Run 1 becomes 0004 (fourth in execution). Old Run 56
becomes 0046 (MLP decomposition moves into position with the analysis
chain). See the renumber map document for the full bijection.

**Why:** the prior numbering was the order runs were designed in, not
the order they execute. Filenames, CSVs, JSONs, paper section refs —
everything — carries ID baggage from historical design order. After
renumber, a run's ID tells you where it sits in the pipeline at a glance.

**Summary of the remap:**
```
Phase A (temp-indep)        old 19, 20, 26     → new 0001–0003
Phase B (ET-source)         old 1–9, 15–17     → new 0004–0015
Phase G (E_t meta-run)      old 48             → new 0016
Phase D (patching)          old 21, 42, 53     → new 0017–0019
Phase EFC (slow→fast)       old 30,31,23,…,11  → new 0020–0040
Per-temp chain              old 45,33,34,…,56  → new 0041–0046
Per-temp indep              old 25, 27, 32     → new 0047–0049
Pooled                      old 47, 40         → new 0050–0051
Cross-model                 old 50, 51, 52     → new 0052–0054
Output                      old 54, 55         → new 0055–0056
```

**Data structures flipped:**
- `cartography.RUN_CSV` — string keys, renumbered, new filename convention
  with R/Q phase prefix reflecting NEW run number (R for 0001-0021, Q for
  0022+). Each entry annotated with `# was Rold_id` for migration cross-ref.
- `cartography.ANALYSIS_JSON` — same treatment, 16 analysis JSON filenames
  remapped.
- `cartography.TEMP_INDEP_RUNS` — `{1, 2, 3}` (was `{19, 20, 26}`).
- `start_here.RUN_MAP` — fully rewritten with phase-annotated structure.
- `start_here.EXECUTION_ORDER` — clean linear `0001..0056` sequence.
- `start_here._PREREQS` — 10 prereq edges remapped via old→new bijection.
- `start_here._ANALYSIS_RUNS` — `{41..54}` (was scattered).
- `start_here._GEN_RUNS` — `{1..40}` (all data-collection runs, incl. meta).
- `scanner._ET_RECOVERY_RUNS` — new IDs with old-ID annotations.
- `export_flask._POOLED_RUNS` / `_CROSS_MODEL_RUNS` / `_OUTPUT_RUNS` —
  renumbered to 50-51 / 52-54 / 55-56 respectively.

**Infrastructure retained from earlier arc work:**
- `DualKeyRunDict` / `DualKeyRunSet` classes — accept int or string keys
  transparently. With renumbered data in place, these still let legacy
  integer call sites work, just pointing at new-canonical values.
- `run_mode_matches` / `run_mode_mask` / `run_mode_mask_any` helpers —
  dual-accept filters for CSV read sites.

**Hardcoded filename literal sweep — 218 substitutions across 14 files:**

`analysis.py` (28), `dependency_map.py` (66), `export_stats.py` (47),
`start_here.py` (23), `migrate_data_keys.py` (10), `runners_p2.py` (8),
`runners_p3.py` (8), `cleanup_analysis.py` (7), `purge_stale_analysis.py`
(7), `export_flask.py` (5), `report.py` (4), `runners_p1.py` (3),
`graft_patching.py` (1), `read_r45.py` (1).

All literals of the form `"R03_introspection.csv"` / `"Q33_decomposition.json"`
flipped to their new-ID equivalents (e.g. `"R0006_introspection.csv"`,
`"Q0042_decomposition.json"`). Format-string templates with variable
substitutions (`f"R{run_num:04d}_foo.csv"`) were already handled in earlier
arc work; this sweep only touched static string literals.

**NEW FILE: `migrate_run_ids.py` — bijective migration**

Rewritten from the pad-only version. Three phases:

1. **File rename.** Handles both legacy 2-digit (`R03_*`) and earlier-arc
   4-digit-old-pad (`R0003_*`) file forms. Renames to new canonical
   ID with correct R/Q prefix for the new number. Idempotent — already-
   canonical files are detected and skipped.

2. **CSV run_mode rewrite.** For every CSV with a run_mode column,
   replaces old integer/padded-old value with new 4-digit canonical ID.
   Atomic writes via `.tmp → os.replace`.

3. **JSON content rewrite.** Walks every `.json` in DATA/:
   - Scalar values under run-ID-named keys (`run_id`, `run_num`,
     `source_run`, `ref_run`, etc.) remapped.
   - String elements of plural-named arrays (`source_runs: ["3","15"]`)
     remapped to new IDs.
   - Dict keys remapped ONLY when the container key is a known
     run-indexed container (`per_run`, `runs`, `by_run`, `run_results`,
     etc.) — protects non-run numeric keys like trial indices from
     being incorrectly remapped.

**Idempotency:** marker file `.migration_0.79.4.0.COMPLETE` written
on successful `--apply`. Re-runs refuse unless `--force` is passed.
Prevents double-migration hazard where an already-remapped ID (e.g. 42)
would be mistaken for an OLD ID and remapped again (to 18).

**Safety flags:**
- `--apply` — actually make changes (default is dry-run)
- `--verbose` — list every affected file individually
- `--data-dir PATH` — override DATA root (testing use)
- `--skip-jsons` — skip Phase 3 (if you want to handle JSONs manually)
- `--force` — bypass the idempotency marker-file check

**Recommended workflow:**
```
# Stop Flask, finish Q8 + Run 0046 (was R56) collection, then:
tar -czf data_backup_pre_0.79.4.0.tar.gz data/
python migrate_run_ids.py                    # dry-run preview
python migrate_run_ids.py --apply            # actually do it
# Restart Flask, Ctrl+Shift+R dashboard, resume normally.
```

**Paper-facing impact:**
- Figure filenames change (`R33_decomposition.png` → `R0042_decomposition.png`).
- Paper section numbers that reference runs by ID flip.
- Master JSONs and cross-model summaries carry the new canonical IDs
  throughout.

**Version bumps:** `0.79.3.0` → `0.79.4.0` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).

---

### 0.79.3.0 — April 2026
`start_here.py`, `export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**"ALL MODELS" CARD + SEQUENTIAL CROSS-MODEL DISPATCH.**

Third functional ship of the 0.79.x → 0.80 redesign arc. Adds a
synthetic "All Models" card at the top of the model list. Clicking
it puts the dashboard into fleet-aggregate mode — the grid shows
cross-model completion status, and launching runs fans them out
across every discovered model in sequence.

**Scope for this ship — sequential, not parallel.**
Parallelization (GPU collection on Model A while no-GPU analysis
on Model B) is deferred to 0.79.3.1 or later. Adding a worker
pool, disk-lock handling, and concurrent-CSV safety is a bigger
change than fits one ship. Sequential gets the fan-out primitive
right first.

**UI — `export_flask.py`:**

*New synthetic card.* `buildModelCards()` prepends an "All Models"
card before iterating real models. Only rendered when ≥2 models
are discovered (no point showing fan-out with one model).
CSS: new `.model-card.all-models` variant — accent border, subtle
gradient, accent-colored name. Card body lists the number of models
and their labels.

*New navigation handler.* `navToAllModels(models)` sets the
`_allModelsMode = {models: [...]}` sentinel, clears per-model state
(`scanSt`, `_tempGrid`, `descs`), navigates to Layer 2, then fires
`/scan` per model with query params. Aggregates the N per-model
status dicts into a fleet view: a run shows `'done'` only when every
model has it done; `'partial'` if any model has started it; `'missing'`
if none. `_navModel` gets set to a synthetic `{family: '*', size: '*',
variant: '*', all_models: true}` so code paths that check `_navModel`
still have something truthy to work with.

*doScan override.* The periodic scan poll (5-10s) would normally
call `/scan` session-scoped — which would overwrite the aggregate
with whichever model's session file happens to be active. Added
a branch at the top of `doScan()` that, when `_allModelsMode` is
active, re-runs the same per-model overlay logic used by
`navToAllModels` to keep the aggregate fresh. Temp-grid poll
suppressed in this branch — session-scoped `/temp_grid` would
similarly corrupt the aggregate.

*doLaunch signal.* `doLaunch()` reads `_allModelsMode`; if truthy,
adds `all_models: true` to the `/run` POST body. Toast message
differentiates ("Launched across all models").

*paintGrid.* New branch: when `_navLayer === 2` and all-models mode
is active, cells use `scanSt` aggregate directly (bypass `_tempGrid`,
which doesn't apply to fleet view). Banner text shows "All Models (N):
X/Y done everywhere, Z partial". Progress bar reflects fleet-done ratio.

**Flask — `export_flask.py`:**

*Extended `/scan` endpoint.* Optional `family`, `size`, `variant`,
`temperature` query params. When provided, scans that specific model
without mutating the session. `size` is resolved via
`cartography.logical_size` so either `'2b'` or `'2b_8bit'` format works.
Response shape matches the session-scoped default (raw `{run_num:
status}` dict) so callers are consistent.

*`/run` endpoint.* Reads `all_models` from request body. When true,
spawns orchestrator with `--all-models <runs_payload>` — the runs
string passes through unchanged (accepts `'auto-temp'`, `'all-temps:N'`,
or plain run list; each model's subprocess re-interprets it per
its own data).

**Orchestrator — `start_here.py`:**

*New CLI arg.* `--all-models <runs_payload>` — outer iteration wrapper
that fans out runs across every discovered model.

*New `_run_all_models(session, runs_payload)` function.* Full
implementation:
- Discovers models via `export_stats._discover_model_data({})` — same
  walker the dashboard's `/models_collected` endpoint uses. Consistency
  between UI card list and orchestrator dispatch list.
- For each model, refreshes session with family/logical_size/variant/quant.
  Quant resolved via `cartography.quant_from_dir` (falls back to session
  default if size dir has no quant suffix). Model path + display name
  resolved via `vault.resolve_variant_path` so each subprocess has the
  correct HF path for its base/instruct/abliterated triple.
- Writes the per-model session atomically, then spawns the appropriate
  single-model sub-CLI based on runs_payload shape: `--auto-temp`,
  `--all-temps-runs`, or `--headless --auto-run`.
- Sequential dispatch — model A's subprocess fully completes (or fails)
  before model B starts. "Smart loader" is the natural consequence of
  outer-loop iteration — all of Model A's runs fire in sequence before
  Model B's first subprocess starts.
- Per-model failure is non-fatal: logged, appended to `failed[]`,
  loop continues to next model.
- After the loop, restores the original session so the user's "active"
  model isn't whatever trailing model the iteration happened to leave.
- Summary banner: completed vs failed model counts + per-model lines.

**Parallelization — NOT in this ship:**

Deferred. The architecture for concurrent GPU+CPU dispatch would need:
a resource-aware worker pool (GPU worker vs no-GPU worker, 2-max), CSV
write-lock coordination if two subprocesses touch the same per-model
paths, a queue scheduler that respects prereq graphs across models,
status pane updates reflecting which model is running what. Each is
moderate complexity on its own; together they're a full ship.

Sequential works today. If a user queues e.g. Run 34 on 3 models,
it fires 34 on Model A (analysis, ~5 min), then 34 on Model B, then
34 on Model C — total ~15 min instead of ~5 min with parallelism.
Acceptable trade for now; parallelization is a quality-of-life
improvement, not a correctness requirement.

**Files touched:**
- `start_here.py` — new CLI arg, new `_run_all_models()` function,
  arg-handler dispatch branch
- `export_flask.py` — CSS for all-models card, UI card prepend,
  `navToAllModels` JS function, `_allModelsMode` sentinel, `doScan`
  branch, `paintGrid` branch, `doLaunch` flag, `/scan` query params,
  `/run` `--all-models` dispatch
- `CHANGELOG.md`, `HANDOFF.md`

**Version bumps:** `0.79.2.0` → `0.79.3.0` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).

**Deployment:** drop-in safe over 0.79.2.0. Restart Flask,
Ctrl+Shift+R on dashboard. "All Models" card appears at the top of
the list when ≥2 models have data on disk. Click → grid shows fleet
status. Select runs → hit play → sequential dispatch. If you only
have one model, the card is hidden and the dashboard behaves
identically to 0.79.2.0.

---

### 0.79.2.0 — April 2026
`scanner.py`, `start_here.py`, `runners.py`, `runners_core.py`,
`export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**E_t RECOVERY PROMOTED TO FIRST-CLASS META-RUN (Run 48).**

Second functional ship of the 0.79.x → 0.80 redesign arc. Reclaims the
run-number gap at 48, replaces the auto-fire ET batch with a visible
queue entry, simplifies per-source status derivation.

**Before (0.79.1.0 and earlier):**
- Scanner marked runs 1-9, 15-17, 20 as `'needs_et'` when their CSV
  was complete but `*_et_base.npy` files were missing.
- `_run_session_isolated` pulled `needs_et` runs out of the subprocess
  loop and routed them to an implicit `_et_direct_batch`.
- Analysis runs (33, etc.) relied on `_check_prereqs` silently accepting
  `'needs_et'`/`'et_partial'` as "satisfied" so they could depend on
  source runs without the ET pass blocking them.
- Dashboard per-run popups independently computed ET status via
  duplicated logic, with a hard-coded `_ET_RECOVERY_TOTAL = {20: 500}`.

**After (0.79.2.0):**
- Scanner no longer emits `'needs_et'` or `'et_partial'` for any run.
  Source runs report honest `'done'`/`'partial'`/`'missing'` based
  purely on CSV + S_t completion.
- New `scanner._run48_status()` aggregates ET coverage across all 13
  sources. For each source with a CSV on disk, counts expected (unique
  non-priming `(trial, turn)` rows) vs actual `*_et_base.npy` files in
  the base-variant hidden_states dir:
    - all sources at 100% → `'done'` (green)
    - any source started but some incomplete → `'partial'` (orange)
    - no source has any ET files yet → `'missing'` (grey)
- `status[48]` emitted at end of every `scan_runs()` call.
- `_ET_RECOVERY_TOTAL` constant removed entirely. Expected counts
  derived from CSV row counts at scan time — no hardcoding.

**Run 48 in RUN_MAP, EXECUTION_ORDER, dispatch:**
- RUN_MAP entry: `48: ("runners_core", "E_t recovery — base-model pass
  for all Phase B sources (meta-run)")`
- EXECUTION_ORDER slot: between Phase B end (17) and Phase D start (21).
  Matches the Phase G position in the locked plan.
- `runners.run()` asserts 48 is accepted, short-circuits BEFORE the
  abliterated `load_model()` call (Run 48 loads its own base model via
  `_run_et_recovery`), then `os._exit(0)` to match the one-run-one-subprocess
  pattern all other runs use.
- Added to `_GEN_RUNS` so auto-temp dispatch picks it up as a normal
  collection run.

**Dependency graph update:**
- `_PREREQS[33]` now lists Run 48 explicitly: `48: "Run 48 (E_t recovery)
  — base-model E_t embeddings required for Phase B sources"`. Previously
  this was implicit via the `needs_et`/`et_partial` acceptance hack.
- `_check_prereqs` tightened: accepts only `'done'` as satisfied.
  `'needs_et'`/`'et_partial'` removed from the acceptance list.

**`_run_session_isolated` simplification:**
- Ripped the `_et_direct_batch` carve-out block (lines 1184-1198 of
  0.79.1.0). ~15 lines of orchestration logic gone.
- Runs flow through the normal subprocess-per-run dispatch — Run 48
  just shows up as one of the queued items and spawns its own
  subprocess like everything else.

**`_run_all_temps_analysis` safety-net rewritten:**
- Was: scan for `r in gpu_runs if _post_status.get(r) == 'needs_et'`.
- Now: check `_post_status.get(48) != 'done' and 48 not in gpu_runs` —
  fire `--single-run 48` if the user excluded Run 48 from their queue.
  Main gpu_runs loop already handles the normal case (48 in queue).

**`_status_header` console output:**
- Dropped `needs_et` / `et_partial` counters.
- Added Run 48 status surface: shows "E_t recovery partial" or
  "E_t recovery pending" instead of summing individual source ET states.

**Dashboard `run_detail_ep` cleanup:**
- Stripped the MC-branch ET short-circuit (was `_ET_RECOVERY_RUNS_MC =
  {20}` + `_ET_RECOVERY_TOTAL_MC = {20: 500}` + per-file glob count).
- Stripped the standard-branch ET short-circuit (same logic pattern
  for runs 1-9, 15-17).
- These blocks had duplicated scanner logic that would've drifted.
  Source runs in the popup now show honest done/partial/missing.

**Dashboard per-temp status banner:**
- Was: count runs with `needs_et` and `et_partial`, append banner text.
- Now: show Run 48 status explicitly alongside the done-count.

**Backward compatibility:**
- Legacy `--et-recovery <runs>` CLI path preserved — unchanged.
  External scripts or historical launchers still work.
- Interactive `_prompt_et_batch_cfg` preserved. Its `'needs_et'→'e'`
  and `'et_partial'→'e'` default-mode branches are dead paths
  (scanner won't emit those statuses) but the `'missing'→'b'` and
  `'partial'→'b'` defaults keep the interactive flow working for
  users who run ET via the console prompt instead of Run 48.
- Legacy `_run_session` ET dispatch block (line 1722+) marked as
  retained-defensively. Headless + Flask-launched sessions go through
  Run 48; interactive console sessions still hit the old code.

**Files touched:**
- `scanner.py` — constants, `_run48_status`, status-dict injection
- `start_here.py` — imports, RUN_MAP, EXECUTION_ORDER, `_PREREQS[33]`,
  `_check_prereqs`, `_GEN_RUNS`, `_run_session_isolated`, legacy block
  comments, `_status_header`, `_run_all_temps_analysis` safety-net
- `runners.py` — `run()` assert tuple, Run 48 dispatch branch
- `runners_core.py` — `_run_et_recovery` docstring
- `export_flask.py` — `EXEC_ORDER`, `PHASES` Data Collection list,
  `run_detail_ep` MC + standard branches, per-temp status banner JS

**Version bumps:** `0.79.1.0` → `0.79.2.0` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).

**Deployment:** drop-in safe over 0.79.1.0. Restart Flask, Ctrl+Shift+R
on dashboard. Run 48 will appear in the Collection grid as a new cell;
its status derives from whatever ET recovery state already exists on
disk — no migration needed. If you already have all ET files for
every source, Run 48 shows green immediately.

---

### 0.79.1.0 — April 2026
`start_here.py`, `export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**PHASE REORDER — EXECUTION_ORDER + dashboard tabs + console presets.**

First functional ship of the 0.79.x → 0.80 redesign arc. Integer run
numbers unchanged. Data on disk untouched. Zero migration.

**`start_here.py:EXECUTION_ORDER` rewritten** to the
psychologically-sequenced phase structure locked in during planning:

```
DATA COLLECTION
  A   19, 20, 26
  B   1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17
  G   (E_t recovery auto-fires here — becomes first-class in 0.79.2.0)
  D   21, 42, 53
  EFC 30, 31, 23, 28, 29, 43, 44, 24, 22,
      12, 13, 14, 18, 41, 36, 39, 38, 37, 35, 10, 11

PER-TEMPERATURE ANALYSIS
  45, 33, 34, 46, 49, 56, 25, 27, 32

POOLED ANALYSIS
  47, 40

CROSS-MODEL ANALYSIS
  50, 51, 52

OUTPUT
  54, 55
```

Dependency chain verified: 21/42/53 land after Run 3 (Phase B),
E_t sources (1-9, 15-17, 20) are complete by end of Phase B, 45
before 33/34/46/49/56, 33 before 34/46/49/56, 41 before 34.

**`export_flask.py` dashboard preset bar — six tabs:**

| Old | New |
|---|---|
| All | All |
| Data | Collection |
| Analysis | Analysis (tighter: adds 56, scopes to per-temp only) |
| Pooled | Pooled (tighter: scoped to {40, 47}) |
| — | Cross-Model (new: {50, 51, 52}) |
| — | Paper (new: {54, 55}) |
| Eₜ Recovery | Eₜ Recovery (unchanged — slated for fold-in at 0.79.2.0) |

Category constants tightened:
- `_POOLED_RUNS` was `{40, 47, 54}` → now `{40, 47}` (54 is Output)
- `_CROSS_MODEL_RUNS` was `{50, 51, 52, 55}` → now `{50, 51, 52}` (55 is Output)
- new `_OUTPUT_RUNS = {54, 55}` carved out of analysis accounting
  so the model card doesn't count 54/55 as "pending pooled" or
  "pending cross-model" — they're terminal paper-pipeline stamps.

**`export_flask.py` dashboard `EXEC_ORDER` + `PHASES` JS constants**
rewritten to match the new Python `EXECUTION_ORDER` exactly. Five
phase rows now visible on the grid: Data Collection, Analysis,
Pooled, Cross-Model, Paper.

**`start_here.py:PRESETS` console menu rewritten** to match the new
tab structure. Old phase presets (1–4) retained for historical
compatibility; new presets are `c` (Collection), `a` (Analysis),
`p` (Pooled), `x` (Cross-Model), `o` (Paper), `m` (Custom).

**Preserved unchanged:**
- Integer run numbers (renumber is 0.79.4.0)
- RUN_MAP contents (labels same)
- RUN_CSV / ANALYSIS_JSON (filenames same)
- All per-run dispatch logic
- E_t auto-fire batch (becomes first-class meta-run in 0.79.2.0)
- Scanner status logic
- PREREQS graph

**Version bumps:** `0.79.0.0` → `0.79.1.0` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).
Minor bump — first functional step of the 0.79.x arc.

**Deployment:** drop-in safe over 0.79.0.0 or 0.78.2.2. Restart
Flask, Ctrl+Shift+R on dashboard for the new tabs to render.
No data migration.

---

### 0.79.0.0 — April 2026
`HANDOFF.md`, `CHANGELOG.md`, `start_here.py`, `export_flask.py`

**PLANNING-ONLY SHIP. Zero code behaviour change.**

Kicks off the 0.79.x → 0.80.0 redesign arc. HANDOFF.md rewritten with:

- **IN FLIGHT — 0.79.x → 0.80.0 REDESIGN** section documenting every
  locked decision: run renumbering (1-55, 4-digit), phase reorder
  with dependency-verified ordering, E_t recovery as first-class run,
  dashboard tab rename (six tabs: All / Collection / Analysis / Pooled
  / Cross-Model / Paper), All-Models model card with smart loader.

- **Build order** table mapping each incremental ship (0.79.1.0 reorder,
  0.79.2.0 E_t meta, 0.79.3.0 All-Models, 0.79.4.0 renumber) through
  to the final 0.80.0.0 integration release.

- **INVENTORY** section with automated pattern-sweep counts across all
  28 .py files. Surfaces blast radius per-file and per-category for
  every upcoming edit class — RUN_MAP entries, dispatch branches,
  filename literals, dashboard HTML, etc. Estimated total edit surface
  ~500-600 string replacements across the arc.

- Preserved 0.78.x content under "PROJECT STATE — retained from 0.78.x"
  for continuity: Q56 paper-pipeline integration (0.78.2.0),
  build_master_jsons absorbed into export_stats (0.78.2.1), BUG-STALE-GRID
  fix (0.78.1.0), BUG-UNLOAD-STALL fix (0.78.2.2), docstring pass
  completion (0.78.0.2).

**Version bumps:** `0.78.2.2` → `0.79.0.0` in `start_here.py` and
`export_flask.py`. Minor bump (second digit) reflects the new arc
opening, not functional change.

**Deployment:** drop-in safe over 0.78.2.2. Restart Flask as usual.
No data changes, no migration. This ship exists purely so the plan
lives on disk alongside the code, not just in a chat transcript.

---

### 0.78.2.2 — April 2026
`orchestration_core.py`, `start_here.py`, `export_flask.py`,
`CHANGELOG.md`, `HANDOFF.md`

**BUG-UNLOAD-STALL fix: `mdl.cpu()` and `torch.cuda.empty_cache()` in
`unload_model()` now wrapped in daemon-thread + timeout.**

**Symptom:** collection hung at end of trial 1 turn 13 of Run 1 with
Gemma 2 2B Q8. Parent orchestrator waiting on `proc.wait()` for a
subprocess that never exited.

**Root cause:** `unload_model()` calls `mdl.cpu()` with no timeout.
For bitsandbytes 8-bit weights specifically, the dequantize-on-offload
path can stall indefinitely (Windows and WSL). `torch.cuda.synchronize()`
already had a 15s timeout from BUG-43B, but `mdl.cpu()` and
`empty_cache()` were unguarded.

Subprocess stuck on `mdl.cpu()` → `finally` in runners.py never
reached → `os._exit(0)` never fires → parent's `proc.wait()` blocks
forever → auto-temp hangs.

**Fix:** both `mdl.cpu()` (10s timeout) and `torch.cuda.empty_cache()`
(5s timeout) now run in daemon threads with the same pattern as
`cuda.synchronize`. If either times out, unload prints a message and
returns — `os._exit(0)` in runners.py forces process termination
regardless.

**Version:** `0.78.2.1` → `0.78.2.2`. Patch bump — single-file bug
fix.

**Deployment:** drop-in safe, restart Flask, resume collection.

---

### 0.78.2.1 — April 2026
`export_stats.py`, `start_here.py`, `export_flask.py`,
`build_master_jsons.py` (REMOVED), `CHANGELOG.md`, `HANDOFF.md`

**REFACTOR: `build_master_jsons.py` integrated into `export_stats.py`
and deleted.** Zero functional change — same outputs, same call path
from Run 55, cleaner surface.

**Rationale:**
0.78.2.0 shipped `build_master_jsons.py` as a standalone 478-line
module. Review found it duplicated work already in `export_stats.py`:
its `_discover_models` reimplemented `_discover_model_data`, its
`_rebuild_per_model` was a thin wrapper around `build_master_results`,
its JSON-loading helper duplicated the NaN-sanitising pattern used
five times elsewhere in `export_stats.py`. Only the cross-architecture
aggregation logic and the `all_models_master.json` writer were
genuinely new.

Keeping the duplication invited drift — a bug fix to
`_discover_model_data` would silently miss `_discover_models`.

**Changes:**

*`export_stats.py` — three new functions added after `build_master_results`:*

  - `_paper_model_key(model_info)` — builds the `family_sizedir`
    filename prefix for `DATA/paper/json/` matching Run 55's existing
    convention.

  - `_cross_arch_summary(all_models_dict)` — derives the
    `ridge_vs_mlp` / `R_per_temperature` / `decomposition_summary` /
    `R_pooled_by_model` sections. Includes the `conjecture_1_verdict`
    threshold check (max |ridge_gap| < 0.05 → `ridge_adequate`).
    Pre-computed for direct S7 cross-architecture paragraph writing
    by the writer bot.

  - `build_all_masters(verbose=True, rebuild_per_model=True)` — top-
    level entry. Uses the existing `_discover_model_data()` for model
    walk (no duplication). Writes
    `DATA/paper/json/all_models_master.json` with schema_version 1.0.
    `rebuild_per_model=False` skips the per-model rebuild — used by
    Run 55 where per-model masters were already rebuilt in-line.

  - `verify_masters()` — read-only status flag matrix [S P 6 F] per
    model (source/paper-copy/has-Q56/fresh-vs-aggregate). Moved from
    the standalone script intact.

*`start_here.py` — Run 55:*
Swapped `import build_master_jsons as _bmj; _bmj.build_all(...)` for
`import export_stats as _es_aggr; _es_aggr.build_all_masters(...)`.
One-line change; same behaviour.

*`build_master_jsons.py` — DELETED.*
The three genuine new pieces are now in export_stats; the rest was
duplication.

**Ad-hoc invocation paths:**

Run 55 calls this automatically; manual triggers when needed:

  ```python
  python -c "import export_stats; export_stats.build_all_masters()"
  python -c "import export_stats; export_stats.verify_masters()"
  ```

Slightly less ergonomic than `python build_master_jsons.py --verify`,
but the standalone CLI wasn't load-bearing — Run 55 was the primary
call path, and manual use was the exception. File count is back to
28 .py.

**Verification:** every .py compiles. Zero duplication of
`_discover_model_data`. Zero behavioural change — same per-model
master contents, same `all_models_master.json` schema, same Run 55
flow.

**Version bump:** `0.78.2.0` → `0.78.2.1` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).
Patch-level bump reflects the refactor — no new features, no
behaviour change, just cleaner internals.

**Deployment:** drop-in safe over 0.78.2.0. If you already deployed
0.78.2.0 and ran it, the `all_models_master.json` on disk is
identical in format — no re-migration needed. Delete
`build_master_jsons.py` from your deploy if you had copied it out,
or just re-extract the zip over the old dir.

---

### 0.78.2.0 — April 2026
`build_master_jsons.py` (NEW), `export_stats.py`, `start_here.py`,
`export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**FEATURE: Run 56 now flows into per-model and all-models master JSONs
for the paper pipeline.**

**Gap addressed:**
As of 0.78.1.0, Run 56 (MLP permutation sensitivity decomposition,
the Conjecture-1 empirical check against Ridge) wrote its output to
`DATA/{family}/{size_quant}/{variant}/{condition}/analysis/Q56_mlp_decomposition.json`
but had zero downstream consumers. `export_stats.py` didn't ingest it
into `master_results.json`, Run 55 paper assembly didn't propagate it,
and there was no cross-architecture aggregate anywhere. A completed
Q56 was visible on the dashboard but invisible to the writer bot.

There was also no single all-models JSON the writer bot could consume
for cross-architecture claims — per-model masters existed but had to
be stitched together by hand for every S7 cross-arch paragraph.

**Changes:**

*`export_stats.py` — one-line wiring (`build_master_results`):*
`'Q56': 'Q56_mlp_decomposition.json'` added to `_ANALYSIS_FILES`.
Every per-model `master_results.json` now ingests Q56 alongside the
existing Q25/Q27/Q32/Q33/Q34/Q45/Q46/Q49 files in the
`per_temperature.{temp}.Q56` slot. Next time Run 54 stats fires,
Q56 is picked up automatically. No schema change — a new top-level
key in a known container.

*`build_master_jsons.py` — NEW standalone script:*
Thin driver over `export_stats.build_master_results()` plus a new
cross-architecture aggregator. Three modes via CLI:
  - default               : rebuild per-model masters + all-models aggregate
  - `--per-model-only`    : skip the aggregate
  - `--all-only`          : skip per-model rebuild (aggregate from existing)
  - `--verify`            : read-only status report

Per-model: rebuilds every discovered model's `master_results.json`
(via the now-Q56-aware build_master_results) and copies to
`DATA/paper/json/{family}_{size_dir}_master_results.json` using the
exact naming convention Run 55 already uses.

All-models: writes `DATA/paper/json/all_models_master.json` with:
  - `generated`, `n_models`, `model_keys`, `schema_version`
  - `models` — full per-model master data, keyed by `{family}_{size}`
  - `cross_architecture`:
      - `R_pooled_by_model`     — mean R across temps per model
      - `R_per_temperature`     — per-model × per-temp R grid
      - `ridge_vs_mlp`          — Ridge vs MLP R means + ridge_gap per
                                  model + `conjecture_1_verdict`
                                  ('ridge_adequate' / 'mlp_materially_different')
                                  for direct S7 cross-arch writing
      - `decomposition_summary` — one-line `E=.., C=.., R=..` per model

Conjecture 1 verdict threshold: `|ridge_gap_max| < 0.05` →
`ridge_adequate`. Writer bot can pull this verdict directly for the
cross-architecture paragraph without re-implementing the check.

*`start_here.py` — Run 55 integration:*
`_run_55_paper_assembly` now calls `build_master_jsons.build_all(
all_only=True)` after the per-model copies but before the R55 stamp.
`all_only=True` because the per-model masters were just rebuilt
in-line; the aggregate is the only new work. Graceful degradation:
if the aggregate build fails, a `ui.warn` is logged and R55 completes
normally — the aggregate is a convenience, not a hard requirement.

**Verification sweep:**
All 29 .py files (28 prior + new `build_master_jsons.py`) pass
`python3 -m py_compile`. Every function in `build_master_jsons.py`
has a docstring. Zero changes to any existing analysis or run logic —
this is additive only.

**Version bumps:** `0.78.1.0` → `0.78.2.0` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).

Minor bump (second digit) reflects the new feature — this is the
second functional ship in the 0.78 line after 0.78.1.0's HTTP-cache
bug fix.

**Deployment:**

Drop-in safe over 0.78.1.0 or 0.77.1.3. No data migration. Restart
Flask.

To backfill existing models with Q56-aware masters, run once after
deploy:
  `python build_master_jsons.py`

To verify state without writing:
  `python build_master_jsons.py --verify`

Run 55 now does the aggregate build automatically; manual invocation
only needed for ad-hoc refreshes between R55 runs (e.g., when Run 56
lands for a new model mid-paper-writing).

---

### 0.78.1.0 — April 2026
`export_flask.py`, `start_here.py`, `CHANGELOG.md`, `HANDOFF.md`

**BUG-STALE-GRID fix: model cards and temp grids now refresh correctly
across model switches without requiring a Flask reboot.**

**Symptom (reported against 0.77.1.3):**
- Model cards stopped reflecting current disk state until Flask was
  rebooted. Adding a new temperature's data, running new analysis,
  or deleting a stale JSON produced no visible change in the cards.
- All-temps grid would display "old" data or, worse, data from a
  previously-selected model — cross-model contamination on switch.
- Only a Flask reboot cleared the state.

**Root cause:**
`/models_collected`, `/models_data`, `/temp_grid`, `/scan`, and every
other JSON endpoint in export_flask.py used Flask's bare `jsonify()`,
which returns `200 OK` with no `Cache-Control` / `Expires` /
`Last-Modified` headers. Under HTTP/1.1 heuristic caching rules
(RFC 7234 §4.2.2), browsers are free to cache such responses for
arbitrary durations, keyed on the URL. Chrome and Firefox cache them
aggressively — minutes to hours. When the user switched models in
the dashboard, the browser returned the previously-cached JSON from
the prior model instead of round-tripping to the server.

The 0.77.1.0 rip-out of the client-side `_modelCache` was correct
but addressed only half the problem: we removed our in-process JS
cache but left the browser's HTTP cache untouched. The frontend
dutifully called `fetch()` on every interaction; the browser
returned stale data without the request reaching the server.

The Flask reboot "fix" worked because it dropped the TCP connection
and forced a page reload, which invalidated the heuristic cache.
The underlying cause never went away — it just got papered over.

**Diagnosis path:**
Server-side verification showed `/models_collected` and `/models_data`
read fresh from disk every call (filesystem walk, no server cache).
That narrowed it to the browser. A single grep for `Cache-Control`
revealed that only SSE (`/stream`, line 406) and the HTML index
(`/`, line 2381) set cache headers. Every JSON endpoint in the app
— 30+ handlers — was uncacheable by policy but actually cacheable
by default.

**Fix:**
One `@app.after_request` hook inserted at line 108 applies
`Cache-Control: no-store, no-cache, must-revalidate, max-age=0` plus
`Pragma: no-cache` and `Expires: 0` to every response uniformly.
`no-store` (not just `no-cache`) forbids the browser from storing
the response at all — the correct policy for a live dashboard where
the underlying data changes continuously.

Zero changes to any endpoint logic. Zero changes to any client JS.
The fix is a blanket policy at the response layer.

**Scope notes:**
- SSE stream (`/stream`) had its own `Cache-Control: no-cache` header
  set per-response. The after_request hook overwrites with `no-store`,
  which is strictly stricter. EventSource handles no-store fine in
  all current browsers.
- The HTML index (`/`) already set the same policy; the hook confirms
  it uniformly.
- No static assets served by this app — everything is JSON or the
  one inlined HTML string. No incidental thrashing.

**Version bumps:** `0.78.0.2` → `0.78.1.0` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).

Minor bump reflects the functional nature of the change: this is the
first non-documentation ship in the 0.78 line.

**Deployment:** drop-in safe over 0.77.1.3 or 0.78.0.2. No data
migration. **Restart Flask** to load the new response hook — the
browser's cache from the old server will be partially bypassed but
a hard refresh (Ctrl+Shift+R) on first page load after the Flask
restart is recommended to clear any in-progress cached responses.

---

### 0.78.0.2 — April 2026
`dedup_all_runs.py`, `start_here.py`, `ui.py`, `orchestration_core.py`,
`export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**DOCSTRING MOPUP — 0.78.0.1 verified complete.**

Audit after 0.78.0.1 ship found 26 residual gaps — one-line chrome
helpers and class methods that the prior pass batched by file and
missed by accident. Zero functional risk, but the CHANGELOG claim of
"full pass on Sessions A+B files" was technically inaccurate. This
ship closes the gap.

Fills applied:
- `dedup_all_runs.py` — `main()` (the CLI entry, bare)
- `start_here.py` — `_find_root()` (bare despite being docstringed in
  every other file)
- `ui.py` — `section`, `opt`, `blank`, `msg`, `warn`, `ok`, `err`
  (one-line chrome, companions to `bar`/`dbar`/`header` from A);
  `Progress.__init__`, `Progress.update`, `Progress._render`,
  `Progress.finish` (class methods — class docstring already present)
- `orchestration_core.py` — `GPUMonitor` class docstring,
  `GPUMonitor._ensure_nvml`, `__init__`, `_run`, `start`, `stop`;
  `TokenTimer` class + `__init__`, `__call__`, `intervals`;
  `MinTokenEnforcer` class + `__init__`, `__call__`;
  `StatusTokenEnforcerFixed.__init__`, `__call__`;
  `TrialState.__post_init__`

**Coverage after this ship:** 253/253 non-trivial top-level functions
and class methods across all 18 Session-A+B files now documented.
Verified via ast.get_docstring sweep.

**Version bumps:** `0.78.0.1` → `0.78.0.2` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).

**Remaining work (Session C — 0.78.0.3 target):**
- `analysis.py` (14 top-level funcs)
- `export_stats.py` (7 top-level funcs)
- `export_flask.py` (30 top-level Flask handlers)

After 0.78.0.3, the codebase has 100% docstring coverage on every
non-trivial top-level function and method across all 28 .py files.

---

### 0.78.0.1 — April 2026
`cleanup_analysis.py`, `copy_hidden_r20_r26.py`, `purge_stale_analysis.py`,
`read_r45.py`, `dedup_all_runs.py`, `migrate_data_keys.py`, `report.py`,
`cartography.py`, `scanner.py`, `runners_core.py`, `runners_p1.py`,
`runners_p2.py`, `graft_patching.py`, `run42_layer_isolation.py`,
`orchestration_core.py`, `ui.py`, `setup.py`, `start_here.py`,
`export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**DOCSTRING PASS — Sessions A + B combined.**

Full-docstring pass across two-thirds of the codebase. Style matched
to `start_here.py`'s archaeology-friendly convention: one-line summary,
blank, then WHY — version markers (`v0.76.0.4:`, `BUG-QWEN fix`, etc.)
and cross-references preserved inline. Paragraph-length docstrings on
anything with non-obvious rationale; one-liners on trivial chrome.
Zero behaviour changes — pure documentation.

**Session A — infrastructure / migrations / utilities / patching / orchestration:**

*Utilities* — `cleanup_analysis` (3 funcs), `copy_hidden_r20_r26` (1),
`purge_stale_analysis` (1), `read_r45` (2), `dedup_all_runs` (4),
`report` (4).

*Migrations* — `migrate_data_keys` (6 funcs). `migrate_folders` is a
script with no functions; unchanged.

*Infrastructure* — `cartography` (12 funcs; every path helper and
embedding I/O function now documents its contract + kind-suffix
convention), `scanner` (4 nested helpers in `scan_runs`),
`orchestration_core` (`set_seed`, `get_status_token_ids`, `cosine_sim`,
`delta_r`, `load_model`, `run_generation`, `get_trials_to_run`,
`get_next_trial_for_condition`, `canonical_read`, `save_npy`,
`print_turn_result`, `_write_status`, `_append_log`, `run_calibration`
— 14 funcs, including the two largest untextured functions in the
codebase).

*Patching modules* — `graft_patching` (`_find_root`, `run`,
`_run_53_random_patching` — Run 21 / Run 53 protocol fully described),
`run42_layer_isolation` (`_find_root`, `run`, nested `_tint` — Run 42
single-layer causal-sufficiency protocol documented).

*Runner helpers* — `runners_core` (2 nested row extractors inside
`_get_trials_for_condition`), `runners_p1` (3 nested variant helpers
inside `_run_null_trivariant`), `runners_p2` (`_run_instance` — Run 30
two-instance VRAM-safe protocol; nested `_cosine`).

**Session B — user-facing shell:**

*`ui.py`* (19 funcs) — `bar`, `dbar`, `header`, `enter_exit`, `pick`,
`confirm`, `load_session`, `save_session`, `session_summary`,
`get_total_vram_gb`, `get_available_vram_gb`, `install_deps`,
`pick_quantization`, `pick_cache_mode`, `_infer_variant`,
`_browse_by_family`, `pick_runs`, `pick_params`, `_delete_run_data`,
`write_crash_marker`, `clear_crash_marker`, `check_crash_recovery`,
`prompt_crash_recovery`, `Progress` class, `_fmt_time`, `setup_logging`.

*`setup.py`* (7 funcs) — all chrome helpers (`bar`, `dbar`, `blank`,
`msg`, `ok`, `warn`, `err`, `header`, `confirm`), `check_missing`,
`install_system`, `create_venv`, `venv_python`, `install_venv`,
`write_bat`, `write_sh`, `write_env`.

*`start_here.py`* (6 top-level funcs) — `_load_module`, `_parse_runs`,
`_ensure_calibration`, `_run_session`, `_clear_stale_status`, `main`.
`main`'s docstring now enumerates every CLI dispatch path
(`--all-stats`, `--et-recovery`, `--single-run`, `--batch-runs`,
`--auto-run`, `--auto-temp`, `--all-temps-runs`, bare interactive).

**Version bumps:** `0.77.1.3` → `0.78.0.1` in `start_here.py` module
header and `export_flask.py` (module docstring + UI logo span).

**Verification:** every .py file in this zip passes `python3 -m
py_compile`. Every edit was a surgical replacement around the function
signature — no file bodies rewrote, no imports reordered, no logic
changed.

**Remaining work (next ship — Session C):**

*0.78.0.2 — analysis / reporting / dashboard:*
- `analysis.py` (14 top-level funcs — decomposition pipeline + cross-
  model discovery)
- `export_stats.py` (7 top-level funcs)
- `export_flask.py` (30 top-level funcs — Flask route handlers +
  session helpers)

After 0.78.0.2 the codebase has 100% docstring coverage on every
non-trivial top-level function, matching the handoff's ~500-edit
estimate.

**STATUS UPDATE (from 0.77.1.3):** Gemma 2 2B FP16 collection is
COMPLETE — all six temperatures landed and Conjecture 1 can proceed
to analysis. Handoff updated accordingly.

**Session recovery note:** 0.78.0.0 shipped Session A as a standalone
zip but `/mnt/project/` was never refreshed between sessions, so
0.78.0.1 re-applies Session A from the 0.77.1.3 baseline alongside
Session B's new content. Net effect: 0.78.0.1 supersedes 0.78.0.0
fully. If both zips are on disk, keep 0.78.0.1 and delete 0.78.0.0.

---

### 0.77.1.3 — April 2026
`scanner.py`, `export_flask.py`, `CHANGELOG.md`

**FIX #1: Cross-model 50/51/52 grey for models the run wasn't launched from.**

0.77.1.2 added 50/51/52 to `RUN_CSV` and the scanner checked
`{variant}/pooled/analysis/Q52_cross_model_summary.json`. That file only
exists for the model the cross-model run was LAUNCHED from — not for
every model it compared against. Result: cross-model cells greened for
the launching model, stayed grey for all other models in the comparison
set.

`DATA/paper/json/{family}_{size_dir}_master_results.json` IS written per-
model for every model assembled into a cross-model comparison. That's
the correct source of truth. Scanner now splits:

  - `_CROSS_MODEL_RUNS = {50, 51, 52}` → check paper/json per-model
  - `_POOLED_PATH_RUNS = {40, 47, 54}` → keep per-variant pooled/analysis

Run 55 (global paper assembly) continues to check its single paper/json
R55_paper_assembly.json. Run 56 (per-temperature MLP decomposition)
continues to check per-temperature ana_dir like other analysis runs.

**FIX #3: Run 56 skips as "already complete" when switching quant on same model.**

Model card click handler (`export_flask.py` line ~4093) sent family/size/
variant to `/session` but omitted `quantization`. Switching from a
Gemma 2B FP16 card to a Gemma 2B Q4 card updated the size string but
left session.quantization='fp16' (the previous value). `_wsess` then
called `set_active_quant('fp16')`, so `get_paths('gemma','2b',...)`
resolved to `data/gemma/2b_fp16/...` and the Run 56 completion gate
found FP16's Q56_mlp_decomposition.json and fired "already complete" —
against data from the wrong quantization.

Fix: card click POST body now includes `quantization: m.quant` (the
field was already exposed on `/models_collected` output at line 1357;
just not being sent). Session now reflects the card the user clicked.

---

### 0.77.1.2 — April 2026
`cartography.py`, `CHANGELOG.md`

**FIX: Cross-model runs 50/51/52 permanently grey in grid.**

`RUN_CSV` was missing entries for runs 50, 51, 52. Scanner iterates
`for run_num, csv_fname in RUN_CSV.items()` so those three run numbers
were never visited, their status was never computed, and the grid
received no data for them — showing grey regardless of whether the
cross-model JSONs actually existed on disk in `pooled/analysis/`.

The 0.77.1.0 fix in scanner.py that added `_POOLED_PATH_RUNS = {40, 47,
54, 50, 51, 52}` was correct logic but unreachable: the code path was
gated behind a loop that never included the run numbers it was trying
to handle.

Fix: add `50: None`, `51: None`, `52: None` to `RUN_CSV` alongside the
existing 55 (cross-model paper assembly) and 56 (MLP decomposition)
entries. Scanner now visits them, hits the `_POOLED_PATH_RUNS`
carve-out, finds the JSONs in `{variant}/pooled/analysis/`, reports
'done' across all 6 temperatures uniformly.

This has been broken since the cross-model runs were added in
v0.71.0.11 — five months. User reported it multiple times without
successful resolution because each attempted fix touched scanner logic
that was already correct. The missing piece was the registry entry
that makes the scanner loop visit the runs.

---

### 0.77.1.1 — April 2026
`export_flask.py`, `CHANGELOG.md`

**FIX: Run 56 missing from frontend registries.** Ship 4 in 0.77.1.0
registered Run 56 across the backend (`cartography.ANALYSIS_JSON`,
`cartography.RUN_CSV`, `start_here._ANALYSIS_RUNS`, `start_here.EXECUTION_ORDER`,
`start_here.RUN_MAP`, `start_here._PREREQS`, `analysis.run()` dispatch,
`REQUIRED_FIELDS`) but missed four places in the frontend JS:

- `ALL` array (line 3405) — iterated by `paintGrid` to find DOM cells; Run
  56 had no cell to render.
- `PHASES` Analysis group (line 3400) — the cell-generation loop uses this
  to emit `<div id="rc56">` elements; without it, no DOM node existed.
- `EXEC_ORDER` (line 3394) — hypothesis launcher queue ordering; bonus fix
  also added missing cross-model runs 50, 51, 52, 55 that were absent
  pre-0.77.
- `_ANA` progress-bar counter (line 4139) and `_ANALYSIS_RUN_SET` (line
  4964) — analysis classification for grid progress calculations.

Result: Gemma 2B FP16 grid appeared empty for Run 56 after 0.77.1.0 deploy
because the cell was never rendered. All five frontend sets now include
Run 56.

---

### 0.77.1.0 — April 2026
`analysis.py`, `cartography.py`, `start_here.py`, `export_flask.py`,
`export_stats.py`, `scanner.py`, `CHANGELOG.md`, `HANDOFF.md`

**Merge release — Ships 1-4 ported onto 0.77.0.0 refactor base.**

Branched from `iota_0_77_0_0.zip` (user's ArXiv terminology refactor) and
ported four ships forward that were not in the refactor tree.

**Ship 1 — ET batch launch / pre-scan / default-mode (was 0.76.0.5).**

`launchEtBatch` now sends `et_mode_overrides` dict keyed by string run number,
matching `_wsess` whitelist at line 219 and `_run_session` override reader at
start_here.py line 1523. Prior (refactor-inherited) code sent
`et_batch=[{run,mode}...]` — not in the session whitelist, silently dropped,
popup mode selections inert. Pre-scan in `_run_session_isolated` now honors
`session['et_mode_overrides']` in addition to `status=='needs_et'`: an
explicit mode='e' selection routes a run to `_et_direct_batch` regardless of
current status, so an ET request against 'partial', 'et_partial', or 'missing'
no longer spawns a full abliterated collection subprocess. `_default_mode` in
`_prompt_et_batch_cfg` now maps 'partial' → 'b' and 'et_partial' → 'e'
instead of falling through to 's' (skip) — interrupted collections and ET
passes now resume cleanly.

**Ship 2 — Layer 2 all-temps analysis routing (was 0.76.0.6).**

`_COLLECT` JS set in export_flask.py now uses the authoritative collection
run list (matches `start_here._GEN_RUNS`) instead of `{1..39}` numeric range.
Prior form treated Runs 25, 27, 32, 33, 34 (analysis) as collection and
excluded Runs 41, 42, 43, 44, 53 (actual collection). Selecting analysis-only
runs from Layer 2 routed to auto-temp, which found collection done, exited
"All temperature rounds already complete", and analysis never ran.

**Ship 3 — Bug sweep + hypothesis metadata cleanup (was 0.76.0.7, adapted).**

Adapted to the refactor's H02→H54 / H04→H55 / H07→H56 / H15→H57 renumbering.

Bug #22: `_qcache_load` in analysis.py now checks cache mtime against every
`.npy` in `hidden_states/` and every `*_et_base.npy` in the base sibling
directory. Stale quadruplet cache from before an ET recovery or a Run 19
pass-4 recomputation no longer silently poisons Run 33/34/46 fractions.
Cache is invalidated (deleted) on mtime mismatch and freshly rebuilt.

Bug #24: `/queue` POST endpoint now peels the `all-temps:` prefix off before
naive comma-split, parses the numeric body, merges with existing queue,
re-attaches prefix. Prior version fed `"all-temps:25,27,32"` into split(',')
— first token failed numeric parse, was dropped, AND the dispatch prefix was
lost. Incompatible dispatch intents (all-temps vs per-temp) detected and the
new caller's intent is taken as authoritative instead of silently merging.

Bug #26: `_PREREQS[33]` expanded from 8 to 16 source runs. Now covers
`SOURCE_RUNS_3WAY ∪ SOURCE_RUNS_2WAY`: Runs 1, 2, 6, 7, 8, 9, 19, 20 added.
Run 33 no longer silently proceeds with empty quadruplets from missing 2WAY
sources.

Hypothesis registry name alignment (export_stats.py). H54 "Is Degenerate" →
"Is Not Degenerate". H55 "Is Condition-Invariant" → "Is Elevated By
Introspection". H14 "Is Layer-Uniform" → "Concentrates In Late Layers"
(code tests late-layer concentration via `peak_idx >= n_layers/2`, not
uniformity). H57 "Is Layer-Uniform" → "Shows Early-To-Late Gradient"
(code tests directional gradient via `late − early > 0.02`). H50 and H51
now state their NULL direction (framework convention): "Does Not Improve
On Ridge By More Than 0.01", "Interaction Information Below 5% Of Joint MI".
All code logic unchanged.

UI alignment (export_flask.py H_PHASES): labels now match the export_stats
registry. H33 from "Compute Efficiency" to the registry's full "Condition A
Has No Task-Equivalent Compute Advantage". H34 from "Tracks Contradiction
Response" to "Does Not Track Contradiction Response" (registry form).

Dashboard hypothesis panel (export_flask.py): `showHypDetail` is now bound
to hypothesis row click (prior: function existed but was never invoked).
Added `#hypDetail` DOM container to the hypothesis tab (prior: target element
missing from markup). Upgraded rendering to a structured multi-row view —
hypothesis ID + colour-coded status chip + metric/value monospace line +
extra fields (p-values, effect sizes, etc.) rendered as a wrapped key/value
grid + full implication text. Numerical hypothesis results are now visible
in the UI for the first time.

**Ship 4 — Run 56: MLP permutation sensitivity decomposition (was 0.76.1.0).**

New parallel E+C+R decomposition using `MLPRegressor` instead of Ridge.
Mirrors Run 34 (pooled, 56a) and Run 46 (per-condition, 56b) using the
same permutation procedure — only the model class differs. Every Ridge
fraction now has an MLP fraction reported alongside it, with
gap-vs-Ridge flagged for each.

**56a (pooled):** one MLP fit per (architecture, seed) on `SOURCE_RUNS_3WAY`
pooled rows. Reads Q34_sobol_partition.json for ridge_reference.

**56b (per-condition):** one MLP fit per (source_run, architecture, seed).
Reads Q46_per_condition_R.json for ridge_reference. Per-condition mirrors
Run 46 exactly — same grouping by `run_num`, same `_MIN_QUADRUPLETS_PER_RUN`
threshold, same skip-if-insufficient behavior.

**Architectures:** MLP-64, MLP-256, MLP-128-64 (the three configs H50 tests).
**Seeds:** 42, 43, 44. **Permutations/component:** 50. **Total per-temperature
compute:** 9 pooled fits + ~81 per-condition fits + ~13,500 predict-only
evaluations. ~30-60 min per temperature on CPU.

**Optional `do_per_condition` flag:** defaults True. Can be disabled via
`session['_r56_per_condition'] = False` to run 56a only — faster when the
pooled Ridge-MLP gap already confirms Ridge adequacy. Batch drivers can
toggle off after the first model.

**Prerequisites:** Run 33 (quadruplets + POOL_DIM), Run 34 (pooled Ridge
reference), Run 46 (per-condition Ridge reference for 56b only). Added to
`_PREREQS[56]`. POOL_DIM calibration from Run 45 gates it via the existing
`_ensure_calibrated` analysis dispatch gate — no code changes needed.

**Output JSON (`Q56_mlp_decomposition.json`):** single file per (model,
temperature) with nested `pooled` and `per_condition` blocks plus a top-level
`mlp_summary`. Incremental flush after every (arch, seed). Resume-safe via
`REQUIRED_FIELDS[56] = ['pooled', 'mlp_summary']`. Error-isolated — one
crashed MLP fit does not abort the run; failed entries get an `error` key
and the summary reports `n_failed`.

**EXECUTION_ORDER placement:** after Run 46, before Runs 35-44/47/40/49/50-52.
`_ANALYSIS_RUNS` and `RUN_MAP[56]` extended. `cartography.ANALYSIS_JSON[56]`
and `RUN_CSV[56]=None` registered. Scanner and model-card counters auto-pick
up Run 56 through ANALYSIS_JSON.

**Model card impact:** analysis denominator bumps from 8 to 9 per temp until
Q56 JSON exists. Existing model cards will go orange until Run 56 runs at
each temp. Correct state reporting.

**Cross-model grid display fix.**

Runs 50, 51, 52 complete on disk (JSON in `pooled/analysis/`) but the
all-temps grid showed them as missing for Gemma 2B FP16. Three bugs:

1. `ANALYSIS_JSON` had no Run 50 entry. `_run_cross_model` writes Run 50
   output to `Q52_cross_model_summary.json` (same file as Run 52 — see
   analysis.py line 4434 `if run_num == 52 or run_num == 50:`). With no
   ANALYSIS_JSON registration, scanner had nothing to check for Run 50.
2. Scanner's pooled-path carve-out was `{40, 47, 54}` only. Runs 50/51/52
   also write to `pooled/analysis/`; scanner was checking per-temperature
   `ana_dir` where their JSONs don't exist. Extended carve-out to
   `_POOLED_PATH_RUNS = {40, 47, 54, 50, 51, 52}` unified across read path
   and marker path.
3. Model card's `_CROSS_MODEL_RUNS` was `{51, 52, 55}` — Run 50 fell into
   the per-temp analysis denominator by default. Extended to `{50, 51, 52, 55}`.

Run 50 now aliased to `Q52_cross_model_summary.json` in ANALYSIS_JSON to
match the code behavior. Follow-up ship can split them for real if needed.

**Grid cache removed.**

User report: stale cross-model data persisting across model switches.
Root cause: `_modelCache` in export_flask.py snapshotted `_tempGrid` and
`scanSt` on every switch/nav/scan with no invalidation. Removed the cache
variable, `_cacheKey`, `_saveModelCache`, `_loadModelCache`, `_curTemp`
helpers, and all 9 call sites. Every view now fetches fresh from
`/temp_grid` and `/scan`. Loading overlay shows during the refetch on
model switch. `_tempGrid` retained as plain in-memory variable populated
per-fetch — no persistence across model navigation. `localStorage` UI-prefs
helpers at lines 5752-5753 retained (separate concern — threshold sliders).

**Verification.** All 29 .py files pass `ast.parse`. Run 56 smoke-tested on
synthetic data (n=300, d=32 quadruplets) — MLP recovered expected fraction
ordering (E > C > R with ground truth E=0.56, C=0.25, R=0.19), R²=0.94,
converged in 286 iterations, zero negative drops.

**Restart Flask after deploying.**

---

### 0.77.0.0 — April 2026
`runners_prompts.py`, `runners_p2.py`, `runners_p3.py`, `runners.py`,
`runners_core.py`, `runners_p1.py`, `export_stats.py`, `export_flask.py`,
`start_here.py`, `orchestration_throughlines.py`, `orchestration_core.py`,
`analysis.py`, `cartography.py`, `dependency_map.py`, `ui.py`, `scanner.py`,
`report.py`, `migrate_data_keys.py`, `IOTA_Hypotheses_v10.md`,
`CHANGELOG.md`, `HANDOFF.md`

**Hypothesis catalog surgery:**
- Unified numbering: code and document now use the same `HNN` scheme.
  Document's legacy `H0-N` prefix retired; all hypothesis entries
  renumbered to match their code-schema counterpart by title.
- Four code-only hypotheses (previously numbered `H02`, `H04`, `H07`,
  `H15`, colliding with `H0-2`, `H0-4`, `H0-7`, `H0-15` which were
  distinct hypotheses) renumbered to `H54`, `H55`, `H56`, `H57`.
- Three new measurement-quality hypotheses added: `H52` (disruption
  magnitude stationarity), `H53` (cross-stochasticity disruption
  monotonicity), `H58` (E+C+R sum-to-1 diagnostic).
- `H40` (random noise patching control, Run 53) added to the document.
- Three framework hypotheses (`H20` signal-per-watt, `H33` compute
  advantage, `H47` coherence transfer efficiency) moved to
  Appendix A as out-of-scope for the measurement paper.
- Revelatory framing removed from all retained hypothesis bodies
  (attractor, flywheel, alignment-state, adversarial, production
  monitoring, etc.).
- Final count: 51 hypotheses in paper scope, 3 in Appendix A.


**SCOPE: Codebase terminology cleanup for ArXiv submission.**

This release culls theory-loaded vocabulary from the code surface. The
measurement apparatus is unchanged. All statistical tests, hypothesis
definitions, and data-collection procedures behave identically. Data
values written into new CSVs use new labels; old CSV data requires the
separate `migrate_data_keys.py` utility to become compatible with this
version.

**Renames — metrics (variables and CSV column headers):**
- `tci_proxy` → `state_similarity_index`
- `ser_proxy` → `signal_entropy_ratio`
- `ser_per_watt` → `signal_per_watt`
- `hesitation_ratio` → `onset_delay_ratio`
- `rc_event` → `disruption_flag`
- `rc_score` → `disruption_magnitude`

**Renames — run functions:**
- `_run_reflexivity` → `_run_self_reference` (Run 22)
- `_run_bound_state` → `_run_cross_instance` (Run 30)
- `_run_flywheel` → `_run_coherence_levels` (Run 31)
- `_run_flywheel_compression` → `_run_coherence_transfer` (Run 43)
- `_run_alignment_entropy` → `_run_contradiction` (Run 44)

**Renames — figure functions:**
- `fig_reflexivity` → `fig_self_reference_phase` (H05)
- `fig_alignment_rc_score` → `fig_self_reference_trajectory` (H34)
- `fig_bound_state` → `fig_cross_instance` (H19)
- `fig_flywheel` → `fig_coherence_levels` (H20)
- `fig_flywheel_compression` → `fig_coherence_transfer` (H33)
- `fig_alignment_entropy` → `fig_contradiction` (H35)
- Dead-code figure functions marked `NOT IN PIPELINE`

**Renames — prompt constants (runners_prompts.py):**
- `BOUND_STATE_SEED` → `CROSS_INSTANCE_SEED`
- `BOUND_STATE_FOLLOWUPS` → `CROSS_INSTANCE_FOLLOWUPS`
- `FLYWHEEL_HIGH_R_PROMPTS` → `HIGH_R_PROMPTS`
- `FLYWHEEL_MID_R_PROMPTS` → `MID_R_PROMPTS`
- `FLYWHEEL_LOW_R_PROMPTS` → `LOW_R_PROMPTS`

**Renames — CSV filenames:**
- `Q22_reflexivity.csv` → `Q22_contradiction.csv`
- `Q30_bound_state.csv` → `Q30_instances.csv`
- `Q31_flywheel.csv` → `Q31_conditions.csv`
- `Q43_flywheel_compression.csv` → `Q43_lengths.csv`
- `Q44_alignment_entropy.csv` → `Q44_recovery.csv`

**Renames — CSV cell values:**
- `condition` column, Q22 rows: `'reflexivity'` → `'self_reference'`
- `r_condition` column, Q43 rows: `'high_r_enforcer'` → `'condition_a'`,
  `'high_r_free'` → `'condition_b'`, `'low_r_verbose'` → `'condition_c'`
- Q31 and Q44 `r_condition` values `'high_r'` / `'mid_r'` / `'low_r'` unchanged

**Renames — throughline keys (orchestration_throughlines.py):**
- `'reflexivity'` → `'self_reference'`
- `'bound_state'` → `'cross_instance'`
- `'flywheel_high'` / `'flywheel_mid'` / `'flywheel_low'` → `'coherence_high'` / `'coherence_mid'` / `'coherence_low'`
- `'alignment_entropy_mid'` → `'contradiction_mid'`

**Renames — JSON analysis keys:**
- `results["flywheel_power"]` → `results["coherence_levels_power"]` (analysis.py)
- `"flywheel_slope_status"` → `"coherence_slope_status"` (export_stats.py H20)
- `"energy_p"` → `"compute_p"` (export_stats.py H33)

**Renames — figure PNG filenames on disk:**
- `H05_reflexivity_tci_phase` → `H05_self_reference_phase`
- `H19_bound_state_coupling` → `H19_cross_instance`
- `H34_alignment_rc_score` → `H34_self_reference_trajectory`
- `H35_alignment_entropy_recovery` → `H35_contradiction`
- `H20_power_by_condition` and `H33_compression_efficiency` unchanged

**Rewrites — comments, docstrings, print strings, UI labels, figure titles,
hypothesis titles:** theory-loaded vocabulary (flywheel, bound state,
reflexivity, alignment, deployable as production monitor, compression
efficiency proven, energy advantage, etc.) replaced with measurement-scope
descriptions throughout. H17 title "Coherence Is Fully Explained..." →
"Consistency Is Fully Explained...". H20 title "SER-Per-Watt Effect" →
"Signal-Per-Watt Effect". H31 title "Trajectory Coherence" → "Trajectory
Consistency". H33 "Energy Advantage" → "Compute Advantage". H34 "Track
Alignment State" → "Track Contradiction Response". H35 "Alignment
Recovery" → "Contradiction Recovery". ICD/COV/Emission_Gate/RRL/CBM/ROP
had zero hits in current codebase; mappings retained as guard against
reintroduction.

**Data migration.** Existing CSVs, JSONs, and figure PNGs on disk use the
old labels and are not compatible with this release of the code. Run the
separate utility `migrate_data_keys.py` (delivered outside this zip) to
rewrite the data on disk before invoking any analysis runs against prior
collections. The migration is atomic, dry-run capable, and produces a
backup of `data/` before any writes.

**Verification.** Zero hits for `flywheel`, `reflexivity`, `bound_state`,
`alignment_entropy`, `tci_proxy`, `ser_proxy`, `ser_per_watt`,
`hesitation_ratio`, `rc_event`, `rc_score` across all 28 `.py` files
(excluding `column alignment` in CSV formatting comments and `NOT IN
PIPELINE` markers on dead-code figure functions). All 28 files pass `ast.parse`.

**Restart Flask after deploying.** The dashboard caches module code in
memory; restart is required for the new symbol names to resolve.

---

### 0.76.0.4 — April 2026
`start_here.py`, `export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**FIX: auto-temp dispatched analysis runs during collection.** The run list
passed to `_run_session_isolated` was `'1-44,53'` which includes analysis runs
(25, 27, 32, 33, 34). EXECUTION_ORDER interleaves them: Run 32 (analysis)
fires right after Run 31 (GPU collection). Run 32 fails because Run 45
(POOL_DIM calibration) hasn't been done. The cascade: analysis failures →
session reports "0 runs done, 39 skipped/failed" → no ET markers written →
ET recovery skipped → auto-temp sees "some may have failed" → advances to
next temp without completing the current one.

Fix: `_run_all_temperatures` now builds the run list from `_GEN_RUNS` (the
39 GPU collection runs) instead of a hardcoded range. Analysis runs are
excluded entirely from the auto-temp collection loop. They run separately
after all temperatures are collected.

**INCLUDES 0.76.0.3:** Run 53 included in `_GEN_RUNS` dispatch.

---

### 0.76.0.3 — April 2026
`start_here.py`, `export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**FIX: auto-temp skipped Run 53.** `_run_all_temperatures` set
`session['runs'] = '1-44'` — Run 53 is outside that range. Fixed to `'1-44,53'`.

---

### 0.76.0.2 — April 2026
`start_here.py`, `export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**FIX: auto-temp dispatcher fed dispatch token as run list.** Flask's /run
endpoint writes `session['runs'] = 'auto-temp'` as the dispatch signal, then
spawns `start_here.py --auto-temp`. `_run_all_temperatures` iterates the six
temperatures and calls `_run_session_isolated(session)` for each. That
function reads `session.get('runs', '1-44')` — which returns `'auto-temp'`
(the dispatch token), not a run spec. `_parse_runs('auto-temp')` returns
empty. Every temperature printed "No valid runs selected" and exited in 3s.

Result: auto-temp advanced through all 6 temperatures in ~20 seconds and
reported "5 round(s) collected" with zero actual collection.

Fix: `_run_all_temperatures` now overrides `session['runs'] = '1-44'` before
each temperature's isolated run, so the isolated session parses a real run
list.

---

### 0.76.0.1 — April 2026
`export_flask.py`, `start_here.py`, `CHANGELOG.md`, `HANDOFF.md`

**FIX: All-temps grid view silently failed on fresh models.** When user
selected runs in the all-temps (Layer 2) view and clicked Run, JS dispatched
`all-temps:<runs>` which invoked `_run_all_temps_analysis` — the ANALYSIS
dispatcher. For a freshly-collected model with only T=0.0 data, this discovered
only the `deterministic` directory (others don't exist yet), ran ET recovery
(which is a no-op with no CSVs), reported "1 temperatures complete" in ~7s,
and exited. The user thought collection had run across all temperatures.

Fix: JS now detects if collection runs (1-39) are in the selection. If so,
routes to `auto-temp` (which runs `_run_all_temperatures` — the collection
dispatcher that iterates all 6 temps and collects missing ones). Analysis-only
selections still route to `all-temps:<runs>` as before.

---

### 0.76.0.0 — April 2026
`cartography.py`, `migrate_folders.py`, `start_here.py`, `export_flask.py`,
`analysis.py`, `export_stats.py`, `cleanup_analysis.py`, `copy_hidden_r20_r26.py`,
`purge_stale_analysis.py`, `read_r45.py`, `runners_p2.py`, `vault.py`,
`orchestration_core.py`, `runners.py`, `runners_p1.py`, `runners_core.py`,
`graft_patching.py`, `run42_layer_isolation.py`, `runners_prompts.py`,
`IOTA_Philosophy_and_Context.md`, `CHANGELOG.md`, `HANDOFF.md`

**ARCH: Root data directory renamed `base/` → `data/`.** Single-line change in
`cartography.py`. Physical rename handled by `migrate_folders.py`.

**ARCH: Quantization in directory paths.** Data paths now include quantization:
`data/{family}/{size}_{quant}/{variant}/{condition}/`. Prevents Q4/FP16 data
collision. `_ACTIVE_QUANT` module global in cartography.py, set at session start.
`_size_dir(size)` builds `{size}_{quant}` with double-append guard.
`logical_size()` and `quant_from_dir()` strip/extract from directory names.

**FIX: 22 manual path constructions bypassed `_size_dir()`.** Every
`os.path.join(DATA, family, size, ...)` across 8 files replaced with
`os.path.join(get_family_size_dir(family, size), ...)`. Files: start_here.py (5),
analysis.py (3), export_flask.py (6), export_stats.py (3), cleanup_analysis.py (1),
copy_hidden_r20_r26.py (2), purge_stale_analysis.py (1), read_r45.py (1).

**FIX: All 9 `load_model()` call sites pass `quant=`.** Calibration
(`_ensure_calibration`) and Run 30 (`runners_p2.py`) were missing the quant
parameter — loaded at Q4 regardless of dropdown setting. Now all call sites
pass `session.get('quantization', '4bit')`.

**FIX: SIZE_ORDER missing real model sizes.** Added 0.5b, 1.5b, 2b, 4b, 9b,
12b, 27b. ALL_MODELS entries corrected: Gemma 2B (3b→2b), Gemma 9B (8b→9b),
Gemma 3 4B (3b→4b), Gemma 3 12B (13b→12b), Gemma 27B (30b→27b),
Qwen 0.5B (1b→0.5b), Qwen 1.5B (1b→1.5b), Mistral Nemo 12B (13b→12b).

**FEATURE: FP16 quantization support.** `_resolve_quant()` maps quant strings
to BitsAndBytesConfig or None. Dashboard dropdown: 4-bit, 8-bit, FP16.
Quant shown in model cards and grid breadcrumb (`[4-bit]`, `[FP16]`).

**FEATURE: Extended null prompts for small base models.** 13 semantically empty
prompts, normalised to 20 tokens. Activates when base tokenizer has chat_template.

**FEATURE: Multi-EOS token support.** Collects all EOS IDs before overwriting
GenerationConfig. MinTokenEnforcer and StatusTokenEnforcer iterate lists.

**FEATURE: Base model raw text formatting.** `_iota_raw_text` flag on tokenizer.
Prompts formatted as content only, no role markers.

**FEATURE: Hidden state diagnostic.** One-time `[HS] path=... | layers=... | dim=...`
print on first extraction per model load.

**FIX: Auto-navigate removed.** Dashboard always starts on Layer 1.

**CONFIRMED: Qwen 2.5 1.5B Q4 numerically dead.** NaN from transformer layer 1
onwards. Only embedding layer returns real values. sim=0.0345 was 1/29 =
NaN layers mapped to 0.0 by nanmean.

`migrate_folders.py` handles the full migration:
1. `base/` → `data/` (root rename)
2. Size+quant renames (handles original, _q4, and _4bit states)
3. Session model_size correction

---

### 0.75.4.1 — April 2026
`export_flask.py`, `start_here.py`, `CHANGELOG.md`, `HANDOFF.md`

**FIX: Quantization dropdown cleaned.** Removed duplicate/ambiguous options
(16-bit, fp32). Three options: 4-bit, 8-bit, FP16. Matches `_resolve_quant()`
values exactly.

**FIX: Quant shown in grid breadcrumb.** All-temps view now shows
`Gemma 2 2B [4-bit]` or `Gemma 2 2B [FP16]` in the navigation breadcrumb.
Formatted display names: 4bit→"4-bit", 8bit→"8-bit", fp16→"FP16".
Model cards already showed quant (pre-existing), now formatted consistently
with breadcrumb.

**HANDOFF updated with full path architecture documentation.**

---

### 0.75.4.0 — April 2026
`vault.py`, `cartography.py`, `orchestration_core.py`, `runners.py`,
`runners_p1.py`, `runners_core.py`, `runners_prompts.py`,
`graft_patching.py`, `run42_layer_isolation.py`,
`export_flask.py`, `start_here.py`, `CHANGELOG.md`, `HANDOFF.md`

**ARCH: Quantization in directory paths.** Data paths now include quantization:
`base/{family}/{size}_{quant}/{variant}/{condition}/`. Prevents Q4/FP16 data
collision that caused data overwrite in v0.75.3.0. Implemented via module-level
`_ACTIVE_QUANT` in cartography.py, set once at session start. All 90 call sites
to `get_paths`/`get_pooled_paths`/`get_family_size_dir` inherit automatically.

**FIX: SIZE_ORDER missing real model sizes.** Added 0.5b, 1.5b, 2b, 4b, 9b,
12b, 27b. Models were bucketed into nearest available size (Gemma 2 2B → "3b",
Gemma 2 9B → "8b", Gemma 3 12B → "13b", Qwen 0.5B/1.5B → "1b"). All
ALL_MODELS entries corrected to actual sizes.

**DATA MIGRATION REQUIRED.** Run `python migrate_folders.py --apply` BEFORE
starting Flask. Renames: gemma/3b/ → gemma/2b_q4/, gemma/8b/ → gemma/9b_q4/,
llama/8b/ → llama/8b_q4/. Dry run without --apply.

**FP16 quantization support.** `load_model()` accepts `quant` string ('4bit',
'8bit', 'fp16'). All 7 call sites pass `session.get('quantization', '4bit')`.
Dashboard dropdown gains FP16 option. `_resolve_quant()` maps to the right
BitsAndBytesConfig or None.

**INCLUDES from 0.75.2.4–0.75.3.0:**
- Multi-EOS token support (BUG-QWEN)
- Auto-navigate removed
- Base model raw text formatting + extended null prompts
- Hidden state diagnostic print
- Qwen 1.5B Q4 confirmed numerically dead (NaN from layer 1)

---

### 0.75.3.0 — April 2026
`orchestration_core.py`, `runners.py`, `runners_p1.py`, `runners_core.py`,
`graft_patching.py`, `run42_layer_isolation.py`, `export_flask.py`,
`runners_prompts.py`, `start_here.py`, `CHANGELOG.md`, `HANDOFF.md`

**FEATURE: FP16 quantization support.** `load_model()` now accepts `quant`
string parameter ('4bit', '8bit', 'fp16', 'fp32'). `_resolve_quant()` maps
to the right BitsAndBytesConfig or None. All 7 call sites updated to pass
`session.get('quantization', '4bit')`. Dashboard dropdown gains FP16 option.

**WHY:** Gemma 2 2B at Q4 produces R ≈ 0.44. Is that the true R or a
quantization ceiling? Same model at FP16 (native training precision, no
compression) answers this. If R changes, Q4 compresses R. If R stays,
0.40 convergence is real. This is Conjecture 1.

**FIX: Qwen 2.5 1.5B Q4 produces NaN from layer 1 onwards.** Diagnostic
confirms: only the embedding layer (layer 0) has real values. All 28
transformer layers output NaN at Q4. The model is numerically broken at
this quantization — not a prompt or format issue. The previously observed
sim=0.0345 (1/29) was NaN layers silently mapped to 0.0 by np.where/nanmean.

**WARNING:** Data paths do not include quantization. Running Gemma 2B at
FP16 will overwrite Q4 data unless the Q4 directory is renamed first.
Rename `base/gemma/2b/` to `base/gemma/2b_q4/` before starting FP16.

**INCLUDES from 0.75.2.4–0.75.2.5:**
- Multi-EOS token support (BUG-QWEN)
- Auto-navigate removed
- Base model raw text formatting + extended null prompts
- Hidden state diagnostic print on first extraction

---

### 0.75.2.5 — April 2026
`runners_p1.py`, `runners_core.py`, `start_here.py`, `export_flask.py`,
`CHANGELOG.md`, `HANDOFF.md`

**FIX: Base model chat_template produces near-random hidden states (Qwen).**
Qwen base model tokenizers (`Qwen/Qwen2.5-1.5B` etc.) ship with a ChatML
`chat_template` even though the base model was never trained on chat format.
When applied, the model receives `<|im_start|>` / `<|im_end|>` tokens it
cannot process — hidden states come back near-random (sim ≈ 1/√d constant).

LLaMA and Gemma base tokenizers don't have `chat_template`, so they hit the
plain-text fallback naturally. Qwen never did.

Fix: `_run_null_trivariant` (Run 19 base pass) and `_run_et_recovery` now
strip `tok.chat_template = None` and set `tok._iota_raw_text = True` after
loading the base model. `run_generation` and `_extract_hidden_states` check
this flag and format prompts as raw content text with no role markers
(`system:`, `user:`, `assistant:`) — just the content, newline-separated.
Base models were trained on raw text completion, not any chat format.

This is consistent with the existing methodology: LLaMA and Gemma base
tokenizers lack `chat_template`, so they hit the role-marker fallback
(`"system: ...\nuser: ...\nassistant:"`), which those base models can
process. Qwen's base tokenizer has ChatML by default, and its base model
can't process ChatML OR role markers. Raw text is the only format it
understands.

Added one-time hidden state diagnostic: `[HS] path=... | steps=... |
layers=... | dim=...` prints on first extraction per model load.
Confirms extraction structure without spamming console.

---

### 0.75.2.4 — April 2026
`orchestration_core.py`, `graft_patching.py`, `run42_layer_isolation.py`,
`start_here.py`, `export_flask.py`, `CHANGELOG.md`, `HANDOFF.md`

**FIX: BUG-QWEN — Multi-EOS token support.** Models using ChatML (Qwen 2.5,
potentially Gemma 3) ship with multiple EOS token IDs in their generation config
(e.g. both `<|endoftext|>` and `<|im_end|>`). The framework overwrote this with
a single `tok.eos_token_id` in `load_model()`, causing generation to blow past
the natural turn boundary on non-status runs — producing garbage continuation.

Root cause: `GenerationConfig(eos_token_id=tok.eos_token_id)` on line 410
crushed the model's multi-EOS list to a single int. `MinTokenEnforcer` and
`StatusTokenEnforcerFixed` also only suppressed/forced one EOS ID.

Fix: `_get_all_eos_ids(tok, mdl)` collects EOS IDs from both the tokenizer and
the model's original generation config BEFORE overwriting. The collected list is
stored as `tok._iota_eos_ids` and used everywhere via `get_eos_ids(tok)`.
`MinTokenEnforcer` and `StatusTokenEnforcerFixed` now accept and iterate over
a list of EOS IDs. `GenerationConfig` receives the full list. Diagnostic print
on load when multi-EOS detected.

Affected call sites (7): `load_model` GenerationConfig, `run_generation` ×2,
`graft_patching.py` ×1, `run42_layer_isolation.py` ×1, plus the two class
definitions. All `pad_token_id` references remain single-int (correct for padding).

**UNBLOCKS:** Qwen 2.5 1.5B collection. Qwen 0.5B FP32 vs Q4 (Conjecture 1).

**REMOVE: Auto-navigate and localStorage model cache.** The v0.75.2.2 auto-navigate
feature jumped to the session's last model on page load, even when that model had
no data (Qwen). Removed: `_autoNavFromCache()` IIFE, localStorage save/restore of
`_modelCache`, auto-set of `_navModel` in `buildModelCards`. In-memory model cache
retained for within-session switching. Dashboard always starts on Layer 1 (model cards).

---

### 0.75.2.3 — April 2026
`export_flask.py`, `start_here.py`, `scanner.py`, `CHANGELOG.md`

**FIX: Run 55 popup triggers on any selection containing 55.** Previously only
triggered when 55 was the sole selected run. Now intercepts whenever 55 is in the
set, stores other runs (50,51,52) and launches them alongside. Diagnostic logging
added: `[R55] doLaunch runs=... has55=true` in browser console.

**FIX: Cross-model JSONs propagated to all models.** Run 55 copies Q50/Q51/Q52
from whichever model generated them to ALL included models' pooled/analysis/.
Scanner shows cross-model runs as "done" in every model's grid, not just the
source model.

**FIX: Auto-navigate to last model on fresh page load.** Dashboard checks session
for active model + localStorage for cached grid → goes straight to Layer 2 with
cached data, background refresh with buttons locked.

**FIX: Grid cache persists in localStorage.** Survives page reloads and Flask
restarts. Console diagnostics: `[cache] Restored`, `[cache] hit`, `[cache] miss`.

**FIX: RDP popup buttons disabled during grid load.** All launch-capable buttons
grey out while `_gridBusy` is true, not just toolbar buttons.

---

### 0.75.2.2 — April 2026
`export_flask.py`, `start_here.py`, `scanner.py`, `CHANGELOG.md`

**FIX: Run 55 checkbox UI fully wired.** Selecting Run 55 and hitting Run opens
a model selection dialog. Fetches `/cross_model_status`, shows checkboxes for each
discovered model (checked + enabled if analysis complete, greyed + locked if not).
User confirms, `cross_model_include` list flows through session to
`_run_55_paper_assembly` which filters to selected models only. Added to session
allowlist in `/run` endpoint.

**FIX: RDP popup run buttons disabled during grid load.** `_setRunBtnEnabled` now
targets `#rdpActs button`, `.gc-btn.go`, and `.set-action` in addition to the main
toolbar run buttons. All launch-capable buttons grey out while `_gridBusy` is true.

**FIX: Grid cache localStorage diagnostics.** Console logs on cache save/load/miss
to debug whether localStorage persistence is working. Messages: `[cache] Restored
from localStorage`, `[cache] hit for llama/8b`, `[cache] miss for gemma/9b`.

**FIX: Auto-navigate to last model on page load.** On fresh page load, dashboard
checks session for the active model, then checks localStorage for cached grid data.
If cache exists, auto-navigates from Layer 1 (model cards) directly to Layer 2
(all-temps grid) with cached data showing immediately. Background fetch updates to
live data with run buttons disabled until refresh completes.

---

### 0.75.2.1 — April 2026
`export_flask.py`, `CHANGELOG.md`

**FIX: Run 55 missing from dashboard.** Run 55 was registered in start_here.py
(RUN_MAP, EXECUTION_ORDER, POOLED_RUNS) but not wired into the dashboard. Six
JS/Python locations fixed: `_CROSS_MODEL_RUNS` (model card count 8/9 → 8/8),
Cross-Model grid section, ALL array, JS `_POOLED_RUNS`, "All" preset (1-55),
"Pooled" preset (47,40,54,55).

---

### 0.75.2.0 — April 2026
`analysis.py`, `start_here.py`, `export_flask.py`, `export_stats.py`,
`cartography.py`, `HANDOFF.md`, `CHANGELOG.md`

**FIX: BUG-RACE-GRIDRUN — Run button disabled during grid load.** Clicking Run
while the all-temps grid was still loading caused Flask to block on concurrent
requests (single-threaded), freezing the entire dashboard. Run button and toolbar
launch button now disabled (greyed, unclickable) while `_gridBusy` is true.
Re-enabled after `loadTempGrid()` completes. Toast warning if user somehow triggers
launch during load. `.catch()` handler ensures buttons re-enable on fetch failure.

**ARCH: Run 54/55 split — per-model vs cross-model paper assembly.**

Run 54 (per-model) retains: stats export at each temp, cross-temp status table,
three-variant comparison, master results, per-model pooled diagnostics, stamp.

Run 55 (new, cross-model) takes: `generate_combined_paper_figures()` (FIG01–FIG13),
paper JSON assembly from ALL models. Session-independent — discovers all models
with analysis data, filters to ready models only. `models_include` parameter
added to `generate_combined_paper_figures()` for selective model inclusion.

New `/cross_model_status` endpoint returns discovered models with analysis
completion status (n_temps_analyzed, has_r54_stamp, ready flag). Supports
dashboard checkbox UI (next push).

Run 55 registered in: RUN_MAP, EXECUTION_ORDER, POOLED_RUNS, ANALYSIS_JSON,
single-run dispatch, all-temps pooled dispatch.

**CLEANUP: CHANGELOG compressed.** 8010 → ~450 lines. Pre-v0.74 history condensed
into Phase 1–9 summaries (4–6 lines each). Duplicate CURRENT STATE headers merged.
Two VERSION HISTORY sections merged. Phase ordering corrected to chronological.

---

### 0.75.1.0 — April 2026
`analysis.py`, `start_here.py`, `export_flask.py`, `HANDOFF.md`, `CHANGELOG.md`

**ARCH: Resume-aware analysis skip gate replaces file-existence gate.**

**Problem:** Analysis runs (25, 27, 32–34, 40, 45–47, 49–52) used a simple
file-existence check to determine completion: if the output JSON existed and
`status != 'running'`, the run was skipped. When new analysis stages were added
to a run (e.g. H50 linearity_check and H51 interaction_info added to Run 33),
existing JSONs passed the existence check even though they were missing the new
fields. The run never re-entered, and the new stages were never computed.

Three redundant skip gates existed:
1. `analysis.py run()` inner gate (v0.71.0.12)
2. `_run_all_temps_analysis` outer gate (per-temp analysis loop)
3. `_run_all_stats` outer gate (with force flag, but default skipped)

**Fix:** Single resume-aware gate in `analysis.py run()`. All outer gates in
`start_here.py` removed — analysis.py is the sole source of truth.

`REQUIRED_FIELDS` dict defines per-run completeness:
- Run 33: requires `r2_D_full_decomposition`, `linearity_check`,
  `interaction_info`, `h16_confound_decomposition`
- Run 34: requires `perm_sens_R`, `bootstrap_ci`
- Run 40: requires `perm_sens_R`
- Other runs: legacy fallback (file exists + status != running)

If a required field is missing or contains an `error` key, the run re-enters.
Internal incremental logic (Run 33's `_merge_q33`) handles skipping completed
stages. Runs without incremental logic recompute fully — correct behaviour when
the old output was incomplete.

**FIX: Quadruplet cache persists across analysis re-entry.**

`qcache_cleanup` calls removed from `start_here.py`. Cache files were deleted
after every analysis pass, forcing minutes-long re-loads from individual .npy
files on re-entry. Cache now persists on disk for instant re-entry (~1 second).

Stale cache management moved into `_qcache_save`: after writing a new cache,
any other `_qcache_*.npz` files in the same directory are deleted. One cache
file per temperature directory. No accumulation. `qcache_cleanup` retained for
manual use but no longer called automatically.

**FIX: Internal resume guards rejected completed JSONs on re-entry.**

Four resume guards inside analysis runs checked `status == 'running'` before
recognising cached data. Completed JSONs have status popped — so on re-entry
(triggered by missing REQUIRED_FIELDS), the guard failed and the entire
analysis recomputed from scratch (200 OLS permutations, bootstrap CIs, etc.).

Fixed in Runs 33, 34, and 46 — all resume guards now check data presence only,
not status. Completed JSONs with cached OLS/permutation/bootstrap data are
recognised immediately.

---

### 0.75.0.6 — April 2026
`export_stats.py`, `CHANGELOG.md`

**FIX: Pooled figure scope — single model only.** `fig_r_vs_temperature`,
`fig_ols_bars`, `fig_ecr_stacked_bar` now accept `single_model=True` parameter.
Pooled task list passes `single_model=True` so pooled figures show only the
session's model, not all models. FIG10 (cross-model) removed from pooled entirely.

**FIX: FIG03 y-axis flipped.** Deeper layers now at bottom via `invert_yaxis()`.

**FIX: FIG12 number placement.** Values moved above bars with clearance instead
of overlapping bar faces.

**FIX: FIG12 thinner bars.** Width reduced to 0.4 with edge lines.

---

### 0.75.0.5 — April 2026
`export_stats.py`, `CHANGELOG.md`

**FIX: base/instruct variants permanently excluded from stats pipeline.**
`export_stats.run()` returns immediately for base/instruct variants.
`_discover_model_data()` skips base/instruct in model scanning.
`generate_paper_figures()` returns 0 for base/instruct. These variants
only store hidden states for Run 19 — no CSV data, no analysis, no figures.

---

### 0.75.0.4 — April 2026
`export_stats.py`, `CHANGELOG.md`

**FIX: All remaining color eliminated.** Replaced #2196F3 (blue), #64B5F6 (light
blue), #EF5350/#E53935/#B71C1C (reds), #1565C0 (dark blue), #7B1FA2 (purple),
#22c55e (green), #ef4444 (red), viridis colormap, RdYlGn, YlOrRd colormaps, and
named 'red'/'green' colors with grayscale equivalents. Zero non-grayscale elements
remain in any figure function.

**FIX: Per-model FIG04 stacked bars replaced with grouped bars.** Three bars per
temperature (E, C, R) side-by-side with hatching. No more stacking.

**FIX: regen_diagnostics.py skips base/instruct variants.** Only abliterated
variants have full CSV data for stats export.

---

### 0.75.0.3 — April 2026
`export_stats.py`, `CHANGELOG.md`

**FIX: Grayscale everywhere — no remaining color.** Replaced viridis colormap in
per-model dim₉₅ curve with grayscale shades + markers + linestyles. Replaced
RdYlGn and YlOrRd colormaps with Greys. Zero colored elements remain in any
figure function — paper, pooled, or per-temp.

---

### 0.75.0.2 — April 2026
`export_stats.py`, `CHANGELOG.md`

**FIX: Global model style registry.** Added `_MODEL_STYLES` dict and
`_get_model_style()` function. Every combined paper figure now uses consistent
symbols per model: LLaMA 8B = black circles solid, Gemma 9B = grey squares dashed,
Gemma 2B = light grey triangles dotted. Qwen slots pre-assigned. No more
symbol-shuffling between figures. All combined figure functions updated to use
`_get_model_style(m, mi)` instead of loop-indexed `_BW_COLORS[mi]`.

**FIX: FIG13 sig figs — removed sharey, forced formatting per panel.** Each panel
now independently sets `FormatStrFormatter('%.3f')` and identical ylim. All panels
show y-axis labels with consistent 3-decimal formatting.

---

### 0.75.0.1 — April 2026
`export_stats.py`, `CHANGELOG.md`

**STYLE: Full grayscale overhaul — all figures.** Seaborn palette switched from
"muted" (colorful) to _BW_COLORS grayscale globally. PAL dict updated to grayscale.
All 10 paper figures + per-temp diagnostics now render in black/grey.

**STYLE: Line charts — markers + dashes on every line.** FIG01, FIG04, FIG05, FIG06,
FIG08 now use _BW_LINESTYLES (solid, dashed, dotted, dash-dot) in addition to
_BW_MARKERS for maximum grayscale differentiation.

**STYLE: Bar charts — thinner bars, hatching, compressed y-axes.** FIG02, FIG09,
FIG10 use narrower bars (0.25–0.5 width), edge lines, hatching patterns per model,
and auto y-axis scaling tight to data range with 15% padding.

**REDESIGN: FIG04 E+C+R — stacked bars replaced with three-panel line plot.**
Three subplots (E, C, R) each showing all models across temperature with
markers + dashes. Shared y-axis. Compressed axes. No more stacked bars.

**FIX: Per-model pooled diagnostics restored in Run 54.** The v0.75.0.0 restructure
removed the `generate_paper_figures(session)` call that populates each model's
`pooled/visuals/` with FIG07, FIG11, FIG12 and other per-model diagnostics. Run 54
now calls both `generate_combined_paper_figures()` (→ base/paper/) and
`generate_paper_figures(session)` (→ model/pooled/visuals/).

**FIX: FIG13 — consistent 3-decimal sig figs across all panels.** Shared y-axis
with FormatStrFormatter('%.3f'). Y-range padded uniformly across panels.

---

### 0.75.0.0 — April 2026
`cartography.py`, `export_stats.py`, `start_here.py`, `CHANGELOG.md`

**STRUCTURAL: Paper figures moved from per-model to global `base/paper/`.** Paper
figures are now 10 combined cross-model charts generated once into `base/paper/visuals/`
instead of 13 per-model figures in each model's `pooled/visuals/`. Summary JSONs
copied to `base/paper/json/` with model-prefixed filenames.

New `get_paper_paths()` in cartography.py returns `base/paper/{visuals,json}`.

New `generate_combined_paper_figures()` in export_stats.py discovers all models,
preloads Q33/Q34/Q45/Q46/Q49 and three-variant data for each, and generates 10
combined figures. Figures use grayscale with distinct markers for print-safe
differentiation. All figures auto-scale y-axes to data range.

The 10 paper figures:
  FIG01 R(T) curve — all models overlay
  FIG02 OLS ΔR² — grouped bars per model
  FIG03 Patching heatmap — side-by-side panels (different layer counts)
  FIG04 E+C+R stacked bar — grouped by model
  FIG05 Held-out validation — overlay per model
  FIG06 dim₉₅ curve — overlay (shows capacity differences)
  FIG08 Bootstrap CI — overlay distributions
  FIG09 Per-condition R — grouped bars per condition per model
  FIG10 Cross-model R — grouped bars all models
  FIG13 Three-variant TCI — multi-panel per model

FIG07 (Per-turn TCI), FIG11 (Fixed-dim cross-temp), FIG12 (Interaction mass)
removed from paper set — remain as per-model diagnostics in pooled/visuals/.

Per-model `paper/` directories inside each model are no longer created. Per-temp
diagnostics in `{temp}/visuals/` and pooled diagnostics in `pooled/visuals/` are
unchanged — kept for researchers.

`_run_54_stats_report` updated to call `generate_combined_paper_figures()` and
assemble JSONs into `base/paper/json/`.

---

### 0.74.0.9 — April 2026
`export_stats.py`, `CHANGELOG.md`

**FIX: FIG13 Three-variant delta key mismatch.** Figure read `entries` key but
`_run_54_stats_report` writes `table` key. Also fixed structure mismatch: data is
long format (one row per model per temp) but figure expected wide format (one row
per temp with tci_abliterated/tci_instruct/tci_base columns). Now pivots the long
table into per-model arrays correctly. Grayscale with distinct markers/linestyles.

---

### 0.74.0.8 — April 2026
`export_stats.py`, `CHANGELOG.md`

**CRITICAL FIX: Missing `import re` in export_stats.py.** The `re` module was
never imported but used in `_load_json_all_temps_for_model` and
`_load_analysis_json_all_temps` for NaN/Infinity sanitisation. Every call to
`re.sub()` raised `NameError`, caught by `except Exception: pass`, silently
skipping every JSON file. This caused all 11 JSON-based paper figures to return
None while the 2 CSV-based figures (FIG03, FIG07) worked. One missing import
killed 11 of 13 paper figures. Fix: add `re` to the import line.

---

### 0.74.0.7 — April 2026
`export_stats.py`, `CHANGELOG.md`

**FIX: Paper figures now enforce grayscale publication style.** `sns.set_theme()`
was overriding the publication rcParams with seaborn's "muted" (blue) palette.
`generate_paper_figures()` now resets to serif fonts, grayscale color cycle, and
no grid before generating, then restores the previous style after. Paper figures
use `_BW_COLORS` and `_BW_MARKERS` for print-safe differentiation.

**FIX: Paper figure diagnostic logging.** Figures that return None (missing data)
now log "returned None (data missing or insufficient)" instead of silently moving
on. Exceptions log the error message. Console now shows exactly which figures
generated and which failed and why.

**REDESIGN: FIG07 Per-turn TCI trajectory.** Previous version was a single blue
line with all data bunched at the top of a 0–1 y-axis. Now shows per-condition
thin lines (grayscale, distinct markers) with bold overall mean + SEM band.
Y-axis auto-scales to the 1st–99th percentile of the data. Legend identifies
each condition.

**FIX: FIG01 R(T) curve auto y-axis.** Y-axis now pads 10% around the data
range instead of defaulting to 0–1. Distinct markers per model for grayscale
differentiation.

---

### 0.74.0.6 — April 2026
`S1_introduction.md`, `S9_conclusion.md`, `CHANGELOG.md`

**FIX: "No prior publication" claim updated.** Six preprints are now published on
Zenodo (ORCID: 0009-0006-7552-6367). S1 and S9 updated to reference the Zenodo
preprints and note this is the first work to present empirical data. The previous
claim ("No prior publication by the present author exists") was accurate when
written but is no longer correct.

---

### 0.74.0.5 — April 2026
`start_here.py`, `CHANGELOG.md`

**FIX: Dashboard startup race condition.** Opening the browser immediately after
spawning Flask allowed the user to click Run before Flask was listening on port
5000. The cached page loaded instantly, the Run request either failed silently or
hit a half-initialized server, crashing the session. Now polls `localhost:5000`
in a loop (0.5s interval, 10s timeout) before opening the browser. If Flask
doesn't start within 10s, prints a warning with the manual URL instead of
opening a browser to a dead server.

---

### 0.74.0.4 — April 2026
`analysis.py`, `CHANGELOG.md`

**FIX: Quadruplet cache load 100x slower than necessary.** `np.load()` returns a
lazy `NpzFile` object — every `d['s_prev'][i]` re-decompresses the full array from
disk. At dim=1024 × 22800 rows, the loop did 90000+ full decompressions of a 90MB
matrix. Cache load took 10+ minutes instead of the expected ~5 seconds. Now forces
all arrays into memory (`np.array(d['key'])`) before the iteration loop. Expected
speedup: 10 min → ~10 sec at dim=1024.

---

### 0.74.0.3 — April 2026
`analysis.py`, `CHANGELOG.md`

**ADD: Run 33 incremental saves and early completion.** Three post-OLS stages
(H16 confound decomposition, H50 linearity check, H51 interaction information)
now save results to the partial JSON after each stage completes. On resume, each
stage checks the partial and skips if already computed. If all three stages are
complete in the partial but the final write failed (status still 'running'), Run 33
writes the final output directly from the partial without loading quadruplets —
eliminates a 2-minute reload on crash recovery at dim=1024.

**FIX: Run 34 early completion bootstrap CI format.** The early completion path
wrote nested format (`{E: {ci_lo, ci_hi}, ...}`) while the normal path and all
downstream code (export_stats.py figure generation) expected flat format
(`{R_ci_lower, R_ci_upper, R_excludes_zero, ...}`). CI bands in figures silently
fell back to point estimates and `R_excludes_zero` always read False. Now writes
flat format matching the normal path. **Existing Run 34 files that completed via
early completion need re-running** — delete Q34_sobol_partition.json and re-run.

---

### 0.74.0.2 — April 2026
`export_flask.py`, `CHANGELOG.md`

**FIX: Console duplicate entries.** Rawlog polling (`.iota_flask.log`) creates DOM
elements with class `ll raw`. When JSONL polling takes over (`LLN > 0`), rawlog
deactivated but left its elements in the DOM. JSONL then appended the same content
as separate elements — visible duplicates. Now clears all `ll raw` elements when
rawlog deactivates.

**FIX: Clear button cascade.** After Clear, clicking the "load history" indicator
set `_clrMode=false` which re-enabled the scroll-up auto-loader. With `scrollTop`
near 0 from the prepend, the scroll handler fired repeatedly — each batch load
triggered another, cascading through the entire log in seconds. Now the indicator
loads one batch without clearing `_clrMode`. Scroll-up auto-loading resumes only
when the user hits the Bottom button (natural "I'm done browsing" signal).

---

### 0.74.0.1 — April 2026
`start_here.py`, `CHANGELOG.md`

**FIX: Paper directory now contains only paper figures.** Previously, Run 54
copied all per-temp diagnostic PNGs (9 × 6 temps = 54) plus all pooled paper
figures (13) into `paper/visuals/`, producing 67 files. Now copies only the 13
pooled paper figures (FIG01–FIG13). The `pooled_` prefix is dropped — filenames
are clean (`FIG01_R_vs_temperature.png`, not `pooled_FIG01_R_vs_temperature.png`).
Per-temp diagnostics remain in their own `{condition}/visuals/` directories.

---

### 0.74.0.0 — April 2026
`export_stats.py`, `start_here.py`, `analysis.py`, `CHANGELOG.md`

**CLEANUP: Paper figures renumbered FIG01–FIG13.** All 13 paper figures generated
by `generate_paper_figures()` now use zero-padded sequential numbering matching
their generation order in `_PAPER_TASKS`. Previous numbering had duplicate prefixes
(two FIG3s, two FIG4s, two FIG7s) — filenames were unique but the shared prefixes
caused confusion. New mapping:

  FIG01 R(T) curve                FIG08 Bootstrap CI distributions
  FIG02 OLS ΔR² bars             FIG09 Per-condition R fractions
  FIG03 Patching heatmap          FIG10 Cross-model R comparison
  FIG04 E+C+R stacked bar         FIG11 Fixed-dim cross-temp
  FIG05 Held-out validation       FIG12 Interaction mass
  FIG06 dim₉₅ curve               FIG13 Three-variant delta
  FIG07 Per-turn TCI trajectory

**CLEANUP: Legacy report figures removed from Run 54.** `report.generate()` call
removed from `_run_54_stats_report()`. The 9 report figures (fig1–fig9 in
pooled/analysis/figures/) had color-only differentiation, compressed y-axes, and
wrong metric choices. Paper figures (FIG01–FIG13) supersede them. `report.py`
retained in codebase for manual REPORT.md generation if needed.

**FIX: Q49 missing from force-delete list.** `_PER_TEMP_JSONS` in `_run_all_stats`
now includes `Q49_fixed_dim_per_condition_R.json`. Previously, ALL STATS with
`--force` left stale Q49 files — computed at old POOL_DIM or with old data.

**FIX: Stale comment in bootstrap.** Line 1842 said "re-pool to 512 dims" but
code uses POOL_DIM since v0.73.1.42. Comment corrected.

---


---

## VERSION HISTORY (v1.0 – v0.73)

Condensed from ~8000 lines. Full detail in git history.

### Phase 1 — Foundation (v1–v19)
Core framework established. E+C+R=1 decomposition design. Run infrastructure,
CSV append, hidden state save. `append_csv`, `ensure_csv_header`, `save_npy`.
Granger probe design (Runs 25, 27). Baseline swap (Run 32).

### Phase 2 — CTM + Resume (v20–v35)
`compute_turn_metrics` added (tci_proxy, ser_proxy, ser_per_watt, hesitation_ratio,
rc_event, rc_score). Resume logic: `get_trials_to_run`, `strip_partial_trial`,
`strip_partial_rows`. Dedup (`dedup_all_runs.py`). Calibration. E+C+R full OLS
decomposition (Run 33). Permutation sensitivity partition (Run 34).

### Phase 3 — Three-Model Redesign (v36–v44)
Run 19 redesigned: three model passes (abliterated + base + instruct) + C_t vectors.
`_run_et_recovery` for true base-model E_t on Runs 1–9, 15–17. `load_embedding`
with kind='E_base'. Held-out validation (Run 41, H0-28). Pooled decomposition (Run 40).
Activation patching (Runs 21, 42). Phase 4 runs: 43, 44, 35–39.

### Phase 4 — v45–v49: Scanner Bugs Fixed
Series of CSV scanner bugs fixed as new runs exposed schema variants:
BUG-INVARIANT9 (result.update before CTM), ser_per_watt added late,
first_token_entropy added, priming-row header issues, multi-condition resume bugs.
895-line scanner accumulated position heuristics for each new variant.

### Phase 5 — v50–v54: Canonical CSV Schema + Stats Pipeline
`normalize_csvs.py` one-time migration: all CSVs rewritten to FTF(0–38) + sorted
extras. `_scan_runs` rewritten to read by column name. Cross-model analysis runs
(51–52). Random noise patching (53). Stats pipeline + dashboard terminal (54).
`ensure_csv_header` added to all runners (BUG-MISSING-CTM fix).

### Phase 6 — v0.59–v0.66: Performance + Multi-Model (March 2026)
CRITICAL: GPUMonitor.stop() blocked 1s/turn — 3-4x speedup. Buffered npy saves.
ET-only runs skip subprocess spawning (1hr → minutes). Run 30 OOM fix. Gemma 2 9B
model path corrected (failspy→IlyaGusev). Multi-model session switching stabilised.

### Phase 7 — v0.67–v0.69: Patching + Dashboard (April 2026)
Run 21 fourth mode: random activation patching (N(μ,σ) per-layer). Run 53: random
noise patching baseline (H40) — 6 modes. CRITICAL: browser cached stale JS after
Flask restart — Cache-Control headers added. Dashboard integration for Runs 21/42/53.

### Phase 8 — v0.70–v0.71: Dashboard Overhaul + Stats (April 2026)
Complete dashboard UI restructuring: 3-layer navigation (Models → All Temps →
Per-Temp). Model setup, settings tab, sparklines, unified grid controls.
Omnibus stats: hypothesis renames, "flywheel" stripped, analysis skip gate,
cross-temp status tables, hypothesis tab grouped by requirement.

### Phase 9 — v0.72–v0.73: Cross-Model + Model Catalog (April 2026)
Run 49: fixed-dim per-condition R. Stats export renumbered to Run 54. Live
hypothesis computation. IOTA_SUBFAMILY_MAP expanded to 26 verified triplets
(5 families). Model cards, 3-dropdown selection, Add Model flow. Idle memory
fixes. 46 dashboard patches (v0.73.1.1–v0.73.1.46).
