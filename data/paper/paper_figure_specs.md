# Paper Figure Specs

Reference document for paper figures. Covers (a) Nature journal rendering
requirements and (b) per-figure content specs. Written as a grep target, not
a narrative — look up what you need, skip the rest.

---

## Nature rendering requirements

Verified against Nature submission guidelines as of early 2026. If the target
shifts to a different journal, start here and diff against that journal's
spec sheet.

### Dimensions

- **Single column:** 89–90 mm wide (sometimes written as 88 mm for Nature
  Communications; 90 mm is the safe default for the main journal)
- **Double column:** 180–183 mm wide
- **Max height:** 170 mm (leaves room for legend below)

Design at exact print dimensions from the start. Matplotlib:

```python
import matplotlib as mpl
mm = 1 / 25.4  # mm to inches conversion for figsize

fig, ax = plt.subplots(figsize=(180 * mm, 120 * mm))  # double column
```

### Resolution / format

Preference order:

1. **Vector** (PDF, EPS, SVG) — resolution-independent, no DPI concerns
2. **Raster combination** (multi-panel with photos + graphs) — 600 DPI
3. **Raster line art** (diagrams) — 1200 DPI
4. **Raster photo** — 300 DPI minimum, 450 DPI for online proofs

Every figure in this paper ships as both **SVG** (primary, fully editable) and
**PDF** (journal upload). No PNG — PNG in a Nature submission is a downgrade.

Matplotlib config for both formats with editable text:

```python
mpl.rcParams['pdf.fonttype'] = 42   # TrueType, editable text in Illustrator
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'  # text as text, not paths

fig.savefig('fig1.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('fig1.svg', bbox_inches='tight', pad_inches=0.05)
```

### Typography

- **Font:** Arial or Helvetica. Symbol font for Greek letters.
- **Body text size:** 5–7 pt (Nature rejects text outside this range)
- **Panel labels:** 8 pt bold lowercase (a, b, c) in upper-left corner

```python
mpl.rcParams['font.family']     = 'Arial'   # Helvetica as fallback
mpl.rcParams['font.size']       = 7
mpl.rcParams['axes.labelsize']  = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
```

Panel labels go in `ax.text(0.02, 0.98, 'a', transform=ax.transAxes,
fontsize=8, fontweight='bold', va='top')`.

### Color

- **Mode:** RGB (Nature converts to CMYK for print)
- **Palette:** Okabe-Ito or Wong 2011 8-color colorblind-safe
- **Avoid:** red/green pairs (8% of male readers cannot distinguish),
  rainbow colormap for continuous data
- **Diverging data centered at zero:** use red-blue (`RdBu_r`), not red-green

Okabe-Ito reference:

```python
OKABE_ITO = {
    'black':       '#000000',
    'orange':      '#E69F00',
    'sky_blue':    '#56B4E9',
    'bluish_green':'#009E73',
    'yellow':      '#F0E442',
    'blue':        '#0072B2',
    'vermillion':  '#D55E00',
    'reddish_purple':'#CC79A7',
}
```

### Other requirements

- **Axes:** tick marks present; axis labels include units in parentheses
  (e.g., `"Temperature (T)"`, not `"Temperature"`)
- **Error bars:** required where applicable. Legend must state what the bars
  represent (95% CI, SEM, SD) and the sample size
- **Statistical tests:** state the test used and the p-value
- **Scale bars:** use scale bars not magnification factors; keep bar + label
  on a separate editable layer (don't flatten into the image)
- **Editable layers:** never flatten figures. Nature production staff adjusts
  lines, arrows, scale bars, text for house style

### File size

Max 10 MB per figure. For this paper's figures, expected size is well under
that (~0.5–2 MB per SVG, similar for PDF).

---

## Per-figure content specs

Five main figures. Each is generated from `data/paper/results.json` via
`fig{N}_*.py` script in 0.80. Each figure ships as `.svg` + `.pdf` at
`data/paper/fig{N}_{name}.{ext}`.

Every figure writes a sidecar JSON next to it: `data/paper/fig{N}_{name}.meta.json`
containing:

```json
{
  "source_fields": ["cells.llama_8b_4bit_abliterated_T1.0.r2_ridge", ...],
  "cell_filter": {"temperature": 1.0},
  "caption_draft": "...",
  "results_json_hash": "sha256 of results.json at render time",
  "generated_at":  "iso timestamp",
  "iota_version":  "0.80.x"
}
```

The hash ensures a reader six months from now can tell whether the figure
matches the current `results.json` or was rendered from an earlier version.

### Figure 1 — Ridge vs MLP by temperature (methodology headline)

The paper's headline made visual. **Four panels, one per configuration**
(LLaMA 8B Q4, Gemma 9B Q4, Gemma 2B Q4, Gemma 2B FP16). Each panel:

- X-axis: temperature (0.0 to 1.0, 6 values)
- Y-axis: R̂ (0 to 0.6, **same scale across all four panels** so the reader
  compares heights without rescaling their eyes)
- Paired bars per temperature: Ridge R̂ (one color) and MLP R̂ (another color)
- Error bars from seed variance; permutation-standard-error if no seed variance

The LLaMA panel should show the ~0.18 Ridge-to-MLP drop at T=1.0 visibly
opening. The Gemma panels show the bars close together or MLP above Ridge.

**Source fields:**
- `cells.{key}.rhat_ridge_insample` (Run 0043)
- `cells.{key}.rhat_mlp_reference` (Run 0046, mlp_256 designated reference)
- Error from `cells.{key}.rhat_ridge_std` / equivalent MLP std

**Format:** double column (180 mm), 2×2 panel grid, ~120 mm total height.

### Figure 2 — Toy verdict heatmap

The 3×3 grid from the toy output rendered at publication quality.

- X-axis: `nl_E` (E-channel nonlinearity, 0.0 / 0.5 / 1.0)
- Y-axis: `nl_S` (S-channel nonlinearity, 0.0 / 0.5 / 1.0)
- Cell values: `Rhat_gap` (MLP S-share − Ridge S-share), annotated in each cell
- Colormap: `RdBu_r` diverging, centered at zero, vmin/vmax = ±max absolute gap
- Corner annotations for predictions: (0,0) should be ≈0, (0,1) predicted
  negative, (1,0) predicted positive, (1,1) predicted small

**Source fields:** `toy_nonlinearity.grid` (3×3 table with Rhat_gap mean per cell).

**Format:** single column (90 mm), square aspect (90 × 90 mm).

### Figure 3 — Mechanism falsification scatter

Two-panel scatter. **Four points per panel** — the four configurations at T=1.0.
Label points directly; no legend.

**Left panel (redundancy hypothesis):**
- X-axis: kNN II fraction at T=1.0 (`cells.{key}.ii_fraction_knn`)
- Y-axis: Ridge→MLP R̂ drop at T=1.0
  (`cells.{key}.rhat_ridge_insample - cells.{key}.rhat_mlp_reference`)
- Interpretation: if redundancy drives the gap, points cluster on a positive slope

**Right panel (nonlinearity hypothesis):**
- X-axis: channel-marginal asymmetry (`cells.{key}.asymmetry` from channel_marginal)
- Y-axis: same Ridge→MLP drop as left panel
- Interpretation: nonlinearity mechanism predicts structure here that left panel lacks

**Source fields:**
- `cells.{key}.ii_fraction_knn` (Run 0042.interaction_info)
- `cells.{key}.asymmetry` (channel_marginal calibration artifact)
- `cells.{key}.rhat_ridge_insample`, `rhat_mlp_reference`

**Format:** double column (180 mm), 1×2 panels, ~80 mm height.

### Figure 4 — Temperature trajectories

Single-panel line plot. Four lines, one per configuration.

- X-axis: temperature (0.0 to 1.0)
- Y-axis: R̂ — use MLP R̂ as primary (cleaner estimator under H50 failure)
- Optional overlay: thin dashed line per configuration with Ridge R̂
  trajectory at half opacity, same color. Reader sees both estimators' paths.
- Optional overlay: small vertical tick where H50 first fails per configuration,
  labeled

**Source fields:**
- `cells.{key}.rhat_mlp_reference` across all temperatures
- `cells.{key}.rhat_ridge_insample` for optional overlay
- `cells.{key}.h50_passes` to locate first-fail tick

**Format:** single or double column (author choice), ~100 mm height.

### Figure 5 — Layer-isolation causal profile

Four-panel grid (one per configuration) or single panel with grouped bars.
Shows output-change rate from Run 0018 single-layer patching.

- X-axis: layer (L8, L16, L24, L31) grouped by temperature
- Y-axis: output change rate
- Finding to illustrate: L16-at-low-T vs L31-at-high-T crossover (the only
  causal result in the paper, deserves its own figure)

**Source fields:**
- `cells.{key}.layer_output_change_rate.{layer}` (Run 0018 output)
- Per-layer binomial test p-value for annotation

**Format:** double column (180 mm), grouped bars, ~100 mm height.

---

## Rendering checklist (per figure, before saving)

- [ ] figsize in mm matches column-width target
- [ ] fonts at 5–7 pt body, 8 pt bold panel labels
- [ ] `pdf.fonttype = 42`, `svg.fonttype = 'none'` set
- [ ] Okabe-Ito or Wong 2011 palette; no red/green pairs
- [ ] Axes labeled with units in parentheses
- [ ] Error bars present where applicable, legend explains what they represent
- [ ] Panel labels a/b/c in upper-left, bold lowercase, 8 pt
- [ ] Both SVG and PDF saved to `data/paper/fig{N}_{name}.{ext}`
- [ ] Sidecar meta.json written with source fields + results.json hash
