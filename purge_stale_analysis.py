"""
IOTA FRAMEWORK — PURGE STALE ANALYSIS
=======================================
Deletes analysis outputs (Q33, Q34, Q46) from temperature rounds
that were computed before per-temperature POOL_DIM calibration.

Targets:
  - temp_0.2, temp_0.4, temp_0.8, temp_1.0: Q33, Q34, Q46 + stats outputs
  - pooled/: ALL analysis outputs (Q40, Q47, report, figures)

Keeps:
  - deterministic (T=0.0): untouched
  - temp_0.6: untouched (calibrated at POOL_DIM=2048)

Usage:
    python purge_stale_analysis.py
"""

import os, sys, json, shutil, glob

def _find_root():
    """Walk up from this file's directory until we find the iota root
    (directory containing start_here.py). Lets the script run from anywhere."""
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(candidate, "start_here.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.dirname(os.path.abspath(__file__))

ROOT = _find_root()

# ── Load session to find data directory ────────────────────────────────────
sess_path = os.path.join(ROOT, 'last_session.json')
if not os.path.exists(sess_path):
    print("  ERROR: last_session.json not found. Run start_here.py first.")
    sys.exit(1)

with open(sess_path) as f:
    session = json.load(f)

family  = session.get('model_family', 'llama')
size    = session.get('model_size', '8b')
variant = session.get('model_variant', 'abliterated')

from cartography import DATA, _size_dir, set_active_quant
set_active_quant(session.get('quantization', '4bit'))
base_dir = os.path.join(DATA, family, _size_dir(size), variant)

if not os.path.isdir(base_dir):
    print(f"  ERROR: Data directory not found: {base_dir}")
    sys.exit(1)

# ── Define purge targets ──────────────────────────────────────────────────

PURGE_TEMPS = ['temp_0.2', 'temp_0.4', 'temp_0.8', 'temp_1.0']
KEEP_TEMPS  = ['deterministic', 'temp_0.6']

# Per-temperature analysis files to delete
PER_TEMP_ANALYSIS = [
    'Q0042_decomposition.json',
    'Q0043_sobol_partition.json',
    'Q0044_per_condition_R.json',
    'hypothesis_outcomes.json',
    'descriptive_statistics.json',
]

# Pooled directory — nuke everything in analysis/
POOLED_ANALYSIS = [
    'Q0051_pooled_decomposition.json',
    'Q0051_pooled_sobol.json',
    'Q0051_temperature_curve.json',
    'Q0050_cross_temp_synthesis.json',
    'REPORT.md',
]

# ── Scan and collect targets ──────────────────────────────────────────────

targets = []  # (path, description)

print(f"\n  IOTA Purge — Stale Analysis Cleanup")
print(f"  Base: {base_dir}")
print(f"  Purge temps: {PURGE_TEMPS}")
print(f"  Keep temps:  {KEEP_TEMPS}")
print()

# Per-temperature analysis files
for temp_dir in PURGE_TEMPS:
    ana_dir = os.path.join(base_dir, temp_dir, 'analysis')
    vis_dir = os.path.join(base_dir, temp_dir, 'visuals')

    if not os.path.isdir(ana_dir):
        continue

    for fname in PER_TEMP_ANALYSIS:
        fpath = os.path.join(ana_dir, fname)
        if os.path.exists(fpath):
            targets.append((fpath, f"{temp_dir}/analysis/{fname}"))

    # Visuals directory — figures are also stale
    if os.path.isdir(vis_dir):
        pngs = glob.glob(os.path.join(vis_dir, '*.png'))
        if pngs:
            targets.append((vis_dir, f"{temp_dir}/visuals/ ({len(pngs)} figures)"))

# Pooled analysis — everything
pooled_ana = os.path.join(base_dir, 'pooled', 'analysis')
if os.path.isdir(pooled_ana):
    for fname in POOLED_ANALYSIS:
        fpath = os.path.join(pooled_ana, fname)
        if os.path.exists(fpath):
            targets.append((fpath, f"pooled/analysis/{fname}"))

    # Pooled figures directory
    pooled_figs = os.path.join(pooled_ana, 'figures')
    if os.path.isdir(pooled_figs):
        n_figs = len(glob.glob(os.path.join(pooled_figs, '*.png')))
        if n_figs:
            targets.append((pooled_figs, f"pooled/analysis/figures/ ({n_figs} figures)"))

# Pooled visuals
pooled_vis = os.path.join(base_dir, 'pooled', 'visuals')
if os.path.isdir(pooled_vis):
    pngs = glob.glob(os.path.join(pooled_vis, '*.png'))
    if pngs:
        targets.append((pooled_vis, f"pooled/visuals/ ({len(pngs)} figures)"))

# ── Display and confirm ──────────────────────────────────────────────────

if not targets:
    print("  Nothing to purge — all clean.")
    sys.exit(0)

print(f"  {len(targets)} targets found:\n")
for _, desc in targets:
    print(f"    ✗  {desc}")

print(f"\n  Keeping:")
for keep in KEEP_TEMPS:
    keep_ana = os.path.join(base_dir, keep, 'analysis')
    if os.path.isdir(keep_ana):
        n_files = len([f for f in os.listdir(keep_ana) if f.endswith('.json')])
        print(f"    ✓  {keep}/analysis/ ({n_files} JSON files)")
    else:
        print(f"    ✓  {keep}/ (no analysis yet)")

print()
raw = input("  [b]ackup (.bak) / [d]elete permanently / [q]uit > ").strip().lower()

if raw in ('b', 'd'):
    use_bak = (raw == 'b')
    count = 0
    for fpath, desc in targets:
        try:
            if use_bak:
                bak = fpath + '.bak'
                # Remove old .bak if it exists (don't accumulate)
                if os.path.exists(bak):
                    if os.path.isdir(bak):
                        shutil.rmtree(bak)
                    else:
                        os.remove(bak)
                os.rename(fpath, bak)
                print(f"    .bak  {desc}")
            else:
                if os.path.isdir(fpath):
                    shutil.rmtree(fpath)
                else:
                    os.remove(fpath)
                print(f"    del   {desc}")
            count += 1
        except Exception as e:
            print(f"    FAIL  {desc} — {e}")

    verb = "backed up" if use_bak else "deleted"
    print(f"\n  Done. {count}/{len(targets)} targets {verb}.")
    print(f"  Next: run 0041 at all temps, then AllStats with overwrite.")
    if use_bak:
        print(f"  To clean backups later: find {base_dir} -name '*.bak' -delete")
else:
    print("  Aborted — nothing touched.")
