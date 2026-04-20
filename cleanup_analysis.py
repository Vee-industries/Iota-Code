#!/usr/bin/env python
"""
IOTA FRAMEWORK — Analysis Cleanup
===================================
Deletes specified analysis JSONs so all-stats recomputes them with
correct per-temperature POOL_DIM calibration.

Usage:
    python cleanup_analysis.py              # interactive — shows what will be deleted
    python cleanup_analysis.py --confirm    # skip confirmation, just do it
    python cleanup_analysis.py --backup     # backup before deleting
"""

import os, sys, glob, shutil, argparse, datetime

def _find_root():
    """Walk up from this file's directory until we find the iota root
    (directory containing start_here.py). Allows the script to be run
    from anywhere in the tree."""
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
sys.path.insert(0, ROOT)

from cartography import DATA, get_family_size_dir

# Files to delete per temperature directory
PER_TEMP_DELETE = [
    "Q0042_decomposition.json",
    "Q0043_sobol_partition.json",
    "Q0044_per_condition_R.json",
]

# Files to delete in pooled directory
POOLED_DELETE = [
    "Q0051_pooled_decomposition.json",
    "Q0051_pooled_sobol.json",
    "Q0051_temperature_curve.json",
    "Q0050_cross_temp_synthesis.json",
]

# Temperatures to skip (already computed correctly)
SKIP_TEMPS = set()  # Add temp floats here to skip, e.g. {1.0}


def find_session():
    """Load last_session.json from the iota root. Returns the raw dict,
    or an empty dict if the file is absent or unreadable. Used to pick
    up the model family/size/variant the user last selected — so the
    cleanup operates on the currently-active model's data tree."""
    sess_path = os.path.join(ROOT, 'last_session.json')
    if os.path.exists(sess_path):
        import json
        with open(sess_path) as f:
            return json.load(f)
    return {}


def main():
    """Interactive analysis-JSON cleaner for recomputation workflows.

    Walks the active model's data tree and flags the following outputs
    for deletion so the next all-stats run recomputes them with
    current POOL_DIM calibration and current code paths:

      per-temperature analysis/:
        Q33_decomposition.json, Q34_sobol_partition.json,
        Q46_per_condition_R.json

      pooled/analysis/:
        Q40_*.json, Q47_cross_temp_synthesis.json,
        REPORT.md + all generated figure PNGs

    --skip-temps preserves select temperatures already computed
    correctly. --backup copies each file to data/.../.bak/{ts}/ before
    deletion so the operation is recoverable.
    """
    parser = argparse.ArgumentParser(description="Clean analysis JSONs for recomputation")
    parser.add_argument('--confirm', action='store_true', help='Skip confirmation')
    parser.add_argument('--backup', action='store_true', help='Backup files before deleting')
    parser.add_argument('--skip-temps', nargs='*', type=float, default=[],
                        help='Temperatures to skip (e.g. --skip-temps 1.0)')
    args = parser.parse_args()

    skip = set(args.skip_temps) | SKIP_TEMPS

    session = find_session()
    family  = session.get('model_family', 'llama')
    size    = session.get('model_size', '8b')
    variant = session.get('model_variant', 'abliterated')

    base_dir = os.path.join(get_family_size_dir(family, size), variant)
    if not os.path.isdir(base_dir):
        print(f"  No data directory: {base_dir}")
        return

    _TEMPS = {
        'deterministic': 0.0, 'temp_0.2': 0.2, 'temp_0.4': 0.4,
        'temp_0.6': 0.6, 'temp_0.8': 0.8, 'temp_1.0': 1.0,
    }

    to_delete = []
    to_keep = []

    # Per-temperature files
    for cond in sorted(os.listdir(base_dir)):
        if cond == 'pooled':
            continue
        temp = _TEMPS.get(cond)
        if temp is None:
            continue
        ana_dir = os.path.join(base_dir, cond, 'analysis')
        if not os.path.isdir(ana_dir):
            continue

        if temp in skip:
            for fname in PER_TEMP_DELETE:
                fpath = os.path.join(ana_dir, fname)
                if os.path.exists(fpath):
                    to_keep.append((fpath, f"SKIP T={temp}"))
            continue

        for fname in PER_TEMP_DELETE:
            fpath = os.path.join(ana_dir, fname)
            if os.path.exists(fpath):
                to_delete.append(fpath)

    # Pooled files
    pooled_ana = os.path.join(base_dir, 'pooled', 'analysis')
    if os.path.isdir(pooled_ana):
        for fname in POOLED_DELETE:
            fpath = os.path.join(pooled_ana, fname)
            if os.path.exists(fpath):
                to_delete.append(fpath)

    # Also delete report figures since they'll be stale
    figures_dir = os.path.join(pooled_ana, 'figures')
    if os.path.isdir(figures_dir):
        for f in glob.glob(os.path.join(figures_dir, '*.png')):
            to_delete.append(f)
        report_md = os.path.join(pooled_ana, 'REPORT.md')
        if os.path.exists(report_md):
            to_delete.append(report_md)

    print(f"\n  IOTA Analysis Cleanup")
    print(f"  Model: {family}/{size}/{variant}")
    print(f"  Skipping temperatures: {sorted(skip) if skip else 'none'}")
    print()

    if to_keep:
        print(f"  KEEPING ({len(to_keep)} files):")
        for fpath, reason in to_keep:
            print(f"    {reason}: {os.path.basename(fpath)}")
        print()

    if not to_delete:
        print("  Nothing to delete.")
        return

    print(f"  WILL DELETE ({len(to_delete)} files):")
    for fpath in to_delete:
        rel = os.path.relpath(fpath, base_dir)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {rel}  ({size_kb:.1f} KB)")
    print()

    if not args.confirm:
        print("  Options:")
        print("    [d] Delete and continue")
        print("    [b] Backup to .bak/ then delete")
        print("    [q] Quit without deleting")
        print()
        choice = input("  Choice [d/b/q]: ").strip().lower()
        if choice == 'q':
            print("  Cancelled.")
            return
        if choice == 'b':
            args.backup = True
        elif choice != 'd':
            print("  Invalid choice. Cancelled.")
            return

    if args.backup:
        bak_dir = os.path.join(base_dir, '.bak',
                               datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(bak_dir, exist_ok=True)
        for fpath in to_delete:
            rel = os.path.relpath(fpath, base_dir)
            dest = os.path.join(bak_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(fpath, dest)
        print(f"  Backed up {len(to_delete)} files to {bak_dir}")

    deleted = 0
    for fpath in to_delete:
        try:
            os.remove(fpath)
            deleted += 1
        except Exception as e:
            print(f"  ERROR: {fpath}: {e}")

    print(f"\n  Deleted {deleted}/{len(to_delete)} files.")
    print(f"  Run all-stats to recompute with per-temperature POOL_DIM.")
    print()


if __name__ == '__main__':
    main()
