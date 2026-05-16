"""
_clear_for_recollection.py — surgically move a specific model's run outputs
aside so a fresh collection can fire without the resume logic seeing stale rows.

Files moved (per run, per affected variant + temp dir):
  csv/R{NNNN}_*.csv               (CSV with trial rows the resume logic reads)
  hidden_states/{R,Q}{NNNN}_*.npy (hidden-state outputs)
  hidden_states/R0001_*.stamp     (Pass 4 sentinel; only Run 0001)
  hidden_states/R0001_*_global_mean.npy  (only Run 0001)

For Run 0001 (three-model trivariant) the script also clears the base/ and
instruct/ sibling variant dirs.

Files go to:
  <data_root>/_archive_pre_v1_recollection_<timestamp>/...

Reversible by moving them back; data is never deleted.

Usage:
  python _clear_for_recollection.py --family gemma --size 2b_4bit \
         --variant abliterated --runs 1 3 39
"""
import argparse
import shutil
import sys
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:/Users/Gaming/Desktop/iota active")
DATA = ROOT / "data"

# Run 0001 is the trivariant null baseline (base / instruct / abliterated
# at T=0). It writes only to deterministic/. Pass 4 computes E_t and C_t
# vectors and drops them in the same deterministic/ hidden_states dir.
# It does NOT fan out to temp_X.X/. Any R0001 files found in temp_X.X
# are stale v28 leftovers and inert -- nothing reads them.
#
# Runs 0003 and 0039 do write to all six temp dirs:
#   R0003 -- internally iterates 5 nonzero temperatures + deterministic.
#   R0039 -- fired per-T, one temp dir per invocation.
#
# Earlier TEMP_INDEP={1,2,3} was wrong for R0003 and caused stale v28
# files to survive in temp_X.X/ across re-collection.
TEMP_INDEP = {1}  # only Run 1 is deterministic-only
TEMP_DIRS = ["deterministic", "temp_0.2", "temp_0.4", "temp_0.6",
             "temp_0.8", "temp_1.0"]


def variants_for_run(run_id):
    """Determine which variant dirs (base / instruct / abliterated) hold
    output for a given run.

    - Run 0001 is the trivariant null isolation; all three variants are
      first-class outputs (E_t from base, C_t from abl - base).
    - Runs 0002, 0004-0015, 0023 write E_t/C_t source data via the
      _standard_trial_loop save_embeddings path, which writes to the
      base and instruct sibling dirs in addition to abliterated.
    - All other runs (0003, 0017+, 0039, ...) write only to abliterated.
    """
    if run_id == 1:
        return ["base", "instruct", "abliterated"]
    if run_id in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23):
        return ["base", "instruct", "abliterated"]
    return ["abliterated"]


def temp_dirs_for_run(run_id):
    return ["deterministic"] if run_id in TEMP_INDEP else TEMP_DIRS


def run_prefix(run_id):
    return "R" if run_id <= 21 else "Q"


def move_run_files(family, size, variant, run_id, archive_root):
    """Move all files for one (variant, run) tuple to the archive."""
    pfx_npy = run_prefix(run_id)  # R or Q for the .npy filename prefix
    rid = f"{run_id:04d}"
    moved = 0
    for temp_dir in temp_dirs_for_run(run_id):
        src_base = DATA / family / size / variant / temp_dir
        if not src_base.exists():
            continue
        dst_base = archive_root / family / size / variant / temp_dir
        # CSVs (always R-prefixed for CSV, regardless of NPY prefix)
        csv_dir = src_base / "csv"
        if csv_dir.exists():
            for f in csv_dir.glob(f"R{rid}_*.csv"):
                dst = dst_base / "csv" / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst))
                moved += 1
            # also catch .csv.bak siblings
            for f in csv_dir.glob(f"R{rid}_*.csv.bak"):
                dst = dst_base / "csv" / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst))
                moved += 1
        # Hidden states: .npy + .stamp + global_mean variants
        hs_dir = src_base / "hidden_states"
        if hs_dir.exists():
            for pat in [f"{pfx_npy}{rid}_*.npy",
                        f"R{rid}_*.npy",        # in case prefix mismatch
                        f"R{rid}_*.stamp",
                        f"R{rid}_*_global_mean.npy"]:
                for f in hs_dir.glob(pat):
                    dst = dst_base / "hidden_states" / f.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dst))
                    moved += 1
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--size", required=True)  # e.g. 2b_4bit
    ap.add_argument("--variant", required=True)  # e.g. abliterated
    ap.add_argument("--runs", type=int, nargs="+", required=True)
    ap.add_argument("--archive-root", default=None,
                    help="Override archive destination; default: "
                         "data/_archive_pre_v1_recollection_<timestamp>")
    args = ap.parse_args()

    if args.archive_root:
        archive_root = Path(args.archive_root)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        archive_root = DATA / f"_archive_pre_v1_recollection_{ts}"
    archive_root.mkdir(parents=True, exist_ok=True)

    print(f"Surgical clear for {args.family}/{args.size}/{args.variant}")
    print(f"  Runs:    {args.runs}")
    print(f"  Archive: {archive_root}")
    print()

    total = 0
    for run_id in args.runs:
        for variant in variants_for_run(run_id):
            if run_id == 1 or variant == args.variant:
                # Run 0001: clear all three sibling variants regardless of
                # the requested --variant. For other runs: only the requested.
                n = move_run_files(args.family, args.size, variant,
                                   run_id, archive_root)
                print(f"  Run {run_id:04d} / {variant}: {n} files moved")
                total += n
    print()
    print(f"Total: {total} files moved to {archive_root}")
    print("Reversible: move files back from archive to their original paths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
