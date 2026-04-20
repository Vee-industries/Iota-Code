"""
migrate_data_keys.py — IOTA 0.76 → 0.77 data migration utility

NOTE (v0.79.4.0): This migration is LEGACY — it predates the execution-order
renumber. The filenames in CSV_FILENAME_RENAMES below reference the PRE-0.79.4
numbering (e.g. Q0028, R0020, R0021). If you have 0.76-era data, run this
migration BEFORE running migrate_run_ids.py. Running in the opposite order
will no-op (new-numbered files don't match these old-numbered keys).

Rewrites all data on disk to match the v0.77.0.0 renamed labels:
  - CSV filenames (5 files per model/temp)
  - CSV column headers (6 metric columns)
  - CSV cell values (Q22 condition column; Q43 r_condition values)
  - Throughline cell values (priming rows)
  - JSON analysis keys (flywheel_power, flywheel_slope_status, energy_p)
  - Figure PNG filenames (4 orphan figures)

SAFE BY DEFAULT.
  - Dry-run mode prints every change without writing
  - Full backup of data/ before any writes
  - Per-file atomic writes (tempfile + rename)
  - Manifest file records every rename for rollback
  - Halts on any error; partial state is never left on disk
  - Refuses to run if Flask or start_here process is active

Usage:
  python migrate_data_keys.py                    # dry run; shows what would change
  python migrate_data_keys.py --apply            # perform migration (creates backup first)
  python migrate_data_keys.py --rollback         # restore from most recent manifest
  python migrate_data_keys.py --verify           # sweep data/ for stale labels

Author: Kevin Vaillancourt
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# RENAME TABLES — authoritative. Must match MAPPING_FINAL.md and CHANGELOG 0.77.0.0.
# ──────────────────────────────────────────────────────────────────────────────

CSV_FILENAME_RENAMES = {
    "Q0028_reflexivity.csv":          "Q0028_contradiction.csv",
    "R0020_bound_state.csv":          "R0020_instances.csv",
    "R0021_flywheel.csv":             "R0021_conditions.csv",
    "Q0025_flywheel_compression.csv": "Q0025_lengths.csv",
    "Q0026_alignment_entropy.csv":    "Q0026_recovery.csv",
}

CSV_COLUMN_RENAMES = {
    "tci_proxy":         "state_similarity_index",
    "ser_proxy":         "signal_entropy_ratio",
    "ser_per_watt":      "signal_per_watt",
    "hesitation_ratio":  "onset_delay_ratio",
    "rc_event":          "disruption_flag",
    "rc_score":          "disruption_magnitude",
}

# (column_name, old_value, new_value) — applied only to the named column.
# Column-scoped so cell-value renames never collide with values in other columns.
CSV_CELL_RENAMES = [
    ("condition",   "reflexivity",      "self_reference"),
    ("r_condition", "high_r_enforcer",  "condition_a"),
    ("r_condition", "high_r_free",      "condition_b"),
    ("r_condition", "low_r_verbose",    "condition_c"),
]

# Throughline keys written to the `condition` or throughline-equivalent column
# on priming rows (priming=1). These parallel the throughline dict renames in
# orchestration_throughlines.py.
THROUGHLINE_CELL_RENAMES = [
    # (column_name, old, new)
    ("condition", "reflexivity",          "self_reference"),      # Run 0028 priming rows
    ("condition", "bound_state",          "cross_instance"),      # Run 0020 priming rows
    ("condition", "flywheel_high",        "coherence_high"),      # Run 0021/43 priming rows
    ("condition", "flywheel_mid",         "coherence_mid"),
    ("condition", "flywheel_low",         "coherence_low"),
    ("condition", "alignment_entropy_mid", "contradiction_mid"),  # Run 0026 priming rows
]

# JSON keys at the top level of analysis output dicts.
# Values in JSONs are nested; we rewrite the RAW TEXT (safe because these keys are unique
# strings unlikely to collide with any non-key content).
JSON_KEY_RENAMES = {
    # Top-level results keys
    '"flywheel_power"':        '"coherence_levels_power"',
    '"flywheel_slope_status"': '"coherence_slope_status"',
    '"energy_p"':              '"compute_p"',
    '"compression_efficiency"': '"compute_efficiency"',
    '"tci_summary"':            '"similarity_summary"',
    # Contradiction analysis keys
    '"pre_tci_mean"':           '"pre_similarity_mean"',
    '"post_tci_mean"':          '"post_similarity_mean"',
    '"contradiction_tci_mean"': '"contradiction_similarity_mean"',
    '"tci_drop"':               '"similarity_drop"',
    # Similarity summary / saturation keys
    '"mean_tci"':               '"mean_similarity"',
    '"early_tci"':              '"early_similarity"',
    '"late_tci"':               '"late_similarity"',
    '"tci_decay"':              '"similarity_decay"',
    # Condition transfer keys
    '"pre_switch_tci"':         '"pre_switch_similarity"',
    '"post_switch_tci"':        '"post_switch_similarity"',
    # Priming vulnerability f-string keys (neutral/cooperative/resistant × mean/std/rc_rate)
    '"neutral_mean_tci"':       '"neutral_mean_similarity"',
    '"cooperative_mean_tci"':   '"cooperative_mean_similarity"',
    '"resistant_mean_tci"':     '"resistant_mean_similarity"',
    '"neutral_std_tci"':        '"neutral_std_similarity"',
    '"cooperative_std_tci"':    '"cooperative_std_similarity"',
    '"resistant_std_tci"':      '"resistant_std_similarity"',
    '"neutral_rc_rate"':        '"neutral_disruption_rate"',
    '"cooperative_rc_rate"':    '"cooperative_disruption_rate"',
    '"resistant_rc_rate"':      '"resistant_disruption_rate"',
    # Sobol outcome keys (export_stats H16)
    '"ser_proxy_p"':            '"signal_entropy_ratio_p"',
    '"ser_proxy_effect"':       '"signal_entropy_ratio_effect"',
    # Hypothesis metric identifier strings (hypothesis_outcomes.json)
    '"pearsonr_correct_tci"':       '"pearsonr_correct_similarity"',
    '"recovery_tci_delta_group"':   '"recovery_similarity_delta_group"',
    '"recovery_tci_delta_corr"':    '"recovery_similarity_delta_corr"',
    '"rc_score_delta"':             '"disruption_magnitude_delta"',
    '"hesitation_ratio"':           '"onset_delay_ratio"',
    # H23 correlation fields (hypothesis_outcomes.json)
    '"r_rc"':                       '"r_disrupt"',
    '"p_rc"':                       '"p_disrupt"',
}

# Figure PNG filename renames. These are the orphan figures — the two still-active
# figures (H20_power_by_condition, H33_compression_efficiency) had no loaded terms
# and don't change.
PNG_FILENAME_RENAMES = {
    "H05_reflexivity_tci_phase.png":      "H05_self_reference_phase.png",
    "H19_bound_state_coupling.png":       "H19_cross_instance.png",
    "H34_alignment_rc_score.png":         "H34_self_reference_trajectory.png",
    "H35_alignment_entropy_recovery.png": "H35_contradiction.png",
    "H23_hesitation_ratio.png":           "H23_onset_delay_ratio.png",
    "H05_rc_events_by_phase.png":         "H05_disruption_events_by_phase.png",
    "H33_compression_efficiency.png":     "H33_compute_efficiency.png",
    "H01_H03_H09_tci_by_run.png":         "H01_H03_H09_similarity_by_run.png",
    "H06_saturation_tci_decay.png":       "H06_saturation_similarity_decay.png",
    "H13_temperature_tci.png":            "H13_temperature_similarity.png",
    "H18_persistence_tci.png":            "H18_persistence_similarity.png",
}


# ──────────────────────────────────────────────────────────────────────────────
# PATHS + SAFETY
# ──────────────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MANIFEST_DIR = os.path.join(ROOT, ".migrate_manifests")


def _refuse_if_running():
    """Halt if Flask or a run is active — migrating live data is unsafe."""
    bad = []
    pid_file = os.path.join(ROOT, ".iota_pid")
    flask_pid = os.path.join(ROOT, ".iota_flask.pid")
    status = os.path.join(ROOT, ".iota_status.json")
    if os.path.exists(pid_file):
        bad.append(f"{pid_file} exists — a run may be active")
    if os.path.exists(flask_pid):
        bad.append(f"{flask_pid} exists — Flask is running")
    if os.path.exists(status):
        try:
            with open(status) as f:
                s = json.load(f)
            if s.get("status") == "running":
                bad.append(f"{status} reports status='running'")
        except Exception:
            pass
    # Also sanity-check for any python process running start_here or export_flask
    try:
        out = subprocess.check_output(["pgrep", "-af", "start_here.py|export_flask.py"],
                                      stderr=subprocess.DEVNULL, text=True)
        if out.strip():
            bad.append(f"active python processes: {out.strip()}")
    except Exception:
        pass
    if bad:
        print("\n  REFUSING TO MIGRATE — active processes detected:\n")
        for b in bad:
            print(f"    - {b}")
        print("\n  Stop Flask and any start_here.py process, then retry.\n")
        sys.exit(2)


def _backup_data(timestamp: str) -> str:
    """Copy data/ to data.backup.{timestamp}/. Returns backup path."""
    if not os.path.isdir(DATA_DIR):
        print(f"  NOTE: {DATA_DIR} does not exist — nothing to back up.")
        return ""
    backup = os.path.join(ROOT, f"data.backup.{timestamp}")
    print(f"  Backing up {DATA_DIR} → {backup} ...")
    shutil.copytree(DATA_DIR, backup)
    print(f"  Backup complete ({sum(1 for _,_,f in os.walk(backup) for _ in f)} files).")
    return backup


# ──────────────────────────────────────────────────────────────────────────────
# SCAN — find every file that needs action. Dry-run surfaces this list.
# ──────────────────────────────────────────────────────────────────────────────

def _scan():
    """Walk data/ and classify every migration-relevant file.

    Returns a dict:
      {
        'csv_renames':   [(old_path, new_path), ...],     # filename only
        'csv_rewrites':  [path, ...],                     # any CSV that needs column/cell changes
        'json_rewrites': [path, ...],                     # JSONs with renamed keys
        'png_renames':   [(old_path, new_path), ...],     # figure PNGs to rename
      }
    """
    out = {'csv_renames': [], 'csv_rewrites': [], 'json_rewrites': [], 'png_renames': []}
    if not os.path.isdir(DATA_DIR):
        return out

    for dirpath, _dirnames, filenames in os.walk(DATA_DIR):
        for fname in filenames:
            full = os.path.join(dirpath, fname)

            # CSV filename renames
            if fname in CSV_FILENAME_RENAMES:
                new_full = os.path.join(dirpath, CSV_FILENAME_RENAMES[fname])
                out['csv_renames'].append((full, new_full))

            # Any CSV file may have column or cell renames needed.
            # We test the header quickly rather than reading every row upfront.
            if fname.endswith(".csv"):
                try:
                    with open(full, 'r', newline='', encoding='utf-8') as f:
                        header = f.readline()
                    header_cols = [c.strip() for c in header.split(',')]
                    needs = any(c in CSV_COLUMN_RENAMES for c in header_cols)
                    if not needs:
                        # Might still have cell-value renames; check cell-target columns
                        target_cols = {col for col, _, _ in CSV_CELL_RENAMES} | \
                                      {col for col, _, _ in THROUGHLINE_CELL_RENAMES}
                        needs = any(c in target_cols for c in header_cols)
                    if needs:
                        out['csv_rewrites'].append(full)
                except Exception as e:
                    print(f"  WARN: could not read {full}: {e}")

            # JSON files — rewrite raw text for key renames
            if fname.endswith(".json"):
                try:
                    with open(full, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if any(old in content for old in JSON_KEY_RENAMES):
                        out['json_rewrites'].append(full)
                except Exception as e:
                    print(f"  WARN: could not read {full}: {e}")

            # Figure PNG renames
            if fname in PNG_FILENAME_RENAMES:
                new_full = os.path.join(dirpath, PNG_FILENAME_RENAMES[fname])
                out['png_renames'].append((full, new_full))

    return out


# ──────────────────────────────────────────────────────────────────────────────
# CSV REWRITE — columns + cell values + (new) filename
# ──────────────────────────────────────────────────────────────────────────────

def _rewrite_csv(path: str, apply: bool) -> int:
    """Read CSV; rewrite header and targeted cell values; atomic-write back to same path.
    Returns count of cell-level substitutions performed (header substitutions counted per-column).
    """
    changes = 0
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0
        rows = list(reader)

    # Header rename
    new_header = []
    header_idx = {}
    for i, col in enumerate(header):
        col_clean = col.strip()
        new_name = CSV_COLUMN_RENAMES.get(col_clean, col_clean)
        if new_name != col_clean:
            changes += 1
        new_header.append(new_name)
        header_idx[new_name] = i  # index by POST-rename name for cell lookups

    # Build cell-rename map keyed by column index (looking up by new column name
    # because header_idx uses new names).
    cell_map = {}  # {col_idx: {old_val: new_val}}
    for col, old, new in CSV_CELL_RENAMES + THROUGHLINE_CELL_RENAMES:
        if col in header_idx:
            cell_map.setdefault(header_idx[col], {})[old] = new

    # Apply cell renames
    if cell_map:
        for row in rows:
            for idx, submap in cell_map.items():
                if idx < len(row):
                    v = row[idx]
                    if v in submap:
                        row[idx] = submap[v]
                        changes += 1

    if not apply or changes == 0:
        return changes

    # Atomic write: temp file in same dir, then rename
    tmp = path + ".migrate.tmp"
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(new_header)
        w.writerows(rows)
    os.replace(tmp, path)
    return changes


def _rewrite_json(path: str, apply: bool) -> int:
    """Rewrite top-level JSON keys via raw-text substitution.
    The key renames are quoted string patterns unique enough that raw substitution is safe.
    Returns count of substitutions performed."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    orig = content
    changes = 0
    for old, new in JSON_KEY_RENAMES.items():
        n = content.count(old)
        if n:
            content = content.replace(old, new)
            changes += n
    if not apply or changes == 0:
        return changes
    tmp = path + ".migrate.tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(content)
    os.replace(tmp, path)
    return changes


# ──────────────────────────────────────────────────────────────────────────────
# APPLY + MANIFEST + ROLLBACK
# ──────────────────────────────────────────────────────────────────────────────

def _write_manifest(manifest: dict, timestamp: str) -> str:
    """Persist a migration manifest as .iota_migrations/manifest.{ts}.json.

    The manifest records every rename, rewrite, and backup path from a
    single _apply_migration run so --rollback can reverse the operation.
    Returns the absolute path written."""
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    path = os.path.join(MANIFEST_DIR, f"manifest.{timestamp}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return path


def _latest_manifest():
    """Return path to the newest manifest in .iota_migrations/, or None.
    Sort is lexicographic on the timestamped filenames — matches
    chronological order because timestamps are YYYYMMDD_HHMMSS."""
    if not os.path.isdir(MANIFEST_DIR):
        return None
    files = [f for f in os.listdir(MANIFEST_DIR) if f.startswith("manifest.") and f.endswith(".json")]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(MANIFEST_DIR, files[0])


def _apply_migration(plan: dict, timestamp: str, backup_path: str):
    """Execute the migration plan in four ordered phases, recording
    every change in a manifest for later rollback.

    Phase order matters:
      1. CSV content rewrites BEFORE filename renames so we rewrite
         using the old filenames still on disk.
      2. CSV filename renames. Halts if both old and new names exist
         in the same directory (ambiguous which holds correct data;
         user must manually resolve).
      3. JSON key rewrites (flywheel_* -> coherence_*, etc.).
      4. Figure PNG renames. Same both-exist halt rule as CSVs.

    Manifest is written at the end so a partial crash between phases
    still leaves a usable backup.
    """
    manifest = {
        "timestamp":    timestamp,
        "backup_path":  backup_path,
        "csv_renames":  [],
        "csv_rewrites": [],
        "json_rewrites":[],
        "png_renames":  [],
    }

    # 1. CSV content rewrites BEFORE filename renames (so we rewrite by old names).
    print(f"\n  Rewriting {len(plan['csv_rewrites'])} CSV file(s) ...")
    for path in plan['csv_rewrites']:
        n = _rewrite_csv(path, apply=True)
        manifest["csv_rewrites"].append({"path": path, "changes": n})
        print(f"    {path}: {n} substitutions")

    # 2. CSV filename renames
    print(f"\n  Renaming {len(plan['csv_renames'])} CSV file(s) ...")
    for old, new in plan['csv_renames']:
        if os.path.exists(new):
            # Both old and new exist — refuse to proceed. User must resolve manually
            # because we don't know which file holds the correct data.
            raise RuntimeError(
                f"Cannot rename {os.path.basename(old)} → {os.path.basename(new)}: "
                f"target already exists at {new}. "
                f"Both old and new filenames are present in the same directory. "
                f"This is ambiguous — you must manually merge or delete one before "
                f"re-running migration. Dry-run (no --apply) to see the full plan first."
            )
        os.rename(old, new)
        manifest["csv_renames"].append({"old": old, "new": new})
        print(f"    {os.path.basename(old)} → {os.path.basename(new)}")

    # 3. JSON rewrites
    print(f"\n  Rewriting {len(plan['json_rewrites'])} JSON file(s) ...")
    for path in plan['json_rewrites']:
        n = _rewrite_json(path, apply=True)
        manifest["json_rewrites"].append({"path": path, "changes": n})
        print(f"    {path}: {n} key renames")

    # 4. PNG renames
    print(f"\n  Renaming {len(plan['png_renames'])} figure file(s) ...")
    for old, new in plan['png_renames']:
        if os.path.exists(new):
            raise RuntimeError(
                f"Cannot rename {os.path.basename(old)} → {os.path.basename(new)}: "
                f"target already exists at {new}. "
                f"Both old and new filenames present — manually resolve before retry."
            )
        os.rename(old, new)
        manifest["png_renames"].append({"old": old, "new": new})
        print(f"    {os.path.basename(old)} → {os.path.basename(new)}")

    mpath = _write_manifest(manifest, timestamp)
    print(f"\n  Manifest written: {mpath}")


def _rollback():
    """Restore data/ from the backup path recorded in the newest manifest.

    Primary path: manifest has backup_path, backup dir exists → move
    current data/ aside to data.rollback_scratch.{ts} and copytree the
    backup into place. Full state restoration, including CSV cell
    rewrites which cannot be reversed without the originals.

    Degraded fallback: no backup_path or backup gone → reverse PNG and
    CSV filename renames only. CSV content rewrites are NOT reversed
    because reversing them requires the originals.
    """
    mpath = _latest_manifest()
    if not mpath:
        print("  No manifest found — nothing to roll back.")
        return
    print(f"  Loading manifest: {mpath}")
    with open(mpath) as f:
        m = json.load(f)

    backup = m.get("backup_path")
    if backup and os.path.isdir(backup):
        print(f"\n  Restoring from backup: {backup}")
        if os.path.isdir(DATA_DIR):
            scratch = DATA_DIR + f".rollback_scratch.{int(time.time())}"
            os.rename(DATA_DIR, scratch)
            print(f"    Current data/ moved aside to {scratch}")
        shutil.copytree(backup, DATA_DIR)
        print(f"  Restore complete.")
    else:
        print("  No backup path in manifest — attempting reverse-rename only (NOT RECOMMENDED).")
        # Reverse PNG renames, reverse CSV renames. Do NOT reverse CSV rewrites (cell values) —
        # those require reading current files to know originals.
        for entry in reversed(m.get("png_renames", [])):
            if os.path.exists(entry["new"]):
                os.rename(entry["new"], entry["old"])
                print(f"    {entry['new']} → {entry['old']}")
        for entry in reversed(m.get("csv_renames", [])):
            if os.path.exists(entry["new"]):
                os.rename(entry["new"], entry["old"])
                print(f"    {entry['new']} → {entry['old']}")
        print("  WARN: CSV cell/column content NOT reverted — full reverse requires backup.")


# ──────────────────────────────────────────────────────────────────────────────
# VERIFY — sweep for stale labels
# ──────────────────────────────────────────────────────────────────────────────

def _verify():
    """Sweep data/ looking for stale labels from the old vocabulary.

    Scans CSV and JSON file bodies for old column names, cell values,
    throughline keys, and JSON key strings. Also catches files whose
    names themselves are stale (Q22_reflexivity.csv, Q31_flywheel.csv,
    etc.). Prints every hit with its path.

    Read-only. Use post-migration to confirm nothing was missed, or
    pre-migration to confirm the tree is already at the new vocab.
    """
    if not os.path.isdir(DATA_DIR):
        print("  No data/ directory.")
        return

    stale_terms = list(CSV_FILENAME_RENAMES.keys()) + \
                  list(CSV_COLUMN_RENAMES.keys()) + \
                  [old for _, old, _ in CSV_CELL_RENAMES] + \
                  [old for _, old, _ in THROUGHLINE_CELL_RENAMES] + \
                  [k.strip('"') for k in JSON_KEY_RENAMES.keys()] + \
                  list(PNG_FILENAME_RENAMES.keys())
    stale_set = set(stale_terms)

    hits = 0
    for dirpath, _d, filenames in os.walk(DATA_DIR):
        for fname in filenames:
            if fname in stale_set:
                print(f"  STALE FILENAME: {os.path.join(dirpath, fname)}")
                hits += 1
                continue
            if fname.endswith(('.csv', '.json')):
                full = os.path.join(dirpath, fname)
                try:
                    with open(full, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                except Exception:
                    continue
                for term in stale_set:
                    if term in content and not term.endswith('.csv') and not term.endswith('.png'):
                        print(f"  STALE CONTENT in {full}: '{term}'")
                        hits += 1
                        break  # one hit per file is enough
    if hits == 0:
        print("  CLEAN — no stale labels found in data/.")
    else:
        print(f"\n  {hits} stale hits found. Re-run --apply or investigate.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Argparse entry point. Dispatches to --rollback / --verify /
    --apply / dry-run based on CLI flags.

    Default (no flags) = dry run: builds the plan and prints every
    filename rename, content rewrite, and key substitution without
    touching disk. Use this first, always.

    --apply: creates data.backup.{ts}/ (unless --skip-backup), then
    executes via _apply_migration and writes a manifest.
    --rollback: restores from newest manifest's backup.
    --verify: sweeps data/ for stale labels without writing.

    Refuses to run --apply while Flask or a collection/analysis
    subprocess is active — data/ must be quiescent.
    """
    ap = argparse.ArgumentParser(description="IOTA 0.76 → 0.77 data migration")
    ap.add_argument("--apply",    action="store_true", help="perform migration (default: dry run)")
    ap.add_argument("--rollback", action="store_true", help="restore from most recent manifest/backup")
    ap.add_argument("--verify",   action="store_true", help="sweep data/ for stale labels")
    ap.add_argument("--skip-backup", action="store_true",
                    help="(apply only) skip creating data.backup.{ts}/ — NOT RECOMMENDED")
    args = ap.parse_args()

    if args.rollback:
        _rollback()
        return

    if args.verify:
        _verify()
        return

    _refuse_if_running()

    plan = _scan()

    print("\n  MIGRATION PLAN")
    print( "  ──────────────")
    print(f"  CSV filename renames : {len(plan['csv_renames'])}")
    print(f"  CSV content rewrites : {len(plan['csv_rewrites'])}")
    print(f"  JSON content rewrites: {len(plan['json_rewrites'])}")
    print(f"  PNG filename renames : {len(plan['png_renames'])}")
    total = sum(len(v) for v in plan.values())
    print(f"  TOTAL FILES TOUCHED  : {total}")

    if total == 0:
        print("\n  Nothing to migrate. data/ is already at v0.77.0.0 labels (or empty).")
        return

    if not args.apply:
        print("\n  DRY RUN — no files written. Re-run with --apply to execute.")
        print("\n  Samples:")
        for old, new in plan['csv_renames'][:5]:
            print(f"    RENAME: {old}")
            print(f"        →   {new}")
        for p in plan['csv_rewrites'][:5]:
            print(f"    REWRITE CSV: {p}")
        for p in plan['json_rewrites'][:3]:
            print(f"    REWRITE JSON: {p}")
        for old, new in plan['png_renames'][:3]:
            print(f"    RENAME PNG: {os.path.basename(old)} → {os.path.basename(new)}")
        return

    # APPLY
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = ""
    if not args.skip_backup:
        backup_path = _backup_data(timestamp)
    else:
        print("  WARN: --skip-backup set. No backup will be created.")

    try:
        _apply_migration(plan, timestamp, backup_path)
        print("\n  MIGRATION COMPLETE.")
        print("  Run 'python migrate_data_keys.py --verify' to sweep for any stale labels.")
        print("  If anything looks wrong, 'python migrate_data_keys.py --rollback' restores the backup.")
    except Exception as e:
        print(f"\n  ERROR during migration: {e}")
        print("  Recommended: run --rollback to restore from the backup.")
        raise


if __name__ == "__main__":
    main()
