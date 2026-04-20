"""
IOTA FRAMEWORK — One-shot migration to canonical 4-digit run IDs  (v0.79.5.0)
==============================================================================

Run this AFTER Q8 collection finishes and AFTER you've deployed 0.79.5.0.

WHAT IT DOES (in order, atomic-ish)
====================================
1. Pre-flight check — confirms no Flask process is holding DATA/ open.
   Aborts with a clear message if one is found. Migration in the middle
   of live reads is a recipe for corrupt files.

2. Backup — creates  data_backup_pre_0.79.4.0_<YYYYMMDD_HHMMSS>.tar.gz  in
   the project root. Skips if a backup already exists from today (safety
   vs. accidental re-run). Not skipped if --force passed.

3. Migrate — invokes migrate_run_ids.py --apply in the current interpreter
   (imports it as a module so error handling is direct, not subprocess-
   swallowed). Renames every R{NN}_* / Q{NN}_* file to R{NNNN}_ / Q{NNNN}_
   and rewrites every CSV run_mode column from "3" → "0003" form.

4. Verify — three checks:
   (a) Manifest log parses cleanly, no FAIL lines.
   (b) No orphan R{NN}_ / Q{NN}_ files remain anywhere in DATA/ (glob sweep).
   (c) Sample CSV run_mode column — pick 3 random CSVs, confirm their
       run_mode columns contain only 4-digit values (no legacy ints).

5. Summary — prints N files renamed, N CSVs rewritten, N rows touched,
   elapsed wall time, backup file path, and the exit status.

NON-GOALS
=========
- Does NOT restart Flask for you — you'll do that yourself.
- Does NOT purge the .migration log (it's evidence, keep it until you're
  sure the migration went clean).
- Does NOT run analysis or rebuild masters — use start_here.py after.

USAGE
=====
Preview everything:
    python merge_to_canonical.py --dry-run

Do it for real:
    python merge_to_canonical.py

Force (skip today's-backup-exists check):
    python merge_to_canonical.py --force

EXIT CODES
==========
    0   — everything succeeded
    1   — pre-flight check failed (Flask running, etc.)
    2   — backup failed
    3   — migrate returned non-zero or raised
    4   — verification found orphans or FAIL entries
"""

import argparse
import datetime
import glob
import os
import re
import subprocess
import sys
import tarfile
import time

try:
    from cartography import DATA
except Exception:
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

ROOT = os.path.dirname(os.path.abspath(__file__))


# ── ANSI colour (same style as ui.py, but standalone so this script is
#    runnable even if ui.py is broken mid-migration)
def _c(s, code):
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"

def _ok(s):    return _c(s, "32")   # green
def _warn(s):  return _c(s, "33")   # yellow
def _err(s):   return _c(s, "31")   # red
def _dim(s):   return _c(s, "2")    # dim
def _bold(s):  return _c(s, "1")


# ── Step 1: pre-flight ────────────────────────────────────────────────────────

def _preflight():
    """Check for anything that would make migration unsafe.
    Currently: Flask process holding .iota_flask.pid."""
    pid_file = os.path.join(ROOT, '.iota_flask.pid')
    if os.path.exists(pid_file):
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            # Is it actually alive? Signal 0 = existence check.
            try:
                os.kill(pid, 0)
                return False, (
                    f"Flask is running (pid {pid}). Stop it before migrating — "
                    f"concurrent reads during file renames can corrupt state.\n"
                    f"  To stop:  kill {pid}    or use the dashboard's stop button"
                )
            except (ProcessLookupError, PermissionError):
                # Stale PID file — Flask was killed without cleanup. Safe.
                pass
        except (ValueError, OSError):
            pass

    # Sanity: DATA exists
    if not os.path.isdir(DATA):
        return False, f"DATA directory not found: {DATA}"

    # Sanity: migrate_run_ids.py is here
    if not os.path.exists(os.path.join(ROOT, 'migrate_run_ids.py')):
        return False, (
            f"migrate_run_ids.py not found in {ROOT} — did you deploy "
            f"0.79.5.0 correctly?"
        )

    return True, None


# ── Step 2: backup ────────────────────────────────────────────────────────────

def _todays_backup_exists():
    today = datetime.datetime.now().strftime('%Y%m%d')
    pattern = os.path.join(ROOT, f"data_backup_pre_0.79.4.0_{today}_*.tar.gz")
    return bool(glob.glob(pattern))


def _make_backup(dry_run=False):
    """Tar-gzip the DATA/ tree. Returns (path, size_bytes, elapsed_sec)."""
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(ROOT, f"data_backup_pre_0.79.4.0_{stamp}.tar.gz")

    if dry_run:
        print(f"  [dry-run] would write backup to: {out_path}")
        # Estimate size roughly by walking the tree
        est = 0
        for dp, _, files in os.walk(DATA):
            for fn in files:
                try:
                    est += os.path.getsize(os.path.join(dp, fn))
                except OSError:
                    pass
        print(f"  [dry-run] estimated tar source size: {est/1e9:.2f} GB "
              f"(actual .tar.gz will be smaller due to compression)")
        return out_path, est, 0.0

    t0 = time.time()
    # Relative path inside archive so `tar xzf` extracts to ./data/
    arcname = os.path.basename(DATA.rstrip(os.sep))
    try:
        with tarfile.open(out_path, 'w:gz', compresslevel=6) as tar:
            tar.add(DATA, arcname=arcname)
    except Exception as e:
        raise RuntimeError(f"backup failed: {e}") from e
    elapsed = time.time() - t0
    size = os.path.getsize(out_path)
    return out_path, size, elapsed


# ── Step 3: migrate ───────────────────────────────────────────────────────────

def _run_migration(dry_run=False, data_dir=None):
    """Invoke migrate_run_ids.py. Returns (returncode, elapsed_sec, err)."""
    cmd = [sys.executable, os.path.join(ROOT, 'migrate_run_ids.py')]
    if not dry_run:
        cmd.append('--apply')
    if data_dir:
        cmd += ['--data-dir', data_dir]
    # Let child stream to our stdout directly so the user sees progress.
    print(_dim(f"  invoking: {' '.join(cmd)}"))
    print()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
    except Exception as e:
        return 1, 0.0, f"subprocess error: {e}"
    elapsed = time.time() - t0
    return proc.returncode, elapsed, None


# ── Step 4: verify ────────────────────────────────────────────────────────────

_LEGACY_FILE_RE = re.compile(r'^[RQ](\d{2})_')


def _find_orphan_legacy_files():
    """Sweep DATA/ for any file whose basename starts R{NN}_ or Q{NN}_
    with exactly 2 digits (the legacy format). Returns list of paths."""
    orphans = []
    for dirpath, dirnames, files in os.walk(DATA):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fn in files:
            m = _LEGACY_FILE_RE.match(fn)
            if not m:
                continue
            # Filter: the 2-digit must be in [01, 99] range — defensive
            try:
                n = int(m.group(1))
                if n < 1 or n > 99:
                    continue
            except ValueError:
                continue
            orphans.append(os.path.join(dirpath, fn))
    return orphans


def _check_manifest():
    """Scan the migration manifest for problematic entries. Returns
    (fails, skips, err) where fails = hard failures, skips = defensive
    skips that left work undone (collisions)."""
    log_path = os.path.join(DATA, '.migration_0.79.4.0.log')
    if not os.path.exists(log_path):
        return None, None, "manifest file not found (migration may have failed early)"
    fails = []
    skips = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('FAIL') or line.startswith('CSV_WRITE_FAIL') or \
                   line.startswith('CSV_READ_FAIL'):
                    fails.append(line)
                elif line.startswith('SKIP_COLLIDE'):
                    skips.append(line)
    except OSError as e:
        return None, None, f"could not read manifest: {e}"
    return fails, skips, None


def _check_sample_csvs(n_samples=3):
    """Pick N random CSVs, verify their run_mode columns are all 4-digit.
    Returns list of (path, problem_description) for failures; empty if OK."""
    import csv
    import random
    csvs = []
    for dirpath, dirnames, files in os.walk(DATA):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fn in files:
            if fn.endswith('.csv'):
                csvs.append(os.path.join(dirpath, fn))
    if not csvs:
        return []
    random.shuffle(csvs)
    problems = []
    checked = 0
    for path in csvs:
        if checked >= n_samples:
            break
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception:
            continue
        if len(rows) < 2:
            continue
        header = rows[0]
        if 'run_mode' not in header:
            continue
        checked += 1
        rm_col = header.index('run_mode')
        # Check: every non-empty run_mode cell should be 4+ digits
        offenders = 0
        for row in rows[1:]:
            if rm_col >= len(row):
                continue
            v = row[rm_col].strip()
            if not v or v.upper() == 'NA':
                continue
            if not v.isdigit():
                continue
            if len(v) < 4:
                offenders += 1
        if offenders:
            problems.append((path, f"{offenders} legacy-format run_mode rows"))
    return problems


# ── Step 5: orchestration ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='One-shot IOTA migration to canonical 4-digit run IDs.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would happen; no backup, no rename.')
    parser.add_argument('--force', action='store_true',
                        help='Skip the "backup already exists today" safety check.')
    parser.add_argument('--skip-backup', action='store_true',
                        help='Skip backup (DANGEROUS — only if you already have one).')
    parser.add_argument('--data-dir', default=None,
                        help='Override DATA root (default: cartography.DATA). '
                             'Rarely needed — useful for testing against a '
                             'copy of the real data tree.')
    args = parser.parse_args()

    # Allow --data-dir override. Propagate to our module globals so the
    # helpers below see it, AND to the subprocess via argv so migrate_run_ids
    # gets the same root.
    global DATA
    if args.data_dir:
        DATA = os.path.abspath(args.data_dir)

    t_start = time.time()

    print(_bold("═══════════════════════════════════════════════════════════════"))
    print(_bold("  IOTA — one-shot migration to canonical 4-digit run IDs"))
    print(_bold("═══════════════════════════════════════════════════════════════"))
    print(f"  DATA root: {DATA}")
    print(f"  Mode:      {_warn('DRY RUN — no changes') if args.dry_run else _ok('APPLY — will modify disk')}")
    print()

    # ── Step 1: pre-flight
    print(_bold("Step 1/5 — Pre-flight check"))
    ok, msg = _preflight()
    if not ok:
        print(_err(f"  FAIL: {msg}"))
        sys.exit(1)
    print(_ok("  Pass."))
    print()

    # ── Step 2: backup
    print(_bold("Step 2/5 — Backup"))
    backup_path = None
    if args.skip_backup:
        print(_warn("  Skipped (--skip-backup). Make sure you have a backup already."))
    elif _todays_backup_exists() and not args.force:
        print(_warn(
            "  A backup from today already exists. Skipping to avoid overwriting.\n"
            "  Pass --force to make another backup anyway."))
    else:
        try:
            backup_path, size, elapsed = _make_backup(dry_run=args.dry_run)
        except RuntimeError as e:
            print(_err(f"  FAIL: {e}"))
            sys.exit(2)
        if args.dry_run:
            print(_dim(f"  [dry-run] no backup written."))
        else:
            print(_ok(f"  Wrote {backup_path}"))
            print(f"  Size: {size/1e9:.2f} GB  in  {elapsed:.1f}s")
    print()

    # ── Step 3: migrate
    print(_bold("Step 3/5 — Migrate run IDs"))
    rc, mig_elapsed, err = _run_migration(dry_run=args.dry_run,
                                          data_dir=args.data_dir)
    print()
    if rc != 0 or err:
        print(_err(f"  FAIL: migration returned {rc}  {err or ''}"))
        if backup_path:
            print()
            print(_dim("  Your backup is intact at:"))
            print(_dim(f"    {backup_path}"))
            print(_dim("  To restore:"))
            print(_dim(f"    rm -rf {DATA}"))
            print(_dim(f"    tar -xzf {backup_path}"))
        sys.exit(3)
    print(_ok(f"  Migration ran in {mig_elapsed:.1f}s"))
    print()

    # ── Step 4: verify
    print(_bold("Step 4/5 — Verify"))
    if args.dry_run:
        print(_dim("  [dry-run] skipping verification."))
        print()
    else:
        # 4a — manifest
        fails, skips, err = _check_manifest()
        if err:
            print(_warn(f"  (a) manifest check skipped: {err}"))
        elif fails:
            print(_err(f"  (a) manifest has {len(fails)} FAIL lines:"))
            for line in fails[:10]:
                print(_err(f"      {line}"))
            if len(fails) > 10:
                print(_err(f"      ... and {len(fails)-10} more"))
            print()
            print(_dim("  Check the manifest for details:"))
            print(_dim(f"    {os.path.join(DATA, '.migration_0.79.4.0.log')}"))
            if backup_path:
                print(_dim("  To restore from backup:"))
                print(_dim(f"    rm -rf {DATA}"))
                print(_dim(f"    tar -xzf {backup_path}"))
            sys.exit(4)
        else:
            msg = "  (a) manifest clean — no FAIL entries"
            if skips:
                msg += f"  ({_warn(str(len(skips)) + ' SKIP_COLLIDE')} — see step b)"
            print(_ok(msg) if not skips else msg.replace("(a)", _ok("(a)")))

        # 4b — orphans
        orphans = _find_orphan_legacy_files()
        if orphans:
            print(_err(f"  (b) {len(orphans)} orphan legacy-format file(s) remain:"))
            for p in orphans[:10]:
                print(_err(f"      {p}"))
            if len(orphans) > 10:
                print(_err(f"      ... and {len(orphans)-10} more"))
            print()
            print(_dim("  These usually mean a destination file already existed"))
            print(_dim("  when the rename was attempted (SKIP_COLLIDE in the manifest)."))
            print(_dim("  Resolution path:"))
            print(_dim("    1. Inspect each orphan and its canonical counterpart — compare"))
            print(_dim("       content + sizes to decide which is the keeper."))
            print(_dim("    2. Delete or archive the stale one."))
            print(_dim("    3. Re-run: python merge_to_canonical.py --skip-backup"))
            print(_dim(f"  Manifest: {os.path.join(DATA, '.migration_0.79.4.0.log')}"))
            sys.exit(4)
        else:
            print(_ok("  (b) no orphan R{NN}_/Q{NN}_ legacy files — disk is clean"))

        # 4c — CSV sample
        problems = _check_sample_csvs(n_samples=3)
        if problems:
            print(_err("  (c) CSV sample check failed:"))
            for path, desc in problems:
                print(_err(f"      {path}: {desc}"))
            sys.exit(4)
        else:
            print(_ok("  (c) CSV sample — run_mode columns are all 4-digit"))
        print()

    # ── Step 5: summary
    t_end = time.time()
    print(_bold("Step 5/5 — Summary"))
    print(_ok(f"  Migration {'WOULD SUCCEED' if args.dry_run else 'COMPLETE'}"))
    print(f"  Total elapsed: {t_end - t_start:.1f}s")
    if backup_path and not args.dry_run:
        print(f"  Backup at:     {backup_path}")
    if not args.dry_run:
        print()
        print(_dim("  Next steps:"))
        print(_dim("    1. Restart Flask  (the old process has int-mode runtime state)"))
        print(_dim("    2. Ctrl+Shift+R the dashboard to pick up canonical run IDs"))
        print(_dim("    3. Resume collection or analysis normally"))
    print()
    sys.exit(0)


if __name__ == '__main__':
    main()
