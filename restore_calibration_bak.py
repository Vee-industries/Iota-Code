"""
restore_calibration_bak.py -- restore cached calibration outputs from a backup.

When Run 0056 detects a stale calibration (upstream hash mismatch), it moves
the current outputs to data/paper/calibration/{name}/_bak/{timestamp}/ before
regenerating. If you decide the upstream change didn't actually affect
calibration, use this script to restore without re-running.

Usage:
  python restore_calibration_bak.py            # interactive picker
  python restore_calibration_bak.py --list     # just list, no restore
"""

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CAL_ROOT = os.path.join(ROOT, 'data', 'paper', 'calibration')

_NAMES = ['v5', 'ridge_bias', 'toy_nonlinearity', 'channel_marginal']


def _list_backups():
    entries = []
    for name in _NAMES:
        bak_root = os.path.join(CAL_ROOT, name, '_bak')
        if not os.path.isdir(bak_root):
            continue
        for stamp in sorted(os.listdir(bak_root)):
            bak_dir = os.path.join(bak_root, stamp)
            if os.path.isdir(bak_dir):
                entries.append((name, stamp, bak_dir))
    return entries


def _restore(name, stamp, bak_dir):
    cal_dir = os.path.join(CAL_ROOT, name)
    # Move current cal_dir contents out of the way (if any non-_bak contents)
    # to a failsafe _pre_restore_bak/{now}/
    import datetime
    failsafe_root = os.path.join(cal_dir, '_bak',
        '_pre_restore_' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(failsafe_root, exist_ok=True)
    moved_any = False
    for item in os.listdir(cal_dir):
        if item == '_bak':
            continue
        src = os.path.join(cal_dir, item)
        dst = os.path.join(failsafe_root, item)
        try:
            shutil.move(src, dst)
            moved_any = True
        except Exception as e:
            print(f"  ! failed to move {src}: {e}", file=sys.stderr)
    if not moved_any:
        # No live contents to preserve; remove empty failsafe
        try: os.rmdir(failsafe_root)
        except Exception: pass

    # Copy bak contents back up
    for item in os.listdir(bak_dir):
        src = os.path.join(bak_dir, item)
        dst = os.path.join(cal_dir, item)
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            print(f"  ! failed to restore {src}: {e}", file=sys.stderr)
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true',
                    help='list backups and exit')
    args = ap.parse_args()

    if not os.path.isdir(CAL_ROOT):
        print(f"! no calibration root: {CAL_ROOT}", file=sys.stderr)
        return 1

    entries = _list_backups()
    if not entries:
        print("No backups found.")
        return 0

    print("Available calibration backups:")
    for i, (name, stamp, path) in enumerate(entries, 1):
        print(f"  [{i:2d}] {name:20s}  {stamp}")

    if args.list:
        return 0

    resp = input("\nRestore which? (number, 'all', 'q' to quit): ").strip()
    if resp.lower() in ('q', 'quit', 'exit', ''):
        print("Cancelled.")
        return 0

    if resp.lower() == 'all':
        targets = entries
    else:
        try:
            i = int(resp)
            if i < 1 or i > len(entries):
                print(f"! out of range: {i}", file=sys.stderr)
                return 1
            targets = [entries[i - 1]]
        except ValueError:
            print(f"! not a number: {resp!r}", file=sys.stderr)
            return 1

    for name, stamp, bak_dir in targets:
        print(f"Restoring {name} from {stamp} ...", end='', flush=True)
        ok = _restore(name, stamp, bak_dir)
        print(" done." if ok else " FAILED.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
