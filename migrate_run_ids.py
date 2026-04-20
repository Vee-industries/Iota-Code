"""
IOTA FRAMEWORK — Run ID Migration Utility  (v0.79.4.0 — RENUMBER EDITION)
===========================================================================

v0.79.4.0 renumbers every run to its execution-order position. This is a
BIJECTIVE REMAP, not a zero-pad. Old run 19 → new 0001, old run 1 →
new 0004, old run 56 → new 0046, etc.

WHAT IT DOES
============

1. **File rename.** Walks DATA/ and renames every legacy file to its
   new 4-digit canonical ID. Handles BOTH the original 2-digit format
   (R03_*) AND the earlier-arc 4-digit pad (R0003_*). New R/Q phase
   prefix: R for new 0001-0021, Q for 0022+.

2. **CSV run_mode rewrite.** For every CSV with a run_mode column, old
   numeric value → new 4-digit canonical ID. Atomic writes.

3. **JSON content rewrite.** Walks every .json in DATA/ and remaps
   string keys that look like run-IDs, plus values under run-ID-named
   keys (run_id, run_num, source_run, ref_run, etc).

SAFETY
======
- Dry-run by default. Pass --apply to make changes.
- Manifest log at DATA/.migration_0.79.4.0.log.
- Atomic writes; no half-modified files.
- Idempotent: already-canonical files/rows/keys are skipped.
"""

import argparse
import csv
import json
import os
import re
import sys

try:
    from cartography import DATA
except Exception:
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ── OLD → NEW mapping (execution-order renumber) ─────────────────────
OLD_TO_NEW = {
    # Phase A — temp-indep
    19:  1,  20:  2,  26:  3,
    # Phase B — ET-source
    1:   4,  2:   5,  3:   6,  4:   7,  5:   8,
    6:   9,  7:  10,  8:  11,  9:  12,
    15: 13, 16: 14, 17: 15,
    # Phase G — E_t meta-run
    48: 16,
    # Phase D — patching cluster
    21: 17, 42: 18, 53: 19,
    # Phase EFC — slow → fast
    30: 20, 31: 21, 23: 22, 28: 23, 29: 24,
    43: 25, 44: 26, 24: 27, 22: 28,
    12: 29, 13: 30, 14: 31, 18: 32,
    41: 33, 36: 34, 39: 35, 38: 36, 37: 37, 35: 38,
    10: 39, 11: 40,
    # Per-temp analysis (chain)
    45: 41, 33: 42, 34: 43, 46: 44, 49: 45, 56: 46,
    # Per-temp analysis (indep)
    25: 47, 27: 48, 32: 49,
    # Pooled
    47: 50, 40: 51,
    # Cross-model
    50: 52, 51: 53, 52: 54,
    # Output
    54: 55, 55: 56,
}

assert len(OLD_TO_NEW) == 56
assert set(OLD_TO_NEW.keys()) == set(range(1, 57))
assert set(OLD_TO_NEW.values()) == set(range(1, 57))

_NEW_VALUES = set(OLD_TO_NEW.values())


def _new_pfx(new_n):
    return "R" if new_n <= 21 else "Q"


# ── Phase 1: file renames ────────────────────────────────────────────

_FNAME_RE = re.compile(r'^([RQ])(\d{2,4})(_.+)$')


def _propose_rename(basename):
    """Return new basename, or None if not a legacy run file or already
    canonical. Accepts 2-digit OR 4-digit-old-number form."""
    m = _FNAME_RE.match(basename)
    if not m:
        return None
    pfx, digits, rest = m.group(1), m.group(2), m.group(3)
    try:
        n = int(digits)
    except ValueError:
        return None
    # If 4-digit and already in image of map AND prefix is correct
    # for that new number, it's canonical new — skip.
    if len(digits) >= 4 and n in _NEW_VALUES:
        if pfx == _new_pfx(n):
            return None
        # Wrong prefix for new number — rewrite prefix.
        return f"{_new_pfx(n)}{n:04d}{rest}"
    # Otherwise, n should be a legacy OLD id.
    if n not in OLD_TO_NEW:
        return None
    new_n = OLD_TO_NEW[n]
    return f"{_new_pfx(new_n)}{new_n:04d}{rest}"


def _walk(root):
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        yield dirpath


def _rename_files(root, apply, verbose, log_lines):
    n_renamed = n_skipped = n_already = 0
    for dirpath in _walk(root):
        try:
            entries = os.listdir(dirpath)
        except OSError:
            continue
        to_do = []
        for name in entries:
            src = os.path.join(dirpath, name)
            if not os.path.isfile(src):
                continue
            new = _propose_rename(name)
            if new is None:
                # Could be either a non-run file or already canonical
                m = _FNAME_RE.match(name)
                if m and len(m.group(2)) >= 4:
                    try:
                        n = int(m.group(2))
                        if n in _NEW_VALUES and m.group(1) == _new_pfx(n):
                            n_already += 1
                    except ValueError:
                        pass
                continue
            to_do.append((src, new, name))
        for src, new, name in to_do:
            dst = os.path.join(dirpath, new)
            if os.path.exists(dst):
                log_lines.append(f"SKIP_COLLIDE\t{src}\t{dst}")
                n_skipped += 1
                if verbose:
                    print(f"  SKIP (exists): {name} → {new}")
                continue
            log_lines.append(f"RENAME\t{src}\t{dst}")
            if verbose:
                print(f"  {name} → {new}")
            if apply:
                try:
                    os.rename(src, dst)
                except OSError as e:
                    log_lines.append(f"FAIL\t{src}\t{e}")
                    n_skipped += 1
                    continue
            n_renamed += 1
    return n_renamed, n_skipped, n_already


# ── Phase 2: CSV run_mode rewrite ────────────────────────────────────

def _remap_cell(value):
    """(new_value, changed) — remap a run_mode CSV cell."""
    if value is None:
        return value, False
    s = str(value).strip()
    if s == "" or s.upper() == "NA" or not s.isdigit():
        return value, False
    try:
        n = int(s)
    except ValueError:
        return value, False
    # Already new canonical? 4-digit string AND in image of map.
    if len(s) >= 4 and n in _NEW_VALUES:
        return value, False
    if n not in OLD_TO_NEW:
        return value, False
    return f"{OLD_TO_NEW[n]:04d}", True


def _rewrite_csvs(root, apply, verbose, log_lines):
    n_scan = n_mod = n_rows = 0
    for dirpath in _walk(root):
        try:
            entries = os.listdir(dirpath)
        except OSError:
            continue
        for name in entries:
            if not name.endswith('.csv'):
                continue
            src = os.path.join(dirpath, name)
            n_scan += 1
            try:
                with open(src, 'r', newline='', encoding='utf-8') as f:
                    rows = list(csv.reader(f))
            except Exception as e:
                log_lines.append(f"CSV_READ_FAIL\t{src}\t{e}")
                continue
            if not rows:
                continue
            header = rows[0]
            if 'run_mode' not in header:
                continue
            rmc = header.index('run_mode')
            changed = 0
            for row in rows[1:]:
                if rmc >= len(row):
                    continue
                nv, c = _remap_cell(row[rmc])
                if c:
                    row[rmc] = nv
                    changed += 1
            if not changed:
                continue
            log_lines.append(f"CSV_UPDATE\t{src}\trows={changed}")
            if verbose:
                print(f"  CSV: {os.path.relpath(src, root)}  ({changed} rows)")
            if apply:
                tmp = src + '.tmp'
                try:
                    with open(tmp, 'w', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerows(rows)
                    os.replace(tmp, src)
                except Exception as e:
                    log_lines.append(f"CSV_WRITE_FAIL\t{src}\t{e}")
                    try: os.unlink(tmp)
                    except OSError: pass
                    continue
            n_mod += 1
            n_rows += changed
    return n_scan, n_mod, n_rows


# ── Phase 3: JSON content rewrite ────────────────────────────────────

_RUN_KEY_NAMES = {
    'run_id', 'run_num', 'run_mode', 'source_run', 'ref_run',
    'run_number', 'source_run_id', 'run',
}


def _remap_token(s):
    """(new_s, changed) — remap a string that might be a run-ID ref.
    Accepts bare numbers ('3', '0003') and R/Q-prefixed ('R3', 'Q0033')."""
    if not isinstance(s, str):
        return s, False
    stripped = s.strip()
    prefix = ''
    core = stripped
    if len(core) >= 2 and core[0] in ('R', 'Q'):
        prefix = core[0]
        core = core[1:]
    if not core.isdigit():
        return s, False
    try:
        n = int(core)
    except ValueError:
        return s, False
    # Already new canonical?
    if len(core) >= 4 and n in _NEW_VALUES:
        return s, False
    if n not in OLD_TO_NEW:
        return s, False
    new_n = OLD_TO_NEW[n]
    new_core = f"{new_n:04d}"
    return (_new_pfx(new_n) + new_core) if prefix else new_core, True


_RUN_CONTAINER_KEYS = {
    'per_run', 'runs', 'by_run', 'run_results', 'per_source_run',
    'source_runs', 'run_map', 'results_by_run',
}


def _walk_tree(obj, parent_key, changes):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            new_k = k
            # v0.79.4.0: only remap dict keys if this dict is a KNOWN
            # run-indexed container. Otherwise numeric keys like "0", "1"
            # (which could be trial indices, turn numbers, etc) are left
            # alone. Safer default.
            if isinstance(k, str) and parent_key in _RUN_CONTAINER_KEYS:
                ns, c = _remap_token(k)
                if c:
                    new_k = ns
                    changes[0] += 1
            out[new_k] = _walk_tree(v, parent_key=k, changes=changes)
        return out
    if isinstance(obj, list):
        return [_walk_tree(x, parent_key, changes) for x in obj]
    if isinstance(obj, str) and parent_key in _RUN_KEY_NAMES:
        ns, c = _remap_token(obj)
        if c:
            changes[0] += 1
            return ns
    # Handle plural-parent case: "source_runs": ["3", "15", "48"]
    if isinstance(obj, str) and parent_key and parent_key.endswith('s') and \
       parent_key[:-1] in _RUN_KEY_NAMES:
        ns, c = _remap_token(obj)
        if c:
            changes[0] += 1
            return ns
    if isinstance(obj, int) and parent_key in _RUN_KEY_NAMES:
        # v0.79.4.0: aggressive remap. Idempotency handled by marker file.
        if obj in OLD_TO_NEW:
            changes[0] += 1
            return OLD_TO_NEW[obj]
    return obj


def _rewrite_jsons(root, apply, verbose, log_lines):
    n_scan = n_mod = n_keys = 0
    for dirpath in _walk(root):
        try:
            entries = os.listdir(dirpath)
        except OSError:
            continue
        for name in entries:
            if not name.endswith('.json'):
                continue
            src = os.path.join(dirpath, name)
            n_scan += 1
            try:
                with open(src, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                log_lines.append(f"JSON_READ_FAIL\t{src}\t{e}")
                continue
            changes = [0]
            new_data = _walk_tree(data, parent_key=None, changes=changes)
            if not changes[0]:
                continue
            log_lines.append(f"JSON_UPDATE\t{src}\tkeys={changes[0]}")
            if verbose:
                print(f"  JSON: {os.path.relpath(src, root)}  ({changes[0]} keys)")
            if apply:
                tmp = src + '.tmp'
                try:
                    with open(tmp, 'w', encoding='utf-8') as f:
                        json.dump(new_data, f, indent=2, default=str)
                    os.replace(tmp, src)
                except Exception as e:
                    log_lines.append(f"JSON_WRITE_FAIL\t{src}\t{e}")
                    try: os.unlink(tmp)
                    except OSError: pass
                    continue
            n_mod += 1
            n_keys += changes[0]
    return n_scan, n_mod, n_keys


def main():
    ap = argparse.ArgumentParser(
        description='Migrate data to v0.79.4.0 execution-order run IDs.')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--data-dir', default=DATA)
    ap.add_argument('--skip-jsons', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='Re-run even if migration marker file exists.')
    args = ap.parse_args()

    root = os.path.abspath(args.data_dir)
    if not os.path.isdir(root):
        print(f"ERROR: data dir not found: {root}", file=sys.stderr)
        sys.exit(2)

    # v0.79.4.0: idempotency guard. Marker file written on successful apply.
    marker = os.path.join(root, '.migration_0.79.4.0.COMPLETE')
    if os.path.exists(marker) and args.apply and not args.force:
        print(f"ERROR: migration already completed (marker exists: {marker})")
        print("Pass --force to re-run anyway (DANGEROUS — remaps already-remapped IDs).")
        sys.exit(3)

    mode = 'APPLY' if args.apply else 'DRY-RUN'
    print(f"IOTA v0.79.4.0 Migration  [{mode}]")
    print(f"  Data root: {root}")
    print(f"  Remap:     execution-order renumber (56 runs, bijective)")
    print()

    log = [f"# iota 0.79.4.0 migration log — mode={mode}"]

    print("=== Phase 1: rename files to new canonical IDs")
    a, b, c = _rename_files(root, args.apply, args.verbose, log)
    print(f"    Files {('renamed' if args.apply else 'would rename')}: {a}")
    if c: print(f"    Already canonical (skipped):                      {c}")
    if b: print(f"    Skipped (collisions/failures):                     {b}")

    print()
    print("=== Phase 2: rewrite CSV run_mode cells")
    d, e, f2 = _rewrite_csvs(root, args.apply, args.verbose, log)
    print(f"    CSVs scanned:  {d}")
    print(f"    CSVs {('modified' if args.apply else 'to modify')}:  {e}")
    print(f"    Rows {('updated' if args.apply else 'to update')}:   {f2}")

    if not args.skip_jsons:
        print()
        print("=== Phase 3: rewrite JSON contents")
        g, h, i = _rewrite_jsons(root, args.apply, args.verbose, log)
        print(f"    JSONs scanned: {g}")
        print(f"    JSONs {('modified' if args.apply else 'to modify')}: {h}")
        print(f"    Keys {('remapped' if args.apply else 'to remap')}:  {i}")

    print()
    logp = os.path.join(root, '.migration_0.79.4.0.log')
    if args.apply or args.verbose:
        try:
            with open(logp, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(log) + '\n')
            print(f"Manifest written: {logp}")
        except Exception as ex:
            print(f"WARN: could not write manifest: {ex}", file=sys.stderr)

    if not args.apply:
        print()
        print("This was a DRY RUN. Re-run with --apply to make changes.")
        print("Recommended before applying:")
        print("    tar -czf data_backup_pre_0.79.4.0.tar.gz data/")
    else:
        # Write idempotency marker
        try:
            with open(marker, 'w', encoding='utf-8') as fh:
                from datetime import datetime as _dt
                fh.write(f"# iota 0.79.4.0 migration complete\n"
                         f"timestamp={_dt.now().isoformat()}\n")
        except Exception as ex:
            print(f"WARN: could not write marker: {ex}", file=sys.stderr)


if __name__ == '__main__':
    main()
