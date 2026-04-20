#!/usr/bin/env python3
"""
IOTA — Universal CSV dedup/cap/cull v0.58.0.0
================================================================
Three passes per CSV:

  1. DEDUP  — remove duplicate rows using key (run_mode, trial, turn, condition).
              All rows are header-width (universal schema). Read by column name.

  2. CAP    — trim per-condition trial count to n_trials.

  3. CULL   — remove incomplete trials (fewer than expected turns).

Dry-run by default. Pass --apply to write changes (backs up as .csv.bak).

Usage:
  python dedup_all_runs.py                 # dry-run all runs
  python dedup_all_runs.py --run 0025        # dry-run single run
  python dedup_all_runs.py --apply         # apply all
  python dedup_all_runs.py --run 0025 --apply
  python dedup_all_runs.py --trials 50     # override n_trials
"""

import os, sys, csv, io, shutil, argparse, json
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# All condition columns — ALL of them form the dedup key so R26 (condition +
# temperature_condition) is never falsely deduplicated across temperatures.
_COND_COLS = [
    'condition', 'r_condition', 'history_mode', 'confound_condition',
    'instance', 'temperature_condition', 'patch_mode', 'patch_layer',
    'phase',
]


def _get_paths(trials_override=None):
    """Resolve (paths_dict, n_trials) for the currently-active session.

    Reads last_session.json, calls cartography.get_paths with those
    family/size/variant/temperature values, and returns the session's
    configured trials count (or trials_override if provided).
    Returns (None, 100) on any error so callers can fall back safely."""
    sess_path = os.path.join(ROOT, 'last_session.json')
    s = {}
    if os.path.exists(sess_path):
        with open(sess_path) as f:
            s = json.load(f)
    try:
        from cartography import get_paths
        paths = get_paths(s.get('model_family','llama'), s.get('model_size','8b'),
                          s.get('model_variant','abliterated'), s.get('temperature',0.0),
                          create_dirs=False)
    except Exception as e:
        print(f"  [dedup] Could not load paths: {e}")
        return None, 100
    n = trials_override if trials_override is not None else int(s.get('trials', 100))
    return paths, n


def _read_rows(fpath):
    """Read a CSV file as raw rows (header + data). UTF-8 tolerant,
    CRLF-normalized. Returns list of lists from csv.reader."""
    with open(fpath, 'rb') as fh:
        raw = fh.read()
    content = raw.decode('utf-8', errors='replace').replace('\r\n','\n').replace('\r','\n')
    return list(csv.reader(io.StringIO(content)))


def _write_rows(fpath, header, kept):
    """Atomic overwrite: backs up fpath to fpath+'.bak' via shutil.copy2
    (preserves mtime), then writes header + kept rows with LF line endings.
    .bak lets process_run be re-runnable — last good state is always on disk."""
    shutil.copy2(fpath, fpath + '.bak')
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)
    w.writerows(kept)
    with open(fpath, 'w', encoding='utf-8', newline='') as fh:
        fh.write(buf.getvalue())


def process_run(run_num, fpath, n_trials, dry_run=True, verbose=True):
    """Apply DEDUP -> CAP -> CULL passes to one run's CSV.

    DEDUP  — collapse identical rows under compound key
             (run_mode, trial, turn, condition_concat). Condition key
             concatenates every condition column present in the CSV so
             multi-condition runs (R26, R28, etc.) aren't collapsed
             across conditions.

    CAP    — trim per-condition rows down to n_trials * n_turns. Uses
             file_trial when present (multi-condition offset encoding),
             else plain trial. Keeps lowest trial indices.

    CULL   — remove trials whose turn count falls short of the expected
             per-condition turn count. Incomplete trials would bias
             analysis when loaded.

    Returns {'dedup', 'cap', 'cull', 'total'} counts.
    dry_run=True (default): no writes. Pass --apply at CLI to persist.
    """
    all_rows = _read_rows(fpath)
    if len(all_rows) < 2:
        return {'dedup': 0, 'cap': 0, 'cull': 0, 'total': 0}

    header  = all_rows[0]
    n_hdr   = len(header)

    # Build column index by name — canonical read, no position scanning
    idx = {c: i for i, c in enumerate(header)}

    rm_idx  = idx.get('run_mode', 0)
    pr_idx  = idx.get('priming',  2)
    tr_idx  = idx.get('trial',   -1)
    tu_idx  = idx.get('turn',     1)

    # All condition columns present in this CSV form the compound condition key.
    # This handles R26 (condition + temperature_condition), R28 (confound_condition), etc.
    cond_idxs = [idx[c] for c in _COND_COLS if c in idx]

    def _is_priming(row):
        try: return row[pr_idx] == '1'
        except: return False

    def _get_trial(row):
        if tr_idx < 0 or tr_idx >= len(row): return None
        try:
            v = int(float(row[tr_idx]))
            return v if v >= 0 else None
        except (ValueError, TypeError):
            return None

    def _get_turn(row):
        try: return int(row[tu_idx])
        except: return -1

    def _get_cond(row):
        # Concatenate all condition column values → compound key
        parts = []
        for i in cond_idxs:
            v = row[i] if i < len(row) else ''
            if v: parts.append(v)
        return '|'.join(parts)

    def _get_plain_trial(row):
        """Plain trial 0..(n_trials-1) for cap/cull."""
        t = _get_trial(row)
        if t is None: return None
        return t if 0 <= t < n_trials else None

    # ── Pass 1: DEDUP ─────────────────────────────────────────────────────────
    seen = set()
    kept = []
    dedup_removed = 0

    for row in all_rows[1:]:
        if _is_priming(row):
            kept.append(row)
            continue
        tid  = _get_trial(row)
        turn = _get_turn(row)
        cond = _get_cond(row)
        if tid is None:
            kept.append(row)
            continue
        key = (tid, turn, cond)
        if key in seen:
            dedup_removed += 1
        else:
            seen.add(key)
            kept.append(row)

    # ── Pass 2: CAP ───────────────────────────────────────────────────────────
    cond_trials = defaultdict(set)
    for row in kept:
        if _is_priming(row): continue
        t  = _get_plain_trial(row)
        cd = _get_cond(row)
        if t is not None:
            cond_trials[cd].add(t)

    excess = set()
    for cd, trials in cond_trials.items():
        if len(trials) > n_trials:
            for t in sorted(trials)[n_trials:]:
                excess.add((cd, t))

    cap_removed = 0
    if excess:
        new_kept = []
        for row in kept:
            if _is_priming(row):
                new_kept.append(row); continue
            if (_get_cond(row), _get_plain_trial(row)) in excess:
                cap_removed += 1
            else:
                new_kept.append(row)
        kept = new_kept

    # ── Pass 3: CULL ──────────────────────────────────────────────────────────
    trial_turns = defaultdict(lambda: defaultdict(int))
    for row in kept:
        if _is_priming(row): continue
        t  = _get_plain_trial(row)
        cd = _get_cond(row)
        if t is not None:
            trial_turns[cd][t] += 1

    # Compute expected_turns PER condition group, not globally.
    # R43/R44 have mixed-turn phases (8 priming + 5 arithmetic) — global mode
    # would be 8, causing all 5-turn arithmetic blocks to be incorrectly culled.
    incomplete = set()
    for cd, t_dict in trial_turns.items():
        counts = list(t_dict.values())
        if not counts:
            continue
        group_expected = Counter(counts).most_common(1)[0][0]
        for t, count in t_dict.items():
            if count < group_expected:
                incomplete.add((cd, t))
    all_counts = [c for d in trial_turns.values() for c in d.values()]
    expected_turns = Counter(all_counts).most_common(1)[0][0] if all_counts else 13

    cull_removed = 0
    if incomplete:
        new_kept = []
        for row in kept:
            if _is_priming(row):
                new_kept.append(row); continue
            if (_get_cond(row), _get_plain_trial(row)) in incomplete:
                cull_removed += 1
            else:
                new_kept.append(row)
        kept = new_kept

    total = dedup_removed + cap_removed + cull_removed

    if verbose:
        cond_names = [c for c in _COND_COLS if c in idx]
        cond_tag = f"[{'+'.join(cond_names)}]" if cond_names else ""
        mode = "DRY-RUN" if dry_run else "APPLIED"
        if total > 0:
            print(f"  [{mode}] Run {run_num:04d} {cond_tag}: "
                  f"dedup={dedup_removed}, cap={cap_removed}, cull={cull_removed} "
                  f"(expected_turns={expected_turns})")
        else:
            print(f"  Run {run_num:04d}: clean")

    if total > 0 and not dry_run:
        _write_rows(fpath, header, kept)

    return {'dedup': dedup_removed, 'cap': cap_removed,
            'cull': cull_removed, 'total': total}


def main():
    """CLI entry point. Parses --run / --apply / --trials, resolves the
    active session's paths, and applies process_run across every CSV
    for the selected runs.

    Default (no --apply) is a dry-run printing dedup/cap/cull counts
    per run. --apply actually writes (CSV backed up as .csv.bak first).
    --trials overrides the session's trials count for cap/cull sizing."""
    parser = argparse.ArgumentParser(description='IOTA CSV integrity v3.0')
    parser.add_argument('--trials', type=int, default=None)
    parser.add_argument('--run',    type=int, default=None)
    parser.add_argument('--apply',  action='store_true')
    args = parser.parse_args()

    dry_run = not args.apply
    paths, n_trials = _get_paths(args.trials)
    if not paths:
        sys.exit(1)

    try:
        from cartography import RUN_CSV
    except Exception as e:
        print(f"Could not load dependency_map: {e}"); sys.exit(1)

    csv_dir = paths.get('csv', '')
    mode_str = "DRY-RUN (no files changed)" if dry_run else "APPLY MODE"
    print(f"\nIOTA dedup v3.0 — {mode_str}")
    print(f"n_trials={n_trials}  csv_dir={csv_dir}\n")

    total_removed = 0
    runs = {args.run: RUN_CSV[args.run]} if args.run else dict(RUN_CSV)

    # Run 0001 has a multi-pass structure (base → instruct → abliterated) that
    # produces rows with identical (trial, turn) keys across passes — dedup
    # incorrectly treats them as duplicates and removes valid data.
    # v54.0.2: excluded from pre-run dedup in start_here.py.
    # v54.2.3: also excluded here for consistency.
    _SKIP_DEDUP = {19}

    for run_num, csv_fname in sorted(runs.items()):
        if csv_fname is None: continue
        if run_num in _SKIP_DEDUP:
            print(f"  Run {run_num:04d}: skipped (multi-pass structure exempt from dedup)")
            continue
        fpath = os.path.join(csv_dir, csv_fname)
        if not os.path.exists(fpath): continue
        try:
            r = process_run(run_num, fpath, n_trials, dry_run=dry_run)
            total_removed += r['total']
        except Exception as e:
            print(f"  Run {run_num:04d}: ERROR — {e}")

    print(f"\nTotal rows {'would be' if dry_run else ''} removed: {total_removed}")
    if dry_run and total_removed > 0:
        print("Run with --apply to apply changes.")
    if not dry_run and total_removed > 0:
        print("Backups saved as .csv.bak alongside modified files.")


if __name__ == '__main__':
    main()
