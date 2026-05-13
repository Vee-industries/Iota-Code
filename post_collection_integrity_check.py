"""
_post_collection_integrity_check.py -- assert cross-temperature distinctness
of hidden-state .npy files after a re-collection run.

Run after every collection batch. If two cells in the same model/variant at
different non-zero temperatures produce byte-identical .npy files for the
same (trial, turn), something is wrong and the data is unfit for any
analysis that treats temperature as a varying axis.

Catches:
  - status_enforcer accidentally still on (collapses outputs to {DONE,
    WAIT, STOP} which makes temperature invisible)
  - seed scheme that doesn't differentiate temperatures
  - directory-routing bug that writes the same file to multiple temp dirs

Usage:
  python _post_collection_integrity_check.py --run 39
  python _post_collection_integrity_check.py --run 3
  python _post_collection_integrity_check.py --run 39 --model llama_8b_q4

Exits with code 0 if integrity holds, code 1 if duplicates found.

v0.82.0.30: written 2026-05-13 as part of the post-status-enforcer audit.
The bug-that-already-happened becomes a guarantee-that-can't-happen-again.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')

# Models to audit. Each entry is (family, size_dir, variant).
MODELS = [
    ('llama', '8b_4bit',  'abliterated'),
    ('gemma', '9b_4bit',  'abliterated'),
    ('gemma', '2b_4bit',  'abliterated'),
    ('gemma', '2b_fp16',  'abliterated'),
]

# Temperature directories to check for cross-temp duplicates.
NONZERO_TEMP_DIRS = ['temp_0.2', 'temp_0.4', 'temp_0.6', 'temp_0.8', 'temp_1.0']

# Sample size per cell: which (trial, turn) tuples to spot-check.
# Full sweep is expensive; this samples enough to catch any global duplication.
SPOT_CHECK = [
    (0, 5),  (0, 10), (0, 14),
    (25, 5), (25, 10), (25, 14),
    (50, 5), (50, 10), (50, 14),
    (75, 5), (75, 10), (75, 14),
    (99, 5), (99, 10), (99, 14),
]


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def find_npy(family, size, variant, temp_dir, run_id, trial, turn):
    """Find the hidden-state .npy file for a given (run, trial, turn) under a
    temperature directory. Returns the path or None."""
    prefix = 'R' if int(run_id) <= 21 else 'Q'
    base = os.path.join(DATA, family, size, variant, temp_dir, 'hidden_states')
    if not os.path.isdir(base):
        return None
    # Match any model-name variant of the file (sanitized name varies).
    pattern_tail = f"_trial{int(trial):04d}_turn{int(turn):02d}.npy"
    pattern_head = f"{prefix}{int(run_id):04d}_"
    try:
        for fname in os.listdir(base):
            if fname.startswith(pattern_head) and fname.endswith(pattern_tail):
                return os.path.join(base, fname)
    except OSError:
        return None
    return None


def audit_run(run_id, models=None):
    """For each model, check that the same (trial, turn) across the 5 nonzero
    temperature directories produces DIFFERENT md5 hashes. Reports any cluster
    of byte-identical files (which would indicate a re-introduced collection
    bug).

    Returns a dict with the audit results."""
    models = models or MODELS
    results = {
        'run_id':         run_id,
        'check_version':  '0.82.0.30',
        'spot_checks':    [],
        'duplicates':     [],
        'errors':         [],
        'verdict':        None,
    }
    n_checked = 0
    n_dup_clusters = 0
    for family, size, variant in models:
        model_tag = f"{family}/{size}/{variant}"
        for trial, turn in SPOT_CHECK:
            paths_md5 = {}
            for tdir in NONZERO_TEMP_DIRS:
                p = find_npy(family, size, variant, tdir, run_id, trial, turn)
                if p is None:
                    continue
                try:
                    paths_md5[tdir] = md5_file(p)
                except OSError as e:
                    results['errors'].append(
                        f"{model_tag} {tdir} t{trial} k{turn}: {e}")
                    continue
            if not paths_md5:
                continue
            n_checked += 1
            md5_to_temps = defaultdict(list)
            for tdir, m in paths_md5.items():
                md5_to_temps[m].append(tdir)
            distinct = len(md5_to_temps)
            results['spot_checks'].append({
                'model':              model_tag,
                'trial':              trial,
                'turn':               turn,
                'n_temps_present':    len(paths_md5),
                'n_distinct_md5':     distinct,
                'md5_groupings':      {m: sorted(t) for m, t in md5_to_temps.items()},
            })
            if distinct < len(paths_md5):
                n_dup_clusters += 1
                for m, temps in md5_to_temps.items():
                    if len(temps) > 1:
                        results['duplicates'].append({
                            'model':       model_tag,
                            'trial':       trial,
                            'turn':        turn,
                            'md5':         m,
                            'temperatures': sorted(temps),
                        })
    if n_checked == 0:
        results['verdict'] = 'NO_DATA'
    elif n_dup_clusters == 0:
        results['verdict'] = 'PASS'
    else:
        dup_rate = n_dup_clusters / n_checked
        if dup_rate >= 0.5:
            results['verdict'] = 'FAIL_MAJOR'
        else:
            results['verdict'] = 'FAIL_PARTIAL'
    results['n_spot_checked'] = n_checked
    results['n_duplicate_clusters'] = n_dup_clusters
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=int, required=True,
                    help='Run ID to audit (e.g. 39 for jolt, 3 for temperature grid)')
    ap.add_argument('--model', type=str, default=None,
                    help='Optional: restrict to one model (e.g. llama_8b_q4)')
    args = ap.parse_args()

    models = MODELS
    if args.model:
        models = []
        for family, size, variant in MODELS:
            tag = f"{family}_{size.replace('_', '').replace('bit','')}"  # rough match
            if args.model.lower() in (f"{family}_{size}", tag, family):
                models.append((family, size, variant))
        if not models:
            print(f"No model matches '{args.model}'. Known: "
                  + ', '.join(f"{f}/{s}/{v}" for f, s, v in MODELS))
            return 2

    print(f"Running cross-temperature integrity check for Run {args.run:04d}...")
    print(f"Models: {', '.join(f'{f}/{s}/{v}' for f, s, v in models)}")
    print(f"Spot-check tuples (trial, turn): {len(SPOT_CHECK)} per model")
    print()

    results = audit_run(args.run, models)

    print(f"Verdict: {results['verdict']}")
    print(f"  Spot-checks performed:   {results['n_spot_checked']}")
    print(f"  Duplicate clusters:      {results['n_duplicate_clusters']}")
    if results['duplicates']:
        print()
        print("Duplicate cross-temperature md5 clusters:")
        for d in results['duplicates'][:20]:
            print(f"  {d['model']} t{d['trial']} k{d['turn']}: "
                  f"{d['md5'][:12]} across {d['temperatures']}")
        if len(results['duplicates']) > 20:
            print(f"  ... and {len(results['duplicates']) - 20} more")
    if results['errors']:
        print()
        print(f"Errors during scan: {len(results['errors'])}")
        for e in results['errors'][:5]:
            print(f"  {e}")

    out_dir = os.path.join(DATA, 'paper', 'data_integrity')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            f'integrity_check_run{args.run:04d}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {out_path}")

    return 0 if results['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
