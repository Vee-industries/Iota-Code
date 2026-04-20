#!/usr/bin/env python3
"""
IOTA — Retroactive hidden state copy for Runs 0002 and 0003
==========================================================
Copies R20_* and Q26_* hidden state files from deterministic to temp_0.2.

Prior to v0.55.0.3, _TEMP_INDEPENDENT_RUNS had copy_hidden=False for Runs 0002
and 0003. Their CSVs were copied but hidden states were not. This means:
  - _load_quadruplets at T=0.2 found no S_t files for Runs 0002/0003
  - Run 0002 (SOURCE_RUNS_2WAY) and Run 0003 (SOURCE_RUNS_3WAY) contributed
    zero rows to the OLS decomposition at non-deterministic temperatures
  - The BLOCKING run (26) was silently absent from the decomposition

This script retroactively copies the missing files.

Run 0002: abliterated + base variants (S_t in abliterated, E_t in base)
Run 0003: abliterated only (S_t + E_t proxy + C_t all in abliterated)

Usage:
  python copy_hidden_r20_r26.py              # dry run
  python copy_hidden_r20_r26.py --apply      # copy the files
"""

import os, sys, glob, json, shutil, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
for _ in range(4):
    if os.path.exists(os.path.join(ROOT, 'start_here.py')):
        break
    ROOT = os.path.dirname(ROOT)

sys.path.insert(0, ROOT)

COPY_SPEC = {
    # v0.79.4.0: renumbered. Old 20 → new 2 (robustness). Old 26 → new 3 (temperature grid).
    # Filenames flipped to new R-prefix canonical form.
    2: {'prefix': 'R0002_*', 'variants': ['abliterated', 'base'], 'legacy_prefix': 'R20_*'},
    3: {'prefix': 'R0003_*', 'variants': ['abliterated'],          'legacy_prefix': 'Q26_*'},
}

SRC_COND = 'deterministic'
DST_COND = 'temp_0.2'


def main():
    """Retroactively copy R20 and Q26 hidden-state .npy files from the
    deterministic/ directory to temp_0.2/ (or --dst target).

    Rationale — pre-v0.55.0.3:
    _TEMP_INDEPENDENT_RUNS marked Runs 0002 and 0003 as temperature-
    indifferent (CSVs were mirrored to every temp dir) but copy_hidden
    was False for both, so the .npy hidden states never made the trip.
    At T=0.2+, _load_quadruplets found zero S_t files for Runs 0002/0003
    and both contributed zero rows to the decomposition — silently.
    Run 0003 is the Phase 2 BLOCKING run; its absence at all non-
    deterministic temperatures is catastrophic for H13.

    Copy spec:
      Run 0002:  R20_* in abliterated + base (S_t in abliterated,
               base-model E_t in base/)
      Run 0003:  Q26_* in abliterated only (S_t + E_t proxy + C_t all
               written by the abliterated-model pass)

    Dry-run by default. --apply performs shutil.copy2. Safe to re-run;
    already-copied files are skipped. Reads last_session.json for
    family/size/quant so it operates on the currently-active model.
    """
    parser = argparse.ArgumentParser(description='Copy R20/Q26 hidden states to temp_0.2')
    parser.add_argument('--apply', action='store_true', help='Actually copy (default: dry run)')
    parser.add_argument('--dst', default=DST_COND, help=f'Destination condition (default: {DST_COND})')
    args = parser.parse_args()

    sess_path = os.path.join(ROOT, 'last_session.json')
    family, size, _quant = 'llama', '8b', '4bit'
    if os.path.exists(sess_path):
        try:
            with open(sess_path) as f:
                s = json.load(f)
            family = s.get('model_family', 'llama')
            size = s.get('model_size', '8b')
            _quant = s.get('quantization', '4bit')
        except Exception:
            pass

    from cartography import DATA, get_family_size_dir, set_active_quant
    set_active_quant(_quant)

    mode = 'APPLY' if args.apply else 'DRY RUN'
    dst_cond = args.dst
    print(f'\nIOTA — Copy R20/Q26 hidden states [{mode}]')
    print(f'Source: {SRC_COND}  →  Destination: {dst_cond}')
    print(f'Base: {get_family_size_dir(family, size)}/\n')

    total = 0
    total_bytes = 0

    for run_num, spec in COPY_SPEC.items():
        prefix = spec['prefix']
        for var in spec['variants']:
            src_dir = os.path.join(get_family_size_dir(family, size), var, SRC_COND, 'hidden_states')
            dst_dir = os.path.join(get_family_size_dir(family, size), var, dst_cond, 'hidden_states')

            if not os.path.isdir(src_dir):
                print(f'  R{run_num:04d}/{var}/{SRC_COND}: source dir not found — skipping')
                continue

            files = glob.glob(os.path.join(src_dir, prefix))
            if not files and 'legacy_prefix' in spec:
                # v0.79.4.0: fall back to legacy pre-migration filename pattern
                # if new-form not found yet (user hasn't run migrate_run_ids.py).
                files = glob.glob(os.path.join(src_dir, spec['legacy_prefix']))
            if not files:
                print(f'  R{run_num:04d}/{var}/{SRC_COND}: no files matching {prefix}')
                continue

            # Check how many already exist in dst
            already = 0
            to_copy = []
            for f in files:
                dst_file = os.path.join(dst_dir, os.path.basename(f))
                if os.path.exists(dst_file):
                    already += 1
                else:
                    to_copy.append(f)

            if not to_copy:
                print(f'  R{run_num:04d}/{var}: all {len(files)} files already present in {dst_cond}')
                continue

            copy_bytes = sum(os.path.getsize(f) for f in to_copy)

            if args.apply:
                os.makedirs(dst_dir, exist_ok=True)
                for f in to_copy:
                    shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))

            print(f'  R{run_num:04d}/{var}: {len(to_copy)} files to copy '
                  f'({copy_bytes / 1024 / 1024:.1f} MB), {already} already present')
            total += len(to_copy)
            total_bytes += copy_bytes

    print(f'\nTotal: {total} files ({total_bytes / 1024 / 1024:.1f} MB)')
    if not args.apply and total > 0:
        print('Dry run — no files copied. Run with --apply to copy.')
    elif args.apply and total > 0:
        print(f'Copied to {dst_cond}. _load_quadruplets will now find S_t/E_t/C_t for these runs.')
    elif total == 0:
        print('Nothing to copy — all files already present or source missing.')


if __name__ == '__main__':
    main()
