"""
IOTA v0.76.0.0 Data Migration

1. Renames base/ → data/ (root data directory)
2. Renames size folders to include quantization suffix
3. Handles all possible starting states

Usage:
  python migrate_folders.py           # dry run
  python migrate_folders.py --apply   # rename
"""

import os, sys, json

ROOT = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.join(ROOT, "last_session.json")
APPLY = "--apply" in sys.argv

print("IOTA v0.76.0.0 Data Migration\n")

# ── Phase 1: base/ → data/ ──────────────────────────────────────────────────

old_root = os.path.join(ROOT, "base")
new_root = os.path.join(ROOT, "data")

if os.path.isdir(old_root) and not os.path.isdir(new_root):
    n = sum(len(files) for _, _, files in os.walk(old_root))
    print(f"  base/ → data/  ({n} files)")
    if APPLY:
        os.rename(old_root, new_root)
        print(f"    RENAMED")
elif os.path.isdir(new_root):
    print(f"  data/ already exists — skipping root rename")
elif not os.path.isdir(old_root):
    print(f"  base/ not found, data/ not found — nothing to rename")

# ── Phase 2: size folder renames ─────────────────────────────────────────────

DATA = new_root  # work from data/ regardless of phase 1

SIZE_RENAMES = [
    # (family, old_name, new_name)
    # Original names (pre-migration)
    ("gemma", "3b",     "2b_4bit"),
    ("gemma", "8b",     "9b_4bit"),
    ("llama", "8b",     "8b_4bit"),
    # Wrong _q4 suffix from v0.75.3.0
    ("gemma", "2b_q4",  "2b_4bit"),
    ("gemma", "9b_q4",  "9b_4bit"),
    ("llama", "8b_q4",  "8b_4bit"),
]

if os.path.isdir(DATA):
    for family, old_size, new_size in SIZE_RENAMES:
        old_path = os.path.join(DATA, family, old_size)
        new_path = os.path.join(DATA, family, new_size)
        if not os.path.isdir(old_path):
            continue
        if os.path.isdir(new_path):
            print(f"  {family}/{old_size}/ → {new_size}/ — TARGET EXISTS, skipping")
            continue
        n = sum(len(files) for _, _, files in os.walk(old_path))
        print(f"  {family}/{old_size}/ → {family}/{new_size}/  ({n} files)")
        if APPLY:
            os.rename(old_path, new_path)
            print(f"    RENAMED")

# ── Phase 3: session update ──────────────────────────────────────────────────

SIZE_FIXES = {
    ("gemma", "3b"): "2b",
    ("gemma", "8b"): "9b",
    ("gemma", "2b_q4"): "2b",
    ("gemma", "9b_q4"): "9b",
    ("llama", "8b_q4"): "8b",
}

if os.path.exists(SESSION):
    with open(SESSION) as f:
        sess = json.load(f)
    fam = sess.get('model_family', '')
    sz = sess.get('model_size', '')
    if (fam, sz) in SIZE_FIXES:
        new_sz = SIZE_FIXES[(fam, sz)]
        print(f"\n  Session: model_size '{sz}' → '{new_sz}'")
        if APPLY:
            sess['model_size'] = new_sz
            with open(SESSION, 'w') as f:
                json.dump(sess, f, indent=2)
            print(f"    UPDATED")
    else:
        print(f"\n  Session: model_size '{sz}' — no change needed")

if not APPLY:
    print("\n  DRY RUN. Run with --apply to rename.")
else:
    print("\n  Migration complete. Restart Flask.")
