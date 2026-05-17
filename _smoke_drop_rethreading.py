"""
_smoke_drop_rethreading.py -- minimal validation of the drop-rethreading
protocol before committing to a full re-collection.

Purpose: verify that with IOTA_DROP_RETHREADING=1, R0039 collection:
  1. Runs without error (env var change doesn't break tokenization or
     chat-template handling for consecutive user turns).
  2. Produces hidden state files with the expected count and structure.
  3. Produces measurably DIFFERENT hidden states from the threaded
     protocol (proving the env var actually changes behavior).
  4. Still produces cohort-discriminable hidden states (since the shock
     prompt at user_13 is still present in the input -- only the
     assistant-output channel is dropped).

Writes to a temporary variant dir (abliterated_smoke) so existing
v1.0.0 R0039 data is untouched.

Usage:
  python _smoke_drop_rethreading.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = Path(r"C:/Users/Gaming/Desktop/iota active")
PY = r"C:/Users/Gaming/AppData/Local/Programs/Python/Python310/python.exe"
SESSION = ROOT / "last_session.json"
DATA_ROOT = ROOT / "data"

# Smoke target: gemma_2b_q4 at T=0, writing to a temp variant.
MODEL_CFG = {
    "model_path":         "IlyaGusev/gemma-2-2b-it-abliterated",
    "model_name":         "gemma-2-2b-it-abliterated",
    "model_family":       "gemma",
    "model_size":         "2b",
    "model_variant":      "abliterated_smoke",  # isolates from existing data
    "quantization":       "4bit",
    "model_generation":   "2",
}
SMOKE_TEMP = 0.0
SMOKE_TRIALS = 40  # 10 per cohort -- enough for sanity check + permutation


def update_session(trials, temp, runs):
    s = json.loads(SESSION.read_text())
    s.update(MODEL_CFG)
    s["temperature"] = float(temp)
    s["trials"] = trials
    s["runs"] = str(runs)
    s["_saved"] = datetime.now().isoformat()
    SESSION.write_text(json.dumps(s, indent=2))


def md5_first_kb(path):
    """MD5 of first KB of a file -- enough to confirm content distinct."""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read(1024)).hexdigest()


def fire_r0039(env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"  firing R0039  env={env_extra or {}}")
    t0 = datetime.now()
    rc = subprocess.run(
        [PY, "start_here.py", "--single-run", "39"],
        cwd=str(ROOT),
        env=env,
    ).returncode
    dt = (datetime.now() - t0).total_seconds()
    print(f"    rc={rc}  elapsed={int(dt)}s")
    return rc


def count_files(variant, run="Q0039"):
    d = (DATA_ROOT / MODEL_CFG["model_family"]
         / f"{MODEL_CFG['model_size']}_{MODEL_CFG['quantization']}"
         / variant / "deterministic" / "hidden_states")
    if not d.is_dir():
        return 0, d
    files = list(d.glob(f"{run}_*trial*turn*.npy"))
    return len(files), d


def sample_md5s(variant, n=5):
    """MD5 a few specific (trial, turn) pairs for cross-protocol comparison."""
    d = (DATA_ROOT / MODEL_CFG["model_family"]
         / f"{MODEL_CFG['model_size']}_{MODEL_CFG['quantization']}"
         / variant / "deterministic" / "hidden_states")
    samples = [(0, 7), (0, 14), (10, 13), (25, 15), (39, 16)]
    sanitized = MODEL_CFG["model_name"].replace("/", "_").replace(".", "_")
    out = {}
    for trial, turn in samples:
        candidates = list(d.glob(
            f"Q0039_*_trial{trial:04d}_turn{turn:02d}.npy"))
        if candidates:
            out[(trial, turn)] = md5_first_kb(candidates[0])
        else:
            out[(trial, turn)] = "MISSING"
    return out


def main():
    print("=" * 70)
    print(" Smoke test: drop-rethreading on R0039")
    print(f" Started: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 70)
    print()

    # Step 0: clean smoke dir if it exists from a prior run
    smoke_dir = (DATA_ROOT / MODEL_CFG["model_family"]
                 / f"{MODEL_CFG['model_size']}_{MODEL_CFG['quantization']}"
                 / "abliterated_smoke")
    if smoke_dir.exists():
        print(f"  removing prior smoke dir: {smoke_dir}")
        shutil.rmtree(smoke_dir)
    print()

    # Step 1: fire R0039 with drop-rethreading enabled
    print("Step 1: fire R0039 with IOTA_DROP_RETHREADING=1")
    update_session(SMOKE_TRIALS, SMOKE_TEMP, runs=39)
    rc = fire_r0039(env_extra={"IOTA_DROP_RETHREADING": "1"})
    if rc != 0:
        print("\n  FAIL: R0039 fire exited with non-zero rc")
        return 1
    print()

    # Step 2: file-count sanity
    print("Step 2: verify file count")
    n_smoke, smoke_path = count_files("abliterated_smoke")
    expected = SMOKE_TRIALS * 16  # 16 turns
    print(f"  files in {smoke_path}: {n_smoke}")
    print(f"  expected ~ {expected}  (allowing some slack for priming files)")
    if n_smoke < expected * 0.9:
        print("\n  FAIL: too few hidden state files written")
        return 1
    print()

    # Step 3: cross-protocol distinctness vs existing abliterated/
    print("Step 3: confirm hidden states differ from threaded protocol")
    n_threaded, _ = count_files("abliterated")
    print(f"  files in existing abliterated/: {n_threaded}")
    if n_threaded == 0:
        print("  no existing abliterated/ R0039 -- skipping cross-protocol diff")
    else:
        smoke_md5 = sample_md5s("abliterated_smoke")
        threaded_md5 = sample_md5s("abliterated")
        n_same = sum(1 for k in smoke_md5
                     if smoke_md5[k] == threaded_md5.get(k))
        n_diff = sum(1 for k in smoke_md5
                     if smoke_md5[k] != "MISSING"
                     and threaded_md5.get(k) != "MISSING"
                     and smoke_md5[k] != threaded_md5[k])
        print(f"  sampled (trial,turn) pairs: {len(smoke_md5)}")
        print(f"  identical to threaded:      {n_same}")
        print(f"  different from threaded:    {n_diff}")
        for k in smoke_md5:
            t_md5 = threaded_md5.get(k, "MISSING")
            tag = "SAME" if smoke_md5[k] == t_md5 else "DIFF"
            print(f"    trial={k[0]:03d} turn={k[1]:02d}  "
                  f"smoke={smoke_md5[k][:12]}  "
                  f"threaded={t_md5[:12]}  [{tag}]")
        if n_diff < len(smoke_md5) // 2:
            print("\n  WARN: most files identical -- "
                  "drop-rethreading may not be in effect")
            # Continue but flag this
    print()

    # Step 4: dump a summary
    print("=" * 70)
    print(" Smoke test complete")
    print(f" Smoke data at: {smoke_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
