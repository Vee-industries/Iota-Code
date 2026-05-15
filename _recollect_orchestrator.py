"""
_recollect_orchestrator.py -- unattended re-collection sequencer.

Fires:
  R0001 (trivariant null isolation)  -- 1 invocation, ~2-3h
  R0003 (temperature grid)           -- 1 invocation (iterates 5 temps internally), ~3-10h
  R0039 × 6 temperatures (jolt)      -- 6 invocations, ~30-60min each

Integrity check after each.

Status log: data/paper/recollection_status_<model>.json
Orchestrator log: .iota_recollect_orchestrator.log
"""
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = Path(r"C:/Users/Gaming/Desktop/iota active")
PY = r"C:/Users/Gaming/AppData/Local/Programs/Python/Python310/python.exe"
SESSION = ROOT / "last_session.json"
LOGFILE = ROOT / ".iota_recollect_orchestrator.log"

# Model under collection (Gemma 2B Q4 abliterated)
FAMILY = "gemma"
SIZE_DIR = "2b_4bit"
VARIANT = "abliterated"
MODEL_TAG = f"{FAMILY}_2b_q4_{VARIANT}"

DATA_ROOT = ROOT / "data"
ABL_DET_HID = (DATA_ROOT / FAMILY / SIZE_DIR / VARIANT
               / "deterministic" / "hidden_states")
R0001_SENTINEL_GLOB = "R0001_*_pass4_ok.stamp"

TEMPS_R0039 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_session(run_num, temperature):
    s = json.loads(SESSION.read_text())
    s["runs"] = str(run_num)
    s["temperature"] = float(temperature)
    SESSION.write_text(json.dumps(s, indent=2))


def fire_run(run_num, temperature, label):
    set_session(run_num, temperature)
    log(f"=== firing Run {run_num:04d} {label} (T={temperature}) ===")
    rc = subprocess.run(
        [PY, "start_here.py", "--single-run", str(run_num)],
        cwd=str(ROOT),
    ).returncode
    log(f"  Run {run_num:04d} exit code: {rc}")
    return rc == 0


def wait_for_r0001_sentinel(timeout_s=14400):
    log(f"Waiting for R0001 Pass 4 sentinel (up to {timeout_s/3600:.1f}h)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if ABL_DET_HID.exists():
            if list(ABL_DET_HID.glob(R0001_SENTINEL_GLOB)):
                log("  R0001 sentinel found.")
                return True
        time.sleep(60)
    log("  R0001 sentinel timeout.")
    return False


def integrity_check(run_num):
    log(f"=== integrity check on Run {run_num:04d} ===")
    rc = subprocess.run(
        [PY, "post_collection_integrity_check.py", "--run", str(run_num)],
        cwd=str(ROOT),
        capture_output=True, text=True,
    )
    log((rc.stdout or "")[-1500:])
    log(f"  integrity check exit code: {rc.returncode}")
    return rc.returncode


def write_status(status):
    out = DATA_ROOT / "paper" / f"recollection_status_{MODEL_TAG}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2))


def main():
    log("=== orchestrator start ===")
    log(f"  model: {FAMILY}/{SIZE_DIR}/{VARIANT}")
    log(f"  sequence: R0001 -> R0003 -> R0039 x {TEMPS_R0039}")

    status = {
        "start_iso": datetime.now().isoformat(),
        "model": f"{FAMILY}/{SIZE_DIR}/{VARIANT}",
        "steps": [],
    }
    write_status(status)

    # Step 1: R0001 (fire + wait for sentinel)
    ok = fire_run(1, 0.0, "(null trivariant)")
    status["steps"].append({"step": "r0001", "result": "OK" if ok else "FAIL"})
    write_status(status)

    # R0001 dispatches to its own no-wait subprocess via _os._exit, so wait
    # for the sentinel file explicitly before proceeding.
    sentinel_ok = wait_for_r0001_sentinel()
    status["steps"].append({"step": "r0001_wait",
                            "result": "OK" if sentinel_ok else "TIMEOUT"})
    write_status(status)
    if not sentinel_ok:
        log("ABORT: R0001 did not complete")
        return 1

    rc = integrity_check(1)
    status["steps"].append({"step": "r0001_integrity",
                            "result": "OK" if rc == 0 else "FAIL_OR_NO_DATA"})
    write_status(status)

    # Step 2: R0003 (temperature grid; iterates internally)
    ok = fire_run(3, 0.0, "(temperature grid)")
    status["steps"].append({"step": "r0003", "result": "OK" if ok else "FAIL"})
    write_status(status)

    if ok:
        rc = integrity_check(3)
        status["steps"].append({"step": "r0003_integrity",
                                "result": "OK" if rc == 0 else "FAIL_OR_NO_DATA"})
        write_status(status)

    # Step 3: R0039 across 6 temperatures
    for t in TEMPS_R0039:
        ok = fire_run(39, t, f"(jolt T={t})")
        status["steps"].append({"step": f"r0039_t{t}",
                                "result": "OK" if ok else "FAIL"})
        write_status(status)

    # Final integrity check on R0039 (cross-temperature audit)
    rc = integrity_check(39)
    status["steps"].append({"step": "r0039_integrity",
                            "result": "OK" if rc == 0 else "FAIL_OR_NO_DATA"})
    status["end_iso"] = datetime.now().isoformat()
    write_status(status)

    log("=== orchestrator complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
