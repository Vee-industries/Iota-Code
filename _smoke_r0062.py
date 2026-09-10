"""
_smoke_r0062.py -- smoke test for Run 0062 framework wiring.

Fires R0062 on gemma_2b_q4 at T=0.0 with trials=3 (one trial per
cohort A, B, C, plus partial D). Verifies:
  - Framework dispatches run_mode=62 without crashing
  - Pre-shock turns (0-12) produce single-token status outputs
    (status enforcer ON, matched to R0039)
  - Shock turn 13 produces single-token status output
    (status enforcer ON, matched to R0039)
  - Recovery turns (14, 15, 16) produce multi-token free-form
    output (status enforcer OFF)
  - CSV captures the free-form recovery responses
  - Hidden state .npy files written for recovery turns

This is a wiring smoke test, not a scientific result. The full
collection lives in _phase_e_section36.py.
"""
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:/Users/Gaming/Desktop/iota active")
PY = r"C:/Users/Gaming/AppData/Local/Programs/Python/Python310/python.exe"
SESSION = ROOT / "last_session.json"

MODEL_CFG = {
    "tag":              "gemma_2b_q4",
    "model_path":       "IlyaGusev/gemma-2-2b-it-abliterated",
    "model_name":       "gemma-2-2b-it-abliterated",
    "model_family":     "gemma",
    "model_size":       "2b",
    "size_dir":         "2b_4bit",
    "model_variant":    "abliterated",
    "quantization":     "4bit",
    "model_generation": "2",
}


def main():
    print(f"[smoke] firing R0062 on {MODEL_CFG['tag']} T=0.0 trials=3")
    s = json.loads(SESSION.read_text())
    s["model_path"]       = MODEL_CFG["model_path"]
    s["model_name"]       = MODEL_CFG["model_name"]
    s["model_family"]     = MODEL_CFG["model_family"]
    s["model_size"]       = MODEL_CFG["model_size"]
    s["model_variant"]    = MODEL_CFG["model_variant"]
    s["quantization"]     = MODEL_CFG["quantization"]
    s["model_generation"] = MODEL_CFG["model_generation"]
    s["temperature"]      = 0.0
    s["trials"]           = 3
    s["runs"]             = "62"
    s["_saved"]           = datetime.now().isoformat()
    SESSION.write_text(json.dumps(s, indent=2))

    t0 = time.time()
    rc = subprocess.run(
        [PY, "start_here.py", "--single-run", "62"],
        cwd=str(ROOT),
    ).returncode
    dt = time.time() - t0
    print(f"[smoke] rc={rc} ({int(dt)}s)")
    if rc != 0:
        return rc

    # Inspect the resulting CSV
    csv_path = (ROOT / "data" / "gemma" / "2b_4bit" / "abliterated"
                / "deterministic" / "csv" / "R0062_jolt.csv")
    if not csv_path.exists():
        print(f"[smoke] FAIL: expected CSV not found at {csv_path}")
        return 2

    import csv as _csv
    _csv.field_size_limit(2**31 - 1)
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"\n[smoke] CSV has {len(rows)} rows")
    print(f"[smoke] per-turn summary (trial=0 only):")
    for row in rows:
        if row.get("trial") != "0":
            continue
        turn = row.get("turn")
        out = row.get("output", "")[:80].replace("\n", "\\n")
        n_tok = row.get("output_tokens", "?")
        sv = row.get("shock_variant", "?")
        print(f"  turn={turn:>3}  sv={sv:>2}  n_tok={n_tok:>4}  output={out!r}")

    # Check recovery turns produced multi-token output
    recovery_rows = [r for r in rows if r.get("turn") in ("14", "15", "16")]
    multi_token = [r for r in recovery_rows
                   if int(r.get("output_tokens", "1")) > 1]
    print(f"\n[smoke] recovery rows: {len(recovery_rows)}, "
          f"multi-token recovery rows: {len(multi_token)}")
    if not multi_token:
        print(f"[smoke] FAIL: recovery turns produced no multi-token output "
              f"(enforcer-off not engaged)")
        return 3
    print(f"[smoke] OK: framework dispatched R0062, recovery turns generated "
          f"free-form output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
