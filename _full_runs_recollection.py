"""
_full_runs_recollection.py -- Phase B: re-collect the 14 ET-source runs.

After the v1.0.0 re-collection of R0001, R0003, R0039 (covered by
_full_recollection.py), Paper A's §4 chain-rule decomposition (Q0042)
still reads hidden states from 14 additional source runs that were
collected under the v28 seed scheme:

  R0002       Robustness sweep
  R0004,5     Null baseline A/B
  R0006-8     Introspection A/B/C
  R0009-12    Arithmetic A/B/C/D
  R0013-15    Priming (neutral / cooperative / resistant)
  R0023       C_t confound isolation

Each is fired once per session-temperature in v28 (writes to the
session-temp dir), so v1.0.0 fidelity requires 14 runs x 6 temps =
84 firings per model x 4 models = 336 firings total. Internal-temp
iterators (R0002, R0006-8, R0023) replicate their 5-internal-temp
output across the 6 session-temp dirs; external-temp consumers
(R0004,5, R0009-12, R0013-15) produce distinct per-temp data.

Resume-safe: per-(model, run, session_temp) status JSON tracks what's
done. Re-launch picks up where the previous run left off without
re-clearing or re-firing completed cells. Per-cell clears happen once
on first encounter; the `cleared` sentinel in the status file prevents
re-clearing.

After all 336 firings: refresh Q0042 (per-cell decomposition) on the
24 cells, then fire Run 0058 (full Bayesian apparatus, all 12 phases)
to produce the refreshed apparatus_aggregator/per_cell.json that
feeds Paper A §4.

Status log: data/paper/recollection_status_phase_b.json
Orchestrator log: .iota_full_runs_recollection.log

Expected total wall time: 2-3 days continuous GPU.
"""
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

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
LOGFILE = ROOT / ".iota_full_runs_recollection.log"
DATA_ROOT = ROOT / "data"
STATUS_PATH = DATA_ROOT / "paper" / "recollection_status_phase_b.json"

MODELS = [
    {
        "tag":           "gemma_2b_q4",
        "model_path":    "IlyaGusev/gemma-2-2b-it-abliterated",
        "model_name":    "gemma-2-2b-it-abliterated",
        "model_family":  "gemma",
        "model_size":    "2b",
        "size_dir":      "2b_4bit",
        "model_variant": "abliterated",
        "quantization":  "4bit",
        "model_generation": "2",
    },
    {
        "tag":           "gemma_2b_fp16",
        "model_path":    "IlyaGusev/gemma-2-2b-it-abliterated",
        "model_name":    "gemma-2-2b-it-abliterated",
        "model_family":  "gemma",
        "model_size":    "2b",
        "size_dir":      "2b_fp16",
        "model_variant": "abliterated",
        "quantization":  "fp16",
        "model_generation": "2",
    },
    {
        "tag":           "llama_8b_q4",
        "model_path":    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
        "model_name":    "Meta-Llama-3-8B-Instruct-abliterated-v3",
        "model_family":  "llama",
        "model_size":    "8b",
        "size_dir":      "8b_4bit",
        "model_variant": "abliterated",
        "quantization":  "4bit",
        "model_generation": "3",
    },
    {
        "tag":           "gemma_9b_q4",
        "model_path":    "IlyaGusev/gemma-2-9b-it-abliterated",
        "model_name":    "gemma-2-9b-it-abliterated",
        "model_family":  "gemma",
        "model_size":    "9b",
        "size_dir":      "9b_4bit",
        "model_variant": "abliterated",
        "quantization":  "4bit",
        "model_generation": "2",
    },
]

RUNS_TO_RECOLLECT = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23]
SESSION_TEMPS    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Temperature-independent runs in our set. The IOTA framework's
# _copy_temperature_independent_runs (called at the start of every
# _run_session) auto-copies R0001 / R0002 / R0003 data from
# deterministic/ into the current temp dir whenever any T > 0 run
# fires. So we only need to fire these runs once at T=0; the framework
# distributes them across the other 5 temp dirs for free. Firing them
# 6 times produces byte-identical content (internal seeds match across
# session temps), wasting 5/6 of the wall time.
#
# R0002 is the only TEMP_INDEP run in Phase B's set. R0001 and R0003
# were handled in Phase A.
TEMP_INDEP_RUNS_LOCAL = {2}


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_status():
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding='utf-8'))
    return {
        "start_iso": datetime.now().isoformat(),
        "models": {},
        "post_collection": {},
    }


def save_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(status, indent=2), encoding='utf-8')
    tmp.replace(STATUS_PATH)


def update_session(model_cfg, run_num, temperature):
    s = json.loads(SESSION.read_text())
    s["model_path"]        = model_cfg["model_path"]
    s["model_name"]        = model_cfg["model_name"]
    s["model_family"]      = model_cfg["model_family"]
    s["model_size"]        = model_cfg["model_size"]
    s["model_variant"]     = model_cfg["model_variant"]
    s["quantization"]      = model_cfg["quantization"]
    s["model_generation"]  = model_cfg["model_generation"]
    s["temperature"]       = float(temperature)
    s["runs"]              = str(run_num)
    s["_saved"]            = datetime.now().isoformat()
    SESSION.write_text(json.dumps(s, indent=2))


def clear_model_runs(model_cfg):
    """Move all v28 data for runs 2, 4-15, 23 aside (per-model archive)."""
    log(f"  clearing {model_cfg['tag']} for runs 2, 4-15, 23")
    args = [PY, "_clear_for_recollection.py",
            "--family",  model_cfg["model_family"],
            "--size",    model_cfg["size_dir"],
            "--variant", model_cfg["model_variant"],
            "--runs"] + [str(r) for r in RUNS_TO_RECOLLECT]
    rc = subprocess.run(args, cwd=str(ROOT)).returncode
    log(f"    clear exit code: {rc}")
    return rc == 0


def fire_run(model_cfg, run_num, temperature):
    update_session(model_cfg, run_num, temperature)
    t0 = time.time()
    log(f"    firing R{run_num:04d} T={temperature}  ({model_cfg['tag']})")
    rc = subprocess.run(
        [PY, "start_here.py", "--single-run", str(run_num)],
        cwd=str(ROOT),
    ).returncode
    dt = time.time() - t0
    log(f"      -> rc={rc}  ({dt:.0f}s)")
    return rc == 0


def run_one_model(model_cfg, status):
    tag = model_cfg["tag"]
    if tag not in status["models"]:
        status["models"][tag] = {
            "start_iso": datetime.now().isoformat(),
            "cleared": False,
            "cells": {},
        }
        save_status(status)
    m_state = status["models"][tag]

    log(f"=== {tag} BEGIN  ({len(RUNS_TO_RECOLLECT)} runs x {len(SESSION_TEMPS)} temps = {len(RUNS_TO_RECOLLECT)*len(SESSION_TEMPS)} firings) ===")

    # Step 0: one-shot clear (only on first encounter)
    if not m_state.get("cleared", False):
        ok = clear_model_runs(model_cfg)
        m_state["cleared"] = ok
        m_state["cleared_iso"] = datetime.now().isoformat()
        save_status(status)
        if not ok:
            log(f"  {tag}: clear failed, aborting model")
            m_state["end_iso"] = datetime.now().isoformat()
            return

    # Step 1: 14 runs x 6 temps, with TEMP_INDEP optimization.
    # For runs in TEMP_INDEP_RUNS_LOCAL (just R0002): fire once at T=0;
    # framework auto-copies to the other 5 temp dirs. Mark the T > 0
    # cells as done-via-auto-copy so resume logic skips them cleanly.
    for run_num in RUNS_TO_RECOLLECT:
        for temp in SESSION_TEMPS:
            cell_key = f"r{run_num:04d}_t{temp}"
            if m_state["cells"].get(cell_key, {}).get("done"):
                continue

            # Skip TEMP_INDEP runs at T > 0; framework's
            # _copy_temperature_independent_runs handles distribution.
            if run_num in TEMP_INDEP_RUNS_LOCAL and temp != 0.0:
                log(f"    skip R{run_num:04d} T={temp}  (TEMP_INDEP -- auto-copy)")
                m_state["cells"][cell_key] = {
                    "done":   True,
                    "iso":    datetime.now().isoformat(),
                    "skipped_via": "auto_copy_temp_indep",
                }
                save_status(status)
                continue

            ok = fire_run(model_cfg, run_num, temp)
            m_state["cells"][cell_key] = {
                "done":   ok,
                "iso":    datetime.now().isoformat(),
            }
            save_status(status)
            if not ok:
                log(f"    !! {cell_key} failed; continuing")

    m_state["end_iso"] = datetime.now().isoformat()
    log(f"=== {tag} END ===")


def refresh_q0042(model_cfg, status):
    """Phase B post-collection: refresh Q0042 (E+C+R decomposition) per cell."""
    tag = model_cfg["tag"]
    log(f"  refreshing Q0042 for {tag} across 6 session-temps")
    for temp in SESSION_TEMPS:
        update_session(model_cfg, 42, temp)
        t0 = time.time()
        rc = subprocess.run(
            [PY, "start_here.py", "--single-run", "42"],
            cwd=str(ROOT),
        ).returncode
        dt = time.time() - t0
        log(f"    Q0042 T={temp}: rc={rc}  ({dt:.0f}s)")
        status["post_collection"].setdefault(tag, {})[f"q0042_t{temp}"] = {
            "rc": rc, "iso": datetime.now().isoformat(),
        }
        save_status(status)


def run_bayesian_apparatus(status):
    """Phase 1-12 of run_bayesian_apparatus.py. Produces the refreshed
    apparatus_aggregator/per_cell.json that feeds Paper A §4."""
    log("  firing run_bayesian_apparatus.py (Run 0058, all 12 phases)")
    t0 = time.time()
    rc = subprocess.run(
        [PY, "run_bayesian_apparatus.py"],
        cwd=str(ROOT),
    ).returncode
    dt = time.time() - t0
    log(f"    Run 0058: rc={rc}  ({dt:.0f}s)")
    status["post_collection"]["run_0058"] = {
        "rc": rc, "iso": datetime.now().isoformat(),
    }
    save_status(status)


def main():
    log("=== Phase B re-collection start ===")
    log(f"  models: {[m['tag'] for m in MODELS]}")
    log(f"  runs:   {RUNS_TO_RECOLLECT}")
    log(f"  temps:  {SESSION_TEMPS}")
    log(f"  total firings: {len(MODELS) * len(RUNS_TO_RECOLLECT) * len(SESSION_TEMPS)}")

    status = load_status()
    if "start_iso" not in status:
        status["start_iso"] = datetime.now().isoformat()
    save_status(status)

    # Collection loop
    for model_cfg in MODELS:
        try:
            run_one_model(model_cfg, status)
        except Exception as e:
            log(f"  EXCEPTION in {model_cfg['tag']}: {type(e).__name__}: {e}")

    # Post-collection: Q0042 refresh per cell
    log("=== post-collection: Q0042 refresh ===")
    for model_cfg in MODELS:
        try:
            refresh_q0042(model_cfg, status)
        except Exception as e:
            log(f"  Q0042 EXCEPTION in {model_cfg['tag']}: {type(e).__name__}: {e}")

    # Post-collection: Run 0058 (Bayesian apparatus, 12 phases)
    log("=== post-collection: Run 0058 (Bayesian apparatus) ===")
    try:
        run_bayesian_apparatus(status)
    except Exception as e:
        log(f"  Run 0058 EXCEPTION: {type(e).__name__}: {e}")

    status["end_iso"] = datetime.now().isoformat()
    save_status(status)
    log("=== Phase B re-collection complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
