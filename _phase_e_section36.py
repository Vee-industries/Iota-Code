"""
_phase_e_section36.py -- Paper A §3.6 follow-up collection.

Fires Run 0062 (Perturbation A with status-token enforcer suppressed on
recovery turns 14-16) on the four deterministic cells: one T=0.0 cell per
model configuration (4 cells total, 100 trials each = 400 trials).

Run 0062 mirrors Run 0039 in every respect EXCEPT the status-token
enforcer is OFF for the three recovery turns. Pre-recovery threading
through turn 13 (including the shock-turn single-token response) is
matched to R0039 at T=0 — the shock-turn response is the same
status token across all cohorts at every deterministic cell (verified
on R0039 substrate: DONE everywhere). Recovery-turn responses are
free-form (LONG_OUT_TOKENS budget) and captured in the CSV `output`
column; recovery-turn pooled hidden states are saved for the §3
permutation test.

The §3.6 question the data answers, at the four deterministic cells:
  - Are free-form responses byte-identical across cohorts under
    threading-matched-to-R0039 conditions? (the §3.6 compound claim
    holds in unqualified form, with no behavioral-channel caveat)
  - Are responses semantically equivalent but not byte-identical?
    (cohort signal surfaces as fine-grained generation variance
    without semantic divergence)
  - Are responses semantically divergent across cohorts? (cohort
    signal surfaces as discriminable output content; this is the
    first empirical brick in a coherence-to-behavior bridge)

Status log: data/paper/recollection_status_phase_e.json
Orchestrator log: .iota_phase_e_section36.log

Expected wall time: 4 cells x 100 trials x ~6 turns of free-form
generation at recovery (plus 13 single-token turns pre-recovery)
~= ~30-90 min depending on model. Per-cell budget LONG_OUT_TOKENS
caps recovery-turn generation length.
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
LOGFILE = ROOT / ".iota_phase_e_section36.log"
DATA_ROOT = ROOT / "data"
STATUS_PATH = DATA_ROOT / "paper" / "recollection_status_phase_e.json"

MODELS = [
    {
        "tag":              "gemma_2b_q4",
        "model_path":       "IlyaGusev/gemma-2-2b-it-abliterated",
        "model_name":       "gemma-2-2b-it-abliterated",
        "model_family":     "gemma",
        "model_size":       "2b",
        "size_dir":         "2b_4bit",
        "model_variant":    "abliterated",
        "quantization":     "4bit",
        "model_generation": "2",
    },
    {
        "tag":              "gemma_2b_fp16",
        "model_path":       "IlyaGusev/gemma-2-2b-it-abliterated",
        "model_name":       "gemma-2-2b-it-abliterated",
        "model_family":     "gemma",
        "model_size":       "2b",
        "size_dir":         "2b_fp16",
        "model_variant":    "abliterated",
        "quantization":     "fp16",
        "model_generation": "2",
    },
    {
        "tag":              "llama_8b_q4",
        "model_path":       "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
        "model_name":       "Meta-Llama-3-8B-Instruct-abliterated-v3",
        "model_family":     "llama",
        "model_size":       "8b",
        "size_dir":         "8b_4bit",
        "model_variant":    "abliterated",
        "quantization":     "4bit",
        "model_generation": "3",
    },
    {
        "tag":              "gemma_9b_q4",
        "model_path":       "IlyaGusev/gemma-2-9b-it-abliterated",
        "model_name":       "gemma-2-9b-it-abliterated",
        "model_family":     "gemma",
        "model_size":       "9b",
        "size_dir":         "9b_4bit",
        "model_variant":    "abliterated",
        "quantization":     "4bit",
        "model_generation": "2",
    },
]
# T=0.0 only: §3.6 is about the deterministic cells where the
# behavioral-silence pattern was observed and the cascade is empirically
# absent. Higher temperatures introduce stochastic decoding noise that
# would conflate the question; if the deterministic readings are clean,
# a temperature sweep is the natural extension.
SESSION_TEMPS = [0.0]


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_status():
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding='utf-8'))
    return {"start_iso": datetime.now().isoformat(), "models": {}}


def save_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(status, indent=2), encoding='utf-8')
    tmp.replace(STATUS_PATH)


def update_session(model_cfg, temperature):
    s = json.loads(SESSION.read_text())
    s["model_path"]       = model_cfg["model_path"]
    s["model_name"]       = model_cfg["model_name"]
    s["model_family"]     = model_cfg["model_family"]
    s["model_size"]       = model_cfg["model_size"]
    s["model_variant"]    = model_cfg["model_variant"]
    s["quantization"]     = model_cfg["quantization"]
    s["model_generation"] = model_cfg["model_generation"]
    s["temperature"]      = float(temperature)
    s["trials"]           = 100
    s["runs"]             = "62"
    s["_saved"]           = datetime.now().isoformat()
    SESSION.write_text(json.dumps(s, indent=2))


def fire_r0062(model_cfg, temperature):
    update_session(model_cfg, temperature)
    t0 = time.time()
    log(f"    firing R0062 T={temperature}  ({model_cfg['tag']})")
    rc = subprocess.run(
        [PY, "start_here.py", "--single-run", "62"],
        cwd=str(ROOT),
    ).returncode
    dt = time.time() - t0
    log(f"      -> rc={rc}  ({int(dt)}s)")
    return rc == 0


def run_one_model(model_cfg, status):
    tag = model_cfg["tag"]
    if tag not in status["models"]:
        status["models"][tag] = {
            "start_iso": datetime.now().isoformat(),
            "cells":     {},
        }
        save_status(status)
    m_state = status["models"][tag]

    log(f"=== {tag} BEGIN  (R0062 at T=0.0 only) ===")

    for temp in SESSION_TEMPS:
        cell_key = f"r0062_t{temp}"
        if m_state["cells"].get(cell_key, {}).get("done"):
            continue
        ok = fire_r0062(model_cfg, temp)
        m_state["cells"][cell_key] = {
            "done": ok,
            "iso":  datetime.now().isoformat(),
        }
        save_status(status)
        if not ok:
            log(f"    !! {cell_key} failed; continuing")

    m_state["end_iso"] = datetime.now().isoformat()
    log(f"=== {tag} END ===")


def main():
    log("=== Phase E (§3.6 follow-up) start ===")
    log(f"  models: {[m['tag'] for m in MODELS]}")
    log(f"  run: R0062 (enforcer-off recovery jolt)")
    log(f"  temps: {SESSION_TEMPS}")
    log(f"  total firings: {len(MODELS) * len(SESSION_TEMPS)}")

    status = load_status()
    save_status(status)

    for model_cfg in MODELS:
        try:
            run_one_model(model_cfg, status)
        except Exception as e:
            log(f"  EXCEPTION in {model_cfg['tag']}: {type(e).__name__}: {e}")

    status["end_iso"] = datetime.now().isoformat()
    save_status(status)
    log("=== Phase E complete ===")
    log("  next: rerun Run 0061 (jolt-carryover analysis) against R0062 substrate")
    log("        and compare per-cell discrimination p-values to R0039 thematic")
    log("        readings. Then compare free-form recovery-turn outputs across")
    log("        cohorts at each cell for the §3.6 behavioral-channel question.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
