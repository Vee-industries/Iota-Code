"""
_phase_d_section7.py -- §7 follow-up collection for Paper A.

Fires Run 0060 (Perturbation A with non-thematic recovery prompts) across
all four model configurations at all six decoding temperatures = 24 cells.

Run 0060 mirrors Run 0039 exactly except for the recovery prompts at
turns 14-16. The §3 cohort discrimination analysis (Run 0061) then
operates on both Run 0039 (thematic) and Run 0060 (non-thematic) and
compares discrimination strength to separate two mechanistic readings:

  - Spontaneous retention: cohort signal embedded in the recomputed
    hidden-state trajectory regardless of recovery-prompt content.
    Prediction: discrimination strength is preserved under non-thematic
    recovery; per-cell p-values remain at the permutation floor for
    quantized cells and follow the same FP16-at-temperature attenuation
    pattern reported in Paper A §3.3.
  - Input-triggered reactivation: recovery-prompt thematic relatedness
    cues the model to re-attend to prior conversational content.
    Prediction: discrimination strength attenuates sharply under non-
    thematic recovery; per-cell p-values move toward chance (p ≈ 0.5)
    across the fleet.

Status log: data/paper/recollection_status_phase_d.json
Orchestrator log: .iota_phase_d_section7.log

Expected wall time: ~3-5h continuous GPU, comparable to one R0039
sweep in Phase A.
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
LOGFILE = ROOT / ".iota_phase_d_section7.log"
DATA_ROOT = ROOT / "data"
STATUS_PATH = DATA_ROOT / "paper" / "recollection_status_phase_d.json"

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
SESSION_TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


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
    s["trials"]           = 100  # matches R0039 trial count; needed for §3 permutation test
    s["runs"]             = "60"
    s["_saved"]           = datetime.now().isoformat()
    SESSION.write_text(json.dumps(s, indent=2))


def fire_r0060(model_cfg, temperature):
    update_session(model_cfg, temperature)
    t0 = time.time()
    log(f"    firing R0060 T={temperature}  ({model_cfg['tag']})")
    rc = subprocess.run(
        [PY, "start_here.py", "--single-run", "60"],
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

    log(f"=== {tag} BEGIN  (R0060 across {len(SESSION_TEMPS)} session temps) ===")

    for temp in SESSION_TEMPS:
        cell_key = f"r0060_t{temp}"
        if m_state["cells"].get(cell_key, {}).get("done"):
            continue
        ok = fire_r0060(model_cfg, temp)
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
    log("=== Phase D (§7 follow-up) start ===")
    log(f"  models: {[m['tag'] for m in MODELS]}")
    log(f"  run: R0060 (non-thematic recovery jolt)")
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
    log("=== Phase D complete ===")
    log("  next: rerun Run 0061 (jolt-carryover analysis) against R0060 substrate")
    log("        to produce per-cell discrimination p-values under non-thematic")
    log("        recovery. Compare to existing R0039 thematic readings for the")
    log("        spontaneous-retention vs input-triggered-reactivation split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
