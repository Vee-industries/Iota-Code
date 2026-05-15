"""
_full_recollection.py -- multi-model re-collection sequencer.

For each of the four planned model configs:
  1. Switch last_session.json to the target model.
  2. Clear its existing run files into a per-model archive.
  3. Fire R0001 (trivariant null, T=0; ~10-30 min).
  4. Wait for the R0001 Pass-4 sentinel.
  5. Fire R0003 (temperature grid; ~1-3 h depending on size).
  6. Fire R0039 across 6 temperatures (~30-60 min per T per model).

No per-step integrity check. The earlier check assumed cross-temperature
hidden-state variation was required; under the corrected apparatus reading
(status enforcer is by design, T-variation lives in occasional sampling
flips that propagate forward, not in every spot-check cell), that check
flags false positives. Re-introduce a smarter R-level check later.

Status log: data/paper/recollection_status_full.json
Orchestrator log: .iota_full_recollection.log

Expected total runtime: 12-20 h across all four models.
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
LOGFILE = ROOT / ".iota_full_recollection.log"
DATA_ROOT = ROOT / "data"

# Four model configs to recollect.
# Each entry drives both the clear script and the session switch.
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

TEMPS_R0039 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_session(model_cfg, run_num, temperature):
    """Rewrite last_session.json for a given model + run + T."""
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


def fire_run(model_cfg, run_num, temperature, label):
    update_session(model_cfg, run_num, temperature)
    log(f"  firing {model_cfg['tag']} Run {run_num:04d} {label} (T={temperature})")
    rc = subprocess.run(
        [PY, "start_here.py", "--single-run", str(run_num)],
        cwd=str(ROOT),
    ).returncode
    log(f"    -> exit code {rc}")
    return rc == 0


def wait_for_r0001_sentinel(model_cfg, timeout_s=14400):
    """Pass 4 drops R0001_*_pass4_ok.stamp under the abliterated/deterministic
    hidden_states dir. Block until it appears or we hit the timeout."""
    abl_det = (DATA_ROOT / model_cfg["model_family"] / model_cfg["size_dir"]
               / model_cfg["model_variant"] / "deterministic" / "hidden_states")
    log(f"  waiting for R0001 Pass 4 sentinel under {abl_det}")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if abl_det.exists() and list(abl_det.glob("R0001_*pass4_ok.stamp")):
            log(f"    sentinel found after {int(time.time()-t0)}s")
            return True
        time.sleep(30)
    log(f"    sentinel TIMEOUT after {timeout_s}s")
    return False


def clear_model(model_cfg):
    """Move existing run-3/39 files for this model to a per-model archive.

    R0001 is intentionally NOT cleared -- v28's R0001 baseline data is
    still valid (deterministic at T=0, apparatus unchanged for T=0), so
    we keep whatever R0001 files exist on disk for each model.
    """
    log(f"  clearing existing {model_cfg['tag']} data (R0003 + R0039 only)")
    rc = subprocess.run(
        [PY, "_clear_for_recollection.py",
         "--family",  model_cfg["model_family"],
         "--size",    model_cfg["size_dir"],
         "--variant", model_cfg["model_variant"],
         "--runs", "3", "39"],
        cwd=str(ROOT),
    ).returncode
    log(f"    clear exit code: {rc}")
    return rc == 0


def write_status(status):
    out = DATA_ROOT / "paper" / "recollection_status_full.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2))


def run_one_model(model_cfg, status):
    tag = model_cfg["tag"]
    model_record = {"tag": tag, "steps": [],
                    "start_iso": datetime.now().isoformat()}
    status["models"].append(model_record)
    write_status(status)

    log(f"=== {tag} BEGIN ===")

    # Step 0: clear
    ok = clear_model(model_cfg)
    model_record["steps"].append({"step": "clear",
                                  "result": "OK" if ok else "FAIL"})
    write_status(status)
    if not ok:
        log(f"  {tag}: clear failed, skipping model")
        model_record["end_iso"] = datetime.now().isoformat()
        return

    # R0001 step removed: the v28 R0001 trivariant null baseline runs at T=0
    # only and is deterministic. Neither the seed fix nor the status enforcer
    # state changed in a way that affects T=0 outputs, so the v28 R0001 data
    # (archived under data/_archive_pre_v1_recollection_2026-05-14_104220/)
    # is still scientifically valid. Restore it from the archive after this
    # batch if needed. Wait for R0001 sentinel is also skipped -- R0003 and
    # R0039 do not consume R0001 outputs.

    # Step 2: R0003 (internal temperature iteration)
    ok = fire_run(model_cfg, 3, 0.0, "(temperature grid)")
    model_record["steps"].append({"step": "r0003",
                                  "result": "OK" if ok else "FAIL"})
    write_status(status)
    if not ok:
        log(f"  {tag}: R0003 fire failed, continuing to R0039")

    # Step 3: R0039 across six temperatures
    for t in TEMPS_R0039:
        ok = fire_run(model_cfg, 39, t, f"(jolt T={t})")
        model_record["steps"].append({"step": f"r0039_t{t}",
                                      "result": "OK" if ok else "FAIL"})
        write_status(status)

    model_record["end_iso"] = datetime.now().isoformat()
    log(f"=== {tag} END ===")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-model", default=None,
                    help="Tag of the first model to process; earlier "
                         "entries in MODELS are skipped. Use this to "
                         "resume after a partial run.")
    args = ap.parse_args()

    models = list(MODELS)
    if args.start_model:
        tags = [m["tag"] for m in models]
        if args.start_model not in tags:
            log(f"ERROR: --start-model={args.start_model} not in {tags}")
            return 2
        start_idx = tags.index(args.start_model)
        models = models[start_idx:]
        log(f"  resuming from model: {args.start_model} "
            f"(skipping {start_idx} earlier entries)")

    log("=== full re-collection start ===")
    log(f"  models: {[m['tag'] for m in models]}")
    log(f"  sequence per model: clear -> R0003 -> R0039 x 6T  "
        f"(R0001 skipped -- v28 baseline reused from archive)")
    status = {
        "start_iso": datetime.now().isoformat(),
        "model_count": len(models),
        "models": [],
    }
    write_status(status)

    for model_cfg in models:
        try:
            run_one_model(model_cfg, status)
        except Exception as e:
            log(f"  EXCEPTION in {model_cfg['tag']}: {e}")

    status["end_iso"] = datetime.now().isoformat()
    write_status(status)
    log("=== full re-collection complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
