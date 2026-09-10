"""
_recollect_run0023.py -- fire Run 0023 (C_t confound isolation) for real on the 24 fleet cells.

Why: the 2026-05 Phase B re-collection archived the April Run 0023 hidden states
(`_archive_pre_v1_recollection_*`) and then "fired" R0023 at every temperature, but each
firing exited after ~20 s with rc=0 because `Q0023_introspection.csv` (mtime 2026-04-20)
was left in place and the gap-aware resume logic (`runners_core._get_trials_for_condition`)
treated every trial as already done. Result: no Q0023 hidden states on any fleet cell.

What this does, per (model, temperature), smallest model first:
  1. moves a stale `csv/Q0023_introspection.csv` (mtime before 2026-05-01) to
     `Q0023_introspection.csv.stale_pre_recollection` so the run starts from trial 0;
  2. writes the session exactly as `_full_runs_recollection.update_session` does;
  3. runs `start_here.py --single-run 23` and logs rc + wall time to `.iota_run0023_recollection.log`.

Usage: py _recollect_run0023.py                 -> all 4 models x 6 temps
       py _recollect_run0023.py --only gemma_2b_q4 --temps 0.0     (test one cell)
       py _recollect_run0023.py --models gemma_2b_q4 gemma_2b_fp16 (subset)
"""
import os, sys, json, time, argparse, subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SESSION = ROOT / "last_session.json"
LOGFILE = ROOT / ".iota_run0023_recollection.log"
STALE_BEFORE = datetime(2026, 5, 1).timestamp()

MODELS = {
    "gemma_2b_q4":   dict(model_path="IlyaGusev/gemma-2-2b-it-abliterated", model_name="gemma-2-2b-it-abliterated",
                          model_family="gemma", model_size="2b", size_dir="2b_4bit", model_variant="abliterated",
                          quantization="4bit", model_generation="2"),
    "gemma_2b_fp16": dict(model_path="IlyaGusev/gemma-2-2b-it-abliterated", model_name="gemma-2-2b-it-abliterated",
                          model_family="gemma", model_size="2b", size_dir="2b_fp16", model_variant="abliterated",
                          quantization="fp16", model_generation="2"),
    "llama_8b_q4":   dict(model_path="failspy/Meta-Llama-3-8B-Instruct-abliterated-v3", model_name="Meta-Llama-3-8B-Instruct-abliterated-v3",
                          model_family="llama", model_size="8b", size_dir="8b_4bit", model_variant="abliterated",
                          quantization="4bit", model_generation="3"),
    "gemma_9b_q4":   dict(model_path="IlyaGusev/gemma-2-9b-it-abliterated", model_name="gemma-2-9b-it-abliterated",
                          model_family="gemma", model_size="9b", size_dir="9b_4bit", model_variant="abliterated",
                          quantization="4bit", model_generation="2"),
}
ORDER = ["gemma_2b_q4", "gemma_2b_fp16", "llama_8b_q4", "gemma_9b_q4"]
TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cell_csv(cfg, temp):
    tdir = "deterministic" if temp == 0.0 else f"temp_{temp}"
    return ROOT / "data" / cfg["model_family"] / cfg["size_dir"] / cfg["model_variant"] / tdir / "csv" / "Q0023_introspection.csv"


def update_session(cfg, run_num, temperature):
    s = json.loads(SESSION.read_text())
    for k in ("model_path", "model_name", "model_family", "model_size", "model_variant", "quantization", "model_generation"):
        s[k] = cfg[k]
    s["temperature"] = float(temperature)
    s["runs"] = str(run_num)
    s["_saved"] = datetime.now().isoformat()
    SESSION.write_text(json.dumps(s, indent=2))


def wait_for_gpu(max_used_mib=1500, timeout_s=180):
    """Both 9B OOMs (2026-09-10 08:31, 09:36) hit the cell fired immediately after a completed 9B cell:
    the previous subprocess's VRAM was not yet released. Poll nvidia-smi until the card has drained."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            used = int(out.split()[0])
        except Exception:
            return
        if used <= max_used_mib:
            if time.time() - t0 > 1:
                log(f"    GPU drained to {used} MiB after {time.time() - t0:.0f}s")
            return
        time.sleep(5)
    log(f"    WARNING: GPU still at {used} MiB after {timeout_s}s; firing anyway")


def fire(tag, temp):
    cfg = MODELS[tag]
    csv = cell_csv(cfg, temp)
    if csv.exists() and os.path.getmtime(csv) < STALE_BEFORE:
        dst = csv.with_name(csv.name + ".stale_pre_recollection")
        if dst.exists():
            dst = csv.with_name(csv.name + f".stale_pre_recollection.{int(time.time())}")
        csv.rename(dst)
        log(f"    moved stale CSV -> {dst.name}")
    update_session(cfg, 23, temp)
    wait_for_gpu()
    t0 = time.time()
    log(f"    firing R0023 T={temp}  ({tag})")
    rc = subprocess.run([PY, "start_here.py", "--single-run", "23"], cwd=str(ROOT)).returncode
    log(f"      -> rc={rc}  ({time.time() - t0:.0f}s)")
    hid = csv.parent.parent / "hidden_states"
    n = len([f for f in os.listdir(hid) if f.startswith("Q0023_") and "_turn" in f and f.endswith(".npy")]) if hid.is_dir() else 0
    log(f"      Q0023 .npy files now in {hid.name}: {n}")
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="one model tag")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--temps", nargs="*", type=float, default=None)
    a = ap.parse_args()
    tags = [a.only] if a.only else (a.models or ORDER)
    temps = a.temps or TEMPS
    log(f"=== Run 0023 re-collection: models {tags}, temps {temps} ===")
    for tag in tags:
        for temp in temps:
            if not fire(tag, temp):
                log(f"    STOP: rc != 0 on {tag} T={temp}")
                return 1
    log("=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
