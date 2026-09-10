"""
reproduce.py -- run the IOTA pipeline end to end, unattended, on one consumer GPU.

    python reproduce.py --plan            # print the stage list and exit
    python reproduce.py                   # run everything, resuming whatever is already done
    python reproduce.py --from analysis   # start at a stage (see --plan for names)
    python reproduce.py --only figures    # one stage (comma-separated for several)
    python reproduce.py --models gemma_2b_q4 gemma_9b_q4   # restrict the per-model stages
    python reproduce.py --skip-collection # analysis and calibration only (states already on disk)

What it does, in order (every stage is idempotent: the framework's own resume logic skips
completed trials, the calibration scripts skip existing outputs, and this driver records each
finished stage in data/paper/reproduce_status.json so a restart continues where it stopped):

  env          check Python packages, CUDA, the session file (created from defaults if absent;
               set HF_TOKEN in the environment for gated models), and run the unit tests
  collect      per model (2B Q4, 2B FP16, 8B Q4, 9B Q4): Run 0001, 0003, 0002 at T=0
               (temperature-independent); then per temperature 0.0..1.0: runs 4-15 and 23
               (E_t-source runs), 39 (jolt), 60 (non-thematic jolt), 16 (E_t recovery on the
               base model); at T=0 also 62 (enforcer-off jolt), 17 and 18 (patching)
  analysis     per model: `start_here.py --all-stats` (runs 41-54 at every temperature,
               including Run 0042, the E+C+R decomposition every downstream stage reads)
  calibration  operating point (split/PCA) -> kNN-MI reliability -> V5 synthetic suite ->
               Run 0056 (methodology calibration: V5b + channel marginal + manifest) ->
               Run 0057 (function-class sensitivity) -> Run 0058 (apparatus, 11 phases) ->
               V5 lambda calibration + finalization -> V5g bridge (n_steps 8000) ->
               tau_H4 derivation
  behavioral   Run 0061 cohort-discrimination analysis on Run 0039, 0060 and 0062
  results      Run 0059: results.json + paper figures
  probes       the 2026-09-10 measurements: Gaussian redundancy on de-duplicated rows,
               multi-step persistence R_k, KV-cache interpolation (GPU), drop-history (GPU)
  verify       compare the rebuilt results.json against the released results8th.json and
               print the headline numbers next to the values the papers quote

Wall clock on an RTX 3080 (10 GB): collection is the cost, roughly 1-2 days per model
(Run 0003 is the long one); analysis and calibration are CPU and take a few hours;
probes and verify take minutes. GPU stages wait for the card to drain before launching
and use IOTA_VRAM_FRACTION=0.93 for the 9B (the 0.85 default OOMs at load there).
Log: .iota_reproduce.log. Stop any time; rerun the same command to continue.
"""
import os, sys, json, time, argparse, subprocess, shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
SESSION = ROOT / "last_session.json"
STATUS = ROOT / "data" / "paper" / "reproduce_status.json"
LOG = ROOT / ".iota_reproduce.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", IOTA_HEADLESS="1")

MODELS = {
    "gemma_2b_q4":   dict(model_path="IlyaGusev/gemma-2-2b-it-abliterated", model_name="gemma-2-2b-it-abliterated",
                          model_family="gemma", model_size="2b", size_dir="2b_4bit", model_variant="abliterated",
                          quantization="4bit", model_generation="2", vram=None),
    "gemma_2b_fp16": dict(model_path="IlyaGusev/gemma-2-2b-it-abliterated", model_name="gemma-2-2b-it-abliterated",
                          model_family="gemma", model_size="2b", size_dir="2b_fp16", model_variant="abliterated",
                          quantization="fp16", model_generation="2", vram=None),
    "llama_8b_q4":   dict(model_path="failspy/Meta-Llama-3-8B-Instruct-abliterated-v3", model_name="Meta-Llama-3-8B-Instruct-abliterated-v3",
                          model_family="llama", model_size="8b", size_dir="8b_4bit", model_variant="abliterated",
                          quantization="4bit", model_generation="3", vram=None),
    "gemma_9b_q4":   dict(model_path="IlyaGusev/gemma-2-9b-it-abliterated", model_name="gemma-2-9b-it-abliterated",
                          model_family="gemma", model_size="9b", size_dir="9b_4bit", model_variant="abliterated",
                          quantization="4bit", model_generation="2", vram="0.93"),
}
ORDER = ["gemma_2b_q4", "gemma_2b_fp16", "llama_8b_q4", "gemma_9b_q4"]
TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TEMP_INDEP = [1, 3, 2]                     # fired once at T=0; the framework copies them across temps
PER_TEMP = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 39, 60]
T0_ONLY = [62, 17, 18]
STAGES = ["env", "collect", "analysis", "calibration", "behavioral", "results", "probes", "verify"]

DEFAULT_SESSION = {
    "model_path": "", "model_name": "", "model_family": "gemma", "model_size": "2b", "model_variant": "abliterated",
    "quantization": "4bit", "model_generation": "2", "temperature": 0.0, "trials": 100, "clear_cache": True,
    "runs": "", "seed": 42, "hf_token": "", "low_priority": True, "force_gc": False, "unload_between_runs": True,
}


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_status():
    try:
        return json.load(open(STATUS))
    except Exception:
        return {"done": {}, "started": datetime.now().isoformat()}


def mark(status, key):
    status["done"][key] = datetime.now().isoformat(timespec="seconds")
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(status, open(STATUS, "w"), indent=1)


def sh(cmd, env=None, label=None):
    t0 = time.time()
    log(f"  $ {' '.join(str(c) for c in cmd)}" if label is None else f"  {label}")
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env or ENV).returncode
    log(f"    -> rc={rc} ({time.time() - t0:.0f}s)")
    return rc == 0


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out.split()[0])
    except Exception:
        return None


def wait_for_gpu(max_used=1500, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = gpu_used_mib()
        if used is None or used <= max_used:
            return
        time.sleep(5)
    log(f"    WARNING: GPU still at {used} MiB after {timeout}s; continuing")


def set_session(cfg, temp, runs):
    s = json.loads(SESSION.read_text()) if SESSION.exists() else dict(DEFAULT_SESSION)
    for k in ("model_path", "model_name", "model_family", "model_size", "model_variant", "quantization", "model_generation"):
        s[k] = cfg[k]
    s["temperature"] = float(temp); s["runs"] = str(runs); s["_saved"] = datetime.now().isoformat()
    if not s.get("hf_token") and os.environ.get("HF_TOKEN"):
        s["hf_token"] = os.environ["HF_TOKEN"]
    SESSION.write_text(json.dumps(s, indent=2))


def model_env(cfg):
    e = dict(ENV)
    if cfg.get("vram"):
        e["IOTA_VRAM_FRACTION"] = cfg["vram"]
    return e


def fire(cfg, temp, run, label):
    set_session(cfg, temp, run)
    wait_for_gpu()
    return sh([PY, "start_here.py", "--single-run", str(run)], env=model_env(cfg), label=f"{label}: run {run:04d} T={temp} ({cfg['model_name']}, {cfg['quantization']})")


def wait_for_sentinel(cfg, timeout=4 * 3600):
    d = ROOT / "data" / cfg["model_family"] / cfg["size_dir"] / cfg["model_variant"] / "deterministic" / "hidden_states"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if d.is_dir() and list(d.glob("R0001_*pass4_ok.stamp")):
            return True
        time.sleep(30)
    log("    WARNING: R0001 pass-4 sentinel not found; continuing")
    return False


# ── stages ──────────────────────────────────────────────────────────────────

def stage_env(models, status):
    import importlib.util
    missing = [m for m in ("numpy", "scipy", "sklearn", "pandas", "torch", "transformers") if importlib.util.find_spec(m) is None]
    if missing:
        log(f"  missing packages: {missing}; run `python start_here.py` once for the setup wizard, or pip install -r requirements.txt")
        return False
    try:
        import torch
        log(f"  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        log(f"  torch import failed: {e}"); return False
    if not SESSION.exists():
        SESSION.write_text(json.dumps(DEFAULT_SESSION, indent=2)); log("  created last_session.json from defaults")
    if not (ROOT / ".iota_env.json").exists():
        log("  NOTE: .iota_env.json absent (the interactive setup wizard has not run). Collection still works headless; the dashboard needs it.")
    ok = sh([PY, "-m", "unittest", "discover", "-s", "tests"], label="unit tests")
    return ok


def stage_collect(models, status):
    for tag in models:
        cfg = MODELS[tag]
        for run in TEMP_INDEP:
            key = f"collect/{tag}/T0.0/{run}"
            if key in status["done"]: continue
            if not fire(cfg, 0.0, run, "collect"): return False
            if run == 1: wait_for_sentinel(cfg)
            mark(status, key)
        for temp in TEMPS:
            runs = PER_TEMP + [16] + (T0_ONLY if temp == 0.0 else [])
            for run in runs:
                key = f"collect/{tag}/T{temp}/{run}"
                if key in status["done"]: continue
                if not fire(cfg, temp, run, "collect"): return False
                mark(status, key)
    return True


def stage_analysis(models, status):
    for tag in models:
        key = f"analysis/{tag}"
        if key in status["done"]: continue
        set_session(MODELS[tag], 0.0, "")
        if not sh([PY, "start_here.py", "--all-stats"], label=f"analysis: --all-stats ({tag})"): return False
        mark(status, key)
    return True


def stage_calibration(models, status):
    set_session(MODELS[models[0]], 0.0, "")
    steps = [
        ("split_pca",   [PY, "run_split_pca_selection.py"],             "data/paper/calibration/split_pca_selection/selection.json"),
        ("knn_mi",      [PY, "run_knn_mi_reliability.py"],              "data/paper/calibration/knn_mi_reliability/reliability.json"),
        ("v5",          [PY, "v5_synthetic_calibration.py"],            "data/paper/calibration/v5/v5_calibration_results.json"),
        ("run0056",     [PY, "start_here.py", "--single-run", "56"],    "data/paper/calibration/channel_marginal/channel_marginal_nonlinearity.csv"),
        ("run0057",     [PY, "start_here.py", "--single-run", "57"],    "data/paper/Q0057_function_class_sensitivity.json"),
        ("run0058",     [PY, "run_bayesian_apparatus.py"],              "data/paper/calibration/apparatus_p2/per_cell.json"),
        ("lambda",      [PY, "_run_v5_lambda_calibration.py"],          "data/paper/calibration/v5/v5_lambda_calibration.json"),
        ("lambda_fin",  [PY, "_finalize_lambda_calibration.py"],        "data/paper/calibration/v5/lambda_calibration_outcome.json"),
        ("v5g",         [PY, "_run_v5g_mid_dim_bridge.py", "--n-steps", "8000"], "data/paper/calibration/v5/v5g_bridge_calibration.json"),
        ("tau_h4",      [PY, "_derive_h4_threshold.py"],                "data/paper/calibration/v5/h4_threshold_derivation.json"),
    ]
    for name, cmd, out in steps:
        key = f"calibration/{name}"
        if key in status["done"] and (ROOT / out).exists(): continue
        env = dict(ENV, IOTA_N_BOOTSTRAP="12") if name == "run0058" else ENV   # the released phase-10 file used 12 resamples
        if not sh(cmd, env=env, label=f"calibration: {name}"): return False
        if not (ROOT / out).exists(): log(f"    WARNING: expected output missing: {out}")
        mark(status, key)
    return True


def stage_behavioral(models, status):
    for run in (39, 60, 62):
        key = f"behavioral/run0061_on_{run}"
        if key in status["done"]: continue
        if not sh([PY, "_run_run0061_jolt_carryover.py", "--run", str(run)], label=f"behavioral: Run 0061 on Run {run:04d}"): return False
        mark(status, key)
    return True


def stage_results(models, status):
    set_session(MODELS[models[0]], 0.0, "")
    if "results/run0059" not in status["done"]:
        if not sh([PY, "start_here.py", "--single-run", "59"], label="results: Run 0059 (results.json + figures)"): return False
        mark(status, "results/run0059")
    for f in sorted(ROOT.glob("fig*_*.py")):
        if f.name == "figures_common.py": continue
        key = f"results/{f.stem}"
        if key in status["done"]: continue
        sh([PY, f.name], label=f"results: {f.name}")
        mark(status, key)
    return True


def stage_probes(models, status):
    steps = [
        ("gauss",     [PY, "_gauss_ii_fleet.py", "fleet"], False),
        ("multistep", [PY, "_multistep_R.py"], False),
        ("kv",        [PY, "_kv_interpolation_test.py"], True),
        ("drop",      [PY, "_drop_history_probe.py"], True),
    ]
    for name, cmd, gpu in steps:
        key = f"probes/{name}"
        if key in status["done"]: continue
        if gpu: wait_for_gpu()
        if not sh(cmd, label=f"probes: {name}"): return False
        mark(status, key)
    return True


def stage_verify(models, status):
    new = ROOT / "data" / "paper" / "results.json"; ref = ROOT / "data" / "paper" / "results8th.json"
    if not ref.exists():
        log("  results8th.json (released) not found; nothing to compare against"); return True
    R = json.load(open(ref))["cells"]
    def summary(cells):
        import statistics
        r = [c["measurements"]["anchored_shares"]["R"] for c in cells.values() if c.get("measurements", {}).get("anchored_shares")]
        return (len(r), statistics.mean(r), min(r), max(r)) if r else (0, float("nan"), float("nan"), float("nan"))
    n, m, lo, hi = summary(R)
    log(f"  released results8th.json: {n} cells, fleet-mean R-hat {m:.4f}, range {lo:.4f}-{hi:.4f}   (papers: 0.41, 0.34-0.48)")
    if new.exists():
        N = json.load(open(new))["cells"]; n2, m2, lo2, hi2 = summary(N)
        log(f"  rebuilt  results.json:     {n2} cells, fleet-mean R-hat {m2:.4f}, range {lo2:.4f}-{hi2:.4f}")
        diffs = []
        for k in R:
            if k in N and N[k].get("measurements", {}).get("anchored_shares"):
                diffs.append(abs(R[k]["measurements"]["anchored_shares"]["R"] - N[k]["measurements"]["anchored_shares"]["R"]))
        if diffs: log(f"  per-cell |delta R-hat| vs released: max {max(diffs):.4f}, mean {sum(diffs)/len(diffs):.4f} over {len(diffs)} cells")
    else:
        log("  results.json not rebuilt yet (run the results stage)")
    for f, what in [("data/paper/calibration/gauss_ii_fleet.json", "Gaussian redundancy (B §5.4: all 24 cells positive at d=8)"),
                    ("data/paper/calibration/multistep_R_fleet.json", "multi-step persistence (B §6.9)"),
                    ("data/paper/calibration/kv_interpolation_gemma_2b_4bit_abliterated_deterministic.json", "KV interpolation (B §3.1)"),
                    ("data/paper/calibration/drop_history_probe_gemma_2b_q4_T0.json", "drop-history probe (C §3.4)"),
                    ("data/paper/calibration/v5/h4_threshold_derivation.json", "tau_H4 = 0.72 (B §5.7)")]:
        log(f"  {'present' if (ROOT / f).exists() else 'MISSING'}: {f}  [{what}]")
    return True


STAGE_FN = {"env": stage_env, "collect": stage_collect, "analysis": stage_analysis, "calibration": stage_calibration,
            "behavioral": stage_behavioral, "results": stage_results, "probes": stage_probes, "verify": stage_verify}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None); ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--skip-collection", action="store_true")
    a = ap.parse_args()
    models = a.models or ORDER
    for m in models:
        if m not in MODELS: sys.exit(f"unknown model tag {m}; choose from {list(MODELS)}")
    stages = STAGES if not a.only else [s.strip() for s in a.only.split(",")]
    if a.start:
        stages = STAGES[STAGES.index(a.start):]
    if a.skip_collection:
        stages = [s for s in stages if s != "collect"]
    if a.plan:
        print("stages:", " -> ".join(stages)); print("models:", models)
        print("collection per model:", "T=0:", TEMP_INDEP + T0_ONLY, "| per temperature:", PER_TEMP + [16], "| temps:", TEMPS)
        return 0
    status = load_status()
    log(f"=== reproduce: stages {stages}, models {models} ===")
    for s in stages:
        log(f"--- stage {s} ---")
        if not STAGE_FN[s](models, status):
            log(f"=== STOPPED in stage {s}; fix and rerun the same command to resume ===")
            return 1
    log("=== all stages complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
