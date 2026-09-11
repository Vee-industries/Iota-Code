"""
_dedup_rerun.py -- the de-duplicated re-run of the apparatus (paper B §10.2 follow-up, executed 2026-09-10).

Environment for every step: IOTA_DEDUP_ROWS=1 (one row per distinct (S_t, E_t, S_t+1, C_t); separate
'_dedup' qcache family), IOTA_SOURCE_RUNS_3WAY=6,7,8,13,14,15,1,3,23 (the nine-run apparatus row set the
released Q0057 / Run 0058 layer was computed on; the r1-23 cache family), IOTA_PREREQ_GATE=off (the
per-cell scanner keys Run 0016 on its bookkeeping CSV, which holds 2 of 13 source runs in every fleet
cell since 2026-05-17; the E_t files Run 0042 actually reads are present in every cell, checked 2026-09-10).

Steps (resumable; finished keys in data/paper/dedup_rerun_status.json):
  cells    per fleet cell: move aside Q0042/Q0043/Q0044/Q0045/Q0046 (they are resume caches) into
           data/_snapshot_pre_dedup_2026-09-10/, run 42, 43, 44, 45, 46 one subprocess each, then rebuild
           the nine-run apparatus cache with _rebuild_apparatus_cache.py (Run 0043's last load leaves the
           held-out cache as the survivor otherwise).
  masters  export_stats.build_all_masters() -> data/paper/results.json from the new per-cell files
           (Run 0057 pulls its Ridge/MLP partitions from results.json; Run 0059 rebuilds it again at the end).
  q56      Run 0056 (methodology calibration: channel marginal + manifest)
  q57      Run 0057 (function-class sensitivity; existing Q0057 moved aside first -- it is a resume cache)
  q58      Run 0058 (apparatus, 11 phases; both kraskov anchors.json moved aside first -- resume caches;
           IOTA_N_BOOTSTRAP=12 as in the released phase-10 file)
  q59      Run 0059 (results.json + figures)

Usage: py _dedup_rerun.py [--plan] [--from STEP] [--only STEP]     STEP in cells,masters,q56,q57,q58,q59
"""
import os, sys, json, time, argparse, subprocess, shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import reproduce as R

STATUS = ROOT / "data" / "paper" / "dedup_rerun_status.json"
SNAP = ROOT / "data" / "_snapshot_pre_dedup_2026-09-10"
LOG = ROOT / ".iota_dedup_rerun.log"
APPARATUS_RUNS = "6,7,8,13,14,15,1,3,23"
ENV = dict(R.ENV, IOTA_DEDUP_ROWS="1", IOTA_SOURCE_RUNS_3WAY=APPARATUS_RUNS, IOTA_PREREQ_GATE="off", IOTA_FRESH="1")
STEPS = ["cells", "cells2", "masters", "q56", "q57", "q58", "q59"]
CELL_RUNS = [(42, "Q0042_decomposition.json"), (43, "Q0043_sobol_partition.json"), (44, "Q0044_per_condition_R.json"),
             (45, "Q0045_fixed_dim_per_condition_R.json"), (46, "Q0046_mlp_decomposition.json")]
# cells2 (2026-09-10 evening): after the Ridge alpha policy change (ridge_policy.py) the Ridge-based per-cell runs are
# redone; Run 0042 (OLS) is untouched. The 'cells' step is left in place so a --plan shows the full history.
CELL_RUNS2 = CELL_RUNS[1:]


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")


def status():
    try: return json.load(open(STATUS))
    except Exception: return {"done": {}}


def mark(st, key):
    st["done"][key] = datetime.now().isoformat(timespec="seconds"); json.dump(st, open(STATUS, "w"), indent=1)


def sh(cmd, label, env=None, capture=False):
    t0 = time.time(); log(f"  {label}")
    if capture:
        p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env or ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
        for line in (p.stdout or "").splitlines():
            if line.startswith("CACHE ") or line.startswith("ERROR"): log(f"    {line}")
        if p.returncode != 0: log("    stderr tail: " + "\n".join((p.stderr or "").splitlines()[-8:]))
        rc = p.returncode
    else:
        rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env or ENV).returncode
    log(f"    -> rc={rc} ({time.time() - t0:.0f}s)"); return rc == 0


def cell_dir(cfg, temp):
    return ROOT / "data" / cfg["model_family"] / cfg["size_dir"] / cfg["model_variant"] / ("deterministic" if temp == 0.0 else f"temp_{temp}")


def move_aside(path, tag):
    """Copy a resume-cache file into the snapshot (first time only) and remove it from the tree."""
    if not path.exists(): return
    SNAP.mkdir(parents=True, exist_ok=True)
    dst = SNAP / f"{path.stem}__{tag}{path.suffix}"
    if not dst.exists(): shutil.copy(path, dst)
    path.unlink(); log(f"    moved aside {path.relative_to(ROOT)}")


def step_cells(st):
    for tag in R.ORDER:
        cfg = R.MODELS[tag]
        for temp in R.TEMPS:
            cd = cell_dir(cfg, temp); ctag = f"{cfg['model_family']}_{cfg['size_dir']}_{cfg['model_variant']}_{cd.name}"
            key = f"cells/{tag}/T{temp}"
            if key in st["done"]: continue
            log(f"--- cell {tag} T={temp} ---")
            for run, fname in CELL_RUNS:
                if f"{key}/run{run}" in st["done"]: continue      # this re-run's own output: keep for resume
                move_aside(cd / "analysis" / fname, ctag)
            R.set_session(cfg, temp, "")
            for run, fname in CELL_RUNS:
                if f"{key}/run{run}" in st["done"]: continue
                R.set_session(cfg, temp, str(run))
                if not sh([R.PY, "start_here.py", "--single-run", str(run)], f"Run {run:04d} dedup: {tag} T={temp}"): return False
                out = cd / "analysis" / fname
                if not out.exists(): log(f"    ERROR: {fname} missing after Run {run:04d}"); return False
                if run == 42:
                    d = json.load(open(out, encoding="utf-8-sig"))
                    log(f"    Q0042 n_3way={d.get('n_3way')} n_2way={d.get('n_2way')} r2_D={d.get('r2_D_full_decomposition')}")
                mark(st, f"{key}/run{run}")
            if not sh([R.PY, "_rebuild_apparatus_cache.py", str(cd.relative_to(ROOT))], f"rebuild nine-run apparatus cache: {tag} T={temp}", capture=True): return False
            mark(st, key)
    return True


def step_cells2(st):
    for tag in R.ORDER:
        cfg = R.MODELS[tag]
        for temp in R.TEMPS:
            cd = cell_dir(cfg, temp); ctag = f"{cfg['model_family']}_{cfg['size_dir']}_{cfg['model_variant']}_{cd.name}"
            key = f"cells2/{tag}/T{temp}"
            if key in st["done"]: continue
            log(f"--- cells2 {tag} T={temp} ---")
            for run, fname in CELL_RUNS2:
                if f"{key}/run{run}" in st["done"]: continue
                move_aside(cd / "analysis" / fname, ctag + "__alpha0.01_dedup")
            for run, fname in CELL_RUNS2:
                if f"{key}/run{run}" in st["done"]: continue
                R.set_session(cfg, temp, str(run))
                if not sh([R.PY, "start_here.py", "--single-run", str(run)], f"Run {run:04d} dedup+ridgeCV: {tag} T={temp}"): return False
                out = cd / "analysis" / fname
                if not out.exists(): log(f"    ERROR: {fname} missing after Run {run:04d}"); return False
                if run == 43:
                    d = json.load(open(out, encoding="utf-8-sig"))
                    log(f"    Q0043 n_3way={d.get('n_3way')} alpha={d.get('ridge_alpha')} E/C/R={d.get('perm_sens_E', {}).get('fraction')}/{d.get('perm_sens_C', {}).get('fraction')}/{d.get('perm_sens_R', {}).get('fraction')}")
                mark(st, f"{key}/run{run}")
            mark(st, key)
    return True


def step_masters(st):
    if "masters" in st["done"]: return True
    R.set_session(R.MODELS[R.ORDER[0]], 0.0, "")
    ok = sh([R.PY, "-c", "import export_stats; export_stats.build_all_masters()"], "build_all_masters -> data/paper/results.json")
    if ok: mark(st, "masters")
    return ok


def step_simple(st, key, cmd, env_extra=None, label=None, aside=()):
    if key in st["done"]: return True
    for p in aside: move_aside(ROOT / p, "pre_dedup")
    R.set_session(R.MODELS[R.ORDER[0]], 0.0, "")
    env = dict(ENV, **env_extra) if env_extra else ENV
    ok = sh(cmd, label or key, env=env)
    if ok: mark(st, key)
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", action="store_true"); ap.add_argument("--from", dest="start"); ap.add_argument("--only")
    a = ap.parse_args()
    steps = STEPS if not a.only else [a.only]
    if a.start: steps = STEPS[STEPS.index(a.start):]
    if a.plan: print("steps:", steps); return 0
    st = status(); log(f"=== dedup re-run: {steps} ===")
    fns = {
        "cells": lambda: step_cells(st),
        "cells2": lambda: step_cells2(st),
        "masters": lambda: step_masters(st),
        "q56": lambda: step_simple(st, "q56", [R.PY, "start_here.py", "--single-run", "56"], label="Run 0056 dedup (methodology calibration)"),
        "q57": lambda: step_simple(st, "q57", [R.PY, "start_here.py", "--single-run", "57"], label="Run 0057 dedup (function-class sensitivity)",
                                   aside=("data/paper/Q0057_function_class_sensitivity.json",)),
        "q58": lambda: step_simple(st, "q58", [R.PY, "run_bayesian_apparatus.py"], env_extra={"IOTA_N_BOOTSTRAP": "12"}, label="Run 0058 dedup (apparatus, 11 phases)",
                                   aside=("data/paper/calibration/kraskov_anchor/anchors.json", "data/paper/calibration/kraskov_anchor_p2/anchors.json",
                                          "data/paper/calibration/function_class_p2/per_cell.json", "data/paper/calibration/bootstrap_variance_p2/per_cell.json",
                                          "data/paper/calibration/estimator_joint_R2_p2/per_cell.json")),
        "q59": lambda: step_simple(st, "q59", [R.PY, "start_here.py", "--single-run", "59"], label="Run 0059 dedup (results.json + figures)"),
    }
    for s in steps:
        log(f"--- {s} ---")
        if not fns[s](): log(f"=== STOPPED at {s} ==="); return 1
    log("=== dedup re-run complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
