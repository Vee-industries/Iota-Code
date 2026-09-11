"""
_compare_dedup_results.py -- compare the de-duplicated re-run (data/paper/results.json) against the
released results8th.json cell by cell, and print every number paper B and paper C quote from the
results file so the paper edits can be made from one table.

Usage: py _compare_dedup_results.py [--new data/paper/results.json] [--old data/paper/results8th.json]
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--new", default="data/paper/results.json"); ap.add_argument("--old", default="data/paper/results8th.json")
a = ap.parse_args()
N = json.load(open(a.new, encoding="utf-8-sig"))["cells"]; O = json.load(open(a.old, encoding="utf-8-sig"))["cells"]
keys = [k for k in O if k in N]
print(f"cells: old {len(O)}, new {len(N)}, common {len(keys)}")

def m(c, path):
    cur = c.get("measurements", {})
    for p in path.split("."):
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None: return None
    return cur

rows = []
for k in keys:
    o, n = O[k], N[k]
    rows.append((k, m(o, "anchored_shares.R"), m(n, "anchored_shares.R"), m(o, "anchored_shares.E"), m(n, "anchored_shares.E"), m(o, "anchored_shares.C"), m(n, "anchored_shares.C"),
                 m(o, "estimator_joint_R2.ridge.heldout"), m(n, "estimator_joint_R2.ridge.heldout"), m(o, "estimator_joint_R2.mlp.heldout"), m(n, "estimator_joint_R2.mlp.heldout"),
                 m(o, "ii_fraction_knn"), m(n, "ii_fraction_knn"), m(n, "n_knn_samples")))
print("\ncell | R old->new | E old->new | C old->new | ridge R2 old->new | mlp R2 old->new | H3 old->new | ii_knn old->new")
def f(x): return "  n/a " if x is None else f"{x:6.3f}"
for r in rows:
    k, Ro, Rn, Eo, En, Co, Cn, ro, rn, mo, mn, io_, in_, nk = r
    h3o = abs(ro - mo) if ro is not None and mo is not None else None; h3n = abs(rn - mn) if rn is not None and mn is not None else None
    print(f"{k:34s} {f(Ro)}->{f(Rn)} | {f(Eo)}->{f(En)} | {f(Co)}->{f(Cn)} | {f(ro)}->{f(rn)} | {f(mo)}->{f(mn)} | {f(h3o)}->{f(h3n)} | {f(io_)}->{f(in_)}")

def summ(vals):
    v = [x for x in vals if x is not None]; return (len(v), float(np.mean(v)), float(np.min(v)), float(np.max(v))) if v else (0, np.nan, np.nan, np.nan)
for lab, idx in (("R", (1, 2)), ("E", (3, 4)), ("C", (5, 6))):
    so, sn = summ([r[idx[0]] for r in rows]), summ([r[idx[1]] for r in rows])
    print(f"\nfleet {lab}: old n={so[0]} mean {so[1]:.4f} range {so[2]:.4f}-{so[3]:.4f} | new n={sn[0]} mean {sn[1]:.4f} range {sn[2]:.4f}-{sn[3]:.4f}")
print("\nper-configuration mean R (old -> new):")
for cfg in ("gemma_2b_4bit", "gemma_2b_fp16", "gemma_9b_4bit", "llama_8b_4bit"):
    ks = [r for r in rows if r[0].startswith(cfg)]
    print(f"  {cfg:14s} {np.mean([r[1] for r in ks if r[1] is not None]):.4f} -> {np.mean([r[2] for r in ks if r[2] is not None]):.4f}")
tau = 0.0154
def h3(r, i, j): return abs(r[i] - r[j]) if r[i] is not None and r[j] is not None else None
out_old = sum(1 for r in rows if (h3(r, 7, 9) or 0) > tau); out_new = sum(1 for r in rows if (h3(r, 8, 10) or 0) > tau)
print(f"\nH3 OUT at tau={tau}: old {out_old}/24, new {out_new}/24")
print("MLP > Ridge held-out: old", sum(1 for r in rows if r[9] is not None and r[7] is not None and r[9] > r[7]), "/24, new", sum(1 for r in rows if r[10] is not None and r[8] is not None and r[10] > r[8]), "/24")
d = [abs(r[1] - r[2]) for r in rows if r[1] is not None and r[2] is not None]
if d: print(f"|delta R| per cell: max {max(d):.4f}, mean {np.mean(d):.4f}")
