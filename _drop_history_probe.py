"""
_drop_history_probe.py -- "context existed at turn x-1 but is not in the input at turn x."

Cohorts 1 and 2 of Run 0039 (Gemma 2B Q4, deterministic) are run normally through turn 13
(their states are on disk and reproduce exactly; see _kv_interpolation_test.py). At turn 14
the model is fed ONLY [system, user_14]: no prior turns, no cache. The architecture predicts
the two cohorts' turn-14 states are byte-identical, because nothing from turn 13 is an input
to the turn-14 computation. Also reported: the same probe with the full history (the normal
protocol), where the cohorts differ, as the positive control.

Usage: py _drop_history_probe.py
"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.argv = [sys.argv[0]]
import numpy as np, pandas as pd, torch
import importlib.util
spec = importlib.util.spec_from_file_location("kv", os.path.join(ROOT, "_kv_interpolation_test.py"))
kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
import orchestration_core as oc
from orchestration_throughlines import NEUTRAL_SYSTEM_PROMPT as SYS

CELL = os.path.join(ROOT, "data", "gemma", "2b_4bit", "abliterated", "deterministic")
df = pd.read_csv(os.path.join(CELL, "csv", "R0039_jolt.csv"))
tA = sorted(df[df.shock_variant == 1].trial.unique())[0]
tB = sorted(df[df.shock_variant == 2].trial.unique())[0]
hf_token = json.load(open(os.path.join(ROOT, "last_session.json"))).get("hf_token")
model, tok = oc.load_model("IlyaGusev/gemma-2-2b-it-abliterated", quant="4bit", token=hf_token); model.eval()
dev = next(model.parameters()).device

def state(text):
    ids = tok(text, return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float().cpu().numpy(), int(ids.shape[1])

res = {"cell": "gemma/2b_4bit/abliterated/deterministic", "trials": [int(tA), int(tB)]}
full, drop = {}, {}
for name, t in (("A", tA), ("B", tB)):
    hist = kv.build_messages(df, t, 13, SYS)
    u14 = str(df[(df.trial == t) & (df.turn == 14)].iloc[0].prompt)
    full[name], nf = state(kv.fmt_for_turn(tok, hist, u14))
    drop[name], nd = state(kv.fmt_for_turn(tok, [{"role": "system", "content": SYS}], u14))
    res[f"tokens_full_{name}"] = nf; res[f"tokens_drop_{name}"] = nd
cosd = kv.cosd
res["full_history_cosd_A_vs_B"] = cosd(full["A"], full["B"])
res["dropped_history_cosd_A_vs_B"] = cosd(drop["A"], drop["B"])
res["dropped_history_bytes_identical"] = bool(np.array_equal(drop["A"], drop["B"]))
res["dropped_vs_full_cosd_A"] = cosd(drop["A"], full["A"])
res["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "data", "paper", "calibration", "drop_history_probe_gemma_2b_q4_T0.json"), "w"), indent=1)
