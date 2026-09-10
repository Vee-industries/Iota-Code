"""
_kv_interpolation_test.py -- state-vs-string test by KV-cache interpolation (designed 2026-09-10).

Question: at turn 14 of the Run 0039 protocol, does the model read its prior state as a
representation (a latent object) or does the next turn only ever see the transcript string?
A plain KV swap cannot answer this (the cache is a deterministic function of the tokens, so
"B's cache + A's text" is just a spliced transcript). What can: a cached state that NO
transcript produces. Take two cohorts whose turn-13 contexts tokenize to the same length,
interpolate their KV caches with alpha in [0, 1], feed the byte-identical turn-14 recovery
prompt on top, and read the final-layer state at the last input token (the same quantity the
collector saves). If that state moves continuously with alpha, the next turn reads the
representation; if it snaps to one endpoint, the string wins.

Also run: alpha in {0, 1} against the saved Q0039 turn-14 states (reconstruction check), and
a same-cohort interpolation (two trials of one cohort at T = 0 are identical, so it must read 0).

Defaults: Gemma 2B Q4, deterministic cell, cohorts 1 and 2 (identical prompt-token counts at
every post-shock turn: 585 / 606 / 631 / 657). Output: data/paper/calibration/kv_interpolation_<tag>.json

Usage: py _kv_interpolation_test.py [--cell gemma/2b_4bit/abliterated/deterministic] [--cohorts 1 2] [--alphas 0 0.25 0.5 0.75 1]
"""
import os, sys, json, argparse, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch

import orchestration_core as oc
from cartography import sanitize


def build_messages(df, trial, upto_turn, sys_prompt):
    """Reconstruct the threaded messages exactly as _standard_trial_loop built them,
    from the CSV's stored (padded) prompts and outputs: system, throughline (turn 0), then turns."""
    rows = df[df.trial == trial].sort_values("turn")
    msgs = [{"role": "system", "content": sys_prompt}] if sys_prompt else []
    for _, r in rows.iterrows():
        if r.turn > upto_turn:
            break
        msgs.append({"role": "user", "content": str(r.prompt)})
        msgs.append({"role": "assistant", "content": str(r.output)})
    return msgs


def fmt_for_turn(tok, msgs_through_prev, next_user_prompt):
    """Formatted text the collector fed at a turn: history + new user prompt + generation prompt."""
    msgs = list(msgs_through_prev) + [{"role": "user", "content": next_user_prompt}]
    if oc._NO_SYSTEM_ROLE:
        msgs = oc._strip_system_role(msgs)
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        if 'System role not supported' in str(e):
            oc._NO_SYSTEM_ROLE = True
            return tok.apply_chat_template(oc._strip_system_role(msgs), tokenize=False, add_generation_prompt=True)
        raise


def fmt_prefix(tok, msgs):
    """Formatted text of a complete history (no generation prompt)."""
    m = oc._strip_system_role(msgs) if oc._NO_SYSTEM_ROLE else msgs
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        if 'System role not supported' in str(e):
            oc._NO_SYSTEM_ROLE = True
            return tok.apply_chat_template(oc._strip_system_role(msgs), tokenize=False, add_generation_prompt=False)
        raise


def last_state(model, ids, cache=None):
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float().cpu().numpy(), out.past_key_values, out.logits[0, -1, :].float().cpu()


def cache_tensors(cache):
    """List of (K, V) per layer from a transformers cache object (v5 DynamicCache layers / legacy attrs)."""
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache)


def make_cache(kv_list, model):
    from transformers import DynamicCache
    try:
        return DynamicCache(ddp_cache_data=[(k, v) for k, v in kv_list], config=model.config)
    except TypeError:
        c = DynamicCache()
        for i, (k, v) in enumerate(kv_list):
            c.update(k, v, i)
        return c


def cosd(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="gemma/2b_4bit/abliterated/deterministic")
    ap.add_argument("--model-path", default="IlyaGusev/gemma-2-2b-it-abliterated")
    ap.add_argument("--model-name", default="gemma-2-2b-it-abliterated")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--cohorts", nargs=2, type=int, default=[1, 2])
    ap.add_argument("--trial-in-cohort", type=int, default=0, help="which trial of each cohort (0 = first)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--shock-turn", type=int, default=13)
    a = ap.parse_args()

    cell = os.path.join(ROOT, "data", *a.cell.split("/"))
    df = pd.read_csv(os.path.join(cell, "csv", "R0039_jolt.csv"))
    hid = os.path.join(cell, "hidden_states")
    mn = sanitize(a.model_name)
    from orchestration_throughlines import NEUTRAL_SYSTEM_PROMPT as sys_prompt   # the default sys_prompt of _standard_trial_loop

    def trials_of(cohort):
        return sorted(df[(df.shock_variant == cohort)].trial.unique())
    tA = trials_of(a.cohorts[0])[a.trial_in_cohort]
    tB = trials_of(a.cohorts[1])[a.trial_in_cohort]
    tA2 = trials_of(a.cohorts[0])[a.trial_in_cohort + 1]     # same-cohort control
    print(f"cohort {a.cohorts[0]} trial {tA} (control trial {tA2}); cohort {a.cohorts[1]} trial {tB}", flush=True)

    hf_token = None
    try:
        hf_token = json.load(open(os.path.join(ROOT, "last_session.json"))).get("hf_token")
    except Exception:
        pass
    model, tok = oc.load_model(a.model_path, quant=a.quant, token=hf_token)
    model.eval()
    dev = next(model.parameters()).device

    def context_ids(trial):
        """Token ids of the full history through the shock turn's assistant response, and the
        turn-14 continuation ids (history + turn-14 user prompt + generation prompt, minus the prefix)."""
        hist = build_messages(df, trial, a.shock_turn, sys_prompt)
        nxt = df[(df.trial == trial) & (df.turn == a.shock_turn + 1)].iloc[0].prompt
        full_txt = fmt_for_turn(tok, hist, str(nxt))
        pre_txt = fmt_prefix(tok, hist)
        full = tok(full_txt, return_tensors="pt")["input_ids"]
        pre = tok(pre_txt, return_tensors="pt")["input_ids"]
        n = pre.shape[1]
        assert torch.equal(full[0, :n], pre[0]), "prefix tokenization is not a prefix of the full tokenization"
        return pre.to(dev), full[:, n:].to(dev), full.to(dev)

    def saved_state(trial, turn):
        return np.load(os.path.join(hid, f"Q0039_{mn}_trial{trial:04d}_turn{turn:02d}.npy")).astype(np.float32)

    results = {"cell": a.cell, "cohorts": a.cohorts, "trials": [int(tA), int(tB)], "control_trial": int(tA2), "alphas": a.alphas}

    # 1. reconstruction check: full re-prefill of turn 14 must reproduce the saved turn-14 state
    preA, sufA, fullA = context_ids(tA)
    preB, sufB, fullB = context_ids(tB)
    assert preA.shape[1] == preB.shape[1], f"prefix lengths differ: {preA.shape[1]} vs {preB.shape[1]}"
    assert torch.equal(sufA, sufB), "turn-14 continuation tokens differ between cohorts (they must be byte-identical)"
    hA_full, _, _ = last_state(model, fullA)
    hB_full, _, _ = last_state(model, fullB)
    results["reconstruction"] = {
        "cosd_A_vs_saved14": cosd(hA_full, saved_state(tA, a.shock_turn + 1)),
        "cosd_B_vs_saved14": cosd(hB_full, saved_state(tB, a.shock_turn + 1)),
        "cosd_A_vs_B_full_prefill": cosd(hA_full, hB_full),
        "prefix_tokens": int(preA.shape[1]), "continuation_tokens": int(sufA.shape[1]),
    }
    print("reconstruction:", results["reconstruction"], flush=True)

    # 2. caches from the two prefixes; cache-on continuation must equal full prefill (sanity)
    _, cacheA, _ = last_state(model, preA)
    _, cacheB, _ = last_state(model, preB)
    kvA, kvB = cache_tensors(cacheA), cache_tensors(cacheB)
    hA_cache, _, _ = last_state(model, sufA, make_cache([(k.clone(), v.clone()) for k, v in kvA], model))
    results["cache_on_equals_prefill"] = {"cosd_A": cosd(hA_cache, hA_full)}
    print("cache-on vs prefill:", results["cache_on_equals_prefill"], flush=True)

    # 3. interpolation sweep
    sweep = []
    for al in a.alphas:
        kv = [(((1 - al) * kA.float() + al * kB.float()).to(kA.dtype), ((1 - al) * vA.float() + al * vB.float()).to(vA.dtype))
              for (kA, vA), (kB, vB) in zip(kvA, kvB)]
        h, _, logits = last_state(model, sufA, make_cache(kv, model))
        top = tok.decode(int(torch.argmax(logits)))
        sweep.append({"alpha": al, "cosd_to_A": cosd(h, hA_full), "cosd_to_B": cosd(h, hB_full),
                      "position": cosd(h, hA_full) / (cosd(h, hA_full) + cosd(h, hB_full) + 1e-12), "argmax_token": top})
        print(f"alpha={al:.2f}: d(A)={sweep[-1]['cosd_to_A']:.6f} d(B)={sweep[-1]['cosd_to_B']:.6f} pos={sweep[-1]['position']:.3f} top='{top}'", flush=True)
    results["sweep"] = sweep

    # 4. same-cohort control (trials tA, tA2 of cohort A): identical at T=0, so interpolation must read 0
    preA2, sufA2, fullA2 = context_ids(tA2)
    _, cacheA2, _ = last_state(model, preA2)
    kvA2 = cache_tensors(cacheA2)
    hA2_full, _, _ = last_state(model, fullA2)
    kv = [(((0.5) * kA.float() + 0.5 * kB.float()).to(kA.dtype), (0.5 * vA.float() + 0.5 * vB.float()).to(vA.dtype))
          for (kA, vA), (kB, vB) in zip(kvA, kvA2)]
    h_ctrl, _, _ = last_state(model, sufA, make_cache(kv, model))
    results["same_cohort_control"] = {"cosd_A_vs_A2_full": cosd(hA_full, hA2_full), "cosd_mid_vs_A": cosd(h_ctrl, hA_full)}
    print("same-cohort control:", results["same_cohort_control"], flush=True)

    out = os.path.join(ROOT, "data", "paper", "calibration", f"kv_interpolation_{a.cell.replace('/', '_')}.json")
    results["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(results, open(out, "w"), indent=1)
    print("Wrote", out)


if __name__ == "__main__":
    main()
