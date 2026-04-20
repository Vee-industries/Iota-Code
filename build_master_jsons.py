"""
build_master_jsons.py — per-model + all-models paper JSON aggregator

Purpose:
  The IOTA paper pipeline consumes a per-model master JSON
  (master_results.json, one per model, aggregated by
  export_stats.build_master_results) and ships it to DATA/paper/json/
  via Run 55. That flow was missing Run 56 (MLP decomposition) as of
  v0.78.1.0, and there was no single cross-architecture aggregate for
  the writer bot to consume.

  This script fixes both gaps in one idempotent command:

    1) Rebuilds master_results.json for EVERY discovered model, pulling
       in Q56 alongside the existing Q25/Q27/Q32/Q33/Q34/Q45/Q46/Q49
       per-temperature analysis files. Safe to re-run — matches
       build_master_results() byte-for-byte for any pre-existing field.

    2) Copies each rebuilt master into DATA/paper/json/ using the
       naming convention Run 55 already uses:
         {family}_{size_dir}_master_results.json

    3) Builds a NEW all-models master at:
         DATA/paper/json/all_models_master.json

       Structure:
         {
           "generated": ISO timestamp,
           "n_models":  integer,
           "models":    { "{family}_{size}": <full per-model master>, ... },
           "cross_architecture": {
              "R_pooled_by_model":     { model_key: R_mean_across_temps },
              "ridge_vs_mlp":          per-model Q56 summary for
                                       Conjecture-1 (Ridge ≈ MLP) check,
              "R_per_temperature":     { model_key: { temp: R }, ... },
              "decomposition_summary": one-line E+C+R string per model,
           }
         }

Usage:
  python build_master_jsons.py                    # rebuild everything
  python build_master_jsons.py --per-model-only   # skip all_models_master
  python build_master_jsons.py --all-only         # skip per-model rebuild
  python build_master_jsons.py --verify           # read-only; report gaps

Design notes:
  - Depends on export_stats.build_master_results() — single source of
    truth for per-model structure. This script is a thin driver over it.
  - Writes DATA/paper/json/ alongside R55's outputs; coexists with R55
    without conflict (same naming scheme).
  - Run 55 calls this script automatically as its last step, so running
    all-stats + Run 55 gives you a fresh all_models_master.json for free.
  - Can also be invoked manually at any time — useful when Run 56 lands
    for a new model mid-ArXiv push and you just want the aggregate
    refreshed without re-running the full paper assembly.

Author: Kevin Vaillancourt
Version: 0.78.2.0
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys


# ── Locate iota root ─────────────────────────────────────────────────────────
def _find_root():
    """Walk up from this file's directory to the iota root (contains
    start_here.py). Lets the script run from any working directory."""
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(candidate, "start_here.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.dirname(os.path.abspath(__file__))


ROOT = _find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Model discovery ──────────────────────────────────────────────────────────
def _discover_models():
    """Walk DATA/ and yield (family, size_dir, variant, label, data_dir)
    for every model with a pooled/analysis/ directory present.

    Returns a list of dicts ready for per-model master rebuild. Skips
    'paper' and any dot-prefixed directories at the DATA root so we
    don't treat paper/ or .backup dirs as models."""
    from cartography import DATA, get_pooled_paths, logical_size, quant_from_dir
    from vault import IOTA_SUBFAMILY_MAP

    # Family-size labels from IOTA_SUBFAMILY_MAP
    _FAM_NAMES = {'llama': 'LLaMA', 'gemma': 'Gemma', 'qwen': 'Qwen',
                  'mistral': 'Mistral', 'phi': 'Phi', 'deepseek': 'DeepSeek'}
    _labels = {}
    for hf_path, info in IOTA_SUBFAMILY_MAP.items():
        sf = info.get('subfamily', '')
        if sf:
            parts = sf.split('-')
            fam_key = parts[0] if parts else ''
            fam_display = _FAM_NAMES.get(fam_key, fam_key.title())
            rest = ' '.join(p.upper() if p[-1:] == 'b' and p[:-1].replace('.','').isdigit() else p
                            for p in parts[1:])
            _labels[(fam_key, parts[-1] if parts else '')] = fam_display + ' ' + rest

    models = []
    if not os.path.isdir(DATA):
        return models
    for family in sorted(os.listdir(DATA)):
        if family.startswith('.') or family == 'paper':
            continue
        fam_dir = os.path.join(DATA, family)
        if not os.path.isdir(fam_dir):
            continue
        for size_dir in sorted(os.listdir(fam_dir)):
            sz_dir = os.path.join(fam_dir, size_dir)
            if not os.path.isdir(sz_dir):
                continue
            for variant in sorted(os.listdir(sz_dir)):
                var_dir = os.path.join(sz_dir, variant)
                if not os.path.isdir(var_dir):
                    continue
                pooled_ana = os.path.join(var_dir, 'pooled', 'analysis')
                if not os.path.isdir(pooled_ana):
                    continue
                lsize = logical_size(size_dir)
                quant = quant_from_dir(size_dir) or ''
                label = _labels.get((family, lsize),
                                    f"{_FAM_NAMES.get(family, family.title())} {lsize.upper()}")
                models.append({
                    'family':     family,
                    'size_dir':   size_dir,    # e.g. '2b_fp16' — disk form
                    'logical_size': lsize,     # e.g. '2b' — logical form
                    'variant':    variant,
                    'quant':      quant,
                    'label':      label,
                    'data_dir':   var_dir,
                    'pooled_ana': pooled_ana,
                    'key':        f"{family}_{size_dir}",  # for paper/json filenames
                })
    return models


# ── Per-model rebuild ────────────────────────────────────────────────────────
def _rebuild_per_model(model):
    """Call export_stats.build_master_results() for one model and return
    the absolute path to the written master_results.json, or None if the
    rebuild failed (missing prereqs, unreadable JSONs, etc.)."""
    import export_stats
    session = {
        'model_family':  model['family'],
        'model_size':    model['logical_size'],
        'model_variant': model['variant'],
        'model_name':    model['label'],
        'quantization':  model['quant'] or '4bit',
    }
    try:
        export_stats.build_master_results(session)
    except Exception as e:
        print(f"    [!] build_master_results failed for {model['key']}: {e}")
        return None
    out_path = os.path.join(model['pooled_ana'], 'master_results.json')
    return out_path if os.path.exists(out_path) else None


def _copy_to_paper(master_path, model, paper_json_dir):
    """Copy a per-model master_results.json into DATA/paper/json/
    using the naming convention Run 55 already uses:
    {family}_{size_dir}_master_results.json. Preserves mtime."""
    os.makedirs(paper_json_dir, exist_ok=True)
    dst = os.path.join(paper_json_dir, f"{model['key']}_master_results.json")
    shutil.copy2(master_path, dst)
    return dst


# ── All-models aggregate ─────────────────────────────────────────────────────
def _load_json_safe(path):
    """Read a JSON file with NaN/Infinity sanitisation. Returns None on
    any I/O or parse error so the aggregator degrades gracefully when a
    model's master is mid-rebuild or corrupted."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        raw = re.sub(r'\bNaN\b', 'null', raw)
        raw = re.sub(r'\bInfinity\b', 'null', raw)
        raw = re.sub(r'\b-Infinity\b', 'null', raw)
        return json.loads(raw)
    except Exception:
        return None


def _cross_arch_summary(all_models_dict):
    """Derive a small cross-architecture section from the per-model
    master dicts. Convenience for the writer bot — no new data, just
    common views pre-computed.

    Sections:
      R_pooled_by_model     — mean R across all temperatures per model
      R_per_temperature     — per-model × per-temp R (from Q33 if present,
                              else Q40 pooled)
      ridge_vs_mlp          — for each model: Ridge R (Q34) vs MLP R (Q56),
                              and ridge_gap — the core Conjecture 1 check
      decomposition_summary — one-line 'E={}, C={}, R={}' string per model
    """
    summary = {
        'R_pooled_by_model':     {},
        'R_per_temperature':     {},
        'ridge_vs_mlp':          {},
        'decomposition_summary': {},
    }

    for key, m in all_models_dict.items():
        per_t = m.get('per_temperature', {}) or {}
        pooled = m.get('pooled', {}) or {}

        # R per temperature from Q33 (or Q40 pooled fallback).
        r_by_temp = {}
        for t_str, t_data in per_t.items():
            q33 = t_data.get('Q33') or {}
            r = q33.get('R_fraction') or q33.get('R') or q33.get('r_fraction')
            if r is not None:
                r_by_temp[t_str] = float(r)
        if r_by_temp:
            summary['R_per_temperature'][key] = r_by_temp
            summary['R_pooled_by_model'][key] = (
                sum(r_by_temp.values()) / len(r_by_temp)
            )

        # Ridge vs MLP — requires Q34 (ridge) and Q56 (MLP) in the same temp.
        # Pool across whatever temperatures we have both for.
        ridge_Rs = []
        mlp_Rs = []
        ridge_gaps = []
        for t_str, t_data in per_t.items():
            q34 = t_data.get('Q34') or {}
            q56 = t_data.get('Q56') or {}
            # Ridge R from Q34 Sobol partition: 'R_fraction' or nested.
            r_ridge = (q34.get('R_fraction')
                       or (q34.get('ridge') or {}).get('R_fraction')
                       or q34.get('r_fraction'))
            # MLP R from Q56 pooled: Q56.pooled.mlp_r_mean (see analysis.py).
            q56_pooled = (q56.get('pooled') or {})
            r_mlp = q56_pooled.get('mlp_r_mean') or q56_pooled.get('R_fraction')
            gap = q56_pooled.get('ridge_gap')
            if r_ridge is not None:
                ridge_Rs.append(float(r_ridge))
            if r_mlp is not None:
                mlp_Rs.append(float(r_mlp))
            if gap is not None:
                ridge_gaps.append(float(gap))

        if ridge_Rs or mlp_Rs:
            entry = {}
            if ridge_Rs:
                entry['ridge_R_mean']  = sum(ridge_Rs) / len(ridge_Rs)
                entry['ridge_n_temps'] = len(ridge_Rs)
            if mlp_Rs:
                entry['mlp_R_mean']    = sum(mlp_Rs) / len(mlp_Rs)
                entry['mlp_n_temps']   = len(mlp_Rs)
            if ridge_gaps:
                entry['ridge_gap_mean'] = sum(ridge_gaps) / len(ridge_gaps)
                entry['ridge_gap_max']  = max(ridge_gaps)
                # Conjecture 1 verdict: Ridge is adequate if max |gap| < 0.05.
                # The writer bot can pull this directly for S7 cross-arch text.
                entry['conjecture_1_verdict'] = (
                    'ridge_adequate' if entry['ridge_gap_max'] < 0.05
                    else 'mlp_materially_different'
                )
            summary['ridge_vs_mlp'][key] = entry

        # E+C+R one-liner from pooled Q40 if available, else first per-temp Q33.
        ecr_src = (pooled.get('Q40_decomposition') or {})
        if not ecr_src:
            for t_data in per_t.values():
                q33 = t_data.get('Q33') or {}
                if q33:
                    ecr_src = q33
                    break
        if ecr_src:
            E = ecr_src.get('E_fraction') or ecr_src.get('E')
            C = ecr_src.get('C_fraction') or ecr_src.get('C')
            R = ecr_src.get('R_fraction') or ecr_src.get('R')
            if E is not None and C is not None and R is not None:
                summary['decomposition_summary'][key] = (
                    f"E={float(E):.3f}, C={float(C):.3f}, R={float(R):.3f}"
                )

    return summary


def build_all(verbose=True, per_model_only=False, all_only=False):
    """Top-level entry: discover models, rebuild per-model masters, copy
    them to DATA/paper/json/, then build DATA/paper/json/all_models_master.json.

    Flags:
      per_model_only — rebuild per-model masters and copy to paper/json/,
                       but skip the all_models_master.json aggregate.
      all_only       — skip the per-model rebuild entirely; just re-read
                       whatever per-model masters already exist in
                       paper/json/ and regenerate the aggregate. Useful
                       when you've already run all-stats and just want
                       the cross-arch view refreshed.

    Returns the path to all_models_master.json (or None if all_only=False
    and per_model_only=True since no aggregate is written in that mode).
    """
    from cartography import DATA
    paper_json = os.path.join(DATA, 'paper', 'json')

    models = _discover_models()
    if verbose:
        print(f"\n  Discovered {len(models)} model(s) under {DATA}")
        for m in models:
            print(f"    - {m['key']}  ({m['label']})")

    # ── Per-model rebuild ────────────────────────────────────────────────
    if not all_only:
        if verbose:
            print(f"\n  Rebuilding per-model master_results.json ...")
        n_ok = 0
        for m in models:
            src = _rebuild_per_model(m)
            if src:
                dst = _copy_to_paper(src, m, paper_json)
                if verbose:
                    print(f"    [+] {m['key']}  →  {dst}")
                n_ok += 1
            else:
                if verbose:
                    print(f"    [x] {m['key']}  no master produced")
        if verbose:
            print(f"\n  {n_ok}/{len(models)} per-model masters rebuilt.")
    else:
        if verbose:
            print(f"\n  Skipping per-model rebuild (--all-only).")

    if per_model_only:
        if verbose:
            print(f"  --per-model-only set; skipping all_models_master.json.")
        return None

    # ── All-models aggregate ─────────────────────────────────────────────
    if verbose:
        print(f"\n  Building all_models_master.json ...")

    all_models_dict = {}
    for m in models:
        src = os.path.join(paper_json, f"{m['key']}_master_results.json")
        if not os.path.exists(src):
            # Fall back to pooled/analysis/ if paper copy absent (happens
            # when user runs this script without having run R55 first)
            src = os.path.join(m['pooled_ana'], 'master_results.json')
        data = _load_json_safe(src)
        if data:
            # Tag per-model entries with their disk location so the
            # writer bot can cross-reference back to source files.
            data['_source_path'] = src
            data['_key'] = m['key']
            all_models_dict[m['key']] = data

    aggregate = {
        'schema_version': '1.0',
        'generated': datetime.datetime.now().isoformat(),
        'n_models': len(all_models_dict),
        'model_keys': sorted(all_models_dict.keys()),
        'cross_architecture': _cross_arch_summary(all_models_dict),
        'models': all_models_dict,
    }

    os.makedirs(paper_json, exist_ok=True)
    out_path = os.path.join(paper_json, 'all_models_master.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(aggregate, f, indent=2, default=str)

    if verbose:
        print(f"\n  [+] {out_path}")
        ca = aggregate['cross_architecture']
        print(f"      R_pooled_by_model: {len(ca['R_pooled_by_model'])} model(s)")
        print(f"      ridge_vs_mlp:      {len(ca['ridge_vs_mlp'])} model(s)")
        if ca['decomposition_summary']:
            print(f"\n      Decomposition summary:")
            for key, s in ca['decomposition_summary'].items():
                print(f"        {key:24s}  {s}")
        if ca['ridge_vs_mlp']:
            print(f"\n      Conjecture 1 verdicts:")
            for key, d in ca['ridge_vs_mlp'].items():
                v = d.get('conjecture_1_verdict', 'insufficient_data')
                gap = d.get('ridge_gap_max')
                gap_s = f"(max |gap| = {gap:.3f})" if gap is not None else ""
                print(f"        {key:24s}  {v}  {gap_s}")
    return out_path


def _verify():
    """Read-only: scan DATA/ for every model and report:
      - whether master_results.json exists in pooled/analysis/
      - whether it's copied to paper/json/
      - whether Q56 is present in that master
      - whether all_models_master.json exists and is fresher than
        every contributing per-model master
    Zero writes. Use to diagnose missing/stale state before triggering
    a rebuild."""
    from cartography import DATA
    paper_json = os.path.join(DATA, 'paper', 'json')
    models = _discover_models()
    print(f"\n  Verifying {len(models)} model(s):\n")
    agg_path = os.path.join(paper_json, 'all_models_master.json')
    agg_mtime = os.path.getmtime(agg_path) if os.path.exists(agg_path) else 0
    stale = []
    for m in models:
        src = os.path.join(m['pooled_ana'], 'master_results.json')
        paper = os.path.join(paper_json, f"{m['key']}_master_results.json")
        has_src = os.path.exists(src)
        has_paper = os.path.exists(paper)
        has_q56 = False
        if has_src:
            data = _load_json_safe(src)
            if data:
                for t_data in (data.get('per_temperature') or {}).values():
                    if 'Q56' in t_data:
                        has_q56 = True
                        break
        mtime = os.path.getmtime(src) if has_src else 0
        fresh = mtime <= agg_mtime and agg_mtime > 0
        if not fresh and has_src:
            stale.append(m['key'])
        flags = (
            ("S" if has_src   else "-") +
            ("P" if has_paper else "-") +
            ("6" if has_q56   else "-") +
            ("F" if fresh     else "-")
        )
        print(f"    [{flags}]  {m['key']:28s}  {m['label']}")
    print(f"\n  Legend: S=pooled master present  P=paper copy  6=has Q56  F=fresh vs aggregate")
    print(f"  all_models_master.json: {'present' if agg_mtime else 'MISSING'}")
    if stale:
        print(f"  Stale (would refresh on next build_all): {stale}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    """Argparse entry point. Default rebuilds everything end-to-end.
    --per-model-only skips the aggregate; --all-only skips the per-model
    rebuild; --verify is read-only diagnostics."""
    ap = argparse.ArgumentParser(
        description="Rebuild per-model master_results.json and "
                    "all_models_master.json for the paper pipeline."
    )
    ap.add_argument('--per-model-only', action='store_true',
                    help='Rebuild per-model masters only; skip aggregate.')
    ap.add_argument('--all-only',       action='store_true',
                    help='Skip per-model rebuild; regenerate aggregate only.')
    ap.add_argument('--verify',         action='store_true',
                    help='Read-only status report.')
    ap.add_argument('-q', '--quiet',    action='store_true',
                    help='Suppress progress output.')
    args = ap.parse_args()

    if args.verify:
        _verify()
        return

    build_all(verbose=not args.quiet,
              per_model_only=args.per_model_only,
              all_only=args.all_only)


if __name__ == '__main__':
    main()
