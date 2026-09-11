"""
_paper_numbers_pack.py -- every fleet quantity papers B and C quote, computed from one results file.

Usage: py _paper_numbers_pack.py [--results data/paper/results.json] [--q57 data/paper/Q0057_function_class_sensitivity.json]
                                 [--anchors data/paper/calibration/kraskov_anchor_p2/anchors.json]
                                 [--boot data/paper/calibration/bootstrap_variance_p2/per_cell.json] [--label new]
Run it on results8th.json (+ the snapshot Q0057) to check it reproduces the released numbers, then on the new files.
"""
import os, sys, json, argparse
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument('--results', default='data/paper/results.json'); ap.add_argument('--q57', default='data/paper/Q0057_function_class_sensitivity.json')
ap.add_argument('--anchors', default='data/paper/calibration/kraskov_anchor_p2/anchors.json'); ap.add_argument('--boot', default='data/paper/calibration/bootstrap_variance_p2/per_cell.json')
ap.add_argument('--label', default='new'); a = ap.parse_args()

R = json.load(open(a.results, encoding='utf-8-sig')); C = R['cells']
Q = json.load(open(a.q57, encoding='utf-8-sig'))['cells'] if os.path.exists(a.q57) else {}
FAM = [('llama_8b_4bit', 'LLaMA 8B Q4'), ('gemma_9b_4bit', 'Gemma 9B Q4'), ('gemma_2b_fp16', 'Gemma 2B FP16'), ('gemma_2b_4bit', 'Gemma 2B Q4')]
TEMPS = ['T0.0', 'T0.2', 'T0.4', 'T0.6', 'T0.8', 'T1.0']
def key(f, t): return f"{f}_abliterated_{t}"
def q57key(f, t): return f"{f.replace('_4bit', '_q4')}_abliterated_t{t[1:].replace('.', '')}"
def m(f, t): return C[key(f, t)]['measurements']
def f3(x): return 'n/a' if x is None else f"{x:.3f}"
def f4(x): return 'n/a' if x is None else f"{x:.4f}"
print(f"===== numbers pack [{a.label}] from {a.results} (generated {R.get('generated_at')}, iota {R.get('iota_version')}) =====\n")

# ── q* (anchored shares) ───────────────────────────────────────────────
print("## anchored q* per cell (E, C, R)")
allR = []
for f, lab in FAM:
    for t in TEMPS:
        s = m(f, t)['anchored_shares']; allR.append(s['R']); print(f"  {key(f,t):34s} {s['E']:.3f} {s['C']:.3f} {s['R']:.3f}")
allR = np.array(allR)
print(f"fleet R: mean {allR.mean():.4f}  min {allR.min():.3f}  max {allR.max():.3f}  range {allR.max()-allR.min():.3f}")
for f, lab in FAM:
    v = [m(f, t)['anchored_shares']['R'] for t in TEMPS]; print(f"  {lab:14s} mean {np.mean(v):.3f}  range [{min(v):.3f}, {max(v):.3f}]  by T: " + ' '.join(f"{x:.3f}" for x in v))
print("per-class R ranges (Ridge in-sample = rhat_ridge_insample, MLP = rhat_mlp_pooled_mean, RF/RKHS partition-B):")
for cls, getter in (('Ridge', lambda mm, q: mm.get('rhat_ridge_insample')), ('MLP', lambda mm, q: mm.get('rhat_mlp_pooled_mean')),
                    ('RF', lambda mm, q: (q.get('permutation_shares', {}).get('rf') or {}).get('R')), ('RKHS', lambda mm, q: (q.get('permutation_shares', {}).get('rkhs_median') or {}).get('R'))):
    vals = {}
    for f, lab in FAM:
        for t in TEMPS:
            v = getter(m(f, t), Q.get(q57key(f, t), {})); vals[key(f, t)] = v
    vv = [v for v in vals.values() if v is not None]
    print(f"  {cls:5s} span {min(vv):.3f} to {max(vv):.3f}")
    for f, lab in FAM:
        print(f"      {lab:14s} " + ' '.join(f3(vals[key(f, t)]) for t in TEMPS))

# ── H3 / H4 / gaps table ───────────────────────────────────────────────
print("\n## recoverability plane: H3 = |ridge - mlp| held-out, IN/OUT at tau 0.0103/0.0128/0.0154, H4 = ii_fraction_knn, gap_E, gap_S")
taus = (0.0103, 0.0128, 0.0154); outc = {tau: 0 for tau in taus}; fam_out = {}
for f, lab in FAM:
    fam_out[lab] = [0, 0, 0]
    for t in TEMPS:
        mm = m(f, t); e = mm['estimator_joint_R2']; h3 = abs(e['ridge']['heldout'] - e['mlp']['heldout'])
        io = ['OUT' if h3 > tau else 'IN' for tau in taus]
        for i, tau in enumerate(taus):
            if h3 > tau: outc[tau] += 1; fam_out[lab][i] += 1
        print(f"  {lab:14s} {t[1:]}  H3={h3:.4f}  {', '.join(io):14s}  H4={f4(mm.get('ii_fraction_knn'))}  gap_E={mm.get('gap_E'):+.4f}  gap_S={mm.get('gap_S'):+.4f}")
print("H3 OUT counts at taus:", {tau: f"{outc[tau]}/24" for tau in taus}, "| by family (0.0103/0.0128/0.0154):", fam_out)
h3s = {key(f, t): abs(m(f, t)['estimator_joint_R2']['ridge']['heldout'] - m(f, t)['estimator_joint_R2']['mlp']['heldout']) for f, _ in FAM for t in TEMPS}
print(f"H3 range {min(h3s.values()):.4f} to {max(h3s.values()):.4f} (max at {max(h3s, key=h3s.get)}); H4 range {min(m(f,t)['ii_fraction_knn'] for f,_ in FAM for t in TEMPS):.3f} to {max(m(f,t)['ii_fraction_knn'] for f,_ in FAM for t in TEMPS):.3f}")

# ── estimator held-out R2 ──────────────────────────────────────────────
print("\n## held-out joint R2 per estimator: fleet mean/median/min/max, pairwise counts, family means, T=0.0 -> T=1.0 trajectories")
E = {cls: {key(f, t): m(f, t)['estimator_joint_R2'][cls]['heldout'] for f, _ in FAM for t in TEMPS} for cls in ('ridge', 'mlp', 'rf', 'rkhs_median')}
for cls in E:
    v = np.array(list(E[cls].values())); print(f"  {cls:12s} mean {v.mean():.4f} median {np.median(v):.4f} min {v.min():.4f} max {v.max():.4f}")
for x, y in (('mlp', 'ridge'), ('mlp', 'rf'), ('mlp', 'rkhs_median'), ('rf', 'rkhs_median'), ('rf', 'ridge'), ('rkhs_median', 'ridge')):
    print(f"  {x} > {y}: {sum(1 for k in E[x] if E[x][k] > E[y][k])}/24")
print("  cells where ridge >= mlp:", [(k, round(E['ridge'][k], 4), round(E['mlp'][k], 4)) for k in E['ridge'] if E['ridge'][k] >= E['mlp'][k]])
print(f"  {'family':14s} ridge  mlp    rf     rkhs   mlp-ridge")
for f, lab in FAM:
    ks = [key(f, t) for t in TEMPS]
    print(f"  {lab:14s} " + ' '.join(f"{np.mean([E[c][k] for k in ks]):.3f}" for c in ('ridge', 'mlp', 'rf', 'rkhs_median')) + f"  {np.mean([E['mlp'][k]-E['ridge'][k] for k in ks]):+.4f}")
print("  trajectories (ridge | mlp) and delta T1.0-T0.0, least-squares slope of ridge vs T:")
Ts = np.array([0, .2, .4, .6, .8, 1.0])
for f, lab in FAM:
    r = [E['ridge'][key(f, t)] for t in TEMPS]; ml = [E['mlp'][key(f, t)] for t in TEMPS]
    print(f"  {lab:14s} ridge " + ' '.join(f"{x:.4f}" for x in r) + f"  d={r[-1]-r[0]:+.4f} slope={np.polyfit(Ts, r, 1)[0]:+.3f} | mlp " + ' '.join(f"{x:.4f}" for x in ml) + f"  d={ml[-1]-ml[0]:+.4f}")
ll = [E['ridge'][key('llama_8b_4bit', t)] for t in TEMPS]
for f, lab in FAM[1:]:
    g = [E['ridge'][key(f, t)] for t in TEMPS]; print(f"  Welch LLaMA ridge vs {lab}: p={stats.ttest_ind(ll, g, equal_var=False).pvalue:.4f}")
q4 = [key('gemma_2b_4bit', t) for t in TEMPS]; fp = [key('gemma_2b_fp16', t) for t in TEMPS]
print("  Q4 vs FP16 (Gemma 2B): " + ', '.join(f"{c}: {np.mean([E[c][k] for k in q4]):.4f} vs {np.mean([E[c][k] for k in fp]):.4f} (d={np.mean([E[c][k] for k in fp])-np.mean([E[c][k] for k in q4]):+.4f})" for c in ('ridge', 'mlp', 'rf', 'rkhs_median')))
# Fisher 4x2 on OUT at 0.0154 and pairwise LLaMA vs 2B Q4
tab = [[fam_out[lab][2], 6 - fam_out[lab][2]] for _, lab in FAM]
try:
    from scipy.stats import chi2_contingency
    print(f"  4x2 OUT-at-0.0154 table {tab}; chi2 p={chi2_contingency(tab)[1]:.4f}; Fisher LLaMA vs Gemma 2B Q4 p={stats.fisher_exact([tab[0], tab[3]])[1]:.4f}")
except Exception as ex: print('  fisher/chi2 failed', ex)

# ── partition-B substitution (§6.3, §6.7 iv) ───────────────────────────
print("\n## partition-B substitution diagnostics")
absdc = {c: [] for c in ('ridge', 'mlp', 'rkhs_median', 'rf')}; two = strict = 0; dom = []; further = 0; cosneg = 0; gshift = {}
for f, lab in FAM:
    for t in TEMPS:
        d = m(f, t).get('partition_b_substitution_diagnostics') or {}
        if not d: continue
        for c in absdc: absdc[c].append(d.get('abs_dC_by_class', {}).get(c))
        two += bool(d.get('two_tier_ordering_holds')); strict += bool(d.get('strict_ordering_holds'))
        if d.get('g_tilde_shift_dC_dominance') is not None: dom.append(d['g_tilde_shift_dC_dominance'])
        gshift[key(f, t)] = d.get('g_tilde_shift_norm')
        further += bool(d.get('kl_delta_anchor', 0) > 0) if 'kl_delta_anchor' in d else 0
        if d.get('anchor_alignment_cosine') is not None and d['anchor_alignment_cosine'] < 0: cosneg += 1
print("  fleet_mean_abs_dC:", {c: round(float(np.mean([x for x in v if x is not None])), 4) for c, v in absdc.items() if any(x is not None for x in v)})
print(f"  two_tier {two}/24  strict {strict}/24  dC_dominance mean {np.mean(dom) if dom else float('nan'):.3f}  moved_further(kl_delta>0) {further}/24  cosine_negative {cosneg}/24")
if gshift:
    groups = [[gshift[key(f, t)] for t in TEMPS] for f, _ in FAM]
    print("  g_tilde_shift_norm family means:", {lab: round(float(np.mean(g)), 3) for (f, lab), g in zip(FAM, groups)}, f"Kruskal H={stats.kruskal(*groups).statistic:.2f} p={stats.kruskal(*groups).pvalue:.4f}")
    d0 = m('llama_8b_4bit', 'T0.0').get('partition_b_substitution_diagnostics') or {}
    print("  sample diag keys:", list(d0.keys()))

# ── geometric diagnostics (§6.4) ───────────────────────────────────────
print("\n## hull membership (geometric_diagnostics)")
inside = outside = degen = 0; mech = {}; pulls = []; pulls_out = []
for f, lab in FAM:
    for t in TEMPS:
        g = m(f, t).get('geometric_diagnostics') or {}
        if not g: continue
        if g.get('hull_degenerate'): degen += 1
        if g.get('q_star_inside_class_hull'): inside += 1
        else:
            outside += 1; mech[g.get('hull_exit_mechanism')] = mech.get(g.get('hull_exit_mechanism'), 0) + 1; pulls_out.append((g.get('anchor_pull_distance'), key(f, t)))
        pulls.append(g.get('anchor_pull_distance'))
print(f"  inside {inside}  outside {outside}  degenerate {degen}  exit mechanisms {mech}  anchor_pull mean {np.mean(pulls):.3f} (outside-only {np.mean([p for p,_ in pulls_out]) if pulls_out else float('nan'):.3f}) max {max(pulls_out) if pulls_out else None}")

# ── anchors (§6.4 top-5 R_kNN) ─────────────────────────────────────────
if os.path.exists(a.anchors):
    A = json.load(open(a.anchors, encoding='utf-8-sig')); pc = A.get('cells') or A.get('anchors_per_cell') or {}
    top = sorted(((v['p_anchor'][2], k) for k, v in pc.items() if isinstance(v, dict) and v.get('p_anchor')), reverse=True)[:5]
    print("\n## top-5 R_kNN (kraskov_anchor_p2):", [(k, round(r, 4)) for r, k in top])

# ── Q0057-derived: per-class trajectories on LLaMA, sign checks (§6.5, §7.1) ──
if Q:
    print("\n## Q0057: four-class R shares by family and T; sign checks")
    for f, lab in FAM:
        for cls in ('ridge', 'mlp', 'rf', 'rkhs_median'):
            v = [(Q.get(q57key(f, t), {}).get('permutation_shares', {}).get(cls) or {}).get('R') for t in TEMPS]
            print(f"  {lab:14s} {cls:12s} " + ' '.join(f3(x) for x in v))
    both = mlp_only = rf_only = neither = 0; gap_holds = asym_holds = 0
    for f, lab in FAM:
        for t in TEMPS:
            c = Q.get(q57key(f, t), {}); sc = c.get('sign_check') or {}
            g = bool(sc.get('mlp_vs_ridge_signs_opposite')); r_ = bool(sc.get('rf_vs_ridge_signs_opposite'))
            gap_holds += g; asym_holds += r_
            both += g and r_; mlp_only += g and not r_; rf_only += r_ and not g; neither += (not g) and (not r_)
    print(f"  sign_check mlp_vs_ridge opposite: {gap_holds}/24; rf_vs_ridge opposite: {asym_holds}/24; both {both}, mlp-only {mlp_only}, rf-only {rf_only}, neither {neither}")
    c0 = Q.get(q57key('llama_8b_4bit', 'T1.0'), {}); print("  sign_check keys:", list((c0.get('sign_check') or {}).keys()), "| asymmetry:", c0.get('asymmetry'), "| R_hat_gap:", c0.get('R_hat_gap'))
    print("  Ridge-minus-MLP R gap by family (Q0057 shares): " + ', '.join(f"{lab}: " + ' '.join(f"{(Q.get(q57key(f,t),{}).get('permutation_shares',{}).get('ridge') or {}).get('R',float('nan')) - (Q.get(q57key(f,t),{}).get('permutation_shares',{}).get('mlp') or {}).get('R',float('nan')):+.3f}" for t in TEMPS) for f, lab in FAM))

# ── lambda sensitivity (§3.4, §9.1) ────────────────────────────────────
print("\n## lambda sensitivity from lambda_sweep")
shifts = []; at1 = []; at10 = []; interior = (0.5, 1.0, 2.0, 5.0)
for f, lab in FAM:
    for t in TEMPS:
        sw = {round(float(x['lambda']), 3): x['q_star']['R'] for x in (m(f, t).get('lambda_sweep') or [])}
        if not sw: continue
        vals = [sw[l] for l in interior if l in sw]
        if vals: shifts.append((max(vals) - min(vals), key(f, t)))
        if 1.0 in sw: at1.append(sw[1.0])
        if 10.0 in sw: at10.append(sw[10.0])
if shifts: print(f"  max interior-range R shift {max(shifts)[0]:.4f} at {max(shifts)[1]}; fleet-mean R at lambda=1 {np.mean(at1):.4f}, at lambda=10 {np.mean(at10) if at10 else float('nan'):.4f}; grid {sorted({round(float(x['lambda']),3) for x in (m('llama_8b_4bit','T0.0').get('lambda_sweep') or [])})}")

# ── bootstrap subset (§6.6) ────────────────────────────────────────────
if os.path.exists(a.boot):
    B = json.load(open(a.boot, encoding='utf-8-sig')); bc = B.get('cells') or {}
    print("\n## bootstrap_variance_p2 subset cells")
    p8 = os.path.join(os.path.dirname(os.path.dirname(a.boot)), 'apparatus_p2', 'per_cell.json')
    P8 = (json.load(open(p8, encoding='utf-8-sig')).get('cells') or {}) if os.path.exists(p8) else {}
    hw = []
    for k, v in bc.items():
        if v.get('n_bootstrap', 0) and str(v.get('variance_source', '')).startswith('phase10'):
            anc = np.array((v.get('per_resample_shares') or {}).get('anchored') or [])
            if anc.size == 0: print(f"  {k:34s} n={v.get('n_bootstrap')} (no per-resample shares)"); continue
            lo, hi = np.percentile(anc[:, 2], [2.5, 97.5]); hw.append((hi - lo) / 2)
            sw8 = (P8.get(k) or {}).get('lambda_sweep') or []; pt8 = next((x['q_star'] for x in sw8 if float(x['lambda']) == 1.0), None)
            pt = [pt8['E'], pt8['C'], pt8['R']] if pt8 else (P8.get(k) or {}).get('q_star')
            print(f"  {k:34s} n={len(anc)} point(phase8 q*R)={f3(pt[2]) if pt else 'n/a'} median={np.median(anc[:,2]):.3f} 95% CI [{lo:.3f}, {hi:.3f}] half-width {(hi-lo)/2:.3f} | anchored var {v.get('anchored_var_per_channel')}")
    if hw: print(f"  half-widths: min {min(hw):.3f} max {max(hw):.3f} mean {np.mean(hw):.3f}")
    k0 = next(iter(bc)); print("  phase10 cell keys:", list(bc[k0].keys())[:30])

# ── simple-baseline partial R2 (§6.7) from Q0042 r2 fields ─────────────
print("\n## simple-baseline partial R2 for S_t (r2_D - r2_C)/(1 - r2_C) from Q0042 fields")
for f, lab in FAM:
    v = []
    for t in TEMPS:
        mm = m(f, t); rc, rd = mm.get('r2_C_external_plus_constraint'), mm.get('r2_D_full_decomposition')
        v.append(None if rc is None or rd is None or rc >= 1 else (rd - rc) / (1 - rc))
    print(f"  {lab:14s} " + ' '.join(f3(x) for x in v))
