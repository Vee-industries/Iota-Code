"""
toy_nonlinearity_asymmetry.py
==============================

Mechanism test for the Ridge-vs-MLP attribution split observed across the
real-data cells where the paper reports Ridge-vs-MLP R² gaps.

HYPOTHESIS
    Ridge permutation-share attribution is biased relative to an MLP
    reference when one of the two input channels (E or S) carries nonlinear
    structure the linear model cannot see. Direction of the bias depends on
    which channel is nonlinear:
        • nonlinear E, linear S   → Ridge OVER-attributes to S (pattern 1)
        • linear E, nonlinear S   → Ridge UNDER-attributes to S
        • both linear or both nonlinear → shares roughly preserved

DESIGN — WHY THIS ISN'T THE OLD TOY
    The original toy parameterised nonlinearity as a *mixture weight*
    between A·S (big linear term) and tanh(A·S) (small bounded term). At
    nl=0.5 the tanh branch contributed O(1) of magnitude against the linear
    term's O(√d), so the "nonlinear" setting was actually a magnitude-
    shrinkage of the channel. The asymmetry it claimed to measure was
    partly a scale artifact.

    Here, nonlinearity is a RESIDUAL. For each channel we build a fixed
    random 2-layer tanh net, then subtract its best linear projection onto
    the linear contribution. The residual is orthogonal to the linear part
    by OLS and invisible to Ridge. We then RMS-rescale the residual to
    match the linear term's RMS and mix with weight nl:

        S_contrib = A·S      +  nl_S · scale_S · residual_S(S)
        E_contrib = B·E      +  nl_E · scale_E · residual_E(E)
        S_next    = S_contrib + E_contrib + σ·noise

    At nl=0 the channel is pure linear (Ridge fits it fully, shares =
    analytic). At nl=1 the channel carries linear + equal-RMS nonlinear
    residual (Ridge captures half the variance, MLP captures all).

    Also fixed vs. the old toy:
      • Ridge(alpha=0.01), StandardScaler on X and y  — matches
        analysis.py::_run_decomposition lines 1570-1608.
      • MLPRegressor(hidden=(256,), tanh, early_stopping) — matches
        analysis.py::linearity_check.
      • No PCA pooling (the E-channel was getting decimated asymmetrically).
      • n_permutes = 30, not 3.
      • TorchMLP guarded behind HAS_TORCH.

CHANNELS
    Two-channel (E, S → S'), not three. The paper's Rhat is under a
    three-channel (E + C + R) decomposition; here we collapse C into the
    background and ask the cleaner question "how do Ridge and MLP split
    E vs S share?". The mechanism this toy probes is estimator-dependence
    under channel-marginal nonlinearity, which is about the E↔S split,
    not about the C channel. Rhat in column names maps to the S-share.

USAGE
    python toy_nonlinearity_asymmetry.py              # full 3×3, 4 seeds
    python toy_nonlinearity_asymmetry.py --quick      # 2×2 corners, 2 seeds
    python toy_nonlinearity_asymmetry.py --seeds 0 1 2 3 4

OUTPUTS
    ./toy_nonlinearity_results/results.csv
    ./toy_nonlinearity_results/summary.txt
    ./toy_nonlinearity_results/heatmap.png        (if matplotlib available)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import warnings

# v0.80.0.20: force UTF-8 stdout/stderr on Windows. Default Python on
# Windows uses cp1252 for stdout, which crashes on Greek letters,
# arrows, and pictographs (⚠ → σ ρ etc.) that calibration scripts use.
# Reconfigure to UTF-8 with replacement chars as final safety net.
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    import io as _io
    if hasattr(_sys.stdout, 'buffer'):
        _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(_sys.stderr, 'buffer'):
        _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ══════════════════════════════════════════════════════════════════════
#  GENERATORS  (all deterministic given seed)
# ══════════════════════════════════════════════════════════════════════

def make_orthogonal_scaled(d, rho, rng):
    """d×d orthogonal matrix scaled by rho (persistence)."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return rho * Q


def make_linear_map(d_out, d_in, rng):
    """Random linear map with iid N(0, 1/d_in) entries — so ‖B·E‖²/d_out ≈ 1
    when E ~ N(0, I_{d_in})."""
    return rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)


def make_tanh_net(d_out, d_in, hidden_mult, rng):
    """Fixed random 2-layer tanh network. Weights are He-scaled per layer.
    Returns f : (n, d_in) → (n, d_out). No training."""
    h = hidden_mult * d_in
    W1 = rng.standard_normal((h,     d_in)) * np.sqrt(2.0 / d_in)
    b1 = rng.standard_normal(h) * 0.1
    W2 = rng.standard_normal((d_out, h))    * np.sqrt(2.0 / h)
    b2 = rng.standard_normal(d_out) * 0.1

    def f(X):
        return np.tanh(X @ W1.T + b1) @ W2.T + b2

    return f


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def compute_residual(nonlinear_raw, linear):
    """Orthogonalise nonlinear_raw against linear via OLS. Returns the
    residual: the part of nonlinear_raw that a linear predictor cannot
    recover from linear. Orthogonal to linear on the training data."""
    proj = Ridge(alpha=1e-6, fit_intercept=True).fit(linear, nonlinear_raw)
    return nonlinear_raw - proj.predict(linear)


# ══════════════════════════════════════════════════════════════════════
#  GENERATIVE SYSTEM
# ══════════════════════════════════════════════════════════════════════

def generate_cell(nl_S, nl_E, rho, sigma, d, n, seed):
    """Generate n iid triplets (S_prev, E, S_next) from
        S_next = A·S_prev + nl_S·scale_S·resS(S_prev)
               + B·E      + nl_E·scale_E·resE(E)
               + σ·noise

    Samples are drawn iid at the analytic steady-state variance, not from
    trajectory rollouts. Ridge/MLP permutation importance depends only on
    the joint distribution (S_prev, E, S_next), not on temporal
    correlations between successive samples.

    Returns S_prev (n,d), E (n,d), S_next (n,d), and a diagnostics dict.
    """
    rng = np.random.default_rng(seed)

    # --- generators ---
    A  = make_orthogonal_scaled(d, rho, rng)
    B  = make_linear_map(d, d, rng)
    NS = make_tanh_net(d, d, hidden_mult=2,
                       rng=np.random.default_rng(seed * 17 + 1001))
    NE = make_tanh_net(d, d, hidden_mult=2,
                       rng=np.random.default_rng(seed * 17 + 2002))

    # --- sample inputs at steady-state-approximated variance ---
    # S steady-state per-element variance v_S = (1 + σ²) / (1 - ρ²) under
    # S_{t+1} = A·S_t + B·E_t + σ·noise with orthogonal A and the scaling
    # convention above. Sample S_prev ~ N(0, v_S·I).
    v_S = (1.0 + sigma ** 2) / (1.0 - rho ** 2)
    S_prev = rng.standard_normal((n, d)) * np.sqrt(v_S)
    E      = rng.standard_normal((n, d))

    # --- linear contributions ---
    lin_S = S_prev @ A.T
    lin_E = E      @ B.T

    # --- nonlinear residuals, orthogonalised against linear ---
    res_S = compute_residual(NS(S_prev), lin_S)
    res_E = compute_residual(NE(E),      lin_E)

    # --- rescale residuals to match RMS of linear contributions ---
    scale_S = _rms(lin_S) / (_rms(res_S) + 1e-12)
    scale_E = _rms(lin_E) / (_rms(res_E) + 1e-12)
    res_S = scale_S * res_S
    res_E = scale_E * res_E

    # --- S_next ---
    noise  = sigma * rng.standard_normal((n, d))
    S_next = (lin_S + nl_S * res_S) + (lin_E + nl_E * res_E) + noise

    info = {
        'rms_lin_S':  _rms(lin_S),
        'rms_lin_E':  _rms(lin_E),
        'rms_res_S':  _rms(res_S),   # equals rms_lin_S by construction
        'rms_res_E':  _rms(res_E),
        'rms_noise':  _rms(noise),
        'rms_Snext':  _rms(S_next),
        'v_S':        v_S,
    }
    return S_prev, E, S_next, info


# ══════════════════════════════════════════════════════════════════════
#  FIT + PERMUTATION IMPORTANCE
# ══════════════════════════════════════════════════════════════════════

def _permutation_shares(model, sx, sy, X_tr, X_te, y_te, slices,
                        n_permutes, seed):
    """Measure permutation R² drop per block on the TEST set.
    Permutation happens in UNSCALED space (matching analysis.py::
    _permutation_sensitivity lines 1598-1605). Shares are clipped at zero
    and renormalised so they sum to 1."""
    y_te_s = sy.transform(y_te)
    X_te_s = sx.transform(X_te)
    r2_base = float(r2_score(y_te_s, model.predict(X_te_s),
                             multioutput='variance_weighted'))
    rng = np.random.default_rng(seed)
    drops = {}
    for name, (a, b) in slices.items():
        ds = []
        for _ in range(n_permutes):
            perm = rng.permutation(len(X_te))
            X_p = X_te.copy()
            X_p[:, a:b] = X_te[perm, a:b]
            pred = model.predict(sx.transform(X_p))
            ds.append(r2_base - float(r2_score(y_te_s, pred,
                                               multioutput='variance_weighted')))
        drops[name] = float(np.mean(ds))
    pos = {k: max(v, 0.0) for k, v in drops.items()}
    tot = sum(pos.values())
    shares = {k: (pos[k] / tot if tot > 1e-10 else float('nan')) for k in drops}
    return r2_base, drops, shares


def fit_and_measure(S_prev, E, S_next, seed, n_permutes=30, test_frac=0.2):
    d = S_prev.shape[1]
    X = np.hstack([S_prev, E]).astype(np.float32)
    y = S_next.astype(np.float32)
    slices = {'S': (0, d), 'E': (d, 2 * d)}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed)

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr)
    X_tr_s = sx.transform(X_tr)
    y_tr_s = sy.transform(y_tr)

    # Ridge (matches analysis.py line 1580)
    ridge = Ridge(alpha=0.01).fit(X_tr_s, y_tr_s)
    r2_ridge = float(r2_score(sy.transform(y_te), ridge.predict(sx.transform(X_te)),
                              multioutput='variance_weighted'))

    # MLP — matches analysis.py linearity_check exactly (default relu, default tol).
    # My earlier override to activation='tanh' tol=1e-5 was a mis-quote and also
    # tended to freeze the net near its small-init linear regime; relu is
    # demonstrably easier to optimise out of that regime in this sample budget.
    mlp = MLPRegressor(
        hidden_layer_sizes=(256,),
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=500,
        random_state=seed,
    ).fit(X_tr_s, y_tr_s)
    r2_mlp = float(r2_score(sy.transform(y_te), mlp.predict(sx.transform(X_te)),
                            multioutput='variance_weighted'))

    _, drops_r, shares_r = _permutation_shares(
        ridge, sx, sy, X_tr, X_te, y_te, slices, n_permutes, seed)
    _, drops_m, shares_m = _permutation_shares(
        mlp,   sx, sy, X_tr, X_te, y_te, slices, n_permutes, seed + 10_000)

    return {
        'R2_ridge':     r2_ridge,
        'R2_mlp':       r2_mlp,
        'R2_gap':       r2_mlp - r2_ridge,
        'Rhat_ridge':   shares_r['S'],
        'Rhat_mlp':     shares_m['S'],
        'Rhat_gap':     shares_m['S'] - shares_r['S'],
        'Ehat_ridge':   shares_r['E'],
        'Ehat_mlp':     shares_m['E'],
        'Ehat_gap':     shares_m['E'] - shares_r['E'],
        'drop_S_ridge': drops_r['S'],
        'drop_E_ridge': drops_r['E'],
        'drop_S_mlp':   drops_m['S'],
        'drop_E_mlp':   drops_m['E'],
    }


# ══════════════════════════════════════════════════════════════════════
#  GRID LOOP
# ══════════════════════════════════════════════════════════════════════

def run_grid(nl_S_levels, nl_E_levels, seeds,
             rho, sigma, d, n, n_permutes, out_csv):
    # v0.80.0.11: resume support. If out_csv already exists with prior
    # rows, load them and skip (nl_S, nl_E, seed) triples already computed.
    rows = []
    existing_keys = set()
    if os.path.exists(out_csv):
        try:
            prev = pd.read_csv(out_csv)
            for _, r in prev.iterrows():
                rows.append(r.to_dict())
                existing_keys.add((float(r.get('nl_S')),
                                   float(r.get('nl_E')),
                                   int(r.get('seed'))))
            if existing_keys:
                print(f"[resume] {len(existing_keys)} cell(s) already in "
                      f"{out_csv}, will skip them", flush=True)
        except Exception as e:
            print(f"[resume] couldn't read existing {out_csv}: {e}", flush=True)

    total, idx, t0 = len(nl_S_levels) * len(nl_E_levels) * len(seeds), 0, time.time()
    for nl_S in nl_S_levels:
        for nl_E in nl_E_levels:
            for seed in seeds:
                idx += 1
                key = (float(nl_S), float(nl_E), int(seed))
                if key in existing_keys:
                    print(f"[{idx:2d}/{total}] SKIP (cached)  "
                          f"nl_S={nl_S:.2f} nl_E={nl_E:.2f} seed={seed}",
                          flush=True)
                    continue
                t_cell = time.time()
                S_prev, E, S_next, info = generate_cell(nl_S, nl_E, rho, sigma, d, n, seed)
                m = fit_and_measure(S_prev, E, S_next, seed=seed, n_permutes=n_permutes)
                row = {
                    'nl_S': nl_S, 'nl_E': nl_E, 'seed': seed,
                    'rho': rho, 'sigma': sigma, 'd': d, 'n': n, **m,
                    **{f'info_{k}': v for k, v in info.items()},
                }
                rows.append(row)
                dt      = time.time() - t_cell
                elapsed = time.time() - t0
                eta     = elapsed / idx * (total - idx)
                print(f"[{idx:2d}/{total}] nl_S={nl_S:.2f} nl_E={nl_E:.2f} seed={seed}  "
                      f"R² ridge={row['R2_ridge']:.3f} mlp={row['R2_mlp']:.3f}  "
                      f"Rhat ridge={row['Rhat_ridge']:.3f} mlp={row['Rhat_mlp']:.3f}  "
                      f"gap={row['Rhat_gap']:+.3f}  ({dt:.1f}s, eta {eta:.0f}s)",
                      flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyze(df, out_dir):
    lines = []
    p = lines.append

    p("=" * 80)
    p("TOY NONLINEARITY ASYMMETRY — MECHANISM TEST")
    p("=" * 80)
    p(f"cells: {len(df)} rows  |  seeds: {sorted(df['seed'].unique())}")
    p(f"nl_S levels: {sorted(df['nl_S'].unique())}")
    p(f"nl_E levels: {sorted(df['nl_E'].unique())}")
    p(f"rho={df['rho'].iloc[0]}  sigma={df['sigma'].iloc[0]}  "
      f"d={df['d'].iloc[0]}  n={df['n'].iloc[0]}")
    p("")

    p("-" * 80)
    p("Rhat_gap  =  MLP S-share − Ridge S-share    (seed-mean per cell)")
    p("  negative → Ridge OVER-attributes to S  (pattern 1: negative asymmetry)")
    p("  positive → Ridge UNDER-attributes to S")
    p("-" * 80)
    gap = df.groupby(['nl_S', 'nl_E'])['Rhat_gap'].mean().unstack()
    p(gap.round(3).to_string()); p("")

    p("-" * 80)
    p("R²_gap  =  R²_MLP − R²_Ridge   (how much nonlinearity exists to find)")
    p("-" * 80)
    r2g = df.groupby(['nl_S', 'nl_E'])['R2_gap'].mean().unstack()
    p(r2g.round(3).to_string()); p("")

    p("-" * 80)
    p("Rhat_ridge  (Ridge S-share) — seed-mean per cell")
    p("-" * 80)
    rr = df.groupby(['nl_S', 'nl_E'])['Rhat_ridge'].mean().unstack()
    p(rr.round(3).to_string()); p("")

    p("-" * 80)
    p("Rhat_mlp    (MLP S-share) — seed-mean per cell")
    p("-" * 80)
    rm = df.groupby(['nl_S', 'nl_E'])['Rhat_mlp'].mean().unstack()
    p(rm.round(3).to_string()); p("")

    # Corner diagnostics
    def _corner(nS, nE):
        sub = df[(df['nl_S'] == nS) & (df['nl_E'] == nE)]
        return {k: float(sub[k].mean()) for k in
                ['Rhat_gap', 'Rhat_ridge', 'Rhat_mlp',
                 'R2_ridge', 'R2_mlp', 'R2_gap']}

    hi_S = max(df['nl_S'].unique())
    hi_E = max(df['nl_E'].unique())
    c_00 = _corner(0.0, 0.0)
    c_0E = _corner(0.0, hi_E)
    c_S0 = _corner(hi_S, 0.0)
    c_SE = _corner(hi_S, hi_E)

    p("-" * 80)
    p("CORNER DIAGNOSTICS")
    p("-" * 80)
    p(f"(0,0)        pure linear    : Rhat_gap={c_00['Rhat_gap']:+.3f}  "
      f"R²_gap={c_00['R2_gap']:+.3f}   ← should be ≈ 0")
    p(f"(0,{hi_E})    E-only nonlin  : Rhat_gap={c_0E['Rhat_gap']:+.3f}  "
      f"R²_gap={c_0E['R2_gap']:+.3f}   ← should be NEG")
    p(f"({hi_S},0)    S-only nonlin  : Rhat_gap={c_S0['Rhat_gap']:+.3f}  "
      f"R²_gap={c_S0['R2_gap']:+.3f}   ← should be POS")
    p(f"({hi_S},{hi_E})   both nonlin    : Rhat_gap={c_SE['Rhat_gap']:+.3f}  "
      f"R²_gap={c_SE['R2_gap']:+.3f}   ← should be small")
    p("")

    # Verdict
    p("-" * 80)
    p("VERDICT")
    p("-" * 80)
    null_ok  = abs(c_00['Rhat_gap']) < 0.05
    e_neg    = c_0E['Rhat_gap'] < -0.03
    s_pos    = c_S0['Rhat_gap'] >  0.03

    # Capacity check: at the off-corners MLP R² must noticeably exceed Ridge R².
    # If not, the MLP has not learned the nonlinearity at all, and any
    # statement about Rhat_gap is meaningless.
    mlp_finds_E = c_0E['R2_gap'] > 0.03
    mlp_finds_S = c_S0['R2_gap'] > 0.03
    capacity_ok = mlp_finds_E or mlp_finds_S

    if not capacity_ok:
        p(f"  [!!] MLP under-capacity: R²_gap is {c_0E['R2_gap']:+.3f} at the E-corner")
        p(f"       and {c_S0['R2_gap']:+.3f} at the S-corner. The MLP is not finding the")
        p("       nonlinearity that exists by construction in these cells. Rhat_gap")
        p("       numbers are not interpretable until MLP fits properly.")
        p("       Try: smaller d (--d 16), or larger n (--n 30000), or wider MLP.")
        p("")
        # still print the rest as diagnostics

    if null_ok:
        p("  [ok] null corner clean (Rhat_gap ~ 0 when both channels linear)")
    else:
        p(f"  [!!] null corner Rhat_gap = {c_00['Rhat_gap']:+.3f}  NOT clean - "
          "suspect Ridge alpha, scaling, or residual orthogonalisation")

    if not capacity_ok:
        p("  [--] skipping asymmetry verdict: MLP didn't fit (see capacity warning above)")
    elif e_neg and s_pos:
        p("  [ok] asymmetry confirmed: Rhat_gap flips sign between channels")
        p(f"       E-only nonlin corner:  {c_0E['Rhat_gap']:+.3f}   "
          "(Ridge over-attributes to S)")
        p(f"       S-only nonlin corner:  {c_S0['Rhat_gap']:+.3f}   "
          "(Ridge under-attributes to S)")
        p("  MECHANISM IS SUFFICIENT to produce the empirical cross-cell Ridge-vs-MLP")
        p(f"  pattern. A channel-marginal E-nonlinearity of nl_E={hi_E} produces")
        p(f"  a Ridge->MLP R-hat drop of up to {-c_0E['Rhat_gap']:.3f} in this toy.")
    elif e_neg and not s_pos:
        p(f"  [~~] partial: E-nonlin corner gives predicted negative gap "
          f"({c_0E['Rhat_gap']:+.3f}) but S-nonlin corner doesn't flip positive "
          f"({c_S0['Rhat_gap']:+.3f}).")
    elif s_pos and not e_neg:
        p(f"  [~~] partial: S-nonlin corner gives predicted positive gap "
          f"({c_S0['Rhat_gap']:+.3f}) but E-nonlin corner doesn't flip negative "
          f"({c_0E['Rhat_gap']:+.3f}).")
    else:
        p(f"  [xx] no asymmetry: E-corner={c_0E['Rhat_gap']:+.3f}, "
          f"S-corner={c_S0['Rhat_gap']:+.3f}")
        p("  Mechanism is NOT sufficient - empirical Ridge-vs-MLP split requires another account.")
    p("")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text + "\n")


def plot_heatmap(df, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    gap = df.groupby(['nl_S', 'nl_E'])['Rhat_gap'].mean().unstack()
    gap = gap.reindex(index=sorted(df['nl_S'].unique()),
                      columns=sorted(df['nl_E'].unique()))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    vmax = max(abs(np.nanmin(gap.values)), abs(np.nanmax(gap.values)), 0.05)
    im = ax.imshow(gap.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_xticks(range(len(gap.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in gap.columns])
    ax.set_yticks(range(len(gap.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in gap.index])
    ax.set_xlabel("nl_E   (E-channel nonlinearity)")
    ax.set_ylabel("nl_S   (S-channel nonlinearity)")
    ax.set_title("Rhat_gap  =  MLP S-share − Ridge S-share\n"
                 "(negative = Ridge over-attributes to S — pattern 1: negative asymmetry)")
    for i in range(len(gap.index)):
        for j in range(len(gap.columns)):
            v = gap.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha='center', va='center',
                        color='white' if abs(v) > vmax * 0.5 else 'black',
                        fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(out_dir, 'heatmap.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nHeatmap saved: {path}")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='./data/paper/calibration/toy_nonlinearity')
    parser.add_argument('--quick', action='store_true',
                        help="2×2 corner sweep × 2 seeds only (smoke test)")
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--d', type=int, default=16,
                        help="dimensionality of S and E. Defaults to 16; at the "
                             "default n=8000, d=64 puts the MLP below capacity "
                             "and it won't learn the nonlinearity (discovered "
                             "empirically — see v2 notes).")
    parser.add_argument('--n', type=int, default=8000,
                        help="samples per cell, pre train/test split")
    parser.add_argument('--rho', type=float, default=0.7)
    parser.add_argument('--sigma', type=float, default=0.3)
    parser.add_argument('--n_permutes', type=int, default=30)
    args = parser.parse_args()

    if args.quick:
        nl_S_levels = [0.0, 1.0]
        nl_E_levels = [0.0, 1.0]
        seeds       = args.seeds or [0, 1]
    else:
        nl_S_levels = [0.0, 0.5, 1.0]
        nl_E_levels = [0.0, 0.5, 1.0]
        seeds       = args.seeds or [0, 1, 2, 3]

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, 'results.csv')

    print("toy_nonlinearity_asymmetry.py")
    print(f"  nl_S levels : {nl_S_levels}")
    print(f"  nl_E levels : {nl_E_levels}")
    print(f"  seeds       : {seeds}")
    print(f"  rho={args.rho}  sigma={args.sigma}  d={args.d}  n={args.n}")
    print(f"  n_permutes  : {args.n_permutes}")
    print(f"  output dir  : {os.path.abspath(args.out)}\n")

    df = run_grid(nl_S_levels, nl_E_levels, seeds,
                  args.rho, args.sigma, args.d, args.n,
                  args.n_permutes, out_csv)
    df.to_csv(out_csv, index=False)
    print(f"\nResults: {out_csv}\n")
    analyze(df, args.out)
    plot_heatmap(df, args.out)


if __name__ == '__main__':
    main()
