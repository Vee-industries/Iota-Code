"""
ridge_bias_toy.py — drop into IOTA base folder, no installs needed.

Tests whether Ridge permutation-importance Rhat is downward-biased in a way that
compresses genuinely different true-R values toward a common apparent ceiling
as output noise rises.

If yes: cross-architecture Rhat ≈ 0.40-0.45 convergence at T=1.0 is at least
partly an estimator artifact, not a transformer property.

System: S_{t+1} = tanh(A·S_t + B·E_t) + σ·noise
  - Persistence ρ: spectral radius of A (orthogonal × ρ, exact)
  - "Temperature" σ: output noise stdev
  - E_t i.i.d. Gaussian (exogenous by construction, matches paper §3.9.1)

Estimators:
  - Ridge (α=1.0): paper's estimator
  - MLP (torch on GPU if available, sklearn fallback): truth proxy per V5b/V5e

Outputs (in ./ridge_bias_results/ by default):
  - results.csv — per-cell raw
  - summary.txt — aggregated + verdict
  - rhat_curves.png — Rhat vs σ, Ridge vs MLP side-by-side (if matplotlib present)

Usage:
  python ridge_bias_toy.py                          # medium preset
  python ridge_bias_toy.py --config small           # quick sanity (~2min on GPU)
  python ridge_bias_toy.py --config large           # more data (~15min on GPU)
  python ridge_bias_toy.py --d_hidden 2048          # transformer hidden dim in 2B param range
  python ridge_bias_toy.py --seeds 0 1 2 3 4        # more seeds
  python ridge_bias_toy.py --out ./my_results       # custom output dir
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

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

warnings.filterwarnings('ignore')

# ---------- detect torch / CUDA ----------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CUDA_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    CUDA_NAME = 'n/a'

if not HAS_TORCH:
    from sklearn.neural_network import MLPRegressor


# ---------- simulation ----------

def simulate_system(d_hidden, d_input, n_steps, rho_A, sigma, seed):
    """S_{t+1} = tanh(A S_t + B E_t) + sigma * eps_t, A orthogonal × rho_A."""
    rng = np.random.default_rng(seed)
    A_raw = rng.standard_normal((d_hidden, d_hidden))
    Q, _ = np.linalg.qr(A_raw)
    A = rho_A * Q  # spectral radius = rho_A exactly
    B = rng.standard_normal((d_hidden, d_input)) / np.sqrt(d_input)

    S = np.zeros((n_steps + 1, d_hidden), dtype=np.float32)
    E = rng.standard_normal((n_steps, d_input)).astype(np.float32)
    S[0] = (rng.standard_normal(d_hidden) * 0.1).astype(np.float32)

    for t in range(n_steps):
        noise = sigma * rng.standard_normal(d_hidden).astype(np.float32)
        S[t + 1] = np.tanh(A @ S[t] + B @ E[t]) + noise

    return S[:-1], E, S[1:]


# ---------- torch MLP (GPU path) ----------

class TorchMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_torch_mlp(X_tr, y_tr, hidden, max_iter, seed, device,
                  batch_size=256, lr=1e-3, tol=1e-5):
    """Train torch MLP on GPU. Returns wrapper with .predict(X_np) -> np.ndarray."""
    torch.manual_seed(seed)
    d_in = X_tr.shape[1]
    d_out = y_tr.shape[1]
    model = TorchMLP(d_in, d_out, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_tr).float().to(device)
    y_t = torch.from_numpy(y_tr).float().to(device)
    n = X_t.shape[0]

    prev_loss = None
    for epoch in range(max_iter):
        perm = torch.randperm(n, device=device)
        total = 0.0
        count = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]
        avg = total / count
        if prev_loss is not None and abs(prev_loss - avg) < tol:
            break
        prev_loss = avg

    class _Wrapper:
        def __init__(self, m, dev):
            self.m = m.eval()
            self.dev = dev

        def predict(self, X_np):
            with torch.no_grad():
                xt = torch.from_numpy(X_np).float().to(self.dev)
                return self.m(xt).cpu().numpy()

    return _Wrapper(model, device)


def fit_sklearn_mlp(X_tr, y_tr, hidden, max_iter, seed):
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden, activation='tanh',
        max_iter=max_iter, early_stopping=False,
        solver='adam', learning_rate_init=0.001,
        random_state=seed, tol=1e-5,
    )
    mlp.fit(X_tr, y_tr)
    return mlp


# ---------- permutation importance ----------

def permutation_share(model, X_test, y_test, slices, n_permutes, rng):
    """Permutation importance shares, shuffled on test, paper's protocol."""
    y_pred = model.predict(X_test)
    r2_base = r2_score(y_test, y_pred, multioutput='variance_weighted')

    drops = {}
    for name, (start, end) in slices.items():
        ds = []
        for _ in range(n_permutes):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(len(X_test))
            X_perm[:, start:end] = X_test[perm_idx, start:end]
            y_perm = model.predict(X_perm)
            r2_perm = r2_score(y_test, y_perm, multioutput='variance_weighted')
            ds.append(r2_base - r2_perm)
        drops[name] = float(np.mean(ds))

    pos = {k: max(v, 0.0) for k, v in drops.items()}
    total = sum(pos.values())
    shares = {k: (pos[k] / total if total > 1e-10 else float('nan')) for k in drops}
    return r2_base, drops, shares


# ---------- single cell ----------

def run_cell(d_hidden, d_input, n_steps, rho_A, sigma, seed,
             mlp_hidden, max_iter, n_permutes, test_frac=0.2, device=None):
    S_prev, E, S_next = simulate_system(d_hidden, d_input, n_steps, rho_A, sigma, seed)
    X = np.concatenate([S_prev, E], axis=1).astype(np.float32)
    y = S_next.astype(np.float32)

    rng = np.random.default_rng(seed + 10_000)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    slices = {'S': (0, d_hidden), 'E': (d_hidden, d_hidden + d_input)}

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    r2_ridge, drops_ridge, shares_ridge = permutation_share(
        ridge, X_te, y_te, slices, n_permutes, rng
    )

    if HAS_TORCH:
        mlp = fit_torch_mlp(X_tr, y_tr, mlp_hidden, max_iter, seed, device)
    else:
        mlp = fit_sklearn_mlp(X_tr, y_tr, mlp_hidden, max_iter, seed)
    r2_mlp, drops_mlp, shares_mlp = permutation_share(
        mlp, X_te, y_te, slices, n_permutes, rng
    )

    return {
        'rho_A': rho_A, 'sigma': sigma, 'seed': seed,
        'R2_ridge': r2_ridge, 'R2_mlp': r2_mlp, 'R2_gap': r2_mlp - r2_ridge,
        'Rhat_ridge': shares_ridge['S'], 'Rhat_mlp': shares_mlp['S'],
        'Rhat_gap': shares_mlp['S'] - shares_ridge['S'],
        'drop_S_ridge': drops_ridge['S'], 'drop_E_ridge': drops_ridge['E'],
        'drop_S_mlp': drops_mlp['S'], 'drop_E_mlp': drops_mlp['E'],
    }


# ---------- configs ----------
# σ calibration vs paper R² regime (signal stdev ~0.7 after tanh):
#   σ=0.00 → R²~1.00 (paper T=0, R²~0.99)
#   σ=0.05 → R²~0.99
#   σ=0.10 → R²~0.98
#   σ=0.20 → R²~0.92
#   σ=0.35 → R²~0.80 (just below paper T=1, R²~0.88)

CONFIGS = {
    'small': dict(
        d_hidden=64, d_input=32, n_steps=4000,
        rho_levels=[0.3, 0.5, 0.7, 0.9],
        sigma_levels=[0.0, 0.05, 0.1, 0.2, 0.35],
        seeds=[0, 1],
        mlp_hidden=(128, 64), max_iter=400, n_permutes=3,
    ),
    'medium': dict(
        d_hidden=256, d_input=128, n_steps=6000,
        rho_levels=[0.3, 0.5, 0.7, 0.9],
        sigma_levels=[0.0, 0.05, 0.1, 0.2, 0.35],
        seeds=[0, 1, 2],
        mlp_hidden=(256, 128), max_iter=500, n_permutes=3,
    ),
    'large': dict(
        d_hidden=512, d_input=256, n_steps=8000,
        rho_levels=[0.3, 0.5, 0.7, 0.9],
        sigma_levels=[0.0, 0.05, 0.1, 0.2, 0.35],
        seeds=[0, 1, 2],
        mlp_hidden=(512, 256), max_iter=600, n_permutes=5,
    ),
    'paper_scale': dict(
        d_hidden=2048, d_input=1024, n_steps=8000,
        rho_levels=[0.3, 0.5, 0.7, 0.9],
        sigma_levels=[0.0, 0.05, 0.1, 0.2, 0.35],
        seeds=[0, 1, 2],
        mlp_hidden=(1024, 512), max_iter=400, n_permutes=3,
    ),
}


# ---------- analysis + reporting ----------

def analyze(df, sigma_levels, out_dir):
    lines = []
    p = lines.append

    p("=" * 80)
    p("PER-CELL RESULTS (mean ± std over seeds)")
    p("=" * 80)
    agg = df.groupby(['rho_A', 'sigma']).agg({
        'R2_ridge': ['mean', 'std'],
        'R2_mlp': ['mean', 'std'],
        'Rhat_ridge': ['mean', 'std'],
        'Rhat_mlp': ['mean', 'std'],
        'Rhat_gap': 'mean',
    }).round(4)
    p(agg.to_string())

    bad = df[df['R2_gap'] < 0]
    if len(bad) > 0:
        p("\n⚠ CELLS WHERE MLP R² < RIDGE R² (MLP underfit — treat with caution):")
        for _, r in bad.iterrows():
            p(f"    ρ={r['rho_A']} σ={r['sigma']} seed={r['seed']}: "
              f"R²_mlp={r['R2_mlp']:.3f} R²_ridge={r['R2_ridge']:.3f}")

    p("\n" + "=" * 80)
    p("KEY DIAGNOSTIC: CROSS-PERSISTENCE SPREAD OF Rhat AT EACH σ")
    p("=" * 80)
    p("If Ridge spread collapses as σ rises while MLP spread preserved → WORLD 2")
    p("  (convergence in the paper is an estimator artifact)")
    p("If both Ridge and MLP spreads collapse → WORLD 3")
    p("  (convergence is a real property of noisy persistent-state systems)")
    p("If Ridge preserves ordering but compresses magnitudes → WORLD 2-lite")
    p("")

    mean_df = df.groupby(['rho_A', 'sigma'])[['Rhat_ridge', 'Rhat_mlp']].mean().reset_index()
    ridge_spreads, mlp_spreads = [], []
    for sigma in sigma_levels:
        sub = mean_df[mean_df['sigma'] == sigma].sort_values('rho_A')
        r_vals = sub['Rhat_ridge'].values
        m_vals = sub['Rhat_mlp'].values
        r_spread = r_vals.max() - r_vals.min()
        m_spread = m_vals.max() - m_vals.min()
        ridge_spreads.append(r_spread)
        mlp_spreads.append(m_spread)
        r_str = ' '.join(f'{v:.3f}' for v in r_vals)
        m_str = ' '.join(f'{v:.3f}' for v in m_vals)
        p(f"  σ={sigma:4.2f}:  Ridge Rhat [{r_str}]  spread={r_spread:.3f}")
        p(f"          MLP   Rhat [{m_str}]  spread={m_spread:.3f}")

    p("\n" + "=" * 80)
    p("VERDICT")
    p("=" * 80)
    r_ratio = ridge_spreads[-1] / ridge_spreads[0] if ridge_spreads[0] > 0 else float('nan')
    m_ratio = mlp_spreads[-1] / mlp_spreads[0] if mlp_spreads[0] > 0 else float('nan')
    p(f"Ridge spread ratio (high σ / low σ): {r_ratio:.3f}")
    p(f"MLP   spread ratio (high σ / low σ): {m_ratio:.3f}")
    p("")
    if r_ratio < 0.5 and m_ratio > 0.7:
        p("→ WORLD 2. Ridge compresses much faster than MLP. The cross-architecture")
        p("  Rhat ≈ 0.40-0.45 convergence in the paper is at least partly a Ridge")
        p("  permutation-importance ceiling under high sampling noise, not a property")
        p("  of transformer systems per se.")
    elif r_ratio < 0.5 and m_ratio < 0.5:
        p("→ WORLD 3. Both estimators compress similarly. The convergence is a real")
        p("  property of noisy persistent-state systems, not Ridge-specific.")
    elif r_ratio > 0.7 and m_ratio > 0.7:
        p("→ NO CONVERGENCE in either estimator on this toy. Likely a different")
        p("  underlying system dynamics than the paper's.")
    else:
        p("→ INTERMEDIATE. Inspect the per-cell data. Consider more seeds or σ levels.")

    p("\n" + "=" * 80)
    p("RIDGE BIAS SIGNATURE (Rhat_mlp - Rhat_ridge, paired)")
    p("=" * 80)
    p("V5a: Ridge undershoots on binary toy by up to 0.146")
    p("V5d: Ridge undershoots on pure XOR by 0.194")
    p("This toy falls somewhere in between if the bias generalizes:")
    p("")
    for rho in sorted(df['rho_A'].unique()):
        sub = df[df['rho_A'] == rho].groupby('sigma')['Rhat_gap'].mean()
        gaps = '  '.join(f'σ={s:.2f}: {g:+.3f}' for s, g in sub.items())
        p(f"  ρ={rho}: {gaps}")

    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')


def plot_results(df, sigma_levels, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    mean_df = df.groupby(['rho_A', 'sigma'])[['Rhat_ridge', 'Rhat_mlp']].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    rho_levels = sorted(mean_df['rho_A'].unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_levels)))
    for rho, c in zip(rho_levels, colors):
        sub = mean_df[mean_df['rho_A'] == rho].sort_values('sigma')
        ax1.plot(sub['sigma'], sub['Rhat_ridge'], 'o-', color=c, label=f'ρ={rho}')
        ax2.plot(sub['sigma'], sub['Rhat_mlp'], 'o-', color=c, label=f'ρ={rho}')
    for ax, title in [(ax1, 'Ridge Rhat (paper estimator)'), (ax2, 'MLP Rhat (truth proxy)')]:
        ax.set_xlabel('σ (noise, temperature analog)')
        ax.set_title(title)
        ax.legend(title='persistence ρ', loc='best')
        ax.grid(alpha=0.3)
    ax1.set_ylabel('Rhat = drop_S / (drop_S + drop_E)')
    fig.suptitle('Rhat vs noise at each persistence level\n'
                 '(if Ridge lines collapse while MLP lines stay apart → estimator artifact)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'rhat_curves.png')
    fig.savefig(path, dpi=140)
    print(f"Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=list(CONFIGS.keys()), default='medium',
                        help="small (~2min GPU), medium (default, ~5min GPU), large (~15min GPU), paper_scale (d=2048)")
    parser.add_argument('--out', default='./data/paper/calibration/ridge_bias',
                        help="Output directory (relative to cwd)")
    parser.add_argument('--d_hidden', type=int, default=None,
                        help="Override hidden dim (e.g., 2048 for 2B-param-class transformer scale)")
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--force_cpu', action='store_true',
                        help="Force torch to use CPU even if CUDA available")
    args = parser.parse_args()

    cfg = dict(CONFIGS[args.config])
    if args.d_hidden is not None:
        cfg['d_hidden'] = args.d_hidden
        cfg['d_input'] = max(args.d_hidden // 2, 32)
        cfg['mlp_hidden'] = (min(args.d_hidden, 1024), min(args.d_hidden // 2, 512))
    if args.n_steps is not None:
        cfg['n_steps'] = args.n_steps
    if args.seeds is not None:
        cfg['seeds'] = args.seeds

    device = DEVICE
    if HAS_TORCH and args.force_cpu:
        device = torch.device('cpu')

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, 'results.csv')

    print(f"ridge_bias_toy.py")
    print(f"  torch:     {'yes' if HAS_TORCH else 'no (using sklearn MLPRegressor CPU)'}")
    if HAS_TORCH:
        print(f"  device:    {device} ({CUDA_NAME if device.type == 'cuda' else 'CPU'})")
    print(f"  config:    {args.config}")
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"  output:    {os.path.abspath(args.out)}")
    print()

    # v0.80.0.11: resume support. Load existing csv_path if present and
    # skip (rho_A, sigma, seed) triples already computed.
    rows = []
    existing_keys = set()
    if os.path.exists(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            for _, r in prev.iterrows():
                rows.append(r.to_dict())
                existing_keys.add((float(r.get('rho_A')),
                                   float(r.get('sigma')),
                                   int(r.get('seed'))))
            if existing_keys:
                print(f"[resume] {len(existing_keys)} cell(s) already in "
                      f"{csv_path}, will skip them", flush=True)
        except Exception as e:
            print(f"[resume] couldn't read existing {csv_path}: {e}", flush=True)

    total_cells = len(cfg['rho_levels']) * len(cfg['sigma_levels']) * len(cfg['seeds'])
    idx = 0
    t_start = time.time()
    for rho in cfg['rho_levels']:
        for sigma in cfg['sigma_levels']:
            for seed in cfg['seeds']:
                idx += 1
                key = (float(rho), float(sigma), int(seed))
                if key in existing_keys:
                    print(f"[{idx}/{total_cells}] SKIP (cached) "
                          f"rho={rho} sigma={sigma} seed={seed}", flush=True)
                    continue
                t_cell = time.time()
                print(f"[{idx}/{total_cells}] ρ={rho} σ={sigma} seed={seed}", flush=True)
                row = run_cell(
                    cfg['d_hidden'], cfg['d_input'], cfg['n_steps'],
                    rho, sigma, seed,
                    cfg['mlp_hidden'], cfg['max_iter'], cfg['n_permutes'],
                    device=device,
                )
                rows.append(row)
                print(f"    R²: ridge={row['R2_ridge']:.3f}  mlp={row['R2_mlp']:.3f}  gap={row['R2_gap']:+.3f}")
                print(f"    Rhat:  ridge={row['Rhat_ridge']:.3f}  mlp={row['Rhat_mlp']:.3f}  gap={row['Rhat_gap']:+.3f}")
                dt = time.time() - t_cell
                elapsed = time.time() - t_start
                eta = elapsed / idx * (total_cells - idx)
                print(f"    cell: {dt:.1f}s  elapsed: {elapsed:.0f}s  eta: {eta:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nResults: {csv_path}\n")

    analyze(df, cfg['sigma_levels'], args.out)
    plot_results(df, cfg['sigma_levels'], args.out)


if __name__ == '__main__':
    main()
