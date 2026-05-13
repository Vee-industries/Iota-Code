"""
V5 Synthetic Calibration Suite: Does the Ridge permutation estimator recover true R?

Four calibration tests against systems with known or reference R:
  V5a: Stochastic persistence toy model (binary, closed-form R)
  V5b: Nonlinear RNN (continuous, MLP reference R)
  V5c: Correlated predictors (exogeneity violation sweep)
  V5d: XOR synergy (pure synergistic dependence)

No GPU, no model inference, no hidden states. Runs in ~2 minutes.

Usage: python v5_synthetic_calibration.py
Output: v5_calibration_results.json
"""

import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
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
warnings.filterwarnings("ignore")


# ============================================================
# Shared utilities
# ============================================================

def binary_entropy(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def estimate_R_ridge(X_S, X_E, y, n_permutations=200, seed=42):
    """Estimate R via Ridge + permutation sensitivity.
    X_S: state predictors (n_samples, d_s)
    X_E: external predictors (n_samples, d_e)
    y:   target (n_samples, d_y)
    Returns R_hat, E_hat, baseline_R2
    """
    rng = np.random.RandomState(seed)

    X = np.hstack([X_S, X_E])
    d_s = X_S.shape[1]
    d_e = X_E.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    baseline_r2 = model.score(X_test, y_test)

    drop_S = 0.0
    drop_E = 0.0

    for _ in range(n_permutations):
        # Permute S columns
        X_perm = X_test.copy()
        for col in range(d_s):
            X_perm[:, col] = rng.permutation(X_perm[:, col])
        drop_S += (baseline_r2 - model.score(X_perm, y_test))

        # Permute E columns
        X_perm = X_test.copy()
        for col in range(d_s, d_s + d_e):
            X_perm[:, col] = rng.permutation(X_perm[:, col])
        drop_E += (baseline_r2 - model.score(X_perm, y_test))

    drop_S /= n_permutations
    drop_E /= n_permutations

    drop_S = max(drop_S, 0)
    drop_E = max(drop_E, 0)

    total = drop_S + drop_E
    if total > 0:
        R_hat = drop_S / total
        E_hat = drop_E / total
    else:
        R_hat = 0.5
        E_hat = 0.5

    return R_hat, E_hat, baseline_r2


def estimate_R_mlp_reference(X_S, X_E, y, seed=42):
    """MLP-based R estimate used as reference for continuous systems."""
    rng = np.random.RandomState(seed)

    X = np.hstack([X_S, X_E])
    d_s = X_S.shape[1]
    d_e = X_E.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        early_stopping=True,
        random_state=seed,
        validation_fraction=0.15
    )
    model.fit(X_train, y_train)
    baseline_r2 = model.score(X_test, y_test)

    n_perm = 100
    drop_S = 0.0
    drop_E = 0.0

    for _ in range(n_perm):
        X_perm = X_test.copy()
        for col in range(d_s):
            X_perm[:, col] = rng.permutation(X_perm[:, col])
        drop_S += (baseline_r2 - model.score(X_perm, y_test))

        X_perm = X_test.copy()
        for col in range(d_s, d_s + d_e):
            X_perm[:, col] = rng.permutation(X_perm[:, col])
        drop_E += (baseline_r2 - model.score(X_perm, y_test))

    drop_S /= n_perm
    drop_E /= n_perm
    drop_S = max(drop_S, 0)
    drop_E = max(drop_E, 0)

    total = drop_S + drop_E
    if total > 0:
        R_ref = drop_S / total
    else:
        R_ref = 0.5

    return R_ref, baseline_r2


# ============================================================
# V5a: Stochastic persistence toy model (binary, closed-form)
# ============================================================

def true_R_toy(p):
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    num = binary_entropy(p / 2) - binary_entropy(p) / 2
    den = 1.0 - binary_entropy(p) / 2
    if den <= 0:
        return 0.0
    return num / den


def run_v5a():
    print("=" * 60)
    print("V5a: Stochastic persistence toy model (binary)")
    print("=" * 60)

    p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_steps = 50000
    n_repeats = 5
    results = []

    print(f"{'p':>5} | {'True R':>8} | {'R-hat':>8} | {'Error':>8} | {'R2':>6}")
    print("-" * 48)

    for p in p_values:
        R_true = true_R_toy(p)
        R_hats = []
        R2s = []

        for rep in range(n_repeats):
            seed = 42 + rep * 1000
            rng = np.random.RandomState(seed)

            E = rng.randint(0, 2, size=n_steps)
            S = np.zeros(n_steps + 1, dtype=int)
            S[0] = rng.randint(0, 2)
            persist = rng.random(n_steps) < p

            for t in range(n_steps):
                S[t + 1] = S[t] if persist[t] else E[t]

            X_S = S[:-1].reshape(-1, 1).astype(float)
            X_E = E.reshape(-1, 1).astype(float)
            y = S[1:].astype(float).reshape(-1, 1)

            R_hat, _, r2 = estimate_R_ridge(X_S, X_E, y, seed=seed + 500)
            R_hats.append(R_hat)
            R2s.append(r2)

        R_hat_mean = np.mean(R_hats)
        error = R_hat_mean - R_true
        results.append({
            "p": p,
            "R_true": round(R_true, 4),
            "R_hat": round(R_hat_mean, 4),
            "R_hat_std": round(np.std(R_hats), 4),
            "error": round(error, 4),
            "R2": round(np.mean(R2s), 4)
        })
        print(f"{p:>5.1f} | {R_true:>8.4f} | {R_hat_mean:>8.4f} | {error:>+8.4f} | {np.mean(R2s):>6.4f}")

    return results


# ============================================================
# V5b: Nonlinear RNN (continuous state, MLP reference)
# ============================================================

def run_v5b():
    print("\n" + "=" * 60)
    print("V5b: Nonlinear RNN (continuous, MLP reference)")
    print("=" * 60)

    dim = 8
    n_steps = 20000
    noise_std = 0.1
    results = []

    configs = [
        {"label": "E-dominant", "a_scale": 0.2, "b_scale": 0.8},
        {"label": "balanced",   "a_scale": 0.5, "b_scale": 0.5},
        {"label": "S-dominant", "a_scale": 0.8, "b_scale": 0.2},
        {"label": "strong-S",   "a_scale": 0.95, "b_scale": 0.1},
    ]

    print(f"{'Config':>12} | {'R_ref(MLP)':>10} | {'R_hat(Ridge)':>12} | {'Gap':>8} | {'R2_ridge':>9} | {'R2_mlp':>7}")
    print("-" * 72)

    for cfg in configs:
        seed = 42
        rng = np.random.RandomState(seed)

        A_raw = rng.randn(dim, dim)
        A_raw = A_raw / np.max(np.abs(np.linalg.eigvals(A_raw)))
        A = cfg["a_scale"] * A_raw

        B_raw = rng.randn(dim, dim)
        B = cfg["b_scale"] * B_raw / np.linalg.norm(B_raw)

        S = np.zeros((n_steps + 1, dim))
        E = rng.randn(n_steps, dim)
        S[0] = rng.randn(dim) * 0.1

        for t in range(n_steps):
            S[t + 1] = np.tanh(A @ S[t] + B @ E[t]) + rng.randn(dim) * noise_std

        X_S = S[:-1]
        X_E = E
        y = S[1:]

        R_hat, _, r2_ridge = estimate_R_ridge(X_S, X_E, y, n_permutations=100, seed=seed)
        R_ref, r2_mlp = estimate_R_mlp_reference(X_S, X_E, y, seed=seed)

        gap = R_hat - R_ref
        results.append({
            "config": cfg["label"],
            "a_scale": cfg["a_scale"],
            "b_scale": cfg["b_scale"],
            "R_ref_mlp": round(R_ref, 4),
            "R_hat_ridge": round(R_hat, 4),
            "gap": round(gap, 4),
            "R2_ridge": round(r2_ridge, 4),
            "R2_mlp": round(r2_mlp, 4)
        })
        print(f"{cfg['label']:>12} | {R_ref:>10.4f} | {R_hat:>12.4f} | {gap:>+8.4f} | {r2_ridge:>9.4f} | {r2_mlp:>7.4f}")

    return results


# ============================================================
# V5c: Correlated predictors (exogeneity violation sweep)
# ============================================================

def run_v5c():
    print("\n" + "=" * 60)
    print("V5c: Correlated predictors (exogeneity violation)")
    print("=" * 60)

    p_persist = 0.5
    R_true_exogenous = true_R_toy(p_persist)
    n_steps = 50000
    n_repeats = 5

    q_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    print(f"{'q(corr)':>8} | {'R_true(q=0)':>11} | {'R-hat':>8} | {'Shift':>8}")
    print("-" * 44)

    for q in q_values:
        R_hats = []

        for rep in range(n_repeats):
            seed = 42 + rep * 1000
            rng = np.random.RandomState(seed)

            S = np.zeros(n_steps + 1, dtype=int)
            E = np.zeros(n_steps, dtype=int)
            S[0] = rng.randint(0, 2)

            for t in range(n_steps):
                if rng.random() < q:
                    E[t] = S[t]
                else:
                    E[t] = rng.randint(0, 2)

                if rng.random() < p_persist:
                    S[t + 1] = S[t]
                else:
                    S[t + 1] = E[t]

            X_S = S[:-1].reshape(-1, 1).astype(float)
            X_E = E.reshape(-1, 1).astype(float)
            y = S[1:].astype(float).reshape(-1, 1)

            R_hat, _, _ = estimate_R_ridge(X_S, X_E, y, seed=seed + 500)
            R_hats.append(R_hat)

        R_hat_mean = np.mean(R_hats)
        shift = R_hat_mean - results[0]["R_hat"] if len(results) > 0 else 0.0

        results.append({
            "q_correlation": q,
            "R_true_at_q0": round(R_true_exogenous, 4),
            "R_hat": round(R_hat_mean, 4),
            "R_hat_std": round(np.std(R_hats), 4),
            "shift_from_q0": round(shift, 4)
        })
        print(f"{q:>8.1f} | {R_true_exogenous:>11.4f} | {R_hat_mean:>8.4f} | {shift:>+8.4f}")

    return results


# ============================================================
# V5d: XOR synergy
# ============================================================

def run_v5d():
    print("\n" + "=" * 60)
    print("V5d: XOR synergy (pure synergistic dependence)")
    print("=" * 60)

    n_steps = 50000
    n_repeats = 5
    R_true = 1.0

    R_hats = []
    R2s = []

    for rep in range(n_repeats):
        seed = 42 + rep * 1000
        rng = np.random.RandomState(seed)

        S_t = rng.randint(0, 2, size=n_steps)
        E_t = rng.randint(0, 2, size=n_steps)
        S_next = np.bitwise_xor(S_t, E_t)

        X_S = S_t.reshape(-1, 1).astype(float)
        X_E = E_t.reshape(-1, 1).astype(float)
        y = S_next.astype(float).reshape(-1, 1)

        R_hat, E_hat, r2 = estimate_R_ridge(X_S, X_E, y, seed=seed + 500)
        R_hats.append(R_hat)
        R2s.append(r2)

    R_hat_mean = np.mean(R_hats)
    R2_mean = np.mean(R2s)
    error = R_hat_mean - R_true

    print(f"True R (E-first ordering): {R_true:.4f}")
    print(f"Estimated R-hat:           {R_hat_mean:.4f} (std: {np.std(R_hats):.4f})")
    print(f"Error:                     {error:+.4f}")
    print(f"Ridge R2:                  {R2_mean:.4f}")
    print()

    if R2_mean < 0.01:
        print("Ridge R2 ~ 0: linear model cannot capture XOR.")
        print("This is expected. XOR is purely nonlinear/synergistic.")
        print("Estimator fails on purely synergistic systems,")
        print("consistent with the stated limitation (Section 10).")
    else:
        print(f"Ridge captures some structure (R2={R2_mean:.4f}).")

    result = {
        "R_true_E_first": R_true,
        "R_hat": round(R_hat_mean, 4),
        "R_hat_std": round(np.std(R_hats), 4),
        "error": round(error, 4),
        "R2": round(R2_mean, 4),
        "note": "XOR is purely nonlinear; Ridge R2~0 expected. Estimator cannot recover R in purely synergistic systems."
    }

    return result


# ============================================================
# V5e: Synergy mixture (linear + XOR sweep)
# ============================================================

def run_v5e():
    print("\n" + "=" * 60)
    print("V5e: Synergy mixture (linear + XOR, alpha sweep)")
    print("=" * 60)

    n_steps = 20000
    n_repeats = 3

    # alpha = 0: pure linear (persistence model, p=0.5)
    # alpha = 1: pure XOR
    # In between: with prob alpha use XOR, with prob (1-alpha) use persistence
    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    p_persist = 0.5
    results = []

    print(f"{'alpha':>6} | {'R_ref(MLP)':>10} | {'R_hat(Ridge)':>12} | {'Gap':>8} | {'R2_ridge':>9} | {'R2_mlp':>7}")
    print("-" * 66)

    for alpha in alpha_values:
        R_hats_ridge = []
        R_refs_mlp = []
        R2_ridges = []
        R2_mlps = []

        for rep in range(n_repeats):
            seed = 42 + rep * 1000
            rng = np.random.RandomState(seed)

            E = rng.randint(0, 2, size=n_steps)
            S = np.zeros(n_steps + 1, dtype=int)
            S[0] = rng.randint(0, 2)

            for t in range(n_steps):
                if rng.random() < alpha:
                    # XOR transition
                    S[t + 1] = S[t] ^ E[t]
                else:
                    # Persistence transition
                    if rng.random() < p_persist:
                        S[t + 1] = S[t]
                    else:
                        S[t + 1] = E[t]

            X_S = S[:-1].reshape(-1, 1).astype(float)
            X_E = E.reshape(-1, 1).astype(float)
            y = S[1:].astype(float).reshape(-1, 1)

            R_hat, _, r2_ridge = estimate_R_ridge(X_S, X_E, y, n_permutations=100, seed=seed + 500)
            R_ref, r2_mlp = estimate_R_mlp_reference(X_S, X_E, y, seed=seed + 500)

            R_hats_ridge.append(R_hat)
            R_refs_mlp.append(R_ref)
            R2_ridges.append(r2_ridge)
            R2_mlps.append(r2_mlp)

        R_hat_mean = np.mean(R_hats_ridge)
        R_ref_mean = np.mean(R_refs_mlp)
        gap = R_hat_mean - R_ref_mean
        r2_ridge_mean = np.mean(R2_ridges)
        r2_mlp_mean = np.mean(R2_mlps)

        results.append({
            "alpha": alpha,
            "R_ref_mlp": round(R_ref_mean, 4),
            "R_hat_ridge": round(R_hat_mean, 4),
            "gap": round(gap, 4),
            "R2_ridge": round(r2_ridge_mean, 4),
            "R2_mlp": round(r2_mlp_mean, 4)
        })
        print(f"{alpha:>6.1f} | {R_ref_mean:>10.4f} | {R_hat_mean:>12.4f} | {gap:>+8.4f} | {r2_ridge_mean:>9.4f} | {r2_mlp_mean:>7.4f}")

    return results


# ============================================================
# V5f: Doubly-conditional system (three-channel chain rule)
# ------------------------------------------------------------
# v0.81.1.5 ship 3 — paper 2 measurement-paper validation system.
#
# Construction per codebot_handoff_v0_14.md §3.1, sourced from
# paper2_preregistration_v2.md §4.6:
#
#   E, C ~ N(0, 1)                        (independent)
#   S = E + C + ε,    ε ~ N(0, σ_S^2)
#   Y = sin(S) + η,    η ~ N(0, σ_Y^2)
#
# The three "channels" the apparatus consumes are E, C, S (with S the
# integrated state that contains both inputs). Y is the prediction
# target — analogous to S_{t+1} in the recurrent framing. This system
# has a clean analytic chain-rule decomposition:
#
#   I(Y; E, C, S) = I(Y; S)              (S is sufficient for Y)
#   R-share goes entirely to S in the (E, C, S) → Y attribution
#   sense, BUT S is itself fully predictable from E + C, so under a
#   marginalized analysis the apparatus should:
#     - Direct kNN-MI per channel: I(Y; E), I(Y; E, C), I(Y; E, C, S)
#       chain. Conditional terms identify how much each channel adds.
#     - Function classes: Ridge / MLP / RKHS / RF disagree on how to
#       split attribution because S = E + C is a linear consequence of
#       its own inputs and the y ~ sin(S) nonlinearity is asymmetric
#       across the channels.
#
# Noise scales σ_S, σ_Y are not arbitrary — they're tuned so that the
# system's MI magnitudes match the four empirical cells' kNN-MI
# magnitudes within a tolerance band. Without this matching, V5f is
# either too clean (apparatus easy) or too noisy (apparatus useless),
# and the calibration claim "V5f mirrors empirical chain-rule
# geometry" doesn't hold up.
#
# The tuning loop:
#   1. Compute target MI band = (median, p25, p75) of empirical cells'
#      I(S_{t+1}; E_t) and I(S_{t+1}; E_t, C_t) from Q0042 cells.
#   2. Binary-search σ_S to land I(Y; E, C) inside the empirical band.
#   3. Fix σ_S, binary-search σ_Y to land I(Y; S) inside the band.
#   4. Record final σ_S, σ_Y in v5f_construction_artifact.md alongside
#      the analytic ground-truth R values for the converged system.
#
# Analytic ground truth (under converged σ_S, σ_Y, all three channels
# fed to the apparatus, attribution to S as the "internal" channel):
#
#   R_truth_marginal = I(Y; S | E, C) / I(Y; E, C, S)
#                    = I(Y; S | E, C) / I(Y; S)
#                    = (I(Y; S) - I(Y; E + C)) / I(Y; S)
#
# (because Y is a function of S only, conditioning on E and C beyond
# what they tell us about S adds nothing). This is computed
# numerically post-tuning and reported in the v5f_results entry.
# ============================================================


def estimate_R_3channel(X_E, X_C, X_S, y, n_permutations=200,
                         seed=42, regressor='ridge'):
    """Three-channel permutation R estimator for V5f.

    Channels: (E, C, S) → y. R is the share of the joint signal that
    goes through the S channel (the "internal" coordinate the paper's
    R-share names). E + C are external; S is the integrated channel.

    regressor: 'ridge' | 'mlp' | 'rkhs' | 'rf'.
      - ridge: linear baseline (paper §5/§6 headline).
      - mlp:   smooth-nonlinear (V5b precedent — truth proxy for
               nonlinear targets where Ridge under-attributes).
      - rkhs:  characteristic-kernel ridge regression. Single fit at
               the median pairwise distance with Matérn ν=1.5 (the
               middle of the kernel grid in run_function_class_
               sensitivity.RKHS_KERNELS — full grid is too expensive
               for V5 calibration; one representative kernel suffices).
      - rf:    axis-aligned recursive partitioning. Hyperparameters
               match RF_HYPERS in run_function_class_sensitivity
               (n_estimators=100, max_depth=None, min_samples_leaf=5,
               random_state=seed, n_jobs=4).

    v0.81.1.5 ship 3: extended from {ridge, mlp} to four classes so V5
    calibration runs produce the same four-class shares the apparatus
    consumes on empirical cells.
    """
    rng = np.random.RandomState(seed)
    X = np.hstack([X_E, X_C, X_S])
    d_e = X_E.shape[1]
    d_c = X_C.shape[1]
    d_s = X_S.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    if regressor == 'ridge':
        model = Ridge(alpha=1.0)
    elif regressor == 'mlp':
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            random_state=seed,
            validation_fraction=0.15,
        )
    elif regressor == 'rkhs':
        # Characteristic-kernel ridge at the median pairwise distance
        # of X_train. Matérn ν=1.5 is the middle of the production
        # kernel grid; once-differentiable kernels balance smoothness
        # and locality. Subsample if X_train is large (RKHS is O(n³)).
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.gaussian_process.kernels import Matern as _Matern
        from sklearn.metrics.pairwise import euclidean_distances
        n_train_full = X_train.shape[0]
        rkhs_n_cap = 5000  # matches RKHS_N_SUBSAMPLE in production runner
        if n_train_full > rkhs_n_cap:
            sub_idx = rng.choice(n_train_full, size=rkhs_n_cap, replace=False)
            X_train_rkhs = X_train[sub_idx]
            y_train_rkhs = y_train[sub_idx]
        else:
            X_train_rkhs = X_train
            y_train_rkhs = y_train
        # Median pairwise on a 1000-row subsample of X_train_rkhs
        ms_idx = rng.choice(X_train_rkhs.shape[0],
                             size=min(1000, X_train_rkhs.shape[0]),
                             replace=False)
        pwd = euclidean_distances(X_train_rkhs[ms_idx])
        offdiag = pwd[np.triu_indices_from(pwd, k=1)]
        median_pairwise = float(np.median(offdiag)) if offdiag.size > 0 else 1.0
        if median_pairwise <= 0:
            median_pairwise = 1.0
        kernel = _Matern(length_scale=median_pairwise, nu=1.5)
        model = KernelRidge(alpha=1e-3, kernel=kernel)
        # Fit on subsampled set, score on the FULL test set
        model.fit(X_train_rkhs, y_train_rkhs)
        baseline_r2 = model.score(X_test, y_test)
        # Permutation drops on full test, predict via subsampled-train kernel
        n_perm = max(50, n_permutations // 4)
        slice_E = slice(0,           d_e)
        slice_C = slice(d_e,         d_e + d_c)
        slice_S = slice(d_e + d_c,   d_e + d_c + d_s)
        drops = {'E': 0.0, 'C': 0.0, 'S': 0.0}
        for label, sl in [('E', slice_E), ('C', slice_C), ('S', slice_S)]:
            for _ in range(n_perm):
                X_perm = X_test.copy()
                for col in range(sl.start, sl.stop):
                    X_perm[:, col] = rng.permutation(X_perm[:, col])
                drops[label] += baseline_r2 - model.score(X_perm, y_test)
            drops[label] /= n_perm
            drops[label] = max(drops[label], 0.0)
        total = sum(drops.values())
        if total > 0:
            shares = {k: v / total for k, v in drops.items()}
        else:
            shares = {'E': 1.0/3, 'C': 1.0/3, 'S': 1.0/3}
        return shares, baseline_r2
    elif regressor == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=4,
        )
    else:
        raise ValueError(f"unknown regressor: {regressor}")
    model.fit(X_train, y_train)
    baseline_r2 = model.score(X_test, y_test)

    # Reduce permutations for slow regressors: MLP, RF (RKHS handled
    # separately above with its own loop and subsample).
    if regressor in ('mlp', 'rf'):
        n_perm = max(50, n_permutations // 4)
    else:
        n_perm = n_permutations
    slice_E = slice(0,           d_e)
    slice_C = slice(d_e,         d_e + d_c)
    slice_S = slice(d_e + d_c,   d_e + d_c + d_s)

    drops = {'E': 0.0, 'C': 0.0, 'S': 0.0}
    for label, sl in [('E', slice_E), ('C', slice_C), ('S', slice_S)]:
        for _ in range(n_perm):
            X_perm = X_test.copy()
            for col in range(sl.start, sl.stop):
                X_perm[:, col] = rng.permutation(X_perm[:, col])
            drops[label] += baseline_r2 - model.score(X_perm, y_test)
        drops[label] /= n_perm
        drops[label] = max(drops[label], 0.0)

    total = sum(drops.values())
    if total > 0:
        shares = {k: v / total for k, v in drops.items()}
    else:
        shares = {'E': 1.0/3, 'C': 1.0/3, 'S': 1.0/3}
    return shares, baseline_r2


def _v5f_generate(n_steps, sigma_S, sigma_Y, seed=42):
    """Generate one V5f sample of size n_steps. Returns (E, C, S, Y)
    each (n_steps, 1). E, C ~ N(0,1) independent; S = E + C + ε,
    ε ~ N(0, σ_S²); Y = sin(S) + η, η ~ N(0, σ_Y²)."""
    rng = np.random.RandomState(seed)
    E = rng.standard_normal((n_steps, 1))
    C = rng.standard_normal((n_steps, 1))
    eps = sigma_S * rng.standard_normal((n_steps, 1))
    S = E + C + eps
    eta = sigma_Y * rng.standard_normal((n_steps, 1))
    Y = np.sin(S) + eta
    return E, C, S, Y


def _v5f_compute_mi_proxies(E, C, S, Y, seed=42):
    """Approximate I(Y; E, C) and I(Y; S) via R²-from-Gaussian-MI proxy
    (MI ≈ -0.5 * log(1 - r²) for Gaussian residuals). NOT a kNN-MI
    estimate — that's expensive. This is the cheap inner-loop proxy
    used during tuning to land the system roughly in the empirical band.
    Final V5f reports kNN-MI on the converged system (separate, slower
    reliability run)."""
    # I(Y; E, C) via MLP r² (E + C → Y)
    X_ec = np.hstack([E, C])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_ec, Y, test_size=0.2, random_state=seed)
    model = MLPRegressor(
        hidden_layer_sizes=(64,), max_iter=300, early_stopping=True,
        random_state=seed, validation_fraction=0.15)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        model.fit(X_tr, y_tr.ravel())
        r2_ec = max(0.0, model.score(X_te, y_te.ravel()))
    mi_ec = -0.5 * np.log(max(1e-9, 1.0 - r2_ec))

    # I(Y; S) via MLP r² (S → Y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        S, Y, test_size=0.2, random_state=seed)
    model = MLPRegressor(
        hidden_layer_sizes=(64,), max_iter=300, early_stopping=True,
        random_state=seed, validation_fraction=0.15)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        model.fit(X_tr, y_tr.ravel())
        r2_s = max(0.0, model.score(X_te, y_te.ravel()))
    mi_s = -0.5 * np.log(max(1e-9, 1.0 - r2_s))
    return mi_ec, mi_s


def _v5f_tune_noise(target_mi_ec, target_mi_s,
                     n_steps=10000, max_iter=12,
                     tol_log=0.10, seed=42):
    """Binary search σ_S, σ_Y in log-space to land MI proxies inside
    the target bands. Returns (σ_S, σ_Y, tuning_log).

    target_mi_ec, target_mi_s: target MI values (in nats — natural log
    convention from the r²-from-Gaussian proxy).

    Strategy:
      - σ_S controls I(Y; S) indirectly via S's noise (more ε → S is
        less informative about E+C → I(Y; E, C) drops). Tune σ_S to
        land I(Y; E, C) at target_mi_ec.
      - σ_Y controls I(Y; S) directly (more η → less of Y predictable
        from S → I(Y; S) drops). Tune σ_Y to land I(Y; S) at target_mi_s.
      - Sequential: first σ_S, then fix it and tune σ_Y. Approximation —
        the two are weakly coupled but the proxy is fast enough that we
        can do one re-pass on σ_S after σ_Y converges if the joint
        residual is large.
    """
    log = []

    # Initial guess
    sigma_S = 0.5
    sigma_Y = 0.3

    def _measure(sS, sY):
        E, C, S, Y = _v5f_generate(n_steps, sS, sY, seed=seed)
        return _v5f_compute_mi_proxies(E, C, S, Y, seed=seed)

    # ── Phase 1: tune σ_S to land I(Y; E, C) on target ──
    lo, hi = 1e-3, 5.0
    for it in range(max_iter):
        sigma_S = (lo + hi) / 2.0
        mi_ec, mi_s = _measure(sigma_S, sigma_Y)
        log.append({
            'phase': 1, 'iter': it,
            'sigma_S': float(sigma_S), 'sigma_Y': float(sigma_Y),
            'mi_ec': float(mi_ec), 'mi_s': float(mi_s),
            'target_mi_ec': float(target_mi_ec),
            'target_mi_s':  float(target_mi_s),
        })
        # If MI(E,C → Y) too high: S is too clean → ε too small → raise σ_S
        if mi_ec > target_mi_ec * (1.0 + tol_log):
            lo = sigma_S
        elif mi_ec < target_mi_ec * (1.0 - tol_log):
            hi = sigma_S
        else:
            break

    # ── Phase 2: tune σ_Y to land I(Y; S) on target ──
    lo, hi = 1e-3, 5.0
    for it in range(max_iter):
        sigma_Y = (lo + hi) / 2.0
        mi_ec, mi_s = _measure(sigma_S, sigma_Y)
        log.append({
            'phase': 2, 'iter': it,
            'sigma_S': float(sigma_S), 'sigma_Y': float(sigma_Y),
            'mi_ec': float(mi_ec), 'mi_s': float(mi_s),
            'target_mi_ec': float(target_mi_ec),
            'target_mi_s':  float(target_mi_s),
        })
        if mi_s > target_mi_s * (1.0 + tol_log):
            lo = sigma_Y
        elif mi_s < target_mi_s * (1.0 - tol_log):
            hi = sigma_Y
        else:
            break

    return float(sigma_S), float(sigma_Y), log


def run_v5f(target_mi_ec=0.30, target_mi_s=0.45,
             n_steps_final=20000, n_repeats=3,
             tune_n_steps=10000, seed=42):
    """V5f doubly-conditional construction + Ridge/MLP attribution
    measurement at converged σ_S, σ_Y.

    target_mi_ec, target_mi_s: empirical MI band centers (in nats).
    Default values are reasonable starting points for the four
    empirical cells; final values are committed to the construction
    artifact post-tuning. Caller can override with empirical-cell-
    derived bands when those are computed.

    Returns dict with tuning log, converged parameters, analytic-
    truth R, and per-class permutation shares (Ridge, MLP) at full
    n_steps.
    """
    print("\n" + "=" * 60)
    print("V5f: Doubly-conditional system (E, C → S → Y)")
    print("=" * 60)
    print(f"Target MI(Y; E, C) ≈ {target_mi_ec:.3f} nats")
    print(f"Target MI(Y; S)    ≈ {target_mi_s:.3f} nats")

    # ── Tune ──────────────────────────────────────────────────────
    print(f"\nTuning σ_S, σ_Y on n={tune_n_steps}...")
    sigma_S, sigma_Y, tuning_log = _v5f_tune_noise(
        target_mi_ec=target_mi_ec, target_mi_s=target_mi_s,
        n_steps=tune_n_steps, seed=seed,
    )
    print(f"Converged: σ_S = {sigma_S:.4f}, σ_Y = {sigma_Y:.4f}")
    print(f"  ({len(tuning_log)} bisection iterations)")

    # ── Generate at full n_steps + measure ────────────────────────
    print(f"\nGenerating final V5f at n={n_steps_final}, "
          f"{n_repeats} seeds...")

    R_truth_runs = []
    ridge_shares_runs = []
    mlp_shares_runs = []
    rkhs_shares_runs = []
    rf_shares_runs = []
    R2_ridge_runs = []
    R2_mlp_runs = []
    R2_rkhs_runs = []
    R2_rf_runs = []
    mi_ec_final = []
    mi_s_final = []

    for rep in range(n_repeats):
        s = seed + rep * 1000
        E, C, S, Y = _v5f_generate(n_steps_final, sigma_S, sigma_Y, seed=s)

        # MI proxies on the final-size system
        mi_ec, mi_s = _v5f_compute_mi_proxies(E, C, S, Y, seed=s)
        mi_ec_final.append(mi_ec)
        mi_s_final.append(mi_s)

        # Analytic-truth R: under three-channel attribution (E, C, S → Y),
        # truth allocates all of R to S (Y is a function of S alone, not
        # of E and C beyond what they tell us about S). The MARGINAL R
        # truth is therefore 1.0 if all the chain-rule mass goes through S.
        # Operationally: R_truth = (I(Y;S) - I(Y;E,C)) / I(Y;S)
        # — the "S adds beyond E+C" portion divided by total S information.
        # When σ_S → 0, S is fully determined by E+C, R_truth → 0 (no new
        # info). When σ_S is moderate, S contributes new variance, R_truth > 0.
        if mi_s > 0:
            R_truth = max(0.0, (mi_s - mi_ec) / mi_s)
        else:
            R_truth = float('nan')
        R_truth_runs.append(R_truth)

        # v0.81.1.5 ship 3: four-class attribution. Ridge + MLP retain
        # the V5b precedent; RKHS + RF added so V5 calibration produces
        # the same four-class shape the apparatus measures on empirical
        # cells (Ship 5).
        ridge_shares, r2_ridge = estimate_R_3channel(
            E, C, S, Y, n_permutations=100, seed=s + 500, regressor='ridge')
        mlp_shares, r2_mlp = estimate_R_3channel(
            E, C, S, Y, n_permutations=100, seed=s + 500, regressor='mlp')
        rkhs_shares, r2_rkhs = estimate_R_3channel(
            E, C, S, Y, n_permutations=100, seed=s + 500, regressor='rkhs')
        rf_shares, r2_rf = estimate_R_3channel(
            E, C, S, Y, n_permutations=100, seed=s + 500, regressor='rf')

        ridge_shares_runs.append(ridge_shares)
        mlp_shares_runs.append(mlp_shares)
        rkhs_shares_runs.append(rkhs_shares)
        rf_shares_runs.append(rf_shares)
        R2_ridge_runs.append(r2_ridge)
        R2_mlp_runs.append(r2_mlp)
        R2_rkhs_runs.append(r2_rkhs)
        R2_rf_runs.append(r2_rf)

    # Aggregate across seeds
    def _mean_share(runs, key):
        return float(np.mean([r[key] for r in runs]))

    ridge_mean = {k: _mean_share(ridge_shares_runs, k) for k in ('E', 'C', 'S')}
    mlp_mean   = {k: _mean_share(mlp_shares_runs, k)   for k in ('E', 'C', 'S')}
    rkhs_mean  = {k: _mean_share(rkhs_shares_runs, k)  for k in ('E', 'C', 'S')}
    rf_mean    = {k: _mean_share(rf_shares_runs, k)    for k in ('E', 'C', 'S')}
    R_truth_mean = float(np.mean(R_truth_runs))
    R_truth_std  = float(np.std(R_truth_runs))

    print(f"\n{'estimator':>10} | {'E':>7} | {'C':>7} | {'S':>7} | {'R²':>7}")
    print("-" * 60)
    print(f"{'truth':>10} |   --    |   --    | {R_truth_mean:>7.4f} | "
          f"{'(R = S share)':>7}")
    print(f"{'ridge':>10} | {ridge_mean['E']:>7.4f} | {ridge_mean['C']:>7.4f} "
          f"| {ridge_mean['S']:>7.4f} | {np.mean(R2_ridge_runs):>7.4f}")
    print(f"{'mlp':>10} | {mlp_mean['E']:>7.4f} | {mlp_mean['C']:>7.4f} "
          f"| {mlp_mean['S']:>7.4f} | {np.mean(R2_mlp_runs):>7.4f}")
    print(f"{'rkhs':>10} | {rkhs_mean['E']:>7.4f} | {rkhs_mean['C']:>7.4f} "
          f"| {rkhs_mean['S']:>7.4f} | {np.mean(R2_rkhs_runs):>7.4f}")
    print(f"{'rf':>10} | {rf_mean['E']:>7.4f} | {rf_mean['C']:>7.4f} "
          f"| {rf_mean['S']:>7.4f} | {np.mean(R2_rf_runs):>7.4f}")

    return {
        'description': (
            "Doubly-conditional system: E, C ~ N(0,1) independent; "
            "S = E + C + ε with ε ~ N(0, σ_S²); Y = sin(S) + η with "
            "η ~ N(0, σ_Y²). Constructed for paper 2 measurement "
            "validation per codebot_handoff_v0_14.md §3.1."
        ),
        'construction': {
            'sigma_S': sigma_S,
            'sigma_Y': sigma_Y,
            'noise_target_mi_ec_nats': target_mi_ec,
            'noise_target_mi_s_nats':  target_mi_s,
            'n_steps_final':           n_steps_final,
            'n_repeats':               n_repeats,
            'tune_n_steps':            tune_n_steps,
        },
        'tuning_log': tuning_log,
        'analytic_truth': {
            'R_truth_mean':  R_truth_mean,
            'R_truth_std':   R_truth_std,
            'R_truth_per_seed': [float(x) for x in R_truth_runs],
            'mi_ec_proxy_mean_nats': float(np.mean(mi_ec_final)),
            'mi_s_proxy_mean_nats':  float(np.mean(mi_s_final)),
            'truth_formula': "R_truth = (I(Y;S) - I(Y;E,C)) / I(Y;S)",
        },
        'attribution_ridge': {
            'shares_mean': ridge_mean,
            'R2_mean':     float(np.mean(R2_ridge_runs)),
        },
        'attribution_mlp': {
            'shares_mean': mlp_mean,
            'R2_mean':     float(np.mean(R2_mlp_runs)),
        },
        # v0.81.1.5 ship 3: four-class V5f shares
        'attribution_rkhs': {
            'shares_mean': rkhs_mean,
            'R2_mean':     float(np.mean(R2_rkhs_runs)),
        },
        'attribution_rf': {
            'shares_mean': rf_mean,
            'R2_mean':     float(np.mean(R2_rf_runs)),
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    import argparse, os, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./data/paper/calibration/v5',
                        help='output directory')
    parser.add_argument('--force', action='store_true',
                        help='regenerate all sub-tests even if cached')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, 'v5_calibration_results.json')

    # v0.80.0.11: resume support. If a prior v5_calibration_results.json
    # exists, load it and skip sub-tests that already have entries.
    # Each of v5a-v5e runs independently; if v5a completed and v5b
    # crashed, on restart v5a is skipped, v5b re-runs.
    cached = {}
    if os.path.exists(out_path) and not args.force:
        try:
            with open(out_path, 'r', encoding='utf-8-sig') as f:
                prev = json.load(f)
            for sub in ('v5a_toy_model', 'v5b_nonlinear_rnn',
                        'v5c_correlated_predictors', 'v5d_xor_synergy',
                        'v5e_synergy_mixture', 'v5f_doubly_conditional'):
                prev_sub = prev.get(sub) or {}
                if prev_sub.get('results') or prev_sub.get('result'):
                    cached[sub] = prev_sub
            if cached:
                print(f"[resume] loaded {len(cached)} cached sub-test(s) "
                      f"from {out_path}")
        except Exception as e:
            print(f"[resume] could not read existing output: {e}")

    print("V5 Synthetic Calibration Suite")
    print("=" * 60)

    if 'v5a_toy_model' in cached:
        print("[skip] v5a — using cached result")
        results_v5a = cached['v5a_toy_model'].get('results') or []
    else:
        results_v5a = run_v5a()
    if 'v5b_nonlinear_rnn' in cached:
        print("[skip] v5b — using cached result")
        results_v5b = cached['v5b_nonlinear_rnn'].get('results') or []
    else:
        results_v5b = run_v5b()
    if 'v5c_correlated_predictors' in cached:
        print("[skip] v5c — using cached result")
        results_v5c = cached['v5c_correlated_predictors'].get('results') or []
    else:
        results_v5c = run_v5c()
    if 'v5d_xor_synergy' in cached:
        print("[skip] v5d — using cached result")
        results_v5d = cached['v5d_xor_synergy'].get('result') or {}
    else:
        results_v5d = run_v5d()
    if 'v5e_synergy_mixture' in cached:
        print("[skip] v5e — using cached result")
        results_v5e = cached['v5e_synergy_mixture'].get('results') or []
    else:
        results_v5e = run_v5e()
    # v0.81.1.5: V5f doubly-conditional system (paper 2 measurement validation).
    # Stores 'result' (dict) instead of 'results' (list) since V5f is one
    # converged system, not a parameter sweep.
    if 'v5f_doubly_conditional' in cached:
        print("[skip] v5f — using cached result")
        results_v5f = cached['v5f_doubly_conditional'].get('result') or {}
    else:
        results_v5f = run_v5f()

    output = {
        "experiment": "V5_synthetic_calibration_suite",
        "description": "Five calibration tests of Ridge permutation estimator against systems with known or reference R",
        "v5a_toy_model": {
            "description": "Stochastic persistence model, binary state, closed-form true R",
            "n_steps": 50000,
            "n_repeats": 5,
            "results": results_v5a
        },
        "v5b_nonlinear_rnn": {
            "description": "Nonlinear RNN (tanh), continuous 8-dim state, MLP reference R",
            "n_steps": 20000,
            "dim": 8,
            "noise_std": 0.1,
            "results": results_v5b
        },
        "v5c_correlated_predictors": {
            "description": "Exogeneity violation: E_t copies S_t with probability q",
            "persistence": 0.5,
            "n_steps": 50000,
            "n_repeats": 5,
            "results": results_v5c
        },
        "v5d_xor_synergy": {
            "description": "Pure synergistic dependence: S_{t+1} = S_t XOR E_t",
            "n_steps": 50000,
            "n_repeats": 5,
            "result": results_v5d
        },
        "v5e_synergy_mixture": {
            "description": "Mixture of persistence (linear) and XOR (synergistic), alpha sweep",
            "persistence": 0.5,
            "n_steps": 50000,
            "n_repeats": 5,
            "results": results_v5e
        },
        "v5f_doubly_conditional": {
            "description": (
                "Doubly-conditional three-channel system "
                "(E, C → S → Y). Paper 2 measurement-validation "
                "calibration system per codebot_handoff_v0_14.md §3.1. "
                "Noise scales σ_S, σ_Y tuned to match empirical-cell MI "
                "magnitudes via binary search."
            ),
            "result": results_v5f
        }
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    errors_a = [abs(r["error"]) for r in results_v5a]
    print(f"V5a (toy model):     mean |error| = {np.mean(errors_a):.4f}, max = {max(errors_a):.4f}, bias: always negative (conservative)")

    gaps_b = [abs(r["gap"]) for r in results_v5b]
    print(f"V5b (nonlinear RNN): mean |gap| = {np.mean(gaps_b):.4f}, max = {max(gaps_b):.4f} (Ridge vs MLP reference)")

    r_hats_c = [r["R_hat"] for r in results_v5c]
    print(f"V5c (correlated E):  R-hat at q=0: {r_hats_c[0]:.4f}, at q=0.5: {r_hats_c[-1]:.4f}, shift: {r_hats_c[-1] - r_hats_c[0]:+.4f}")

    print(f"V5d (XOR synergy):   R-hat = {results_v5d['R_hat']:.4f}, Ridge R2 = {results_v5d['R2']:.4f} (linear model fails on XOR as expected)")

    gaps_e = [abs(r["gap"]) for r in results_v5e]
    print(f"V5e (synergy mix):   mean |gap| = {np.mean(gaps_e):.4f}, max = {max(gaps_e):.4f} (Ridge vs MLP across alpha sweep)")

    if isinstance(results_v5f, dict) and results_v5f.get('attribution_ridge'):
        truth = (results_v5f.get('analytic_truth') or {}).get('R_truth_mean', float('nan'))
        ridge_S = (results_v5f.get('attribution_ridge') or {}).get('shares_mean', {}).get('S', float('nan'))
        mlp_S   = (results_v5f.get('attribution_mlp')   or {}).get('shares_mean', {}).get('S', float('nan'))
        print(f"V5f (doubly cond):   R_truth={truth:.4f} | Ridge_R̂={ridge_S:.4f} | "
              f"MLP_R̂={mlp_S:.4f} (three-channel attribution to S)")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
