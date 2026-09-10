"""
_derive_h4_threshold.py -- re-derive the H4 synergy-gate threshold tau_H4 = 0.72.

Paper B (§5.7, App E) states: tau_H4 is V5e's plug-in MI synergy ratio at the
mixture midpoint alpha = 0.5, obtained by linear interpolation between the
alpha = 0.4 ratio (0.587) and the alpha = 0.6 ratio (0.852). Until 2026-09-09
neither the two ratios nor the 0.72 were persisted anywhere on disk; the
appendix-data H4 gate entry pointed at a CHANGELOG entry that does not
contain them. This script regenerates all three numbers from the V5e
construction in v5_synthetic_calibration.run_v5e (same seeds, same n_steps,
same three repeats) and also reports the closed-form values.

Convention: the ratio is SYNERGY-POSITIVE, -II / I(S_{t+1}; S_t, E_t) with
II = I(S_{t+1}; S_t) + I(S_{t+1}; E_t) - I(S_{t+1}; S_t, E_t) (redundancy-positive
interaction information, the convention results8th.json's ii_fraction_knn /
redundancy_fraction_knn use). A fleet cell is compared against tau_H4 after
the sign flip; see Paper B §4.3 / §6.2.

Output: data/paper/calibration/v5/h4_threshold_derivation.json
Usage:  py _derive_h4_threshold.py
"""
import os
import json
import datetime

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT, 'data', 'paper', 'calibration', 'v5',
                        'h4_threshold_derivation.json')

N_STEPS = 20000          # run_v5e
N_REPEATS = 3            # run_v5e
SEEDS = [42 + rep * 1000 for rep in range(N_REPEATS)]   # run_v5e
P_PERSIST = 0.5          # run_v5e
ALPHAS = [0.4, 0.5, 0.6]
ALPHA_CRIT = 0.5


def _h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def closed_form(alpha):
    """Exact plug-in ratios for the V5e transition kernel with S_t, E_t iid Bernoulli(1/2).
    P(S'=1 | S,E): (0,0)->0, (1,1)->1-alpha, (0,1),(1,0)->(1+alpha)/2; marginal P(S'=1)=1/2."""
    cond = [0.0, 1 - alpha, (1 + alpha) / 2, (1 + alpha) / 2]
    I_SE = 1.0 - float(np.mean([_h2(c) for c in cond]))
    I_S = 1.0 - 0.5 * (_h2((1 + alpha) / 4) + _h2((3 - alpha) / 4))
    I_E = I_S
    II = I_S + I_E - I_SE
    return {'I_S': I_S, 'I_E': I_E, 'I_joint': I_SE,
            'II_redundancy_positive': II,
            'synergy_ratio': -II / I_SE}


def _mi_binary(x, y):
    mi = 0.0
    for a in (0, 1):
        for b in (0, 1):
            pab = float(np.mean((x == a) & (y == b)))
            if pab > 0:
                mi += pab * np.log2(pab / (float(np.mean(x == a)) * float(np.mean(y == b))))
    return mi


def _mi_joint(s, e, y):
    j = s * 2 + e
    mi = 0.0
    for v in range(4):
        for b in (0, 1):
            pj = float(np.mean((j == v) & (y == b)))
            if pj > 0:
                mi += pj * np.log2(pj / (float(np.mean(j == v)) * float(np.mean(y == b))))
    return mi


def sample_v5e(alpha, seed):
    """Byte-for-byte the generator in v5_synthetic_calibration.run_v5e."""
    rng = np.random.RandomState(seed)
    E = rng.randint(0, 2, size=N_STEPS)
    S = np.zeros(N_STEPS + 1, dtype=int)
    S[0] = rng.randint(0, 2)
    for t in range(N_STEPS):
        if rng.random() < alpha:
            S[t + 1] = S[t] ^ E[t]
        else:
            if rng.random() < P_PERSIST:
                S[t + 1] = S[t]
            else:
                S[t + 1] = E[t]
    return S[:-1], E, S[1:]


def plug_in(alpha):
    per_seed = []
    for seed in SEEDS:
        s, e, y = sample_v5e(alpha, seed)
        I_S, I_E, I_SE = _mi_binary(s, y), _mi_binary(e, y), _mi_joint(s, e, y)
        II = I_S + I_E - I_SE
        per_seed.append({'seed': seed, 'I_S': I_S, 'I_E': I_E, 'I_joint': I_SE,
                         'II_redundancy_positive': II, 'synergy_ratio': -II / I_SE})
    return {'per_seed': per_seed,
            'synergy_ratio_mean': float(np.mean([p['synergy_ratio'] for p in per_seed]))}


def main():
    out = {
        'experiment': 'H4_threshold_derivation',
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'convention': 'synergy-positive: -II / I(S_next; S, E), II = I(S_next;S) + I(S_next;E) - I(S_next;S,E)',
        'construction': 'V5e synergy mixture, v5_synthetic_calibration.run_v5e (n_steps 20000, 3 repeats, seeds 42/1042/2042, p_persist 0.5)',
        'alpha_crit': ALPHA_CRIT,
        'closed_form': {str(a): closed_form(a) for a in ALPHAS},
        'plug_in': {str(a): plug_in(a) for a in ALPHAS},
    }
    r04 = out['plug_in']['0.4']['synergy_ratio_mean']
    r06 = out['plug_in']['0.6']['synergy_ratio_mean']
    interp = 0.5 * (r04 + r06)
    out['tau_H4'] = {
        'ratio_alpha_0.4_plug_in': round(r04, 3),
        'ratio_alpha_0.6_plug_in': round(r06, 3),
        'linear_interpolation_at_alpha_0.5': interp,
        'threshold_value': round(interp, 2),
        'direct_plug_in_at_alpha_0.5': out['plug_in']['0.5']['synergy_ratio_mean'],
        'closed_form_at_alpha_0.5': out['closed_form']['0.5']['synergy_ratio'],
        'note': ('Paper B §5.7 / App E state the interpolated value 0.72. The direct '
                 'alpha=0.5 ratio is 0.73; the paper uses the interpolation. '
                 'Fleet comparison: results8th.json redundancy_fraction_knn is '
                 'redundancy-positive; negate before comparing with tau_H4.'),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"alpha 0.4 plug-in ratio {r04:.4f} (closed form {out['closed_form']['0.4']['synergy_ratio']:.4f})")
    print(f"alpha 0.6 plug-in ratio {r06:.4f} (closed form {out['closed_form']['0.6']['synergy_ratio']:.4f})")
    print(f"interpolated at alpha 0.5: {interp:.4f} -> tau_H4 = {round(interp, 2)}")
    print(f"direct alpha 0.5 plug-in {out['tau_H4']['direct_plug_in_at_alpha_0.5']:.4f}, closed form {out['tau_H4']['closed_form_at_alpha_0.5']:.4f}")
    print(f"Wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
