"""
IOTA FRAMEWORK — Run 0041 Sweep Reader  v0.75.0.6
=================================================
Reads Q45_pool_dim_sweep.json from every temperature directory.
Reports exactly what the JSONs contain. Read-only.

Usage:
    python read_r45.py
    python read_r45.py --path /path/to/data/llama/8b_4bit/abliterated
"""

import os, sys, json


def _find_root():
    """Walk up from this file's directory until we find the iota root
    (directory containing start_here.py). Lets the script run from anywhere."""
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(candidate, "start_here.py")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.dirname(os.path.abspath(__file__))


def main():
    """Print Run 0041 (POOL_DIM calibration sweep) results for every
    temperature in a single stacked table.

    Reads Q45_pool_dim_sweep.json from each temp dir under the active
    model's variant root. Produces tables of R fraction / E fraction /
    C fraction / interaction mass / R²-full-model at each projection
    dimension tested. Followed by per-temperature summary: dim_95
    (smallest dim retaining 95% of peak R), R*, R_max, dim_star, and
    the JSON's own verdict string.

    Read-only. Never modifies disk. This is the reference view the
    paper's FIG06 (dim_95 curve) was derived from.

    Usage:
      python read_r45.py
      python read_r45.py --path /data/llama/8b_4bit/abliterated
    """
    root = _find_root()
    base_dir = None

    if '--path' in sys.argv:
        idx = sys.argv.index('--path')
        if idx + 1 < len(sys.argv):
            base_dir = sys.argv[idx + 1]

    if base_dir is None:
        sess_path = os.path.join(root, 'last_session.json')
        if os.path.exists(sess_path):
            with open(sess_path) as f:
                s = json.load(f)
            family  = s.get('model_family', 'llama')
            size    = s.get('model_size', '8b')
            variant = s.get('model_variant', 'abliterated')
            from cartography import DATA, _size_dir, set_active_quant
            set_active_quant(s.get('quantization', '4bit'))
            base_dir = os.path.join(DATA, family, _size_dir(size), variant)

    if base_dir is None or not os.path.isdir(base_dir):
        print(f"  Cannot find data directory: {base_dir}")
        print(f"  Use: python read_r45.py --path /path/to/data/family/size_quant/variant")
        return

    print(f"\n  Reading Run 0041 results from: {base_dir}\n")

    _COND_TEMPS = {
        'deterministic': 0.0, 'temp_0.2': 0.2, 'temp_0.4': 0.4,
        'temp_0.6': 0.6, 'temp_0.8': 0.8, 'temp_1.0': 1.0,
        'temp_1.2': 1.2, 'temp_1.4': 1.4, 'temp_1.6': 1.6,
        'temp_1.8': 1.8, 'temp_2.0': 2.0,
    }

    all_temps = []

    for cond in sorted(os.listdir(base_dir)):
        if cond == 'pooled':
            continue
        q45_path = os.path.join(base_dir, cond, 'analysis', 'Q0041_pool_dim_sweep.json')
        if not os.path.exists(q45_path):
            continue

        temp = _COND_TEMPS.get(cond)
        if temp is None:
            try:
                temp = float(cond.replace('temp_', ''))
            except ValueError:
                temp = -1.0

        with open(q45_path) as f:
            d = json.load(f)

        sweep = d.get('results', [])
        if not sweep:
            continue

        all_temps.append({
            'temp': temp,
            'cond': cond,
            'optimal_dim': d.get('optimal_pool_dim'),
            'dim_95': d.get('dim_95'),
            'R_at_dim95': d.get('R_at_dim95'),
            'R_max': d.get('R_max'),
            'dim_star': d.get('dim_star'),
            'verdict': d.get('verdict', ''),
            'r_range': d.get('r_fraction_range'),
            'r_std': d.get('r_fraction_std'),
            'sweep': sweep,
        })

    if not all_temps:
        print("  No Q45_pool_dim_sweep.json files found.")
        return

    all_temps.sort(key=lambda r: r['temp'])

    # ── R fraction at each dimension ──────────────────────────────
    dims_all = sorted({r['pool_dim'] for t in all_temps for r in t['sweep'] if 'error' not in r})

    print(f"  R fraction at each dimension")
    print(f"  {'T':>4}  " + "  ".join(f"{d:>7}" for d in dims_all) + "   optimal")
    print(f"  {'─'*4}  " + "  ".join('─'*7 for _ in dims_all) + "   {'─'*7}")

    for t in all_temps:
        dim_map = {r['pool_dim']: r for r in t['sweep'] if 'error' not in r}
        cells = []
        for d in dims_all:
            r = dim_map.get(d)
            if r is None:
                cells.append(f"{'—':>7}")
            else:
                cells.append(f"{r['frac_R']:7.3f}")
        print(f"  {t['temp']:4.1f}  " + "  ".join(cells) + f"   {t['optimal_dim']:>7}")

    # ── E fraction ────────────────────────────────────────────────
    print(f"\n  E fraction at each dimension")
    print(f"  {'T':>4}  " + "  ".join(f"{d:>7}" for d in dims_all))
    print(f"  {'─'*4}  " + "  ".join('─'*7 for _ in dims_all))

    for t in all_temps:
        dim_map = {r['pool_dim']: r for r in t['sweep'] if 'error' not in r}
        cells = []
        for d in dims_all:
            r = dim_map.get(d)
            if r is None:
                cells.append(f"{'—':>7}")
            else:
                cells.append(f"{r['frac_E']:7.3f}")
        print(f"  {t['temp']:4.1f}  " + "  ".join(cells))

    # ── C fraction ────────────────────────────────────────────────
    print(f"\n  C fraction at each dimension")
    print(f"  {'T':>4}  " + "  ".join(f"{d:>7}" for d in dims_all))
    print(f"  {'─'*4}  " + "  ".join('─'*7 for _ in dims_all))

    for t in all_temps:
        dim_map = {r['pool_dim']: r for r in t['sweep'] if 'error' not in r}
        cells = []
        for d in dims_all:
            r = dim_map.get(d)
            if r is None:
                cells.append(f"{'—':>7}")
            else:
                cells.append(f"{r['frac_C']:7.3f}")
        print(f"  {t['temp']:4.1f}  " + "  ".join(cells))

    # ── Interaction mass ──────────────────────────────────────────
    print(f"\n  Interaction mass at each dimension")
    print(f"  {'T':>4}  " + "  ".join(f"{d:>7}" for d in dims_all))
    print(f"  {'─'*4}  " + "  ".join('─'*7 for _ in dims_all))

    for t in all_temps:
        dim_map = {r['pool_dim']: r for r in t['sweep'] if 'error' not in r}
        cells = []
        for d in dims_all:
            r = dim_map.get(d)
            if r is None:
                cells.append(f"{'—':>7}")
            else:
                cells.append(f"{r['interaction_mass']:7.1f}")
        print(f"  {t['temp']:4.1f}  " + "  ".join(cells))

    # ── R² full model ─────────────────────────────────────────────
    print(f"\n  R-squared (full model) at each dimension")
    print(f"  {'T':>4}  " + "  ".join(f"{d:>7}" for d in dims_all))
    print(f"  {'─'*4}  " + "  ".join('─'*7 for _ in dims_all))

    for t in all_temps:
        dim_map = {r['pool_dim']: r for r in t['sweep'] if 'error' not in r}
        cells = []
        for d in dims_all:
            r = dim_map.get(d)
            if r is None:
                cells.append(f"{'—':>7}")
            else:
                cells.append(f"{r['r2_full']:7.4f}")
        print(f"  {t['temp']:4.1f}  " + "  ".join(cells))

    # ── JSON-reported summary per temperature ─────────────────────
    print(f"\n\n  JSON-reported summary per temperature\n")
    for t in all_temps:
        dim95 = t.get('dim_95')
        r_star = t.get('R_at_dim95')
        r_max = t.get('R_max')
        d_star = t.get('dim_star')
        # dim₉₅ fields only exist in v0.65+ JSONs
        if dim95 is not None:
            print(f"  T={t['temp']:.1f}  dim₉₅={dim95}  R*={r_star:.3f}  "
                  f"R_max={r_max:.3f} at dim={d_star}")
        else:
            print(f"  T={t['temp']:.1f}  optimal_pool_dim={t['optimal_dim']}  "
                  f"(pre-v0.65 — no dim₉₅)")
        print(f"         R_range={t['r_range']:.4f}  R_std={t['r_std']:.4f}")
        print(f"         {t['verdict']}")
        print()

    print()


if __name__ == '__main__':
    main()
