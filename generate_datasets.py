"""
Dataset generator for Full Pipeline experiments.

Produces a reproducible grid of non-stationary causal time series and saves
them as CSV files that run_experiments.py can load without re-generating data.

Parameter grid (defaults — 17 280 scenarios)
---------------------------------------------
  p              : 3..10              (8 values)
  n_changes      : 1..6               (6 values — n_regimes = n_changes + 1 = 2..7)
  samples_regime : 500, 2500, 5000    (3 values — MAX samples per regime; each regime
                                       gets a random size in [MIN_REGIME_SAMPLES, samples_regime])
  base_pconn     : 0.20, 0.35, 0.50, 0.75   (4 values)
  noise_fraction : 0.02, 0.08, 0.20  (3 values — low / medium / high)
  seeds          : 10 per combination

Grid math: 8 × 6 × 3 × 4 × 3 × 10 = 17 280 scenarios
(matches the causalmorph SyntheticCausalScenarios_mixed_v2_5 reference at 17 280)

Constraint: samples_regime >= 500 for every regime (change-point
detection needs enough data per window; values below 500 are rejected).

Mapping to causalmorph reference parameters
-------------------------------------------
  p              ↔ num_vars
  n_changes      ↔ (no direct analog — unique to non-stationary setting)
  samples_regime ↔ nsamples
  base_pconn     ↔ pconn
  noise_fraction ↔ deviation  (structural noise level)
  seeds          ↔ repetitions

Output layout
-------------
  {output_dir}/index.csv               — one row per scenario (all scalar metadata)
  {output_dir}/{scenario_id}-dat.csv   — time series  (V1..Vp columns, T rows)
  {output_dir}/{scenario_id}-am.csv    — adjacency matrices stacked by regime
                                         columns: regime_idx + V1..Vp
                                         row index (from_var): cause variable

Scenario ID format
------------------
  p{p}_r{n_regimes}_n{samples}_pc{pconn_int:02d}_nf{noise_int:02d}_s{seed:06d}
  e.g.  p5_r3_n2500_pc35_nf08_s000042

Usage
-----
  python generate_datasets.py                      # full default grid (17 280 scenarios)
  python generate_datasets.py --p_min 3 --p_max 5 # restrict node range
  python generate_datasets.py --n_seeds 5          # fewer seeds per combo
  python generate_datasets.py --output_dir my_data # custom output directory
"""

import os
import sys
import time
import argparse
import traceback
import itertools

import numpy as np
import pandas as pd

MIN_REGIME_SAMPLES = 250  # hard floor on samples per regime

# ── Path setup (mirrors full_pipeline.py) ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(_HERE, "causalmorph"))

from full_pipeline import build_nonstationary_scenario


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_scenario_id(
    p: int,
    n_regimes: int,
    samples_regime: int,
    pconn: float,
    noise_fraction: float,
    seed: int,
) -> str:
    """Encode all parameters into a short, filesystem-safe identifier."""
    return (
        f"p{p}_r{n_regimes}_n{samples_regime}"
        f"_pc{int(pconn * 100):02d}"
        f"_nf{int(noise_fraction * 100):02d}"
        f"_s{seed:06d}"
    )


def generate_and_save(
    p: int,
    n_regimes: int,
    samples_regime: int,
    pconn: float,
    noise_fraction: float,
    seed: int,
    output_dir: str,
) -> dict:
    """
    Generate one scenario and write its CSV files.

    samples_regime is used for both min_samples and max_samples so every
    regime has an exact fixed length — matches the controlled design of the
    causalmorph reference (nsamples is exact, not a range).

    Returns an index-row dict on success, raises on failure.
    """
    sid = make_scenario_id(p, n_regimes, samples_regime, pconn, noise_fraction, seed)

    if samples_regime < MIN_REGIME_SAMPLES:
        raise ValueError(
            f"samples_regime={samples_regime} is below the minimum regime floor "
            f"({MIN_REGIME_SAMPLES})"
        )

    # Constrained random partition: sizes are random but sum exactly to
    # n_regimes * samples_regime, each regime >= MIN_REGIME_SAMPLES.
    # Uses stars-and-bars on the budget above the per-regime minimum.
    rng = np.random.default_rng(seed)
    total_samples = n_regimes * samples_regime
    budget = total_samples - n_regimes * MIN_REGIME_SAMPLES
    cuts = sorted(rng.integers(0, budget + 1, size=n_regimes - 1).tolist())
    boundaries = [0] + cuts + [budget]
    computed_sizes = [
        MIN_REGIME_SAMPLES + (boundaries[i + 1] - boundaries[i])
        for i in range(n_regimes)
    ]

    (
        X,
        true_cps,
        variable_names,
        regime_sizes,
        _scenario,
        _scenario_gen,
        true_adjs,
        _change_infos,
    ) = build_nonstationary_scenario(
        p=p,
        n_regimes=n_regimes,
        min_samples=MIN_REGIME_SAMPLES,
        max_samples=samples_regime,
        base_pconn=pconn,
        change_pcts=[0, 30, 25, 35, 30],
        seed=seed,
        noise_fraction=noise_fraction,
        regime_sizes=computed_sizes,
    )

    # ── dat.csv: full time series ─────────────────────────────────────────────
    dat_path = os.path.join(output_dir, f"{sid}-dat.csv")
    pd.DataFrame(X, columns=variable_names).to_csv(dat_path, index=False)

    # ── am.csv: per-regime adjacency matrices stacked ─────────────────────────
    # Each regime contributes p rows.  regime_idx is the first column.
    # Row index (from_var) is the cause variable; columns V1..Vp are effects.
    am_blocks = []
    for r_idx, adj in enumerate(true_adjs):
        adj_arr = np.asarray(adj.values if hasattr(adj, "values") else adj)
        block = pd.DataFrame(adj_arr, index=variable_names, columns=variable_names)
        block.insert(0, "regime_idx", r_idx)
        am_blocks.append(block)
    am_df = pd.concat(am_blocks)
    am_df.index.name = "from_var"
    am_path = os.path.join(output_dir, f"{sid}-am.csv")
    am_df.to_csv(am_path)

    return {
        "scenario_id":       sid,
        "p":                 p,
        "n_regimes":         n_regimes,
        "n_changes":         n_regimes - 1,
        "samples_regime":    samples_regime,
        "base_pconn":        pconn,
        "noise_fraction":    noise_fraction,
        "seed":              seed,
        "total_samples":     len(X),
        "n_true_cps":        len(true_cps),
        "change_points":     ";".join(map(str, true_cps)),
        "regime_sizes":      ";".join(map(str, regime_sizes)),
        "variable_names":    ";".join(variable_names),
        "min_samples_regime": MIN_REGIME_SAMPLES,   # floor used; run_experiments uses this for min_window
    }


# ── Build the parameter grid ──────────────────────────────────────────────────

def build_grid(
    p_min: int,
    p_max: int,
    n_changes_min: int,
    n_changes_max: int,
    samples_list,
    pconn_list,
    noise_list,
    n_seeds: int,
    base_seed: int,
) -> list:
    """
    Return a flat list of dicts, one per scenario, with deterministic seeds.

    Iteration order is fixed so seeds never collide even if the grid is
    regenerated with different bounds.
    """
    p_list         = list(range(p_min, p_max + 1))
    n_regimes_list = list(range(n_changes_min + 1, n_changes_max + 2))

    combos = list(itertools.product(
        p_list,
        n_regimes_list,
        sorted(samples_list),
        sorted(pconn_list),
        sorted(noise_list),
    ))

    rows = []
    for combo_idx, (p, n_regimes, samples, pconn, noise) in enumerate(combos):
        for seed_num in range(n_seeds):
            seed = base_seed + combo_idx * n_seeds + seed_num
            rows.append({
                "p":              p,
                "n_regimes":      n_regimes,
                "samples_regime": samples,
                "pconn":          pconn,
                "noise_fraction": noise,
                "seed":           seed,
            })
    return rows


# ── Main batch generator ──────────────────────────────────────────────────────

def run_generate(
    p_min: int           = 3,
    p_max: int           = 10,
    n_changes_min: int   = 1,
    n_changes_max: int   = 6,
    samples_list         = (500, 2500, 5000),
    pconn_list           = (0.20, 0.35, 0.50, 0.75),
    noise_list           = (0.02, 0.08, 0.20),
    n_seeds: int         = 10,
    base_seed: int       = 0,
    output_dir: str      = "datasets",
):
    """
    Generate all scenarios in the parameter grid and save to output_dir.

    Already-existing scenario files are skipped — safe to resume after
    interruption.  Total scenarios = combinations × n_seeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "index.csv")

    grid  = build_grid(
        p_min, p_max, n_changes_min, n_changes_max,
        samples_list, pconn_list, noise_list, n_seeds, base_seed,
    )
    total    = len(grid)
    n_combos = total // n_seeds

    print("=" * 70)
    print("Dataset Generator — Full Pipeline")
    print("=" * 70)
    print(f"  p                : {p_min}..{p_max}  ({p_max - p_min + 1} values)")
    print(f"  n_changes        : {n_changes_min}..{n_changes_max}  "
          f"({n_changes_max - n_changes_min + 1} values → n_regimes {n_changes_min+1}..{n_changes_max+1})")
    print(f"  samples/regime   : {sorted(samples_list)}")
    print(f"  base_pconn       : {sorted(pconn_list)}")
    print(f"  noise_fraction   : {sorted(noise_list)}")
    print(f"  seeds per combo  : {n_seeds}  (base_seed={base_seed})")
    print(f"  combinations     : {n_combos}")
    print(f"  total scenarios  : {total}")
    print(f"  output           : {output_dir}/")
    print()

    # Load existing index
    if os.path.exists(index_path):
        existing_index = pd.read_csv(index_path, dtype=str)
        done_ids = set(existing_index["scenario_id"].tolist())
        remaining = total - sum(
            1 for r in grid
            if make_scenario_id(
                r["p"], r["n_regimes"], r["samples_regime"],
                r["pconn"], r["noise_fraction"], r["seed"],
            ) in done_ids
        )
        print(f"  Resuming: {len(done_ids)} already done, {remaining} remaining\n")
    else:
        existing_index = pd.DataFrame()
        done_ids = set()

    new_rows = []
    errors   = 0
    t0       = time.time()

    for idx, params in enumerate(grid, start=1):
        p              = params["p"]
        n_regimes      = params["n_regimes"]
        samples        = params["samples_regime"]
        pconn          = params["pconn"]
        noise          = params["noise_fraction"]
        seed           = params["seed"]

        sid     = make_scenario_id(p, n_regimes, samples, pconn, noise, seed)
        elapsed = time.time() - t0
        prefix  = f"  [{idx:>6}/{total}]  {sid}  elapsed={elapsed:.0f}s"

        if sid in done_ids:
            print(f"{prefix}  SKIP")
            continue

        print(prefix, end="  ", flush=True)

        try:
            row = generate_and_save(
                p=p,
                n_regimes=n_regimes,
                samples_regime=samples,
                pconn=pconn,
                noise_fraction=noise,
                seed=seed,
                output_dir=output_dir,
            )
            new_rows.append(row)
            print(f"T={row['total_samples']}  cps=[{row['change_points']}]")
        except Exception:
            errors += 1
            msg = traceback.format_exc(limit=2).strip().splitlines()[-1]
            print(f"ERROR: {msg}")

    # Write updated index
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if not existing_index.empty:
            df_index = pd.concat([existing_index, df_new], ignore_index=True)
        else:
            df_index = df_new
        df_index.to_csv(index_path, index=False)
        print(f"\nIndex written → {index_path}  ({len(df_index)} total scenarios)")
    elif done_ids:
        print(f"\nAll scenarios already exist — nothing new written.")
    else:
        print(f"\nNo scenarios generated (check errors above).")

    elapsed_total = time.time() - t0
    print(f"Finished in {elapsed_total:.0f}s  ({errors} errors)")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate non-stationary causal datasets for batch experiments.\n"
            "Default grid: 8×6×3×4×3×10 = 17 280 scenarios."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--p_min",          type=int,   default=3,
                        help="Min nodes (default 3)")
    parser.add_argument("--p_max",          type=int,   default=10,
                        help="Max nodes inclusive (default 10)")
    parser.add_argument("--n_changes_min",  type=int,   default=1,
                        help="Min change points per scenario (default 1)")
    parser.add_argument("--n_changes_max",  type=int,   default=6,
                        help="Max change points per scenario (default 6)")
    parser.add_argument("--samples",        type=int,   nargs="+",
                        default=[500, 2500, 5000],
                        help="Exact samples per regime (default: 500 2500 5000)")
    parser.add_argument("--pconn",          type=float, nargs="+",
                        default=[0.20, 0.35, 0.50, 0.75],
                        help="Edge probabilities (default: 0.20 0.35 0.50 0.75)")
    parser.add_argument("--noise",          type=float, nargs="+",
                        default=[0.02, 0.08, 0.20],
                        help="Pink noise fraction values (default: 0.02 0.08 0.20)")
    parser.add_argument("--n_seeds",        type=int,   default=10,
                        help="Seeds per parameter combination (default 10)")
    parser.add_argument("--base_seed",      type=int,   default=0,
                        help="Starting seed offset (default 0)")
    parser.add_argument("--output_dir",     type=str,   default="datasets",
                        help="Output directory (default: datasets/)")
    args = parser.parse_args()

    bad = [s for s in args.samples if s < 500]
    if bad:
        parser.error(f"--samples values must be >= 500 (got {bad})")

    n_combos = (
        (args.p_max - args.p_min + 1)
        * (args.n_changes_max - args.n_changes_min + 1)
        * len(args.samples)
        * len(args.pconn)
        * len(args.noise)
    )
    print(f"Grid: {n_combos} combinations × {args.n_seeds} seeds = {n_combos * args.n_seeds} scenarios")

    run_generate(
        p_min          = args.p_min,
        p_max          = args.p_max,
        n_changes_min  = args.n_changes_min,
        n_changes_max  = args.n_changes_max,
        samples_list   = args.samples,
        pconn_list     = args.pconn,
        noise_list     = args.noise,
        n_seeds        = args.n_seeds,
        base_seed      = args.base_seed,
        output_dir     = args.output_dir,
    )
