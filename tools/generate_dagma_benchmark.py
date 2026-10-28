"""
External piecewise-stationary benchmark built from the DAGMA simulator.

Every regime is an independent draw of the linear-SEM benchmark of
Bello, Aragam and Ravikumar (NeurIPS 2022, Appendix C.1), using the
simulator shipped with the `dagma` package unchanged:

  graph   dagma.utils.simulate_dag(d, k*d, 'ER'|'SF')
  weights dagma.utils.simulate_parameter(B)      Unif[-2,-0.5] U [0.5,2]
  data    dagma.utils.simulate_linear_sem(W, n, 'gauss'|'exp'|'gumbel')  scale 1

The regimes are concatenated in time; nothing else is added (no observation
noise, no rescaling). The only choices made here are the ones the
stationary benchmark does not have: the number of regimes K and the regime
lengths, drawn uniformly in n*(1 -+ jitter) around the n = 1000 of the
original benchmark (jitter 0 gives regimes of exactly n samples).

Output layout is the one of generate_datasets.py (index.csv, -dat.csv,
-am.csv), so run_experiments.py and tools/run_cdnod.py read it unchanged.

  python tools/generate_dagma_benchmark.py --output_dir datasets_dagma
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from dagma import utils as du

GRAPHS = {"ER2": ("ER", 2), "ER4": ("ER", 4), "SF2": ("SF", 2), "SF4": ("SF", 4)}
SEMS = ("gauss", "exp", "gumbel")


def scenario_id(d, graph, sem, k, seed):
    return f"d{d}_{graph}_{sem}_r{k}_s{seed:03d}"


def generate_one(d, graph, sem, k, n, jitter, seed, output_dir):
    gtype, per_node = GRAPHS[graph]
    sizes = np.random.default_rng(seed ^ 0xA110C).integers(int(n * (1 - jitter)), int(n * (1 + jitter)) + 1, size=k)
    du.set_random_seed(seed)
    blocks, adjs = [], []
    for size in sizes:
        B = du.simulate_dag(d, per_node * d, gtype)
        W = du.simulate_parameter(B)
        blocks.append(du.simulate_linear_sem(W, int(size), sem))
        adjs.append((W != 0).astype(float))
    X = np.vstack(blocks)
    names = [f"V{i + 1}" for i in range(d)]
    sid = scenario_id(d, graph, sem, k, seed)

    pd.DataFrame(X, columns=names).to_csv(os.path.join(output_dir, f"{sid}-dat.csv"), index=False)
    am = []
    for r, A in enumerate(adjs):
        block = pd.DataFrame(A, index=names, columns=names)
        block.insert(0, "regime_idx", r)
        am.append(block)
    am_df = pd.concat(am)
    am_df.index.name = "from_var"
    am_df.to_csv(os.path.join(output_dir, f"{sid}-am.csv"))

    return {
        "scenario_id": sid,
        "p": d,
        "n_regimes": k,
        "n_changes": k - 1,
        "samples_regime": n,
        "base_pconn": round(per_node * d / (d * (d - 1) / 2), 4),
        "noise_fraction": 0.0,
        "seed": seed,
        "total_samples": len(X),
        "n_true_cps": k - 1,
        "change_points": ";".join(str(c) for c in np.cumsum(sizes)[:-1]),
        "regime_sizes": ";".join(str(int(x)) for x in sizes),
        "variable_names": ";".join(names),
        "min_samples_regime": int(sizes.min()),
        "graph": graph,
        "sem_type": sem,
        "mean_edges": float(np.mean([A.sum() for A in adjs])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default="datasets_dagma")
    # capped at the largest p of the main grid (12); ER4 needs d >= 9 (4d <= d(d-1)/2)
    ap.add_argument("--d", type=int, nargs="+", default=[10, 12])
    ap.add_argument("--graphs", nargs="+", default=list(GRAPHS))
    ap.add_argument("--sems", nargs="+", default=list(SEMS))
    ap.add_argument("--regimes", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--n", type=int, default=1000, help="samples per regime (n of the original benchmark)")
    ap.add_argument("--jitter", type=float, default=0.4, help="regime length drawn in n*(1 -+ jitter); 0 = exactly n")
    ap.add_argument("--n_seeds", type=int, default=10, help="repetitions per setting (10 in the original)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    grid = itertools.product(args.d, args.graphs, args.sems, args.regimes, range(args.n_seeds))
    for seed, (d, graph, sem, k, _) in enumerate(grid):  # one seed per scenario
        rows.append(generate_one(d, graph, sem, k, args.n, args.jitter, seed, args.output_dir))
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "index.csv"), index=False)
    print(f"{len(rows)} scenarios -> {args.output_dir}")


if __name__ == "__main__":
    main()
