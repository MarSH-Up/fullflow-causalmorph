#!/usr/bin/env python3
"""
Stage 0 with the Azadkia-Chatterjee dependence measure (xi) as the sink score instead of
HSIC, on the true regime windows (standardised, permuted) of the dataset directories given as
name=dir: nSHD of Stage 0's DAG, orientation accuracy on the true edges and seconds per window.

  OMP_NUM_THREADS=1 python tools/xi_score_check.py --workers 10 name=dir ...
"""
import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph")); sys.path.insert(0, os.path.join(ROOT, "tools"))
from run_experiments import load_scenario  # noqa: E402
from full_pipeline import _standardize_window, compute_shd  # noqa: E402
from core.stage0 import stage0  # noqa: E402
from causalmorph_diagnosis import orient_acc  # noqa: E402


def one(args):
    name, ds, sid = args
    index = pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).set_index("scenario_id")
    X, names, cps, adjs, _, _ = load_scenario(sid, ds, index.loc[sid], permute_variables=True)
    bounds = [0] + list(cps) + [len(X)]
    rows = []
    for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        W = _standardize_window(pd.DataFrame(X[a:b], columns=names)).to_numpy(float); T = adjs[r].to_numpy(float)
        row = {"cell": name, "scenario_id": sid, "regime": r, "n": len(W)}
        for sc in ("hsic", "xi"):
            t0 = time.time(); A, _ = stage0(W, regressor="mlp", seed=r, score_kind=sc); row[f"t_{sc}"] = time.time() - t0
            row[sc] = compute_shd(A, T)["normalized_shd"]; row[f"{sc}_ok"], row[f"{sc}_tot"] = orient_acc(T, A)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cells", nargs="+"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "xi_score_check.csv"))
    a = ap.parse_args()
    jobs = []
    for c in a.cells:
        name, ds = c.split("=", 1)
        jobs += [(name, ds, s) for s in pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).scenario_id]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    D = pd.DataFrame(rows); D.to_csv(a.out, index=False)
    g = D.groupby("cell", sort=False)
    out = g[["hsic", "xi"]].mean().round(3)
    out["orient hsic"] = (g.hsic_ok.sum() / g.hsic_tot.sum()).round(2); out["orient xi"] = (g.xi_ok.sum() / g.xi_tot.sum()).round(2)
    out["s/window hsic"] = g.t_hsic.mean().round(2); out["s/window xi"] = g.t_xi.mean().round(2)
    out["xi - hsic"] = g.apply(lambda x: f"{(x.xi - x.hsic).mean():+.3f} ({(x.xi < x.hsic).mean():.2f}/{(x.xi > x.hsic).mean():.2f})")
    pd.set_option("display.width", 200); print("Stage 0 nSHD per window, sink score HSIC vs xi\n" + out.to_string())


if __name__ == "__main__":
    main()
