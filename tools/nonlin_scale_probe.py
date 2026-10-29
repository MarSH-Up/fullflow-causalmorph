#!/usr/bin/env python3
"""
What the preconditioning buys when the nonlinearity is active. On the true regime
windows (standardised, variables permuted) of the scenarios in each dataset directory
given: nSHD of DirectLiNGAM on the window (dl), after CausalMorph v2cap (v2), after
v2cap with the polynomial residual kept in Stage I (v2poly), PC (CPDAG scored as in
the pipeline), and the empty graph; the orientation accuracy of dl on the true edges;
and how nonlinear the mechanism is (R² gain of a cubic over a linear fit on the true
parents).

  OMP_NUM_THREADS=1 python tools/nonlin_scale_probe.py --workers 10 name1=dir1 name2=dir2 ...
"""
import argparse
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph")); sys.path.insert(0, os.path.join(ROOT, "tools"))
from run_experiments import load_scenario  # noqa: E402
from full_pipeline import _to_causalmorph, _standardize_window, compute_shd  # noqa: E402
from core.causalmorph_v2 import causalmorph_v2  # noqa: E402
from causalmorph_diagnosis import v2poly, dl_fit, orient_acc  # noqa: E402
from orienter_tuning import pc_skeleton  # noqa: E402


def pc_cpdag(X):
    import contextlib, io
    from causallearn.search.ConstraintBased.PC import pc
    from cdnod_baseline import cdnod_to_adjacency
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        cg = pc(X, alpha=0.05, indep_test="fisherz", show_progress=False)
    return cdnod_to_adjacency(cg.G.graph, X.shape[1])[0].astype(float)


def one(args):
    name, ds, sid = args
    index = pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).set_index("scenario_id")
    X, names, cps, adjs, _, _ = load_scenario(sid, ds, index.loc[sid], permute_variables=True)
    bounds = [0] + list(cps) + [len(X)]
    rows = []
    for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        W = _standardize_window(pd.DataFrame(X[a:b], columns=names)); V = W.to_numpy(float)
        T = adjs[r].to_numpy(float)
        row = {"cell": name, "scenario_id": sid, "regime": r, "n": len(W), "p": len(names)}
        gains = []
        for j in range(len(names)):
            pa = np.where(T[:, j] != 0)[0]
            if len(pa):
                P, y = V[:, pa], V[:, j]
                gains.append(make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(P, y).score(P, y) - LinearRegression().fit(P, y).score(P, y))
        row["r2_gain_cubic"] = float(np.mean(gains)) if gains else np.nan
        B, order = dl_fit(W)
        est = (np.abs(B) > 0.05).astype(float)
        row["dl"] = compute_shd(B, T)["normalized_shd"]
        row["dl_ok"], row["dl_tot"] = orient_acc(T, est)
        prior = _to_causalmorph(pd.DataFrame(est, index=names, columns=names))
        Z, det = causalmorph_v2(W, order, prior, return_details=True, stage1_cap=True)
        row["v2"] = compute_shd(dl_fit(Z)[0], T)["normalized_shd"]
        recs = [d for d in det.values() if d["skipped"] is None]
        row["gate"] = sum(d["gated"] for d in recs) / len(recs) if recs else np.nan
        Zp, degs, _ = v2poly(W, order, prior)
        row["v2poly"] = compute_shd(dl_fit(Zp)[0], T)["normalized_shd"]
        row["poly_deg"] = float(np.mean(degs)) if degs else np.nan
        row["pc"] = compute_shd(pc_cpdag(V), T)["normalized_shd"]
        row["empty"] = compute_shd(np.zeros_like(T), T)["normalized_shd"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", help="name=dataset_dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "nonlin_scale_probe.csv"))
    a = ap.parse_args()
    jobs = []
    for c in a.cells:
        name, ds = c.split("=", 1)
        ids = pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).scenario_id
        jobs += [(name, ds, s) for s in ids]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    D = pd.DataFrame(rows); D.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    g = D.groupby("cell", sort=False)
    out = g[["dl", "v2", "v2poly", "pc", "empty"]].mean().round(3)
    out["windows"] = g.size()
    out["orient dl"] = (g.dl_ok.sum() / g.dl_tot.sum()).round(2)
    out["gate rate"] = g.gate.mean().round(2)
    out["R2 gain cubic"] = g.r2_gain_cubic.mean().round(3)
    out["poly deg"] = g.poly_deg.mean().round(2)
    for k in ("v2", "v2poly"):
        out[f"{k}-dl"] = g.apply(lambda x: f"{(x[k] - x.dl).mean():+.3f} ({(x[k] < x.dl).mean():.2f}/{(x[k] > x.dl).mean():.2f})")
    print("mean nSHD per window; paired diff (share of windows lower/higher)\n" + out.to_string())


if __name__ == "__main__":
    main()
