#!/usr/bin/env python3
"""
Which identifiability source each window offers, and what each path costs, on the true
regime windows (standardised, variables permuted) of every scenario of the dataset
directories given as name=dir. Per window: the tentative DirectLiNGAM graph's gate share
(residuals the Stage II gate calls Gaussian) and curvature share (nodes whose MDL
polynomial has degree >= 2), and the nSHD of the three paths: lingam (DirectLiNGAM after
the paper's v2cap transform), lingam3 (after v3), s0 (Stage 0 DAG) and pc (PC's CPDAG).
The report simulates routing rules on those numbers.

  OMP_NUM_THREADS=1 python tools/routing_diagnosis.py --workers 10 --out results/routing_diagnosis.csv name=dir ...
  python tools/routing_diagnosis.py --report results/routing_diagnosis.csv
"""
import argparse
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
EXTRA = os.environ.get("EXTRA", "0") == "1"   # also DAGMA (paper configuration) and CausalMorph v2 + ICA-LiNGAM per window
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph")); sys.path.insert(0, os.path.join(ROOT, "tools"))
from run_experiments import load_scenario  # noqa: E402
from full_pipeline import _to_causalmorph, _standardize_window, compute_shd, _apply_transform, _fit_window_learner, _make_learner, _from_lingam  # noqa: E402
from core.causalmorph_v2 import causalmorph_v2, gate_share, curvature_share, curvature_gain  # noqa: E402
from core.stage0 import stage0  # noqa: E402
from causalmorph_diagnosis import dl_fit  # noqa: E402
from nonlin_scale_probe import pc_cpdag  # noqa: E402


def one(args):
    name, ds, sid = args
    index = pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).set_index("scenario_id")
    X, names, cps, adjs, _, _ = load_scenario(sid, ds, index.loc[sid], permute_variables=True)
    bounds = [0] + list(cps) + [len(X)]
    rows = []
    for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        W = _standardize_window(pd.DataFrame(X[a:b], columns=names)); T = adjs[r].to_numpy(float)
        B, order = dl_fit(W)
        est = pd.DataFrame((np.abs(B) > 0.05).astype(float), index=names, columns=names)
        _, det = causalmorph_v2(W, order, _to_causalmorph(est), stage1="poly", stage1_cap=True, return_details=True)
        recs = [d for d in det.values() if d["skipped"] is None]
        row = {"cell": name, "scenario_id": sid, "regime": r, "n": len(W), "p": len(names), "n_nonroot": len(recs),
               "gate": gate_share(det), "curve": curvature_share(det), "r2": curvature_gain(det),
               "r2_share05": float(np.mean([d["r2_gain"] >= 0.05 for d in recs])) if recs else np.nan}
        row["lingam"] = compute_shd(dl_fit(_apply_transform("v2cap", W, order, est, r))[0], T)["normalized_shd"]
        row["lingam3"] = compute_shd(dl_fit(_apply_transform("v3", W, order, est, r))[0], T)["normalized_shd"]
        A0, _ = stage0(W.to_numpy(float), regressor="mlp", seed=r)
        row["s0"] = compute_shd(A0, T)["normalized_shd"]
        row["pc"] = compute_shd(pc_cpdag(W.to_numpy(float)), T)["normalized_shd"]
        row["empty"] = compute_shd(np.zeros_like(T), T)["normalized_shd"]
        if EXTRA:
            row["dagma"] = compute_shd(_fit_window_learner("dagma_paper", W, names, r)[0], T)["normalized_shd"]
            ica = _make_learner("icalingam", r).fit(W)                                   # the paper's previous pipeline: CausalMorph v2 + ICA-LiNGAM
            est_i = pd.DataFrame((np.abs(_from_lingam(ica.adjacency_matrix_)) > 0.05).astype(float), index=names, columns=names)
            Zi = _apply_transform("v2cap", W, list(ica.causal_order_), est_i, r)
            row["ica"] = compute_shd(_from_lingam(_make_learner("icalingam", r).fit(Zi).adjacency_matrix_), T)["normalized_shd"]
        rows.append(row)
    return rows


def route(D, tc, tg, nmin, col="r2"):
    """curvature (col) >= tc -> s0; else gate >= tg -> pc; else lingam. Windows with fewer than nmin processed nodes stay on lingam."""
    ok = D.n_nonroot >= nmin
    return np.where(ok & (D[col] >= tc), D.s0, np.where(ok & (D.gate >= tg), D.pc, D.lingam))


def report(D):
    pd.set_option("display.width", 250)
    order = list(dict.fromkeys(D.cell)); g = D.groupby("cell", sort=False)
    out = g[[c for c in ("lingam", "lingam3", "s0", "pc", "dagma", "ica", "empty") if c in D]].mean()
    out["best of 3"] = g.apply(lambda x: x[["lingam", "s0", "pc"]].min(axis=1).mean())
    out["gate"] = g.gate.mean(); out["curve"] = g.curve.mean(); out["r2 gain"] = g.r2.mean(); out["windows"] = g.size()
    print("mean nSHD per path (oracle windows), mean gate share, curvature share (degree >= 2) and R² gain\n" + out.round(3).reindex(order).to_string())
    for col, grid in (("r2", (0.01, 0.02, 0.05, 0.1)), ("r2_share05", (0.34, 0.5)), ("curve", (0.67, 1.01))):
        print(f"\nrouting: {col} >= tc -> Stage 0; else gate >= tg -> PC CPDAG; else LiNGAM (nodes >= nmin). mean nSHD (share to s0 / to pc)")
        rows = []
        for tc in grid:
            for tg in (0.5, 0.67):
                for nmin in (1, 3):
                    r = {"tc": tc, "tg": tg, "nmin": nmin}
                    for c in order:
                        x = D[D.cell == c]; ok = x.n_nonroot >= nmin
                        to_s0 = (ok & (x[col] >= tc)).mean(); to_pc = (ok & ~(x[col] >= tc) & (x.gate >= tg)).mean()
                        r[c] = f"{route(x, tc, tg, nmin, col).mean():.3f} ({to_s0:.2f}/{to_pc:.2f})"
                    rows.append(r)
        print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "routing_diagnosis.csv")); ap.add_argument("--report")
    a = ap.parse_args()
    if a.report:
        report(pd.read_csv(a.report)); return
    jobs = []
    for c in a.cells:
        name, ds = c.split("=", 1)
        jobs += [(name, ds, s) for s in pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).scenario_id]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    D = pd.DataFrame(rows); D.to_csv(a.out, index=False); report(D)


if __name__ == "__main__":
    main()
