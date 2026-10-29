#!/usr/bin/env python3
"""
Tuning of the nonlinear orientation stage (CausalMorph Stage 0) on the same windows as
tools/causalmorph_diagnosis.py: the tentative skeleton is oriented by iterative sink
removal, where the sink among the remaining nodes is the one whose residual, after a
nonparametric regression on its remaining neighbours, is most independent of them
(normalised HSIC). Variants: skeleton from DirectLiNGAM or PC (Fisher-z), regressor
kernel ridge (RBF, median heuristic) or polynomial ridge.

Per window and variant: orientation accuracy on the true edges, nSHD of the oriented
skeleton, nSHD of DirectLiNGAM after CausalMorph (polynomial residual) driven by it,
and the gate-routed versions (the nonlinear source only when the Stage II gate calls
the window Gaussian).

  OMP_NUM_THREADS=1 python tools/orienter_tuning.py --k 30 --workers 10 --out results/orienter_tuning.csv
"""
import argparse
import contextlib
import io
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph")); sys.path.insert(0, os.path.join(ROOT, "tools"))
from run_experiments import load_scenario  # noqa: E402
from full_pipeline import _to_causalmorph, _standardize_window, compute_shd  # noqa: E402
from core.causalmorph_v2 import causalmorph_v2  # noqa: E402
from causalmorph_diagnosis import CELLS, N_SUB, hsic, v2poly, dl_fit, orient_acc  # noqa: E402


KINDS = os.environ.get("KINDS", "krr poly").split()
SKELS = os.environ.get("SKELS", "dl pc").split()


def resid(kind, X, y):
    """Residual of y on X for the regressor `kind`: krr / krr3 (kernel ridge, alpha 0.1 / 1e-3),
    poly / poly3 (polynomial ridge, alpha 1 / 1e-3), mlp (one tanh layer, lbfgs)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    if kind.startswith("krr"):
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); pos = d2[d2 > 0]
        m = KernelRidge(alpha=1e-3 if kind == "krr3" else 0.1, kernel="rbf", gamma=1.0 / np.median(pos) if len(pos) else 1.0).fit(X, y)
    elif kind.startswith("poly"):
        deg = 3 if X.shape[1] <= 5 else 2
        m = make_pipeline(PolynomialFeatures(deg, include_bias=False), Ridge(alpha=1e-3 if kind == "poly3" else 1.0)).fit(X, y)
    elif kind == "mlp":
        m = MLPRegressor(hidden_layer_sizes=(10,), activation="tanh", solver="lbfgs", alpha=1e-4, max_iter=500, random_state=0).fit(X, y)
    else:
        raise ValueError(kind)
    return y - m.predict(X)


def orient_skeleton(X, skel, kind):
    """DAG A[from, to] on the skeleton by iterative sink removal; returns (A, order)."""
    p = skel.shape[0]
    R = set(range(p)); A = np.zeros((p, p)); order = []
    score = {}

    def nb(v):
        return sorted(j for j in R if skel[v, j] and j != v)

    def sc(v):
        N = nb(v)
        if not N:
            return -np.inf
        return hsic(X[:, N], resid(kind, X[:, N], X[:, v]))[0]

    for v in R:
        score[v] = sc(v)
    while R:
        v = min(R, key=lambda u: score[u])
        N = nb(v)
        A[N, v] = 1.0
        order.insert(0, v); R.remove(v)
        for u in N:
            score[u] = sc(u)
    return A, order


def pc_skeleton(X):
    from causallearn.search.ConstraintBased.PC import pc
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        cg = pc(X, alpha=0.05, indep_test="fisherz", show_progress=False)
    g = np.asarray(cg.G.graph)
    return ((g != 0) | (g.T != 0)).astype(float)


def skel_f1(S, T):
    Tu = ((T + T.T) > 0); Su = S > 0
    iu = np.triu_indices_from(S, 1)
    tp = (Tu & Su)[iu].sum(); fp = (~Tu & Su)[iu].sum(); fn = (Tu & ~Su)[iu].sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def one(args):
    cell, sid, ds = args
    index = pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).set_index("scenario_id")
    X, names, cps, adjs, _, _ = load_scenario(sid, ds, index.loc[sid], permute_variables=True)
    bounds = [0] + list(cps) + [len(X)]
    rows = []
    for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        W = _standardize_window(pd.DataFrame(X[a:b], columns=names))
        T = adjs[r].to_numpy(float)
        sub = np.random.default_rng(r).choice(len(W), min(N_SUB, len(W)), replace=False)
        Ws = W.to_numpy(float)[sub]
        row = {"cell": cell, "scenario_id": sid, "regime": r, "n": len(W), "p": len(names), "edges": int(T.sum()),
               "nf": float(index.loc[sid, "noise_fraction"])}
        B, order = dl_fit(W)
        est = (np.abs(B) > 0.05).astype(float)
        row["dl"] = compute_shd(B, T)["normalized_shd"]
        row["dl_ok"], row["dl_tot"] = orient_acc(T, est)
        _, det = causalmorph_v2(W, order, _to_causalmorph(pd.DataFrame(est, index=names, columns=names)), return_details=True, stage1_cap=True)
        recs = [d for d in det.values() if d["skipped"] is None]
        row["gauss"] = bool(recs) and sum(d["gated"] for d in recs) / len(recs) >= 0.5
        skels = {"dl": ((est + est.T) > 0).astype(float), "pc": pc_skeleton(Ws)}
        for sk in SKELS:
            S = skels[sk]
            row[f"skel_{sk}_f1"] = skel_f1(S, T)
            for kind in KINDS:
                t0 = time.time()
                A, o = orient_skeleton(Ws, S, kind)
                row[f"t_{sk}_{kind}"] = time.time() - t0
                row[f"or_{sk}_{kind}"] = compute_shd(A, T)["normalized_shd"]
                row[f"or_{sk}_{kind}_ok"], row[f"or_{sk}_{kind}_tot"] = orient_acc(T, A)
                Z, _, _ = v2poly(W, o, _to_causalmorph(pd.DataFrame(A, index=names, columns=names)))
                row[f"cm_{sk}_{kind}"] = compute_shd(dl_fit(Z)[0], T)["normalized_shd"]
        row["empty"] = compute_shd(np.zeros_like(T), T)["normalized_shd"]
        rows.append(row)
    return rows


def report(D):
    pd.set_option("display.width", 250)
    order = list(dict.fromkeys(D.cell))
    g = D.groupby("cell")
    variants = [c[3:] for c in D if c.startswith("or_") and not c.endswith(("_ok", "_tot"))]
    print("skeleton F1:", g[[c for c in D if c.startswith("skel_")]].mean().round(3).reindex(order).to_string(), sep="\n")
    acc = pd.DataFrame({"dl": g.dl_ok.sum() / g.dl_tot.sum()})
    for v in variants:
        acc[v] = g[f"or_{v}_ok"].sum() / g[f"or_{v}_tot"].sum()
    print("\norientation accuracy on true edges (all windows)\n" + acc.round(2).reindex(order).to_string())
    G = D[D.gauss]
    for by in (["cell"], ["cell", "nf"]):
        gg = G.groupby(by)
        accg = pd.DataFrame({"dl": gg.dl_ok.sum() / gg.dl_tot.sum()})
        for v in variants:
            accg[v] = gg[f"or_{v}_ok"].sum() / gg[f"or_{v}_tot"].sum()
        accg["windows"] = gg.size()
        print(f"\norientation accuracy, Gaussian-gated windows only, by {by}\n" + accg.round(2).to_string())
    cols = ["dl"] + [f"{m}_{v}" for v in variants for m in ("or", "cm")] + ["empty"]
    print("\nmean nSHD, unrouted\n" + g[cols].mean().round(3).reindex(order).to_string())
    R = pd.DataFrame(index=order)
    for c in cols[1:-1]:
        routed = np.where(D.gauss, D[c], D.dl)
        R[c] = pd.Series(routed, index=D.index).groupby(D.cell).mean().round(3)
    R["dl"] = g.dl.mean().round(3); R["routed share"] = g.gauss.mean().round(2)
    print("\nmean nSHD, gate-routed (nonlinear source only in Gaussian-gated windows)\n" + R.to_string())
    print("\nseconds per window:", g[[c for c in D if c.startswith("t_")]].mean().round(2).reindex(order).to_string(), sep="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "orienter_tuning.csv"))
    ap.add_argument("--cells", default=" ".join(CELLS))
    a = ap.parse_args()
    ids = [l.strip() for l in open(os.path.join(ROOT, "results/robustness/grid_ids.txt")) if l.strip()][:a.k]
    jobs = []
    for c in a.cells.split():          # a grid cell name, or name=dataset_dir (every scenario of that directory)
        if "=" in c:
            name, ds = c.split("=", 1)
            jobs += [(name, s, ds) for s in pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).scenario_id]
        else:
            jobs += [(c, s, os.path.join(ROOT, CELLS[c])) for s in ids]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    D = pd.DataFrame(rows); D.to_csv(a.out, index=False)
    report(D)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--report":
        report(pd.read_csv(sys.argv[2]))
    else:
        main()
