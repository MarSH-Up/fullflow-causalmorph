#!/usr/bin/env python3
"""
Why CausalMorph v2 adds nothing on top of DirectLiNGAM, and what two changes would buy.
Measured on the true regime windows (standardised, variables permuted) of the first K
scenarios per cell of the local 150-scenario grid (results/robustness/grid_ids.txt).

Per window, nSHD against the regime's graph for
  dl           DirectLiNGAM on the window
  v2           DirectLiNGAM after CausalMorph v2cap (the paper's configuration)
  v2poly       v2cap whose Stage I keeps the polynomial residual: the window becomes
               tangent-plane mean + (noise-shaped) polynomial residual, instead of
               tangent-plane mean + (curvature + noise), which is what v2 outputs
  resit        RESIT (Peters et al. 2014): kernel-ridge regression + HSIC, order then pruning
  v2poly@resit v2poly driven by RESIT's order and graph instead of the tentative DirectLiNGAM fit
  empty
and the orientation accuracy on the true edges for the tentative DirectLiNGAM fit and
for a pairwise ANM test (kernel ridge both ways, HSIC on the residual), plus the Stage I
degree MDL picks and the share of residuals the Stage II gate maps.

  OMP_NUM_THREADS=1 python tools/causalmorph_diagnosis.py --k 30 --workers 8 --out results/causalmorph_diagnosis.csv
"""
import argparse
import os
import sys
import warnings
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph"))
from run_experiments import load_scenario  # noqa: E402
from full_pipeline import _make_learner, _from_lingam, _to_causalmorph, _standardize_window, compute_shd  # noqa: E402
from core.causalmorph_v2 import causalmorph_v2, _max_degree_for, _stage2, _stage3  # noqa: E402
from core.causalmorph_algorithm import best_polynomial_degree_mdl  # noqa: E402

CELLS = {"lin_laplace": "datasets", "nonlin_laplace": "datasets_rob150_nonlin_laplace",
         "lin_normal": "datasets_rob150_lin_normal", "nonlin_normal": "datasets_rob150_nonlin_normal"}
N_SUB = 600       # rows used by the kernel methods
ALPHA = 0.01      # HSIC level for RESIT's pruning
HSIC_MODE = os.environ.get("HSIC_MODE", "stat")      # stat: rank by the normalised statistic; p: by the p-value
KRR_GAMMA = os.environ.get("KRR_GAMMA", "median")    # median: median heuristic per regression; k: 1 / n_features
KRR_ALPHA = float(os.environ.get("KRR_ALPHA", "0.01"))


# ── HSIC with the gamma approximation (Gretton et al. 2008) ──────────────────────
def _gram(x):
    x = np.asarray(x, float).reshape(len(x), -1)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    pos = d2[d2 > 0]
    s2 = np.median(pos) / 2 if len(pos) else 1.0
    return np.exp(-d2 / (2 * s2))


def hsic(x, y):
    """(normalised statistic, p-value) of independence; p from the gamma approximation of the biased HSIC."""
    K, L = _gram(x), _gram(y); m = len(K)
    H = np.eye(m) - 1.0 / m
    Kc, Lc = H @ K @ H, H @ L @ H
    stat = (Kc * Lc).sum() / m
    nstat = (Kc * Lc).sum() / np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    v = (Kc * Lc / 6) ** 2
    var = (v.sum() - np.trace(v)) / m / (m - 1) * 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3)
    K0, L0 = K - np.diag(np.diag(K)), L - np.diag(np.diag(L))
    mux, muy = K0.sum() / m / (m - 1), L0.sum() / m / (m - 1)
    mean = (1 + mux * muy - mux - muy) / m
    if var <= 0 or mean <= 0:
        return nstat, 1.0
    return nstat, float(1 - gamma_dist.cdf(stat, mean ** 2 / var, scale=var * m / mean))


def hsic_p(x, y):
    return hsic(x, y)[1]


def indep_score(x, y):
    """Higher = more independent, in the mode HSIC_MODE."""
    st, pv = hsic(x, y)
    return pv if HSIC_MODE == "p" else -st


def krr_resid(X, y):
    X = np.asarray(X, float).reshape(len(y), -1)
    if KRR_GAMMA == "median":
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); pos = d2[d2 > 0]
        g = 1.0 / np.median(pos) if len(pos) else 1.0
    else:
        g = 1.0 / X.shape[1]
    m = KernelRidge(alpha=KRR_ALPHA, kernel="rbf", gamma=g).fit(X, y)
    return y - m.predict(X)


# ── RESIT ─────────────────────────────────────────────────────────────────────────
def resit(X):
    """Binary adjacency A[from, to] from RESIT with kernel ridge + HSIC."""
    n, p = X.shape
    S, order = list(range(p)), []
    while len(S) > 1:
        best = max(S, key=lambda k: indep_score(X[:, [j for j in S if j != k]], krr_resid(X[:, [j for j in S if j != k]], X[:, k])))
        order.insert(0, best); S.remove(best)
    order.insert(0, S[0])
    A = np.zeros((p, p))
    for i, k in enumerate(order):
        pa = order[:i]
        keep = list(pa)
        for j in pa:
            rest = [q for q in keep if q != j]
            r = krr_resid(X[:, rest], X[:, k]) if rest else X[:, k] - X[:, k].mean()
            if hsic_p(X[:, pa], r) > ALPHA:        # residual independent of all predecessors without j: drop j
                keep.remove(j)
        A[keep, k] = 1.0
    return A, order


# ── CausalMorph v2 with the polynomial residual ────────────────────────────────────
def v2poly(X, order, adj_cm):
    V = X.to_numpy(float); Xn = V.copy(); n = len(V); adj = np.asarray(adj_cm, float)
    degs, gated = [], []
    for idx in order:
        pa = np.where(adj[idx] != 0)[0]
        if len(pa) == 0:
            continue
        P, y = Xn[:, pa], V[:, idx]
        if np.any(P.var(0) < 1e-8):
            continue
        Ps = StandardScaler().fit_transform(P)
        d = best_polynomial_degree_mdl(Ps, y, max_degree=_max_degree_for(n, len(pa)), lam=2.0)
        m = make_pipeline(PolynomialFeatures(d, include_bias=False), LinearRegression()).fit(Ps, y)
        x0 = np.median(Ps, 0); f0 = m.predict(x0[None])[0]
        J = np.array([(m.predict((x0 + 1e-2 * np.eye(len(pa))[i])[None])[0] - f0) / 1e-2 for i in range(len(pa))])
        y_lin = f0 + (Ps - x0) @ J
        eps = y - m.predict(Ps)
        if eps.std() < 1e-8:
            continue
        e2, g, _ = _stage2(eps, "rankmap", 0.01)
        e3 = _stage3(e2, P, eps.std())
        if e3 is None:
            continue
        degs.append(d); gated.append(g)
        Xn[:, idx] = y_lin + e3
    return pd.DataFrame(Xn, columns=X.columns), degs, gated


def dl_fit(W):
    m = _make_learner("directlingam", 0).fit(W)
    return _from_lingam(m.adjacency_matrix_), list(m.causal_order_)


def orient_acc(A_true, A_est):
    """Among true edges present in the estimate in either direction, share oriented right."""
    ok = tot = 0
    for i, j in zip(*np.where(A_true != 0)):
        if A_est[i, j] or A_est[j, i]:
            tot += 1; ok += int(bool(A_est[i, j]) and not A_est[j, i])
    return ok, tot


def one(args):
    cell, sid = args
    ds = os.path.join(ROOT, CELLS[cell])
    index = pd.read_csv(os.path.join(ds, "index.csv"), dtype=str).set_index("scenario_id")
    X, names, cps, adjs, _, _ = load_scenario(sid, ds, index.loc[sid], permute_variables=True)
    bounds = [0] + list(cps) + [len(X)]
    rows = []
    for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        W = _standardize_window(pd.DataFrame(X[a:b], columns=names))
        T = adjs[r].to_numpy(float); p = len(names)
        sub = np.random.default_rng(r).choice(len(W), min(N_SUB, len(W)), replace=False)
        Ws = W.to_numpy(float)[sub]
        row = {"cell": cell, "scenario_id": sid, "regime": r, "n": len(W), "p": p, "edges": int(T.sum())}
        # tentative fit and the paper's transform
        B, order = dl_fit(W)
        row["dl"] = compute_shd(B, T)["normalized_shd"]
        row["dl_ok"], row["dl_tot"] = orient_acc(T, np.abs(B) > 0.05)
        est = pd.DataFrame((np.abs(B) > 0.05).astype(float), index=names, columns=names)
        skel = ((est.values + est.values.T) > 0).astype(float)
        row["dl_undir"] = compute_shd(skel, T)["normalized_shd"]          # abstain from orientation: skeleton as undirected edges
        Z, det = causalmorph_v2(W, order, _to_causalmorph(est), return_details=True, stage1_cap=True)
        row["v2"] = compute_shd(dl_fit(Z)[0], T)["normalized_shd"]
        recs = [d for d in det.values() if d["skipped"] is None]
        row["gate_n"], row["gate_on"] = len(recs), sum(int(d["gated"]) for d in recs)
        gauss = row["gate_n"] > 0 and row["gate_on"] / row["gate_n"] >= 0.5
        row["dl_route"] = row["dl_undir"] if gauss else row["dl"]        # abstain only when the gate calls the window Gaussian
        row["deg_gt1"] = sum(int(d["max_degree"] > 1) for d in recs)
        Zp, degs, gated = v2poly(W, order, _to_causalmorph(est))
        row["v2poly"] = compute_shd(dl_fit(Zp)[0], T)["normalized_shd"]
        row["poly_deg_mean"] = float(np.mean(degs)) if degs else np.nan
        # pairwise ANM orientation on the true edges
        ok = 0
        for i, j in zip(*np.where(T != 0)):
            p_ij = indep_score(Ws[:, i], krr_resid(Ws[:, i], Ws[:, j]))
            p_ji = indep_score(Ws[:, j], krr_resid(Ws[:, j], Ws[:, i]))
            ok += int(p_ij > p_ji)
        row["anm_ok"], row["anm_tot"] = ok, int(T.sum())
        # RESIT, and v2poly driven by it
        A_r, order_r = resit(Ws)
        row["resit"] = compute_shd(A_r, T)["normalized_shd"]
        row["resit_ok"], row["resit_tot"] = orient_acc(T, A_r)
        Zr, _, _ = v2poly(W, order_r, _to_causalmorph(pd.DataFrame(A_r, index=names, columns=names)))
        row["v2poly@resit"] = compute_shd(dl_fit(Zr)[0], T)["normalized_shd"]
        row["empty"] = compute_shd(np.zeros_like(T), T)["normalized_shd"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "causalmorph_diagnosis.csv"))
    ap.add_argument("--cells", default=" ".join(CELLS))
    a = ap.parse_args()
    ids = [l.strip() for l in open(os.path.join(ROOT, "results/robustness/grid_ids.txt")) if l.strip()][:a.k]
    jobs = [(c, s) for c in a.cells.split() for s in ids]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    D = pd.DataFrame(rows); D.to_csv(a.out, index=False)
    report(D)


def report(D):
    pd.set_option("display.width", 200)
    keys = ["dl", "v2", "v2poly", "dl_undir", "dl_route", "resit", "v2poly@resit", "empty"]
    g = D.groupby("cell")
    out = g[keys].mean().round(3)
    out["windows"] = g.size()
    out["orient dl"] = (g.dl_ok.sum() / g.dl_tot.sum()).round(2)
    out["orient anm"] = (g.anm_ok.sum() / g.anm_tot.sum()).round(2)
    out["orient resit"] = (g.resit_ok.sum() / g.resit_tot.sum()).round(2)
    out["gate rate"] = (g.gate_on.sum() / g.gate_n.sum()).round(2)
    out["stage I deg>1"] = (g.deg_gt1.sum() / g.gate_n.sum()).round(2)
    out["poly deg"] = g.poly_deg_mean.mean().round(2)
    print("mean nSHD per window (lower is better), orientation accuracy on true edges, Stage I/II diagnostics")
    print(out.reindex([c for c in CELLS if c in out.index]).to_string())


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--report":
        report(pd.read_csv(sys.argv[2]))
    else:
        main()
