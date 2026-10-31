#!/usr/bin/env python3
"""
Benchmark diagnostics per scenario (on the true regimes of the raw series; permutation-free):
  varsortability     share of directed paths along which the marginal variance increases (Reisach et al. 2021)
  r2sortability      the same with the R^2 of each variable on all the others (Reisach et al. 2023)
  cubic_r2_gain      mean over non-root nodes of R^2(cubic polynomial of the true parents) - R^2(linear)
  density            mean edge density of the regime graphs
  nshd_empty         mean nSHD of the empty graph over the regimes
  nshd_single_oracle mean nSHD of the best single DAG for the whole scenario (majority vote of the
                     regime graphs, which minimises the summed SHD): the ceiling of any single-graph output
plus the number of edges that flip direction between consecutive regimes (always 0 in this generator).

  python tools/benchmark_diagnostics.py --dataset_dir datasets --output_dir results/v3/lin_laplace/diagnostics \
      --scenario_ids_file ids.txt --workers 12
"""
import argparse
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sortability(values: np.ndarray, A: np.ndarray) -> float:
    """Share of directed paths (i ... j, any length) with values[i] < values[j]; nan without paths."""
    p = len(values); E = (A != 0).astype(float); M = E.copy(); num = den = 0.0
    for _ in range(p - 1):
        src, dst = np.where(M > 0)
        if len(src) == 0:
            break
        num += np.sum(values[src] < values[dst]); den += len(src)
        M = (M @ E > 0).astype(float)
    return num / den if den else np.nan


def r2_all_others(X: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        others = [k for k in range(X.shape[1]) if k != j]
        out[j] = LinearRegression().fit(X[:, others], X[:, j]).score(X[:, others], X[:, j])
    return out


def cubic_gain(X: np.ndarray, A: np.ndarray) -> float:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    gains = []
    for j in range(A.shape[0]):
        pa = np.where(A[:, j] != 0)[0]
        if len(pa) == 0:
            continue
        P, y = X[:, pa], X[:, j]
        gains.append(make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(P, y).score(P, y) - LinearRegression().fit(P, y).score(P, y))
    return float(np.mean(gains)) if gains else np.nan


def nshd(est: np.ndarray, true: np.ndarray) -> float:
    p = true.shape[0]
    return float(((est != 0) != (true != 0)).sum() - 0) / (p * (p - 1))


def _job(sid, index_row, dataset_dir):
    from run_experiments import load_scenario
    t0 = time.time()
    try:
        X, names, cps, adjs, _, _ = load_scenario(sid, dataset_dir, index_row)
        bounds = [0] + list(cps) + [len(X)]
        T = [a.to_numpy(float) for a in adjs]; p = len(names)
        vs, r2s, cg, dens = [], [], [], []
        for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            W = X[a:b]
            vs.append(sortability(W.var(0), T[r])); r2s.append(sortability(r2_all_others(W), T[r]))
            cg.append(cubic_gain(W, T[r])); dens.append((T[r] != 0).sum() / (p * (p - 1)))
        vote = (np.mean([(t != 0) for t in T], axis=0) > 0.5).astype(float)   # majority graph: minimises the summed SHD
        flips = sum(int(((T[k] != 0) & (T[k + 1].T != 0)).sum()) for k in range(len(T) - 1))
        return {"scenario_id": sid, "p": p, "n_regimes": len(T), "status": "ok",
                "varsortability": float(np.nanmean(vs)), "r2sortability": float(np.nanmean(r2s)),
                "cubic_r2_gain": float(np.nanmean(cg)), "density": float(np.mean(dens)),
                "nshd_empty": float(np.mean([nshd(np.zeros_like(t), t) for t in T])),
                "nshd_single_oracle": float(np.mean([nshd(vote, t) for t in T])), "edge_flips": flips,
                "seconds": round(time.time() - t0, 1)}
    except Exception:  # noqa: BLE001
        return {"scenario_id": sid, "status": "error", "error_msg": traceback.format_exc(limit=3).strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True); ap.add_argument("--output_dir", required=True)
    ap.add_argument("--scenario_ids_file"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    index = pd.read_csv(os.path.join(a.dataset_dir, "index.csv"), dtype=str)
    if a.scenario_ids_file:
        keep = {ln.strip() for ln in open(a.scenario_ids_file) if ln.strip()}
        index = index[index.scenario_id.isin(keep)]
    os.makedirs(a.output_dir, exist_ok=True); out = os.path.join(a.output_dir, "results.csv")
    if a.resume and os.path.exists(out):
        index = index[~index.scenario_id.isin(set(pd.read_csv(out, dtype=str).scenario_id))]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_job, r.scenario_id, r, a.dataset_dir) for _, r in index.iterrows()]
        for f in as_completed(futs):
            rows.append(f.result())
    D = pd.DataFrame(rows)
    D.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    ok = D[D.status == "ok"]
    print(f"{len(ok)} scenarios -> {out}")
    if len(ok):
        print(ok[["varsortability", "r2sortability", "cubic_r2_gain", "density", "nshd_empty", "nshd_single_oracle", "edge_flips"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
