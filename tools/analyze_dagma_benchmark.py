#!/usr/bin/env python3
"""
Analysis of the external benchmark built from the DAGMA simulator
(run_dagma_benchmark.sh): the pipeline against DAGMA on the same windows and
CD-NOD with Fisher-z, by noise type, graph model and number of variables,
paired by scenario. Writes results/dagma_benchmark/summary.json.

  python tools/analyze_dagma_benchmark.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "dagma_benchmark")
DATA = os.path.join(ROOT, "datasets_dagma")


def load(path):
    d = pd.read_csv(path)
    return d[d.status == "ok"].drop_duplicates("scenario_id", keep="first").set_index("scenario_id")


def paired(a, b, n_boot=2000, seed=0):
    v = (np.asarray(a, float) - np.asarray(b, float))
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    nz = v[v != 0]
    return dict(n=len(v), mean=round(float(v.mean()), 4),
                ci=[round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
                p=float(f"{wilcoxon(nz).pvalue:.2e}") if len(nz) > 10 else None,
                better=int((v < 0).sum()), worse=int((v > 0).sum()), ties=int((v == 0).sum()))


def prf(d, pre):
    tp, fp, fn = d[f"{pre}_tp"].sum(), d[f"{pre}_fp"].sum(), d[f"{pre}_fn"].sum()
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return dict(P=round(P, 3), R=round(R, 3), F1=round(2 * P * R / (P + R) if P + R else 0.0, 3))


def block(m, c):
    s = {"n": int(len(m))}
    for metric, col in (("nshd", "mean_norm_shd"), ("wnshd", "windowed_nshd_weighted")):
        pipe, dag, cd = m[col], m[f"dagma_{col}"], c[col]
        s[metric] = {"pipeline": round(float(pipe.mean()), 4), "dagma": round(float(dag.mean()), 4),
                     "cdnod": round(float(cd.mean()), 4),
                     "pipe_minus_dagma": paired(pipe, dag), "pipe_minus_cdnod": paired(pipe, cd),
                     "rel_vs_dagma": round(float(1 - pipe.mean() / dag.mean()), 3),
                     "rel_vs_cdnod": round(float(1 - pipe.mean() / cd.mean()), 3)}
    s["shd"] = {"pipeline": round(float(m.mean_shd.mean()), 2), "dagma": round(float(m.dagma_mean_shd.mean()), 2)}
    s["detection_raw"] = prf(m, "det")
    s["detection_confirmed"] = prf(m, "chow0001")
    return s


def main():
    idx = pd.read_csv(os.path.join(DATA, "index.csv")).set_index("scenario_id")
    m = load(os.path.join(OUT, "main", "results.csv"))
    c = load(os.path.join(OUT, "cdnod_fz", "results.csv"))
    ids = sorted(set(m.index) & set(c.index))
    m, c, meta = m.loc[ids], c.loc[ids], idx.loc[ids]
    print(f"{len(ids)} scenarios with both rows (of {len(idx)})")

    summary = {"all": block(m, c)}
    for factor in ("sem_type", "graph", "p", "n_regimes"):
        summary[factor] = {str(v): block(m[meta[factor] == v], c[meta[factor] == v])
                           for v in sorted(meta[factor].unique())}
    for sem in sorted(meta.sem_type.unique()):
        for p in sorted(meta.p.unique()):
            k = (meta.sem_type == sem) & (meta.p == p)
            summary.setdefault("sem_type_x_p", {})[f"{sem}/p{p}"] = block(m[k], c[k])

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    rows = []
    for group in ("all", "sem_type", "graph", "p", "sem_type_x_p"):
        items = {"all": summary["all"]} if group == "all" else summary[group]
        for name, s in items.items():
            n, w = s["nshd"], s["wnshd"]
            rows.append({"group": group, "level": name, "n": s["n"],
                         "nSHD pipe": n["pipeline"], "nSHD DAGMA": n["dagma"], "nSHD CD-NOD": n["cdnod"],
                         "d vs DAGMA": n["pipe_minus_dagma"]["mean"], "CI": n["pipe_minus_dagma"]["ci"],
                         "better/worse": f'{n["pipe_minus_dagma"]["better"]}/{n["pipe_minus_dagma"]["worse"]}',
                         "wnSHD pipe": w["pipeline"], "wnSHD DAGMA": w["dagma"],
                         "F1 conf": s["detection_confirmed"]["F1"], "F1 raw": s["detection_raw"]["F1"]})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
