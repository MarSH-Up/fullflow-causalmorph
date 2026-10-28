#!/usr/bin/env python3
"""
Tables of run_robustness.sh: mean nSHD of every learner on raw and on
column-standardised windows, per grid cell and for the external benchmark
(also by noise type), and detection against a uniform-spacing reference.

  python tools/analyze_robustness.py [results/robustness]
Writes <out>/summary.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "robustness"))
TARGETS = ["lin_laplace", "nonlin_laplace", "lin_normal", "nonlin_normal", "external"]
TOL = 200


def load(target, arm):
    f = os.path.join(OUT, target, arm, "results.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[d.status == "ok"].drop_duplicates("scenario_id").set_index("scenario_id")


def columns(target):
    """{label: series of per-scenario nSHD}."""
    cols = {}
    for arm, tag in (("raw", "raw"), ("std", "std")):
        d = load(target, arm)
        if d is None:
            continue
        cols[f"pipeline ICA ({tag})"] = d.mean_norm_shd
        for m, name in (("icalingam", "ICA-LiNGAM"), ("directlingam", "DirectLiNGAM"), ("dagma", "DAGMA rescaled"),
                        ("dagma_paper", "DAGMA paper cfg"), ("sortnregress", "sortnregress"), ("randomregress", "randomregress"), ("empty", "empty graph")):
            if f"{m}_mean_norm_shd" in d:
                cols[f"{name} ({tag})"] = d[f"{m}_mean_norm_shd"]
    for arm, tag in (("raw_dl", "raw"), ("std_dl", "std")):
        d = load(target, arm)
        if d is not None:
            cols[f"pipeline Direct ({tag})"] = d.mean_norm_shd
    d = load(target, "cdnod")
    if d is not None:
        cols["CD-NOD Fisher-z"] = d.mean_norm_shd
    return cols


def parse(s):
    return [int(x) for x in str(s).split(";") if x.strip() and x != "nan"]


def match(tc, dc, tol=TOL):
    used, tp = set(), 0
    for t in sorted(tc):
        cand = [(abs(x - t), j) for j, x in enumerate(sorted(dc)) if j not in used and abs(x - t) <= tol]
        if cand:
            used.add(min(cand)[1]); tp += 1
    return tp, len(dc) - tp, len(tc) - tp


def detection(target):
    d = load(target, "raw")
    if d is None:
        return {}
    out = {}
    for name, get in (("detector (raw)", lambda r: parse(r.detected_cps)),
                      ("uniform, true count", lambda r: [int(round(r.total_samples * k / r.n_regimes)) for k in range(1, int(r.n_regimes))])):
        TP = FP = FN = 0
        for _, r in d.iterrows():
            a, b, c = match(parse(r.true_cps), get(r)); TP += a; FP += b; FN += c
        P = TP / (TP + FP) if TP + FP else 0.0; R = TP / (TP + FN) if TP + FN else 0.0
        out[name] = round(2 * P * R / (P + R) if P + R else 0.0, 3)
    tp, fp, fn = d.chow0001_tp.sum(), d.chow0001_fp.sum(), d.chow0001_fn.sum()
    P, R = tp / (tp + fp), tp / (tp + fn)
    out["confirmed"] = round(2 * P * R / (P + R), 3)
    return out


def main():
    rows = []
    for t in TARGETS:
        cols = columns(t)
        if not cols:
            continue
        ids = sorted(set.intersection(*[set(s.dropna().index) for s in cols.values()]))
        rows.append({"target": t, "n": len(ids), **{k: round(float(v.loc[ids].mean()), 3) for k, v in cols.items()},
                     **{f"F1 {k}": v for k, v in detection(t).items()}})
        if t == "external":
            sem = pd.read_csv(os.path.join(ROOT, open(os.path.join(OUT, "RUN_INFO.txt")).read().split("external: ")[1].split(" |")[0],
                                           "index.csv")).set_index("scenario_id").sem_type.reindex(ids)
            for s in sorted(sem.unique()):
                k = sem.index[sem == s]
                rows.append({"target": f"external/{s}", "n": len(k), **{c: round(float(v.loc[k].mean()), 3) for c, v in cols.items()}})
    T = pd.DataFrame(rows).set_index("target").T
    T.to_csv(os.path.join(OUT, "summary.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(T.to_string())


if __name__ == "__main__":
    main()
