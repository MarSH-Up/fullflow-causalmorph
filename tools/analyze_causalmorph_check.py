#!/usr/bin/env python3
"""
Tables of run_causalmorph_check.sh: mean nSHD and skeleton F1 of every
configuration per cell, and the paired differences that isolate the
preconditioning (v2 - none), Stage II (v2 - lite) and the tentative orientation
(shuffle - v2), for each learner. Negative nSHD differences favour the first term.

  python tools/analyze_causalmorph_check.py [results/causalmorph_check]
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "causalmorph_check"))
CELLS = ["lin_laplace", "nonlin_laplace", "lin_normal", "nonlin_normal"]
LEARNERS = ["directlingam", "icalingam"]
VARIANTS = ["v2", "lite", "none", "shuffle"]
CONTRASTS = [("v2", "none", "preconditioning (v2 - none)"), ("v2", "lite", "Stage II (v2 - lite)"),
             ("shuffle", "v2", "tentative orientation (shuffle - v2)")]


def load(cell, cfg):
    f = os.path.join(OUT, cell, cfg, "results.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[d.status == "ok"].drop_duplicates("scenario_id").set_index("scenario_id")


def paired(a, b, n_boot=2000, seed=0):
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    v = (j.iloc[:, 0] - j.iloc[:, 1]).to_numpy(float)
    if len(v) == 0:
        return "-"
    boot = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f"{v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {int((v < 0).sum())}/{int((v > 0).sum())}"


def main():
    means, contrasts = [], []
    for cell in CELLS:
        data = {f"{l}_{v}": load(cell, f"{l}_{v}") for l in LEARNERS for v in VARIANTS}
        for cfg, d in data.items():
            if d is not None:
                means.append({"cell": cell, "config": cfg, "n": len(d), "nSHD": round(d.mean_norm_shd.mean(), 3),
                              "skeleton F1": round(d.mean_skeleton_f1.mean(), 3)})
        for l in LEARNERS:
            for a, b, label in CONTRASTS:
                da, db = data.get(f"{l}_{a}"), data.get(f"{l}_{b}")
                if da is not None and db is not None:
                    contrasts.append({"learner": l, "contrast": label, "cell": cell,
                                      "nSHD diff": paired(da.mean_norm_shd, db.mean_norm_shd)})
    pd.set_option("display.width", 250)
    if means:
        M = pd.DataFrame(means).pivot(index="config", columns="cell", values="nSHD").reindex(columns=CELLS)
        S = pd.DataFrame(means).pivot(index="config", columns="cell", values="skeleton F1").reindex(columns=CELLS)
        print("mean nSHD (lower is better)\n" + M.to_string() + "\n\nskeleton F1\n" + S.to_string())
    if contrasts:
        C = pd.DataFrame(contrasts).pivot(index=["learner", "contrast"], columns="cell", values="nSHD diff").reindex(columns=CELLS)
        print("\npaired differences in nSHD (mean [95% CI] scenarios lower/higher; negative favours the first term)\n" + C.to_string())
        C.to_csv(os.path.join(OUT, "contrasts.csv"))


if __name__ == "__main__":
    main()
