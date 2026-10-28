#!/usr/bin/env python3
"""
Pick ONE DAGMA setting (l1 weight, weight threshold) for all cells from the
calibration sweep of run_review_extras.sh: the setting with the lowest mean nSHD
averaged over the cells, the same rule the method's constants followed. Prints
the table and, on the last line, the learner name (e.g. dagma_l0.02_t0.2).

  python tools/pick_dagma.py results/review_extras lin_laplace nonlin_laplace lin_normal nonlin_normal
"""
import os
import re
import sys

import pandas as pd

out, cells = sys.argv[1], sys.argv[2:]
means = {}
for c in cells:
    d = pd.read_csv(os.path.join(out, c, "dagma_sweep", "results.csv"))
    d = d[d.status == "ok"]
    for col in d.columns:
        m = re.fullmatch(r"(dagma_l[0-9.]+_t[0-9.]+)_mean_norm_shd", col)
        if m:
            means.setdefault(m.group(1), {})[c] = d[col].mean()
T = pd.DataFrame(means).T
T["mean over cells"] = T.mean(axis=1)
T = T.sort_values("mean over cells")
print(T.round(4).to_string(), file=sys.stderr)
print(T.index[0])
