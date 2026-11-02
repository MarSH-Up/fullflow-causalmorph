#!/usr/bin/env python3
"""
Pick ONE edge threshold w for CASTOR for all cells from the calibration sweep of
run_complements.sh (step castor_cal): the threshold with the lowest mean nSHD over the
scenarios that finished, averaged over the cells, the same rule as tools/pick_dagma.py.
Prints the table (with the time-outs per setting) on stderr and, on the last line of
stdout, the chosen threshold (e.g. 0.2).

  python tools/pick_castor.py results/v3 lin_laplace nonlin_laplace lin_normal nonlin_normal
"""
import glob
import os
import re
import sys

import pandas as pd

out, cells = sys.argv[1], sys.argv[2:]
means, timeouts = {}, {}
for c in cells:
    for d in sorted(glob.glob(os.path.join(out, c, "castor_cal_w*"))):
        m = re.search(r"castor_cal_w(\d+)$", d)
        f = os.path.join(d, "results.csv")
        if not m or not os.path.exists(f):
            continue
        w = int(m.group(1)) / 10.0
        r = pd.read_csv(f).drop_duplicates("scenario_id")
        means.setdefault(w, {})[c] = r[r.status == "ok"].mean_norm_shd.mean()
        timeouts.setdefault(w, {})[c] = int((r.status != "ok").sum())
T = pd.DataFrame(means).T
T["mean over cells"] = T.mean(axis=1)
T["time-outs"] = pd.DataFrame(timeouts).T.sum(axis=1)
T = T.sort_values("mean over cells")
print(T.round(4).to_string(), file=sys.stderr)
print(T.index[0])
