#!/usr/bin/env python3
"""
Detection table of run_stage0_check.sh: per cell and detector configuration (v1O from the
v2_auto run, det_* configurations), change-point F1 with precision/recall, confirmation F1
(Chow, alpha 0.001), windows per true regime and the resulting nSHD. Prints whatever has
finished so far.

  python tools/analyze_detector_xi.py results/stage0_check
"""
import os
import sys

import pandas as pd


def f1(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0, P, R


def main(out):
    cells = [c for c in sorted(os.listdir(out)) if os.path.isdir(os.path.join(out, c)) and c != "logs"]
    order = ["lin_laplace", "cube_laplace", "lin_normal", "cube_normal", "tanh_unit_laplace", "tanh_unit_normal",
             "cube_unit_laplace", "cube_unit_normal"]
    cells = [c for c in order if c in cells] + [c for c in cells if c not in order]
    rows = []
    for c in cells:
        cfgs = [("v1O (today)", "v2_auto")] + [(d.replace("det_", ""), d) for d in sorted(os.listdir(os.path.join(out, c))) if d.startswith("det_")]
        for lab, cfg in cfgs:
            f = os.path.join(out, c, cfg, "results.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f); d = d[d.status == "ok"]
            if d.empty:
                continue
            F, P, R = f1(d.det_tp.sum(), d.det_fp.sum(), d.det_fn.sum())
            C, _, _ = f1(d.chow0001_tp.sum(), d.chow0001_fp.sum(), d.chow0001_fn.sum())
            rows.append({"cell": c, "detector": lab, "n": len(d), "det F1": round(F, 3), "prec": round(P, 2), "rec": round(R, 2),
                         "conf F1": round(C, 3), "windows/regime": round((d.n_structures / d.n_regimes).mean(), 2),
                         "nSHD": round(d.mean_norm_shd.mean(), 3)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "stage0_check"))
