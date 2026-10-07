#!/usr/bin/env python3
"""Paired statistics for the CausalMorph v2 arms on benchmark v2.
Run: cd Full-Flow && OMP_NUM_THREADS=1 python tools/v2_paired_stats.py
Writes results/hardening_validation/v2_paired_stats.csv and prints the table."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scipy.stats import wilcoxon
from analyze_hardening_validation import load

VAL = os.path.join(ROOT, "results", "hardening_validation")
CELLS = [("lin_laplace", "LB_v1L25_v2"), ("nonlin_laplace", "LB_nonlin_laplace"),
         ("lin_normal", "LB_lin_normal"), ("nonlin_normal", "LB_nonlin_normal")]
rng = np.random.default_rng(0)


def paired_stats(a, b, n_boot=2000):
    d = (a - b).dropna(); n = len(d); nz = d[d != 0]
    p = wilcoxon(nz).pvalue if len(nz) else 1.0
    boot = [d.sample(n, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    return n, d.mean(), lo, hi, p, w, l, t, (w - l) / max(w + l, 1)


def main():
    rows = []
    for cell, lb in CELLS:
        L = load(os.path.join(VAL, lb, "results.csv"))
        col = lambda name: load(os.path.join(VAL, name, "results.csv")).mean_struct_f1.reindex(L.index)
        comps = [("v2+ICA vs ICA alone", col(f"P_v2_ica_{cell}"), L.icalingam_mean_f1),
                 ("v2+DL vs DL alone", col(f"P_v2_{cell}"), L.directlingam_mean_f1),
                 ("lite+ICA vs ICA alone", col(f"P_lite_ica_{cell}"), L.icalingam_mean_f1),
                 ("v2+ICA vs v1+ICA", col(f"P_v2_ica_{cell}"), col(f"P_cold_ica_{cell}")),
                 ("v2+ICA vs lite+ICA", col(f"P_v2_ica_{cell}"), col(f"P_lite_ica_{cell}"))]
        for lab, a, b in comps:
            rows.append((cell, lab) + paired_stats(a, b))
    df = pd.DataFrame(rows, columns=["cell", "comparison", "n", "dF1", "ci_lo", "ci_hi", "p_wilcoxon",
                                     "win", "loss", "tie", "rank_biserial"])
    df.to_csv(os.path.join(VAL, "v2_paired_stats.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, formatters={"dF1": "{:+.3f}".format, "ci_lo": "{:+.3f}".format,
          "ci_hi": "{:+.3f}".format, "p_wilcoxon": "{:.1e}".format, "rank_biserial": "{:+.2f}".format}))


if __name__ == "__main__":
    main()
