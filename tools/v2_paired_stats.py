#!/usr/bin/env python3
"""Paired statistics for the CausalMorph v2 arms on benchmark v2.
Run: cd Full-Flow && OMP_NUM_THREADS=1 python tools/v2_paired_stats.py
Writes results/hardening_validation/v2_paired_stats.csv (consensus F1) and
v2_paired_stats_wnshd.csv (windowed nSHD) and prints both tables."""
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


# metric -> (arm column, ICA-alone column, DirectLiNGAM-alone column, output file, digits)
METRICS = {
    "f1": ("mean_struct_f1", "icalingam_mean_f1", "directlingam_mean_f1", "v2_paired_stats.csv", 3),
    "wnshd": ("windowed_nshd_weighted", "icalingam_windowed_nshd_weighted", "directlingam_windowed_nshd_weighted",
              "v2_paired_stats_wnshd.csv", 4),
}


def table(metric):
    """Paired comparisons on one metric. For F1 a positive delta favours the first arm;
    for windowed nSHD a negative delta does (win/loss count first-arm higher/lower)."""
    arm_col, ica_col, dl_col, _, _ = METRICS[metric]
    rows = []
    for cell, lb in CELLS:
        L = load(os.path.join(VAL, lb, "results.csv"))
        col = lambda name: load(os.path.join(VAL, name, "results.csv"))[arm_col].reindex(L.index)
        ica, dl = L[ica_col], L[dl_col]
        comps = [("v2+ICA vs ICA alone", col(f"P_v2_ica_{cell}"), ica),
                 ("v2+DL vs DL alone", col(f"P_v2_{cell}"), dl),
                 ("lite+ICA vs ICA alone", col(f"P_lite_ica_{cell}"), ica),
                 ("v2+ICA vs v1+ICA", col(f"P_v2_ica_{cell}"), col(f"P_cold_ica_{cell}")),
                 ("v2+ICA vs lite+ICA", col(f"P_v2_ica_{cell}"), col(f"P_lite_ica_{cell}"))]
        if os.path.exists(os.path.join(VAL, f"P_v2cap_ica_{cell}", "results.csv")):
            comps += [("v2cap+ICA vs ICA alone", col(f"P_v2cap_ica_{cell}"), ica),
                      ("v2cap+DL vs DL alone", col(f"P_v2cap_{cell}"), dl),
                      ("v2cap+ICA vs v2+ICA (Stage I cap)", col(f"P_v2cap_ica_{cell}"), col(f"P_v2_ica_{cell}"))]
        if os.path.exists(os.path.join(VAL, f"P_litecap_ica_{cell}", "results.csv")):
            comps += [("litecap+ICA vs ICA alone", col(f"P_litecap_ica_{cell}"), ica),
                      ("v2cap+ICA vs litecap+ICA", col(f"P_v2cap_ica_{cell}"), col(f"P_litecap_ica_{cell}"))]
        for lab, a, b in comps:
            rows.append((cell, lab) + paired_stats(a, b))
    return pd.DataFrame(rows, columns=["cell", "comparison", "n", "dF1" if metric == "f1" else "dWNSHD", "ci_lo",
                                       "ci_hi", "p_wilcoxon", "win", "loss", "tie", "rank_biserial"])


def main():
    pd.set_option("display.width", 200)
    for metric, (_, _, _, fname, digits) in METRICS.items():
        df = table(metric)
        df.to_csv(os.path.join(VAL, fname), index=False)
        f = ("{:+.%df}" % digits).format
        print(f"\n=== {metric}: " + ("higher is better" if metric == "f1" else "lower is better (negative delta favours the first arm)"))
        print(df.to_string(index=False, formatters={df.columns[3]: f, "ci_lo": f, "ci_hi": f,
              "p_wilcoxon": "{:.1e}".format, "rank_biserial": "{:+.2f}".format}))


if __name__ == "__main__":
    main()
