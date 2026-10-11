#!/usr/bin/env python3
"""
The pipeline against external structure learners on the 600-scenario benchmark
(four cells), all scored with the same metric functions and paired by scenario.

  pipeline             M_<cell>: v2 with the Stage I cap + ICA-LiNGAM (every change)
  ICA-LiNGAM alone     M_<cell> icalingam_* columns (same detected windows)
  DirectLiNGAM alone   M_<cell> directlingam_* columns (same detected windows)
  DAGMA                DAGMAs_<cell> dagma_* columns (linear DAGMA on the window rescaled by one global factor)
  CD-NOD (Fisher-z)    CDNOD_fz_<cell>: one graph for the whole series, C = time index
  CD-NOD (KCI)         CDNOD_kci_<cell>: same, KCI on a 500-row time-uniform subsample
                       (only the scenarios that have run; the paired n is printed)

    cd Full-Flow && OMP_NUM_THREADS=1 python tools/plot_baselines.py

Writes figures/M_baselines.png (windowed nSHD, skeleton F1, transition F1),
figures/N_baselines_paired_wnshd.png (paired differences, pipeline minus each
method) and baselines_paired_wnshd.csv.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from analyze_hardening_validation import load  # noqa: E402
from v2_paired_stats import paired_stats  # noqa: E402

VAL = os.path.join(ROOT, "results", "hardening_validation")
OUT = os.path.join(VAL, "figures")
CELLS = [("lin_laplace", "linear / Laplace"), ("nonlin_laplace", "nonlinear / Laplace"),
         ("lin_normal", "linear / Gaussian"), ("nonlin_normal", "nonlinear / Gaussian")]
# categorical slots in fixed order, validated with the dataviz palette validator (light surface);
# slots 3–5 are below 3:1 on the surface, so every bar carries its value as a label
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
METHODS = [("Pipeline (v2 with Stage I cap + ICA-LiNGAM)", BLUE), ("ICA-LiNGAM alone", YELLOW),
           ("DirectLiNGAM alone", AQUA), ("DAGMA (linear)", ORANGE), ("CD-NOD (Fisher-z)", MAGENTA),
           ("CD-NOD (KCI, 500-row subsample)", GREEN)]
PIPE = METHODS[0][0]
MIN_COVERAGE = 0.99   # a method is shown in a cell only once it covers the pipeline's scenarios there
METRICS = [("wnSHD", "windowed nSHD (lower is better)"), ("skelF1", "skeleton F1 per regime (higher is better)"),
           ("trF1", "transition F1 (higher is better)")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _frame(df, prefix):
    if prefix:
        cols = {"wnSHD": f"{prefix}_windowed_nshd_weighted", "skelF1": f"{prefix}_mean_skeleton_f1",
                "trF1": f"{prefix}_transition_f1", "F1": f"{prefix}_mean_f1"}
    else:
        cols = {"wnSHD": "windowed_nshd_weighted", "skelF1": "mean_skeleton_f1", "trF1": "transition_f1",
                "F1": "mean_struct_f1"}
    return pd.DataFrame({k: df[c] for k, c in cols.items() if c in df})


def cell_data(cell):
    """{method label: DataFrame(wnSHD, skelF1, trF1, F1)} indexed by scenario_id; pipeline's scenarios."""
    M = load(os.path.join(VAL, f"M_{cell}", "results.csv"))
    if M is None or not len(M):
        return {}
    out = {PIPE: _frame(M, None), "ICA-LiNGAM alone": _frame(M, "icalingam"),
           "DirectLiNGAM alone": _frame(M, "directlingam")}
    for label, row, prefix in (("DAGMA (linear)", f"DAGMAs_{cell}", "dagma"),
                               ("CD-NOD (Fisher-z)", f"CDNOD_fz_{cell}", None),
                               ("CD-NOD (KCI, 500-row subsample)", f"CDNOD_kci_{cell}", None)):
        d = load(os.path.join(VAL, row, "results.csv"))
        if d is not None and len(d):
            f = _frame(d, prefix)
            f = f[f.index.isin(M.index)]
            if len(f) >= MIN_COVERAGE * len(M):
                out[label] = f
            else:
                print(f"  {cell}: {label} not shown yet ({len(f)} of {len(M)} scenarios)")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {c: cell_data(c) for c, _ in CELLS}
    cells = [(c, l) for c, l in CELLS if data[c]]
    if not cells:
        sys.exit("no M_<cell> rows yet")

    # ── M: three metrics, one panel each ──────────────────────────────────────
    present = [(m, col) for m, col in METHODS if any(m in data[c] for c, _ in cells)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
    h = 0.82 / len(present); ys = np.arange(len(cells))[::-1]
    for ax, (key, title) in zip(axes, METRICS):
        top = 0.0
        for i, (m, col) in enumerate(present):
            vals = [data[c][m][key].mean() if m in data[c] and key in data[c][m] else np.nan for c, _ in cells]
            top = np.nanmax([top] + vals)
            bars = ax.barh(ys + ((len(present) - 1) / 2 - i) * h, vals, h * 0.9, color=col, label=m)
            for b_, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b_.get_width() + 0.006 * (top or 1), b_.get_y() + b_.get_height() / 2,
                            f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, top * 1.18); ax.set_title(title, loc="left", fontsize=10.5); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [l for _, l in cells])
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, fontsize=9.5)
    fig.suptitle("Pipeline against per-window learners on the same detected windows and against CD-NOD, "
                 "same 600 scenarios per cell (a method appears once it has run on all of them)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(os.path.join(OUT, "M_baselines.png"), dpi=200); plt.close(fig)

    # ── paired differences in windowed nSHD: pipeline minus each method ──────
    rows = []
    for c, _ in cells:
        p = data[c][PIPE]
        for m, _col in METHODS[1:]:
            if m in data[c]:
                q = data[c][m]["wnSHD"]
                common = q.dropna().index.intersection(p.wnSHD.dropna().index)
                n, d, lo, hi, pv, w, l, t, rb = paired_stats(p.wnSHD.loc[common], q.loc[common])
                rows.append(dict(cell=c, method=m, n=n, delta=d, ci_lo=lo, ci_hi=hi, p_wilcoxon=pv,
                                 pipeline_better=l, pipeline_worse=w, ties=t, rank_biserial=rb))
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(VAL, "baselines_paired_wnshd.csv"), index=False)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    others = [(m, col) for m, col in METHODS[1:] if (st.method == m).any()]
    step = 0.8 / len(others)
    for i, (m, col) in enumerate(others):
        for y, (c, _) in zip(ys, cells):
            r = st[(st.cell == c) & (st.method == m)]
            if not len(r):
                continue
            r = r.iloc[0]; yy = y + ((len(others) - 1) / 2 - i) * step
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.delta, yy, "o", ms=6.5, color=col, mec=SURFACE, mew=1.3, label=f"pipeline − {m}" if c == cells[0][0] else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ys, [l for _, l in cells]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired difference in windowed nSHD per scenario (mean, 95% bootstrap CI); negative = pipeline better")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=9.5)
    ax.set_title("Pipeline minus each method, same scenarios", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "N_baselines_paired_wnshd.png"), dpi=200); plt.close(fig)

    pd.set_option("display.width", 220)
    print(st.to_string(index=False, formatters={"delta": "{:+.4f}".format, "ci_lo": "{:+.4f}".format,
          "ci_hi": "{:+.4f}".format, "p_wilcoxon": "{:.1e}".format, "rank_biserial": "{:+.2f}".format}))
    print(os.path.join(OUT, "M_baselines.png")); print(os.path.join(OUT, "N_baselines_paired_wnshd.png"))


if __name__ == "__main__":
    main()
