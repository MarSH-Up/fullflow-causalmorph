#!/usr/bin/env python3
"""
Analysis of the full CausalMorph v2 experiments (run_full_experiments.sh).

Layout read: <root>/<cell>/<arm>/results.csv, arms main | lite | v1.
  main: v2 + ICA-LiNGAM (mean_struct_*), with ICA-LiNGAM alone (icalingam_*) and
        DirectLiNGAM alone (directlingam_*) on the same windows
  lite: v2 with Stage II off + ICA-LiNGAM (mean_struct_*)
  v1:   published CausalMorph + chain prior + DirectLiNGAM (mean_struct_*)

Writes to <root>/analysis/: arms.csv, paired.csv, breakdown.csv, detection.csv
and figures H_pipeline_before_after.png, I_arms_by_cell.png, J_paired_forest.png.
Cells or arms that have not run yet are skipped, so it can be run mid-way.

    python tools/analyze_full_v2.py --root results/full_v2
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from analyze_hardening_validation import load, micro_prf  # noqa: E402
from v2_paired_stats import paired_stats  # noqa: E402

CELLS = [("lin_laplace", "linear / Laplace"), ("nonlin_laplace", "nonlinear / Laplace"),
         ("lin_normal", "linear / Gaussian"), ("nonlin_normal", "nonlinear / Gaussian")]
METRICS = [("F1", "mean_struct_f1", "mean_f1"), ("P", "mean_struct_precision", "mean_precision"),
           ("R", "mean_struct_recall", "mean_recall"), ("wnSHD", "windowed_nshd_weighted", "windowed_nshd_weighted"),
           ("trF1", "transition_f1", "transition_f1")]


def series(root, cell):
    """{name: DataFrame with columns F1 P R wnSHD trF1 (+ meta)} for one cell, on the main arm's scenarios."""
    main = load(os.path.join(root, cell, "main", "results.csv"))
    if main is None or not len(main):
        return None, {}
    meta = main[["p", "n_regimes", "samples_regime"]]

    def own(d):
        return pd.DataFrame({k: d[c] for k, c, _ in METRICS}).reindex(main.index)

    def learner(prefix):
        return pd.DataFrame({k: main[f"{prefix}_{suffix}"] for k, _, suffix in METRICS if f"{prefix}_{suffix}" in main})

    out = {"v2 + ICA-LiNGAM": own(main)}
    if "icalingam_mean_f1" in main:
        out["ICA-LiNGAM alone"] = learner("icalingam")
    if "directlingam_mean_f1" in main:
        out["DirectLiNGAM alone"] = learner("directlingam")
    for arm, name in (("lite", "lite + ICA-LiNGAM"), ("v1", "October pipeline (v1 + chain + DirectLiNGAM)")):
        d = load(os.path.join(root, cell, arm, "results.csv"))
        if d is not None and len(d):
            out[name] = own(d)
    return main.assign(**meta), out


def n_failed(root, cell, arm):
    d = load(os.path.join(root, cell, arm, "results.csv"))
    return None if d is None or "n_transform_failed" not in d else int(d.n_transform_failed.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/full_v2")
    ap.add_argument("--no_figures", action="store_true")
    a = ap.parse_args()
    root = a.root if os.path.isabs(a.root) else os.path.join(ROOT, a.root)
    out = os.path.join(root, "analysis"); os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 220)

    arms_rows, paired_rows, break_rows, det_rows, data = [], [], [], [], {}
    for cell, label in CELLS:
        main_df, s = series(root, cell)
        if main_df is None:
            print(f"\n{label}: not run yet"); continue
        data[cell] = s
        P, R, F = micro_prf(main_df)
        det_rows.append(dict(cell=cell, n=len(main_df), P=P, R=R, F1=F,
                             under=float((main_df.n_structures < main_df.n_regimes).mean()),
                             over=float((main_df.n_structures > main_df.n_regimes).mean())))
        for name, d in s.items():
            arms_rows.append(dict(cell=cell, arm=name, n=int(d.F1.notna().sum()), **{k: float(d[k].mean()) for k in d}))
        comps = [("v2 vs ICA alone", "v2 + ICA-LiNGAM", "ICA-LiNGAM alone"),
                 ("lite vs ICA alone", "lite + ICA-LiNGAM", "ICA-LiNGAM alone"),
                 ("v2 vs lite (Stage II effect)", "v2 + ICA-LiNGAM", "lite + ICA-LiNGAM"),
                 ("v2 vs October pipeline", "v2 + ICA-LiNGAM", "October pipeline (v1 + chain + DirectLiNGAM)")]
        for lab, x, y in comps:
            if x in s and y in s:
                n, dm, lo, hi, p, w, l, t, rb = paired_stats(s[x].F1, s[y].F1)
                paired_rows.append(dict(cell=cell, comparison=lab, n=n, dF1=dm, ci_lo=lo, ci_hi=hi, p_wilcoxon=p,
                                        win=w, loss=l, tie=t, rank_biserial=rb))
        if "ICA-LiNGAM alone" in s:
            diff = (s["v2 + ICA-LiNGAM"].F1 - s["ICA-LiNGAM alone"].F1).rename("dF1")
            for factor in ("p", "n_regimes", "samples_regime"):
                for lvl, g in pd.concat([diff, main_df[factor]], axis=1).dropna().groupby(factor):
                    break_rows.append(dict(cell=cell, factor=factor, level=lvl, n=len(g), dF1=float(g.dF1.mean()),
                                           win=int((g.dF1 > 0).sum()), loss=int((g.dF1 < 0).sum())))
        fails = {arm: n_failed(root, cell, arm) for arm in ("main", "lite")}
        print(f"\n{label}  (scenarios: {len(main_df)};  transform fallbacks main={fails['main']} lite={fails['lite']})")
        print(f"  detection v1L: P={P:.3f} R={R:.3f} F1={F:.3f}")
        print(pd.DataFrame([r for r in arms_rows if r["cell"] == cell]).drop(columns="cell")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    pd.DataFrame(arms_rows).to_csv(os.path.join(out, "arms.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(os.path.join(out, "paired.csv"), index=False)
    pd.DataFrame(break_rows).to_csv(os.path.join(out, "breakdown.csv"), index=False)
    pd.DataFrame(det_rows).to_csv(os.path.join(out, "detection.csv"), index=False)
    if paired_rows:
        print("\nPaired consensus F1 (Wilcoxon signed-rank on non-zero differences, 95% bootstrap CI):")
        print(pd.DataFrame(paired_rows).to_string(index=False, formatters={
            "dF1": "{:+.3f}".format, "ci_lo": "{:+.3f}".format, "ci_hi": "{:+.3f}".format,
            "p_wilcoxon": "{:.1e}".format, "rank_biserial": "{:+.2f}".format}))
    if break_rows:
        b = pd.DataFrame(break_rows)
        print("\nv2 + ICA-LiNGAM minus ICA-LiNGAM alone, mean ΔF1 by factor level:")
        for factor in ("p", "n_regimes", "samples_regime"):
            t = b[b.factor == factor].pivot(index="level", columns="cell", values="dF1")
            print(f"\n  by {factor}\n" + t.to_string(float_format=lambda v: f"{v:+.3f}"))
    if not a.no_figures and data:
        figures(data, pd.DataFrame(paired_rows), out)
    print(f"\nwritten to {out}")


def figures(data, paired, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "legend.frameon": False})
    cells = [(c, l) for c, l in CELLS if c in data]
    ticks = [l.replace(" / ", "\n") for _, l in cells]
    OCT, NEW = "October pipeline (v1 + chain + DirectLiNGAM)", "v2 + ICA-LiNGAM"

    if all(OCT in data[c] for c, _ in cells):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        x = np.arange(len(cells)); w = 0.36
        for ax, (m, title) in zip(axes, [("F1", "consensus F1 (higher is better)"),
                                         ("wnSHD", "windowed nSHD (lower is better)"),
                                         ("trF1", "transition F1 (higher is better)")]):
            before = [data[c][OCT][m].mean() for c, _ in cells]; after = [data[c][NEW][m].mean() for c, _ in cells]
            top = max(before + after)
            for xs, vals, col, lab in ((x - w / 2 - 0.01, before, ORANGE, "October pipeline (v1 + chain + DirectLiNGAM)"),
                                       (x + w / 2 + 0.01, after, BLUE, "New pipeline (v2 + ICA-LiNGAM)")):
                bars = ax.bar(xs, vals, w, color=col, label=lab)
                for b_, v in zip(bars, vals):
                    ax.text(b_.get_x() + b_.get_width() / 2, v + top * 0.015, f"{v:.2f}", ha="center", fontsize=8.5,
                            color=INK2)
            ax.set_xticks(x, ticks); ax.set_ylim(0, top * 1.15); ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, fontsize=9.5)
        fig.suptitle("Pipeline before and after, full grid", x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig(os.path.join(out, "H_pipeline_before_after.png"), dpi=200)
        plt.close(fig)

    arms = [(NEW, BLUE), ("lite + ICA-LiNGAM", AQUA), ("ICA-LiNGAM alone", YELLOW), (OCT, ORANGE)]
    fig, ax = plt.subplots(figsize=(10, 5))
    h = 0.19; ys = np.arange(len(cells))[::-1]
    for i, (name, col) in enumerate(arms):
        vals = [data[c][name].F1.mean() if name in data[c] else np.nan for c, _ in cells]
        bars = ax.barh(ys + (1.5 - i) * (h + 0.012), vals, h, color=col, label=name)
        for b_, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b_.get_width() + 0.008, b_.get_y() + b_.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ys, [l for _, l in cells]); ax.set_xlim(0, 1.1); ax.set_xlabel("consensus F1"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, fontsize=9.5)
    ax.set_title("Consensus F1 by arm, full grid", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out, "I_arms_by_cell.png"), dpi=200); plt.close(fig)

    if len(paired):
        comps = [("v2 vs ICA alone", BLUE), ("lite vs ICA alone", AQUA), ("v2 vs lite (Stage II effect)", ORANGE)]
        fig, ax = plt.subplots(figsize=(10, 4.8)); step = 0.24
        for i, (comp, col) in enumerate(comps):
            for y, (c, _) in zip(ys, cells):
                r = paired[(paired.cell == c) & (paired.comparison == comp)]
                if not len(r):
                    continue
                r = r.iloc[0]; yy = y + (1 - i) * step
                ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
                ax.plot(r.dF1, yy, "o", ms=7, color=col, mec=SURFACE, mew=1.5, label=comp if c == cells[0][0] else None)
        ax.axvline(0, color=INK2, lw=1); ax.set_yticks(ys, [l for _, l in cells]); ax.grid(axis="y", visible=False)
        ax.set_xlabel("paired ΔF1 per scenario (mean, 95% bootstrap CI)")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), fontsize=9.5)
        ax.set_title("Paired differences, full grid", loc="left", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(out, "J_paired_forest.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
