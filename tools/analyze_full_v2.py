#!/usr/bin/env python3
"""
Analysis of the full CausalMorph v2 experiments (run_full_experiments.sh).

Layout read: <root>/<cell>/<arm>/results.csv, arms main | lite | v1 | v2.
  main: v2 with the Stage I cap + ICA-LiNGAM (mean_struct_*), with ICA-LiNGAM
        alone (icalingam_*) and DirectLiNGAM alone (directlingam_*) on the same windows
  lite: the same with Stage II off + ICA-LiNGAM (mean_struct_*)
  v1:   published CausalMorph + chain prior + DirectLiNGAM (mean_struct_*)
  v2:   optional, v2 without the Stage I cap + ICA-LiNGAM (mean_struct_*)

Writes to <root>/analysis/: arms.csv, paired.csv and breakdown.csv (both windowed
nSHD and consensus F1, column `metric`), detection.csv, and figures
H_pipeline_before_after.png, I_arms_by_cell[_wnshd].png, J_paired_forest[_wnshd].png.
Cells or arms that have not run yet are skipped, so it can be run mid-way.

    python tools/analyze_full_v2.py                      # the newest results/full_* run
    python tools/analyze_full_v2.py --root results/full_20261011_0300
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
NEW = "v2 + ICA-LiNGAM"                      # with the Stage I cap: every change
LITE = "lite + ICA-LiNGAM"
OCT = "October pipeline (v1 + chain + DirectLiNGAM)"
UNCAPPED = "v2 without Stage I cap + ICA-LiNGAM"
METRICS = [("F1", "mean_struct_f1", "mean_f1"), ("P", "mean_struct_precision", "mean_precision"),
           ("R", "mean_struct_recall", "mean_recall"), ("wnSHD", "windowed_nshd_weighted", "windowed_nshd_weighted"),
           ("trF1", "transition_f1", "transition_f1"), ("skelF1", "mean_skeleton_f1", "mean_skeleton_f1")]
DAGMA = "DAGMA (linear) alone"
CDFZ = "CD-NOD (Fisher-z)"
CDKCI = "CD-NOD (KCI, subsample of scenarios)"


def series(root, cell, exclude=frozenset()):
    """{name: DataFrame with columns F1 P R wnSHD trF1 skelF1} for one cell, on the main arm's
    scenarios minus `exclude` (the tuning sample). Methods that ran on a subset (CD-NOD with KCI)
    carry NaN elsewhere; every paired comparison drops the NaN pairs and reports its n."""
    main = load(os.path.join(root, cell, "main", "results.csv"))
    if main is None or not len(main):
        return None, {}
    main = main[~main.index.isin(exclude)]
    meta = main[["p", "n_regimes", "samples_regime"]]

    def own(d):
        return pd.DataFrame({k: d[c] for k, c, _ in METRICS}).reindex(main.index)

    def learner(prefix):
        return pd.DataFrame({k: main[f"{prefix}_{suffix}"] for k, _, suffix in METRICS if f"{prefix}_{suffix}" in main})

    out = {NEW: own(main)}
    if "icalingam_mean_f1" in main:
        out["ICA-LiNGAM alone"] = learner("icalingam")
    if "directlingam_mean_f1" in main:
        out["DirectLiNGAM alone"] = learner("directlingam")
    if "dagma_mean_f1" in main:
        out[DAGMA] = learner("dagma")
    for arm, name in (("lite", LITE), ("v1", OCT), ("v2", UNCAPPED), ("cdnod_fz", CDFZ), ("cdnod_kci", CDKCI)):
        d = load(os.path.join(root, cell, arm, "results.csv"))
        if d is not None and len(d):
            out[name] = own(d)
    return main.assign(**meta), out


def n_failed(root, cell, arm):
    d = load(os.path.join(root, cell, arm, "results.csv"))
    return None if d is None or "n_transform_failed" not in d else int(d.n_transform_failed.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="run folder; default: the newest results/full_* run")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--exclude_ids", default="results/hardening_validation/sample_ids.txt",
                    help="scenario ids left out of the analysis (the tuning sample); '' keeps all")
    a = ap.parse_args()
    if not a.root:
        from full_status import latest_run
        a.root = latest_run(os.path.join(ROOT, "results"))
        if not a.root:
            raise SystemExit("no results/full_* run found; pass --root")
    ex_path = a.exclude_ids if not a.exclude_ids or os.path.isabs(a.exclude_ids) else os.path.join(ROOT, a.exclude_ids)
    exclude = frozenset(l.strip() for l in open(ex_path) if l.strip()) if ex_path and os.path.exists(ex_path) else frozenset()
    print(f"analysis excludes {len(exclude)} tuning scenarios ({ex_path or 'none'})")
    root = a.root if os.path.isabs(a.root) else os.path.join(ROOT, a.root)
    out = os.path.join(root, "analysis"); os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 220)

    arms_rows, paired_rows, break_rows, det_rows, data = [], [], [], [], {}
    for cell, label in CELLS:
        main_df, s = series(root, cell, exclude)
        if main_df is None:
            print(f"\n{label}: not run yet"); continue
        data[cell] = s
        P, R, F = micro_prf(main_df)
        det_rows.append(dict(cell=cell, n=len(main_df), P=P, R=R, F1=F,
                             under=float((main_df.n_structures < main_df.n_regimes).mean()),
                             over=float((main_df.n_structures > main_df.n_regimes).mean())))
        for name, d in s.items():
            arms_rows.append(dict(cell=cell, arm=name, n=int(d.F1.notna().sum()), **{k: float(d[k].mean()) for k in d}))
            if name == CDKCI:   # its scenarios are a subset: the pipeline on the same ones, for a like-for-like row
                sub = s[NEW].loc[d.F1.dropna().index]
                arms_rows.append(dict(cell=cell, arm=f"{NEW} (same scenarios as KCI)", n=len(sub),
                                      **{k: float(sub[k].mean()) for k in sub}))
        comps = [("v2 vs ICA alone", NEW, "ICA-LiNGAM alone"),
                 ("lite vs ICA alone", LITE, "ICA-LiNGAM alone"),
                 ("v2 vs lite (Stage II effect)", NEW, LITE),
                 ("v2 vs October pipeline", NEW, OCT),
                 ("Stage I cap effect (v2 vs uncapped v2)", NEW, UNCAPPED),
                 ("v2 vs DirectLiNGAM alone", NEW, "DirectLiNGAM alone"),
                 ("v2 vs DAGMA", NEW, DAGMA),
                 ("v2 vs CD-NOD (Fisher-z)", NEW, CDFZ),
                 ("v2 vs CD-NOD (KCI)", NEW, CDKCI)]
        for metric in ("wnSHD", "F1"):          # windowed nSHD first: the headline metric (lower is better)
            for lab, x, y in comps:
                if x in s and y in s:
                    n, dm, lo, hi, p, w, l, t, rb = paired_stats(s[x][metric], s[y][metric])
                    paired_rows.append(dict(metric=metric, cell=cell, comparison=lab, n=n, delta=dm, ci_lo=lo,
                                            ci_hi=hi, p_wilcoxon=p, higher=w, lower=l, tie=t, rank_biserial=rb))
            if "ICA-LiNGAM alone" in s:
                diff = (s[NEW][metric] - s["ICA-LiNGAM alone"][metric]).rename("delta")
                for factor in ("p", "n_regimes", "samples_regime"):
                    for lvl, g in pd.concat([diff, main_df[factor]], axis=1).dropna().groupby(factor):
                        break_rows.append(dict(metric=metric, cell=cell, factor=factor, level=lvl, n=len(g),
                                               delta=float(g.delta.mean()), higher=int((g.delta > 0).sum()),
                                               lower=int((g.delta < 0).sum())))
        fails = {arm: n_failed(root, cell, arm) for arm in ("main", "lite", "v2")}
        print(f"\n{label}  (scenarios: {len(main_df)};  transform fallbacks "
              + " ".join(f"{k}={v}" for k, v in fails.items() if v is not None) + ")")
        print(f"  detection (main arm, raw): P={P:.3f} R={R:.3f} F1={F:.3f}")
        print(pd.DataFrame([r for r in arms_rows if r["cell"] == cell]).drop(columns="cell")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    pd.DataFrame(arms_rows).to_csv(os.path.join(out, "arms.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(os.path.join(out, "paired.csv"), index=False)
    pd.DataFrame(break_rows).to_csv(os.path.join(out, "breakdown.csv"), index=False)
    pd.DataFrame(det_rows).to_csv(os.path.join(out, "detection.csv"), index=False)
    TXT = {"wnSHD": "windowed nSHD (lower is better: negative delta favours the first arm)",
           "F1": "consensus F1 (higher is better)"}
    for metric in ("wnSHD", "F1"):
        rows = [r for r in paired_rows if r["metric"] == metric]
        if rows:
            f = "{:+.4f}".format if metric == "wnSHD" else "{:+.3f}".format
            print(f"\nPaired {TXT[metric]}; Wilcoxon signed-rank on non-zero differences, 95% bootstrap CI:")
            print(pd.DataFrame(rows).drop(columns="metric").to_string(index=False, formatters={
                "delta": f, "ci_lo": f, "ci_hi": f, "p_wilcoxon": "{:.1e}".format, "rank_biserial": "{:+.2f}".format}))
        brows = [r for r in break_rows if r["metric"] == metric]
        if brows:
            b = pd.DataFrame(brows)
            print(f"\nv2 + ICA-LiNGAM minus ICA-LiNGAM alone, mean delta in {metric} by factor level:")
            for factor in ("p", "n_regimes", "samples_regime"):
                t = b[b.factor == factor].pivot(index="level", columns="cell", values="delta")
                print(f"\n  by {factor}\n" + t.to_string(float_format=lambda v: f"{v:+.4f}"))
    if not a.no_figures and data:
        figures(data, pd.DataFrame(paired_rows), out)
        baseline_figures(data, pd.DataFrame(paired_rows), out)
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

    if all(OCT in data[c] for c, _ in cells):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        x = np.arange(len(cells)); w = 0.36
        for ax, (m, title) in zip(axes, [("F1", "consensus F1 (higher is better)"),
                                         ("wnSHD", "windowed nSHD (lower is better)"),
                                         ("trF1", "transition F1 (higher is better)")]):
            before = [data[c][OCT][m].mean() for c, _ in cells]; after = [data[c][NEW][m].mean() for c, _ in cells]
            top = max(before + after)
            for xs, vals, col, lab in ((x - w / 2 - 0.01, before, ORANGE, "October pipeline (v1 + chain + DirectLiNGAM)"),
                                       (x + w / 2 + 0.01, after, BLUE, "New pipeline (v2 with Stage I cap + ICA-LiNGAM)")):
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

    for metric, label, suffix in (("wnSHD", "windowed nSHD (lower is better)", "_wnshd"),
                                  ("F1", "consensus F1 (higher is better)", "")):
        _arms_and_forest(data, paired[paired.metric == metric] if len(paired) else paired, out, cells, metric, label,
                         suffix, plt, (BLUE, ORANGE, AQUA, YELLOW, SURFACE, INK2))


def _arms_and_forest(data, paired, out, cells, metric, label, suffix, plt, colors):
    BLUE, ORANGE, AQUA, YELLOW, SURFACE, INK2 = colors
    arms = [(NEW, BLUE), (LITE, AQUA), ("ICA-LiNGAM alone", YELLOW), (OCT, ORANGE)]
    fig, ax = plt.subplots(figsize=(10, 5))
    h = 0.19; ys = np.arange(len(cells))[::-1]
    for i, (name, col) in enumerate(arms):
        vals = [data[c][name][metric].mean() if name in data[c] else np.nan for c, _ in cells]
        bars = ax.barh(ys + (1.5 - i) * (h + 0.012), vals, h, color=col, label=name)
        for b_, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b_.get_width() * 1.01 + 0.002, b_.get_y() + b_.get_height() / 2,
                        f"{v:.4f}" if metric == "wnSHD" else f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    top = np.nanmax([data[c][n][metric].mean() for c, _ in cells for n, _ in arms if n in data[c]])
    ax.set_yticks(ys, [l for _, l in cells]); ax.set_xlim(0, top * 1.22); ax.set_xlabel(label); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, fontsize=9.5)
    ax.set_title(f"{label.split(' (')[0]} by arm, full grid", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"I_arms_by_cell{suffix}.png"), dpi=200); plt.close(fig)

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
                ax.plot(r.delta, yy, "o", ms=7, color=col, mec=SURFACE, mew=1.5, label=comp if c == cells[0][0] else None)
        ax.axvline(0, color=INK2, lw=1); ax.set_yticks(ys, [l for _, l in cells]); ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"paired difference in {label} per scenario (mean, 95% bootstrap CI)")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), fontsize=9.5)
        ax.set_title(f"Paired differences in {label.split(' (')[0]}, full grid", loc="left", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"J_paired_forest{suffix}.png"), dpi=200); plt.close(fig)


def baseline_figures(data, paired, out):
    """M: pipeline vs per-window learners and CD-NOD (Fisher-z) on three metrics; N: paired windowed
    nSHD differences against every external method (CD-NOD with KCI on its own scenario subset)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
    SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "legend.frameon": False})
    cells = [(c, l) for c, l in CELLS if c in data]
    methods = [(NEW, BLUE), ("ICA-LiNGAM alone", YELLOW), ("DirectLiNGAM alone", AQUA), (DAGMA, ORANGE), (CDFZ, MAGENTA)]
    methods = [(m, col) for m, col in methods if any(m in data[c] for c, _ in cells)]
    ys = np.arange(len(cells))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
    h = 0.82 / len(methods)
    for ax, (key, title) in zip(axes, [("wnSHD", "windowed nSHD (lower is better)"),
                                       ("skelF1", "skeleton F1 per regime (higher is better)"),
                                       ("trF1", "transition F1 (higher is better)")]):
        top = 0.0
        for i, (m, col) in enumerate(methods):
            vals = [data[c][m][key].mean() if m in data[c] and key in data[c][m] else np.nan for c, _ in cells]
            top = np.nanmax([top] + vals)
            bars = ax.barh(ys + ((len(methods) - 1) / 2 - i) * h, vals, h * 0.9, color=col, label=m)
            for b_, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b_.get_width() + 0.006 * (top or 1), b_.get_y() + b_.get_height() / 2, f"{v:.3f}",
                            va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, top * 1.18); ax.set_title(title, loc="left", fontsize=10.5); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [l for _, l in cells])
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, fontsize=9.5)
    fig.suptitle("Pipeline vs per-window learners (same detected windows) and CD-NOD, held-out scenarios",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95)); fig.savefig(os.path.join(out, "M_baselines.png"), dpi=200); plt.close(fig)

    if not len(paired):
        return
    comps = [("v2 vs ICA alone", YELLOW), ("v2 vs DirectLiNGAM alone", AQUA), ("v2 vs DAGMA", ORANGE),
             ("v2 vs CD-NOD (Fisher-z)", MAGENTA), ("v2 vs CD-NOD (KCI)", GREEN)]
    pw = paired[paired.metric == "wnSHD"]
    comps = [(c, col) for c, col in comps if (pw.comparison == c).any()]
    fig, ax = plt.subplots(figsize=(11, 5.8)); step = 0.8 / max(len(comps), 1)
    for i, (comp, col) in enumerate(comps):
        for y, (c, _) in zip(ys, cells):
            r = pw[(pw.cell == c) & (pw.comparison == comp)]
            if not len(r):
                continue
            r = r.iloc[0]; yy = y + ((len(comps) - 1) / 2 - i) * step
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.delta, yy, "o", ms=6.5, color=col, mec=SURFACE, mew=1.3,
                    label=f"pipeline − {comp[6:]}" if c == cells[0][0] else None)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(ys, [l for _, l in cells]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired difference in windowed nSHD per scenario (mean, 95% bootstrap CI); negative = pipeline better")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=9.5)
    ax.set_title("Pipeline minus each method, held-out scenarios (KCI: its scenario subset)", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out, "N_baselines_paired_wnshd.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
