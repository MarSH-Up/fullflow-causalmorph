#!/usr/bin/env python3
"""
Figures comparing CausalMorph v2 with the October pipeline, the published
Stage II (v1), the lite ablation and the learners alone.

Reads the benchmark rows in results/hardening_validation/, the paired
statistics CSV (tools/v2_paired_stats.py) and the stationary replication CSV
(tools/kbs_replication.py); writes PNGs to results/hardening_validation/figures/.

    cd Full-Flow && OMP_NUM_THREADS=1 python tools/plot_v2_comparison.py

H_pipeline_before_after.png  October pipeline vs v2 + ICA-LiNGAM, three metrics per cell
I_arms_by_cell.png           consensus F1 of the four arms, one panel per learner
J_paired_forest.png          paired ΔF1 with 95% CI per cell
K_stationary.png             stationary replication, true vs estimated prior (ICA-LiNGAM)
L_stage1_cap.png             learner alone vs v2 vs v2 with the Stage I cap (when the v2cap rows exist)

When the v2cap rows exist, H's "new pipeline" is v2 with the Stage I cap (every change).
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
sys.path.insert(0, ROOT)
from analyze_hardening_validation import load  # noqa: E402

VAL = os.path.join(ROOT, "results", "hardening_validation")
OUT = os.path.join(VAL, "figures")
STATIONARY = os.path.join(ROOT, "results", "kbs_replication", "kbs_replication_arms.csv")

CELLS = [("lin_laplace", "LB_v1L25_v2", "linear\nLaplace"), ("nonlin_laplace", "LB_nonlin_laplace", "nonlinear\nLaplace"),
         ("lin_normal", "LB_lin_normal", "linear\nGaussian"), ("nonlin_normal", "LB_nonlin_normal", "nonlinear\nGaussian")]

# categorical slots in fixed order (validated: dataviz validate_palette.js, light surface);
# slots 3–4 sit below 3:1 on the surface, so every bar carries its value as a label
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def _row(name):
    return load(os.path.join(VAL, name, "results.csv"))


def _paired(cell, lb, row):
    """Restrict an arm's rows to the LB row's scenarios (same 600 ids)."""
    d = _row(row)
    return d[d.index.isin(lb.index)]


def _label_hbars(ax, bars, values, fmt="{:.3f}"):
    for b, v in zip(bars, values):
        ax.text(b.get_width() + 0.008, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", ha="left",
                fontsize=8.5, color=INK2)


def _has(row):
    return os.path.exists(os.path.join(VAL, row, "results.csv"))


def _new_row(cell):
    return f"P_v2cap_ica_{cell}" if _has(f"P_v2cap_ica_{cell}") else f"P_v2_ica_{cell}"


def fig_before_after():
    metrics = [("consensus F1", "mean_struct_f1", "f1", True),
               ("windowed nSHD", "windowed_nshd_weighted", "windowed_nshd_weighted", False),
               ("transition F1", "transition_f1", "transition_f1", True)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    x = np.arange(len(CELLS)); w = 0.36
    for ax, (title, col, _, hib) in zip(axes, metrics):
        before, after = [], []
        for cell, lbn, _ in CELLS:
            lb = _row(lbn)
            before.append(lb[col].mean())                              # October pipeline: v1 + chain + DirectLiNGAM
            after.append(_paired(cell, lb, _new_row(cell))[col].mean())
        b1 = ax.bar(x - w / 2 - 0.01, before, w, color=ORANGE, label="October pipeline (v1 + chain + DirectLiNGAM)")
        capped = all(_has(f"P_v2cap_ica_{c}") for c, _, _ in CELLS)
        b2 = ax.bar(x + w / 2 + 0.01, after, w, color=BLUE,
                    label="New pipeline (v2 with Stage I cap + ICA-LiNGAM)" if capped else "New pipeline (v2 + ICA-LiNGAM)")
        top = max(before + after)
        for bars, vals in ((b1, before), (b2, after)):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.015, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8.5, color=INK2)
        ax.set_xticks(x, [c[2] for c in CELLS])
        ax.set_ylim(0, top * 1.15)
        ax.set_title(f"{title} ({'higher' if hib else 'lower'} is better)", loc="left")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, fontsize=9.5)
    fig.suptitle("Pipeline before and after, same v1L windows, 600 paired scenarios per cell", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(OUT, "H_pipeline_before_after.png"), dpi=200)
    plt.close(fig)


def fig_arms():
    arms = [("v2", "P_v2", BLUE), ("v1 (published Stage II)", "P_cold", ORANGE), ("lite (Stage II off)", "P_lite", AQUA),
            ("learner alone", None, YELLOW)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (learner, suffix, alone_col) in zip(axes, [("ICA-LiNGAM", "_ica", "icalingam_mean_f1"),
                                                       ("DirectLiNGAM", "", "directlingam_mean_f1")]):
        h = 0.19
        ys = np.arange(len(CELLS))[::-1]
        for i, (lab, prefix, col) in enumerate(arms):
            vals = []
            for cell, lbn, _ in CELLS:
                lb = _row(lbn)
                vals.append(lb[alone_col].mean() if prefix is None else _paired(cell, lb, f"{prefix}{suffix}_{cell}").mean_struct_f1.mean())
            bars = ax.barh(ys + (1.5 - i) * (h + 0.012), vals, h, color=col, label=lab)
            _label_hbars(ax, bars, vals)
        ax.set_yticks(ys, [c[2].replace("\n", " / ") for c in CELLS])
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("consensus F1")
        ax.set_title(learner, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4, fontsize=9.5)
    fig.suptitle("Consensus F1 by preconditioner, same windows, 600 paired scenarios per cell", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(OUT, "I_arms_by_cell.png"), dpi=200)
    plt.close(fig)


def fig_forest():
    st = pd.read_csv(os.path.join(VAL, "v2_paired_stats.csv"))
    comps = [("v2+ICA vs ICA alone", "v2 + ICA-LiNGAM vs ICA-LiNGAM alone", BLUE),
             ("lite+ICA vs ICA alone", "lite + ICA-LiNGAM vs ICA-LiNGAM alone", AQUA),
             ("v2+ICA vs lite+ICA", "v2 vs lite, ICA-LiNGAM (the Stage II effect)", ORANGE)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    step = 0.24
    ys = np.arange(len(CELLS))[::-1]
    for i, (comp, lab, col) in enumerate(comps):
        for y, (cell, _, _) in zip(ys, CELLS):
            r = st[(st.cell == cell) & (st.comparison == comp)].iloc[0]
            yy = y + (1 - i) * step
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.dF1, yy, "o", ms=7, color=col, mec=SURFACE, mew=1.5, label=lab if y == ys[0] else None)
            if comp == "v2+ICA vs ICA alone":
                ax.text(max(r.ci_hi, 0) + 0.004, yy, f"{r.dF1:+.3f}  r={r.rank_biserial:+.2f}", va="center",
                        fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ys, [c[2].replace("\n", " / ") for c in CELLS])
    ax.set_xlabel("paired ΔF1 per scenario (mean, 95% bootstrap CI)")
    ax.set_xlim(-0.07, 0.10)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=1, fontsize=9.5)
    ax.set_title("Where v2 helps: paired differences, n = 600 per cell", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "J_paired_forest.png"), dpi=200)
    plt.close(fig)


def fig_stationary():
    k = pd.read_csv(STATIONARY)
    blocks = [(("linear", "laplace"), "linear / Laplace (45)"), (("nonlinear", "laplace"), "nonlinear / Laplace (12)"),
              (("linear", "normal"), "linear / Gaussian (12)")]
    arms = [("v2", BLUE), ("v1", ORANGE), ("lite", AQUA), ("none", YELLOW)]
    names = {"v2": "v2", "v1": "v1 (published Stage II)", "lite": "lite (Stage II off)", "none": "learner alone"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (prior, title) in zip(axes, [("true", "true graph as prior"), ("est", "learner's own estimate as prior")]):
        h = 0.19
        ys = np.arange(len(blocks))[::-1]
        for i, (arm, col) in enumerate(arms):
            vals = [k[(k["mode"] == m) & (k["dist"] == d)][f"icalingam_{prior}_{arm}"].mean() for (m, d), _ in blocks]
            bars = ax.barh(ys + (1.5 - i) * (h + 0.012), vals, h, color=col, label=names[arm])
            _label_hbars(ax, bars, vals)
        ax.set_yticks(ys, [b[1] for b in blocks])
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("F1, ICA-LiNGAM")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=4, fontsize=9.5)
    fig.suptitle("Stationary replication (one regime per run; runs per block in brackets)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(OUT, "K_stationary.png"), dpi=200)
    plt.close(fig)


def fig_stage1_cap():
    if not all(_has(f"P_v2cap_ica_{c}") and _has(f"P_v2cap_{c}") for c, _, _ in CELLS):
        return None
    series = [("learner alone", None, YELLOW), ("v2", "P_v2", BLUE), ("v2 with Stage I cap", "P_v2cap", MAGENTA)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (learner, suffix, alone_col) in zip(axes, [("ICA-LiNGAM", "_ica", "icalingam_mean_f1"),
                                                       ("DirectLiNGAM", "", "directlingam_mean_f1")]):
        h = 0.25
        ys = np.arange(len(CELLS))[::-1]
        for i, (lab, prefix, col) in enumerate(series):
            vals = []
            for cell, lbn, _ in CELLS:
                lb = _row(lbn)
                vals.append(lb[alone_col].mean() if prefix is None else _paired(cell, lb, f"{prefix}{suffix}_{cell}").mean_struct_f1.mean())
            bars = ax.barh(ys + (1 - i) * (h + 0.012), vals, h, color=col, label=lab)
            _label_hbars(ax, bars, vals)
        ax.set_yticks(ys, [c[2].replace("\n", " / ") for c in CELLS])
        ax.set_xlim(0, 1.08); ax.set_xlabel("consensus F1"); ax.set_title(learner, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, fontsize=9.5)
    fig.suptitle("Effect of the Stage I degree cap, same windows, 600 paired scenarios per cell", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(OUT, "L_stage1_cap.png"), dpi=200)
    plt.close(fig)
    return "L_stage1_cap.png"


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_before_after(); fig_arms(); fig_forest(); fig_stationary()
    extra = fig_stage1_cap()
    for f in ("H_pipeline_before_after.png", "I_arms_by_cell.png", "J_paired_forest.png", "K_stationary.png") + ((extra,) if extra else ()):
        print(os.path.join(OUT, f))


if __name__ == "__main__":
    main()
