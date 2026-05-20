#!/usr/bin/env python3
"""
Presentation plots — Slides 16 & 17.

Slide 16  slide16_framework.png
          Clean three-pillar comparison table (no dashboard).

Slide 17  slide17_performance.png
          Two side-by-side bar charts from batch_ablation.csv:
          norm SHD (lower=better) and struct F1 (higher=better)
          for CausalMorph vs DirectLiNGAM vs ICA-LiNGAM.

Run from Full-Flow/:
    python presentation_plots_s16_s17.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "batch_ablation.csv")

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE 16  —  THREE-PILLAR TABLE
# ═══════════════════════════════════════════════════════════════════════════

PILLARS = [
    dict(
        title  = "Gatekeeper v1-F",
        status = "✓  Built & tested",
        sc     = "#1B5E20",        # status text/dot colour
        hc     = "#1565C0",        # header colour
        role   = "Non-stationarity detection",
        items  = [
            "MAD artifact rejection",
            "Rolling moments  (mean · var · skew · kurt)",
            "CWT Morlet scalogram",
            "Fourier surrogate calibration",
            "Signed log-ratio  E⁺(t) − E⁻(t)",
            "Derivative peak detection",
            "K-of-channels consensus gate",
        ],
    ),
    dict(
        title  = "CausalMorph / LiNGAM",
        status = "⚙  Integrated",
        sc     = "#E65100",
        hc     = "#E65100",
        role   = "Regime-wise causal structure learning",
        items  = [
            "CausalMorph preconditioning",
            "DirectLiNGAM  (non-Gaussian ICA)",
            "Warm-start prior chaining",
            "Window overlap (25 %)",
            "Hybrid cold / warm model selection",
            "Residual independence scoring",
        ],
    ),
    dict(
        title  = "nsDCBN / Bayesian aggregation",
        status = "◑  Partial",
        sc     = "#6A1B9A",
        hc     = "#6A1B9A",
        role   = "Consensus structure across regimes",
        items  = [
            "Beta-Bernoulli edge posterior",
            "Sample-size weighted voting",
            "Edge probability heatmap",
            "Threshold consensus graph (0.20)",
            "SHD / F1 evaluation vs ground truth",
        ],
    ),
]

# How far down the y-axis each text element sits (figure coords, top=1)
_HEADER_Y  = 0.90   # header bar top
_STATUS_Y  = 0.77
_ROLE_Y    = 0.72
_ITEMS_Y0  = 0.65   # first bullet
_ITEM_DY   = 0.073  # vertical step per bullet


def plot_slide16(path="slide16_framework.png"):
    N = len(PILLARS)
    fig = plt.figure(figsize=(14, 7.5))
    fig.patch.set_facecolor("white")

    # No visible axes — we draw everything in figure coordinates
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    col_w   = 0.30
    col_gap = (1.0 - N * col_w) / (N + 1)

    # Title
    ax.text(0.5, 0.97, "Integrated Framework — Three Pillars",
            ha="center", va="top", fontsize=15, fontweight="bold",
            color="#1A1A1A")

    for k, p in enumerate(PILLARS):
        x0 = col_gap + k * (col_w + col_gap)
        xc = x0 + col_w / 2

        # ── coloured header bar ──────────────────────────────────────────────
        ax.add_patch(FancyBboxPatch(
            (x0, _HEADER_Y - 0.07), col_w, 0.09,
            boxstyle="round,pad=0.005", lw=0,
            facecolor=p["hc"], transform=ax.transAxes,
        ))
        ax.text(xc, _HEADER_Y - 0.025, p["title"],
                ha="center", va="center", color="white",
                fontsize=11.5, fontweight="bold", transform=ax.transAxes)

        # ── status badge ─────────────────────────────────────────────────────
        ax.text(xc, _STATUS_Y, p["status"],
                ha="center", va="center", color=p["sc"],
                fontsize=10.5, fontweight="bold", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec=p["sc"], lw=1.5, alpha=0.95))

        # ── role label ───────────────────────────────────────────────────────
        ax.text(xc, _ROLE_Y, p["role"],
                ha="center", va="top", color="#444444",
                fontsize=9, style="italic", transform=ax.transAxes)

        # ── separator line ───────────────────────────────────────────────────
        ax.plot([x0 + 0.01, x0 + col_w - 0.01], [_ROLE_Y - 0.025] * 2,
                color="#CCCCCC", lw=0.8, transform=ax.transAxes)

        # ── bullet list ──────────────────────────────────────────────────────
        for i, item in enumerate(p["items"]):
            ax.text(x0 + 0.015, _ITEMS_Y0 - i * _ITEM_DY,
                    f"· {item}",
                    ha="left", va="top", color="#1A1A1A",
                    fontsize=8.8, transform=ax.transAxes)

        # ── thin left border stripe ──────────────────────────────────────────
        ax.plot([x0, x0],
                [0.08, _HEADER_Y - 0.07],
                color=p["hc"], lw=2.5, alpha=0.4,
                transform=ax.transAxes)

    # ── bottom flow line ─────────────────────────────────────────────────────
    ax.text(0.5, 0.04,
            "Detection   →   Per-regime structure learning   →   Bayesian consensus",
            ha="center", va="center", color="#777777",
            fontsize=9.5, style="italic", transform=ax.transAxes)

    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE 17  —  FULL PIPELINE PERFORMANCE  (from batch_ablation.csv)
# ═══════════════════════════════════════════════════════════════════════════

# Detection colour (Gatekeeper stage)
_DET_COLOR = "#1B5E20"

# Method configs reused across stages 2 and 3
METHOD_CFG = [
    dict(label="CausalMorph\n+ DirectLiNGAM",
         nshd_col="causalmorph_mean_norm_shd",
         cons_f1_col="causalmorph_cons_f1",
         color="#1565C0", hatch=""),
    dict(label="DirectLiNGAM\n(no prior)",
         nshd_col="directlingam_mean_norm_shd",
         cons_f1_col="directlingam_cons_f1",
         color="#2E7D32", hatch="//"),
    dict(label="ICA-LiNGAM\n(no prior)",
         nshd_col="icalingam_mean_norm_shd",
         cons_f1_col="icalingam_cons_f1",
         color="#9E9E9E", hatch="xx"),
]


def _bar(ax, x, mu, se, colors, hatches, bar_w=0.55):
    bars = ax.bar(
        x, mu, bar_w,
        yerr=se, capsize=5,
        color=colors, hatch=hatches,
        edgecolor="white", linewidth=0.8, alpha=0.88,
        error_kw=dict(lw=1.6, capthick=1.6, ecolor="#333333"),
    )
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in mu],
                 padding=4, fontsize=9.5, fontweight="bold")
    return bars


def plot_slide17(path="slide17_performance.png"):
    if not os.path.isfile(CSV_PATH):
        print(f"  [warn] {CSV_PATH} not found — skipping slide 17")
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df["status"] == "ok"].copy()
    n  = len(df)
    print(f"  Loaded {n} ok experiments")

    # ── Stage 1 data: precision by number of nodes ───────────────────────────
    det_grp    = df.groupby("p")["det_precision"].agg(["mean", "sem"]).reset_index()
    det_p      = det_grp["p"].tolist()
    det_mu     = det_grp["mean"].tolist()
    det_se     = det_grp["sem"].tolist()
    det_colors = [_DET_COLOR] * len(det_p)

    # ── Stage 2 data: per-regime norm SHD ────────────────────────────────────
    m_labels  = [m["label"]                       for m in METHOD_CFG]
    m_colors  = [m["color"]                       for m in METHOD_CFG]
    m_hatches = [m["hatch"]                       for m in METHOD_CFG]
    nshd_mu   = [df[m["nshd_col"]].mean()         for m in METHOD_CFG]
    nshd_se   = [df[m["nshd_col"]].sem()          for m in METHOD_CFG]

    x3 = np.arange(3)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.44),
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"End-to-end pipeline performance  "
        f"(n = {n} runs,  p = 4–10,  5 regimes,  Laplace noise)",
        fontsize=12.5, fontweight="bold", y=1.03,
    )

    # ── Panel 1: Gatekeeper precision by # nodes ─────────────────────────────
    xd = np.arange(len(det_p))
    _bar(ax1, xd, det_mu, det_se, det_colors, [""] * len(det_p), bar_w=0.6)
    ax1.plot(xd, det_mu, color=_DET_COLOR, lw=1.5, alpha=0.5,
             marker="o", markersize=4, zorder=5)
    ax1.set_xticks(xd)
    ax1.set_xticklabels([str(p) for p in det_p], fontsize=10.5)
    ax1.set_xlabel("# nodes  (p)", fontsize=10.5)
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.set_title("(1) Gatekeeper precision", fontsize=11.5,
                  fontweight="bold", pad=8, color=_DET_COLOR)
    ax1.set_ylim(0, 1.15)
    ax1.axhline(1.0, color="#CCCCCC", lw=0.7, linestyle="--")
    ax1.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax1.set_axisbelow(True)

    # ── Panel 2: per-regime norm SHD ─────────────────────────────────────────
    _bar(ax2, x3, nshd_mu, nshd_se, m_colors, m_hatches)
    # Δ annotation between CausalMorph and best baseline
    if nshd_mu[0] < nshd_mu[1]:
        delta = nshd_mu[1] - nshd_mu[0]
        y_mid = (nshd_mu[0] + nshd_mu[1]) / 2
        ax2.annotate("",
            xy=(0, nshd_mu[0] + 0.004), xytext=(1, nshd_mu[1] - 0.004),
            arrowprops=dict(arrowstyle="<->", color="#1565C0",
                            lw=1.3, mutation_scale=9))
        ax2.text(0.5, y_mid + 0.01, f"Δ {delta:.3f}",
                 ha="center", fontsize=8.5, color="#1565C0", fontweight="bold")
    ax2.set_xticks(x3)
    ax2.set_xticklabels(m_labels, fontsize=9.5)
    ax2.set_ylabel("Mean norm SHD", fontsize=11)
    ax2.set_title("(2) Per-regime nSHD  (lower ↓)", fontsize=11.5,
                  fontweight="bold", pad=8, color="#1A1A1A")
    ax2.set_ylim(0, max(nshd_mu) * 1.5 + 0.02)
    ax2.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax2.set_axisbelow(True)

    # ── pipeline flow arrow ───────────────────────────────────────────────────
    fig.canvas.draw()
    x_left  = ax1.get_position().x1
    x_right = ax2.get_position().x0
    fig.text((x_left + x_right) / 2, 0.48, "→", ha="center", va="center",
             fontsize=20, color="#AAAAAA", fontweight="bold",
             transform=fig.transFigure)

    # ── shared method legend ──────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=m["color"], hatch=m["hatch"],
                       edgecolor="white", label=m["label"].replace("\n", "  "))
        for m in METHOD_CFG
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9.5, framealpha=0.9,
               bbox_to_anchor=(0.62, -0.10))

    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating slide 16 (three pillars)...")
    plot_slide16()

    print("Generating slide 17 (method comparison)...")
    plot_slide17()

    print("\nDone:")
    print("  slide16_framework.png")
    print("  slide17_performance.png")
