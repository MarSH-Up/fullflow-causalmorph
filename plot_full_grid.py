#!/usr/bin/env python3
"""
Every figure for the full-grid extraction ablation and the method comparison.

Reads the two run_experiments.py arms, reuses compare_full_grid.py for the
paired statistics, and writes publication-quality PNGs to --out.

Palette follows the deck convention (proposed = blue, baseline = orange). Both
the four-method and the five-metric categorical sets pass the dataviz validator
on a light surface (worst adjacent pair #0f8a7a/#eb6834, protan dE 9.1; worst
normal-vision pair dE 16.8).

Usage:
    python plot_full_grid.py \
        --baseline results/results_october_5/v1h_full_baseline/results.csv \
        --winning  results/results_october_5/v1h_full_winning/results.csv \
        --out      results/results_october_5/figures
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_full_grid import load_pair, headline_table, stratified_table, METRICS

# ── palette ───────────────────────────────────────────────────────────────────

BLUE, ORANGE, TEAL, PURPLE = "#2a78d6", "#eb6834", "#0f8a7a", "#8b5bd6"
MAGENTA = "#d1458f"   # fifth categorical slot, validated with the other four
SURFACE, INK, INK_2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, GOOD = "#e1e0d9", "#c3c2b7", "#006300"

METHOD_COLOR = {"Pipeline": BLUE, "CausalMorph": ORANGE,
                "DirectLiNGAM": TEAL, "ICA-LiNGAM": PURPLE}

# Per-method column names, same metric in the same slot.
METHOD_COLS = {
    "Pipeline":     dict(nshd="mean_norm_shd", f1="mean_struct_f1",
                         prec="mean_struct_precision", rec="mean_struct_recall",
                         cons_f1="consensus_f1", wnshd="windowed_nshd_weighted",
                         tshd="transition_shd", dyn="dynamic_score"),
    "CausalMorph":  dict(nshd="causalmorph_mean_norm_shd", f1="causalmorph_mean_f1",
                         prec="causalmorph_mean_precision", rec="causalmorph_mean_recall",
                         cons_f1="causalmorph_cons_f1", wnshd="causalmorph_windowed_nshd_weighted",
                         tshd="causalmorph_transition_shd", dyn="causalmorph_dynamic_score"),
    "DirectLiNGAM": dict(nshd="directlingam_mean_norm_shd", f1="directlingam_mean_f1",
                         prec="directlingam_mean_precision", rec="directlingam_mean_recall",
                         cons_f1="directlingam_cons_f1", wnshd="directlingam_windowed_nshd_weighted",
                         tshd="directlingam_transition_shd", dyn="directlingam_dynamic_score"),
    "ICA-LiNGAM":   dict(nshd="icalingam_mean_norm_shd", f1="icalingam_mean_f1",
                         prec="icalingam_mean_precision", rec="icalingam_mean_recall",
                         cons_f1="icalingam_cons_f1", wnshd="icalingam_windowed_nshd_weighted",
                         tshd="icalingam_transition_shd", dyn="icalingam_dynamic_score"),
}

BAR_W, GAP = 0.36, 0.02   # 2px-equivalent surface gap between adjacent bars


def style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })


def header(fig, title, subtitle=None):
    """Title block spaced in points, so short figures do not collide the two lines."""
    h = fig.get_figheight() * 72.0
    fig.suptitle(title, x=0.005, y=1 + 17/h, ha="left", va="bottom",
                 fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.005, 1 + 4/h, subtitle, ha="left", va="bottom",
                 fontsize=9.5, color=INK_2)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  {name}")


# ── 1. headline: baseline vs winning ──────────────────────────────────────────

def fig_headline(head, out):
    hi = head[head.higher_is_better]
    lo = head[~head.higher_is_better]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3),
                             gridspec_kw={"width_ratios": [len(hi), len(lo)]})

    for ax, sel, title in ((axes[0], hi, "Precision  (higher is better)"),
                           (axes[1], lo, "Structural error  (lower is better)")):
        x = np.arange(len(sel))
        ax.bar(x - BAR_W/2 - GAP, sel.baseline_mean, BAR_W, color=ORANGE, label="Baseline")
        ax.bar(x + BAR_W/2 + GAP, sel.winning_mean, BAR_W, color=BLUE,
               label="Bootstrap + overlap")
        for i, (_, r) in enumerate(sel.iterrows()):
            change = r.rel_improvement_pct if r.higher_is_better else -r.rel_improvement_pct
            sign = "+" if change >= 0 else "−"
            top = max(r.baseline_mean, r.winning_mean)
            ax.text(i + BAR_W/2 + GAP, r.winning_mean + top*0.035,
                    f"{sign}{abs(change):.1f}%", ha="center", fontsize=9.5,
                    color=GOOD, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n", 1) for m in sel.metric])
        ax.set_ylim(0, max(sel.baseline_mean.max(), sel.winning_mean.max()) * 1.26)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("mean over 18,000 scenarios")
    axes[0].legend(loc="upper left", ncol=1)
    header(fig, "Bootstrap + overlap improves every metric on the full grid",
           "18,000 scenarios, paired by scenario. Green = relative change; "
           "every metric p < 1e-250 (Wilcoxon signed-rank).")
    fig.tight_layout()
    save(fig, out, "01_ablation_headline.png")


# ── 2. win-rates ──────────────────────────────────────────────────────────────

def fig_winrates(head, out):
    fig, ax = plt.subplots(figsize=(11.4, 3.6))
    y = np.arange(len(head))[::-1]
    tot = head.wins + head.losses + head.ties
    ax.barh(y, head.wins/tot, 0.5, color=BLUE, label="Bootstrap + overlap wins")
    ax.barh(y, head.losses/tot, 0.5, left=head.wins/tot + 0.002,
            color=ORANGE, label="Baseline wins")
    ax.barh(y, head.ties/tot, 0.5, left=(head.wins+head.losses)/tot + 0.004,
            color=GRID, label="Tie")
    for yi, (_, r) in zip(y, head.iterrows()):
        ax.text(0.012, yi, f"{r.wins:,} / {r.losses:,}", va="center",
                fontsize=9.5, color=SURFACE, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(head.metric)
    ax.set_xlim(0, 1); ax.set_xlabel("share of the 18,000 scenarios")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    header(fig, "Per-scenario win-rates, not just better averages",
           "Win-rate among decided scenarios ranges 0.67–0.72. Ties are scenarios "
           "where both arms scored identically.")
    fig.tight_layout()
    save(fig, out, "02_ablation_winrates.png")


# ── 3. stratifications ────────────────────────────────────────────────────────

def fig_strata(base, win, out):
    specs = [("p", "Nodes  $p$"), ("n_regimes", "Regimes  $R$"),
             ("samples_regime", "Samples per regime")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), sharey=True)
    colors = [BLUE, ORANGE, TEAL, PURPLE, MAGENTA]
    for ax, (by, label) in zip(axes, specs):
        t = stratified_table(base, win, by)
        for (metric, g), c in zip(t.groupby("metric", sort=False), colors):
            ax.plot(g[by], g.rel_improvement_pct, "-o", color=c, lw=2,
                    ms=5, label=metric)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlabel(label)
        ax.set_xticks(sorted(t[by].unique()))
    axes[0].set_ylabel("relative improvement (%)")
    axes[0].legend(loc="upper right", ncol=1)
    header(fig, "The gain holds across every slice of the grid",
           "Positive = bootstrap + overlap is better. Per-regime precision is flat "
           "at ~9%; the SHD metrics pay off most at low $p$.")
    fig.tight_layout()
    save(fig, out, "03_ablation_strata.png")


# ── 4. paired deltas ──────────────────────────────────────────────────────────

def fig_paired_deltas(base, win, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    picks = [("mean_struct_precision", "Per-regime precision", True),
             ("windowed_nshd_weighted", "Windowed nSHD", False),
             ("transition_shd", "Transition SHD", False)]
    for ax, (col, label, hib) in zip(axes, picks):
        d = (win[col] - base[col]).dropna()
        gain = d if hib else -d
        # Exact ties dominate the zero bin and flatten the shape; the win-rate
        # figure already reports them, so this panel shows decided scenarios only.
        n_tie = int((gain == 0).sum())
        gain = gain[gain != 0]
        ax.hist(gain, bins=70, color=BLUE, alpha=0.85)
        ax.text(0.02, 0.97, f"{len(gain):,} decided\n{n_tie:,} ties excluded",
                transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
        ax.axvline(0, color=INK, lw=1.2)
        ax.axvline(gain.mean(), color=ORANGE, lw=2, ls="--")
        ax.text(gain.mean(), ax.get_ylim()[1]*0.92, f"  mean {gain.mean():+.3f}",
                color=ORANGE, fontsize=9.5, fontweight="bold")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("per-scenario gain")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("scenarios")
    header(fig, "The gain is a shifted distribution, not a handful of outliers",
           "Per-scenario gain over baseline, scenarios where the two arms differ. "
           "Mass right of zero is a win; the black line is parity.")
    fig.tight_layout()
    save(fig, out, "04_ablation_paired_deltas.png")


# ── 5. method comparison ──────────────────────────────────────────────────────

def fig_methods(win, out):
    """Two panels, because mixing higher-is-better and lower-is-better on one axis
    makes a tall bar mean 'good' on the left half and 'bad' on the right."""
    hi = [("f1", "mean F1"), ("prec", "mean precision"), ("rec", "mean recall")]
    lo = [("nshd", "mean nSHD"), ("wnshd", "windowed nSHD"),
          ("tshd", "transition SHD"), ("dyn", "dynamic score")]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2),
                             gridspec_kw={"width_ratios": [len(hi), len(lo)]})
    w = 0.2
    for ax, keys, title in ((axes[0], hi, "Structure recovery  (higher is better)"),
                            (axes[1], lo, "Structural error  (lower is better)")):
        x = np.arange(len(keys))
        for i, (m, cols) in enumerate(METHOD_COLS.items()):
            ax.bar(x + (i - 1.5)*(w + GAP), [win[cols[k]].mean() for k, _ in keys],
                   w, color=METHOD_COLOR[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl.replace(" ", "\n", 1) for _, lbl in keys])
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("mean over 18,000 scenarios")
    header(fig, "The pipeline against the three baselines",
           "ICA-LiNGAM collapses on non-stationary data (F1 0.014) and DirectLiNGAM "
           "is far behind; the real contest is with CausalMorph alone.")
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "05_methods_overview.png")


def fig_methods_by_p(win, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))
    for ax, (k, label, _) in zip(axes, [("nshd", "mean nSHD  (lower is better)", False),
                                        ("f1", "mean F1  (higher is better)", True)]):
        for m, cols in METHOD_COLS.items():
            g = win.groupby("p")[cols[k]].mean()
            ax.plot(g.index, g.values, "-o", color=METHOD_COLOR[m], lw=2, ms=5, label=m)
        ax.set_xlabel("Nodes  $p$"); ax.set_title(label, fontsize=11)
        ax.set_xticks(sorted(win.p.unique()))
    axes[0].legend(loc="upper left", ncol=1)
    header(fig, "The lead over the LiNGAM baselines is widest on small problems",
           "Means per node count. The nSHD gap to DirectLiNGAM compresses from "
           "0.370 at $p$=3 to 0.179 at $p$=12 \u2014 but the gap to CausalMorph alone "
           "goes the other way, 0.007 to 0.016.")
    fig.tight_layout()
    save(fig, out, "06_methods_by_p.png")


def fig_pipeline_vs_cm(win, out):
    """The honest panel: where the pipeline beats standalone CausalMorph and where it does not."""
    keys = [("prec", "mean\nprecision", True), ("rec", "mean\nrecall", True),
            ("f1", "mean F1", True), ("cons_f1", "consensus\nF1", True),
            ("nshd", "mean\nnSHD", False), ("wnshd", "windowed\nnSHD", False),
            ("tshd", "transition\nSHD", False), ("dyn", "dynamic\nscore", False)]
    rels, labels, colors = [], [], []
    for k, label, hib in keys:
        a = win[METHOD_COLS["Pipeline"][k]].astype(float)
        b = win[METHOD_COLS["CausalMorph"][k]].astype(float)
        m = a.notna() & b.notna()
        gain = (a[m] - b[m]) if hib else (b[m] - a[m])
        rels.append(100 * gain.mean() / abs(b[m].mean()))
        labels.append(label)
        colors.append(BLUE if rels[-1] >= 0 else ORANGE)
    fig, ax = plt.subplots(figsize=(11.4, 3.9))
    x = np.arange(len(labels))
    ax.bar(x, rels, 0.56, color=colors)
    ax.axhline(0, color=INK, lw=1.2)
    for xi, v in zip(x, rels):
        ax.text(xi, v + (0.25 if v >= 0 else -0.45), f"{v:+.1f}%", ha="center",
                fontsize=9.5, color=INK_2, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("pipeline vs CausalMorph alone (%)")
    ax.set_ylim(min(rels) - 1.6, max(rels) + 1.2)
    ax.grid(axis="x", visible=False)
    ax.annotate("consensus F1: p = 0.37,\nno significant difference",
                xy=(3, rels[3] - 0.15), xytext=(4.1, min(rels) - 0.9),
                fontsize=9.5, color=INK_2, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1,
                                connectionstyle="arc3,rad=0.2"))
    header(fig, "Against CausalMorph alone the pipeline trades recall for precision",
           "Positive = pipeline better. Precision and the dynamic metrics gain; "
           "recall loses; consensus F1 is a tie.")
    fig.tight_layout()
    save(fig, out, "07_pipeline_vs_causalmorph.png")


def fig_distributions(win, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))
    for ax, (k, label) in zip(axes, [("nshd", "mean nSHD  (lower is better)"),
                                     ("f1", "mean F1  (higher is better)")]):
        data = [win[METHOD_COLS[m][k]].dropna() for m in METHOD_COLS]
        parts = ax.violinplot(data, showmedians=True, widths=0.78)
        for body, m in zip(parts["bodies"], METHOD_COLOR):
            body.set_facecolor(METHOD_COLOR[m]); body.set_alpha(0.75)
            body.set_edgecolor(SURFACE); body.set_linewidth(2)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color(INK_2); parts[key].set_linewidth(1.2)
        ax.set_xticks(range(1, len(METHOD_COLOR) + 1))
        ax.set_xticklabels(list(METHOD_COLOR), fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="x", visible=False)
    header(fig, "Full distributions, not just means",
           "Each violin is 18,000 scenarios. The pipeline's mass sits lower on nSHD "
           "and higher on F1 than every baseline.")
    fig.tight_layout()
    save(fig, out, "08_methods_distributions.png")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=os.path.join(here, "results/v1h_full_baseline/results.csv"))
    ap.add_argument("--winning",  default=os.path.join(here, "results/v1h_full_winning/results.csv"))
    ap.add_argument("--out",      default=os.path.join(here, "results/figures"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    style()
    print("\nLoading:")
    base, win = load_pair(args.baseline, args.winning)
    head = headline_table(base, win)

    print(f"\nWriting figures to {args.out}/")
    fig_headline(head, args.out)
    fig_winrates(head, args.out)
    fig_strata(base, win, args.out)
    fig_paired_deltas(base, win, args.out)
    fig_methods(win, args.out)
    fig_methods_by_p(win, args.out)
    fig_pipeline_vs_cm(win, args.out)
    fig_distributions(win, args.out)
    print()


if __name__ == "__main__":
    main()
