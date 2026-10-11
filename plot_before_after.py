#!/usr/bin/env python3
"""
Before / after figures for the pipeline hardening, on the 600-scenario
validation sample. "Before" is the October-5 full-grid run (oracle anchor, old
consensus, old pruning, v1H) restricted to the same scenarios; "after" is the
chosen detector with cold start, merge-for-extraction and the 0.5 consensus.

Usage:
    python plot_before_after.py --det_row D_v1L_g2.5 --l3_row L3_cold_v1L25 \
        --out results/hardening_validation/figures
    # optional, for the decomposition panel:
        --c1_row C1_oracle_anchor_v1L --c2_row C2_oracle_seg_v1L --l2_row L2_cold_v1L
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_hardening_validation import load, parse, match, paired, HEAD

BLUE, ORANGE, TEAL, PURPLE, MAGENTA = "#2a78d6", "#eb6834", "#0f8a7a", "#8b5bd6", "#d1458f"
SURFACE, INK, INK_2, MUTED, GRID, AXIS, GOOD = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#006300"
BAR_W, GAP = 0.36, 0.02


def style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })


def header(fig, title, subtitle=None):
    h = fig.get_figheight() * 72.0
    fig.suptitle(title, x=0.005, y=1 + 17 / h, ha="left", va="bottom", fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.005, 1 + 4 / h, subtitle, ha="left", va="bottom", fontsize=9.5, color=INK_2)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name)); plt.close(fig); print(f"  {name}")


# ── detection helpers ─────────────────────────────────────────────────────────

def micro(d, tol=200):
    TP = FP = FN = 0
    for tc, dc in zip(d.true_cps.map(parse), d.detected_cps.map(parse)):
        a, b, c = match(tc, dc, tol); TP += a; FP += b; FN += c
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def recall_by_position(d, tol=200):
    hits = {}
    for tc, dc in zip(d.true_cps.map(parse), d.detected_cps.map(parse)):
        for k, t in enumerate(sorted(tc)):
            hits.setdefault(k, []).append(any(abs(x - t) <= tol for x in dc))
    return {k: (np.mean(v), len(v)) for k, v in hits.items() if len(v) >= 10}


# ── figures ───────────────────────────────────────────────────────────────────

def fig_detection_summary(before, after, label_after, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1.3]})

    # (a) P / R / F1
    ax = axes[0]
    pb, pa = micro(before), micro(after)
    x = np.arange(3)
    ax.bar(x - BAR_W / 2 - GAP, pb, BAR_W, color=ORANGE, label="Before")
    ax.bar(x + BAR_W / 2 + GAP, pa, BAR_W, color=BLUE, label=label_after)
    for i in range(3):
        d = pa[i] - pb[i]
        ax.text(i + BAR_W / 2 + GAP, pa[i] + 0.02, f"{d:+.2f}", ha="center", fontsize=9,
                color=GOOD if d >= 0 else ORANGE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["precision", "recall", "F1"])
    ax.set_ylim(0, 1.38); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Detection, tolerance 200", fontsize=11)
    ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.0))

    # (b) recall by change-point position
    ax = axes[1]
    for d, c, lab in ((before, ORANGE, "Before"), (after, BLUE, label_after)):
        r = recall_by_position(d)
        ks = sorted(r)
        ax.plot(ks, [r[k][0] for k in ks], "-o", color=c, lw=2, ms=5, label=lab)
    ax.set_ylim(0, 1.05); ax.set_xlabel("position of the true change point in the series")
    ax.set_title("Recall by change-point position", fontsize=11)
    ax.legend(loc="lower left")

    # (c) recall by p
    ax = axes[2]
    for d, c, lab in ((before, ORANGE, "Before"), (after, BLUE, label_after)):
        ps = sorted(d.p.unique())
        ax.plot(ps, [micro(d[d.p == p_])[1] for p_ in ps], "-o", color=c, lw=2, ms=5, label=lab)
    ax.set_ylim(0, 1.05); ax.set_xlabel("nodes  $p$"); ax.set_xticks(sorted(after.p.unique()))
    ax.set_title("Recall by problem size", fontsize=11)
    ax.legend(loc="lower left")

    header(fig, "Change-point detection: before and after",
           "600 stratified scenarios (60 per p), same scenarios in both. "
           "The positional decay and the collapse at large p are gone.")
    fig.tight_layout()
    save(fig, out, "A_detection_before_after.png")


def fig_detection_tolerance(before, after, label_after, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    tols = [25, 50, 100, 200, 300]
    for d, c, lab in ((before, ORANGE, "Before"), (after, BLUE, label_after)):
        ax.plot(tols, [micro(d, t)[1] for t in tols], "-o", color=c, lw=2, ms=5, label=lab)
    ax.set_xscale("log"); ax.set_xticks(tols); ax.set_xticklabels([str(t) for t in tols])
    ax.set_xlabel("matching tolerance (samples)"); ax.set_ylabel("recall"); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    header(fig, "Localisation is sharp in both versions",
           "Recall barely moves with tolerance: when a change point is found it is within a few samples. "
           "The gain is in finding them at all.")
    fig.tight_layout()
    save(fig, out, "B_detection_tolerance.png")


def fig_segmentation_outcome(before, after, label_after, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    cats = ["under-segmented", "exact", "over-segmented", "no detection"]
    def shares(d):
        return [(d.n_structures < d.n_regimes).mean(), (d.n_structures == d.n_regimes).mean(),
                (d.n_structures > d.n_regimes).mean(), (d.n_detected_cps == 0).mean()]
    sb, sa = shares(before), shares(after)
    x = np.arange(len(cats))
    ax.bar(x - BAR_W / 2 - GAP, sb, BAR_W, color=ORANGE, label="Before")
    ax.bar(x + BAR_W / 2 + GAP, sa, BAR_W, color=BLUE, label=label_after)
    for i in range(len(cats)):
        ax.text(i - BAR_W / 2 - GAP, sb[i] + 0.012, f"{sb[i]:.0%}", ha="center", fontsize=9, color=INK_2)
        ax.text(i + BAR_W / 2 + GAP, sa[i] + 0.012, f"{sa[i]:.0%}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylim(0, max(sb + sa) * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("share of scenarios"); ax.legend(loc="upper right")
    header(fig, "How often the segmentation matches the true number of regimes",
           "Under-segmentation is the error extraction cannot repair: a window spanning two regimes is "
           "scored against one of them.")
    fig.tight_layout()
    save(fig, out, "C_segmentation_outcome.png")


def fig_structure(before, after, label_after, out):
    hi = [(c, m) for c, m, hib in HEAD if hib and c in after and c in before]
    lo = [(c, m) for c, m, hib in HEAD if not hib and c in after and c in before]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [len(hi), len(lo)]})
    for ax, sel, title in ((axes[0], hi, "Recovery  (higher is better)"), (axes[1], lo, "Error  (lower is better)")):
        x = np.arange(len(sel))
        vb = [before[c].mean() for c, _ in sel]; va = [after[c].mean() for c, _ in sel]
        ax.bar(x - BAR_W / 2 - GAP, vb, BAR_W, color=ORANGE, label="Before")
        ax.bar(x + BAR_W / 2 + GAP, va, BAR_W, color=BLUE, label=label_after)
        top = max(vb + va)
        for i, (c, _) in enumerate(sel):
            r = paired(before, after, c, c in [h[0] for h in hi])
            if r:
                sign = "+" if r["rel"] >= 0 else "−"
                ax.text(i + BAR_W / 2 + GAP, va[i] + top * 0.03, f"{sign}{abs(r['rel']):.0f}%",
                        ha="center", fontsize=9, color=GOOD if r["rel"] >= 0 else ORANGE, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([m.replace(" ", "\n", 1) for _, m in sel])
        ax.set_ylim(0, top * 1.25); ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("mean over the sample"); axes[0].legend(loc="upper right")
    header(fig, "Structure recovery: before and after",
           "Green/orange = relative change, positive = better. Both versions scored under the corrected "
           "adjacency convention; 'before' is the old configuration (oracle seed, old consensus, old pruning).")
    fig.tight_layout()
    save(fig, out, "D_structure_before_after.png")


def fig_consensus_windows(before, after, label_after, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6), sharey=True)
    for ax, d, lab, c in ((axes[0], before, "Before", ORANGE), (axes[1], after, label_after, BLUE)):
        nw = (d["n_windows_ext"] if "n_windows_ext" in d else d["n_structures"]).clip(upper=4)
        g = d.groupby(nw).agg(recall=("consensus_recall", "mean"), precision=("consensus_precision", "mean"))
        x = np.arange(len(g))
        ax.bar(x - BAR_W / 2 - GAP, g.recall, BAR_W, color=c, label="consensus recall")
        ax.bar(x + BAR_W / 2 + GAP, g.precision, BAR_W, color=c, alpha=0.45, label="consensus precision")
        for i, v in enumerate(g.recall):
            ax.text(i - BAR_W / 2 - GAP, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK_2)
        ax.set_xticks(x); ax.set_xticklabels([f"{int(k)}" if k < 4 else "4+" for k in g.index])
        ax.set_xlabel("extraction windows in the scenario"); ax.set_title(lab, fontsize=11)
        ax.set_ylim(0, 1.15); ax.legend(loc="upper right")
    header(fig, "The consensus graph with one or two windows",
           "Before: recall exactly 1.000 at 1 and 2 windows — the aggregator admitted every edge (posterior of an "
           "unvoted edge, 1/(n+2), exceeded the 0.20 threshold). After: majority rule.")
    fig.tight_layout()
    save(fig, out, "E_consensus_by_windows.png")


def remap_learner(d, method):
    """Expose a per-window learner's ablation columns under the pipeline's names."""
    if d is None or method == "pipeline":
        return d
    d = d.copy()
    m = {f"{method}_mean_precision": "mean_struct_precision", f"{method}_mean_recall": "mean_struct_recall",
         f"{method}_mean_f1": "mean_struct_f1", f"{method}_mean_norm_shd": "mean_norm_shd",
         f"{method}_cons_precision": "consensus_precision", f"{method}_cons_recall": "consensus_recall",
         f"{method}_cons_f1": "consensus_f1", f"{method}_cons_norm_shd": "consensus_norm_shd",
         f"{method}_windowed_nshd_weighted": "windowed_nshd_weighted", f"{method}_transition_shd": "transition_shd",
         f"{method}_transition_f1": "transition_f1", f"{method}_dynamic_score": "dynamic_score"}
    for src, dst in m.items():
        if src in d:
            d[dst] = d[src]
    return d


def fig_decomposition(after, after_pipe, c2, before, boot, label_after, out):
    """Mean paired differences in metric units, each row against 'after'.
    Positive = that row is better than 'after'. Absolute differences, not
    relative: the chosen configuration's error is small, so ratios explode."""
    # (tick label, legend description, row, reference, colour)
    # C2 carries only the pipeline's own learner, so its headroom is measured against the pipeline, not the per-window learner
    comps = [("oracle\nsegmentation", "oracle segmentation: the pipeline on the true change points, vs the pipeline on detected ones (headroom for the pipeline)", c2, after_pipe if after_pipe is not None else after, TEAL),
             ("old\nconfig", "old configuration: oracle seed, consensus 0.20, 10p\u00b2 deletion, v1H", before, after, ORANGE),
             ("bootstrap\n+ overlap", "bootstrap 20 @ 0.60 + overlap 0.15, pipeline on the same windows", boot, after_pipe, MAGENTA),
             ("CausalMorph\n+ chain", "CausalMorph transform + chain prior, vs the per-window learner on the same windows", after_pipe, after, PURPLE)]
    comps = [c for c in comps if c[2] is not None and c[3] is not None and c[2] is not c[3]]
    if not comps:
        return
    mets = [("mean_struct_precision", "per-regime precision", True), ("mean_struct_recall", "per-regime recall", True),
            ("windowed_nshd_weighted", "windowed nSHD", False)]
    fig, axes = plt.subplots(1, len(mets), figsize=(11.4, 4.0))
    for ax, (col, mlab, hib) in zip(axes, mets):
        vals, cols = [], []
        for lab, desc, d, ref, c in comps:
            r = paired(ref, d, col, hib)
            delta = (r["other"] - r["ref"]) if r else 0.0
            vals.append(delta if hib else -delta)
            cols.append(c)
        x = np.arange(len(comps))
        ax.bar(x, vals, 0.55, color=cols)
        ax.axhline(0, color=INK, lw=1.1)
        span = max(abs(v) for v in vals) or 0.1
        for xi, v in zip(x, vals):
            ax.text(xi, v + (0.04 if v >= 0 else -0.09) * span, f"{v:+.3f}", ha="center", fontsize=9,
                    color=INK_2, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([lab for lab, *_ in comps], fontsize=8.5)
        ax.set_title(mlab, fontsize=11); ax.grid(axis="x", visible=False)
        lo, hi = min(vals + [0]), max(vals + [0]); ax.set_ylim(lo - 0.18 * span, hi + 0.18 * span)
    axes[0].set_ylabel("paired difference vs 'after'  (metric units; + = better)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=desc) for _, desc, _, _, c in comps],
               loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    header(fig, "What each piece is worth, on the same scenarios",
           "Each bar is one change against the chosen configuration, same 600 scenarios, paired. "
           "A bar below zero is a piece that was hurting.")
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    save(fig, out, "F_decomposition.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", default="results/hardening_validation")
    ap.add_argument("--before", default="results/results_october_5")
    ap.add_argument("--before_row", default=None,
                    help="use this validation row as 'before' (e.g. C0_legacy_v2: the old configuration re-scored "
                         "under the corrected adjacency convention) instead of the October-5 CSV, which scored "
                         "transposed adjacencies and is not comparable")
    ap.add_argument("--det_row", required=True, help="detection-only row for the after detector, e.g. D_v1L_g2.5")
    ap.add_argument("--l3_row", required=True, help="L3 row for the after pipeline, e.g. L3_cold_v1L25")
    ap.add_argument("--label", default="After")
    ap.add_argument("--after_method", default="pipeline", choices=["pipeline", "directlingam", "icalingam"],
                    help="which learner's per-window columns to show as 'after' structure (the row must have ablation on)")
    ap.add_argument("--boot_row", default=None,
                    help="row with bootstrap 0.60 + overlap on the same windows, for the decomposition panel")
    ap.add_argument("--c1_row", default=None); ap.add_argument("--c2_row", default=None); ap.add_argument("--l2_row", default=None)
    ap.add_argument("--out", default="results/hardening_validation/figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True); style()
    ids = {ln.strip() for ln in open(os.path.join(args.val, "sample_ids.txt")) if ln.strip()}
    if args.before_row:
        before = load(os.path.join(args.val, args.before_row, "results.csv"))
    else:
        before = load(os.path.join(args.before, "v1h_full_winning/results.csv"), ids)
    det = load(os.path.join(args.val, args.det_row, "results.csv"))
    l3_raw = load(os.path.join(args.val, args.l3_row, "results.csv"))
    l3 = remap_learner(l3_raw, args.after_method)
    common = set(before.index) & set(det.index) & set(l3.index)
    before, det, l3 = before[before.index.isin(common)], det[det.index.isin(common)], l3[l3.index.isin(common)]
    # with the pipeline itself as 'after', the bootstrap bar compares against it directly
    l3_pipe = l3_raw[l3_raw.index.isin(common)] if args.after_method != "pipeline" else l3
    print(f"scenarios in every row: {len(common)}\nwriting to {args.out}/")

    fig_detection_summary(before, det, args.label, args.out)
    fig_detection_tolerance(before, det, args.label, args.out)
    fig_segmentation_outcome(before, l3, args.label, args.out)
    fig_structure(before, l3, args.label, args.out)
    fig_consensus_windows(before, l3, args.label, args.out)
    c2 = remap_learner(load(os.path.join(args.val, args.c2_row, "results.csv")), args.after_method) if args.c2_row else None
    boot = load(os.path.join(args.val, args.boot_row, "results.csv")) if args.boot_row else None
    c2 = c2[c2.index.isin(common)] if c2 is not None else None
    boot = boot[boot.index.isin(common)] if boot is not None else None
    fig_decomposition(l3, l3_pipe, c2, before, boot, args.label, args.out)


if __name__ == "__main__":
    main()
