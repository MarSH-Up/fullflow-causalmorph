#!/usr/bin/env python3
"""
Paired baseline-vs-winning comparison for the extraction ablation.

Pairs two run_experiments.py outputs by scenario_id and reports, for each
headline metric: mean +/- SD per arm, paired delta, relative change, per-scenario
win-rate and a Wilcoxon signed-rank test. Also stratifies by p, samples per
regime and number of regimes, and checks that every metric moved in the same
direction as the 180-scenario subset result.

This is the script behind Table `tab:winning_config` and slide-20 figure F.

Outputs (in --out):
  headline.csv        — one row per metric: means, delta, win-rate, Wilcoxon
  by_p.csv            — paired delta per node count
  by_samples.csv      — paired delta per samples-per-regime
  by_regimes.csv      — paired delta per number of regimes
  plot_F_values.txt   — PRECISION/ERROR literals to paste into
                        Presentation/plot_scripts/plot_F_bootstrap_ablation.py

Usage:
    python compare_full_grid.py --baseline results/v1h_full_baseline/results.csv \
                                --winning  results/v1h_full_winning/results.csv \
                                --out      results/full_grid_comparison
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# ── metrics ───────────────────────────────────────────────────────────────────

# (column, label, higher_is_better, subset_baseline, subset_winning)
# The last two are the 180-scenario values from results/v1h_phase1 and
# results/v1h_winning, used only for the direction sanity check.
METRICS = [
    ("mean_struct_precision",  "Per-regime precision", True,  0.328, 0.363),
    ("consensus_precision",    "Consensus precision",  True,  0.147, 0.175),
    ("windowed_nshd_weighted", "Windowed nSHD",        False, 0.161, 0.151),
    ("transition_shd",         "Transition SHD",       False, 0.116, 0.108),
    ("dynamic_score",          "Dynamic score",        False, 0.217, 0.210),
]

STRATA = [("p", "by_p"), ("samples_regime", "by_samples"), ("n_regimes", "by_regimes")]


# ── loading ───────────────────────────────────────────────────────────────────

def load_pair(baseline_path, winning_path):
    """Load both arms, keep only scenarios that succeeded in both, and align them."""
    base = pd.read_csv(baseline_path)
    win  = pd.read_csv(winning_path)

    for name, df, path in (("baseline", base, baseline_path), ("winning", win, winning_path)):
        missing = [c for c, *_ in METRICS if c not in df.columns]
        if missing:
            raise SystemExit(
                f"{path} is missing {missing}. That file predates the dynamic metrics — "
                f"re-run it with the current run_experiments.py."
            )
        n_bad = (df["status"] != "ok").sum()
        print(f"  {name:8s}: {len(df)} rows, {n_bad} not ok  ({path})")

    base = base[base["status"] == "ok"]
    win  = win[win["status"] == "ok"]

    shared = sorted(set(base["scenario_id"]) & set(win["scenario_id"]))
    if not shared:
        raise SystemExit("No scenario_id is present and ok in both files.")

    base = base.set_index("scenario_id").loc[shared]
    win  = win.set_index("scenario_id").loc[shared]
    print(f"  paired  : {len(shared)} scenarios ok in both arms\n")
    return base, win


# ── statistics ────────────────────────────────────────────────────────────────

def paired_stats(b, w, higher_is_better):
    """Paired summary of one metric. `b`/`w` are aligned Series."""
    mask = b.notna() & w.notna()
    b, w = b[mask].astype(float), w[mask].astype(float)
    diff = w - b                                   # winning minus baseline
    gain = diff if higher_is_better else -diff     # positive gain = winning is better

    wins   = int((gain > 0).sum())
    losses = int((gain < 0).sum())
    ties   = int((gain == 0).sum())

    # Wilcoxon needs at least one non-zero difference.
    if wins + losses:
        stat, pval = stats.wilcoxon(w, b)
    else:
        stat, pval = float("nan"), float("nan")

    rel = 100.0 * gain.mean() / abs(b.mean()) if b.mean() else float("nan")
    return dict(
        n=len(b),
        baseline_mean=b.mean(), baseline_sd=b.std(),
        winning_mean=w.mean(),  winning_sd=w.std(),
        delta=diff.mean(), rel_improvement_pct=rel,
        wins=wins, losses=losses, ties=ties,
        win_rate=wins / (wins + losses) if wins + losses else float("nan"),
        wilcoxon_stat=stat, wilcoxon_p=pval,
    )


def headline_table(base, win):
    rows = []
    for col, label, hib, sub_b, sub_w in METRICS:
        r = paired_stats(base[col], win[col], hib)
        sub_gain = (sub_w - sub_b) if hib else (sub_b - sub_w)
        r.update(metric=label, column=col, higher_is_better=hib,
                 subset_baseline=sub_b, subset_winning=sub_w,
                 direction_matches_subset=bool(np.sign(r["rel_improvement_pct"]) == np.sign(sub_gain)))
        rows.append(r)
    cols = ["metric", "column", "n", "baseline_mean", "baseline_sd", "winning_mean",
            "winning_sd", "delta", "rel_improvement_pct", "wins", "losses", "ties",
            "win_rate", "wilcoxon_stat", "wilcoxon_p", "higher_is_better",
            "subset_baseline", "subset_winning", "direction_matches_subset"]
    return pd.DataFrame(rows)[cols]


def stratified_table(base, win, by):
    rows = []
    for level in sorted(base[by].unique()):
        sel = base[by] == level
        for col, label, hib, *_ in METRICS:
            r = paired_stats(base.loc[sel, col], win.loc[sel, col], hib)
            r.update(**{by: level}, metric=label)
            rows.append(r)
    cols = [by, "metric", "n", "baseline_mean", "winning_mean", "delta",
            "rel_improvement_pct", "wins", "losses", "win_rate", "wilcoxon_p"]
    return pd.DataFrame(rows)[cols]


# ── reporting ─────────────────────────────────────────────────────────────────

def print_headline(t):
    print("=" * 96)
    print("Headline: baseline vs bootstrap+overlap (paired by scenario_id)")
    print("=" * 96)
    print(f"{'Metric':<22}{'Baseline':>17}{'Winning':>17}{'Rel':>9}{'Win-rate':>13}{'Wilcoxon p':>14}")
    for _, r in t.iterrows():
        arrow = "^" if r.higher_is_better else "v"
        print(f"{r.metric + ' ' + arrow:<22}"
              f"{r.baseline_mean:>10.4f} ±{r.baseline_sd:<5.3f}"
              f"{r.winning_mean:>10.4f} ±{r.winning_sd:<5.3f}"
              f"{r.rel_improvement_pct:>+8.1f}%"
              f"{str(r.wins) + '/' + str(r.losses):>13}"
              f"{r.wilcoxon_p:>14.2e}")
    print("=" * 96)

    bad = t[~t.direction_matches_subset]
    if len(bad):
        print("\n  !! DIRECTION REVERSAL vs the 180-scenario subset — investigate before publishing:")
        for _, r in bad.iterrows():
            print(f"     {r.metric}: subset {r.subset_baseline:.3f} -> {r.subset_winning:.3f}, "
                  f"full grid {r.baseline_mean:.3f} -> {r.winning_mean:.3f}")
    else:
        print("\n  All five metrics moved in the same direction as the 180-scenario subset.")


def write_plot_f_values(t, path):
    """Emit the literals for plot_F_bootstrap_ablation.py (values are hard-coded there)."""
    def block(name, sel):
        out = [f"{name} = ["]
        for _, r in sel.iterrows():
            label = r.metric.replace(" ", "\\n", 1)
            # The slide labels the signed change in the metric itself, not the
            # improvement: a lower-is-better metric that improved reads "-6.7%".
            change = r.rel_improvement_pct if r.higher_is_better else -r.rel_improvement_pct
            sign = "+" if change >= 0 else "−"
            out.append(f'    ("{label}", {r.baseline_mean:.3f}, {r.winning_mean:.3f}, '
                       f'"{sign}{abs(change):.1f}%"),')
        out.append("]")
        return "\n".join(out)

    text = (
        "# Generated by compare_full_grid.py — paste into\n"
        "# Presentation/plot_scripts/plot_F_bootstrap_ablation.py (lines 18-25)\n\n"
        + block("PRECISION", t[t.higher_is_better]) + "\n"
        + block("ERROR",     t[~t.higher_is_better]) + "\n"
    )
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  plot F literals -> {path}")
    print(text)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=os.path.join(here, "results/v1h_full_baseline/results.csv"),
                    help="results.csv of the no-bootstrap, no-overlap arm")
    ap.add_argument("--winning",  default=os.path.join(here, "results/v1h_full_winning/results.csv"),
                    help="results.csv of the bootstrap+overlap arm")
    ap.add_argument("--out",      default=os.path.join(here, "results/full_grid_comparison"),
                    help="directory for the CSV outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("\nLoading:")
    base, win = load_pair(args.baseline, args.winning)

    head = headline_table(base, win)
    print_headline(head)
    head.to_csv(os.path.join(args.out, "headline.csv"), index=False)

    for by, name in STRATA:
        tab = stratified_table(base, win, by)
        tab.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        print(f"\n--- {name} (relative improvement %, positive = winning better) ---")
        print(tab.pivot(index=by, columns="metric", values="rel_improvement_pct")
                 .round(1).to_string())

    write_plot_f_values(head, os.path.join(args.out, "plot_F_values.txt"))
    print(f"\nWrote CSVs to {args.out}/")


if __name__ == "__main__":
    main()
