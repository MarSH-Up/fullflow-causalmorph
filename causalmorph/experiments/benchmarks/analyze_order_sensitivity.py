"""
Statistical Analysis of CausalMorph Sensitivity to Errors in Tentative Causal Order

Reads the output CSV from run_order_sensitivity.py and produces:
  1. Aggregate tables  (mean, median, CI) per error-rate level
  2. Paired statistical tests (Wilcoxon signed-rank) at each level
  3. Effect-size measures  (Cohen's d)
  4. Publication-quality figures

Usage:
    python experiments/benchmarks/analyze_order_sensitivity.py <results_csv> [output_dir]
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Visual style (aligned with project palette)
# ---------------------------------------------------------------------------
COLOR_ORIG = "#2563EB"       # Blue  – baseline / original
COLOR_MORPH = "#F97316"      # Orange – CausalMorph / transformed
COLOR_IMPROVE = "#10B981"    # Green  – improvement
COLOR_DEGRADE = "#EF4444"    # Red    – degradation
COLOR_NEUTRAL = "#6B7280"    # Gray

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cohens_d(x, y):
    """Paired Cohen's d (mean difference / pooled SD)."""
    diff = np.asarray(x) - np.asarray(y)
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
    return d


def boot_ci(arr, stat_fn=np.mean, n_boot=5000, ci=0.95, seed=42):
    """Bootstrap confidence interval for *stat_fn*."""
    rng = np.random.RandomState(seed)
    arr = np.asarray(arr)
    boot = np.array([stat_fn(rng.choice(arr, size=len(arr), replace=True))
                     for _ in range(n_boot)])
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return lo, hi


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paired_permutation_test(x, y, n_perm=10_000, seed=0):
    """
    Paired permutation test for H0: median(x - y) == 0.

    Under the null, the sign of each difference d_i = x_i - y_i is
    equally likely to be + or -.  We sample n_perm sign-flip vectors
    and record the distribution of the permuted mean difference.

    Parameters
    ----------
    x, y   : array-like, paired observations (same length)
    n_perm : int, number of permutations
    seed   : int, RNG seed for reproducibility

    Returns
    -------
    observed_stat : float  – mean(x - y) on the real data
    p_value       : float  – two-sided p-value
    """
    rng = np.random.RandomState(seed)
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    observed = np.mean(d)

    # Each permutation randomly flips the sign of each difference
    signs = rng.choice([-1, 1], size=(n_perm, len(d)))
    null_dist = (signs * d[np.newaxis, :]).mean(axis=1)

    # Two-sided: fraction of |null| >= |observed|
    p_value = (np.abs(null_dist) >= np.abs(observed)).mean()
    return observed, p_value


def sensitivity_analysis(summary_df, sweep_col, out_dir, label="Order Error Rate"):
    """
    Test whether performance degrades monotonically across error-rate levels.

    For every consecutive pair of levels (η_low, η_high) a paired permutation
    test is run on the *improvement* values.  This establishes that the
    degradation trend is statistically significant, not just noise.

    Returns a DataFrame of pairwise comparison results.
    """
    # Reload raw data is not available here, so accept pre-aggregated summary.
    # This function is a hook for callers that pass raw per-experiment data.
    pass  # extended version implemented inline in analyze_sweep


# ---------------------------------------------------------------------------
# Core analysis per sweep
# ---------------------------------------------------------------------------
def analyze_sweep(df, sweep_col, fixed_col, label, ax_shd, df_real=None):
    """
    Analyse one sweep dimension (order error).

    Parameters
    ----------
    df : DataFrame  – full results (controlled experiments only)
    sweep_col : str – column being varied  (e.g. "adj_error_rate")
    fixed_col : str – column held at 0     (e.g. "order_error_rate")
    label : str     – human-readable name  (e.g. "Order Error Rate")
    ax_shd : matplotlib Axes for the SHD plot
    df_real : DataFrame or None – real pipeline results (plotted as horizontal line)

    Returns
    -------
    summary : DataFrame with per-level statistics
    """
    sub = df[df[fixed_col] == 0.0].copy()
    rates = sorted(sub[sweep_col].unique())

    rows = []
    for rate in rates:
        g = sub[sub[sweep_col] == rate]
        n = len(g)

        # SHD
        shd_orig = g["shd_original"].values
        shd_trans = g["shd_transformed"].values
        improvement = g["improvement"].values  # orig - trans (positive = better)

        mean_imp = np.mean(improvement)
        median_imp = np.median(improvement)
        ci_lo, ci_hi = boot_ci(improvement)
        pct_improved = (improvement > 0).mean() * 100
        pct_equal = (improvement == 0).mean() * 100

        # Wilcoxon signed-rank (paired, two-sided)
        non_zero = improvement[improvement != 0]
        if len(non_zero) > 10:
            w_stat, w_p = stats.wilcoxon(non_zero, alternative="two-sided")
        else:
            w_stat, w_p = np.nan, np.nan

        # Paired permutation test (sign-flip, 10 000 resamples)
        _, perm_p = paired_permutation_test(shd_orig, shd_trans, n_perm=10_000,
                                            seed=int(rate * 1000))

        d_shd = cohens_d(shd_orig, shd_trans)

        rows.append({
            sweep_col: rate,
            "n": n,
            "shd_orig_mean": np.mean(shd_orig),
            "shd_trans_mean": np.mean(shd_trans),
            "shd_imp_mean": mean_imp,
            "shd_imp_median": median_imp,
            "shd_imp_ci_lo": ci_lo,
            "shd_imp_ci_hi": ci_hi,
            "pct_improved": pct_improved,
            "pct_equal": pct_equal,
            "pct_degraded": 100 - pct_improved - pct_equal,
            "wilcoxon_p_shd": w_p,
            "permutation_p_shd": perm_p,
            "cohens_d_shd": d_shd,
        })

    summary = pd.DataFrame(rows)

    # ---- Plot: SHD (3 independent lines) ----
    # Line 1 (Blue): Original DirectLiNGAM on raw data
    ax_shd.plot(
        summary[sweep_col], summary["shd_orig_mean"],
        "o-", color=COLOR_ORIG, label="DirectLiNGAM (baseline)", markersize=6,
    )
    # Line 2 (Orange): Controlled — ground-truth order/adj perturbed at known rate
    ax_shd.plot(
        summary[sweep_col], summary["shd_trans_mean"],
        "s-", color=COLOR_MORPH, label="CausalMorph + DirectLiNGAM (controlled)", markersize=6,
    )
    # Line 3 (Green): Real pipeline — LiNGAM estimates fed to CausalMorph,
    #   x-position = measured error rate of LiNGAM's output vs ground truth
    if df_real is not None and len(df_real) > 0:
        # Use measured error rate column matching the sweep dimension
        real_rate_col = {"adj_error_rate": "measured_adj_error_rate",
                         "order_error_rate": "measured_order_error_rate"}.get(
            sweep_col, sweep_col)
        # Fall back to sweep_col if measured column doesn't exist (new CSV format)
        if real_rate_col not in df_real.columns:
            real_rate_col = sweep_col

        # Bin measured rates to nearest controlled error rate for alignment
        bin_edges = (
            [-np.inf]
            + [(a + b) / 2 for a, b in zip(rates[:-1], rates[1:])]
            + [np.inf]
        )
        df_rp = df_real.copy()
        df_rp["rate_bin"] = pd.cut(
            df_rp[real_rate_col], bins=bin_edges, labels=rates,
            include_lowest=True,
        )
        rp_agg = (df_rp.groupby("rate_bin", observed=True)["shd_transformed"]
                   .mean().dropna())
        if len(rp_agg) > 0:
            ax_shd.plot(
                rp_agg.index.astype(float), rp_agg.values,
                "^--", color=COLOR_IMPROVE, markersize=7, lw=2,
                label="CausalMorph + LiNGAM (real pipeline)",
            )
    # Stars
    for _, row in summary.iterrows():
        stars = significance_stars(row["wilcoxon_p_shd"])
        if stars != "ns":
            y_pos = max(row["shd_orig_mean"], row["shd_trans_mean"]) + 0.005
            ax_shd.text(row[sweep_col], y_pos, stars, ha="center", fontsize=9,
                        color=COLOR_IMPROVE)

    ax_shd.set_xlabel(label)
    ax_shd.set_ylabel("Normalized SHD (lower is better)")
    ax_shd.set_title(f"SHD vs {label}")
    ax_shd.legend(fontsize=9)
    ax_shd.grid(True, alpha=0.2)

    return summary


# ---------------------------------------------------------------------------
# Breakdown by graph size / density
# ---------------------------------------------------------------------------
def plot_improvement_by_factor(df, factor_col, factor_label, metric, out_dir):
    """Box-plots of improvement grouped by a categorical factor."""
    sub = df[df["adj_error_rate"] == 0.0].copy()
    if factor_col == "density_bin":
        sub["density_bin"] = pd.cut(sub["density"], bins=4)
    elif factor_col == "p_bin":
        sub["p_bin"] = pd.cut(sub["p"], bins=[0, 10, 25, 50, 200])

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = sorted(sub[factor_col].dropna().unique(), key=lambda x: x.left if hasattr(x, "left") else x)
    data_for_box = [sub.loc[sub[factor_col] == g, metric].dropna().values for g in groups]
    bp = ax.boxplot(data_for_box, tick_labels=[str(g) for g in groups], patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_IMPROVE)
        patch.set_alpha(0.5)
    ax.axhline(0, ls="--", color=COLOR_NEUTRAL, lw=1)
    ax.set_xlabel(factor_label)
    ax.set_ylabel(f"{metric} (positive = CausalMorph better)")
    ax.set_title(f"CausalMorph Improvement by {factor_label} (Order Error Rate sweep)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"improvement_by_{factor_col}.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Improvement distribution per error level (ridge / joy plot)
# ---------------------------------------------------------------------------
def plot_improvement_distributions(df, sweep_col, fixed_col, label, out_dir):
    """Ridge plot (joy plot) of SHD improvement at each error level."""
    from scipy.stats import gaussian_kde

    sub = df[df[fixed_col] == 0.0].copy()
    rates = sorted(sub[sweep_col].unique())

    n_rates = len(rates)
    overlap = 0.6  # how much adjacent ridges overlap
    fig, axes = plt.subplots(n_rates, 1, figsize=(10, 1.2 * n_rates + 1),
                             sharex=True, gridspec_kw={"hspace": -overlap})

    # Common x range across all levels
    all_imp = sub["improvement"].dropna().values
    x_min, x_max = np.percentile(all_imp, [1, 99])
    pad = (x_max - x_min) * 0.15
    xs = np.linspace(x_min - pad, x_max + pad, 300)

    cmap = plt.cm.get_cmap("YlOrRd", n_rates + 2)

    for i, rate in enumerate(reversed(rates)):  # top = lowest error
        ax = axes[i]
        vals = sub.loc[sub[sweep_col] == rate, "improvement"].dropna().values

        if len(vals) > 2 and np.std(vals) > 1e-9:
            kde = gaussian_kde(vals, bw_method=0.3)
            density = kde(xs)
        else:
            density = np.zeros_like(xs)

        color = cmap(n_rates - 1 - i + 1)
        ax.fill_between(xs, density, alpha=0.7, color=color)
        ax.plot(xs, density, color="k", lw=0.8)
        ax.axvline(0, ls="--", color=COLOR_NEUTRAL, lw=0.8, alpha=0.6)

        # Mean marker
        mean_val = np.mean(vals)
        ax.axvline(mean_val, ls="-", color=COLOR_ORIG, lw=1.2, alpha=0.8)

        ax.set_yticks([])
        ax.set_ylabel(f"{rate:.0%}", rotation=0, ha="right", va="center", fontsize=10)
        ax.patch.set_alpha(0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("SHD Improvement (positive = CausalMorph better)")
    axes[0].set_title(f"Distribution of SHD Improvement vs {label}", fontsize=12)

    # Legend for reference lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], ls="--", color=COLOR_NEUTRAL, lw=0.8, label="Zero (no change)"),
        Line2D([0], [0], ls="-", color=COLOR_ORIG, lw=1.2, label="Mean improvement"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=8, loc="upper right",
                   framealpha=0.7)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"ridge_{sweep_col}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Win / Tie / Loss stacked bars
# ---------------------------------------------------------------------------
def plot_win_tie_loss(df, sweep_col, fixed_col, label, out_dir):
    sub = df[df[fixed_col] == 0.0].copy()
    rates = sorted(sub[sweep_col].unique())

    wins, ties, losses = [], [], []
    for r in rates:
        g = sub[sub[sweep_col] == r]["improvement"]
        n = len(g)
        wins.append((g > 0).sum() / n * 100)
        ties.append((g == 0).sum() / n * 100)
        losses.append((g < 0).sum() / n * 100)

    x = np.arange(len(rates))
    width = 0.6
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, wins, width, label="Win (CausalMorph better)", color=COLOR_IMPROVE, alpha=0.8)
    ax.bar(x, ties, width, bottom=wins, label="Tie", color=COLOR_NEUTRAL, alpha=0.5)
    ax.bar(x, losses, width, bottom=np.array(wins) + np.array(ties),
           label="Loss (CausalMorph worse)", color=COLOR_DEGRADE, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in rates])
    ax.set_xlabel(label)
    ax.set_ylabel("Percentage of scenarios")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(f"Win / Tie / Loss vs {label}")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"win_tie_loss_{sweep_col}.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Additional metrics table
# ---------------------------------------------------------------------------
def extended_metrics_table(df, sweep_col, fixed_col):
    """Return a DataFrame with MCC, precision, recall, specificity per level."""
    sub = df[df[fixed_col] == 0.0]
    rates = sorted(sub[sweep_col].unique())
    rows = []
    for r in rates:
        g = sub[sub[sweep_col] == r]
        rows.append({
            sweep_col: r,
            "n": len(g),
            "mcc_orig": g["mcc_original"].mean(),
            "mcc_trans": g["mcc_transformed"].mean(),
            "prec_orig": g["precision_original"].mean(),
            "prec_trans": g["precision_transformed"].mean(),
            "recall_orig": g["recall_original"].mean(),
            "recall_trans": g["recall_transformed"].mean(),
            "spec_orig": g["specificity_original"].mean(),
            "spec_trans": g["specificity_transformed"].mean(),
            "mae_deg_orig": g["mae_degree_original"].mean(),
            "mae_deg_trans": g["mae_degree_transformed"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real pipeline analysis
# ---------------------------------------------------------------------------
def analyze_real_pipeline(df_real, out_dir):
    """
    Produce a summary table and plot for the real pipeline experiments.

    Parameters
    ----------
    df_real : DataFrame – rows with experiment_type == "real_pipeline"
    out_dir : str – output directory for plots
    """
    if len(df_real) == 0:
        print("  No real pipeline experiments found.")
        return None

    n = len(df_real)
    improvement = df_real["improvement"].values
    pct_improved = (improvement > 0).mean() * 100
    pct_equal = (improvement == 0).mean() * 100

    non_zero = improvement[improvement != 0]
    if len(non_zero) > 10:
        w_stat, w_p = stats.wilcoxon(non_zero, alternative="two-sided")
    else:
        w_stat, w_p = np.nan, np.nan

    shd_orig_arr = df_real["shd_original"].values
    shd_trans_arr = df_real["shd_transformed"].values
    _, perm_p = paired_permutation_test(shd_orig_arr, shd_trans_arr,
                                        n_perm=10_000, seed=42)
    d_shd = cohens_d(shd_orig_arr, shd_trans_arr)

    summary = {
        "n": n,
        "measured_order_error_mean": df_real["measured_order_error_rate"].mean(),
        "measured_order_error_std": df_real["measured_order_error_rate"].std(),
        "measured_adj_error_mean": df_real["measured_adj_error_rate"].mean(),
        "measured_adj_error_std": df_real["measured_adj_error_rate"].std(),
        "kendall_tau_mean": df_real["kendall_tau"].mean(),
        "f1_input_adj_mean": df_real["f1_input_adj"].mean(),
        "shd_orig_mean": df_real["shd_original"].mean(),
        "shd_trans_mean": df_real["shd_transformed"].mean(),
        "shd_imp_mean": np.mean(improvement),
        "shd_imp_median": np.median(improvement),
        "pct_improved": pct_improved,
        "pct_degraded": 100 - pct_improved - pct_equal,
        "wilcoxon_p": w_p,
        "permutation_p": perm_p,
        "cohens_d": d_shd,
        "f1_orig_mean": df_real["f1_original"].mean(),
        "f1_trans_mean": df_real["f1_transformed"].mean(),
        "f1_imp_mean": df_real["f1_improvement"].mean(),
    }

    print(f"\n  Experiments:                 {n}")
    print(f"  Measured order error rate:   "
          f"{summary['measured_order_error_mean']:.4f} +/- "
          f"{summary['measured_order_error_std']:.4f}")
    print(f"  Measured adj error rate:     "
          f"{summary['measured_adj_error_mean']:.4f} +/- "
          f"{summary['measured_adj_error_std']:.4f}")
    print(f"  Kendall tau (order):         {summary['kendall_tau_mean']:.4f}")
    print(f"  F1 of input adj matrix:      {summary['f1_input_adj_mean']:.4f}")
    print(f"\n  SHD original (mean):         {summary['shd_orig_mean']:.4f}")
    print(f"  SHD transformed (mean):      {summary['shd_trans_mean']:.4f}")
    print(f"  SHD improvement (mean):      {summary['shd_imp_mean']:.4f}")
    print(f"  Win rate:                    {summary['pct_improved']:.1f}%")
    print(f"  Wilcoxon p:                  {summary['wilcoxon_p']:.2e}  "
          f"{significance_stars(summary['wilcoxon_p'])}")
    print(f"  Permutation p (paired):      {summary['permutation_p']:.2e}  "
          f"{significance_stars(summary['permutation_p'])}")
    print(f"  Cohen's d:                   {summary['cohens_d']:.3f}")
    print(f"\n  F1 original (mean):          {summary['f1_orig_mean']:.4f}")
    print(f"  F1 transformed (mean):       {summary['f1_trans_mean']:.4f}")
    print(f"  F1 improvement (mean):       {summary['f1_imp_mean']:.4f}")

    # Save summary
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "real_pipeline_summary.csv"), index=False
    )

    # Scatter: measured error rates vs improvement
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Real Pipeline: LiNGAM Error vs CausalMorph Improvement",
                 fontsize=13, fontweight="bold")

    for ax, xcol, xlabel in [
        (axes[0], "measured_order_error_rate", "Measured Order Error Rate"),
        (axes[1], "measured_adj_error_rate", "Measured Adj Matrix Error Rate"),
    ]:
        ax.scatter(df_real[xcol], df_real["improvement"], alpha=0.4, s=25,
                   color=COLOR_MORPH, edgecolors="black", linewidths=0.3)
        ax.axhline(0, ls="--", color=COLOR_NEUTRAL, lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SHD Improvement (positive = better)")
        ax.grid(True, alpha=0.2)
        # Trend line
        mask = np.isfinite(df_real[xcol].values) & np.isfinite(df_real["improvement"].values)
        if mask.sum() > 5:
            z = np.polyfit(df_real[xcol].values[mask],
                           df_real["improvement"].values[mask], 1)
            xline = np.linspace(df_real[xcol].min(), df_real[xcol].max(), 100)
            ax.plot(xline, np.polyval(z, xline), color=COLOR_DEGRADE, lw=2,
                    ls="--", alpha=0.7, label=f"Trend (slope={z[0]:.3f})")
            ax.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(out_dir, "real_pipeline_error_vs_improvement.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fname}")

    return summary


# ---------------------------------------------------------------------------
# Permutation-based monotone-degradation test across error levels
# ---------------------------------------------------------------------------
def pairwise_permutation_tests(df, sweep_col, fixed_col, out_dir,
                               n_perm=10_000, label="Order Error Rate"):
    """
    For every pair of consecutive error levels (η_low < η_high), test whether
    CausalMorph's improvement at η_low is significantly greater than at η_high.

    H0: mean_improvement(η_low) == mean_improvement(η_high)
    Test statistic: mean(d_low) - mean(d_high)
    Null distribution: generated by pooling the two groups' differences and
                       drawing without replacement (unpaired permutation test).

    Returns
    -------
    result_df : DataFrame with columns [eta_low, eta_high, obs_diff, perm_p]
    """
    sub = df[df[fixed_col] == 0.0].copy()
    rates = sorted(sub[sweep_col].unique())

    rows = []
    for i in range(len(rates) - 1):
        r_lo, r_hi = rates[i], rates[i + 1]
        d_lo = sub.loc[sub[sweep_col] == r_lo, "improvement"].values.astype(float)
        d_hi = sub.loc[sub[sweep_col] == r_hi, "improvement"].values.astype(float)

        obs_stat = np.mean(d_lo) - np.mean(d_hi)
        pooled = np.concatenate([d_lo, d_hi])
        n_lo = len(d_lo)

        rng = np.random.RandomState(int((r_lo + r_hi) * 1000))
        null_dist = np.empty(n_perm)
        for k in range(n_perm):
            perm = rng.permutation(pooled)
            null_dist[k] = perm[:n_lo].mean() - perm[n_lo:].mean()

        # One-sided: observed should be > 0 (improvement degrades with more error)
        p_one = (null_dist >= obs_stat).mean()
        rows.append({
            "eta_low": r_lo, "eta_high": r_hi,
            "mean_imp_low": np.mean(d_lo), "mean_imp_high": np.mean(d_hi),
            "obs_diff": obs_stat,
            "perm_p_one_sided": p_one,
        })

    result_df = pd.DataFrame(rows)

    # ---- Print summary ----
    print("\n--- Pairwise Permutation Tests (monotone degradation across η) ---")
    print(f"  n_permutations = {n_perm:,}  |  H1: improvement(η_low) > improvement(η_high)")
    print(f"  {'η_low':>6}  {'η_high':>7}  {'Δmean':>8}  {'p (one-sided)':>16}  sig")
    for _, r in result_df.iterrows():
        sig = significance_stars(r["perm_p_one_sided"])
        print(f"  {r['eta_low']:>6.0%}  {r['eta_high']:>7.0%}  "
              f"{r['obs_diff']:>8.4f}  {r['perm_p_one_sided']:>16.4f}  {sig}")

    # ---- Plot null distributions for key pairs ----
    n_pairs = len(rows)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4), sharey=False)
    if n_pairs == 1:
        axes = [axes]

    for ax, row_dict in zip(axes, rows):
        r_lo, r_hi = row_dict["eta_low"], row_dict["eta_high"]
        d_lo = sub.loc[sub[sweep_col] == r_lo, "improvement"].values.astype(float)
        d_hi = sub.loc[sub[sweep_col] == r_hi, "improvement"].values.astype(float)

        pooled = np.concatenate([d_lo, d_hi])
        n_lo = len(d_lo)
        rng = np.random.RandomState(int((r_lo + r_hi) * 1000))
        null_dist = np.array([
            rng.permutation(pooled)[:n_lo].mean() - rng.permutation(pooled)[n_lo:].mean()
            for _ in range(n_perm)
        ])
        obs = row_dict["obs_diff"]

        ax.hist(null_dist, bins=60, color=COLOR_NEUTRAL, alpha=0.6,
                edgecolor="none", label="Null distribution")
        ax.axvline(obs, color=COLOR_MORPH, lw=2, label=f"Observed Δ = {obs:.4f}")
        ax.set_xlabel("Δ mean improvement")
        ax.set_ylabel("Count")
        ax.set_title(f"η {r_lo:.0%} → {r_hi:.0%}\n"
                     f"p = {row_dict['perm_p_one_sided']:.4f} "
                     f"{significance_stars(row_dict['perm_p_one_sided'])}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Permutation Test: Monotone Degradation across {label}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"permutation_test_{sweep_col}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fname}")

    result_df.to_csv(
        os.path.join(out_dir, f"pairwise_permutation_{sweep_col}.csv"), index=False)

    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 80)
    print("CausalMorph Sensitivity Analysis — Statistical Report")
    print("=" * 80)
    print(f"Input : {csv_path}")
    print(f"Output: {out_dir}/")

    df = pd.read_csv(csv_path)
    print(f"Rows  : {len(df):,}")
    print(f"Unique scenarios: {df['scenario_name'].nunique()}")

    # Split controlled vs real pipeline
    has_type_col = "experiment_type" in df.columns
    if has_type_col:
        df_controlled = df[df["experiment_type"] == "controlled"].copy()
        df_real = df[df["experiment_type"] == "real_pipeline"].copy()
        print(f"Controlled experiments: {len(df_controlled):,}")
        print(f"Real pipeline experiments: {len(df_real):,}")
    else:
        df_controlled = df.copy()
        df_real = pd.DataFrame()
        print("(No experiment_type column — treating all as controlled)")

    print(f"Variables p     : {sorted(df_controlled['p'].unique())}")
    print(f"Error rates     : {sorted(df_controlled['adj_error_rate'].dropna().unique())}")

    # ---- 1. Main sweep plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    print("\n--- Order Error Rate Sweep (order_error_rate = 0) ---")
    sum_adj = analyze_sweep(
        df_controlled, "adj_error_rate", "order_error_rate", "Order Error Rate", ax,
        df_real=df_real if len(df_real) > 0 else None,
    )
    print(sum_adj.to_string(index=False, float_format="%.4f"))

    fig.suptitle("CausalMorph Sensitivity to Perturbations in Prior Knowledge",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = os.path.join(out_dir, "sensitivity_main.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fname}")

    # ---- 2. Violin distributions ----
    print("\nGenerating distribution plots...")
    plot_improvement_distributions(
        df_controlled, "adj_error_rate", "order_error_rate", "Order Error Rate", out_dir)

    # ---- 3. Win / Tie / Loss ----
    print("Generating win/tie/loss plots...")
    plot_win_tie_loss(
        df_controlled, "adj_error_rate", "order_error_rate", "Order Error Rate", out_dir)

    # ---- 4. Breakdown by graph size and density ----
    print("Generating factor breakdowns...")
    try:
        plot_improvement_by_factor(df_controlled, "p_bin", "Graph Size (p)", "improvement", out_dir)
    except Exception as e:
        print(f"  Skipped p_bin plot: {e}")
    try:
        plot_improvement_by_factor(df_controlled, "density_bin", "Graph Density", "improvement", out_dir)
    except Exception as e:
        print(f"  Skipped density_bin plot: {e}")

    # ---- 5. Extended metrics table ----
    print("\n--- Extended Metrics: Order Error Rate Sweep ---")
    ext_adj = extended_metrics_table(df_controlled, "adj_error_rate", "order_error_rate")
    print(ext_adj.to_string(index=False, float_format="%.4f"))

    # ---- 6. Pairwise permutation tests across error levels ----
    print("\nRunning pairwise permutation tests (this may take ~30 s)...")
    perm_adj = pairwise_permutation_tests(
        df_controlled, "adj_error_rate", "order_error_rate", out_dir,
        n_perm=10_000, label="Order Error Rate",
    )

    # ---- 7. Save summary tables ----
    sum_adj.to_csv(os.path.join(out_dir, "summary_order_error_sweep.csv"), index=False)
    ext_adj.to_csv(os.path.join(out_dir, "extended_metrics_order_error.csv"), index=False)
    print(f"\nAll CSV tables saved to {out_dir}/")

    # ---- 8. Real pipeline analysis ----
    if len(df_real) > 0:
        print("\n" + "=" * 80)
        print("REAL PIPELINE ANALYSIS (LiNGAM -> CausalMorph -> LiNGAM)")
        print("=" * 80)
        real_summary = analyze_real_pipeline(df_real, out_dir)

    # ---- 9. Key takeaways ----
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    baseline = sum_adj[sum_adj["adj_error_rate"] == 0.0].iloc[0]
    worst = sum_adj[sum_adj["adj_error_rate"] == sum_adj["adj_error_rate"].max()].iloc[0]

    print(f"\n  Baseline (0% error):")
    print(f"    SHD improvement : {baseline['shd_imp_mean']:.4f}  "
          f"[{baseline['shd_imp_ci_lo']:.4f}, {baseline['shd_imp_ci_hi']:.4f}]")
    print(f"    Win rate        : {baseline['pct_improved']:.1f}%")
    print(f"    Wilcoxon p (SHD): {baseline['wilcoxon_p_shd']:.2e}  "
          f"{significance_stars(baseline['wilcoxon_p_shd'])}")
    print(f"    Permutation p   : {baseline['permutation_p_shd']:.2e}  "
          f"{significance_stars(baseline['permutation_p_shd'])}")

    print(f"\n  Worst order error rate ({worst['adj_error_rate']:.0%}):")
    print(f"    SHD improvement : {worst['shd_imp_mean']:.4f}  "
          f"[{worst['shd_imp_ci_lo']:.4f}, {worst['shd_imp_ci_hi']:.4f}]")
    print(f"    Win rate        : {worst['pct_improved']:.1f}%")
    print(f"    Cohen's d (SHD) : {worst['cohens_d_shd']:.3f}")

    still_positive = sum_adj[sum_adj["shd_imp_mean"] > 0]
    if len(still_positive) == len(sum_adj):
        print("\n  CausalMorph maintains positive mean SHD improvement at ALL order error rate levels.")
    else:
        crossover = sum_adj[sum_adj["shd_imp_mean"] <= 0].iloc[0]["adj_error_rate"]
        print(f"\n  CausalMorph benefit disappears at order error rate >= {crossover:.0%}.")

    if len(df_real) > 0:
        real_imp = df_real["improvement"].mean()
        real_err = df_real["measured_adj_error_rate"].mean()
        print(f"\n  Real Pipeline (mean adj error = {real_err:.2%}):")
        print(f"    SHD improvement : {real_imp:.4f}")
        print(f"    Win rate        : {(df_real['improvement'] > 0).mean() * 100:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/benchmarks/analyze_order_sensitivity.py "
              "<results_csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(csv_path), "sensitivity_analysis"
    )
    main(csv_path, out_dir)
