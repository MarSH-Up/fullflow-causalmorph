#!/Users/mdlsh/miniconda3/envs/IYCC-env/bin/python
"""
Full analysis of batch_ablation.csv.

Produces 7 figures and 2 statistics CSVs in ./plots/:
  fig1_overview        — setup: sample sizes, detected CPs, TP/FP/FN breakdown
  fig2_detection       — change-point detection F1 / Precision / Recall vs p
  fig3_per_regime      — per-regime structure recovery (all 4 methods, 5 metrics)
  fig4_consensus       — consensus structure recovery (shared + Pipeline-only metrics)
  fig5_distributions   — violin distributions per method × p (4 key metrics)
  fig6_significance    — Bonferroni-corrected Wilcoxon heatmap + effect sizes
  fig7_comparison      — SLIDE-READY: grouped bar chart Pipeline vs baselines + detection
  stats_descriptive    — mean/std/median/IQR/CI per (method, p, metric)
  stats_tests          — Wilcoxon p-values, Bonferroni correction, effect sizes

Usage:
    python analyze_ablation.py [--csv PATH] [--out DIR]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# ── method / column mapping ────────────────────────────────────────────────────

# Each key is a canonical metric name; value is the CSV column for that method.
METHODS = {
    'Pipeline': {
        'mean_f1':   'mean_struct_f1',
        'mean_prec': 'mean_struct_precision',
        'mean_rec':  'mean_struct_recall',
        'mean_nshd': 'mean_norm_shd',
        'mean_shd':  'mean_shd',
        'cons_f1':   'consensus_f1',
        'cons_prec': 'consensus_precision',
        'cons_rec':  'consensus_recall',
        'cons_nshd': 'consensus_norm_shd',
        'cons_shd':  'consensus_shd',
    },
    'CausalMorph': {
        'mean_f1':   'causalmorph_mean_f1',
        'mean_prec': 'causalmorph_mean_precision',
        'mean_rec':  'causalmorph_mean_recall',
        'mean_nshd': 'causalmorph_mean_norm_shd',
        'mean_shd':  'causalmorph_mean_shd',
        'cons_f1':   'causalmorph_cons_f1',
        'cons_nshd': 'causalmorph_cons_norm_shd',
    },
    'DirectLiNGAM': {
        'mean_f1':   'directlingam_mean_f1',
        'mean_prec': 'directlingam_mean_precision',
        'mean_rec':  'directlingam_mean_recall',
        'mean_nshd': 'directlingam_mean_norm_shd',
        'mean_shd':  'directlingam_mean_shd',
        'cons_f1':   'directlingam_cons_f1',
        'cons_nshd': 'directlingam_cons_norm_shd',
    },
    'ICALiNGAM': {
        'mean_f1':   'icalingam_mean_f1',
        'mean_prec': 'icalingam_mean_precision',
        'mean_rec':  'icalingam_mean_recall',
        'mean_nshd': 'icalingam_mean_norm_shd',
        'mean_shd':  'icalingam_mean_shd',
        'cons_f1':   'icalingam_cons_f1',
        'cons_nshd': 'icalingam_cons_norm_shd',
    },
}

BASELINES     = ['CausalMorph', 'DirectLiNGAM', 'ICALiNGAM']
SHARED_METRICS = ['mean_f1', 'mean_prec', 'mean_rec', 'mean_nshd', 'mean_shd',
                  'cons_f1', 'cons_nshd']

METRIC_LABEL = {
    'mean_f1':   'Mean F1',
    'mean_prec': 'Mean Precision',
    'mean_rec':  'Mean Recall',
    'mean_nshd': 'Mean Norm. SHD',
    'mean_shd':  'Mean SHD',
    'cons_f1':   'Consensus F1',
    'cons_prec': 'Consensus Precision',
    'cons_rec':  'Consensus Recall',
    'cons_nshd': 'Consensus Norm. SHD',
    'cons_shd':  'Consensus SHD',
}
HIGHER_BETTER = {
    'mean_f1', 'mean_prec', 'mean_rec',
    'cons_f1', 'cons_prec', 'cons_rec',
}

COLORS  = {'Pipeline': '#2166ac', 'CausalMorph': '#d6604d',
           'DirectLiNGAM': '#4dac26', 'ICALiNGAM': '#8073ac'}
MARKERS = {'Pipeline': 'o', 'CausalMorph': 's', 'DirectLiNGAM': '^', 'ICALiNGAM': 'D'}

# ── helpers ────────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    df = df[df['status'] == 'ok'].copy()
    df['p'] = df['p'].astype(int)
    return df.sort_values('p').reset_index(drop=True)


def ci95(arr):
    return 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))


def agg(series):
    v = np.asarray(series, float)
    return np.mean(v), ci95(v)


def wilcoxon_test(a, b):
    """Paired two-sided Wilcoxon; returns (bonf-ready p, rank-biserial r)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, 0.0
    try:
        stat, pval = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')
        n = len(nz)
        r_mag = 1.0 - 2.0 * stat / (n * (n + 1) / 2.0)
        r = float(np.clip(r_mag * np.sign(np.mean(d)), -1, 1))
        return pval, r
    except Exception:
        return 1.0, 0.0


def sig_star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def line_ci(ax, xs, means, cis, color, marker, label):
    ax.plot(xs, means, color=color, marker=marker, label=label, zorder=3, lw=1.6, ms=5)
    ax.fill_between(xs, means - cis, means + cis, color=color, alpha=0.15, zorder=2)


def style(ax, xticks, ylabel='', title=''):
    ax.set_xticks(xticks)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Number of nodes $p$', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_fig(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, f'{name}.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  {name}.png')


# ── plot style ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'lines.linewidth': 1.6, 'lines.markersize': 5, 'figure.dpi': 150,
})

# ══════════════════════════════════════════════════════════════════════════════

def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df     = load(csv_path)
    P      = sorted(df['p'].unique())
    grp    = df.groupby('p')

    def gv(p, col):
        return grp.get_group(p)[col].to_numpy(float)

    # ── pre-compute Wilcoxon tests (for annotation in figs 3 & 4) ──────────

    raw_pv, eff_r = {}, {}
    for bl in BASELINES:
        for mkey in SHARED_METRICS:
            for p in P:
                a = gv(p, METHODS['Pipeline'][mkey])
                b = gv(p, METHODS[bl][mkey])
                raw_pv[(bl, mkey, p)], eff_r[(bl, mkey, p)] = wilcoxon_test(a, b)

    n_tests  = len(raw_pv)
    corr_pv  = {k: min(v * n_tests, 1.0) for k, v in raw_pv.items()}

    def any_sig(mkey, p):
        """True if Pipeline differs from any baseline at this (metric, p)."""
        return any(corr_pv[(bl, mkey, p)] < 0.05 for bl in BASELINES
                   if (bl, mkey, p) in corr_pv)

    # ── Figure 1 — Experimental overview ─────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Experimental Overview', fontweight='bold')

    # 1a — sample-size distribution
    ax = axes[0]
    ax.boxplot([gv(p, 'total_samples') for p in P], positions=P, widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor='#cce5ff', color='#2166ac'),
               medianprops=dict(color='#d6604d', lw=2),
               flierprops=dict(marker='.', color='grey', alpha=0.5))
    style(ax, P, ylabel='Total samples', title='Sample size per $p$')

    # 1b — detected vs true change-points
    ax = axes[1]
    ms = np.array([gv(p, 'n_detected_cps').mean() for p in P])
    cs = np.array([ci95(gv(p, 'n_detected_cps')) for p in P])
    ax.axhline(4, color='grey', lw=1.2, ls='--', label='True CPs = 4')
    line_ci(ax, P, ms, cs, '#d6604d', 'o', 'Detected CPs (mean ± 95% CI)')
    ax.legend(fontsize=8)
    style(ax, P, ylabel='# Change-points', title='Detected vs true CPs')

    # 1c — TP / FP / FN stacked bar
    ax = axes[2]
    tp = np.array([gv(p, 'det_tp').mean() for p in P])
    fp = np.array([gv(p, 'det_fp').mean() for p in P])
    fn = np.array([gv(p, 'det_fn').mean() for p in P])
    x  = np.arange(len(P))
    ax.bar(x, tp, 0.6, label='TP', color='#4dac26', alpha=0.85)
    ax.bar(x, fp, 0.6, bottom=tp, label='FP', color='#d6604d', alpha=0.85)
    ax.bar(x, fn, 0.6, bottom=tp + fp, label='FN', color='#aaa', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(P)
    ax.legend(fontsize=8)
    style(ax, x, ylabel='Count (mean)', title='Detection TP / FP / FN')
    ax.set_xlabel('Number of nodes $p$', fontsize=9)

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig1_overview')

    # ── Figure 2 — Detection performance ─────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle('Change-point Detection Performance (Pipeline only)', fontweight='bold')

    det_cfg = [
        ('det_f1',        'F1',        '#1b7837'),
        ('det_precision', 'Precision', '#762a83'),
        ('det_recall',    'Recall',    '#d6604d'),
    ]
    for ax, (col, label, color) in zip(axes, det_cfg):
        ms = np.array([agg(gv(p, col))[0] for p in P])
        cs = np.array([agg(gv(p, col))[1] for p in P])
        line_ci(ax, P, ms, cs, color, 'o', label)
        ax.set_ylim(0, 1.05)
        style(ax, P, ylabel=label, title=f'Detection {label}')

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig2_detection')

    # ── shared helper: line-panel for all 4 methods ───────────────────────

    def method_panel(ax, mkey, sig_annot=True):
        """Line + CI for all methods that have mkey; optional sig stars."""
        for method in METHODS:
            col = METHODS[method].get(mkey)
            if col is None:
                continue
            ms = np.array([agg(gv(p, col))[0] for p in P])
            cs = np.array([agg(gv(p, col))[1] for p in P])
            line_ci(ax, P, ms, cs, COLORS[method], MARKERS[method], method)
        ax.legend(fontsize=7.5, loc='best', framealpha=0.85)
        style(ax, P, ylabel=METRIC_LABEL[mkey].split()[-1],
              title=METRIC_LABEL[mkey])

        if sig_annot and mkey in SHARED_METRICS:
            ylo, yhi = ax.get_ylim()
            for p in P:
                if any_sig(mkey, p):
                    ymax = max(
                        agg(gv(p, METHODS[m][mkey]))[0] + agg(gv(p, METHODS[m][mkey]))[1]
                        for m in METHODS if METHODS[m].get(mkey)
                    )
                    ax.annotate('*', xy=(p, ymax + 0.025 * (yhi - ylo)),
                                ha='center', va='bottom', fontsize=8, color='#333')

    # ── Figure 3 — Per-regime structure (all 4 methods) ──────────────────────

    per_regime = ['mean_f1', 'mean_prec', 'mean_rec', 'mean_nshd', 'mean_shd']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Per-regime Structure Recovery — All Methods\n'
                 '(* at least one baseline differs, Bonferroni p < 0.05)',
                 fontweight='bold')
    for mkey, ax in zip(per_regime, axes.flat):
        method_panel(ax, mkey)
    axes.flat[-1].set_visible(False)

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig3_per_regime')

    # ── Figure 4 — Consensus structure ───────────────────────────────────────

    cons_shared   = ['cons_f1', 'cons_nshd']
    cons_pipeline = ['cons_prec', 'cons_rec', 'cons_shd']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Consensus Structure Recovery\n'
                 '(* at least one baseline differs, Bonferroni p < 0.05)',
                 fontweight='bold')
    axflat = list(axes.flat)

    for i, mkey in enumerate(cons_shared):
        method_panel(axflat[i], mkey)

    for i, mkey in enumerate(cons_pipeline):
        ax = axflat[len(cons_shared) + i]
        col = METHODS['Pipeline'][mkey]
        ms = np.array([agg(gv(p, col))[0] for p in P])
        cs = np.array([agg(gv(p, col))[1] for p in P])
        line_ci(ax, P, ms, cs, COLORS['Pipeline'], MARKERS['Pipeline'], 'Pipeline')
        ax.legend(fontsize=7.5)
        style(ax, P, ylabel=METRIC_LABEL[mkey].split()[-1],
              title=METRIC_LABEL[mkey] + ' (Pipeline only)')

    axflat[-1].set_visible(False)
    plt.tight_layout()
    save_fig(fig, out_dir, 'fig4_consensus')

    # ── Figure 5 — Violin distributions ──────────────────────────────────────

    violin_cfg = [
        ('mean_f1',   'Per-regime F1'),
        ('mean_nshd', 'Per-regime Norm. SHD'),
        ('cons_f1',   'Consensus F1'),
        ('cons_nshd', 'Consensus Norm. SHD'),
    ]
    n_m    = len(METHODS)
    gapw   = 1.3
    grpw   = n_m + gapw

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Score Distributions by Method and Graph Size\n'
                 '(violins = density, bar = IQR, dot = median)',
                 fontweight='bold')

    for (mkey, title), ax in zip(violin_cfg, axes.flat):
        positions, vdata, vcolors = [], [], []
        for i, p in enumerate(P):
            for j, (method, cols) in enumerate(METHODS.items()):
                col = cols.get(mkey)
                if col is None:
                    continue
                positions.append(i * grpw + j)
                vdata.append(gv(p, col))
                vcolors.append(COLORS[method])

        parts = ax.violinplot(vdata, positions=positions, widths=0.72,
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts['bodies'], vcolors):
            pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor('none')
        parts['cmedians'].set_color('#111'); parts['cmedians'].set_linewidth(1.4)

        q1s  = [np.percentile(v, 25) for v in vdata]
        q3s  = [np.percentile(v, 75) for v in vdata]
        meds = [np.median(v)          for v in vdata]
        ax.vlines(positions, q1s, q3s, color='#444', lw=1.0, alpha=0.6, zorder=4)
        ax.scatter(positions, meds, s=10, color='#111', zorder=5)

        ticks = [i * grpw + (n_m - 1) / 2 for i in range(len(P))]
        ax.set_xticks(ticks); ax.set_xticklabels(P)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Number of nodes $p$', fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        handles = [Patch(facecolor=COLORS[m], alpha=0.6, label=m) for m in METHODS]
        ax.legend(handles=handles, fontsize=7.5, loc='best', framealpha=0.85)

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig5_distributions')

    # ── Figure 6 — Significance heatmap + effect sizes ───────────────────────

    nrows = len(BASELINES) * len(SHARED_METRICS)
    heat_p = np.zeros((nrows, len(P)))
    heat_r = np.zeros((nrows, len(P)))
    ylabels = []
    for i, bl in enumerate(BASELINES):
        for j, mkey in enumerate(SHARED_METRICS):
            row = i * len(SHARED_METRICS) + j
            ylabels.append(f'{bl}  /  {METRIC_LABEL[mkey]}')
            for k, p in enumerate(P):
                heat_p[row, k] = corr_pv[(bl, mkey, p)]
                heat_r[row, k] = eff_r[(bl, mkey, p)]

    log_p = -np.log10(np.clip(heat_p, 1e-15, 1.0))
    vmax  = max(3.0, np.nanmax(log_p))
    ext   = [P[0] - 0.5, P[-1] + 0.5, nrows - 0.5, -0.5]

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(17, 9))
    fig.suptitle(
        'Pipeline vs Baselines — Bonferroni-corrected Wilcoxon tests (n = 100 per cell)\n'
        'Left: significance  |  Right: rank-biserial effect size'
        ' (+1 = Pipeline always better, −1 = always worse)',
        fontweight='bold', fontsize=10,
    )

    im1 = ax_p.imshow(log_p, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax, extent=ext)
    ax_p.set_yticks(range(nrows)); ax_p.set_yticklabels(ylabels, fontsize=7.5)
    ax_p.set_xticks(P); ax_p.set_xlabel('Number of nodes $p$')
    ax_p.set_title('$-\\log_{10}(p_{\\mathrm{Bonf}})$', fontsize=10)
    cb1 = plt.colorbar(im1, ax=ax_p, fraction=0.025, pad=0.02)
    cb1.ax.axhline(-np.log10(0.05),  color='white', lw=1.5, ls='--')
    cb1.ax.axhline(-np.log10(0.01),  color='white', lw=1.0, ls=':')
    cb1.set_label('$-\\log_{10}(p)$  [dashed = 0.05]', fontsize=8)

    # add separator lines between baseline blocks
    for i in range(1, len(BASELINES)):
        ax_p.axhline(i * len(SHARED_METRICS) - 0.5, color='white', lw=1.8)
    for i in range(nrows):
        for k, p in enumerate(P):
            s = sig_star(heat_p[i, k])
            if s != 'ns':
                ax_p.text(p, i, s, ha='center', va='center', fontsize=6.5,
                          color='white' if log_p[i, k] > vmax * 0.55 else 'black',
                          fontweight='bold')

    im2 = ax_r.imshow(heat_r, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, extent=ext)
    ax_r.set_yticks(range(nrows)); ax_r.set_yticklabels(ylabels, fontsize=7.5)
    ax_r.set_xticks(P); ax_r.set_xlabel('Number of nodes $p$')
    ax_r.set_title('Rank-biserial correlation $r$ (effect size)', fontsize=10)
    plt.colorbar(im2, ax=ax_r, fraction=0.025, pad=0.02, label='$r$')
    for i in range(1, len(BASELINES)):
        ax_r.axhline(i * len(SHARED_METRICS) - 0.5, color='grey', lw=1.8)
    for i in range(nrows):
        for k, p in enumerate(P):
            rv = heat_r[i, k]
            ax_r.text(p, i, f'{rv:+.2f}', ha='center', va='center', fontsize=6,
                      color='white' if abs(rv) > 0.6 else '#222')

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig6_significance')

    # ── Figure 7 — Slide-ready comparison summary ────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Pipeline vs Baselines — At a Glance', fontweight='bold', fontsize=13)

    # ── left: grouped bar chart per metric, all methods, pooled across all p ──
    # ── left: mean_nshd per p per method — grouped bars ──────────────────────
    ax = axes[0]
    bl_methods = BASELINES
    n_m  = len(bl_methods)
    bw   = 0.22
    offs = np.linspace(-(n_m - 1) / 2 * bw, (n_m - 1) / 2 * bw, n_m)
    dx   = np.arange(len(P))

    for j, method in enumerate(bl_methods):
        col = METHODS[method].get('mean_nshd')
        if col is None:
            continue
        ms  = np.array([agg(gv(p, col))[0] for p in P])
        cis = np.array([agg(gv(p, col))[1] for p in P])
        ax.bar(dx + offs[j], ms, bw,
               color=COLORS[method], alpha=0.75,
               label=method, zorder=3)
        ax.errorbar(dx + offs[j], ms, yerr=cis, fmt='none',
                    color='#333', capsize=3, lw=1.0, zorder=4)

    ax.set_xticks(dx)
    ax.set_xticklabels(P, fontsize=10)
    ax.set_xlabel('Number of nodes $p$', fontsize=9.5)
    ax.set_ylabel('Mean Norm. SHD (mean ± 95% CI)', fontsize=9.5)
    ax.set_title('Structure Recovery — Mean Norm. SHD', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.92)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.annotate('SHD: lower is better',
                xy=(0.01, 0.01), xycoords='axes fraction',
                fontsize=7.5, color='#666', ha='left', va='bottom')

    # ── right: detection precision per p (Pipeline) ──────────────────────────
    ax = axes[1]
    dp_means = np.array([agg(gv(p, 'det_precision'))[0] for p in P])
    dp_cis   = np.array([agg(gv(p, 'det_precision'))[1] for p in P])
    dx = np.arange(len(P))
    bars = ax.bar(dx, dp_means, 0.55,
                  color=COLORS['Pipeline'], alpha=0.88,
                  edgecolor='black', linewidth=1.2, zorder=3)
    ax.errorbar(dx, dp_means, yerr=dp_cis, fmt='none',
                color='#333', capsize=4, lw=1.2, zorder=4)
    for bar, val in zip(bars, dp_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color='#222')
    ax.set_xticks(dx)
    ax.set_xticklabels(P, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('Number of nodes $p$', fontsize=9.5)
    ax.set_ylabel('Precision (mean ± 95% CI)', fontsize=9.5)
    ax.set_title('Change-point Detection Precision\n(Pipeline)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig7_comparison')

    # ── stats_descriptive.csv ─────────────────────────────────────────────────

    desc_rows = []
    for method, cols in METHODS.items():
        for mkey, col in cols.items():
            for p in P:
                v = gv(p, col)
                desc_rows.append({
                    'method': method, 'p': p, 'metric': mkey, 'column': col,
                    'n': len(v),
                    'mean':   round(np.mean(v), 5),
                    'std':    round(np.std(v, ddof=1), 5),
                    'ci95':   round(ci95(v), 5),
                    'median': round(np.median(v), 5),
                    'q1':     round(np.percentile(v, 25), 5),
                    'q3':     round(np.percentile(v, 75), 5),
                    'min':    round(v.min(), 5),
                    'max':    round(v.max(), 5),
                })
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(os.path.join(out_dir, 'stats_descriptive.csv'), index=False)
    print(f'  stats_descriptive.csv ({len(desc_df)} rows)')

    # ── stats_tests.csv ───────────────────────────────────────────────────────

    test_rows = []
    for bl in BASELINES:
        for mkey in SHARED_METRICS:
            for p in P:
                cp = corr_pv[(bl, mkey, p)]
                test_rows.append({
                    'baseline': bl, 'metric': mkey, 'p': p,
                    'raw_p':       round(raw_pv[(bl, mkey, p)], 8),
                    'bonf_p':      round(cp, 8),
                    'effect_r':    round(eff_r[(bl, mkey, p)], 4),
                    'significant': cp < 0.05,
                    'stars':       sig_star(cp),
                    'n_tests_total': n_tests,
                })
    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv(os.path.join(out_dir, 'stats_tests.csv'), index=False)
    print(f'  stats_tests.csv ({len(tests_df)} rows, '
          f'{(tests_df.bonf_p < 0.05).sum()} significant)')

    # ── console summary ───────────────────────────────────────────────────────

    print('\n── Pooled mean ± 95% CI across all p ────────────────────────────────────')
    print(f"{'Method':<15} {'Mean F1':>10} {'Mean nSHD':>10} {'Cons F1':>10} {'Cons nSHD':>10}")
    print('─' * 58)
    for method, cols in METHODS.items():
        def pool(mkey):
            col = cols.get(mkey)
            if col is None:
                return '   —    '
            v = df[col].to_numpy(float)
            return f'{np.mean(v):.3f}±{ci95(v):.3f}'
        print(f'{method:<15} {pool("mean_f1"):>10} {pool("mean_nshd"):>10} '
              f'{pool("cons_f1"):>10} {pool("cons_nshd"):>10}')

    print(f'\n── Detection (pooled) ───────────────────────────────────────────────────')
    for col, label in [('det_f1', 'F1'), ('det_precision', 'Precision'), ('det_recall', 'Recall')]:
        v = df[col].to_numpy(float)
        print(f'  {label:<12} {np.mean(v):.3f} ± {ci95(v):.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=os.path.join(os.path.dirname(__file__),
                                                   'batch_ablation.csv'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'plots'))
    args = ap.parse_args()
    print(f'Reading  {args.csv}')
    print(f'Writing  {args.out}/')
    main(args.csv, args.out)
