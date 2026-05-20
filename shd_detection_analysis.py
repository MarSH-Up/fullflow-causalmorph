#!/Users/mdlsh/miniconda3/envs/IYCC-env/bin/python
"""
Focused analysis of batch_ablation.csv: SHD metrics and detection capacity.

Outputs (./plots/):
  fig_shd.{pdf,png}        — SHD line plots (mean + consensus) + violin distributions
  fig_detection.{pdf,png}  — Detection F1/P/R + CP count + TP/FP/FN breakdown
  stats_shd.csv            — descriptive stats + Wilcoxon tests for all SHD metrics
  stats_detection.csv      — descriptive stats for detection metrics vs p

Usage:
    python shd_detection_analysis.py [--csv PATH] [--out DIR]
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

# ── column map ─────────────────────────────────────────────────────────────────

# (method_name, label, color, marker) → {metric: column}
METHODS = {
    'Pipeline':    ('#2166ac', 'o', {
        'mean_nshd': 'mean_norm_shd',
        'mean_shd':  'mean_shd',
        'cons_nshd': 'consensus_norm_shd',
        'cons_shd':  'consensus_shd',          # pipeline-only
    }),
    'CausalMorph': ('#d6604d', 's', {
        'mean_nshd': 'causalmorph_mean_norm_shd',
        'mean_shd':  'causalmorph_mean_shd',
        'cons_nshd': 'causalmorph_cons_norm_shd',
    }),
    'DirectLiNGAM': ('#4dac26', '^', {
        'mean_nshd': 'directlingam_mean_norm_shd',
        'mean_shd':  'directlingam_mean_shd',
        'cons_nshd': 'directlingam_cons_norm_shd',
    }),
    'ICALiNGAM': ('#8073ac', 'D', {
        'mean_nshd': 'icalingam_mean_norm_shd',
        'mean_shd':  'icalingam_mean_shd',
        'cons_nshd': 'icalingam_cons_norm_shd',
    }),
}

# metrics shared by all 4 methods (used for significance testing)
SHARED_SHD = ['mean_nshd', 'mean_shd', 'cons_nshd']
BASELINES  = ['CausalMorph', 'DirectLiNGAM', 'ICALiNGAM']

# ── helpers ────────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    df = df[df['status'] == 'ok'].copy()
    df['p'] = df['p'].astype(int)
    return df.sort_values('p').reset_index(drop=True)


def gv(grp, p, col):
    return grp.get_group(p)[col].to_numpy(float)


def ci95(v):
    return 1.96 * np.std(v, ddof=1) / np.sqrt(len(v))


def agg(v):
    return np.mean(v), ci95(v)


def wilcoxon_test(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, 0.0
    stat, pval = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')
    n   = len(nz)
    r   = float(np.clip((1.0 - 2.0 * stat / (n * (n + 1) / 2)) * np.sign(np.mean(d)), -1, 1))
    return pval, r


def sig_star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def line_ci(ax, xs, means, cis, color, marker, label, lw=1.8, ms=6):
    ax.plot(xs, means, color=color, marker=marker, label=label, lw=lw, ms=ms, zorder=3)
    ax.fill_between(xs, means - cis, means + cis, color=color, alpha=0.18, zorder=2)


def tidy(ax, P, ylabel='', title='', ylim_bottom=None):
    ax.set_xticks(P)
    ax.set_xlabel('Number of nodes $p$', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)


def save_fig(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, f'{name}.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  {name}.png')


# ── style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'lines.linewidth': 1.8, 'lines.markersize': 6, 'figure.dpi': 150,
})

# ══════════════════════════════════════════════════════════════════════════════

def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df   = load(csv_path)
    P    = sorted(df['p'].unique())
    grp  = df.groupby('p')

    # ── Wilcoxon tests for SHD metrics ────────────────────────────────────────

    raw_p, eff_r = {}, {}
    for bl in BASELINES:
        _, _, bl_cols = METHODS[bl]
        _, _, pi_cols = METHODS['Pipeline']
        for mkey in SHARED_SHD:
            for p in P:
                a = gv(grp, p, pi_cols[mkey])
                b = gv(grp, p, bl_cols[mkey])
                raw_p[(bl, mkey, p)], eff_r[(bl, mkey, p)] = wilcoxon_test(a, b)

    n_tests = len(raw_p)
    corr_p  = {k: min(v * n_tests, 1.0) for k, v in raw_p.items()}

    def any_sig(mkey, p):
        return any(corr_p[(bl, mkey, p)] < 0.05 for bl in BASELINES)

    def annot_sig(ax, mkey, P, ypad=0.04):
        """Add * above each p where Pipeline differs from any baseline."""
        ylo, yhi = ax.get_ylim()
        rng = yhi - ylo
        for p in P:
            if not any_sig(mkey, p):
                continue
            # y-position: above tallest CI bar
            ymax = max(
                agg(gv(grp, p, METHODS[m][2][mkey]))[0] + agg(gv(grp, p, METHODS[m][2][mkey]))[1]
                for m in METHODS if mkey in METHODS[m][2]
            )
            ax.annotate('*', xy=(p, ymax + ypad * rng),
                        ha='center', va='bottom', fontsize=9, color='#333', fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1 — SHD
    # Layout: 3 rows × 3 cols
    #   Row 0: line plots — Mean nSHD, Mean SHD, Consensus nSHD (all 4 methods)
    #   Row 1: violin     — same three metrics
    #   Row 2: pipeline-only — Consensus raw SHD line + violin
    # ══════════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        'SHD Analysis — All Methods vs Graph Size\n'
        '(* Pipeline differs from ≥1 baseline, Bonferroni-corrected Wilcoxon p < 0.05)',
        fontweight='bold', fontsize=12,
    )

    shd_cfg = [
        ('mean_nshd', 'Mean Norm. SHD',      'Norm. SHD'),
        ('mean_shd',  'Mean SHD (raw)',       'SHD'),
        ('cons_nshd', 'Consensus Norm. SHD',  'Norm. SHD'),
    ]

    # ── row 0: line plots ──────────────────────────────────────────────────────
    for col_i, (mkey, title, ylabel) in enumerate(shd_cfg):
        ax = axes[0, col_i]
        for method, (color, marker, cols) in METHODS.items():
            if mkey not in cols:
                continue
            ms = np.array([agg(gv(grp, p, cols[mkey]))[0] for p in P])
            cs = np.array([agg(gv(grp, p, cols[mkey]))[1] for p in P])
            line_ci(ax, P, ms, cs, color, marker, method)
        tidy(ax, P, ylabel=ylabel, title=title, ylim_bottom=0)
        ax.legend(loc='upper left', framealpha=0.9)
        annot_sig(ax, mkey, P)

    # ── row 1: violin distributions ───────────────────────────────────────────
    n_m  = len(METHODS)
    gapw = 1.3
    grpw = n_m + gapw

    for col_i, (mkey, title, ylabel) in enumerate(shd_cfg):
        ax = axes[1, col_i]
        positions, vdata, vcolors = [], [], []
        for i, p in enumerate(P):
            for j, (method, (color, marker, cols)) in enumerate(METHODS.items()):
                if mkey not in cols:
                    continue
                positions.append(i * grpw + j)
                vdata.append(gv(grp, p, cols[mkey]))
                vcolors.append(color)

        parts = ax.violinplot(vdata, positions=positions, widths=0.72,
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts['bodies'], vcolors):
            pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor('none')
        parts['cmedians'].set_color('#111'); parts['cmedians'].set_linewidth(1.5)

        q1s  = [np.percentile(v, 25) for v in vdata]
        q3s  = [np.percentile(v, 75) for v in vdata]
        meds = [np.median(v) for v in vdata]
        ax.vlines(positions, q1s, q3s, color='#444', lw=1.1, alpha=0.55, zorder=4)
        ax.scatter(positions, meds, s=12, color='#111', zorder=5)

        ticks = [i * grpw + (n_m - 1) / 2 for i in range(len(P))]
        ax.set_xticks(ticks); ax.set_xticklabels(P)
        ax.set_xlabel('Number of nodes $p$', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f'{title} — distributions', fontsize=10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        handles = [Patch(facecolor=METHODS[m][0], alpha=0.6, label=m) for m in METHODS
                   if mkey in METHODS[m][2]]
        ax.legend(handles=handles, fontsize=7.5, loc='upper left', framealpha=0.85)

    # ── row 2: consensus raw SHD (pipeline only) + significance heatmap ───────

    # Panel (2,0): Consensus SHD pipeline line
    ax = axes[2, 0]
    color, marker, cols = METHODS['Pipeline']
    ms = np.array([agg(gv(grp, p, cols['cons_shd']))[0] for p in P])
    cs = np.array([agg(gv(grp, p, cols['cons_shd']))[1] for p in P])
    line_ci(ax, P, ms, cs, color, marker, 'Pipeline')
    tidy(ax, P, ylabel='SHD', title='Consensus SHD — raw (Pipeline only)', ylim_bottom=0)
    ax.legend(loc='upper left')

    # Panel (2,1): Consensus SHD violin (pipeline only)
    ax = axes[2, 1]
    positions2, vdata2 = [], []
    for i, p in enumerate(P):
        positions2.append(i)
        vdata2.append(gv(grp, p, cols['cons_shd']))
    parts = ax.violinplot(vdata2, positions=positions2, widths=0.65,
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color); pc.set_alpha(0.55); pc.set_edgecolor('none')
    parts['cmedians'].set_color('#111'); parts['cmedians'].set_linewidth(1.5)
    q1s2  = [np.percentile(v, 25) for v in vdata2]
    q3s2  = [np.percentile(v, 75) for v in vdata2]
    meds2 = [np.median(v) for v in vdata2]
    ax.vlines(positions2, q1s2, q3s2, color='#444', lw=1.1, alpha=0.55, zorder=4)
    ax.scatter(positions2, meds2, s=12, color='#111', zorder=5)
    ax.set_xticks(range(len(P))); ax.set_xticklabels(P)
    ax.set_xlabel('Number of nodes $p$', fontsize=9)
    ax.set_ylabel('SHD', fontsize=9)
    ax.set_title('Consensus SHD — distribution (Pipeline only)', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel (2,2): significance heatmap for SHD metrics
    ax = axes[2, 2]
    nrows = len(BASELINES) * len(SHARED_SHD)
    heat  = np.zeros((nrows, len(P)))
    heat_r = np.zeros((nrows, len(P)))
    ylabels = []
    row_labels = {'mean_nshd': 'Mean nSHD', 'mean_shd': 'Mean SHD', 'cons_nshd': 'Cons nSHD'}
    for i, bl in enumerate(BASELINES):
        for j, mkey in enumerate(SHARED_SHD):
            ridx = i * len(SHARED_SHD) + j
            ylabels.append(f'{bl}\n{row_labels[mkey]}')
            for k, p in enumerate(P):
                heat[ridx, k]   = corr_p[(bl, mkey, p)]
                heat_r[ridx, k] = eff_r[(bl, mkey, p)]

    log_p = -np.log10(np.clip(heat, 1e-15, 1.0))
    vmax  = max(3.0, np.nanmax(log_p))
    ext   = [P[0] - 0.5, P[-1] + 0.5, nrows - 0.5, -0.5]

    im = ax.imshow(log_p, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax, extent=ext)
    ax.set_yticks(range(nrows)); ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xticks(P); ax.set_xlabel('Number of nodes $p$', fontsize=9)
    ax.set_title('Pipeline significance vs baselines\n'
                 '$-\\log_{10}(p_{\\mathrm{Bonf}})$  |  cells: effect $r$', fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.axhline(-np.log10(0.05), color='white', lw=1.5, ls='--')
    cb.set_label('$-\\log_{10}(p)$', fontsize=8)
    for i in range(1, len(BASELINES)):
        ax.axhline(i * len(SHARED_SHD) - 0.5, color='white', lw=1.8)
    for ridx in range(nrows):
        for k, p in enumerate(P):
            rv = heat_r[ridx, k]
            ax.text(p, ridx, f'{rv:+.2f}', ha='center', va='center', fontsize=6.5,
                    color='white' if log_p[ridx, k] > vmax * 0.6 else '#111',
                    fontweight='bold' if heat[ridx, k] < 0.05 else 'normal')

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig_shd')

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Detection
    # Layout: 3 rows × 2 cols
    #   Row 0: F1 line, Precision line, Recall line
    #   Row 1: Detected CPs vs true, TP/FP/FN stacked bar, Detection F1 violin
    # ══════════════════════════════════════════════════════════════════════════

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle('Change-point Detection Capacity (Pipeline)', fontweight='bold', fontsize=12)

    det_cfg = [
        ('det_f1',        'F1',        '#1b7837'),
        ('det_precision', 'Precision', '#762a83'),
        ('det_recall',    'Recall',    '#d6604d'),
    ]

    # ── row 0: F1, Precision, Recall line + CI ────────────────────────────────
    for ax, (col, label, color) in zip(axes2[0], det_cfg):
        ms = np.array([agg(gv(grp, p, col))[0] for p in P])
        cs = np.array([agg(gv(grp, p, col))[1] for p in P])
        line_ci(ax, P, ms, cs, color, 'o', label)

        # annotate mean values on the line
        for i, (p, m) in enumerate(zip(P, ms)):
            ax.annotate(f'{m:.2f}', xy=(p, m), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=7, color=color)

        ax.set_ylim(0, 1.12)
        tidy(ax, P, ylabel=label, title=f'Detection {label}')

    # ── row 1, col 0: detected CPs vs true ────────────────────────────────────
    ax = axes2[1, 0]
    ms = np.array([agg(gv(grp, p, 'n_detected_cps'))[0] for p in P])
    cs = np.array([agg(gv(grp, p, 'n_detected_cps'))[1] for p in P])
    ax.axhline(4, color='#555', lw=1.4, ls='--', zorder=1, label='True CPs = 4')
    line_ci(ax, P, ms, cs, '#d6604d', 'o', 'Detected CPs (mean ± 95% CI)')
    # shade the gap
    ax.fill_between(P, ms, 4, alpha=0.10, color='grey', label='Missed CPs (gap)')
    for p, m in zip(P, ms):
        ax.annotate(f'{m:.1f}', xy=(p, m), xytext=(0, -12),
                    textcoords='offset points', ha='center', fontsize=7.5, color='#d6604d')
    tidy(ax, P, ylabel='# Change-points', title='Detected vs True CPs')
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=8, loc='lower left')

    # ── row 1, col 1: TP/FP/FN stacked bar ────────────────────────────────────
    ax = axes2[1, 1]
    tp = np.array([gv(grp, p, 'det_tp').mean() for p in P])
    fp = np.array([gv(grp, p, 'det_fp').mean() for p in P])
    fn = np.array([gv(grp, p, 'det_fn').mean() for p in P])
    x  = np.arange(len(P))
    b1 = ax.bar(x, tp, 0.6, label='TP', color='#4dac26', alpha=0.85)
    b2 = ax.bar(x, fp, 0.6, bottom=tp, label='FP', color='#d6604d', alpha=0.85)
    b3 = ax.bar(x, fn, 0.6, bottom=tp + fp, label='FN', color='#aaa', alpha=0.85)

    # value labels inside bars
    for i, (t, f1, f2) in enumerate(zip(tp, fp, fn)):
        if t > 0.2: ax.text(i, t / 2,      f'{t:.1f}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        if f1 > 0.1: ax.text(i, t + f1/2,   f'{f1:.1f}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        if f2 > 0.2: ax.text(i, t+f1+f2/2,  f'{f2:.1f}', ha='center', va='center', fontsize=7, color='#333', fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(P)
    ax.set_xlabel('Number of nodes $p$', fontsize=10)
    ax.set_ylabel('Count (mean per experiment)', fontsize=10)
    ax.set_title('Detection TP / FP / FN breakdown', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── row 1, col 2: violin of detection F1 by p ────────────────────────────
    ax = axes2[1, 2]
    f1_by_p = [gv(grp, p, 'det_f1') for p in P]
    parts   = ax.violinplot(f1_by_p, positions=P, widths=0.5,
                            showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#1b7837'); pc.set_alpha(0.55); pc.set_edgecolor('none')
    parts['cmedians'].set_color('#111'); parts['cmedians'].set_linewidth(1.5)
    q1s = [np.percentile(v, 25) for v in f1_by_p]
    q3s = [np.percentile(v, 75) for v in f1_by_p]
    meds = [np.median(v) for v in f1_by_p]
    ax.vlines(P, q1s, q3s, color='#444', lw=1.2, alpha=0.6, zorder=4)
    ax.scatter(P, meds, s=14, color='#111', zorder=5)
    ax.set_ylim(-0.05, 1.1)
    tidy(ax, P, ylabel='F1', title='Detection F1 — distribution by $p$')

    plt.tight_layout()
    save_fig(fig2, out_dir, 'fig_detection')

    # ══════════════════════════════════════════════════════════════════════════
    # CSV statistics
    # ══════════════════════════════════════════════════════════════════════════

    # ── SHD descriptive + tests ───────────────────────────────────────────────
    shd_rows = []
    for method, (color, marker, cols) in METHODS.items():
        for mkey, col in cols.items():
            for p in P:
                v = gv(grp, p, col)
                row = {'method': method, 'p': p, 'metric': mkey, 'column': col,
                       'n': len(v), 'mean': round(np.mean(v), 5),
                       'std': round(np.std(v, ddof=1), 5), 'ci95': round(ci95(v), 5),
                       'median': round(np.median(v), 5),
                       'q1': round(np.percentile(v, 25), 5),
                       'q3': round(np.percentile(v, 75), 5),
                       'min': round(v.min(), 5), 'max': round(v.max(), 5)}
                # attach test results for Pipeline vs this method (if applicable)
                if method != 'Pipeline' and mkey in SHARED_SHD:
                    row['bonf_p']      = round(corr_p[(method, mkey, p)], 8)
                    row['effect_r']    = round(eff_r[(method, mkey, p)], 4)
                    row['stars']       = sig_star(corr_p[(method, mkey, p)])
                    row['significant'] = corr_p[(method, mkey, p)] < 0.05
                shd_rows.append(row)

    shd_df = pd.DataFrame(shd_rows)
    shd_df.to_csv(os.path.join(out_dir, 'stats_shd.csv'), index=False)
    print(f'  stats_shd.csv ({len(shd_df)} rows)')

    # ── Detection descriptive ─────────────────────────────────────────────────
    det_rows = []
    for col in ('det_f1', 'det_precision', 'det_recall', 'det_tp', 'det_fp', 'det_fn', 'n_detected_cps'):
        for p in P:
            v = gv(grp, p, col)
            det_rows.append({'metric': col, 'p': p, 'n': len(v),
                             'mean': round(np.mean(v), 5), 'std': round(np.std(v, ddof=1), 5),
                             'ci95': round(ci95(v), 5), 'median': round(np.median(v), 5),
                             'q1': round(np.percentile(v, 25), 5),
                             'q3': round(np.percentile(v, 75), 5),
                             'min': round(v.min(), 5), 'max': round(v.max(), 5)})
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(os.path.join(out_dir, 'stats_detection.csv'), index=False)
    print(f'  stats_detection.csv ({len(det_df)} rows)')

    # ── console summary ───────────────────────────────────────────────────────
    print('\n── SHD pooled mean ± 95% CI (all p) ─────────────────────────────────────')
    print(f"{'Method':<15} {'Mean nSHD':>12} {'Mean SHD':>10} {'Cons nSHD':>12}")
    print('─' * 52)
    for method, (_, _, cols) in METHODS.items():
        def pm(mkey):
            col = cols.get(mkey)
            if col is None: return '    —     '
            v = df[col].to_numpy(float)
            return f'{np.mean(v):.3f}±{ci95(v):.3f}'
        print(f'{method:<15} {pm("mean_nshd"):>12} {pm("mean_shd"):>10} {pm("cons_nshd"):>12}')

    print('\n── Detection pooled mean ± 95% CI (all p) ───────────────────────────────')
    for col, lbl in [('det_f1','F1'), ('det_precision','Precision'), ('det_recall','Recall')]:
        v = df[col].to_numpy(float)
        print(f'  {lbl:<12} {np.mean(v):.3f} ± {ci95(v):.3f}  '
              f'[range {v.min():.2f}–{v.max():.2f}]')

    sig_total = sum(1 for k, v in corr_p.items() if v < 0.05)
    print(f'\n  Significant SHD tests (Bonf. p<0.05): {sig_total}/{n_tests}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'batch_ablation.csv'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'plots'))
    args = ap.parse_args()
    print(f'Reading  {args.csv}\nWriting  {args.out}/')
    main(args.csv, args.out)
