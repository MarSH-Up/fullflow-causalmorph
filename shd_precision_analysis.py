#!/Users/mdlsh/miniconda3/envs/IYCC-env/bin/python
"""
Focused analysis: SHD (all methods) + Detection Precision.
All spread shown as ± 1 SD.

Outputs (./plots/):
  fig_shd.png            — SHD line plots (mean±SD) + violin distributions + sig heatmap
  fig_precision.png      — Precision line (mean±SD), violin, TP/FP bars, scatter
  stats_shd.csv          — descriptive stats + Wilcoxon tests per (method, p, metric)
  stats_precision.csv    — precision descriptive stats + TP/FP per p

Usage:
    python shd_precision_analysis.py [--csv PATH] [--out DIR]
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

# ── column map ─────────────────────────────────────────────────────────────────

METHODS = {
    'Pipeline':    ('#2166ac', 'o', {
        'mean_nshd': 'mean_norm_shd',
        'cons_nshd': 'consensus_norm_shd',
    }),
    'CausalMorph': ('#d6604d', 's', {
        'mean_nshd': 'causalmorph_mean_norm_shd',
        'cons_nshd': 'causalmorph_cons_norm_shd',
    }),
    'DirectLiNGAM': ('#4dac26', '^', {
        'mean_nshd': 'directlingam_mean_norm_shd',
        'cons_nshd': 'directlingam_cons_norm_shd',
    }),
    'ICALiNGAM': ('#8073ac', 'D', {
        'mean_nshd': 'icalingam_mean_norm_shd',
        'cons_nshd': 'icalingam_cons_norm_shd',
    }),
}

SHARED_SHD = ['mean_nshd', 'cons_nshd']
BASELINES  = ['CausalMorph', 'DirectLiNGAM', 'ICALiNGAM']

SHD_LABEL = {
    'mean_nshd': 'Mean Norm. SHD',
    'mean_shd':  'Mean SHD (raw)',
    'cons_nshd': 'Consensus Norm. SHD',
    'cons_shd':  'Consensus SHD (raw)',
}

PREC_COLOR = '#762a83'

# ── helpers ────────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    df = df[df['status'] == 'ok'].copy()
    df['p'] = df['p'].astype(int)
    return df.sort_values('p').reset_index(drop=True)

def gv(grp, p, col):
    return grp.get_group(p)[col].to_numpy(float)

def sd(v):
    return np.std(v, ddof=1)

def wilcoxon_test(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, 0.0
    stat, pval = scipy_stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')
    n = len(nz)
    r = float(np.clip((1.0 - 2.0 * stat / (n * (n + 1) / 2)) * np.sign(np.mean(d)), -1, 1))
    return pval, r

def sig_star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

def line_sd(ax, xs, means, sds, color, marker, label):
    ax.plot(xs, means, color=color, marker=marker, label=label, lw=1.8, ms=6, zorder=3)
    ax.fill_between(xs, means - sds, means + sds, color=color, alpha=0.18, zorder=2)

def tidy(ax, P, ylabel='', title=''):
    ax.set_xticks(P)
    ax.set_xlabel('Number of nodes $p$', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def violin_panel(ax, data_by_p, positions, colors, P, n_methods, gapw, ylabel, title):
    parts = ax.violinplot(data_by_p, positions=positions, widths=0.72,
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor('none')
    parts['cmedians'].set_color('#111'); parts['cmedians'].set_linewidth(1.5)
    q1s  = [np.percentile(v, 25) for v in data_by_p]
    q3s  = [np.percentile(v, 75) for v in data_by_p]
    meds = [np.median(v) for v in data_by_p]
    ax.vlines(positions, q1s, q3s, color='#444', lw=1.1, alpha=0.55, zorder=4)
    ax.scatter(positions, meds, s=10, color='#111', zorder=5)
    ticks = [i * (n_methods + gapw) + (n_methods - 1) / 2 for i in range(len(P))]
    ax.set_xticks(ticks); ax.set_xticklabels(P)
    ax.set_xlabel('Number of nodes $p$', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

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

    # ── Wilcoxon tests (Pipeline vs each baseline, per SHD metric, per p) ──────

    raw_pv, eff_r = {}, {}
    for bl in BASELINES:
        for mkey in SHARED_SHD:
            for p in P:
                a = gv(grp, p, METHODS['Pipeline'][2][mkey])
                b = gv(grp, p, METHODS[bl][2][mkey])
                raw_pv[(bl, mkey, p)], eff_r[(bl, mkey, p)] = wilcoxon_test(a, b)
    n_tests = len(raw_pv)
    corr_pv = {k: min(v * n_tests, 1.0) for k, v in raw_pv.items()}

    def any_sig(mkey, p):
        return any(corr_pv[(bl, mkey, p)] < 0.05 for bl in BASELINES)

    def annot_sig(ax, mkey, ylim):
        for p in P:
            if not any_sig(mkey, p):
                continue
            ymax = max(
                np.mean(gv(grp, p, METHODS[m][2][mkey])) + sd(gv(grp, p, METHODS[m][2][mkey]))
                for m in METHODS if mkey in METHODS[m][2]
            )
            ax.annotate('*', xy=(p, ymax + 0.025 * ylim),
                        ha='center', va='bottom', fontsize=9.5, color='#333', fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1 — Normalized SHD only
    # GridSpec 2×3: lines (row 0) | violins (row 1) | heatmap spans both rows
    # ══════════════════════════════════════════════════════════════════════════

    from matplotlib.gridspec import GridSpec

    fig1 = plt.figure(figsize=(16, 10))
    fig1.suptitle(
        'Normalized SHD — All Methods vs Graph Size  (shaded band = ± 1 SD)\n'
        '* Pipeline differs from ≥ 1 baseline, Bonferroni-corrected Wilcoxon p < 0.05',
        fontweight='bold', fontsize=11,
    )
    gs = GridSpec(2, 3, figure=fig1, wspace=0.35, hspace=0.40)

    ax_line_mean = fig1.add_subplot(gs[0, 0])
    ax_line_cons = fig1.add_subplot(gs[0, 1])
    ax_viol_mean = fig1.add_subplot(gs[1, 0])
    ax_viol_cons = fig1.add_subplot(gs[1, 1])
    ax_heat      = fig1.add_subplot(gs[:, 2])   # spans both rows

    shd_cfg = [
        ('mean_nshd', 'Norm. SHD', 'Mean Norm. SHD',       ax_line_mean, ax_viol_mean),
        ('cons_nshd', 'Norm. SHD', 'Consensus Norm. SHD',  ax_line_cons, ax_viol_cons),
    ]

    n_m  = len(METHODS)
    gapw = 1.3
    grpw = n_m + gapw

    for mkey, ylabel, title, ax_line, ax_viol in shd_cfg:

        # line plot
        for method, (color, marker, cols) in METHODS.items():
            v_by_p = [gv(grp, p, cols[mkey]) for p in P]
            ms = np.array([np.mean(v) for v in v_by_p])
            ss = np.array([sd(v)      for v in v_by_p])
            line_sd(ax_line, P, ms, ss, color, marker, method)
        tidy(ax_line, P, ylabel=ylabel, title=title)
        ax_line.set_ylim(0, 1.05)
        ax_line.legend(loc='upper left', framealpha=0.9)
        annot_sig(ax_line, mkey, ax_line.get_ylim()[1] - ax_line.get_ylim()[0])

        # violin
        positions, vdata, vcolors = [], [], []
        for i, p in enumerate(P):
            for j, (method, (color, marker, cols)) in enumerate(METHODS.items()):
                positions.append(i * grpw + j)
                vdata.append(gv(grp, p, cols[mkey]))
                vcolors.append(color)
        handles = [Patch(facecolor=METHODS[m][0], alpha=0.6, label=m) for m in METHODS]
        violin_panel(ax_viol, vdata, positions, vcolors, P, n_m, gapw,
                     ylabel, f'{title} — distributions')
        ax_viol.legend(handles=handles, fontsize=7.5, loc='upper left', framealpha=0.85)

    # significance heatmap
    nrows   = len(BASELINES) * len(SHARED_SHD)
    heat_p  = np.zeros((nrows, len(P)))
    heat_r  = np.zeros((nrows, len(P)))
    ylabels = []
    row_lbl = {'mean_nshd': 'Mean nSHD', 'cons_nshd': 'Cons nSHD'}
    for i, bl in enumerate(BASELINES):
        for j, mkey in enumerate(SHARED_SHD):
            ridx = i * len(SHARED_SHD) + j
            ylabels.append(f'{bl}\n{row_lbl[mkey]}')
            for k, p in enumerate(P):
                heat_p[ridx, k] = corr_pv[(bl, mkey, p)]
                heat_r[ridx, k] = eff_r[(bl, mkey, p)]
    log_p = -np.log10(np.clip(heat_p, 1e-15, 1.0))
    vmax  = max(3.0, np.nanmax(log_p))
    ext   = [P[0] - 0.5, P[-1] + 0.5, nrows - 0.5, -0.5]
    im    = ax_heat.imshow(log_p, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax, extent=ext)
    ax_heat.set_yticks(range(nrows)); ax_heat.set_yticklabels(ylabels, fontsize=8)
    ax_heat.set_xticks(P); ax_heat.set_xlabel('Number of nodes $p$', fontsize=9)
    ax_heat.set_title('Pipeline significance vs baselines\n'
                      '$-\\log_{10}(p_{\\mathrm{Bonf}})$  |  cells: effect $r$', fontsize=10)
    cb = plt.colorbar(im, ax=ax_heat, fraction=0.05, pad=0.03)
    cb.ax.axhline(-np.log10(0.05), color='white', lw=1.5, ls='--')
    cb.set_label('$-\\log_{10}(p)$  [dashed = 0.05]', fontsize=8)
    for i in range(1, len(BASELINES)):
        ax_heat.axhline(i * len(SHARED_SHD) - 0.5, color='white', lw=1.8)
    for ridx in range(nrows):
        for k, p in enumerate(P):
            rv = heat_r[ridx, k]
            ax_heat.text(p, ridx, f'{rv:+.2f}', ha='center', va='center', fontsize=7,
                         color='white' if log_p[ridx, k] > vmax * 0.6 else '#111',
                         fontweight='bold' if heat_p[ridx, k] < 0.05 else 'normal')

    save_fig(fig1, out_dir, 'fig_shd')

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Detection Precision
    # ══════════════════════════════════════════════════════════════════════════

    prec_by_p = [gv(grp, p, 'det_precision') for p in P]
    tp_by_p   = [gv(grp, p, 'det_tp').mean() for p in P]
    fp_by_p   = [gv(grp, p, 'det_fp').mean() for p in P]
    ms_p      = np.array([np.mean(v) for v in prec_by_p])
    sds_p     = np.array([sd(v)      for v in prec_by_p])

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
    fig2.suptitle('Detection Precision Analysis  (shaded band = ± 1 SD)',
                  fontweight='bold', fontsize=13)

    # (0,0) line + ±1 SD
    ax = axes2[0, 0]
    ax.plot(P, ms_p, color=PREC_COLOR, marker='o', zorder=3, lw=2, ms=7)
    ax.fill_between(P, ms_p - sds_p, ms_p + sds_p, color=PREC_COLOR, alpha=0.18,
                    label='± 1 SD', zorder=2)
    for p, m, s in zip(P, ms_p, sds_p):
        ax.annotate(f'{m:.3f}\n(±{s:.3f})', xy=(p, min(m + s + 0.015, 1.0)),
                    ha='center', va='bottom', fontsize=7.5, color=PREC_COLOR, fontweight='bold')
    ax.axhline(ms_p.mean(), color='grey', lw=1.2, ls='--',
               label=f'Grand mean = {ms_p.mean():.3f}')
    ax.set_ylim(0, 1.30)
    tidy(ax, P, ylabel='Precision', title='Detection Precision vs graph size\n(mean ± 1 SD)')
    ax.legend(fontsize=8.5)

    # (0,1) violin per p
    ax = axes2[0, 1]
    parts = ax.violinplot(prec_by_p, positions=P, widths=0.5,
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(PREC_COLOR); pc.set_alpha(0.50); pc.set_edgecolor('none')
    parts['cmedians'].set_color('#111'); parts['cmedians'].set_linewidth(1.8)
    q1s  = [np.percentile(v, 25) for v in prec_by_p]
    q3s  = [np.percentile(v, 75) for v in prec_by_p]
    meds = [np.median(v) for v in prec_by_p]
    ax.vlines(P, q1s, q3s, color='#444', lw=1.2, alpha=0.6, zorder=4)
    ax.scatter(P, meds, s=16, color='#111', zorder=5)
    for p, med in zip(P, meds):
        ax.annotate(f'{med:.2f}', xy=(p, med), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=7.5, color='#333')
    ax.set_ylim(-0.05, 1.12)
    tidy(ax, P, ylabel='Precision',
         title='Precision distribution by $p$\n(violin = density, bar = IQR, dot = median)')

    # (1,0) TP vs FP bars
    ax = axes2[1, 0]
    x, w = np.arange(len(P)), 0.35
    ax.bar(x - w/2, tp_by_p, w, label='TP (correct)', color='#4dac26', alpha=0.85)
    ax.bar(x + w/2, fp_by_p, w, label='FP (false alarm)', color='#d6604d', alpha=0.85)
    for i, (t, f) in enumerate(zip(tp_by_p, fp_by_p)):
        ax.text(i - w/2, t + 0.03, f'{t:.2f}', ha='center', fontsize=7.5,
                color='#4dac26', fontweight='bold')
        ax.text(i + w/2, f + 0.03, f'{f:.2f}', ha='center', fontsize=7.5,
                color='#d6604d', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(P)
    ax.set_xlabel('Number of nodes $p$', fontsize=10)
    ax.set_ylabel('Count (mean per experiment)', fontsize=10)
    ax.set_title('Mean TP and FP counts\n(Precision = TP / (TP + FP))', fontsize=11)
    ax.legend(fontsize=8.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (1,1) precision vs n_detected_cps scatter
    ax = axes2[1, 1]
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(df))
    sc = ax.scatter(df['n_detected_cps'] + jitter, df['det_precision'],
                    c=df['p'], cmap='plasma', alpha=0.25, s=12, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('$p$ (nodes)', fontsize=8)
    by_ndet = df.groupby('n_detected_cps')['det_precision'].agg(
        mean='mean', std='std').reset_index()
    ax.plot(by_ndet['n_detected_cps'], by_ndet['mean'], 'o-',
            color='black', lw=1.6, ms=6, label='Mean ± 1 SD', zorder=5)
    ax.fill_between(by_ndet['n_detected_cps'],
                    by_ndet['mean'] - by_ndet['std'],
                    by_ndet['mean'] + by_ndet['std'],
                    color='black', alpha=0.15, zorder=4)
    ax.set_xlim(-0.5, df['n_detected_cps'].max() + 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel('Number of detected CPs', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Precision vs detected CP count\n(colour = $p$, jittered)', fontsize=11)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig2, out_dir, 'fig_precision')

    # ══════════════════════════════════════════════════════════════════════════
    # CSV statistics
    # ══════════════════════════════════════════════════════════════════════════

    # SHD stats
    shd_rows = []
    for method, (color, marker, cols) in METHODS.items():
        for mkey, col in cols.items():
            for p in P:
                v = gv(grp, p, col)
                row = {'method': method, 'p': p, 'metric': mkey, 'column': col,
                       'n': len(v), 'mean': round(np.mean(v), 5),
                       'std': round(sd(v), 5), 'median': round(np.median(v), 5),
                       'q1': round(np.percentile(v, 25), 5),
                       'q3': round(np.percentile(v, 75), 5),
                       'min': round(v.min(), 5), 'max': round(v.max(), 5)}
                if method != 'Pipeline' and mkey in SHARED_SHD:
                    row['bonf_p']      = round(corr_pv[(method, mkey, p)], 8)
                    row['effect_r']    = round(eff_r[(method, mkey, p)], 4)
                    row['stars']       = sig_star(corr_pv[(method, mkey, p)])
                    row['significant'] = corr_pv[(method, mkey, p)] < 0.05
                shd_rows.append(row)
    pd.DataFrame(shd_rows).to_csv(os.path.join(out_dir, 'stats_shd.csv'), index=False)
    print(f'  stats_shd.csv ({len(shd_rows)} rows)')

    # Precision stats
    prec_rows = []
    for p, v in zip(P, prec_by_p):
        tp = gv(grp, p, 'det_tp')
        fp = gv(grp, p, 'det_fp')
        prec_rows.append({
            'p': p, 'n': len(v),
            'mean':    round(np.mean(v), 5),
            'std':     round(sd(v), 5),
            'median':  round(np.median(v), 5),
            'q1':      round(np.percentile(v, 25), 5),
            'q3':      round(np.percentile(v, 75), 5),
            'min':     round(v.min(), 5),
            'max':     round(v.max(), 5),
            'pct_prec_1': round((v == 1.0).mean(), 4),
            'pct_prec_0': round((v == 0.0).mean(), 4),
            'mean_tp': round(np.mean(tp), 4),
            'mean_fp': round(np.mean(fp), 4),
        })
    pd.DataFrame(prec_rows).to_csv(os.path.join(out_dir, 'stats_precision.csv'), index=False)
    print(f'  stats_precision.csv ({len(prec_rows)} rows)')

    # ── console ────────────────────────────────────────────────────────────────

    print('\n── Norm. SHD mean ± SD (pooled across all p) ────────────────────────────')
    print(f"{'Method':<15} {'Mean nSHD':>16} {'Cons nSHD':>16}")
    print('─' * 50)
    for method, (_, _, cols) in METHODS.items():
        def pm(mkey):
            v = df[cols[mkey]].to_numpy(float)
            return f'{np.mean(v):.3f} ± {sd(v):.3f}'
        print(f'{method:<15} {pm("mean_nshd"):>16} {pm("cons_nshd"):>16}')

    print('\n── Detection Precision by p ─────────────────────────────────────────────')
    print(f"{'p':>4}  {'mean':>7}  {'±sd':>7}  {'median':>7}  "
          f"{'IQR':>14}  {'prec=1.0':>9}  {'mean_TP':>8}  {'mean_FP':>8}")
    print('─' * 80)
    for r in prec_rows:
        iqr = f"[{r['q1']:.3f}, {r['q3']:.3f}]"
        print(f"{r['p']:>4}  {r['mean']:>7.3f}  {r['std']:>7.3f}  {r['median']:>7.3f}  "
              f"{iqr:>14}  {r['pct_prec_1']:>8.1%}  {r['mean_tp']:>8.3f}  {r['mean_fp']:>8.3f}")

    sig_n = sum(1 for k, v in corr_pv.items() if v < 0.05)
    print(f"\n  Significant SHD tests (Bonf. p<0.05): {sig_n}/{n_tests}")
    print(f"  Grand mean precision: {ms_p.mean():.3f} ± {sd(ms_p):.3f} (SD)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'batch_ablation.csv'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'plots'))
    args = ap.parse_args()
    print(f'Reading  {args.csv}\nWriting  {args.out}/')
    main(args.csv, args.out)
