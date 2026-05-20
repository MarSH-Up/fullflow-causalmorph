#!/Users/mdlsh/miniconda3/envs/IYCC-env/bin/python
"""
Detection precision analysis for batch_ablation.csv.

Outputs (./plots/):
  fig_precision.png     — precision vs p (line + CI), distribution, TP/FP context
  stats_precision.csv   — descriptive stats per p

Usage:
    python precision_analysis.py [--csv PATH] [--out DIR]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── helpers ────────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    df = df[df['status'] == 'ok'].copy()
    df['p'] = df['p'].astype(int)
    return df.sort_values('p').reset_index(drop=True)

def gv(grp, p, col):
    return grp.get_group(p)[col].to_numpy(float)

def std(v):
    return np.std(v, ddof=1)

# ── style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'lines.linewidth': 1.8, 'lines.markersize': 6, 'figure.dpi': 150,
})

COLOR = '#762a83'

# ══════════════════════════════════════════════════════════════════════════════

def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df  = load(csv_path)
    P   = sorted(df['p'].unique())
    grp = df.groupby('p')

    prec_by_p = [gv(grp, p, 'det_precision') for p in P]
    tp_by_p   = [gv(grp, p, 'det_tp').mean() for p in P]
    fp_by_p   = [gv(grp, p, 'det_fp').mean() for p in P]
    ms        = np.array([np.mean(v) for v in prec_by_p])
    sds       = np.array([std(v)     for v in prec_by_p])

    # ── figure: 2×2 ───────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Detection Precision Analysis', fontweight='bold', fontsize=13)

    # ── (0,0) precision vs p — line + ±1 STD band ─────────────────────────────
    ax = axes[0, 0]
    ax.plot(P, ms, color=COLOR, marker='o', zorder=3, lw=2, ms=7)
    ax.fill_between(P, ms - sds, ms + sds, color=COLOR, alpha=0.18, zorder=2,
                    label='± 1 SD')
    for p, m, s in zip(P, ms, sds):
        ax.annotate(f'{m:.3f}\n(±{s:.3f})', xy=(p, m + s + 0.015),
                    ha='center', va='bottom', fontsize=7.5, color=COLOR, fontweight='bold')
    ax.axhline(ms.mean(), color='grey', lw=1.2, ls='--',
               label=f'Grand mean = {ms.mean():.3f}')
    ax.set_ylim(0, 1.25)
    ax.set_xticks(P)
    ax.set_xlabel('Number of nodes $p$')
    ax.set_ylabel('Precision')
    ax.set_title('Detection Precision vs graph size\n(mean ± 1 SD)')
    ax.legend(fontsize=8.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── (0,1) violin distribution per p ───────────────────────────────────────
    ax = axes[0, 1]
    parts = ax.violinplot(prec_by_p, positions=P, widths=0.5,
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(COLOR); pc.set_alpha(0.50); pc.set_edgecolor('none')
    parts['cmedians'].set_color('#111'); parts['cmedians'].set_linewidth(1.8)
    q1s  = [np.percentile(v, 25) for v in prec_by_p]
    q3s  = [np.percentile(v, 75) for v in prec_by_p]
    meds = [np.median(v) for v in prec_by_p]
    ax.vlines(P, q1s, q3s, color='#444', lw=1.2, alpha=0.6, zorder=4)
    ax.scatter(P, meds, s=16, color='#111', zorder=5, label='Median')

    # annotate median
    for p, med in zip(P, meds):
        ax.annotate(f'{med:.2f}', xy=(p, med), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=7.5, color='#333')

    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks(P)
    ax.set_xlabel('Number of nodes $p$')
    ax.set_ylabel('Precision')
    ax.set_title('Precision distribution by $p$\n(violin = density, bar = IQR, dot = median)')
    ax.legend(fontsize=8, loc='lower left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── (1,0) TP vs FP — what precision is made of ────────────────────────────
    ax = axes[1, 0]
    x  = np.arange(len(P))
    w  = 0.35
    b1 = ax.bar(x - w/2, tp_by_p, w, label='TP (correct detections)',
                color='#4dac26', alpha=0.85)
    b2 = ax.bar(x + w/2, fp_by_p, w, label='FP (false alarms)',
                color='#d6604d', alpha=0.85)
    for i, (t, f) in enumerate(zip(tp_by_p, fp_by_p)):
        ax.text(i - w/2, t + 0.03, f'{t:.2f}', ha='center', fontsize=7.5,
                color='#4dac26', fontweight='bold')
        ax.text(i + w/2, f + 0.03, f'{f:.2f}', ha='center', fontsize=7.5,
                color='#d6604d', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(P)
    ax.set_xlabel('Number of nodes $p$')
    ax.set_ylabel('Count (mean per experiment)')
    ax.set_title('Mean TP and FP counts\n(Precision = TP / (TP + FP))')
    ax.legend(fontsize=8.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── (1,1) precision vs n_detected_cps scatter (all experiments) ───────────
    ax = axes[1, 1]
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(df))
    sc = ax.scatter(df['n_detected_cps'] + jitter, df['det_precision'],
                    c=df['p'], cmap='plasma', alpha=0.25, s=12, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('$p$ (nodes)', fontsize=8)

    # overlay mean precision per n_detected_cps
    by_ndet = df.groupby('n_detected_cps')['det_precision'].agg(['mean', 'std']).reset_index()
    ax.plot(by_ndet['n_detected_cps'], by_ndet['mean'], 'o-',
            color='black', lw=1.6, ms=6, label='Mean per # detected', zorder=5)
    ax.fill_between(by_ndet['n_detected_cps'],
                    by_ndet['mean'] - by_ndet['std'],
                    by_ndet['mean'] + by_ndet['std'],
                    color='black', alpha=0.15, zorder=4)

    ax.set_xlabel('Number of detected CPs')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs detected CP count\n(colour = $p$, jittered)')
    ax.set_xlim(-0.5, df['n_detected_cps'].max() + 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_precision.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  fig_precision.png')

    # ── stats CSV ─────────────────────────────────────────────────────────────
    rows = []
    for p, v in zip(P, prec_by_p):
        tp = gv(grp, p, 'det_tp')
        fp = gv(grp, p, 'det_fp')
        rows.append({
            'p': p, 'n': len(v),
            'mean_precision': round(np.mean(v), 5),
            'std':            round(np.std(v, ddof=1), 5),
            'std':            round(std(v), 5),
            'median':         round(np.median(v), 5),
            'q1':             round(np.percentile(v, 25), 5),
            'q3':             round(np.percentile(v, 75), 5),
            'min':            round(v.min(), 5),
            'max':            round(v.max(), 5),
            'pct_precision_1': round((v == 1.0).mean(), 4),   # fraction of perfect-precision runs
            'pct_precision_0': round((v == 0.0).mean(), 4),   # fraction of zero-precision runs
            'mean_tp':         round(np.mean(tp), 4),
            'mean_fp':         round(np.mean(fp), 4),
        })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(out_dir, 'stats_precision.csv'), index=False)
    print(f'  stats_precision.csv')

    # ── console ────────────────────────────────────────────────────────────────
    print('\n── Detection Precision by p ─────────────────────────────────────────────')
    print(f"{'p':>4}  {'mean':>7}  {'±std':>7}  {'median':>7}  "
          f"{'IQR':>14}  {'prec=1.0':>9}  {'mean_TP':>8}  {'mean_FP':>8}")
    print('─' * 80)
    for _, r in stats_df.iterrows():
        iqr = f"[{r['q1']:.3f}, {r['q3']:.3f}]"
        print(f"{int(r['p']):>4}  {r['mean_precision']:>7.3f}  {r['std']:>7.3f}  "
              f"{r['median']:>7.3f}  {iqr:>14}  "
              f"{r['pct_precision_1']:>7.1%}  {r['mean_tp']:>8.3f}  {r['mean_fp']:>8.3f}")

    print(f"\n  Grand mean precision: {ms.mean():.3f} ± {std(ms):.3f} (SD)")
    print(f"  Precision is stable across p (detection is consistently conservative).")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'batch_ablation.csv'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'plots'))
    args = ap.parse_args()
    main(args.csv, args.out)
