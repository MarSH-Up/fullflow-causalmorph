#!/Users/mdlsh/miniconda3/envs/IYCC-env/bin/python
"""
Formal ablation comparison for batch_ablation.csv.

Methods compared (per-regime mean and consensus):
  Pipeline  — NSD_Wavelets changepoint detection + CausalMorph + Bayesian aggregation
              (columns: mean_struct_*, consensus_*)
  CausalMorph   — standalone (causalmorph_*)
  DirectLiNGAM  — standalone (directlingam_*)
  ICALiNGAM     — standalone (icalingam_*)

Ablation axis: p (number of nodes), values 3–10, 100 seeds each.

Outputs (in ./plots/):
  ablation_comparison.{pdf,png}    — line plots + significance heatmap
  ablation_distributions.{pdf,png} — violin distributions per method × p
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────────

CSV_PATH = os.path.join(os.path.dirname(__file__), 'batch_ablation.csv')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ── method definitions ─────────────────────────────────────────────────────────

# Each method maps metric keys → CSV column names
METHODS = {
    'Pipeline':    {
        'mean_f1':  'mean_struct_f1',
        'mean_shd': 'mean_norm_shd',
        'cons_f1':  'consensus_f1',
        'cons_shd': 'consensus_norm_shd',
    },
    'CausalMorph': {
        'mean_f1':  'causalmorph_mean_f1',
        'mean_shd': 'causalmorph_mean_norm_shd',
        'cons_f1':  'causalmorph_cons_f1',
        'cons_shd': 'causalmorph_cons_norm_shd',
    },
    'DirectLiNGAM': {
        'mean_f1':  'directlingam_mean_f1',
        'mean_shd': 'directlingam_mean_norm_shd',
        'cons_f1':  'directlingam_cons_f1',
        'cons_shd': 'directlingam_cons_norm_shd',
    },
    'ICALiNGAM': {
        'mean_f1':  'icalingam_mean_f1',
        'mean_shd': 'icalingam_mean_norm_shd',
        'cons_f1':  'icalingam_cons_f1',
        'cons_shd': 'icalingam_cons_norm_shd',
    },
}

BASELINES = ['CausalMorph', 'DirectLiNGAM', 'ICALiNGAM']
METRICS   = ['mean_f1', 'mean_shd', 'cons_f1', 'cons_shd']
METRIC_LABELS = {
    'mean_f1':  'Mean F1',
    'mean_shd': 'Mean nSHD',
    'cons_f1':  'Cons F1',
    'cons_shd': 'Cons nSHD',
}

COLORS = {
    'Pipeline':    '#2166ac',
    'CausalMorph': '#d6604d',
    'DirectLiNGAM': '#4dac26',
    'ICALiNGAM':   '#8073ac',
}
MARKERS = {
    'Pipeline':    'o',
    'CausalMorph': 's',
    'DirectLiNGAM': '^',
    'ICALiNGAM':   'D',
}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_data(path):
    """Returns dict: p -> list of row dicts (ok rows only)."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['status'] != 'ok':
                continue
            p = int(row['p'])
            data.setdefault(p, []).append(row)
    return data


def vals(rows, col):
    return np.array([float(r[col]) for r in rows])


def mean_ci(arr):
    m  = np.mean(arr)
    ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))
    return m, ci


def wilcoxon_p(a, b):
    d = a - b
    if np.all(d == 0):
        return 1.0
    _, p = stats.wilcoxon(a, b, alternative='two-sided')
    return p


def sig_star(p):
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return 'ns'


# ── load + precompute ──────────────────────────────────────────────────────────

data   = load_data(CSV_PATH)
p_vals = sorted(data.keys())
n_p    = len(p_vals)

# (method, metric, p) -> (mean, ci95)
stats_cache = {}
for method, cols in METHODS.items():
    for mkey, col in cols.items():
        for p in p_vals:
            stats_cache[(method, mkey, p)] = mean_ci(vals(data[p], col))

# detection (pipeline only — single series)
det_cache = {}
for col in ('det_f1', 'det_precision', 'det_recall'):
    det_cache[col] = {p: mean_ci(vals(data[p], col)) for p in p_vals}

# Wilcoxon paired tests: Pipeline vs each baseline, per metric, per p
# Bonferroni correction over n_tests = n_baselines × n_metrics × n_p
raw_pvals = {}
for baseline in BASELINES:
    for mkey in METRICS:
        for p in p_vals:
            pipe = vals(data[p], METHODS['Pipeline'][mkey])
            base = vals(data[p], METHODS[baseline][mkey])
            raw_pvals[(baseline, mkey, p)] = wilcoxon_p(pipe, base)

n_tests   = len(raw_pvals)
corrected = {k: min(v * n_tests, 1.0) for k, v in raw_pvals.items()}

# ── style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'legend.fontsize':   9,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'lines.linewidth':   1.8,
    'lines.markersize':  6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
})

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — line plots + significance heatmap
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(3, 2, figsize=(13, 15))
fig1.suptitle(
    'Ablation Study — Structure Recovery vs Graph Size\n'
    '(n = 100 seeds, Bonferroni-corrected Wilcoxon tests, * p<0.05, ** p<0.01, *** p<0.001)',
    fontsize=12, fontweight='bold', y=0.995,
)


def line_panel(ax, mkey, title, ylabel, higher_better):
    """Line plot with 95% CI shading for all four methods."""
    ax.set_title(title)
    ax.set_xlabel('Number of nodes $p$')
    ax.set_ylabel(ylabel)
    ax.set_xticks(p_vals)

    for method in METHODS:
        ms = np.array([stats_cache[(method, mkey, p)][0] for p in p_vals])
        cs = np.array([stats_cache[(method, mkey, p)][1] for p in p_vals])
        ax.plot(p_vals, ms, color=COLORS[method], marker=MARKERS[method],
                label=method, zorder=3)
        ax.fill_between(p_vals, ms - cs, ms + cs,
                        color=COLORS[method], alpha=0.15, zorder=2)

    ax.legend(loc='best', framealpha=0.9)

    # significance annotation: mark p values where Pipeline differs from
    # at least one baseline (Bonferroni-corrected)
    ylims = ax.get_ylim()
    y_range = ylims[1] - ylims[0]
    for p in p_vals:
        sig_level = 1.0
        for baseline in BASELINES:
            sig_level = min(sig_level, corrected[(baseline, mkey, p)])
        star = sig_star(sig_level)
        if star == 'ns':
            continue
        ymax_data = max(
            stats_cache[(m, mkey, p)][0] + stats_cache[(m, mkey, p)][1]
            for m in METHODS
        )
        ax.annotate(
            star,
            xy=(p, ymax_data + 0.03 * y_range),
            ha='center', va='bottom', fontsize=8, color='#333333',
        )


line_panel(axes[0, 0], 'mean_f1',  'Per-regime Struct. F1 (mean)',  'F1',        higher_better=True)
line_panel(axes[0, 1], 'mean_shd', 'Per-regime Norm. SHD (mean)',   'Norm. SHD', higher_better=False)
line_panel(axes[1, 0], 'cons_f1',  'Consensus Struct. F1',          'F1',        higher_better=True)
line_panel(axes[1, 1], 'cons_shd', 'Consensus Norm. SHD',           'Norm. SHD', higher_better=False)

# ── detection panel ────────────────────────────────────────────────────────────

ax_det = axes[2, 0]
det_colors = {'det_f1': '#1b7837', 'det_precision': '#762a83', 'det_recall': '#d6604d'}
det_names  = {'det_f1': 'F1',       'det_precision': 'Precision', 'det_recall': 'Recall'}
for col, color in det_colors.items():
    ms = np.array([det_cache[col][p][0] for p in p_vals])
    cs = np.array([det_cache[col][p][1] for p in p_vals])
    ax_det.plot(p_vals, ms, marker='o', color=color, label=det_names[col])
    ax_det.fill_between(p_vals, ms - cs, ms + cs, alpha=0.15, color=color)
ax_det.set_title('Change-point Detection (Pipeline)')
ax_det.set_xlabel('Number of nodes $p$')
ax_det.set_ylabel('Score')
ax_det.set_xticks(p_vals)
ax_det.set_ylim(0, 1.05)
ax_det.legend(loc='best', framealpha=0.9)

# ── significance heatmap ───────────────────────────────────────────────────────

ax_heat = axes[2, 1]

# rows: baseline × metric; cols: p values
heat = np.full((len(BASELINES) * len(METRICS), n_p), np.nan)
ytick_labels = []
for i, baseline in enumerate(BASELINES):
    for j, mkey in enumerate(METRICS):
        row_idx = i * len(METRICS) + j
        ytick_labels.append(f'{baseline} / {METRIC_LABELS[mkey]}')
        for k, p in enumerate(p_vals):
            heat[row_idx, k] = corrected[(baseline, mkey, p)]

# use -log10 so significant cells pop out
log_heat = -np.log10(np.clip(heat, 1e-10, 1.0))
vmax = max(3.0, np.nanmax(log_heat))

im = ax_heat.imshow(
    log_heat, aspect='auto', cmap='YlOrRd',
    vmin=0, vmax=vmax,
    extent=[p_vals[0] - 0.5, p_vals[-1] + 0.5,
            len(ytick_labels) - 0.5, -0.5],
)
ax_heat.set_yticks(range(len(ytick_labels)))
ax_heat.set_yticklabels(ytick_labels, fontsize=8)
ax_heat.set_xticks(p_vals)
ax_heat.set_xlabel('Number of nodes $p$')
ax_heat.set_title('Pipeline vs Baselines\n($-\\log_{10}$ Bonferroni-corrected $p$)')

# dashed line at significance threshold (-log10(0.05) ≈ 1.3)
cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cb.ax.axhline(y=-np.log10(0.05),  color='grey',  lw=1, ls='--', label='0.05')
cb.ax.axhline(y=-np.log10(0.01),  color='black', lw=1, ls='--', label='0.01')
cb.set_label('$-\\log_{10}(p)$', fontsize=9)

for i in range(heat.shape[0]):
    for k, p in enumerate(p_vals):
        cp = heat[i, k]
        star = sig_star(cp)
        if star == 'ns':
            continue
        ax_heat.text(p, i, star, ha='center', va='center', fontsize=7,
                     color='black' if log_heat[i, k] < vmax * 0.6 else 'white',
                     fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.993])
fig1.savefig(os.path.join(OUT_DIR, 'ablation_comparison.pdf'), bbox_inches='tight')
fig1.savefig(os.path.join(OUT_DIR, 'ablation_comparison.png'), dpi=200, bbox_inches='tight')
print(f"[1/2] Saved ablation_comparison to {OUT_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — violin distribution plots
# ══════════════════════════════════════════════════════════════════════════════

from matplotlib.patches import Patch

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(
    'Score Distributions by Method and Graph Size',
    fontsize=13, fontweight='bold',
)


def violin_panel(ax, mkey, title, ylabel):
    method_list = list(METHODS.keys())
    n_methods   = len(method_list)
    group_width = n_methods + 1.2      # spacing between p-groups

    positions, violin_data, colors_list = [], [], []
    for i, p in enumerate(p_vals):
        rows = data[p]
        for j, method in enumerate(method_list):
            v = vals(rows, METHODS[method][mkey])
            positions.append(i * group_width + j)
            violin_data.append(v)
            colors_list.append(COLORS[method])

    parts = ax.violinplot(violin_data, positions=positions,
                          widths=0.75, showmedians=True, showextrema=False)
    for pc, color in zip(parts['bodies'], colors_list):
        pc.set_facecolor(color)
        pc.set_edgecolor('none')
        pc.set_alpha(0.60)
    parts['cmedians'].set_color('#111111')
    parts['cmedians'].set_linewidth(1.5)

    # thin IQR box
    q1_q3 = [(np.percentile(v, 25), np.percentile(v, 75)) for v in violin_data]
    meds  = [np.median(v) for v in violin_data]
    ax.vlines(positions, [q[0] for q in q1_q3], [q[1] for q in q1_q3],
              color='black', linewidth=1.2, alpha=0.5, zorder=4)
    ax.scatter(positions, meds, color='black', s=12, zorder=5)

    # x-tick at center of each p-group
    tick_pos = [i * group_width + (n_methods - 1) / 2 for i in range(n_p)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(p) for p in p_vals])
    ax.set_xlabel('Number of nodes $p$')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    handles = [Patch(facecolor=COLORS[m], alpha=0.65, label=m) for m in method_list]
    ax.legend(handles=handles, loc='best', fontsize=9, framealpha=0.9)


violin_panel(axes2[0, 0], 'mean_f1',  'Per-regime Struct. F1',  'F1')
violin_panel(axes2[0, 1], 'mean_shd', 'Per-regime Norm. SHD',   'Norm. SHD')
violin_panel(axes2[1, 0], 'cons_f1',  'Consensus F1',           'F1')
violin_panel(axes2[1, 1], 'cons_shd', 'Consensus Norm. SHD',    'Norm. SHD')

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'ablation_distributions.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(OUT_DIR, 'ablation_distributions.png'), dpi=200, bbox_inches='tight')
print(f"[2/2] Saved ablation_distributions to {OUT_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Mean ± 95% CI (pooled across all p) ───────────────────────────────────')
header = f"{'Method':<15}  {'mean F1':>10}  {'mean nSHD':>10}  {'cons F1':>10}  {'cons nSHD':>10}"
print(header)
print('─' * len(header))
for method in METHODS:
    all_mf1  = np.concatenate([vals(data[p], METHODS[method]['mean_f1'])  for p in p_vals])
    all_mshd = np.concatenate([vals(data[p], METHODS[method]['mean_shd']) for p in p_vals])
    all_cf1  = np.concatenate([vals(data[p], METHODS[method]['cons_f1'])  for p in p_vals])
    all_cshd = np.concatenate([vals(data[p], METHODS[method]['cons_shd']) for p in p_vals])
    m1, c1   = mean_ci(all_mf1)
    m2, c2   = mean_ci(all_mshd)
    m3, c3   = mean_ci(all_cf1)
    m4, c4   = mean_ci(all_cshd)
    print(f'{method:<15}  {m1:.3f}±{c1:.3f}  {m2:.3f}±{c2:.3f}  '
          f'{m3:.3f}±{c3:.3f}  {m4:.3f}±{c4:.3f}')

print('\n── Bonferroni-corrected Wilcoxon p-values (Pipeline vs baseline, by metric) ──')
for mkey in METRICS:
    print(f'\n  {METRIC_LABELS[mkey]}:')
    for baseline in BASELINES:
        row = [f'p={p}: {corrected[(baseline, mkey, p)]:.3g} {sig_star(corrected[(baseline, mkey, p)])}'
               for p in p_vals]
        print(f'    vs {baseline:<13} | ' + '  '.join(row))

print('\nDone.')
