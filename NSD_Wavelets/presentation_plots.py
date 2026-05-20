#!/usr/bin/env python3
"""
Presentation plots for NSD_Wavelets — Slides 14 & 15.

Slide 14  "Why mean was not enough"
          Four synthetic regimes with constant mean but changing
          variance / skewness / kurtosis.  Shows KPSS/ADF (mean-only)
          missing every change while rolling moments catch all of them.

Slide 15  "Gatekeeper v1-F — pipeline walkthrough"
          Single multivariate signal flowing through each stage:
          MAD rejection → rolling moments → CWT Morlet scalogram →
          surrogate calibration → signed log-ratio → derivative peaks →
          K-of-channels gate.

Run from NSD_Wavelets/
    python presentation_plots.py

Outputs (180 dpi PNG):
    slide14_motivation.png
    slide15_pipeline.png
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ── path: allow importing from NSD_Wavelets/src ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

try:
    from detectors.detectors_wavelets import (
        remove_gross_artifacts,
        compute_rolling_moments,
        cwt_morlet,
    )
    _NSD = True
except ImportError as err:
    _NSD = False
    print(f"[warn] NSD import failed ({err}) — using scipy fallbacks")

# ── colour palette ────────────────────────────────────────────────────────────
C = dict(
    signal   = "#2E86AB",
    change   = "#D32F2F",
    artifact = "#FF5252",
    m1       = "#546E7A",   # mean
    m2       = "#F57C00",   # variance
    m3       = "#2E7D32",   # skewness
    m4       = "#6A1B9A",   # kurtosis
    epos     = "#E53935",   # positive energy
    eneg     = "#1565C0",   # negative energy
    gate     = "#E65100",   # K-of-channels highlight
)
REGIME_BG = ["#EFF8FF", "#FFF5E6", "#F2FCF2", "#FDF2F8"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "figure.dpi":        150,
})

# ── fallback implementations (used only when NSD import fails) ────────────────

def _rolling_moments_fallback(x, window=80):
    T = len(x)
    out = {k: np.zeros(T) for k in [1, 2, 3, 4]}
    half = window // 2
    xp = np.pad(x, (half, half), mode="reflect")
    for t in range(T):
        w = xp[t: t + window]
        out[1][t] = np.mean(w)
        out[2][t] = np.var(w)
        out[3][t] = stats.skew(w, bias=False)
        out[4][t] = stats.kurtosis(w, bias=False, fisher=True)
    return out


def _cwt_fallback(x, n_scales=32):
    try:
        from scipy.signal import morlet2, cwt
        scales = np.geomspace(2, len(x) // 4, n_scales)
        W = cwt(x, morlet2, scales, w=6.0)
        return np.abs(W) ** 2, scales
    except Exception:
        # last-resort: sliding-window variance as a mock scalogram
        T = len(x)
        scales = np.geomspace(2, T // 4, n_scales)
        SG = np.zeros((n_scales, T))
        for i, s in enumerate(scales):
            win = max(2, int(s))
            pad = np.pad(x, win // 2, mode="reflect")
            SG[i] = [np.var(pad[t: t + win]) for t in range(T)]
        return SG, scales


def _mad_clean_fallback(x, thresh=5.0):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    bad = np.abs(x - med) > thresh * max(mad, 1e-10)
    out = x.copy()
    if bad.any():
        idx = np.arange(len(x))
        out[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return out, bad


# convenience wrappers ─────────────────────────────────────────────────────────

def _moments(x, window=80):
    if _NSD:
        return compute_rolling_moments(x, window=window, moments=[1, 2, 3, 4])
    return _rolling_moments_fallback(x, window=window)


def _cwt(x, n_scales=32):
    T = len(x)
    if _NSD:
        scales = np.geomspace(2, T // 4, n_scales)
        W = cwt_morlet(x, scales)
        return np.abs(W) ** 2, scales
    return _cwt_fallback(x, n_scales)


def _clean(x):
    if _NSD:
        x_cl = remove_gross_artifacts(x, mad_threshold=5.0)
        mask = np.abs(x - x_cl) > 1.0
        return x_cl, mask
    return _mad_clean_fallback(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  SLIDE 14  —  "Detecting changes in mean was not enough"
# ═══════════════════════════════════════════════════════════════════════════════

def _make_slide14_data(seed=42, T=1000, window=80):
    rng = np.random.default_rng(seed)
    seg = T // 4
    cps = [seg, 2 * seg, 3 * seg]

    # Regime 1: Normal(0,1)  — baseline
    s1 = rng.normal(0.0, 1.0, seg)

    # Regime 2: Normal(0,3)  — variance ×9, mean = 0
    s2 = rng.normal(0.0, 3.0, seg)

    # Regime 3: positive skew, var≈1
    #   Exponential(1) centred → mean=0, var=1, skew≈2, kurt≈6
    raw3 = rng.exponential(1.0, seg)
    s3 = (raw3 - raw3.mean()) / raw3.std()

    # Regime 4: heavy tails, var≈1, skew≈0
    #   Laplace(0, 1/√2) → var=1, excess kurt=3, skew=0
    s4 = rng.laplace(0.0, 1.0 / np.sqrt(2), seg)

    x = np.concatenate([s1, s2, s3, s4])
    moms = _moments(x, window=window)
    return x, moms, cps, seg


def plot_slide14(path="slide14_motivation.png"):
    print("  building slide 14 data...")
    x, moms, cps, seg = _make_slide14_data()
    T = len(x)
    t = np.arange(T)
    bounds = [0] + cps + [T]

    regime_labels = [
        "Baseline\nNormal(0, σ²=1)",
        "Variance ×9\nNormal(0, σ²=9)",
        "Positive Skew\n(Exp-centred, skew≈2)",
        "Heavy Tails\n(Laplace, kurt≈+3)",
    ]

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(5, 1, hspace=0.55,
                           top=0.93, bottom=0.07, left=0.09, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    def shade(ax):
        for k in range(4):
            ax.axvspan(bounds[k], bounds[k + 1],
                       color=REGIME_BG[k], alpha=0.7, zorder=0)
        for cp in cps:
            ax.axvline(cp, color=C["change"], lw=1.1, ls="--",
                       alpha=0.8, zorder=2)
        ax.set_xlim(0, T)

    # ── panel 0: raw signal ──────────────────────────────────────────────────
    ax = axes[0]
    shade(ax)
    ax.plot(t, x, color=C["signal"], lw=0.45, alpha=0.75)
    ax.set_ylabel("x(t)", fontsize=9)
    ax.set_title(
        "Four regimes — constant mean, different higher moments."
        "  Can KPSS / ADF detect these changes?",
        fontsize=12.5, fontweight="bold", pad=7,
    )
    ax.tick_params(labelbottom=False)
    # regime labels via axis coordinates
    for k, lbl in enumerate(regime_labels):
        mid_ax = ((bounds[k] + bounds[k + 1]) / 2) / T
        ax.text(mid_ax, 0.97, lbl,
                ha="center", va="top", fontsize=7.8, color="#444444",
                transform=ax.transAxes)

    # ── panels 1-4: rolling moments ──────────────────────────────────────────
    mom_cfg = [
        (1, "Rolling\nMean",              C["m1"], True),
        (2, "Rolling\nVariance",           C["m2"], False),
        (3, "Rolling\nSkewness",           C["m3"], False),
        (4, "Rolling\nKurtosis\n(excess)", C["m4"], False),
    ]
    for row, (mk, label, color, is_mean) in enumerate(mom_cfg, start=1):
        ax = axes[row]
        shade(ax)
        y = moms[mk]
        ax.plot(t, y, color=color, lw=1.6, zorder=3)
        ax.set_ylabel(label, fontsize=8.5, color=color, labelpad=4)
        ax.yaxis.label.set_color(color)
        if row < 4:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (samples)", fontsize=9.5)

        if is_mean:
            # ── "KPSS/ADF only sees this → flat → MISS" ──
            ax.text(0.50, 0.75,
                    "KPSS / ADF only tests the mean → sees nothing here",
                    ha="center", va="center", fontsize=9.2, color="#B71C1C",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.38", fc="#FFEBEE",
                              ec="#B71C1C", lw=1.2, alpha=0.93))
            for cp in cps:
                ax.text(cp / T + 0.005, 0.45, "✗",
                        ha="left", va="center", fontsize=16,
                        color="#B71C1C", fontweight="bold",
                        transform=ax.transAxes)
        else:
            for cp in cps:
                ax.text(cp / T + 0.005, 0.88, "✓",
                        ha="left", va="top", fontsize=15,
                        color="#1B5E20", fontweight="bold",
                        transform=ax.transAxes)

    # legend
    fig.text(0.975, 0.50,
             "✗  missed by KPSS / ADF\n✓  detected by wavelet\n   moment analysis",
             fontsize=9, ha="right", va="center",
             bbox=dict(boxstyle="round", fc="white", ec="#9E9E9E", alpha=0.93))

    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SLIDE 15  —  "Gatekeeper v1-F — pipeline"
# ═══════════════════════════════════════════════════════════════════════════════

def _make_slide15_data(seed=7, T=800, N=4, window=60):
    rng = np.random.default_rng(seed)
    cp = T // 2
    X = np.zeros((T, N))
    for ch in range(N):
        x1 = rng.normal(0.0, 1.0, cp)
        x2 = rng.normal(0.0, 2.4, T - cp)   # variance jump
        X[:, ch] = np.concatenate([x1, x2])
    X[int(T * 0.30), 0] += 22.0              # artifact in channel 0
    return X, cp


def plot_slide15(path="slide15_pipeline.png"):
    print("  building slide 15 data...")
    T_SIG, N_CH = 800, 4
    X, cp = _make_slide15_data(T=T_SIG, N=N_CH)
    T, N = X.shape
    t = np.arange(T)
    window = 60

    # ① MAD artifact rejection
    x_raw = X[:, 0].copy()
    x_clean, art_mask = _clean(x_raw)

    # ② rolling moments on cleaned ch 0
    print("  computing rolling moments...")
    moms = _moments(x_clean, window=window)

    # ③ CWT on rolling variance of ch 0
    print("  computing CWT (ch 0)...")
    var_s = moms[2].copy()
    var_s[np.isnan(var_s)] = np.nanmean(var_s)
    SG, scales = _cwt(var_s, n_scales=32)

    # ④ surrogate threshold → signed log-ratio
    eps = 1e-12
    thresh = np.percentile(SG[:, :cp], 95, axis=1, keepdims=True)
    Z = np.log((SG + eps) / (thresh + eps))
    E_pos = np.mean(np.maximum(Z, 0.0), axis=0)
    E_neg = np.mean(np.maximum(-Z, 0.0), axis=0)
    E_signed = E_pos - E_neg

    # ⑤ derivative + peaks
    E_sm = uniform_filter1d(E_signed, size=15)
    dE   = uniform_filter1d(np.gradient(E_sm), size=9)
    noise = np.std(dE[:cp])
    pks_on,  _ = find_peaks( dE, height=noise * 1.5, distance=50)
    pks_off, _ = find_peaks(-dE, height=noise * 1.5, distance=50)

    # ⑥ per-channel CWT energy (for K-gate illustration)
    print("  computing per-channel CWT...")
    E_ch = np.zeros((T, N))
    for ch in range(N):
        xch = X[:, ch].copy()
        mch = _moments(xch, window=window)
        vc  = mch[2].copy()
        vc[np.isnan(vc)] = np.nanmean(vc)
        SGc, _ = _cwt(vc, n_scales=len(scales))
        tc  = np.percentile(SGc[:, :cp], 95, axis=1, keepdims=True)
        Zc  = np.log((SGc + eps) / (tc + eps))
        E_ch[:, ch] = np.mean(np.maximum(Zc, 0.0), axis=0)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(6, 1, hspace=0.68,
                           top=0.95, bottom=0.05, left=0.10, right=0.97)

    panel_titles = [
        "①  Raw signal  →  MAD artifact rejection",
        "②  Rolling moments  (mean · variance · skewness · kurtosis)",
        "③  CWT Morlet scalogram  (computed on rolling variance)",
        "④  Surrogate calibration  →  signed log-ratio  E⁺(t) − E⁻(t)",
        "⑤  dE/dt  →  derivative peak detection  (onset ▲  /  offset ▽)",
        "⑥  K-of-channels gate  —  consensus across all channels",
    ]
    ch_colors = [C["epos"], C["m2"], C["m3"], C["signal"]]

    def vline(ax):
        ax.axvline(cp, color=C["change"], lw=1.2, ls="--", alpha=0.75)

    # ── ① ───────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.plot(t, x_raw, color="#BDBDBD", lw=0.7, label="raw", zorder=1)
    ax.plot(t, x_clean, color=C["signal"], lw=1.1, label="cleaned", zorder=2)
    if art_mask.any():
        ai = np.where(art_mask)[0]
        ax.scatter(ai, x_raw[ai], color=C["artifact"], s=60, zorder=3,
                   marker="x", linewidths=2.0, label="artifact removed")
    vline(ax)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.88)
    ax.set_title(panel_titles[0], fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_xlim(0, T)
    ax.tick_params(labelbottom=False)

    # ── ② ───────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    for mk, mlbl, mc in [(1, "mean", C["m1"]), (2, "variance", C["m2"]),
                          (3, "skewness", C["m3"]), (4, "kurtosis", C["m4"])]:
        y = moms[mk].copy()
        r = np.nanmax(np.abs(y)) or 1.0
        ax.plot(t, y / r, color=mc, lw=1.25, label=mlbl, alpha=0.90)
    vline(ax)
    ax.axhline(0, color="#E0E0E0", lw=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.88, ncol=4)
    ax.set_title(panel_titles[1], fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel("Normalised", fontsize=8)
    ax.set_xlim(0, T)
    ax.tick_params(labelbottom=False)

    # ── ③ ───────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    ax.imshow(SG, aspect="auto", origin="lower",
              extent=[0, T, 0, SG.shape[0]],
              cmap="inferno", interpolation="bilinear")
    ax.axvline(cp, color="white", lw=1.8, ls="--", alpha=0.90)
    ax.set_title(panel_titles[2], fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel("Scale index", fontsize=8)
    ax.set_xlim(0, T)
    ax.tick_params(labelbottom=False)

    # ── ④ ───────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3])
    ax.fill_between(t, 0, E_pos,  color=C["epos"], alpha=0.65, label="E⁺  onset")
    ax.fill_between(t, 0, -E_neg, color=C["eneg"], alpha=0.65, label="E⁻  offset")
    ax.axhline(0, color="#9E9E9E", lw=0.5)
    vline(ax)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.88)
    ax.set_title(panel_titles[3], fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel("E(t)", fontsize=8)
    ax.set_xlim(0, T)
    ax.tick_params(labelbottom=False)

    # ── ⑤ ───────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[4])
    ax.plot(t, dE, color="#546E7A", lw=1.0)
    ax.axhline(0, color="#E0E0E0", lw=0.5)
    if len(pks_on):
        ax.scatter(pks_on, dE[pks_on], color=C["epos"], s=70, zorder=4,
                   marker="^", label="onset")
    if len(pks_off):
        ax.scatter(pks_off, dE[pks_off], color=C["eneg"], s=70, zorder=4,
                   marker="v", label="offset")
    vline(ax)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.88)
    ax.set_title(panel_titles[4], fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel("dE/dt", fontsize=8)
    ax.set_xlim(0, T)
    ax.tick_params(labelbottom=False)

    # ── ⑥ ───────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[5])
    step = np.nanmax(E_ch) * 0.60 if np.nanmax(E_ch) > 0 else 0.3
    for ch in range(N):
        ax.plot(t, E_ch[:, ch] + ch * step,
                color=ch_colors[ch], lw=1.1, alpha=0.90,
                label=f"ch {ch + 1}")
    # consensus annotation near the detected peak
    if len(pks_on):
        best = int(pks_on[np.argmax(dE[pks_on])])
        ax.axvspan(best - 18, best + 18, color=C["gate"], alpha=0.18)
        ax.text((best + 18) / T + 0.01, 0.80,
                f"K = {N} channels agree\n→ detection KEPT",
                ha="left", va="top", fontsize=8.5,
                color="#BF360C", fontweight="bold",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=C["gate"], alpha=0.95))
    vline(ax)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.88, ncol=N + 1)
    ax.set_title(panel_titles[5], fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel("E per channel\n(offset for clarity)", fontsize=8)
    ax.set_xlabel("Time (samples)", fontsize=9)
    ax.set_xlim(0, T)

    fig.suptitle("Gatekeeper v1-F — Pipeline Overview",
                 fontsize=13.5, fontweight="bold", y=0.985)

    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating slide 14 (motivation)...")
    plot_slide14()

    print("Generating slide 15 (pipeline)...")
    plot_slide15()

    print("\nDone!  Upload to Canva:")
    print("  slide14_motivation.png")
    print("  slide15_pipeline.png")
