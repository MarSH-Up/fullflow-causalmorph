"""
generate_presentation_figures.py
=================================
Generates all 14 figures for the presentation plan in docs/presentation_plan.md.

Output directory: presentation_figures/

Figures produced
----------------
Standalone wavelet-explanation figures (no CausalMorph required):
  fig_01_morlet_wavelet.png          Morlet wavelet shapes at 4 scales
  fig_02_rolling_moments.png         4 rolling moments of a non-stationary signal
  fig_03_cwt_scalogram.png           CWT scalogram zoomed around a change point
  fig_04_surrogate_calibration.png   Fourier surrogates + calibrated thresholds
  fig_05_signed_energy_computation.png  step-by-step: scalogram → Z → E_signed
  fig_06_multimoment_aggregation.png  factorial weights + 4 moment energies combined
  fig_07_derivative_detection.png    E_signed → dE → find_peaks
  fig_08_channel_gate.png            K-of-channels gate with per-channel z-scores

Pipeline-run figures (full_pipeline.py execution):
  fig_09_detection_diagnostic.png    8-panel diagnostic
  fig_10_timeseries.png              time series with GT / detected CPs
  fig_11_structures_comparison.png   true vs learned DAGs
  fig_12_consensus.png               Bayesian consensus 3-panel
  fig_13_shd_metrics.png             SHD / F1 / Precision / Recall bar chart

Architecture figure:
  fig_14_pipeline_architecture.png   full pipeline flow diagram

Run with:
  python generate_presentation_figures.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from math import factorial

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(_HERE, "causalmorph"))

from detectors.detectors_wavelets import (
    morlet_wavelet,
    cwt_morlet,
    compute_rolling_moments_fast,
    fourier_surrogate,
    get_moment_weights,
    compute_energy_derivative,
    calibrate_multi_moment_thresholds,
)

# ── Output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(_HERE, "presentation_figures")
os.makedirs(OUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":     150,
})

BLUE   = "#2980B9"
RED    = "#E74C3C"
GREEN  = "#27AE60"
PURPLE = "#8E44AD"
ORANGE = "#E67E22"
GRAY   = "#95A5A6"
DARK   = "#2C3E50"
LIGHT  = "#ECF0F1"

MOMENT_COLORS = {1: RED, 2: BLUE, 3: GREEN, 4: PURPLE}
MOMENT_NAMES  = {1: "Mean", 2: "Variance", 3: "Skewness", 4: "Kurtosis"}

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  ✓  {name}")


# =============================================================================
# fig_01 — Morlet wavelet shapes + scalogram
# =============================================================================
def fig_01_morlet_wavelet():
    scales  = [4, 8, 16, 32]
    colors  = [BLUE, GREEN, ORANGE, PURPLE]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Top: real part of Morlet at each scale
    ax = axes[0]
    for s, c in zip(scales, colors):
        L = int(10 * s) | 1          # odd length
        w = morlet_wavelet(L, s, w=6.0)
        t = (np.arange(L) - (L - 1) / 2)
        ax.plot(t, w.real, color=c, lw=1.8, label=f"scale s={s}")
        ax.fill_between(t, w.real, alpha=0.08, color=c)
    ax.axhline(0, color=DARK, lw=0.5, linestyle="--")
    ax.set_title("Morlet Wavelet — Real Part at Four Scales", fontweight="bold")
    ax.set_xlabel("Relative time (samples)")
    ax.set_ylabel("Amplitude")
    ax.legend(ncol=4, fontsize=10, frameon=False)
    ax.set_xlim(-80, 80)

    # Bottom: scalogram of a synthetic variance-step signal
    rng = np.random.default_rng(0)
    T = 600
    cp = 300
    x = np.concatenate([rng.normal(0, 0.5, cp), rng.normal(0, 2.0, T - cp)])

    sc_arr = np.logspace(np.log2(3), np.log2(48), 20, base=2)
    W = cwt_morlet(x, sc_arr, omega0=6.0)
    SG = np.abs(W) ** 2

    ax = axes[1]
    im = ax.imshow(
        SG, aspect="auto",
        extent=[0, T, sc_arr[-1], sc_arr[0]],
        cmap="hot", interpolation="bilinear",
    )
    ax.axvline(cp, color="cyan", lw=2, linestyle="--", label="Change point")
    ax.set_title("Scalogram 𝒲(t, s)² — Variance Step at t=300", fontweight="bold")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Scale s")
    ax.legend(fontsize=10, frameon=False)
    plt.colorbar(im, ax=ax, label="Power", shrink=0.8)

    plt.tight_layout()
    save("fig_01_morlet_wavelet.png")


# =============================================================================
# fig_02 — Rolling moments of a non-stationary signal
# =============================================================================
def fig_02_rolling_moments():
    rng = np.random.default_rng(42)
    T  = 700
    cp = 350
    # regime 0: mean=0, std=0.5; regime 1: mean=1, std=1.8 (mean+variance shift)
    x = np.concatenate([rng.normal(0, 0.5, cp), rng.normal(1.0, 1.8, T - cp)])

    window = 50
    moments_dict = compute_rolling_moments_fast(x, window, [1, 2, 3, 4], causal=True)

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # Panel 0: raw signal
    ax = axes[0]
    ax.plot(x, color=DARK, lw=0.8, alpha=0.9)
    ax.axvline(cp, color=RED, lw=2, linestyle="--", label="Change point")
    ax.set_ylabel("Signal", fontsize=11)
    ax.set_title("Raw Signal  (mean shift + variance increase at t=350)", fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.2)

    # Panels 1–4: each moment
    for i, m in enumerate([1, 2, 3, 4]):
        ax = axes[i + 1]
        ax.plot(moments_dict[m], color=MOMENT_COLORS[m], lw=1.4)
        ax.fill_between(range(T), moments_dict[m], alpha=0.15, color=MOMENT_COLORS[m])
        ax.axvline(cp, color=RED, lw=2, linestyle="--", alpha=0.7)
        ax.set_ylabel(f"M{m}: {MOMENT_NAMES[m]}", fontsize=11)
        ax.set_title(
            f"Rolling {MOMENT_NAMES[m]}  (w={1/factorial(m):.3f} = 1/{m}!)",
            fontsize=11
        )
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (samples)", fontsize=12)
    plt.suptitle(
        "Rolling Moments of a Non-Stationary Signal — Window W=50",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save("fig_02_rolling_moments.png")


# =============================================================================
# fig_03 — CWT scalogram close-up around a change point
# =============================================================================
def fig_03_cwt_scalogram():
    rng = np.random.default_rng(7)
    T  = 800
    cp = 400
    x  = np.concatenate([rng.normal(0, 0.6, cp), rng.normal(0, 2.2, T - cp)])

    # compute rolling variance (moment 2) then CWT
    var_series = compute_rolling_moments_fast(x, 50, [2], causal=True)[2]

    sc_arr = np.logspace(np.log2(3), np.log2(64), 24, base=2)
    W_coef = cwt_morlet(var_series, sc_arr, omega0=6.0)
    SG     = np.abs(W_coef) ** 2

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                              gridspec_kw={"height_ratios": [1, 0.8, 1.5]})

    # Panel 1: raw signal
    ax = axes[0]
    ax.plot(x, color=DARK, lw=0.7, alpha=0.9)
    ax.axvline(cp, color=RED, lw=2.5, linestyle="--", label="Change point")
    ax.set_ylabel("Signal", fontsize=11)
    ax.set_title("Raw Signal", fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.2)

    # Panel 2: rolling variance
    ax = axes[1]
    ax.plot(var_series, color=BLUE, lw=1.2)
    ax.fill_between(range(T), var_series, alpha=0.2, color=BLUE)
    ax.axvline(cp, color=RED, lw=2.5, linestyle="--")
    ax.set_ylabel("Rolling Variance", fontsize=11)
    ax.set_title("Rolling Variance M²(t)  [W=50 causal window]", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Panel 3: scalogram
    ax = axes[2]
    im = ax.imshow(
        np.log1p(SG),
        aspect="auto",
        extent=[0, T, sc_arr[-1], sc_arr[0]],
        cmap="inferno", interpolation="bilinear",
    )
    ax.axvline(cp, color="cyan", lw=2.5, linestyle="--", label="Change point")
    ax.set_ylabel("CWT Scale s", fontsize=11)
    ax.set_xlabel("Time (samples)", fontsize=12)
    ax.set_title(
        "CWT Scalogram of Rolling Variance  log(1 + 𝒲²) — energy concentrates at change point",
        fontweight="bold"
    )
    ax.legend(fontsize=10, frameon=False)
    plt.colorbar(im, ax=ax, label="log(1+power)", shrink=0.9)

    plt.suptitle(
        "Multi-Scale Wavelet Analysis: Energy Localisation at the Change Point",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save("fig_03_cwt_scalogram.png")


# =============================================================================
# fig_04 — Surrogate calibration
# =============================================================================
def fig_04_surrogate_calibration():
    rng   = np.random.default_rng(0)
    T_b   = 400                              # baseline length
    x_base = rng.normal(0, 1.0, T_b)

    # Generate 3 surrogates (show phase randomisation)
    surr_rng = np.random.default_rng(5)
    surrogates = [fourier_surrogate(x_base, surr_rng) for _ in range(3)]

    sc_arr = np.logspace(np.log2(3), np.log2(40), 16, base=2)
    n_surr = 100
    alpha  = 0.40

    # Calibrate thresholds (use variance moment for illustration)
    var_base = compute_rolling_moments_fast(x_base, 50, [2], causal=True)[2]
    surr_quantiles = np.zeros((n_surr, len(sc_arr)))
    for i in range(n_surr):
        x_s   = fourier_surrogate(var_base, rng)
        W_s   = cwt_morlet(x_s, sc_arr, omega0=6.0)
        SG_s  = np.abs(W_s) ** 2
        surr_quantiles[i, :] = np.quantile(SG_s, 0.95, axis=1)
    thresholds = np.quantile(surr_quantiles, 1 - alpha, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: original + surrogates
    ax = axes[0]
    t = np.arange(T_b)
    ax.plot(t, x_base, color=DARK, lw=1.2, zorder=5, label="Baseline signal")
    surr_colors = [RED, BLUE, GREEN]
    for j, (s, c) in enumerate(zip(surrogates, surr_colors)):
        ax.plot(t, s, color=c, lw=0.9, alpha=0.65, label=f"Surrogate {j+1}")
    ax.set_xlabel("Time (samples)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_title(
        "Fourier Surrogates\nSame power spectrum, randomised phase → stationary by construction",
        fontweight="bold"
    )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.2)

    # Right: thresholds per scale
    ax = axes[1]
    ax.step(range(len(sc_arr)), thresholds, color=PURPLE, lw=2.2, where="mid",
            label=f"θ_s  (1−α={1-alpha:.2f} quantile)")
    ax.fill_between(range(len(sc_arr)), thresholds, alpha=0.2, color=PURPLE, step="mid")
    ax.set_xticks(range(len(sc_arr)))
    ax.set_xticklabels([f"{s:.1f}" for s in sc_arr], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("CWT Scale s", fontsize=12)
    ax.set_ylabel("Threshold θ_s", fontsize=12)
    ax.set_title(
        f"Calibrated Thresholds per Scale\n"
        f"α={alpha}, K={n_surr} surrogates  (variance moment)",
        fontweight="bold"
    )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        "FWER-Controlled Threshold Calibration via Fourier Surrogates",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    save("fig_04_surrogate_calibration.png")


# =============================================================================
# fig_05 — Signed energy computation step-by-step
# =============================================================================
def fig_05_signed_energy_computation():
    rng = np.random.default_rng(3)
    T  = 600
    cp = 300
    x  = np.concatenate([rng.normal(0, 0.5, cp), rng.normal(0, 2.0, T - cp)])

    sc_arr = np.logspace(np.log2(3), np.log2(40), 16, base=2)
    eps    = 1e-10

    # Use variance moment
    var_series = compute_rolling_moments_fast(x, 50, [2], causal=True)[2]

    # Calibrate threshold
    baseline_var = var_series[:cp // 2]
    rng2 = np.random.default_rng(99)
    surr_q = np.zeros((50, len(sc_arr)))
    for i in range(50):
        xs   = fourier_surrogate(baseline_var, rng2)
        SGs  = np.abs(cwt_morlet(xs, sc_arr)) ** 2
        surr_q[i] = np.quantile(SGs, 0.95, axis=1)
    threshold = np.quantile(surr_q, 0.60, axis=0)   # α=0.40 → 1-0.40=0.60

    W   = cwt_morlet(var_series, sc_arr, omega0=6.0)
    SG  = np.abs(W) ** 2

    Z   = np.log((SG + eps) / (threshold[:, np.newaxis] + eps))
    Z_pos = np.maximum(Z, 0.0)
    Z_neg = np.maximum(-Z, 0.0)
    E_pos = Z_pos.sum(axis=0)
    E_neg = Z_neg.sum(axis=0)
    E_sig = E_pos - E_neg

    fig = plt.figure(figsize=(14, 12))
    gs  = GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.30)

    def vline(ax):
        ax.axvline(cp, color=RED, lw=2, linestyle="--", alpha=0.8, label="Change point")

    # 1. Rolling variance
    ax = fig.add_subplot(gs[0, :])
    ax.plot(var_series, color=BLUE, lw=1.2)
    ax.fill_between(range(T), var_series, alpha=0.15, color=BLUE)
    vline(ax)
    ax.set_title("Step 1 — Rolling Variance M²(t)", fontweight="bold")
    ax.set_ylabel("M²(t)")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)

    # 2. Scalogram SG
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(np.log1p(SG), aspect="auto",
                   extent=[0, T, sc_arr[-1], sc_arr[0]],
                   cmap="hot", interpolation="bilinear")
    ax.axvline(cp, color="cyan", lw=2, linestyle="--")
    plt.colorbar(im, ax=ax, label="log(1+𝒲²)", shrink=0.9)
    ax.set_title("Step 2 — Scalogram 𝒲(t,s)²", fontweight="bold")
    ax.set_ylabel("Scale s")
    ax.set_xlabel("Time")

    # 3. Z = log(SG / threshold)
    ax = fig.add_subplot(gs[1, 1])
    vmax = np.percentile(np.abs(Z), 98)
    im = ax.imshow(Z, aspect="auto",
                   extent=[0, T, sc_arr[-1], sc_arr[0]],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.axvline(cp, color=DARK, lw=2, linestyle="--")
    plt.colorbar(im, ax=ax, label="Z = log(𝒲/θ)", shrink=0.9)
    ax.set_title("Step 3 — Signed Log-Ratio Z(t,s)", fontweight="bold")
    ax.set_ylabel("Scale s")
    ax.set_xlabel("Time")

    # 4. E_pos and E_neg
    ax = fig.add_subplot(gs[2, :])
    ax.fill_between(range(T), E_pos, alpha=0.35, color=RED, label="E⁺(t)  [above threshold]")
    ax.fill_between(range(T), -E_neg, alpha=0.35, color=BLUE, label="−E⁻(t)  [below threshold]")
    ax.plot(E_pos,  color=RED,  lw=1.0)
    ax.plot(-E_neg, color=BLUE, lw=1.0)
    ax.axhline(0, color=DARK, lw=0.5)
    vline(ax)
    ax.set_title("Step 4 — Positive and Negative Energy Channels", fontweight="bold")
    ax.set_ylabel("Energy")
    ax.legend(fontsize=9, ncol=3, frameon=False)
    ax.grid(True, alpha=0.2)

    # 5. E_signed
    ax = fig.add_subplot(gs[3, :])
    ax.fill_between(range(T), E_sig, where=(E_sig > 0),
                    alpha=0.4, color=RED, label="E_signed > 0 (onset)")
    ax.fill_between(range(T), E_sig, where=(E_sig < 0),
                    alpha=0.4, color=BLUE, label="E_signed < 0 (offset)")
    ax.plot(E_sig, color=DARK, lw=1.0)
    ax.axhline(0, color=DARK, lw=0.5)
    vline(ax)
    ax.set_title("Step 5 — Signed Aggregate Energy E_signed(t) = E⁺ − E⁻", fontweight="bold")
    ax.set_ylabel("E_signed")
    ax.set_xlabel("Time (samples)")
    ax.legend(fontsize=9, ncol=3, frameon=False)
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Signed Log-Ratio Energy Computation — Step by Step (Variance Moment)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    save("fig_05_signed_energy_computation.png")


# =============================================================================
# fig_06 — Multi-moment aggregation: factorial weights + combined E_signed
# =============================================================================
def fig_06_multimoment_aggregation():
    rng  = np.random.default_rng(42)
    T    = 700
    cp   = 350
    N    = 3                  # 3 channels
    eps  = 1e-10

    # Multichannel signal
    X = np.zeros((T, N))
    for ch in range(N):
        X[:cp,  ch] = rng.normal(0, 0.5 + 0.1 * ch, cp)
        X[cp:,  ch] = rng.normal(0.5 * ch, 1.5 + 0.2 * ch, T - cp)

    sc_arr   = np.logspace(np.log2(3), np.log2(40), 16, base=2)
    moments  = [1, 2, 3, 4]
    weights  = get_moment_weights(moments)
    baseline_end = cp // 2

    # Calibrate each moment threshold (fast, small n_surr)
    X_baseline = X[:baseline_end, :]
    calib_rng  = np.random.default_rng(0)

    moment_thresholds = {}
    for m in moments:
        thr_m = np.zeros((N, len(sc_arr)))
        for ch in range(N):
            x_b = X_baseline[:, ch]
            ms  = compute_rolling_moments_fast(x_b, 50, [m], causal=True)[m]
            sq  = np.zeros((40, len(sc_arr)))
            for i in range(40):
                xs  = fourier_surrogate(ms, calib_rng)
                sq[i] = np.quantile(np.abs(cwt_morlet(xs, sc_arr)) ** 2, 0.95, axis=1)
            thr_m[ch] = np.quantile(sq, 0.60, axis=0)
        moment_thresholds[m] = thr_m

    # Compute per-moment signed energies
    E_m_signed = {}
    for m in moments:
        E_pos_m = np.zeros(T)
        E_neg_m = np.zeros(T)
        for ch in range(N):
            ms  = compute_rolling_moments_fast(X[:, ch], 50, [m], causal=True)[m]
            W   = cwt_morlet(ms, sc_arr, omega0=6.0)
            SG  = np.abs(W) ** 2
            Z   = np.log((SG + eps) / (moment_thresholds[m][ch, :, np.newaxis] + eps))
            E_pos_m += np.maximum(Z, 0.0).sum(axis=0)
            E_neg_m += np.maximum(-Z, 0.0).sum(axis=0)
        E_m_signed[m] = E_pos_m - E_neg_m

    E_weighted = sum(weights[m] * E_m_signed[m] for m in moments)

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # Left top: factorial weights bar chart
    ax = fig.add_subplot(gs[0, 0])
    ms_list = [1, 2, 3, 4]
    ws_list = [weights[m] for m in ms_list]
    bars = ax.bar(
        [f"M{m}\n1/{m}!" for m in ms_list],
        ws_list,
        color=[MOMENT_COLORS[m] for m in ms_list],
        edgecolor="white", linewidth=1.5,
    )
    for bar, w in zip(bars, ws_list):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{w:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Weight  w_m = 1/m!", fontsize=11)
    ax.set_title("Factorial-Inverse Weights\nLower moments weighted higher", fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)

    # Right top: per-moment energy traces
    ax = fig.add_subplot(gs[0, 1])
    for m in moments:
        ax.plot(E_m_signed[m], color=MOMENT_COLORS[m], lw=1.1, alpha=0.8,
                label=f"m={m}: {MOMENT_NAMES[m]}")
    ax.axvline(cp, color=RED, lw=2, linestyle="--", label="Change point")
    ax.set_title("Per-Moment Signed Energies E_signed^(m)(t)", fontweight="bold")
    ax.set_ylabel("E_signed^(m)")
    ax.legend(fontsize=9, frameon=False, ncol=2)
    ax.grid(True, alpha=0.2)

    # Middle row: 4 individual panels
    for i, m in enumerate(moments):
        col = i % 2
        row = 1 + i // 2
        ax = fig.add_subplot(gs[row, col])
        ax.fill_between(range(T), E_m_signed[m],
                        where=(E_m_signed[m] >= 0), alpha=0.35, color=MOMENT_COLORS[m])
        ax.fill_between(range(T), E_m_signed[m],
                        where=(E_m_signed[m] < 0), alpha=0.20, color=GRAY)
        ax.plot(E_m_signed[m], color=MOMENT_COLORS[m], lw=1.0)
        ax.axvline(cp, color=RED, lw=2, linestyle="--")
        ax.set_title(
            f"{MOMENT_NAMES[m]}  (w={weights[m]:.3f} = 1/{m}!)",
            fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.2)
        if row == 2:
            ax.set_xlabel("Time (samples)")

    plt.suptitle(
        "Multi-Moment Aggregation with Factorial-Inverse Weights",
        fontsize=14, fontweight="bold", y=1.01,
    )
    save("fig_06_multimoment_aggregation.png")


# =============================================================================
# fig_07 — Derivative-based peak detection
# =============================================================================
def fig_07_derivative_detection():
    rng = np.random.default_rng(10)
    T   = 800
    cp  = 400

    # Create a realistic E_signed trace: step at cp
    E_signed = np.zeros(T)
    E_signed[:cp]   = rng.normal(0, 1.0, cp)
    E_signed[cp:] = rng.normal(8.0, 1.2, T - cp)
    # smooth
    k = np.ones(20) / 20
    E_signed = np.convolve(E_signed, k, mode="same")

    dE = compute_energy_derivative(E_signed, smooth_window=12)

    # find_peaks on dE
    eps_pos = np.quantile(dE[dE > 0], 0.90) if (dE > 0).any() else 1.0
    peak_idx, props = find_peaks(dE, height=eps_pos, distance=150, prominence=eps_pos * 0.3)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Panel 1: E_signed
    ax = axes[0]
    ax.fill_between(range(T), E_signed, where=(E_signed >= 0),
                    alpha=0.25, color=RED)
    ax.fill_between(range(T), E_signed, where=(E_signed < 0),
                    alpha=0.25, color=BLUE)
    ax.plot(E_signed, color=DARK, lw=1.2)
    ax.axvline(cp, color=GREEN, lw=2.5, linestyle="--", label="True change point")
    ax.set_ylabel("E_signed(t)", fontsize=11)
    ax.set_title("Signed Aggregate Energy E_signed(t)  — sustained step at change point",
                 fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.2)

    # Panel 2: dE
    ax = axes[1]
    ax.fill_between(range(T), dE, where=(dE > 0), alpha=0.4, color=RED, label="dE > 0")
    ax.fill_between(range(T), dE, where=(dE < 0), alpha=0.4, color=BLUE, label="dE < 0")
    ax.plot(dE, color=DARK, lw=0.9)
    ax.axhline(eps_pos,  color=RED,  linestyle=":", lw=1.8, alpha=0.8,
               label=f"ε threshold = {eps_pos:.2f}")
    ax.axhline(-eps_pos, color=BLUE, linestyle=":", lw=1.8, alpha=0.8)
    ax.axvline(cp, color=GREEN, lw=2.5, linestyle="--")
    ax.set_ylabel("ΔE(t)", fontsize=11)
    ax.set_title(
        "Energy Derivative ΔE(t) = E_signed(t) − E_signed(t−1)  — spike at transition",
        fontweight="bold"
    )
    ax.legend(fontsize=9, frameon=False, ncol=3)
    ax.grid(True, alpha=0.2)

    # Panel 3: peaks highlighted
    ax = axes[2]
    ax.plot(dE, color=DARK, lw=0.9, alpha=0.8)
    ax.axhline(eps_pos, color=RED, linestyle=":", lw=1.8, alpha=0.8,
               label=f"ε_on = {eps_pos:.2f}")
    for pk in peak_idx:
        ax.axvline(pk, color=RED, lw=2.5, linestyle="--",
                   label=f"Detected peak τ={pk}" if pk == peak_idx[0] else "")
        ax.plot(pk, dE[pk], "^", color=RED, markersize=14, zorder=10)
    ax.axvline(cp, color=GREEN, lw=2.5, linestyle="--", label=f"True CP t={cp}")
    ax.set_ylabel("ΔE(t)", fontsize=11)
    ax.set_xlabel("Time (samples)", fontsize=12)
    ax.set_title(
        "Peak Detection via scipy.signal.find_peaks  (height=ε, refractory=150)",
        fontweight="bold"
    )
    ax.legend(fontsize=9, frameon=False, ncol=3)
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Derivative-Based Detection: Transitions, Not Sustained States",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save("fig_07_derivative_detection.png")


# =============================================================================
# fig_08 — K-of-channels consistency gate
# =============================================================================
def fig_08_channel_gate():
    rng = np.random.default_rng(21)
    T   = 500
    cp  = 250
    N   = 5

    # Channels: 3 active (clear step), 2 noisy (no step)
    E_ch = np.zeros((T, N))
    active = [0, 1, 3]
    noise  = [2, 4]
    for ch in active:
        E_ch[:cp, ch]  = rng.normal(0, 0.5, cp)
        E_ch[cp:, ch]  = rng.normal(3.5 + rng.uniform(-0.5, 0.5), 0.6, T - cp)
    for ch in noise:
        E_ch[:, ch] = rng.normal(0, 0.8, T)

    # Compute z-scores
    gap, pre_w, post_w = 10, 160, 160
    tau = cp
    z_scores = np.zeros(N)
    deltas   = np.zeros(N)
    for ch in range(N):
        pre  = E_ch[max(0, tau - gap - pre_w) : tau - gap, ch]
        post = E_ch[tau + gap : min(T, tau + gap + post_w), ch]
        if len(pre) == 0 or len(post) == 0:
            continue
        delta = float(np.median(post) - np.median(pre))
        sigma = max(1.4826 * np.median(np.abs(pre - np.median(pre))), 1e-3)
        z_scores[ch] = delta / sigma
        deltas[ch]   = delta

    k_z    = 1.5
    n_active = int(np.sum(z_scores > k_z))
    total_delta = np.abs(deltas).sum()
    conc_ratio  = np.abs(deltas).max() / total_delta if total_delta > 1e-10 else 1.0

    ch_names   = [f"V{i+1}" for i in range(N)]
    ch_colors  = [GREEN if z_scores[i] > k_z else GRAY for i in range(N)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

    # Left: per-channel E_signed traces
    ax = axes[0]
    offsets = np.arange(N) * 6
    for ch in range(N):
        col  = GREEN if ch in active else GRAY
        lbl  = "Active channel (step detected)" if ch == active[0] else (
               "Inactive channel (noise only)" if ch == noise[0] else "")
        ax.plot(E_ch[:, ch] + offsets[ch], color=col, lw=1.2, alpha=0.85, label=lbl)
        ax.axhline(offsets[ch], color=col, lw=0.4, linestyle=":")
        ax.text(-15, offsets[ch], f"V{ch+1}", ha="right", va="center",
                fontsize=11, fontweight="bold", color=col)
    ax.axvline(cp, color=RED, lw=2.5, linestyle="--", label="Change point τ")
    ax.set_ylabel("E_n(t) + offset", fontsize=11)
    ax.set_xlabel("Time (samples)", fontsize=12)
    ax.set_title(
        f"Per-Channel Signed Energy at Change Point τ={cp}\n"
        f"Active (green): {n_active}/{N}  ·  Concentration R={conc_ratio:.2f}",
        fontweight="bold"
    )
    handles, labels = ax.get_legend_handles_labels()
    # deduplicate
    seen = {}
    hl = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.update({l: True})]
    ax.legend(*zip(*hl), fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-30, T)

    # Right: z-score bar chart
    ax = axes[1]
    bars = ax.barh(ch_names, z_scores, color=ch_colors, edgecolor="white", linewidth=1.5)
    ax.axvline(k_z,  color=RED,  lw=2.0, linestyle="--", label=f"k_z = {k_z}")
    ax.axvline(-k_z, color=RED,  lw=2.0, linestyle="--")
    ax.axvline(0,    color=DARK, lw=0.7)
    for i, (bar, z) in enumerate(zip(bars, z_scores)):
        ax.text(z + 0.1 if z >= 0 else z - 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{z:.1f}", va="center",
                ha="left" if z >= 0 else "right",
                fontsize=10, fontweight="bold", color=ch_colors[i])
    ax.set_xlabel("z-score  (step / σ_pre)", fontsize=11)
    ax.set_title(
        f"Per-Channel z-Scores\n"
        f"Active ≥ k_z={k_z}:  {n_active}/{N} channels  →  ACCEPT",
        fontweight="bold"
    )
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.2)

    green_patch = mpatches.Patch(color=GREEN, label="Active (z ≥ 1.5)")
    gray_patch  = mpatches.Patch(color=GRAY,  label="Inactive (z < 1.5)")
    ax.legend(handles=[green_patch, gray_patch,
                        plt.Line2D([0], [0], color=RED, lw=2, linestyle="--",
                                   label=f"k_z = {k_z}")],
              fontsize=9, frameon=False)

    plt.suptitle(
        "K-of-Channels Consistency Gate (v1-F)\n"
        "Rejects peaks driven by a single channel",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    save("fig_08_channel_gate.png")


# =============================================================================
# fig_14 — Pipeline architecture (clean flow diagram)
# =============================================================================
def fig_14_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(cx, cy, w, h, text, color, fontsize=11, text_color="white"):
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="white", linewidth=2, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=text_color, zorder=4, wrap=True,
                multialignment="center")

    def arrow(x0, y0, x1, y1):
        ax.annotate("",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=DARK,
                            lw=2.0, mutation_scale=18),
            zorder=2,
        )

    def label(x, y, text, color=DARK, fontsize=9):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=color, style="italic", zorder=5)

    # ── Main pipeline boxes (vertical flow, left column) ──────────────────────
    boxes = [
        (5.0, 9.0, 8.0, 0.8, "Multi-Regime Causal Time Series  X ∈ ℝ^{T × p}", DARK),
        (5.0, 7.5, 8.0, 0.9,
         "[1]  GENERATE SCENARIO\nLearning trajectory  G_init → G_target  ·  R=5 regimes  ·  p=5  ·  600–800 samples/regime",
         BLUE),
        (5.0, 6.0, 8.0, 0.9,
         "[2]  DETECT CHANGE POINTS  (Gatekeeper v1-F)\nRolling moments → CWT (Morlet) → Surrogate calibration → Signed energy → dE peaks → Gates",
         "#C0392B"),
        (5.0, 4.5, 8.0, 0.9,
         "[3]  EXTRACT CAUSAL STRUCTURES  (CausalMorph + DirectLiNGAM)\nIterative warm-start: each window's output is prior for the next",
         GREEN),
        (5.0, 3.0, 8.0, 0.9,
         "[4]  BAYESIAN AGGREGATION\nBeta-Bernoulli: length-weighted votes per edge → posterior P(edge) → consensus graph",
         PURPLE),
        (5.0, 1.5, 8.0, 0.9,
         "[5]  EVALUATE\nSHD, F1, Precision, Recall  per regime  ·  Detection F1",
         ORANGE),
    ]
    for (cx, cy, w, h, text, col) in boxes:
        box(cx, cy, w, h, text, col, fontsize=9.5)

    # ── Arrows ────────────────────────────────────────────────────────────────
    arrow_pairs = [
        (5.0, 8.60, 5.0, 7.95),
        (5.0, 7.05, 5.0, 6.45),
        (5.0, 5.55, 5.0, 4.95),
        (5.0, 4.05, 5.0, 3.45),
        (5.0, 2.55, 5.0, 1.95),
    ]
    labels_on_arrows = [
        "X [T × p]",
        "True CPs  {τ₁, …, τ_{R-1}}",
        "Detected CPs  {τ̂₁, …}",
        "List[RegimeStructure]",
        "Consensus graph  +  per-regime structs",
    ]
    for (x0, y0, x1, y1), lbl in zip(arrow_pairs, labels_on_arrows):
        arrow(x0, y0, x1, y1)
        label((x0 + x1) / 2 + 0.3, (y0 + y1) / 2, lbl, fontsize=8.5)

    # ── Side annotations ──────────────────────────────────────────────────────
    side_items = [
        (9.5, 6.0,  "7 internal steps:\n"
                    "1. Artifact rejection\n"
                    "2. Rolling moments (M1–M4)\n"
                    "3. CWT Morlet multi-scale\n"
                    "4. Fourier surrogate calibration\n"
                    "5. Signed log-ratio Z = log(𝒲/θ)\n"
                    "6. dE derivative + find_peaks\n"
                    "7. Step + K-of-channels gates",
         "#C0392B"),
        (9.5, 4.5,  "Per-window flow:\n"
                    "X_window → CausalMorph(prior)\n"
                    "→ X' = (I − B̂) X\n"
                    "→ DirectLiNGAM on X'\n"
                    "→ new (π, B̂) → next prior",
         GREEN),
    ]
    for (x, y, text, col) in side_items:
        ax.text(x, y, text, ha="left", va="center", fontsize=7.5,
                color=col, zorder=5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=col, alpha=0.85, lw=1.5))

    ax.set_title(
        "Full Pipeline: Non-Stationary Causal Discovery",
        fontsize=16, fontweight="bold", pad=8, color=DARK,
    )
    save("fig_14_pipeline_architecture.png")


# =============================================================================
# Pipeline-run figures (fig_09 – fig_13) via run_full_pipeline
# =============================================================================
def run_pipeline_figures():
    print("\n  Running full pipeline (may take ~2 min for surrogate calibration)…")

    import full_pipeline as fp
    from full_pipeline import (
        plot_detection_diagnostics,
        plot_timeseries,
        plot_structures_comparison,
        plot_consensus_structure,
        plot_shd_metrics,
        aggregate_structures_bayesian,
    )

    def _save_current(name):
        path = os.path.join(OUT, name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close("all")
        print(f"  ✓  {name}")

    # Silence plt.show so nothing pops up during generation
    original_show = plt.show
    plt.show = lambda *a, **kw: None

    try:
        result = fp.run_full_pipeline(
            p=5,
            n_regimes=5,
            min_samples=600,
            max_samples=800,
            base_pconn=0.35,
            seed=42,
            verbose=False,
            show_plots=False,          # we draw manually below
        )

        X            = result["X"]
        det_result   = result["detection_result"]
        true_cps     = result["true_change_points"]
        detected_cps = result["detected_change_points"]
        structures   = result["structures"]
        var_names    = result["variable_names"]
        true_adjs    = result.get("true_adjs", [])
        consensus_adj = result["consensus_adj"]
        edge_probs    = result["edge_probs"]
        scenario      = fp.build_nonstationary_scenario(
            p=5, n_regimes=5, min_samples=600, max_samples=800,
            base_pconn=0.35, change_pcts=[0,30,25,35,30], seed=42,
        )[4]   # index 4 = scenario dict

        # fig_09: detection diagnostic
        print("[Plots] Detection diagnostics (v1-F multi-moment)...")
        plot_detection_diagnostics(
            X, det_result, true_cps, detected_cps,
            title="Full Pipeline v1-F: 5 regimes, p=5",
            tolerance=125,
        )
        _save_current("fig_09_detection_diagnostic.png")

        # fig_10: time series
        print("[Plots] Time series with change points...")
        plot_timeseries(X, var_names, true_cps, detected_cps)
        _save_current("fig_10_timeseries.png")

        # fig_11: true vs learned structures
        print("[Plots] True vs Learned causal structures...")
        plot_structures_comparison(scenario, structures, var_names)
        _save_current("fig_11_structures_comparison.png")

        # fig_12: Bayesian consensus
        print("[Plots] Bayesian consensus structure...")
        plot_consensus_structure(
            consensus_adj, edge_probs, var_names,
            len(structures), true_adjs=true_adjs,
        )
        _save_current("fig_12_consensus.png")

        # fig_13: SHD metrics bar chart
        print("[Plots] SHD metrics...")
        plot_shd_metrics(structures)
        _save_current("fig_13_shd_metrics.png")

    except Exception as exc:
        print(f"  ✗  Pipeline run failed: {exc}")
        import traceback; traceback.print_exc()
    finally:
        plt.show = original_show


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"\nGenerating presentation figures → {OUT}/\n")

    print("── Standalone wavelet-explanation figures ────────────────────────")
    fig_01_morlet_wavelet()
    fig_02_rolling_moments()
    fig_03_cwt_scalogram()
    fig_04_surrogate_calibration()
    fig_05_signed_energy_computation()
    fig_06_multimoment_aggregation()
    fig_07_derivative_detection()
    fig_08_channel_gate()

    print("\n── Architecture diagram ──────────────────────────────────────────")
    fig_14_pipeline_architecture()

    print("\n── Pipeline-run figures (full pipeline execution, ~2 min) ──────────")
    run_pipeline_figures()

    print(f"\nDone. All figures saved to:  {OUT}/")
