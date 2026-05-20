"""
Gatekeeper v1-F: CWT+Morlet Wavelet Non-Stationarity Detector
  Multi-Moment Analysis + Multichannel Consistency Gate

Features:
- K-OF-CHANNELS GATE (v1-F): Rejects peaks where too few channels show a
  significant step — reduces false positives from single-channel noise artifacts
- Per-event multichannel diagnostics: n_active_channels, concentration_ratio, channel_z_scores
- MULTI-MOMENT ANALYSIS: Detects changes in mean, variance, skewness, and kurtosis
- Factorial-inverse weighting: 1/m! for moment m (mean=1, var=0.5, skew~0.17, kurt~0.04)
- Signed wavelet deviations (log-ratio): Z = log(SG/T) for symmetric up/down detection
- Differential change score: detect on dE (derivative), not E level
- PEAK-BASED detection: uses scipy.signal.find_peaks with prominence + distance
- STEP VALIDATION: filters peaks by sustained E_signed level shift (kills transient FPs)
- Peak-height calibration: eps thresholds from surrogate peak heights (not max-stat)
- Time-quantile SG thresholds (not max-over-time, which is too conservative)
- Separate E_pos/E_neg channels for up-shifts and down-shifts
- Forgetting severity trace (leaky integrator) instead of cumulative sum
- Edge padding (reflect) + edge margin to avoid boundary artifacts
- K-of-scales gate to reduce isolated false positives
- Vectorized operations (10-100x faster)
- Robust artifact rejection (median + MAD)

Key changes from v1-E:
- NEW: K-of-channels consistency gate (filter_peaks_by_channel_consistency)
  Requires k_channels_min channels to show a significant per-channel step
  (MAD-scaled z-score > delta_ch_k) before accepting a candidate peak.
  Auto-disabled for N=1; clamped to max(1, N//2) for small N.
- NEW: Per-event diagnostics: n_active_channels, frac_active_channels,
  channel_z_scores, concentration_ratio stored in Event dataclass
- Pipeline order: peaks -> step validation -> channel gate -> SNR filter

Key changes from v1-D (preserved):
- Multi-moment detection (mean, variance, skewness, kurtosis)
- Factorial-inverse weighting (1/m!) instead of voting
- Rolling window statistics for moment computation

Key changes from v1-C (preserved):
- Z is now signed (log-ratio), not one-sided max(0, ...)
- Detection operates on dE (derivative spike at transitions), not E (level)
- Uses find_peaks with prominence for event extraction (not hysteresis)
- STEP VALIDATION: only keep peaks with sustained median(post) - median(pre) shift
- Calibrates eps on surrogate PEAK HEIGHTS (not max-stat)
- SG thresholds use time-quantile (not max-stat)
- Severity uses forgetting (leaky integrator), not cumsum
- Two-sided event detection: onset (dE > eps), offset (-dE > eps)

"""

import numpy as np
from math import factorial
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple, Dict


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Event:
    """A detected non-stationarity event."""
    t_start: int
    t_end: int
    tau: int                                    # Change point (derivative peak)
    event_type: str                             # "onset" (up-shift) or "offset" (down-shift)
    channels_ranked: List[int]                  # Channels by contribution
    scales_ranked: List[int]                    # Scales by contribution
    peak_dE: float                              # Peak derivative magnitude
    peak_E: float                               # Peak energy level at tau
    area: float                                 # Total |dE| in event (severity)
    channel_scores: Optional[np.ndarray] = None
    scale_scores: Optional[np.ndarray] = None
    # v1-F: Multichannel consistency diagnostics
    n_active_channels: Optional[int] = None         # Channels with significant step
    frac_active_channels: Optional[float] = None    # n_active / N
    channel_z_scores: Optional[np.ndarray] = None   # Per-channel MAD-scaled step z-scores [N]
    concentration_ratio: Optional[float] = None     # max|Δ_ch| / Σ|Δ_ch| (1=single-ch, 1/N=distributed)


@dataclass
class CalibrationResult:
    """Calibration thresholds from baseline using max-stat for FWER control."""
    thresholds: np.ndarray          # Shape: [N_channels, M_scales] - max-stat quantiles
    eps_on_pos: float               # Threshold for positive dE (onset detection)
    eps_on_neg: float               # Threshold for negative dE (offset detection)
    n_surrogates: int
    baseline_length: int
    alpha: float                    # Per-scale alpha used
    # Multi-moment thresholds (v1-E)
    moment_thresholds: Optional[Dict[int, np.ndarray]] = None  # {moment: [N, M]}
    moments_used: Optional[List[int]] = None  # Which moments were calibrated


@dataclass
class DetectionResult:
    """Complete detection result."""
    events: List[Event]
    E_pos: np.ndarray               # Up-shift energy time series [T]
    E_neg: np.ndarray               # Down-shift energy time series [T]
    E_signed: np.ndarray            # Signed energy (E_pos - E_neg) [T]
    dE: np.ndarray                  # Smoothed derivative of E_signed [T]
    severity: np.ndarray            # Forgetting severity trace [T]
    E_ch_pos: np.ndarray            # Per-channel up-shift energy [T, N]
    E_ch_neg: np.ndarray            # Per-channel down-shift energy [T, N]
    calibration: CalibrationResult
    is_nonstationary: bool
    change_points: List[int]        # tau values from events (onset and offset)
    onset_points: List[int]         # Onset change points only
    offset_points: List[int]        # Offset change points only
    edge_margin: int                # Samples ignored at edges
    # Multi-moment results (v1-E)
    E_by_moment: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None  # {m: (E_pos, E_neg)}
    moments_used: Optional[List[int]] = None  # Which moments were used
    moment_weights: Optional[Dict[int, float]] = None  # Factorial-inverse weights
    # Per-moment voting diagnostics (v1-G)
    moment_change_points: Optional[Dict[int, List[int]]] = None  # {m: [change_point_indices]}
    moment_dE: Optional[Dict[int, np.ndarray]] = None             # {m: dE_m [T]} for plotting


# =============================================================================
# STEP 0: Artifact Rejection (Robust: Median + MAD)
# =============================================================================

def remove_gross_artifacts(
    x: np.ndarray,
    mad_threshold: float = 5.0,
) -> np.ndarray:
    """
    Remove gross artifacts using robust Median + MAD.

    MAD (Median Absolute Deviation) is more robust to heavy tails than z-score.

    Parameters
    ----------
    x : np.ndarray
        1D signal
    mad_threshold : float
        Threshold in MAD units for outlier detection

    Returns
    -------
    np.ndarray
        Cleaned signal (outliers interpolated)
    """
    x_clean = x.copy().astype(float)

    # Detect NaN/Inf
    bad_mask = ~np.isfinite(x_clean)

    # Robust outlier detection via MAD
    valid = x_clean[np.isfinite(x_clean)]
    if len(valid) > 0:
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        # Scale MAD to be comparable to std (for normal distribution)
        mad_scaled = 1.4826 * mad if mad > 0 else 1e-10

        deviation = np.abs(x_clean - median) / mad_scaled
        bad_mask |= (deviation > mad_threshold)

    # Interpolate bad values
    if np.any(bad_mask):
        good_idx = np.where(~bad_mask)[0]
        bad_idx = np.where(bad_mask)[0]
        if len(good_idx) >= 2:
            x_clean[bad_idx] = np.interp(bad_idx, good_idx, x_clean[good_idx])
        elif len(good_idx) == 1:
            x_clean[bad_idx] = x_clean[good_idx[0]]

    return x_clean


# =============================================================================
# STEP 0b: Rolling Moments Computation
# =============================================================================

def compute_rolling_moments(
    x: np.ndarray,
    window: int = 50,
    moments: List[int] = [1, 2, 3, 4],
) -> Dict[int, np.ndarray]:
    """
    Compute rolling statistical moments for a 1D signal.

    Parameters
    ----------
    x : np.ndarray
        1D signal of length T
    window : int
        Rolling window size
    moments : List[int]
        Which moments to compute (1=mean, 2=variance, 3=skewness, 4=kurtosis)

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping moment number to time series [T]
    """
    T = len(x)
    result = {}

    # Pad signal to handle edges
    half_win = window // 2
    x_padded = np.pad(x, (half_win, half_win), mode='reflect')

    for m in moments:
        moment_series = np.zeros(T)

        for t in range(T):
            # Extract window centered at t
            window_data = x_padded[t:t + window]

            if m == 1:
                # Mean (1st moment)
                moment_series[t] = np.mean(window_data)
            elif m == 2:
                # Variance (2nd central moment)
                moment_series[t] = np.var(window_data)
            elif m == 3:
                # Skewness (3rd standardized moment)
                # Use scipy.stats.skew with bias=False for sample skewness
                std = np.std(window_data)
                if std > 1e-10:
                    moment_series[t] = skew(window_data, bias=False)
                else:
                    moment_series[t] = 0.0
            elif m == 4:
                # Kurtosis (4th standardized moment, excess kurtosis)
                std = np.std(window_data)
                if std > 1e-10:
                    moment_series[t] = kurtosis(window_data, bias=False, fisher=True)
                else:
                    moment_series[t] = 0.0

        result[m] = moment_series

    return result


def compute_rolling_moments_fast(
    x: np.ndarray,
    window: int = 50,
    moments: List[int] = [1, 2, 3, 4],
    causal: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Fast vectorized rolling moments computation using convolution.

    For mean and variance, uses efficient convolution.
    For skewness and kurtosis, uses optimized rolling computation.

    Parameters
    ----------
    x : np.ndarray
        1D signal of length T
    window : int
        Rolling window size
    moments : List[int]
        Which moments to compute (1=mean, 2=variance, 3=skewness, 4=kurtosis)
    causal : bool
        If True, use backward-looking (causal) windows: at time t, uses samples [t-window+1, t].
        This prevents early detection at change points.
        If False, use centered windows (original behavior).

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping moment number to time series [T]
    """
    T = len(x)
    result = {}

    # Kernel for moving average
    kernel = np.ones(window) / window

    if causal:
        # CAUSAL: Backward-looking windows
        # At time t, use samples [t-window+1, t] (inclusive)
        # Pad at the beginning only
        x_padded = np.pad(x, (window - 1, 0), mode='reflect')
    else:
        # CENTERED: Original behavior
        x_padded = np.pad(x, (window // 2, window - window // 2 - 1), mode='reflect')

    if 1 in moments:
        # Rolling mean (fast via convolution)
        result[1] = np.convolve(x_padded, kernel, mode='valid')[:T]

    if 2 in moments:
        # Rolling variance: E[X^2] - E[X]^2
        if causal:
            x2_padded = np.pad(x**2, (window - 1, 0), mode='reflect')
        else:
            x2_padded = np.pad(x**2, (window // 2, window - window // 2 - 1), mode='reflect')
        mean_x = np.convolve(x_padded, kernel, mode='valid')[:T]
        mean_x2 = np.convolve(x2_padded, kernel, mode='valid')[:T]
        result[2] = np.maximum(mean_x2 - mean_x**2, 0)  # Ensure non-negative

    # For skewness and kurtosis, use strided view approach
    if 3 in moments or 4 in moments:
        # Create strided view for efficient window access
        from numpy.lib.stride_tricks import sliding_window_view

        if causal:
            # CAUSAL: Pad at the beginning only
            x_pad = np.pad(x, (window - 1, 0), mode='reflect')
        else:
            # CENTERED: Original behavior
            half_win = window // 2
            x_pad = np.pad(x, (half_win, window - half_win - 1), mode='reflect')

        # Get sliding windows [T, window]
        windows = sliding_window_view(x_pad, window)[:T]

        if 3 in moments:
            # Vectorized skewness
            mean_w = windows.mean(axis=1, keepdims=True)
            std_w = windows.std(axis=1, keepdims=True)
            std_w = np.maximum(std_w, 1e-10)
            z = (windows - mean_w) / std_w
            # Skewness = E[z^3]
            result[3] = np.mean(z**3, axis=1)

        if 4 in moments:
            # Vectorized excess kurtosis
            mean_w = windows.mean(axis=1, keepdims=True)
            std_w = windows.std(axis=1, keepdims=True)
            std_w = np.maximum(std_w, 1e-10)
            z = (windows - mean_w) / std_w
            # Excess kurtosis = E[z^4] - 3
            result[4] = np.mean(z**4, axis=1) - 3.0

    return result


def get_moment_weights(moments: List[int] = [1, 2, 3, 4]) -> Dict[int, float]:
    """
    Get factorial-inverse weights for each moment.

    Weight for moment m = 1/m!

    Parameters
    ----------
    moments : List[int]
        Which moments to weight

    Returns
    -------
    Dict[int, float]
        Dictionary mapping moment number to weight
    """
    return {m: 1.0 / factorial(m) for m in moments}


# =============================================================================
# STEP 1: Calibration via Fourier Surrogates
# =============================================================================

def fourier_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Fourier surrogate: preserves |FFT|, randomizes phase.
    """
    n = len(x)
    X_fft = np.fft.rfft(x)
    amplitudes = np.abs(X_fft)

    random_phases = rng.uniform(0, 2 * np.pi, len(X_fft))
    random_phases[0] = 0  # DC
    if n % 2 == 0:
        random_phases[-1] = 0  # Nyquist

    X_surr = amplitudes * np.exp(1j * random_phases)
    return np.fft.irfft(X_surr, n=n).real


def morlet_wavelet(M: int, s: float, w: float = 6.0) -> np.ndarray:
    """
    Complex Morlet wavelet (replaces scipy.signal.morlet2).

    Parameters
    ----------
    M : int
        Length of the wavelet
    s : float
        Scale parameter
    w : float
        Central frequency (omega0)

    Returns
    -------
    np.ndarray
        Complex Morlet wavelet of length M
    """
    # Time vector centered at 0
    t = np.arange(M) - (M - 1) / 2
    t = t / s  # Scale time

    # Morlet wavelet: exp(i*w*t) * exp(-t^2/2)
    # Normalization: 1/sqrt(s) for energy preservation
    wavelet = np.exp(1j * w * t) * np.exp(-t**2 / 2)

    # Normalize
    wavelet = wavelet / np.sqrt(s)

    return wavelet


def cwt_morlet(
    x: np.ndarray,
    scales: np.ndarray,
    omega0: float = 6.0,
) -> np.ndarray:
    """
    Continuous Wavelet Transform using Morlet wavelet via FFT convolution.

    This implementation works with any scipy version (morlet2 was removed
    in scipy 1.12+).

    Parameters
    ----------
    x : np.ndarray
        1D signal of length T
    scales : np.ndarray
        Array of scales [M]
    omega0 : float
        Morlet wavelet central frequency

    Returns
    -------
    np.ndarray
        Complex CWT coefficients, shape [M, T]
    """
    T = len(x)
    M = len(scales)

    # Compute wavelet support for padding
    max_scale = np.max(scales)
    pad_len = int(3 * max_scale)

    # Reflect-pad to avoid edge effects
    x_padded = np.pad(x, pad_len, mode='reflect')
    T_padded = len(x_padded)

    coeffs = np.zeros((M, T), dtype=complex)

    for i, scale in enumerate(scales):
        # Generate Morlet wavelet at this scale
        # Wavelet length should be ~10*scale for good support
        wavelet_len = min(int(10 * scale), T_padded)
        if wavelet_len % 2 == 0:
            wavelet_len += 1  # Make odd for symmetry

        wavelet = morlet_wavelet(wavelet_len, scale, w=omega0)

        # Convolve using FFT (faster for long signals)
        conv_result = scipy_signal.fftconvolve(x_padded, wavelet, mode='same')

        # Crop back to original length
        coeffs[i, :] = conv_result[pad_len:pad_len + T]

    return coeffs


def calibrate_thresholds_maxstat(
    X_baseline: np.ndarray,
    scales: np.ndarray,
    n_surrogates: int = 100,
    alpha: float = 0.05,
    omega0: float = 6.0,
    smooth_window: int = 7,
    seed: Optional[int] = None,
) -> CalibrationResult:
    """
    Build per-channel, per-scale thresholds and derivative-based event thresholds.

    v1-D changes:
    - Still uses max-stat for scalogram thresholds
    - Now calibrates eps_on_pos and eps_on_neg on DERIVATIVE extremes (not level)
    - This ensures detection triggers on transitions, not sustained regimes

    Parameters
    ----------
    alpha : float
        Controls threshold strictness via quantile (smaller = stricter)
    smooth_window : int
        Smoothing window for derivative computation
    """
    rng = np.random.default_rng(seed)
    T_base, N = X_baseline.shape
    M = len(scales)
    eps = 1e-10

    thresholds = np.zeros((N, M))

    # Collect surrogate statistics for scalogram thresholds
    # v1-D FIX: Use TIME-QUANTILE instead of max-stat to avoid "everything below threshold"
    # Max-stat is too conservative: threshold = max_t(SG_surr) → almost all SG < T → E_neg dominates
    # Time-quantile: threshold = quantile_t(SG_surr, 0.95) → more balanced E_pos/E_neg
    all_surr_quantiles = np.zeros((N, n_surrogates, M))
    time_quantile = 0.95  # Use 95th percentile over time instead of max

    for ch in range(N):
        x_base = remove_gross_artifacts(X_baseline[:, ch])

        for i in range(n_surrogates):
            x_surr = fourier_surrogate(x_base, rng)
            W_surr = cwt_morlet(x_surr, scales, omega0)
            SG_surr = np.abs(W_surr) ** 2  # [M, T_base]
            # Use time-quantile instead of max for more balanced thresholds
            all_surr_quantiles[ch, i, :] = np.quantile(SG_surr, time_quantile, axis=1)

    # Compute scalogram thresholds: use (1-alpha) quantile of surrogate time-quantiles
    for ch in range(N):
        thresholds[ch, :] = np.quantile(all_surr_quantiles[ch], 1 - alpha, axis=0)

    # Compute eps_on_pos and eps_on_neg from PEAK HEIGHTS of surrogate dE
    # v1-D FIX: Calibrate on peak heights, not max(dE), to match peak-based detection
    # Using max(dE) is too conservative (inflates with T), causing threshold > max|dE|
    peak_heights_pos = []
    peak_heights_neg = []
    n_eps_surr = min(n_surrogates, 50)  # Speed cap

    # Refractory period for peak detection (same as detection will use)
    peak_refractory = 20

    # Pre-clean baseline channels
    x_base_clean = [remove_gross_artifacts(X_baseline[:, ch]) for ch in range(N)]

    for i in range(n_eps_surr):
        E_ch_pos_surr = np.zeros((T_base, N), dtype=float)
        E_ch_neg_surr = np.zeros((T_base, N), dtype=float)

        for ch in range(N):
            x_surr = fourier_surrogate(x_base_clean[ch], rng)
            W_surr = cwt_morlet(x_surr, scales, omega0)
            SG_surr = np.abs(W_surr) ** 2  # [M, T_base]

            # Log-ratio deviation (signed)
            thresh_ch = thresholds[ch, :, np.newaxis]  # [M, 1]
            Z_surr = np.log((SG_surr + eps) / (thresh_ch + eps))  # [M, T]

            # Separate positive and negative
            Z_pos = np.maximum(Z_surr, 0.0)
            Z_neg = np.maximum(-Z_surr, 0.0)

            E_ch_pos_surr[:, ch] = Z_pos.sum(axis=0)
            E_ch_neg_surr[:, ch] = Z_neg.sum(axis=0)

        # Aggregate across channels
        E_pos_surr = E_ch_pos_surr.sum(axis=1)
        E_neg_surr = E_ch_neg_surr.sum(axis=1)
        E_signed_surr = E_pos_surr - E_neg_surr

        # Compute derivative of signed energy
        dE_surr = compute_energy_derivative(E_signed_surr, smooth_window)

        # Exclude edge samples to avoid boundary artifacts from CWT
        edge_margin = max(10, int(np.max(scales)))
        dE_interior = dE_surr[edge_margin:-edge_margin] if T_base > 2 * edge_margin else dE_surr

        # Collect PEAK HEIGHTS (not max) from surrogate dE
        # This matches the detection decision rule (find_peaks with height threshold)
        idx_pos, props_pos = find_peaks(dE_interior, distance=peak_refractory, height=0)
        idx_neg, props_neg = find_peaks(-dE_interior, distance=peak_refractory, height=0)

        if "peak_heights" in props_pos and len(props_pos["peak_heights"]) > 0:
            peak_heights_pos.extend(props_pos["peak_heights"].tolist())
        if "peak_heights" in props_neg and len(props_neg["peak_heights"]) > 0:
            peak_heights_neg.extend(props_neg["peak_heights"].tolist())

    # Fallback if no peaks found (rare but possible)
    if len(peak_heights_pos) == 0:
        # Use 99.9th percentile of |dE| as fallback
        all_dE = []
        for i in range(min(5, n_eps_surr)):
            E_ch_pos_surr = np.zeros((T_base, N), dtype=float)
            for ch in range(N):
                x_surr = fourier_surrogate(x_base_clean[ch], rng)
                W_surr = cwt_morlet(x_surr, scales, omega0)
                SG_surr = np.abs(W_surr) ** 2
                thresh_ch = thresholds[ch, :, np.newaxis]
                Z_surr = np.log((SG_surr + eps) / (thresh_ch + eps))
                E_ch_pos_surr[:, ch] = np.maximum(Z_surr, 0.0).sum(axis=0)
            dE_tmp = compute_energy_derivative(E_ch_pos_surr.sum(axis=1), smooth_window)
            all_dE.extend(np.abs(dE_tmp).tolist())
        peak_heights_pos = [np.quantile(all_dE, 0.999)] if all_dE else [1.0]

    if len(peak_heights_neg) == 0:
        peak_heights_neg = peak_heights_pos.copy()  # Use same fallback

    # eps_on = (1-alpha) quantile of surrogate PEAK heights
    # This is less conservative than max-stat, matches detector's decision rule
    eps_on_pos = float(np.quantile(peak_heights_pos, 1 - alpha))
    eps_on_neg = float(np.quantile(peak_heights_neg, 1 - alpha))

    # Minimum thresholds to avoid spurious triggers
    eps_on_pos = max(eps_on_pos, 0.1)
    eps_on_neg = max(eps_on_neg, 0.1)

    return CalibrationResult(
        thresholds=thresholds,
        eps_on_pos=eps_on_pos,
        eps_on_neg=eps_on_neg,
        n_surrogates=n_surrogates,
        baseline_length=T_base,
        alpha=alpha,
    )


def calibrate_multi_moment_thresholds(
    X_baseline: np.ndarray,
    scales: np.ndarray,
    moments: List[int] = [1, 2, 3, 4],
    moment_window: int = 50,
    n_surrogates: int = 100,
    alpha: float = 0.05,
    omega0: float = 6.0,
    smooth_window: int = 7,
    seed: Optional[int] = None,
) -> CalibrationResult:
    """
    Calibrate thresholds for multi-moment detection (v1-E).

    Computes:
    1. Per-moment, per-channel, per-scale thresholds
    2. eps_on_pos/neg thresholds for weighted dE

    Parameters
    ----------
    X_baseline : np.ndarray
        Baseline data [T, N]
    scales : np.ndarray
        CWT scales
    moments : List[int]
        Which moments to calibrate (1=mean, 2=var, 3=skew, 4=kurt)
    moment_window : int
        Window size for rolling moment computation
    n_surrogates : int
        Number of surrogates
    alpha : float
        Significance level
    omega0 : float
        Morlet central frequency
    smooth_window : int
        Smoothing window for dE
    seed : int, optional
        Random seed

    Returns
    -------
    CalibrationResult
        Calibration with multi-moment thresholds
    """
    rng = np.random.default_rng(seed)
    T_base, N = X_baseline.shape
    M = len(scales)
    eps = 1e-10

    weights = get_moment_weights(moments)
    time_quantile = 0.95

    # Calibrate thresholds for each moment
    moment_thresholds = {}

    for m in moments:
        thresholds_m = np.zeros((N, M))

        for ch in range(N):
            x_base = remove_gross_artifacts(X_baseline[:, ch])

            # Compute rolling moment
            moments_dict = compute_rolling_moments_fast(x_base, moment_window, [m])
            moment_series = moments_dict[m]

            # Collect surrogate statistics
            surr_quantiles = np.zeros((n_surrogates, M))

            for i in range(n_surrogates):
                x_surr = fourier_surrogate(moment_series, rng)
                W_surr = cwt_morlet(x_surr, scales, omega0)
                SG_surr = np.abs(W_surr) ** 2
                surr_quantiles[i, :] = np.quantile(SG_surr, time_quantile, axis=1)

            thresholds_m[ch, :] = np.quantile(surr_quantiles, 1 - alpha, axis=0)

        moment_thresholds[m] = thresholds_m

    # Also store variance thresholds as the primary "thresholds" for backwards compatibility
    if 2 in moment_thresholds:
        thresholds = moment_thresholds[2]
    else:
        thresholds = moment_thresholds[moments[0]]

    # Calibrate eps thresholds on weighted dE from surrogates
    peak_heights_pos = []
    peak_heights_neg = []
    n_eps_surr = min(n_surrogates, 50)
    peak_refractory = 20

    # Pre-compute baseline moment series for each channel and moment
    baseline_moments = {m: {} for m in moments}
    for m in moments:
        for ch in range(N):
            x_base = remove_gross_artifacts(X_baseline[:, ch])
            moments_dict = compute_rolling_moments_fast(x_base, moment_window, [m])
            baseline_moments[m][ch] = moments_dict[m]

    for i in range(n_eps_surr):
        E_weighted_pos = np.zeros(T_base)
        E_weighted_neg = np.zeros(T_base)

        for m in moments:
            weight = weights[m]
            E_m_pos_ch = np.zeros((T_base, N))
            E_m_neg_ch = np.zeros((T_base, N))

            for ch in range(N):
                # Generate surrogate of moment series
                moment_series = baseline_moments[m][ch]
                x_surr = fourier_surrogate(moment_series, rng)
                W_surr = cwt_morlet(x_surr, scales, omega0)
                SG_surr = np.abs(W_surr) ** 2

                thresh_ch = moment_thresholds[m][ch, :, np.newaxis]
                Z_surr = np.log((SG_surr + eps) / (thresh_ch + eps))

                Z_pos = np.maximum(Z_surr, 0.0)
                Z_neg = np.maximum(-Z_surr, 0.0)

                E_m_pos_ch[:, ch] = Z_pos.sum(axis=0)
                E_m_neg_ch[:, ch] = Z_neg.sum(axis=0)

            # Aggregate and weight
            E_weighted_pos += weight * E_m_pos_ch.sum(axis=1)
            E_weighted_neg += weight * E_m_neg_ch.sum(axis=1)

        # Compute dE for surrogates
        E_signed_surr = E_weighted_pos - E_weighted_neg
        dE_surr = compute_energy_derivative(E_signed_surr, smooth_window)

        # Exclude edges
        edge_margin = max(10, int(np.max(scales)))
        dE_interior = dE_surr[edge_margin:-edge_margin] if T_base > 2 * edge_margin else dE_surr

        # Collect peak heights
        idx_pos, props_pos = find_peaks(dE_interior, distance=peak_refractory, height=0)
        idx_neg, props_neg = find_peaks(-dE_interior, distance=peak_refractory, height=0)

        if "peak_heights" in props_pos and len(props_pos["peak_heights"]) > 0:
            peak_heights_pos.extend(props_pos["peak_heights"].tolist())
        if "peak_heights" in props_neg and len(props_neg["peak_heights"]) > 0:
            peak_heights_neg.extend(props_neg["peak_heights"].tolist())

    # Compute eps thresholds
    if len(peak_heights_pos) == 0:
        peak_heights_pos = [1.0]
    if len(peak_heights_neg) == 0:
        peak_heights_neg = peak_heights_pos.copy()

    eps_on_pos = float(np.quantile(peak_heights_pos, 1 - alpha))
    eps_on_neg = float(np.quantile(peak_heights_neg, 1 - alpha))

    eps_on_pos = max(eps_on_pos, 0.1)
    eps_on_neg = max(eps_on_neg, 0.1)

    return CalibrationResult(
        thresholds=thresholds,
        eps_on_pos=eps_on_pos,
        eps_on_neg=eps_on_neg,
        n_surrogates=n_surrogates,
        baseline_length=T_base,
        alpha=alpha,
        moment_thresholds=moment_thresholds,
        moments_used=moments,
    )


# =============================================================================
# STEP 2: Detection - Compute Signed Energy (Log-Ratio)
# =============================================================================

def compute_signed_energy(
    X: np.ndarray,
    scales: np.ndarray,
    thresholds: np.ndarray,
    omega0: float = 6.0,
    aggregation_mode: Literal["sum_all", "max", "topK"] = "sum_all",
    top_k: int = 3,
    k_scales_min: int = 0,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute SIGNED energy using log-ratio deviation.

    Log-ratio deviation: Z = log((SG + eps) / (threshold + eps))
    - Z > 0: more energy than baseline threshold (potential onset)
    - Z < 0: less energy than baseline threshold (potential offset)

    This is symmetric and scale-free, allowing detection of both
    increases AND decreases relative to baseline.

    We separate into E_pos (up-shifts) and E_neg (down-shifts) to prevent
    cancellation when aggregating across scales.

    Parameters
    ----------
    X : np.ndarray
        Multichannel time series [T, N]
    scales : np.ndarray
        CWT scales
    thresholds : np.ndarray
        Per-channel, per-scale thresholds [N, M]
    omega0 : float
        Morlet central frequency
    aggregation_mode : str
        Channel aggregation: "sum_all", "max", or "topK"
    top_k : int
        Number of top channels for "topK" mode
    k_scales_min : int
        Minimum number of scales that must be significant.
        If fewer than k_scales_min scales are significant, Z is zeroed.
    eps : float
        Small constant for numerical stability in log

    Returns
    -------
    E_pos : np.ndarray
        Up-shift energy [T] (aggregated positive deviations)
    E_neg : np.ndarray
        Down-shift energy [T] (aggregated negative deviations, as positive values)
    E_signed : np.ndarray
        Signed energy [T] = E_pos - E_neg
    E_ch_pos : np.ndarray
        Per-channel up-shift energy [T, N]
    E_ch_neg : np.ndarray
        Per-channel down-shift energy [T, N]
    edge_margin : int
        Samples to ignore at edges (0 with reflect padding)
    """
    T, N = X.shape

    # Edge margin: reflect-padding handles edges
    edge_margin = 0

    E_ch_pos = np.zeros((T, N))
    E_ch_neg = np.zeros((T, N))

    for ch in range(N):
        W = cwt_morlet(X[:, ch], scales, omega0)
        SG = np.abs(W) ** 2  # [M, T]

        # SIGNED LOG-RATIO DEVIATION: Z = log((SG + eps) / (threshold + eps))
        thresh_broadcast = thresholds[ch, :, np.newaxis]  # [M, 1]
        Z = np.log((SG + eps) / (thresh_broadcast + eps))  # [M, T]

        # K-of-scales gate: require at least k scales to be significantly different
        if k_scales_min > 0:
            # Count how many scales have |Z| > some significance threshold
            # Use |Z| > 0.5 (i.e., SG > 1.65*threshold or SG < 0.6*threshold)
            sig = np.abs(Z) > 0.5  # [M, T] boolean
            k_sig = sig.sum(axis=0)  # [T] count of significant scales
            # Zero out Z where fewer than k_scales_min are significant
            Z[:, k_sig < k_scales_min] = 0.0

        # Separate positive and negative deviations
        Z_pos = np.maximum(Z, 0.0)  # [M, T]
        Z_neg = np.maximum(-Z, 0.0)  # [M, T] (flip sign so it's positive)

        # Sum over scales
        E_ch_pos[:, ch] = Z_pos.sum(axis=0)
        E_ch_neg[:, ch] = Z_neg.sum(axis=0)

    # Aggregate across channels (vectorized)
    if aggregation_mode == "sum_all":
        E_pos = E_ch_pos.sum(axis=1)
        E_neg = E_ch_neg.sum(axis=1)
    elif aggregation_mode == "max":
        E_pos = E_ch_pos.max(axis=1)
        E_neg = E_ch_neg.max(axis=1)
    elif aggregation_mode == "topK":
        sorted_pos = np.sort(E_ch_pos, axis=1)[:, ::-1]
        sorted_neg = np.sort(E_ch_neg, axis=1)[:, ::-1]
        E_pos = sorted_pos[:, :min(top_k, N)].sum(axis=1)
        E_neg = sorted_neg[:, :min(top_k, N)].sum(axis=1)
    else:
        E_pos = E_ch_pos.sum(axis=1)
        E_neg = E_ch_neg.sum(axis=1)

    # Combined signed energy
    E_signed = E_pos - E_neg

    return E_pos, E_neg, E_signed, E_ch_pos, E_ch_neg, edge_margin


# =============================================================================
# STEP 2b: Multi-Moment Signed Energy (NEW in v1-E)
# =============================================================================

def calibrate_moment_thresholds(
    X_baseline: np.ndarray,
    scales: np.ndarray,
    moment: int,
    moment_window: int = 50,
    n_surrogates: int = 100,
    alpha: float = 0.05,
    omega0: float = 6.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Calibrate thresholds for a specific moment's wavelet representation.

    Parameters
    ----------
    X_baseline : np.ndarray
        Baseline data [T, N]
    scales : np.ndarray
        CWT scales
    moment : int
        Which moment (1=mean, 2=variance, 3=skewness, 4=kurtosis)
    moment_window : int
        Window size for rolling moment computation
    n_surrogates : int
        Number of surrogates for calibration
    alpha : float
        Significance level
    omega0 : float
        Morlet central frequency
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Thresholds [N, M] for this moment
    """
    rng = np.random.default_rng(seed)
    T_base, N = X_baseline.shape
    M = len(scales)
    eps = 1e-10

    thresholds = np.zeros((N, M))
    time_quantile = 0.95

    for ch in range(N):
        x_base = remove_gross_artifacts(X_baseline[:, ch])

        # Compute rolling moment for this channel
        moments_dict = compute_rolling_moments_fast(x_base, moment_window, [moment])
        moment_series = moments_dict[moment]

        # Collect surrogate statistics
        surr_quantiles = np.zeros((n_surrogates, M))

        for i in range(n_surrogates):
            # Generate surrogate of the moment time series
            x_surr = fourier_surrogate(moment_series, rng)
            W_surr = cwt_morlet(x_surr, scales, omega0)
            SG_surr = np.abs(W_surr) ** 2  # [M, T_base]
            surr_quantiles[i, :] = np.quantile(SG_surr, time_quantile, axis=1)

        thresholds[ch, :] = np.quantile(surr_quantiles, 1 - alpha, axis=0)

    return thresholds


def compute_moment_energy(
    moment_series: np.ndarray,
    scales: np.ndarray,
    threshold: np.ndarray,
    omega0: float = 6.0,
    k_scales_min: int = 0,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute signed energy for a single moment time series.

    Parameters
    ----------
    moment_series : np.ndarray
        1D moment time series [T]
    scales : np.ndarray
        CWT scales
    threshold : np.ndarray
        Per-scale thresholds [M]
    omega0 : float
        Morlet central frequency
    k_scales_min : int
        Minimum scales that must be significant
    eps : float
        Numerical stability constant

    Returns
    -------
    E_pos : np.ndarray
        Up-shift energy [T]
    E_neg : np.ndarray
        Down-shift energy [T]
    """
    T = len(moment_series)

    W = cwt_morlet(moment_series, scales, omega0)
    SG = np.abs(W) ** 2  # [M, T]

    # Signed log-ratio deviation
    thresh_broadcast = threshold[:, np.newaxis]  # [M, 1]
    Z = np.log((SG + eps) / (thresh_broadcast + eps))  # [M, T]

    # K-of-scales gate
    if k_scales_min > 0:
        sig = np.abs(Z) > 0.5
        k_sig = sig.sum(axis=0)
        Z[:, k_sig < k_scales_min] = 0.0

    # Separate positive and negative deviations
    Z_pos = np.maximum(Z, 0.0)
    Z_neg = np.maximum(-Z, 0.0)

    # Sum over scales
    E_pos = Z_pos.sum(axis=0)
    E_neg = Z_neg.sum(axis=0)

    return E_pos, E_neg


def compute_multi_moment_energy(
    X: np.ndarray,
    scales: np.ndarray,
    moment_thresholds: Dict[int, np.ndarray],
    moment_window: int = 50,
    moments: List[int] = [1, 2, 3, 4],
    omega0: float = 6.0,
    aggregation_mode: Literal["sum_all", "max", "topK"] = "sum_all",
    top_k: int = 3,
    k_scales_min: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]], int]:
    """
    Compute signed energy across multiple statistical moments with factorial-inverse weighting.

    For each moment m ∈ {1,2,3,4}:
    1. Compute rolling moment time series for each channel
    2. Apply CWT and compute signed deviations
    3. Weight by 1/m!

    Parameters
    ----------
    X : np.ndarray
        Multichannel time series [T, N]
    scales : np.ndarray
        CWT scales
    moment_thresholds : Dict[int, np.ndarray]
        Thresholds for each moment {m: [N, M]}
    moment_window : int
        Window size for rolling moment computation
    moments : List[int]
        Which moments to use (1=mean, 2=var, 3=skew, 4=kurt)
    omega0 : float
        Morlet central frequency
    aggregation_mode : str
        Channel aggregation mode
    top_k : int
        Number of top channels for topK mode
    k_scales_min : int
        Minimum significant scales

    Returns
    -------
    E_pos : np.ndarray
        Total weighted up-shift energy [T]
    E_neg : np.ndarray
        Total weighted down-shift energy [T]
    E_signed : np.ndarray
        Signed energy [T] = E_pos - E_neg
    E_ch_pos : np.ndarray
        Per-channel up-shift energy [T, N]
    E_ch_neg : np.ndarray
        Per-channel down-shift energy [T, N]
    E_by_moment : Dict[int, Tuple[np.ndarray, np.ndarray]]
        Per-moment energies {m: (E_pos_m, E_neg_m)}
    edge_margin : int
        Samples to ignore at edges
    """
    T, N = X.shape
    weights = get_moment_weights(moments)

    # Initialize accumulators
    E_ch_pos = np.zeros((T, N))
    E_ch_neg = np.zeros((T, N))
    E_by_moment = {}

    for m in moments:
        if m not in moment_thresholds:
            continue

        weight = weights[m]
        thresholds_m = moment_thresholds[m]

        E_m_pos_ch = np.zeros((T, N))
        E_m_neg_ch = np.zeros((T, N))

        for ch in range(N):
            # Compute rolling moment for this channel
            x_ch = remove_gross_artifacts(X[:, ch])
            moments_dict = compute_rolling_moments_fast(x_ch, moment_window, [m])
            moment_series = moments_dict[m]

            # Compute energy for this moment
            E_pos_ch, E_neg_ch = compute_moment_energy(
                moment_series, scales, thresholds_m[ch, :],
                omega0, k_scales_min
            )

            E_m_pos_ch[:, ch] = E_pos_ch
            E_m_neg_ch[:, ch] = E_neg_ch

        # Aggregate across channels for this moment
        if aggregation_mode == "sum_all":
            E_m_pos = E_m_pos_ch.sum(axis=1)
            E_m_neg = E_m_neg_ch.sum(axis=1)
        elif aggregation_mode == "max":
            E_m_pos = E_m_pos_ch.max(axis=1)
            E_m_neg = E_m_neg_ch.max(axis=1)
        elif aggregation_mode == "topK":
            sorted_pos = np.sort(E_m_pos_ch, axis=1)[:, ::-1]
            sorted_neg = np.sort(E_m_neg_ch, axis=1)[:, ::-1]
            E_m_pos = sorted_pos[:, :min(top_k, N)].sum(axis=1)
            E_m_neg = sorted_neg[:, :min(top_k, N)].sum(axis=1)
        else:
            E_m_pos = E_m_pos_ch.sum(axis=1)
            E_m_neg = E_m_neg_ch.sum(axis=1)

        # Store per-moment results
        E_by_moment[m] = (E_m_pos, E_m_neg)

        # Accumulate with factorial-inverse weight
        E_ch_pos += weight * E_m_pos_ch
        E_ch_neg += weight * E_m_neg_ch

    # Aggregate across channels
    if aggregation_mode == "sum_all":
        E_pos = E_ch_pos.sum(axis=1)
        E_neg = E_ch_neg.sum(axis=1)
    elif aggregation_mode == "max":
        E_pos = E_ch_pos.max(axis=1)
        E_neg = E_ch_neg.max(axis=1)
    elif aggregation_mode == "topK":
        sorted_pos = np.sort(E_ch_pos, axis=1)[:, ::-1]
        sorted_neg = np.sort(E_ch_neg, axis=1)[:, ::-1]
        E_pos = sorted_pos[:, :min(top_k, N)].sum(axis=1)
        E_neg = sorted_neg[:, :min(top_k, N)].sum(axis=1)
    else:
        E_pos = E_ch_pos.sum(axis=1)
        E_neg = E_ch_neg.sum(axis=1)

    E_signed = E_pos - E_neg
    edge_margin = 0

    return E_pos, E_neg, E_signed, E_ch_pos, E_ch_neg, E_by_moment, edge_margin


# =============================================================================
# STEP 2c: Compute Energy Derivative and Forgetting Severity
# =============================================================================

def compute_energy_derivative(
    E: np.ndarray,
    smooth_window: int = 7,
) -> np.ndarray:
    """
    Compute smoothed derivative of energy for transition detection.

    In a sustained high-variance regime, E stays high but dE ≈ 0,
    so we don't re-trigger repeatedly. At boundaries, dE spikes,
    giving localized change points.

    Parameters
    ----------
    E : np.ndarray
        Energy time series [T]
    smooth_window : int
        Moving average window for smoothing before differentiation

    Returns
    -------
    dE : np.ndarray
        Smoothed derivative [T]
    """
    # Smooth the energy
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        E_smooth = np.convolve(E, kernel, mode='same')
    else:
        E_smooth = E

    # Compute derivative (first difference)
    dE = np.diff(E_smooth, prepend=E_smooth[0])

    return dE


def compute_forgetting_severity(
    dE: np.ndarray,
    decay: float = 0.98,
    mode: Literal["leaky", "rolling"] = "leaky",
    rolling_window: int = 50,
) -> np.ndarray:
    """
    Compute forgetting severity trace (replaces cumsum).

    Instead of cumulative energy that only grows, this provides
    a severity measure that decays over time, focusing attention
    on recent changes.

    Parameters
    ----------
    dE : np.ndarray
        Energy derivative [T]
    decay : float
        Memory decay factor for leaky integrator (0.9-0.99 typical)
    mode : str
        "leaky" for exponential forgetting, "rolling" for finite window
    rolling_window : int
        Window size for rolling mode

    Returns
    -------
    severity : np.ndarray
        Forgetting severity trace [T]
    """
    T = len(dE)
    abs_dE = np.abs(dE)

    if mode == "leaky":
        # Leaky integrator: sev[t] = decay * sev[t-1] + |dE[t]|
        severity = np.zeros(T)
        for t in range(T):
            if t == 0:
                severity[t] = abs_dE[t]
            else:
                severity[t] = decay * severity[t - 1] + abs_dE[t]
    else:  # rolling
        # Rolling sum (finite memory)
        kernel = np.ones(rolling_window)
        severity = np.convolve(abs_dE, kernel, mode='same')

    return severity


# =============================================================================
# STEP 3: Peak-Based Event Extraction on Derivative (v1-D Fix)
# =============================================================================

def extract_derivative_peaks(
    dE: np.ndarray,
    eps_on_pos: float,
    eps_on_neg: float,
    refractory_period: int = 120,
    edge_margin: int = 0,
    prominence_ratio: float = 0.3,
) -> Tuple[List[int], List[int]]:
    """
    Extract change points as PEAKS in dE using scipy.signal.find_peaks.

    This is the correct approach for derivative-based detection because:
    - dE produces spike-like peaks at transitions, NOT sustained plateaus
    - Hysteresis logic (m_on/r_off/min_event_len) is wrong for spikes
    - find_peaks + distance is the natural way to detect spikes

    Parameters
    ----------
    dE : np.ndarray
        Smoothed derivative of energy [T]
    eps_on_pos : float
        Height threshold for positive dE peaks (onset detection)
    eps_on_neg : float
        Height threshold for negative dE peaks (offset detection)
    refractory_period : int
        Minimum distance between peaks (enforces separation)
    edge_margin : int
        Samples to ignore at edges
    prominence_ratio : float
        If > 0, require prominence = ratio * threshold (robustness to drift)

    Returns
    -------
    onset_points : List[int]
        Indices of onset peaks (positive dE spikes)
    offset_points : List[int]
        Indices of offset peaks (negative dE spikes)
    """
    T = len(dE)
    lo = edge_margin
    hi = T - edge_margin

    prom_pos = eps_on_pos * prominence_ratio
    prom_neg = eps_on_neg * prominence_ratio

    pos, _ = find_peaks(
        dE,
        height=eps_on_pos,
        distance=refractory_period,
        prominence=prom_pos if prom_pos > 0 else None,
    )
    neg, _ = find_peaks(
        -dE,
        height=eps_on_neg,
        distance=refractory_period,
        prominence=prom_neg if prom_neg > 0 else None,
    )

    pos = [int(t) for t in pos if lo <= t < hi]
    neg = [int(t) for t in neg if lo <= t < hi]

    return pos, neg


def _robust_scale(x: np.ndarray) -> float:
    """Robust sigma via MAD (avoid tiny sigma)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return max(float(sigma), 1e-3)


def _merge_nearby_change_points(change_points: List[int], min_distance: int = 200) -> List[int]:
    """Merge change points within min_distance samples into a single midpoint detection."""
    if not change_points:
        return []
    sorted_cps = sorted(change_points)
    merged = [sorted_cps[0]]
    for cp in sorted_cps[1:]:
        if cp - merged[-1] < min_distance:
            merged[-1] = (merged[-1] + cp) // 2  # collapse to midpoint
        else:
            merged.append(cp)
    return merged


def filter_peaks_by_step(
    peaks: List[int],
    E_signed: np.ndarray,
    baseline_idx: np.ndarray,
    direction: str,
    post_win: int = 100,
    gap: int = 10,
    delta_k: float = 1.5,
    edge_margin: int = 0,
) -> List[int]:
    """
    Keep peak tau only if there is a sustained level shift in E_signed.

    This compares the post-peak region against the BASELINE (not pre-peak),
    which handles cases where dE peaks are shifted from true change points
    in causal data.

    - onset: median(post) - median(baseline) > +threshold
    - offset: median(post) - median(baseline) < -threshold

    Parameters
    ----------
    peaks : List[int]
        Candidate peak indices from extract_derivative_peaks
    E_signed : np.ndarray
        Signed energy time series [T]
    baseline_idx : np.ndarray
        Baseline indices (stationary reference)
    direction : str
        "onset" (expect positive step) or "offset" (expect negative step)
    post_win : int
        Window size after peak for median calculation
    gap : int
        Gap after peak to exclude from window (avoid transition itself)
    delta_k : float
        Threshold = delta_k * baseline_scale
    edge_margin : int
        Samples to exclude at edges

    Returns
    -------
    List[int]
        Filtered peak indices that pass step validation
    """
    if len(peaks) == 0:
        return []

    # Compute baseline statistics
    base_E = E_signed[baseline_idx]
    baseline_median = np.median(base_E)
    scale = _robust_scale(base_E)
    thr = delta_k * scale

    T = len(E_signed)
    kept = []

    for tau in peaks:
        # Check if we have enough room for post window
        if tau > T - edge_margin - post_win - gap - 1:
            continue
        if tau < edge_margin:
            continue

        # Compute median after the peak
        post = E_signed[tau + gap : tau + gap + post_win]

        if len(post) == 0:
            continue

        # Compare post-peak region against baseline
        delta = np.median(post) - baseline_median

        # Keep only if delta matches expected direction and exceeds threshold
        if direction == "onset":
            if delta > thr:
                kept.append(tau)
        else:  # offset
            if delta < -thr:
                kept.append(tau)

    return kept

def filter_peaks_by_local_step_sequential(
    onset_peaks: List[int],
    offset_peaks: List[int],
    E_signed: np.ndarray,
    baseline_idx: np.ndarray,
    *,
    pre_win: int = 160,
    post_win: int = 160,
    gap: int = 10,
    delta_k: float = 2.5,
    edge_margin: int = 0,
    min_level_change: float = 0.0,   # optional absolute floor
) -> Tuple[List[int], List[int]]:
    """
    Accept a peak only if it produces a local step:
      onset:  post - pre  > delta_k * sigma_pre
      offset: pre - post  > delta_k * sigma_pre

    Also sequentially updates the current regime level so we don't accept
    multiple peaks inside the same regime.
    """
    T = len(E_signed)

    # current reference level = baseline median
    baseline_vals = E_signed[baseline_idx]
    current_level = float(np.median(baseline_vals))

    cands = [(t, "onset") for t in onset_peaks] + [(t, "offset") for t in offset_peaks]
    cands.sort(key=lambda z: z[0])

    kept_on, kept_off = [], []

    for tau, typ in cands:
        # bounds check for pre/post windows
        if tau - gap - pre_win < edge_margin:
            continue
        if tau + gap + post_win >= T - edge_margin:
            continue

        pre = E_signed[tau - gap - pre_win : tau - gap]
        post = E_signed[tau + gap : tau + gap + post_win]

        pre_med = float(np.median(pre))
        post_med = float(np.median(post))
        sigma_pre = _robust_scale(pre)

        step = post_med - pre_med  # positive means up-step

        if typ == "onset":
            ok_local = step > delta_k * sigma_pre
            ok_level = (post_med - current_level) > delta_k * sigma_pre
            if min_level_change > 0:
                ok_local = ok_local and (step > min_level_change)
                ok_level = ok_level and ((post_med - current_level) > min_level_change)
            if ok_local and ok_level:
                kept_on.append(tau)
                current_level = post_med  # update regime level
        else:  # offset
            ok_local = (-step) > delta_k * sigma_pre  # pre - post
            ok_level = (current_level - post_med) > delta_k * sigma_pre
            if min_level_change > 0:
                ok_local = ok_local and ((-step) > min_level_change)
                ok_level = ok_level and ((current_level - post_med) > min_level_change)
            if ok_local and ok_level:
                kept_off.append(tau)
                current_level = post_med  # update regime level

    return kept_on, kept_off

def filter_peaks_by_local_step(
    peaks,
    E_signed,
    direction: str,            # "onset" or "offset"
    pre_win: int = 160,
    post_win: int = 160,
    gap: int = 10,
    delta_k: float = 2.0,
    edge_margin: int = 0,
):
    """
    Keep peaks whose *local* mean shift in E_signed is large vs local pre-variance.
    onset: mean(post) - mean(pre) >= delta_k * std(pre)
    offset: mean(post) - mean(pre) <= -delta_k * std(pre)
    """
    T = len(E_signed)
    kept = []

    for tau in sorted(peaks):
        if tau < edge_margin + pre_win + gap:
            continue
        if tau > T - edge_margin - post_win - gap - 1:
            continue

        pre = E_signed[tau - gap - pre_win : tau - gap]
        post = E_signed[tau + gap : tau + gap + post_win]

        mu_pre = float(np.mean(pre))
        mu_post = float(np.mean(post))
        delta = mu_post - mu_pre

        sigma = float(np.std(pre))
        sigma = max(sigma, 1e-6)

        if direction == "onset":
            if delta >= delta_k * sigma:
                kept.append(int(tau))
        else:  # "offset"
            if delta <= -delta_k * sigma:
                kept.append(int(tau))

    return kept


# =============================================================================
# STEP 6c: K-of-Channels Consistency Gate (v1-F)
# =============================================================================

def filter_peaks_by_channel_consistency(
    peaks: List[int],
    E_ch_pos: np.ndarray,
    E_ch_neg: np.ndarray,
    direction: str,
    k_channels_min: int = 2,
    delta_ch_k: float = 1.5,
    pre_win: int = 160,
    post_win: int = 160,
    gap: int = 10,
    edge_margin: int = 0,
) -> Tuple[List[int], List[dict]]:
    """
    K-of-channels gate: keep peak only if >= k_channels_min channels
    show a significant local step in per-channel signed energy.

    For each candidate peak tau:
      1. Compute per-channel step: Delta_ch = median(post) - median(pre)
      2. Normalize: z_ch = Delta_ch / sigma_ch  (MAD-scaled)
      3. Count active channels matching the expected direction
      4. Gate: keep if n_active >= k_channels_min

    Parameters
    ----------
    peaks : List[int]
        Candidate peak indices (already passed step validation on aggregate)
    E_ch_pos : np.ndarray
        Per-channel up-shift energy [T, N]
    E_ch_neg : np.ndarray
        Per-channel down-shift energy [T, N]
    direction : str
        "onset" (expect positive step) or "offset" (expect negative step)
    k_channels_min : int
        Minimum number of channels that must show a significant step.
        Clamped internally to max(1, N // 2) for small N.
    delta_ch_k : float
        Per-channel z-score threshold (in MAD-scaled units)
    pre_win : int
        Window size before peak
    post_win : int
        Window size after peak
    gap : int
        Gap around peak to exclude transition
    edge_margin : int
        Samples to exclude at edges

    Returns
    -------
    kept_peaks : List[int]
        Peaks that pass the channel consistency gate
    channel_info : List[dict]
        Diagnostic info per kept peak: {n_active, frac_active, z_scores, concentration_ratio}
    """
    if len(peaks) == 0:
        return [], []

    T, N = E_ch_pos.shape

    # Clamp k_channels_min for small N
    k_eff = min(k_channels_min, max(1, N // 2))

    # Auto-pass for single channel
    if N <= 1:
        info = [{'n_active': 1, 'frac_active': 1.0,
                 'z_scores': np.array([0.0]), 'concentration_ratio': 1.0}
                for _ in peaks]
        return list(peaks), info

    # Per-channel signed energy
    E_ch_signed = E_ch_pos - E_ch_neg  # [T, N]

    kept = []
    kept_info = []

    for tau in sorted(peaks):
        # Bounds check
        if tau - gap - pre_win < edge_margin:
            continue
        if tau + gap + post_win >= T - edge_margin:
            continue

        z_scores = np.zeros(N)
        deltas = np.zeros(N)

        for ch in range(N):
            pre = E_ch_signed[tau - gap - pre_win : tau - gap, ch]
            post = E_ch_signed[tau + gap : tau + gap + post_win, ch]

            if len(pre) == 0 or len(post) == 0:
                continue

            delta_ch = float(np.median(post) - np.median(pre))
            sigma_ch = _robust_scale(pre)

            z_scores[ch] = delta_ch / max(sigma_ch, 1e-6)
            deltas[ch] = delta_ch

        # Count active channels matching expected direction
        if direction == "onset":
            n_active = int(np.sum(z_scores > delta_ch_k))
        else:
            n_active = int(np.sum(z_scores < -delta_ch_k))

        # Concentration ratio: max|Δ| / Σ|Δ|
        abs_deltas = np.abs(deltas)
        total_delta = abs_deltas.sum()
        concentration = float(abs_deltas.max() / total_delta) if total_delta > 1e-10 else 1.0

        if n_active >= k_eff:
            kept.append(int(tau))
            kept_info.append({
                'n_active': n_active,
                'frac_active': n_active / N,
                'z_scores': z_scores.copy(),
                'concentration_ratio': concentration,
            })

    return kept, kept_info


# Legacy function for backwards compatibility (deprecated)
def extract_derivative_events(
    dE: np.ndarray,
    eps_on_pos: float,
    eps_on_neg: float,
    eps_off_ratio: float = 0.3,
    m_on: int = 3,
    r_off: int = 3,
    edge_margin: int = 0,
    min_event_len: int = 10,
    refractory_period: int = 20,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    DEPRECATED: Use extract_derivative_peaks() instead.

    This hysteresis-based extraction is incompatible with spike-like dE signals.
    Kept for backwards compatibility only.
    """
    # Use peak-based extraction and wrap results in event tuples
    onset_points, offset_points = extract_derivative_peaks(
        dE, eps_on_pos, eps_on_neg, refractory_period, edge_margin
    )

    # Build (t_start, t_end, tau) tuples with small windows around peaks
    w = 3  # Small window around peak
    T = len(dE)

    pos_events = []
    for tau in onset_points:
        t_start = max(0, tau - w)
        t_end = min(T - 1, tau + w)
        pos_events.append((t_start, t_end, tau))

    neg_events = []
    for tau in offset_points:
        t_start = max(0, tau - w)
        t_end = min(T - 1, tau + w)
        neg_events.append((t_start, t_end, tau))

    return pos_events, neg_events


# =============================================================================
# STEP 4: Event Reporting (Vectorized)
# =============================================================================

def build_event_report(
    t_start: int,
    t_end: int,
    tau: int,
    event_type: str,
    dE: np.ndarray,
    E_signed: np.ndarray,
    E_ch_pos: np.ndarray,
    E_ch_neg: np.ndarray,
    scales: np.ndarray,
) -> Event:
    """
    Build detailed event report for derivative-based detection.

    Parameters
    ----------
    t_start, t_end : int
        Event boundaries
    tau : int
        Change point (argmax/argmin of dE within event)
    event_type : str
        "onset" or "offset"
    dE : np.ndarray
        Energy derivative [T]
    E_signed : np.ndarray
        Signed energy [T]
    E_ch_pos, E_ch_neg : np.ndarray
        Per-channel energies [T, N]
    scales : np.ndarray
        CWT scales
    """
    M = len(scales)

    # Use appropriate channel scores based on event type
    if event_type == "onset":
        E_ch = E_ch_pos
    else:
        E_ch = E_ch_neg

    # Channel attribution (vectorized)
    channel_scores = E_ch[t_start:t_end + 1, :].sum(axis=0)
    channels_ranked = list(np.argsort(channel_scores)[::-1])

    # Scale scores placeholder (could be extended to track per-scale contributions)
    scale_scores = np.zeros(M)
    scales_ranked = list(range(M))

    # Peak derivative magnitude within event
    dE_interval = dE[t_start:t_end + 1]
    if event_type == "onset":
        peak_dE = float(np.max(dE_interval))
    else:
        peak_dE = float(np.max(-dE_interval))

    # Peak energy level at tau
    peak_E = float(E_signed[tau]) if 0 <= tau < len(E_signed) else 0.0

    # Severity: total |dE| in event
    area = float(np.sum(np.abs(dE_interval)))

    return Event(
        t_start=t_start,
        t_end=t_end,
        tau=tau,
        event_type=event_type,
        channels_ranked=channels_ranked,
        scales_ranked=scales_ranked,
        peak_dE=peak_dE,
        peak_E=peak_E,
        area=area,
        channel_scores=channel_scores,
        scale_scores=scale_scores,
    )


# =============================================================================
# Per-Moment Vote Diagnostics
# =============================================================================

def compute_per_moment_detections(
    E_by_moment: Dict[int, Tuple[np.ndarray, np.ndarray]],
    baseline_idx: np.ndarray,
    smooth_window: int = 15,
    refractory_period: int = 200,
    edge_margin: int = 0,
    eps_factor: float = 2.0,
    baseline_quantile: float = 0.95,
) -> Tuple[Dict[int, List[int]], Dict[int, np.ndarray]]:
    """
    Run independent peak detection on each moment's signed energy for voting visualization.

    For each moment m, thresholds are derived from the baseline portion of that
    moment's dE so the scale differences across moments are automatically handled.

    Parameters
    ----------
    E_by_moment : Dict[int, Tuple[np.ndarray, np.ndarray]]
        Per-moment channel-aggregated energies {m: (E_pos_m, E_neg_m)}
    baseline_idx : np.ndarray
        Baseline sample indices (used to calibrate per-moment eps)
    smooth_window : int
        Smoothing window for dE computation (must match main pipeline)
    refractory_period : int
        Minimum distance between peaks
    edge_margin : int
        Edge samples to exclude
    eps_factor : float
        Multiplier on baseline dE quantile to set the per-moment threshold
    baseline_quantile : float
        Quantile of |dE_baseline| used as the base threshold

    Returns
    -------
    moment_change_points : Dict[int, List[int]]
        Per-moment detected change points {m: [indices]}
    moment_dE : Dict[int, np.ndarray]
        Per-moment derivative of signed energy {m: dE_m [T]}
    """
    moment_change_points = {}
    moment_dE_out = {}

    for m, (E_m_pos, E_m_neg) in E_by_moment.items():
        E_signed_m = E_m_pos - E_m_neg
        dE_m = compute_energy_derivative(E_signed_m, smooth_window)
        moment_dE_out[m] = dE_m

        # Threshold from baseline distribution of this moment's dE
        dE_base = np.abs(dE_m[baseline_idx])
        eps_m = float(np.quantile(dE_base, baseline_quantile)) * eps_factor
        eps_m = max(eps_m, 1e-4)

        onset_m, offset_m = extract_derivative_peaks(
            dE_m,
            eps_on_pos=eps_m,
            eps_on_neg=eps_m,
            refractory_period=refractory_period,
            edge_margin=edge_margin,
            prominence_ratio=0.2,
        )

        moment_change_points[m] = sorted(set(onset_m + offset_m))

    return moment_change_points, moment_dE_out


# =============================================================================
# Main Detection Function
# =============================================================================

def detect_nonstationarity(
    X: np.ndarray,
    baseline_idx: np.ndarray,
    scales: Optional[np.ndarray] = None,
    min_scale: float = 4.0,
    n_surrogates: int = 100,
    alpha: float = 0.01,           # Stricter than 0.05 (reduces FPs)
    eps_on_pos: Optional[float] = None,
    eps_on_neg: Optional[float] = None,
    k_scales_min: int = 3,         # Require more scales to be significant
    min_snr: float = 2.0,          # SNR threshold (step validation is primary filter)
    aggregation_mode: Literal["sum_all", "max", "topK"] = "sum_all",
    top_k: int = 3,
    omega0: float = 6.0,
    smooth_window: int = 15,       # More smoothing reduces zig-zag peaks
    severity_decay: float = 0.98,
    refractory_period: int = 200,  # Larger distance between peaks
    seed: Optional[int] = None,
    # v1-F: Multichannel consistency gate
    k_channels_min: int = 2,       # Min channels with significant step
    delta_ch_k: float = 1.5,       # Per-channel z-score threshold (MAD-scaled)
    # Step validation threshold (exposed so callers can tune per-scenario)
    step_delta_k: float = 2.5,     # Step validation: keep peaks with step > step_delta_k * sigma_pre
    # Deprecated parameters (kept for backwards compatibility, no longer used)
    eps_off_ratio: float = 0.3,
    m_on: int = 1,
    r_off: int = 1,
    min_event_len: int = 1,
) -> DetectionResult:
    """
    Full Gatekeeper v1-D/F detection pipeline: Signed Deviations + Peak-Based Detection + Channel Gate.

    Key improvements in v1-D (over v1-C):
    - SIGNED wavelet deviations (log-ratio): Z = log(SG/T) for symmetric detection
    - DIFFERENTIAL detection: operates on dE (derivative), not E (level)
    - PEAK-BASED extraction: uses find_peaks on dE (not hysteresis, which fails for spikes)
    - Peak-height calibration: eps thresholds from surrogate peak heights (not max-stat)
    - Separate E_pos/E_neg channels for up-shifts and down-shifts
    - FORGETTING severity trace (leaky integrator), not cumulative sum
    - TWO-SIDED thresholds calibrated on surrogate peak heights

    This fixes the v1-C failure modes:
    - No cascading FPs in sustained high-variance regimes (dE ≈ 0 inside regime)
    - Detects BOTH onset (increase) AND offset (decrease) regime changes
    - Thresholds are not inflated by max-stat (peak-height calibration matches detection)
    - Spike-like dE signals are properly detected (find_peaks, not hysteresis)
    - Severity doesn't accumulate forever; it decays and focuses on recent changes

    Parameters
    ----------
    X : np.ndarray
        Multichannel time series, shape [T, N]
    baseline_idx : np.ndarray
        Indices for baseline segment
    scales : np.ndarray, optional
        CWT scales. If None, auto-generated
    min_scale : float
        Minimum scale to use (drops high-freq noise)
    n_surrogates : int
        Number of Fourier surrogates for calibration
    alpha : float
        Significance level for thresholds (controls FWER)
    eps_on_pos : float, optional
        Threshold for positive dE peaks (onset). If None, uses surrogate-calibrated value
    eps_on_neg : float, optional
        Threshold for negative dE peaks (offset). If None, uses surrogate-calibrated
    k_scales_min : int
        Min scales that must be significant (K-of-scales gate)
    min_snr : float
        Minimum signal-to-noise ratio for derivative events
    aggregation_mode : str
        Channel aggregation: "sum_all", "max", or "topK"
    top_k : int
        Number of top channels for "topK" mode
    omega0 : float
        Morlet central frequency
    smooth_window : int
        Window for smoothing energy before differentiation
    severity_decay : float
        Decay factor for leaky integrator severity (0.95-0.99)
    refractory_period : int
        Minimum gap between peaks (distance parameter for find_peaks)
    seed : int, optional
        Random seed for reproducibility

    Deprecated Parameters (no longer used, kept for backwards compatibility)
    -------------------------------------------------------------------------
    eps_off_ratio, m_on, r_off, min_event_len : deprecated
        These hysteresis parameters were for plateau detection.
        v1-D uses peak detection (find_peaks) which doesn't need them.

    Returns
    -------
    DetectionResult
        Detection result with E_pos, E_neg, E_signed, dE, severity, events, change_points
    """
    # Silence warnings about deprecated params
    _ = eps_off_ratio, m_on, r_off, min_event_len
    T = X.shape[0]

    # Default scales (drop very small scales to reduce FPs)
    if scales is None:
        max_scale = min(64, max(T // 20, 8))
        n_scales = min(20, max(5, int(np.log2(max_scale / min_scale) * 4)))
        scales = np.logspace(np.log2(min_scale), np.log2(max_scale), n_scales, base=2)
    else:
        # Filter out scales below min_scale
        scales = scales[scales >= min_scale]

    # STEP 1: Calibration with MAX-STAT + derivative-based eps thresholds
    X_baseline = X[baseline_idx, :]
    calibration = calibrate_thresholds_maxstat(
        X_baseline, scales, n_surrogates, alpha, omega0, smooth_window, seed
    )

    # STEP 2: Compute SIGNED energy (log-ratio deviation) + K-of-scales gate
    E_pos, E_neg, E_signed, E_ch_pos, E_ch_neg, _ = compute_signed_energy(
        X, scales, calibration.thresholds, omega0, aggregation_mode, top_k, k_scales_min
    )

    # STEP 3: Compute energy DERIVATIVE (for differential detection)
    dE = compute_energy_derivative(E_signed, smooth_window)

    # Compute edge margin to exclude CWT boundary artifacts from peak detection
    edge_margin = max(10, int(np.max(scales)))

    # STEP 4: Compute FORGETTING severity (leaky integrator, not cumsum)
    severity = compute_forgetting_severity(dE, decay=severity_decay, mode="leaky")

    # STEP 5: Use surrogate-calibrated thresholds (unless user overrides)
    if eps_on_pos is None:
        eps_on_pos = calibration.eps_on_pos
    if eps_on_neg is None:
        eps_on_neg = calibration.eps_on_neg

    # STEP 6: Extract change points using PEAK DETECTION on dE
    # v1-D FIX: Use find_peaks with prominence + distance (dE is spike-like)
    onset_points_raw, offset_points_raw = extract_derivative_peaks(
        dE,
        eps_on_pos=eps_on_pos,
        eps_on_neg=eps_on_neg,
        refractory_period=refractory_period,
        edge_margin=edge_margin,
        prominence_ratio=0.3,
    )

    # STEP 6b: Apply STEP VALIDATION to filter out transient spikes
    onset_points_raw = filter_peaks_by_local_step(
        onset_points_raw, E_signed, "onset",
        pre_win=160, post_win=160, gap=10, delta_k=step_delta_k, edge_margin=edge_margin
    )
    offset_points_raw = filter_peaks_by_local_step(
        offset_points_raw, E_signed, "offset",
        pre_win=160, post_win=160, gap=10, delta_k=step_delta_k, edge_margin=edge_margin
    )

    # STEP 6c: K-of-channels consistency gate (v1-F)
    # Reject peaks where too few channels show a significant step
    if X.shape[1] > 1 and k_channels_min > 0:
        onset_points_raw, onset_ch_info = filter_peaks_by_channel_consistency(
            onset_points_raw, E_ch_pos, E_ch_neg, "onset",
            k_channels_min=k_channels_min, delta_ch_k=delta_ch_k,
            pre_win=160, post_win=160, gap=10, edge_margin=edge_margin,
        )
        offset_points_raw, offset_ch_info = filter_peaks_by_channel_consistency(
            offset_points_raw, E_ch_pos, E_ch_neg, "offset",
            k_channels_min=k_channels_min, delta_ch_k=delta_ch_k,
            pre_win=160, post_win=160, gap=10, edge_margin=edge_margin,
        )
    else:
        onset_ch_info = [None] * len(onset_points_raw)
        offset_ch_info = [None] * len(offset_points_raw)

    # STEP 7: Build event reports with SNR filtering
    # SNR is computed on dE relative to baseline dE statistics
    # Exclude edge samples from baseline to avoid CWT boundary artifacts
    baseline_interior_mask = (baseline_idx >= edge_margin) & (baseline_idx < T - edge_margin)
    baseline_dE_interior = dE[baseline_idx[baseline_interior_mask]] if np.any(baseline_interior_mask) else dE[baseline_idx]
    baseline_dE_std = np.std(baseline_dE_interior) if len(baseline_dE_interior) > 0 else 1.0
    baseline_dE_std = max(baseline_dE_std, 0.01)  # Avoid division by zero

    # Small window around peaks for event reporting
    T = len(dE)
    w = max(3, smooth_window)

    events = []
    onset_points = []
    offset_points = []

    # Process onset events (positive dE peaks)
    for i, tau in enumerate(onset_points_raw):
        t_start = max(0, tau - w)
        t_end = min(T - 1, tau + w)

        event = build_event_report(
            t_start, t_end, tau, "onset",
            dE, E_signed, E_ch_pos, E_ch_neg, scales
        )

        # SNR filter on derivative peak
        if min_snr > 0:
            event_snr = event.peak_dE / baseline_dE_std
            if event_snr < min_snr:
                continue

        # Attach channel consistency diagnostics (v1-F)
        if i < len(onset_ch_info) and onset_ch_info[i] is not None:
            event.n_active_channels = onset_ch_info[i]['n_active']
            event.frac_active_channels = onset_ch_info[i]['frac_active']
            event.channel_z_scores = onset_ch_info[i]['z_scores']
            event.concentration_ratio = onset_ch_info[i]['concentration_ratio']

        events.append(event)
        onset_points.append(tau)

    # Process offset events (negative dE peaks)
    for i, tau in enumerate(offset_points_raw):
        t_start = max(0, tau - w)
        t_end = min(T - 1, tau + w)

        event = build_event_report(
            t_start, t_end, tau, "offset",
            dE, E_signed, E_ch_pos, E_ch_neg, scales
        )

        # SNR filter on derivative peak
        if min_snr > 0:
            event_snr = event.peak_dE / baseline_dE_std
            if event_snr < min_snr:
                continue

        # Attach channel consistency diagnostics (v1-F)
        if i < len(offset_ch_info) and offset_ch_info[i] is not None:
            event.n_active_channels = offset_ch_info[i]['n_active']
            event.frac_active_channels = offset_ch_info[i]['frac_active']
            event.channel_z_scores = offset_ch_info[i]['z_scores']
            event.concentration_ratio = offset_ch_info[i]['concentration_ratio']

        events.append(event)
        offset_points.append(tau)

    # Combine all change points; merge any within 200 samples (single event)
    change_points = _merge_nearby_change_points(sorted(set(onset_points + offset_points)))
    is_nonstationary = len(events) > 0

    return DetectionResult(
        events=events,
        E_pos=E_pos,
        E_neg=E_neg,
        E_signed=E_signed,
        dE=dE,
        severity=severity,
        E_ch_pos=E_ch_pos,
        E_ch_neg=E_ch_neg,
        calibration=calibration,
        is_nonstationary=is_nonstationary,
        change_points=change_points,
        onset_points=onset_points,
        offset_points=offset_points,
        edge_margin=edge_margin,
    )


# =============================================================================
# Multi-Moment Detection Function (v1-E)
# =============================================================================

def detect_nonstationarity_multimoment(
    X: np.ndarray,
    baseline_idx: np.ndarray,
    scales: Optional[np.ndarray] = None,
    min_scale: float = 4.0,
    moments: List[int] = [1, 2, 3, 4],
    moment_window: int = 50,
    n_surrogates: int = 100,
    alpha: float = 0.01,
    eps_on_pos: Optional[float] = None,
    eps_on_neg: Optional[float] = None,
    k_scales_min: int = 3,
    min_snr: float = 2.0,
    aggregation_mode: Literal["sum_all", "max", "topK"] = "sum_all",
    top_k: int = 3,
    omega0: float = 6.0,
    smooth_window: int = 15,
    severity_decay: float = 0.98,
    refractory_period: int = 200,
    seed: Optional[int] = None,
    # v1-F: Multichannel consistency gate
    k_channels_min: int = 2,       # Min channels with significant step
    delta_ch_k: float = 1.5,       # Per-channel z-score threshold (MAD-scaled)
    # Step validation threshold (lower = more sensitive, less precise)
    step_delta_k: float = 2.5,     # Step validation: keep peaks with step > step_delta_k * sigma_pre
) -> DetectionResult:
    """
    Full Gatekeeper v1-F detection pipeline: Multi-Moment Analysis + Multichannel Consistency Gate.

    This extends v1-E by adding a K-of-channels consistency gate that rejects
    peaks where too few channels show a significant local step, reducing false
    positives caused by single-channel noise artifacts.

    v1-E base: analyzes changes in the first 4 statistical moments:
    - Mean (1st moment): weight = 1/1! = 1.0
    - Variance (2nd moment): weight = 1/2! = 0.5
    - Skewness (3rd moment): weight = 1/3! ≈ 0.167
    - Kurtosis (4th moment): weight = 1/4! ≈ 0.042

    The factorial-inverse weighting emphasizes lower moments (which are more
    reliable/stable) while still incorporating higher moments (which can
    capture distributional changes not visible in mean/variance).

    Workflow:
    1. For each channel, compute rolling moments (mean, var, skew, kurt)
    2. For each moment, apply CWT and compute signed deviations
    3. Aggregate across moments using factorial-inverse weights
    4. Compute dE (derivative) and detect peaks
    5. Apply step validation and SNR filtering

    Parameters
    ----------
    X : np.ndarray
        Multichannel time series, shape [T, N]
    baseline_idx : np.ndarray
        Indices for baseline segment
    scales : np.ndarray, optional
        CWT scales. If None, auto-generated
    min_scale : float
        Minimum scale to use
    moments : List[int]
        Which moments to use (1=mean, 2=var, 3=skew, 4=kurt)
    moment_window : int
        Window size for rolling moment computation
    n_surrogates : int
        Number of Fourier surrogates for calibration
    alpha : float
        Significance level for thresholds
    eps_on_pos : float, optional
        Threshold for positive dE peaks. If None, uses surrogate-calibrated
    eps_on_neg : float, optional
        Threshold for negative dE peaks. If None, uses surrogate-calibrated
    k_scales_min : int
        Min scales that must be significant
    min_snr : float
        Minimum signal-to-noise ratio for events
    aggregation_mode : str
        Channel aggregation: "sum_all", "max", or "topK"
    top_k : int
        Number of top channels for "topK" mode
    omega0 : float
        Morlet central frequency
    smooth_window : int
        Window for smoothing energy before differentiation
    severity_decay : float
        Decay factor for leaky integrator severity
    refractory_period : int
        Minimum gap between peaks
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    DetectionResult
        Detection result with multi-moment energies and change points
    """
    T, N = X.shape

    # Default scales
    if scales is None:
        max_scale = min(64, max(T // 20, 8))
        n_scales = min(20, max(5, int(np.log2(max_scale / min_scale) * 4)))
        scales = np.logspace(np.log2(min_scale), np.log2(max_scale), n_scales, base=2)
    else:
        scales = scales[scales >= min_scale]

    # STEP 1: Multi-moment calibration
    X_baseline = X[baseline_idx, :]
    calibration = calibrate_multi_moment_thresholds(
        X_baseline, scales, moments, moment_window,
        n_surrogates, alpha, omega0, smooth_window, seed
    )

    # STEP 2: Compute multi-moment weighted energy
    E_pos, E_neg, E_signed, E_ch_pos, E_ch_neg, E_by_moment, _ = compute_multi_moment_energy(
        X, scales, calibration.moment_thresholds, moment_window,
        moments, omega0, aggregation_mode, top_k, k_scales_min
    )

    # STEP 3: Compute energy derivative
    dE = compute_energy_derivative(E_signed, smooth_window)

    # Edge margin for peak detection
    edge_margin = max(10, int(np.max(scales)))

    # STEP 4: Compute forgetting severity
    severity = compute_forgetting_severity(dE, decay=severity_decay, mode="leaky")

    # STEP 5: Use surrogate-calibrated thresholds
    if eps_on_pos is None:
        eps_on_pos = calibration.eps_on_pos
    if eps_on_neg is None:
        eps_on_neg = calibration.eps_on_neg

    # STEP 6: Extract peaks
    onset_points_raw, offset_points_raw = extract_derivative_peaks(
        dE,
        eps_on_pos=eps_on_pos,
        eps_on_neg=eps_on_neg,
        refractory_period=refractory_period,
        edge_margin=edge_margin,
        prominence_ratio=0.3,
    )

    # STEP 6b: Step validation
    onset_points_raw = filter_peaks_by_local_step(
        onset_points_raw, E_signed, "onset",
        pre_win=160, post_win=160, gap=10, delta_k=step_delta_k, edge_margin=edge_margin
    )
    offset_points_raw = filter_peaks_by_local_step(
        offset_points_raw, E_signed, "offset",
        pre_win=160, post_win=160, gap=10, delta_k=step_delta_k, edge_margin=edge_margin
    )

    # STEP 6c: K-of-channels consistency gate (v1-F)
    if X.shape[1] > 1 and k_channels_min > 0:
        onset_points_raw, onset_ch_info = filter_peaks_by_channel_consistency(
            onset_points_raw, E_ch_pos, E_ch_neg, "onset",
            k_channels_min=k_channels_min, delta_ch_k=delta_ch_k,
            pre_win=160, post_win=160, gap=10, edge_margin=edge_margin,
        )
        offset_points_raw, offset_ch_info = filter_peaks_by_channel_consistency(
            offset_points_raw, E_ch_pos, E_ch_neg, "offset",
            k_channels_min=k_channels_min, delta_ch_k=delta_ch_k,
            pre_win=160, post_win=160, gap=10, edge_margin=edge_margin,
        )
    else:
        onset_ch_info = [None] * len(onset_points_raw)
        offset_ch_info = [None] * len(offset_points_raw)

    # STEP 7: Build events with SNR filtering
    baseline_interior_mask = (baseline_idx >= edge_margin) & (baseline_idx < T - edge_margin)
    baseline_dE_interior = dE[baseline_idx[baseline_interior_mask]] if np.any(baseline_interior_mask) else dE[baseline_idx]
    baseline_dE_std = np.std(baseline_dE_interior) if len(baseline_dE_interior) > 0 else 1.0
    baseline_dE_std = max(baseline_dE_std, 0.01)

    w = max(3, smooth_window)

    events = []
    onset_points = []
    offset_points = []

    for i, tau in enumerate(onset_points_raw):
        t_start = max(0, tau - w)
        t_end = min(T - 1, tau + w)

        event = build_event_report(
            t_start, t_end, tau, "onset",
            dE, E_signed, E_ch_pos, E_ch_neg, scales
        )

        if min_snr > 0:
            event_snr = event.peak_dE / baseline_dE_std
            if event_snr < min_snr:
                continue

        # Attach channel consistency diagnostics (v1-F)
        if i < len(onset_ch_info) and onset_ch_info[i] is not None:
            event.n_active_channels = onset_ch_info[i]['n_active']
            event.frac_active_channels = onset_ch_info[i]['frac_active']
            event.channel_z_scores = onset_ch_info[i]['z_scores']
            event.concentration_ratio = onset_ch_info[i]['concentration_ratio']

        events.append(event)
        onset_points.append(tau)

    for i, tau in enumerate(offset_points_raw):
        t_start = max(0, tau - w)
        t_end = min(T - 1, tau + w)

        event = build_event_report(
            t_start, t_end, tau, "offset",
            dE, E_signed, E_ch_pos, E_ch_neg, scales
        )

        if min_snr > 0:
            event_snr = event.peak_dE / baseline_dE_std
            if event_snr < min_snr:
                continue

        # Attach channel consistency diagnostics (v1-F)
        if i < len(offset_ch_info) and offset_ch_info[i] is not None:
            event.n_active_channels = offset_ch_info[i]['n_active']
            event.frac_active_channels = offset_ch_info[i]['frac_active']
            event.channel_z_scores = offset_ch_info[i]['z_scores']
            event.concentration_ratio = offset_ch_info[i]['concentration_ratio']

        events.append(event)
        offset_points.append(tau)

    # Merge any change points within 200 samples (single event)
    change_points = _merge_nearby_change_points(sorted(set(onset_points + offset_points)))
    is_nonstationary = len(events) > 0

    moment_weights = get_moment_weights(moments)

    # Per-moment voting: independent detection on each moment's dE
    mom_cps, mom_dE = compute_per_moment_detections(
        E_by_moment, baseline_idx,
        smooth_window=smooth_window,
        refractory_period=refractory_period,
        edge_margin=edge_margin,
    )

    return DetectionResult(
        events=events,
        E_pos=E_pos,
        E_neg=E_neg,
        E_signed=E_signed,
        dE=dE,
        severity=severity,
        E_ch_pos=E_ch_pos,
        E_ch_neg=E_ch_neg,
        calibration=calibration,
        is_nonstationary=is_nonstationary,
        change_points=change_points,
        onset_points=onset_points,
        offset_points=offset_points,
        edge_margin=edge_margin,
        E_by_moment=E_by_moment,
        moments_used=moments,
        moment_weights=moment_weights,
        moment_change_points=mom_cps,
        moment_dE=mom_dE,
    )


# =============================================================================
# v1-G: Adaptive Step Validation + Two-Pass Detection
# =============================================================================
#
# Addresses the primary recall bottleneck identified in batch experiments:
# step validation (filter_peaks_by_local_step) kills 88% of raw peaks,
# including many true change points.
#
# Three improvements over v1-F:
#   1. ADAPTIVE step threshold: strong dE peaks require less step evidence.
#      delta_k_eff = delta_k * (1 - discount), where discount grows with
#      peak prominence relative to the eps threshold.
#   2. SHORTER pre/post windows (100 vs 160): reduces dead zones near edges
#      from 234 to 174 samples, and better suits 600-800 sample regimes.
#   3. TWO-PASS detection: first pass with standard thresholds (high precision),
#      second pass with relaxed step validation targeting suspiciously long
#      gaps between first-pass detections (likely missed CPs).

def filter_peaks_by_local_step_adaptive(
    peaks: List[int],
    E_signed: np.ndarray,
    dE: np.ndarray,
    eps_threshold: float,
    direction: str,
    pre_win: int = 100,
    post_win: int = 100,
    gap: int = 10,
    delta_k: float = 1.5,
    edge_margin: int = 0,
    max_discount: float = 0.6,
) -> List[int]:
    """
    Adaptive step validation (v1-G): scale delta_k inversely with peak strength.

    Strong dE peaks (well above eps) are likely real transitions even if the
    sustained energy shift is modest. Weak peaks still need full step evidence.

    peak_strength = |dE[tau]| / eps_threshold   (always > 1 since peak passed height filter)

    discount = min(max_discount, 0.2 * (peak_strength - 1))
    delta_k_eff = delta_k * (1 - discount)

    A peak 4x threshold → delta_k reduced by 60% (1.5 → 0.6)
    A peak 2x threshold → delta_k reduced by 20% (1.5 → 1.2)
    A peak barely above  → no reduction
    """
    T = len(E_signed)
    kept = []

    for tau in sorted(peaks):
        if tau < edge_margin + pre_win + gap:
            continue
        if tau > T - edge_margin - post_win - gap - 1:
            continue

        pre = E_signed[tau - gap - pre_win : tau - gap]
        post = E_signed[tau + gap : tau + gap + post_win]

        mu_pre = float(np.mean(pre))
        mu_post = float(np.mean(post))
        delta = mu_post - mu_pre

        sigma = float(np.std(pre))
        sigma = max(sigma, 1e-6)

        peak_dE = abs(float(dE[tau]))
        peak_strength = peak_dE / max(eps_threshold, 1e-6)
        discount = min(max_discount, 0.2 * max(0, peak_strength - 1.0))
        delta_k_eff = delta_k * (1.0 - discount)

        if direction == "onset":
            if delta >= delta_k_eff * sigma:
                kept.append(int(tau))
        else:
            if delta <= -delta_k_eff * sigma:
                kept.append(int(tau))

    return kept


def _second_pass_peaks(
    dE: np.ndarray,
    E_signed: np.ndarray,
    E_ch_pos: np.ndarray,
    E_ch_neg: np.ndarray,
    first_pass_cps: List[int],
    T: int,
    eps_on_pos: float,
    eps_on_neg: float,
    refractory_period: int,
    edge_margin: int,
    step_delta_k: float,
    k_channels_min: int,
    delta_ch_k: float,
    min_gap_ratio: float = 2.0,
    mean_regime_len: float = 700.0,
) -> Tuple[List[int], List[int]]:
    """
    Two-pass rescue: find missed CPs in suspiciously long gaps.

    After the first pass, gaps between consecutive CPs longer than
    min_gap_ratio * mean_regime_len likely contain a missed CP. This
    function re-scans those gaps with relaxed thresholds.
    """
    if not first_pass_cps:
        boundaries = [(0, T)]
    else:
        boundaries = []
        all_bounds = [0] + sorted(first_pass_cps) + [T]
        for i in range(len(all_bounds) - 1):
            boundaries.append((all_bounds[i], all_bounds[i + 1]))

    min_gap = min_gap_ratio * mean_regime_len
    rescued_on = []
    rescued_off = []

    for seg_start, seg_end in boundaries:
        seg_len = seg_end - seg_start
        if seg_len < min_gap:
            continue

        dE_seg = dE[seg_start:seg_end]
        E_seg = E_signed[seg_start:seg_end]

        relaxed_eps_pos = eps_on_pos * 0.6
        relaxed_eps_neg = eps_on_neg * 0.6

        local_on, local_off = extract_derivative_peaks(
            dE_seg,
            eps_on_pos=relaxed_eps_pos,
            eps_on_neg=relaxed_eps_neg,
            refractory_period=refractory_period,
            edge_margin=max(10, edge_margin - seg_start) if seg_start < edge_margin else 0,
            prominence_ratio=0.2,
        )

        # Map back to global indices
        local_on = [t + seg_start for t in local_on]
        local_off = [t + seg_start for t in local_off]

        # Adaptive step validation with relaxed delta_k
        relaxed_delta_k = step_delta_k * 0.6
        local_on = filter_peaks_by_local_step_adaptive(
            local_on, E_signed, dE, relaxed_eps_pos, "onset",
            pre_win=100, post_win=100, gap=10, delta_k=relaxed_delta_k,
            edge_margin=edge_margin, max_discount=0.7)
        local_off = filter_peaks_by_local_step_adaptive(
            local_off, E_signed, dE, relaxed_eps_neg, "offset",
            pre_win=100, post_win=100, gap=10, delta_k=relaxed_delta_k,
            edge_margin=edge_margin, max_discount=0.7)

        # Channel gate with relaxed k
        N = E_ch_pos.shape[1]
        if N > 1 and k_channels_min > 0:
            relaxed_k = max(1, k_channels_min - 1)
            local_on, _ = filter_peaks_by_channel_consistency(
                local_on, E_ch_pos, E_ch_neg, "onset",
                k_channels_min=relaxed_k, delta_ch_k=delta_ch_k * 0.8,
                pre_win=100, post_win=100, gap=10, edge_margin=edge_margin)
            local_off, _ = filter_peaks_by_channel_consistency(
                local_off, E_ch_pos, E_ch_neg, "offset",
                k_channels_min=relaxed_k, delta_ch_k=delta_ch_k * 0.8,
                pre_win=100, post_win=100, gap=10, edge_margin=edge_margin)

        # Keep only the single strongest peak per gap (avoid FP floods)
        if local_on or local_off:
            all_local = [(t, "onset") for t in local_on] + [(t, "offset") for t in local_off]
            best_t, best_dir = max(all_local, key=lambda x: abs(dE[x[0]]))
            if best_dir == "onset":
                rescued_on.append(best_t)
            else:
                rescued_off.append(best_t)

    return rescued_on, rescued_off


def detect_nonstationarity_v1G(
    X: np.ndarray,
    baseline_idx: np.ndarray,
    scales: Optional[np.ndarray] = None,
    min_scale: float = 4.0,
    moments: List[int] = [1, 2, 3, 4],
    moment_window: int = 50,
    n_surrogates: int = 100,
    alpha: float = 0.01,
    eps_on_pos: Optional[float] = None,
    eps_on_neg: Optional[float] = None,
    k_scales_min: int = 3,
    min_snr: float = 2.0,
    aggregation_mode: Literal["sum_all", "max", "topK"] = "sum_all",
    top_k: int = 3,
    omega0: float = 6.0,
    smooth_window: int = 15,
    severity_decay: float = 0.98,
    refractory_period: int = 200,
    seed: Optional[int] = None,
    k_channels_min: int = 2,
    delta_ch_k: float = 1.5,
    step_delta_k: float = 2.5,
    # v1-G specific
    step_pre_win: int = 100,
    step_post_win: int = 100,
    adaptive_max_discount: float = 0.6,
    two_pass: bool = True,
    mean_regime_len: Optional[float] = None,
    min_gap_ratio: float = 2.0,
) -> DetectionResult:
    """
    Gatekeeper v1-G: Adaptive Step Validation + Two-Pass Detection.

    Extends v1-F with three targeted recall improvements while preserving
    precision. The detection pipeline is identical through step 6 (peak
    extraction). Changes begin at step 6b:

    v1-G changes from v1-F:
    - STEP 6b: Adaptive step validation — delta_k scales inversely with
      peak strength (|dE[tau]|/eps). Strong peaks need less sustained-shift
      evidence. Recovers true CPs with strong dE spikes but modest level shifts.
    - STEP 6b: Shorter pre/post windows (100 vs 160) — reduces edge dead zones
      from 234 to 174 samples, better suited for 600–800 sample regimes.
    - STEP 8 (NEW): Two-pass rescue — after the first pass, re-scan
      suspiciously long gaps (> 2× mean regime length) with relaxed thresholds.
      Only the single strongest peak per gap is rescued to avoid FP floods.
    """
    T, N = X.shape

    if scales is None:
        max_scale = min(64, max(T // 20, 8))
        n_scales = min(20, max(5, int(np.log2(max_scale / min_scale) * 4)))
        scales = np.logspace(np.log2(min_scale), np.log2(max_scale), n_scales, base=2)
    else:
        scales = scales[scales >= min_scale]

    # STEP 1: Multi-moment calibration (same as v1-F)
    X_baseline = X[baseline_idx, :]
    calibration = calibrate_multi_moment_thresholds(
        X_baseline, scales, moments, moment_window,
        n_surrogates, alpha, omega0, smooth_window, seed
    )

    # STEP 2: Multi-moment weighted energy (same as v1-F)
    E_pos, E_neg, E_signed, E_ch_pos, E_ch_neg, E_by_moment, _ = compute_multi_moment_energy(
        X, scales, calibration.moment_thresholds, moment_window,
        moments, omega0, aggregation_mode, top_k, k_scales_min
    )

    # STEP 3: Energy derivative (same as v1-F)
    dE = compute_energy_derivative(E_signed, smooth_window)

    edge_margin = max(10, int(np.max(scales)))

    # STEP 4: Forgetting severity (same as v1-F)
    severity = compute_forgetting_severity(dE, decay=severity_decay, mode="leaky")

    # STEP 5: Surrogate-calibrated thresholds (same as v1-F)
    if eps_on_pos is None:
        eps_on_pos = calibration.eps_on_pos
    if eps_on_neg is None:
        eps_on_neg = calibration.eps_on_neg

    # STEP 6: Peak extraction (same as v1-F)
    onset_points_raw, offset_points_raw = extract_derivative_peaks(
        dE,
        eps_on_pos=eps_on_pos,
        eps_on_neg=eps_on_neg,
        refractory_period=refractory_period,
        edge_margin=edge_margin,
        prominence_ratio=0.3,
    )

    # STEP 6b: ADAPTIVE step validation (v1-G — replaces v1-F fixed step)
    onset_points_raw = filter_peaks_by_local_step_adaptive(
        onset_points_raw, E_signed, dE, eps_on_pos, "onset",
        pre_win=step_pre_win, post_win=step_post_win, gap=10,
        delta_k=step_delta_k, edge_margin=edge_margin,
        max_discount=adaptive_max_discount,
    )
    offset_points_raw = filter_peaks_by_local_step_adaptive(
        offset_points_raw, E_signed, dE, eps_on_neg, "offset",
        pre_win=step_pre_win, post_win=step_post_win, gap=10,
        delta_k=step_delta_k, edge_margin=edge_margin,
        max_discount=adaptive_max_discount,
    )

    # STEP 6c: K-of-channels consistency gate
    # Use v1-F window sizes (160) — per-channel estimates need longer windows
    # for stable mean/std; shorter step_pre_win is only for step validation.
    ch_win = 160
    if N > 1 and k_channels_min > 0:
        onset_points_raw, onset_ch_info = filter_peaks_by_channel_consistency(
            onset_points_raw, E_ch_pos, E_ch_neg, "onset",
            k_channels_min=k_channels_min, delta_ch_k=delta_ch_k,
            pre_win=ch_win, post_win=ch_win, gap=10,
            edge_margin=edge_margin,
        )
        offset_points_raw, offset_ch_info = filter_peaks_by_channel_consistency(
            offset_points_raw, E_ch_pos, E_ch_neg, "offset",
            k_channels_min=k_channels_min, delta_ch_k=delta_ch_k,
            pre_win=ch_win, post_win=ch_win, gap=10,
            edge_margin=edge_margin,
        )
    else:
        onset_ch_info = [None] * len(onset_points_raw)
        offset_ch_info = [None] * len(offset_points_raw)

    # STEP 7: SNR filtering + event building (same as v1-F)
    baseline_interior_mask = (baseline_idx >= edge_margin) & (baseline_idx < T - edge_margin)
    baseline_dE_interior = dE[baseline_idx[baseline_interior_mask]] if np.any(baseline_interior_mask) else dE[baseline_idx]
    baseline_dE_std = np.std(baseline_dE_interior) if len(baseline_dE_interior) > 0 else 1.0
    baseline_dE_std = max(baseline_dE_std, 0.01)

    w = max(3, smooth_window)
    events = []
    onset_points = []
    offset_points = []

    for i, tau in enumerate(onset_points_raw):
        t_start = max(0, tau - w)
        t_end = min(T - 1, tau + w)
        event = build_event_report(t_start, t_end, tau, "onset", dE, E_signed, E_ch_pos, E_ch_neg, scales)
        if min_snr > 0:
            if event.peak_dE / baseline_dE_std < min_snr:
                continue
        if i < len(onset_ch_info) and onset_ch_info[i] is not None:
            event.n_active_channels = onset_ch_info[i]['n_active']
            event.frac_active_channels = onset_ch_info[i]['frac_active']
            event.channel_z_scores = onset_ch_info[i]['z_scores']
            event.concentration_ratio = onset_ch_info[i]['concentration_ratio']
        events.append(event)
        onset_points.append(tau)

    for i, tau in enumerate(offset_points_raw):
        t_start = max(0, tau - w)
        t_end = min(T - 1, tau + w)
        event = build_event_report(t_start, t_end, tau, "offset", dE, E_signed, E_ch_pos, E_ch_neg, scales)
        if min_snr > 0:
            if event.peak_dE / baseline_dE_std < min_snr:
                continue
        if i < len(offset_ch_info) and offset_ch_info[i] is not None:
            event.n_active_channels = offset_ch_info[i]['n_active']
            event.frac_active_channels = offset_ch_info[i]['frac_active']
            event.channel_z_scores = offset_ch_info[i]['z_scores']
            event.concentration_ratio = offset_ch_info[i]['concentration_ratio']
        events.append(event)
        offset_points.append(tau)

    # STEP 8 (v1-G NEW): Two-pass rescue for missed CPs in long gaps
    if two_pass:
        first_pass_cps = sorted(set(onset_points + offset_points))
        est_regime_len = mean_regime_len if mean_regime_len else (T / max(len(first_pass_cps) + 1, 2))

        rescued_on, rescued_off = _second_pass_peaks(
            dE, E_signed, E_ch_pos, E_ch_neg,
            first_pass_cps, T,
            eps_on_pos, eps_on_neg,
            refractory_period, edge_margin,
            step_delta_k, k_channels_min, delta_ch_k,
            min_gap_ratio=min_gap_ratio,
            mean_regime_len=est_regime_len,
        )

        # Build events for rescued peaks (tag them)
        for tau in rescued_on:
            if tau not in onset_points:
                t_start = max(0, tau - w)
                t_end = min(T - 1, tau + w)
                event = build_event_report(t_start, t_end, tau, "onset", dE, E_signed, E_ch_pos, E_ch_neg, scales)
                events.append(event)
                onset_points.append(tau)

        for tau in rescued_off:
            if tau not in offset_points:
                t_start = max(0, tau - w)
                t_end = min(T - 1, tau + w)
                event = build_event_report(t_start, t_end, tau, "offset", dE, E_signed, E_ch_pos, E_ch_neg, scales)
                events.append(event)
                offset_points.append(tau)

    change_points = _merge_nearby_change_points(sorted(set(onset_points + offset_points)))
    is_nonstationary = len(events) > 0
    moment_weights = get_moment_weights(moments)

    # Per-moment voting: independent detection on each moment's dE
    mom_cps, mom_dE = compute_per_moment_detections(
        E_by_moment, baseline_idx,
        smooth_window=smooth_window,
        refractory_period=refractory_period,
        edge_margin=edge_margin,
    )

    return DetectionResult(
        events=events,
        E_pos=E_pos,
        E_neg=E_neg,
        E_signed=E_signed,
        dE=dE,
        severity=severity,
        E_ch_pos=E_ch_pos,
        E_ch_neg=E_ch_neg,
        calibration=calibration,
        is_nonstationary=is_nonstationary,
        change_points=change_points,
        onset_points=onset_points,
        offset_points=offset_points,
        edge_margin=edge_margin,
        E_by_moment=E_by_moment,
        moments_used=moments,
        moment_weights=moment_weights,
        moment_change_points=mom_cps,
        moment_dE=mom_dE,
    )


# =============================================================================
# Piecewise Recalibration (handles deviation jumps)
# =============================================================================

def detect_with_recalibration(
    X: np.ndarray,
    baseline_idx: np.ndarray,
    n_iterations: int = 2,
    quiet_quantile: float = 0.7,
    min_segment_len: int = 100,
    **kwargs,
) -> DetectionResult:
    """
    Iterative piecewise recalibration for handling deviation jumps.

    NOTE: With v1-D's differential detection, recalibration is less necessary
    because dE ≈ 0 inside sustained regimes (no cascading FPs). However, this
    function remains useful for very long recordings with multiple regime changes.

    After a variance/deviation jump, thresholds calibrated on the old regime
    become invalid. This function:

    1. Initial detection with baseline calibration
    2. Split signal at detected change points
    3. For each segment, recalibrate thresholds using "quiet" subset
    4. Re-run detection within each segment
    5. Rebuild global arrays from segment results
    6. Iterate until change points stabilize

    Parameters
    ----------
    X : np.ndarray
        Multichannel time series [T, N]
    baseline_idx : np.ndarray
        Initial baseline indices
    n_iterations : int
        Number of recalibration iterations (1-3 typical)
    quiet_quantile : float
        Fraction of lowest |dE| samples to use for recalibration (0.6-0.8)
    min_segment_len : int
        Minimum segment length for recalibration
    **kwargs
        Additional arguments passed to detect_nonstationarity

    Returns
    -------
    DetectionResult
        Final detection result after recalibration
    """
    T, N = X.shape

    # Initial detection
    result = detect_nonstationarity(X, baseline_idx, **kwargs)
    change_points = result.change_points

    # Use |dE| for quiet sample selection (derivative-based)
    current_dE = result.dE.copy()
    all_events = result.events[:]

    for _ in range(n_iterations):
        # Create segments from change points
        boundaries = [0] + sorted(change_points) + [T]
        segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        # Filter tiny segments
        segments = [(s, e) for s, e in segments if (e - s) >= min_segment_len]

        if len(segments) <= 1:
            break  # No meaningful segmentation

        # Re-detect within each segment with local calibration
        new_change_points = []
        all_events = []
        onset_points = []
        offset_points = []

        # Reset arrays for this iteration
        new_E_pos = np.zeros(T, dtype=float)
        new_E_neg = np.zeros(T, dtype=float)
        new_E_signed = np.zeros(T, dtype=float)
        new_dE = np.zeros(T, dtype=float)
        new_severity = np.zeros(T, dtype=float)
        new_E_ch_pos = np.zeros((T, N), dtype=float)
        new_E_ch_neg = np.zeros((T, N), dtype=float)

        for seg_start, seg_end in segments:
            seg_data = X[seg_start:seg_end, :]
            seg_T = seg_data.shape[0]

            if seg_T < min_segment_len:
                continue

            # Find "quiet" subset using |dE| (derivative magnitude)
            seg_dE = np.abs(current_dE[seg_start:seg_end])
            threshold = np.quantile(seg_dE, quiet_quantile)
            quiet_mask = seg_dE <= threshold

            quiet_idx = np.where(quiet_mask)[0]
            if len(quiet_idx) < min_segment_len // 2:
                quiet_idx = np.arange(min(seg_T // 3, min_segment_len))

            # Run detection on segment with local calibration
            seg_result = detect_nonstationarity(seg_data, quiet_idx, **kwargs)

            # Copy segment values to global arrays
            new_E_pos[seg_start:seg_end] = seg_result.E_pos
            new_E_neg[seg_start:seg_end] = seg_result.E_neg
            new_E_signed[seg_start:seg_end] = seg_result.E_signed
            new_dE[seg_start:seg_end] = seg_result.dE
            new_severity[seg_start:seg_end] = seg_result.severity
            new_E_ch_pos[seg_start:seg_end, :] = seg_result.E_ch_pos
            new_E_ch_neg[seg_start:seg_end, :] = seg_result.E_ch_neg

            # Translate segment-local events to global indices
            for event in seg_result.events:
                global_event = Event(
                    t_start=seg_start + event.t_start,
                    t_end=seg_start + event.t_end,
                    tau=seg_start + event.tau,
                    event_type=event.event_type,
                    channels_ranked=event.channels_ranked,
                    scales_ranked=event.scales_ranked,
                    peak_dE=event.peak_dE,
                    peak_E=event.peak_E,
                    area=event.area,
                    channel_scores=event.channel_scores,
                    scale_scores=event.scale_scores,
                )
                all_events.append(global_event)

                if event.event_type == "onset":
                    onset_points.append(seg_start + event.tau)
                else:
                    offset_points.append(seg_start + event.tau)

            # Translate segment-local change points to global indices
            for cp in seg_result.change_points:
                global_cp = seg_start + cp
                if global_cp > seg_start + 10 and global_cp < seg_end - 10:
                    new_change_points.append(global_cp)

        # Update arrays for next iteration
        current_dE = new_dE

        # Merge with segment boundaries
        all_cps = set(new_change_points)
        for s, _ in segments[1:]:
            if s > 0:
                all_cps.add(s)

        new_change_points = sorted(all_cps)

        # Check for convergence
        if set(new_change_points) == set(change_points):
            break

        change_points = new_change_points

    # Build final result
    final_change_points = sorted(set(change_points))

    return DetectionResult(
        events=all_events,
        E_pos=new_E_pos if 'new_E_pos' in dir() else result.E_pos,
        E_neg=new_E_neg if 'new_E_neg' in dir() else result.E_neg,
        E_signed=new_E_signed if 'new_E_signed' in dir() else result.E_signed,
        dE=new_dE if 'new_dE' in dir() else result.dE,
        severity=new_severity if 'new_severity' in dir() else result.severity,
        E_ch_pos=new_E_ch_pos if 'new_E_ch_pos' in dir() else result.E_ch_pos,
        E_ch_neg=new_E_ch_neg if 'new_E_ch_neg' in dir() else result.E_ch_neg,
        calibration=result.calibration,
        is_nonstationary=len(final_change_points) > 0,
        change_points=final_change_points,
        onset_points=onset_points if 'onset_points' in dir() else result.onset_points,
        offset_points=offset_points if 'offset_points' in dir() else result.offset_points,
        edge_margin=result.edge_margin,
    )


# =============================================================================
# Convenience Wrapper
# =============================================================================

def analyze_signal_windows_wavelets(
    signal: np.ndarray,
    window_size: int,
    overlap: int = 0,  # Kept for API compatibility with moments detector
    baseline_fraction: float = 0.2,
    **kwargs,
) -> DetectionResult:
    """
    Convenience wrapper matching moments detector interface.

    Note: overlap parameter is kept for API compatibility but not used
    (wavelet detection operates on full signal, not windows).
    """
    _ = overlap  # Silence unused parameter warning

    # Ensure shape is [T, N]
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    elif signal.shape[0] < signal.shape[1]:
        signal = signal.T

    T = signal.shape[0]
    baseline_idx = np.arange(int(T * baseline_fraction))

    if "scales" not in kwargs:
        min_scale = max(2, window_size // 20)
        max_scale = window_size // 2
        n_scales = min(12, max(4, int(np.log2(max_scale / min_scale) * 2) + 1))
        kwargs["scales"] = np.logspace(
            np.log2(min_scale), np.log2(max_scale), n_scales, base=2
        )

    return detect_nonstationarity(signal, baseline_idx, **kwargs)


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    print("Gatekeeper v1-D: Testing Signed Wavelet Deviations + Differential Detection...")
    print("=" * 70)

    np.random.seed(42)
    T, N = 1000, 4
    onset_point = 400
    offset_point = 700

    # Create signal with TWO change points:
    # - onset at t=400 (variance increases)
    # - offset at t=700 (variance returns to baseline)
    X = np.vstack([
        np.random.randn(onset_point, N) * 0.5,                          # Low variance
        np.random.randn(offset_point - onset_point, N) * 2.0,           # High variance
        np.random.randn(T - offset_point, N) * 0.5                      # Return to low
    ])

    baseline_idx = np.arange(int(T * 0.2))
    result = detect_nonstationarity(
        X, baseline_idx,
        n_surrogates=50,
        alpha=0.05,
        k_scales_min=2,
        seed=42
    )

    print(f"\nGround truth:")
    print(f"  Onset (variance increase) at t={onset_point}")
    print(f"  Offset (variance decrease) at t={offset_point}")

    print(f"\nDetected change points: {result.change_points}")
    print(f"  Onset points: {result.onset_points}")
    print(f"  Offset points: {result.offset_points}")

    print(f"\nEvents detected: {len(result.events)}")
    for i, event in enumerate(result.events):
        print(f"  Event {i+1}: type={event.event_type}, tau={event.tau}, "
              f"peak_dE={event.peak_dE:.2f}")

    print(f"\nCalibration thresholds:")
    print(f"  eps_on_pos (onset): {result.calibration.eps_on_pos:.2f}")
    print(f"  eps_on_neg (offset): {result.calibration.eps_on_neg:.2f}")

    print(f"\nSeverity trace (forgetting): max={result.severity.max():.2f}, "
          f"decays to {result.severity[-1]:.2f} at end")

    # Test that it doesn't produce cascading FPs in the high-variance regime
    high_var_region = slice(onset_point + 50, offset_point - 50)
    dE_inside_regime = result.dE[high_var_region]
    print(f"\nInside high-variance regime (t={onset_point+50} to {offset_point-50}):")
    print(f"  Mean |dE|: {np.mean(np.abs(dE_inside_regime)):.4f} (should be low)")
    print(f"  Max |dE|: {np.max(np.abs(dE_inside_regime)):.4f}")

    print("\n" + "=" * 70)
    print("Test complete.")
