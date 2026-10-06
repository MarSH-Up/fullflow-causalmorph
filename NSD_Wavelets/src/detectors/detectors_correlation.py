"""
Correlation-based change-point detector.

Independent of the wavelet detector — uses the pairwise channel correlation
matrix as the change signal. At each time t, compute correlation in a past
window [t-W, t] vs a future window [t, t+W], and measure their Frobenius
distance. True change points produce peaks in this distance signal.

The wavelet detector uses per-channel energy in 4 statistical moments; this
detector uses INTER-channel correlation. These are complementary signals —
if one channel's edge is added/removed, the correlation with other channels
changes even when the per-channel energy doesn't move much.
"""
import numpy as np
from scipy.signal import find_peaks


def standardized_corr_change(X: np.ndarray, window: int, studentize: bool = False) -> np.ndarray:
    """
    Per-pair change in correlation between the windows before and after t,
    standardised on the Fisher scale:

        zeta_ij(t) = sqrt((W-3)/2) * ( artanh r_ij[t, t+W) - artanh r_ij[t-W, t) )

    Under no change and approximate normality, Var(artanh r) ~ 1/(W-3), the two
    windows are independent, so zeta_ij ~ N(0, 1). Returns an array [T, m] over
    the m = N(N-1)/2 upper-triangular pairs; rows outside [W, T-W) are NaN.

    studentize : divide each pair by its own robust scale over the whole
        series, s_ij = 1.4826 * MAD_t zeta_ij(t). The Fisher variance is the
        Gaussian, i.i.d. result; with heavy-tailed innovations, strong
        correlations or temporally autocorrelated noise the true null scale is
        larger and grows with p (measured on the grid: pooled sd 1.22 / 1.20 /
        1.30 / 1.66 at p = 3 / 6 / 9 / 12; 0.87 / 0.87 / 0.83 / 0.91 after
        studentising). With sparse changes the MAD is dominated by the null, so
        no baseline segment is needed; the change-point bumps make s_ij a
        little conservative, in the safe direction.
    """
    T, N = X.shape
    triu = np.triu_indices(N, k=1)
    out = np.full((T, len(triu[0])), np.nan)
    if N < 2 or T < 3 * window:
        return out
    scale = np.sqrt(max(window - 3, 1) / 2.0)
    clip = 1.0 - 1e-6
    for t in range(window, T - window):
        c_past = np.nan_to_num(np.corrcoef(X[t - window:t].T), nan=0.0)[triu]
        c_fut = np.nan_to_num(np.corrcoef(X[t:t + window].T), nan=0.0)[triu]
        out[t] = scale * (np.arctanh(np.clip(c_fut, -clip, clip))
                          - np.arctanh(np.clip(c_past, -clip, clip)))
    if studentize:
        ok = np.isfinite(out).all(axis=1)
        if ok.any():
            med = np.median(out[ok], axis=0)
            s_hat = 1.4826 * np.median(np.abs(out[ok] - med), axis=0)
            out[ok] = out[ok] / np.maximum(s_hat, 1e-6)
    return out


def detect_correlation_changes(
    X: np.ndarray,
    window: int = 200,
    refractory_period: int = 150,
    edge_margin: int = 0,
    threshold_mad_k: float = 4.0,
    min_threshold: float = 0.3,
    baseline_idx: np.ndarray = None,
    statistic: str = "frobenius",
    gamma: float = 2.0,
    adaptive_null: bool = False,
    studentize: bool = False,
):
    """
    Detect change points from shifts in pairwise channel correlations.

    Parameters
    ----------
    X : [T, N] multichannel time series
    window : rolling window size (each side of t)
    refractory_period : min distance between detected peaks
    edge_margin : samples to ignore at series boundaries
    threshold_mad_k : peak threshold = baseline_med + k * MAD
    min_threshold : floor on the threshold (rejects noise spikes)
    baseline_idx : indices of baseline samples (used for threshold calibration);
                   if None, uses the first window samples after edge_margin
    statistic : "frobenius" — ||C_future - C_past||_F over the upper triangle
                (the v1H statistic; threshold from the baseline median + k*MAD).
                "max" — max_ij |zeta_ij(t)| of the Fisher-standardised per-pair
                change; threshold is analytic, sqrt(2 ln m) + gamma, so it is
                calibration-free in both the number of pairs and the window.
    gamma : additive margin for the "max" threshold (ignored otherwise)
    studentize : for "max", divide each pair by its robust null scale first
                 (see standardized_corr_change); ignored for "frobenius"
    adaptive_null : re-estimate the frobenius threshold from the segment since
                    the last accepted change point (with a guard band of one
                    window on each side) instead of from regime 0 only.
                    Ignored for "max", whose threshold does not use a baseline.

    Returns
    -------
    change_points : List[int]
    change_signal : np.ndarray [T]  — Frobenius distance per time step
    threshold : float — peak height threshold used
    """
    T, N = X.shape
    if N < 2 or T < 3 * window:
        return [], np.zeros(T), 0.0

    if statistic == "max":
        m = N * (N - 1) // 2
        zeta = standardized_corr_change(X, window, studentize=studentize)
        change_signal = np.nan_to_num(np.nanmax(np.abs(zeta), axis=1), nan=0.0)
        threshold = float(np.sqrt(2.0 * np.log(m)) + gamma)
        peaks, _ = find_peaks(change_signal, height=threshold, distance=refractory_period)
        peaks = [int(p) for p in peaks if edge_margin <= p < T - edge_margin]
        return peaks, change_signal, threshold
    if statistic != "frobenius":
        raise ValueError(f"statistic must be 'frobenius' or 'max', got {statistic!r}")

    triu_idx = np.triu_indices(N, k=1)
    change_signal = np.zeros(T)

    # For each interior time t, compute |C(future) - C(past)|_F
    for t in range(window, T - window):
        past = X[t - window:t, :]
        future = X[t:t + window, :]
        # np.corrcoef handles zero-variance via warnings; clip with epsilon
        c_past = np.corrcoef(past.T)
        c_future = np.corrcoef(future.T)
        # Replace NaNs (from zero variance) with 0
        c_past = np.nan_to_num(c_past, nan=0.0)
        c_future = np.nan_to_num(c_future, nan=0.0)
        diff = c_future[triu_idx] - c_past[triu_idx]
        change_signal[t] = np.linalg.norm(diff)

    # Threshold from baseline (after the first window of edge_margin)
    if baseline_idx is not None and len(baseline_idx) > 10:
        baseline_vals = change_signal[baseline_idx]
        baseline_vals = baseline_vals[baseline_vals > 0]
    else:
        b_start = edge_margin + window
        b_end = min(b_start + window, T - window)
        baseline_vals = change_signal[b_start:b_end]
        baseline_vals = baseline_vals[baseline_vals > 0]

    if len(baseline_vals) > 5:
        med = float(np.median(baseline_vals))
        mad = float(np.median(np.abs(baseline_vals - med)))
        threshold = max(med + threshold_mad_k * 1.4826 * max(mad, 1e-6), min_threshold)
    else:
        threshold = min_threshold

    if not adaptive_null:
        peaks, _ = find_peaks(change_signal, height=threshold, distance=refractory_period)
        peaks = [int(p) for p in peaks if edge_margin <= p < T - edge_margin]
        return peaks, change_signal, threshold

    # Adaptive null: candidate peaks are local maxima with the refractory
    # spacing; each is accepted against a threshold re-estimated from the
    # signal between the last accepted change point and the candidate, with a
    # guard band of one window on each side. The regime-0 threshold is the
    # fallback while that segment is too short to estimate from.
    candidates, _ = find_peaks(change_signal, distance=refractory_period)
    accepted = []
    last = edge_margin
    for c in (int(c) for c in candidates):
        if not (edge_margin <= c < T - edge_margin):
            continue
        seg = change_signal[last + window:c - window]
        seg = seg[seg > 0]
        if len(seg) >= window:
            med_t = float(np.median(seg))
            mad_t = float(np.median(np.abs(seg - med_t)))
            thr_t = max(med_t + threshold_mad_k * 1.4826 * max(mad_t, 1e-6), min_threshold)
        elif not accepted:
            # nothing accepted yet: the regime-0 calibration is the right null
            thr_t = threshold
        else:
            # fewer than one window of the new regime observed since the last
            # accepted change point: its null is not estimable yet, and the
            # regime-0 threshold is the wrong one to fall back on. Defer.
            continue
        if change_signal[c] >= thr_t:
            accepted.append(c)
            last = c
    return accepted, change_signal, threshold
