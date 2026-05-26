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


def detect_correlation_changes(
    X: np.ndarray,
    window: int = 200,
    refractory_period: int = 150,
    edge_margin: int = 0,
    threshold_mad_k: float = 4.0,
    min_threshold: float = 0.3,
    baseline_idx: np.ndarray = None,
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

    Returns
    -------
    change_points : List[int]
    change_signal : np.ndarray [T]  — Frobenius distance per time step
    threshold : float — peak height threshold used
    """
    T, N = X.shape
    if N < 2 or T < 3 * window:
        return [], np.zeros(T), 0.0

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

    peaks, _ = find_peaks(change_signal, height=threshold, distance=refractory_period)
    peaks = [int(p) for p in peaks if edge_margin <= p < T - edge_margin]
    return peaks, change_signal, threshold
