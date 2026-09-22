"""
Wavelet-coherence-based change-point detector (v1I).

Generalises the correlation detector (v1H) by replacing time-domain Pearson
correlation with squared wavelet coherence (Grinsted et al., 2004), a
frequency-resolved joint-distribution measure between channels.

Motivation
----------
v1G (per-channel wavelet moments) misses structural rewires because edge
changes shift the *joint* distribution while leaving the marginals nearly
unchanged.  v1H (Pearson correlation) recovers most of the joint shift but
collapses all scales into a single, band-integrated number and assumes
stationarity inside each window.  Wavelet coherence resolves the joint
structure across scales (Morlet CWT) and is intrinsically non-stationary
— matching the substrate already used by v1G.

This addresses the methodological concern that the change between regimes is
not in the marginals but in the joint distribution of the linked variables,
following the line of Kaminski/Baccalá-style frequency-domain coupling
measures.

Mathematics
-----------
For two centred channels x_i, x_j with continuous wavelet transforms
W_i(t, s), W_j(t, s) (Morlet, scale s, time t):

    cross-wavelet:   X_ij(t, s) = W_i(t, s) · W_j*(t, s)
    smoothed power:  P_i,W(s)   = (1/W) Σ_{τ∈window} |W_i(τ, s)|²
    smoothed cross:  X_ij,W(s)  = (1/W) Σ_{τ∈window}  X_ij(τ, s)
    coherence:       C_ij,W(s)  = |X_ij,W(s)|² / ( P_i,W(s) · P_j,W(s) )  ∈ [0, 1]
    scale-average:   C̄_ij,W    = (1/S) Σ_s C_ij,W(s)

For each interior time t we compute C̄_ij over the past window [t-W, t)
and the future window [t, t+W), then take the Frobenius distance between
those two coherence matrices.  Peaks of this signal are calibrated via MAD
on a baseline interval (same calibration as v1H).
"""
import numpy as np
from scipy.signal import find_peaks

from .detectors_wavelets import cwt_morlet


def detect_coherence_changes(
    X: np.ndarray,
    window: int = 200,
    refractory_period: int = 150,
    edge_margin: int = 0,
    threshold_mad_k: float = 6.0,
    min_threshold: float = 0.15,
    baseline_idx: np.ndarray = None,
    min_scale: float = 5.0,
    max_scale: float = None,
    n_scales: int = 12,
    omega0: float = 6.0,
    aggregation: str = "mean",
):
    """
    Detect change points from shifts in pairwise wavelet coherence.

    Parameters
    ----------
    X : np.ndarray, shape [T, N]
        Multichannel time series.
    window : int
        Past/future window size (each side of t).  Each window of length W
        provides 1/W weighting of cross-wavelet products.
    refractory_period : int
        Minimum distance between detected peaks (non-maximum suppression).
    edge_margin : int
        Samples ignored at the series boundaries.
    threshold_mad_k : float
        Peak threshold = median + k · 1.4826 · MAD over baseline.
    min_threshold : float
        Floor on the threshold; the coherence signal is in a much smaller
        magnitude range than the v1H correlation signal so the default
        floor is correspondingly lower.
    baseline_idx : np.ndarray, optional
        Indices of baseline (assumed stationary) samples used for the MAD
        calibration.  If None, uses the first `window` interior samples.
    min_scale, max_scale, n_scales : float, float, int
        CWT scale grid (log-spaced).  If max_scale is None, defaults to
        max(2·min_scale, window/2).
    omega0 : float
        Morlet wavelet central frequency.
    aggregation : {"mean", "max", "median"}
        How to reduce the per-scale coherence to a single per-pair value
        before computing the Frobenius distance.  "mean" dilutes the signal
        across uninformative scales; "max" picks the strongest scale at
        each window (more selective); "median" is a robust compromise.

    Returns
    -------
    change_points : List[int]
    change_signal : np.ndarray, shape [T]   — Frobenius distance per time
    threshold : float                        — peak threshold actually used
    """
    T, N = X.shape
    if N < 2 or T < 3 * window:
        return [], np.zeros(T), 0.0

    if max_scale is None:
        max_scale = max(min_scale * 2.0, window / 2.0)
    scales = np.geomspace(min_scale, max_scale, n_scales)
    S = len(scales)

    # Standardise per channel so coherence is not dominated by amplitude
    # differences (coherence is scale-invariant, but standardisation also
    # stabilises the MAD calibration downstream).
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # ── 1. CWT for each channel ─────────────────────────────────────────
    W = np.empty((N, S, T), dtype=complex)
    for c in range(N):
        W[c] = cwt_morlet(Xn[:, c], scales, omega0=omega0)
    P = (W.real ** 2 + W.imag ** 2)                       # [N, S, T] real

    # ── 2. Cross-wavelet for all upper-triangle pairs ───────────────────
    triu = np.triu_indices(N, k=1)
    M = len(triu[0])                                       # = N(N-1)/2
    X_cross = W[triu[0]] * np.conj(W[triu[1]])             # [M, S, T] complex

    # Free the raw CWT once we have what we need
    del W

    # ── 3. Sliding window sums via cumulative sums ──────────────────────
    # For an array A indexed on axis T, define cum[k] = Σ_{u<k} A[u].
    # Then Σ_{u∈[a,b)} A[u] = cum[b] - cum[a].
    def _cumsum_with_lead_zero(arr, axis):
        zero_shape = list(arr.shape)
        zero_shape[axis] = 1
        zeros = np.zeros(zero_shape, dtype=arr.dtype)
        return np.concatenate([zeros, np.cumsum(arr, axis=axis)], axis=axis)

    cumP = _cumsum_with_lead_zero(P,        axis=2)        # [N, S, T+1]
    cumX = _cumsum_with_lead_zero(X_cross,  axis=2)        # [M, S, T+1] complex
    del P, X_cross

    t_idx = np.arange(window, T - window)
    # Window sums (the 1/W normalisation cancels in the coherence ratio):
    P_past   = cumP[:, :, t_idx]            - cumP[:, :, t_idx - window]
    P_future = cumP[:, :, t_idx + window]   - cumP[:, :, t_idx]
    X_past   = cumX[:, :, t_idx]            - cumX[:, :, t_idx - window]
    X_future = cumX[:, :, t_idx + window]   - cumX[:, :, t_idx]
    del cumP, cumX

    # ── 4. Squared coherence per pair / scale / time ────────────────────
    num_past   = X_past.real ** 2  + X_past.imag ** 2      # [M, S, len(t_idx)]
    num_future = X_future.real ** 2 + X_future.imag ** 2

    den_past   = P_past[triu[0]]   * P_past[triu[1]]   + 1e-12
    den_future = P_future[triu[0]] * P_future[triu[1]] + 1e-12

    coh_past   = num_past   / den_past                     # ∈ [0, 1]
    coh_future = num_future / den_future

    # Scale-aggregated coherence per pair, per time
    if aggregation == "mean":
        Cbar_past   = coh_past.mean(axis=1)
        Cbar_future = coh_future.mean(axis=1)
    elif aggregation == "max":
        Cbar_past   = coh_past.max(axis=1)
        Cbar_future = coh_future.max(axis=1)
    elif aggregation == "median":
        Cbar_past   = np.median(coh_past,   axis=1)
        Cbar_future = np.median(coh_future, axis=1)
    else:
        raise ValueError(f"Unknown aggregation {aggregation!r}; expected mean|max|median")

    # ── 5. Frobenius distance between past/future coherence matrices ───
    diff = Cbar_future - Cbar_past                         # [M, len(t_idx)]
    change_signal = np.zeros(T)
    change_signal[t_idx] = np.linalg.norm(diff, axis=0)

    # ── 6. MAD-calibrated peak threshold ────────────────────────────────
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
