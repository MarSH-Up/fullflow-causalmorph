# Gatekeeper v1-C: Wavelet Non-Stationarity Detector

## Overview

Gatekeeper is a wavelet-based detector for identifying structural change points in multivariate time series. It uses Continuous Wavelet Transform (CWT) with Morlet wavelets combined with Fourier-surrogate calibration to detect when the spectral characteristics of a signal deviate significantly from a baseline.

## Algorithm Pipeline

```
Input Signal X[T, N]
       │
       ▼
┌─────────────────────────────────┐
│  1. CALIBRATION (Baseline)      │
│  - Generate Fourier surrogates  │
│  - Compute CWT scalograms       │
│  - Build max-stat thresholds    │
│  - Calibrate eps_on from null   │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  2. DETECTION (Full Signal)     │
│  - Compute CWT scalogram        │
│  - Relative exceedance Z        │
│  - K-of-scales gate             │
│  - Aggregate across channels    │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  3. EVENT EXTRACTION            │
│  - Hysteresis state machine     │
│  - Debouncing (m_on, r_off)     │
│  - Minimum event length filter  │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  4. CHANGE POINT REPORTING      │
│  - Report onset (start)         │
│  - Report offset (end) if       │
│    event is substantial         │
└─────────────────────────────────┘
       │
       ▼
Output: List of change points
```

## Key Components

### 1. Fourier Surrogate Calibration

**Purpose**: Establish a null distribution for "what the signal looks like under stationarity."

**Method**:
- Take baseline signal segment
- Generate surrogates by preserving |FFT| amplitudes but randomizing phases
- This preserves power spectrum (autocorrelation) while destroying temporal structure

```python
X_fft = np.fft.rfft(x)
amplitudes = np.abs(X_fft)
random_phases = rng.uniform(0, 2π, len(X_fft))
X_surrogate = amplitudes * exp(1j * random_phases)
x_surrogate = np.fft.irfft(X_surrogate)
```

### 2. Max-Stat Thresholds (FWER Control)

**Purpose**: Control family-wise error rate across all time points and scales.

**Method**:
- For each surrogate, compute `max_t SG(t)` (maximum over time, per scale)
- Threshold = (1-α) quantile of these maxima

**Interpretation**: Under the null (stationary signal), P(any false positive) ≤ α

```python
# Per channel, per scale
for each surrogate:
    SG = |CWT(x_surrogate)|²
    max_sg[i, :] = SG.max(axis=1)  # max over time

thresholds[ch, :] = quantile(max_sg, 1 - alpha, axis=0)
```

### 3. Relative Exceedance

**Purpose**: Normalize across scales so large scales don't dominate.

**Formula**:
```
Z(t, scale) = max(0, SG(t)/T(scale) - 1)
```

Where:
- `SG(t)` = scalogram power at time t
- `T(scale)` = threshold for that scale
- `Z` measures "how many times above threshold"

**Interpretation**: Z=1 means 2x threshold, Z=2 means 3x threshold, etc.

### 4. K-of-Scales Gate

**Purpose**: Reduce isolated false positives by requiring multiple scales to be significant simultaneously.

**Method**:
```python
sig = (SG > threshold)           # Boolean: which scales exceed threshold
k_sig = sig.sum(axis=0)          # Count significant scales at each time
Z[:, k_sig < k_scales_min] = 0   # Zero out if too few scales are significant
```

**Interpretation**: A true structural change should affect multiple frequency bands, not just one.

### 5. Hysteresis Event Extraction

**Purpose**: Smooth detection and avoid rapid on/off toggling.

**State Machine**:
```
     E(t) > eps_on for m_on samples
OFF ─────────────────────────────────> ON
                                        │
     E(t) < eps_off for r_off samples   │
ON <─────────────────────────────────── ON
```

**Parameters**:
- `eps_on`: Threshold to enter ON state
- `eps_off`: Threshold to exit ON state (typically 0.8 × eps_on)
- `m_on`: Debounce count to enter ON (must exceed threshold m_on consecutive times)
- `r_off`: Debounce count to exit ON

### 6. Onset/Offset Reporting

**Key Insight**: A sustained "active" regime followed by return to "quiet" has TWO transitions.

**Example**:
```
quiet (0-400) → active (400-800) → quiet (800-1200)
```
- **Onset at ~400**: quiet → active transition
- **Offset at ~800**: active → quiet transition

**Rules**:
- Always report onset (event.t_start)
- Report offset (event.t_end) only if:
  1. Event duration ≥ 2 × min_event_len (substantial regime)
  2. Event doesn't end at recording boundary

## Handling Deviation Jumps: Piecewise Recalibration

### The Problem

When noise variance changes abruptly:
1. Thresholds calibrated on low-variance baseline become invalid
2. High-variance regime causes sustained exceedance
3. Cascading false positives occur even in subsequent regimes

### The Solution: `detect_with_recalibration()`

**Algorithm**:
```
1. Initial detection with baseline calibration
2. Split signal at detected change points into segments
3. For each segment:
   a. Find "quiet" subset (lowest E(t) samples)
   b. Recalibrate thresholds using quiet subset
   c. Re-run detection within segment
4. Merge change points
5. Iterate until convergence (typically 2 iterations)
```

**Quiet Subset Selection**:
```python
threshold = quantile(E_segment, quiet_quantile)  # e.g., 0.7
quiet_mask = E_segment <= threshold
quiet_idx = where(quiet_mask)
# Recalibrate using only X[quiet_idx]
```

## Parameters Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.05 | FWER significance level (smaller = stricter) |
| `n_surrogates` | 100 | Number of surrogates for calibration |
| `min_scale` | 4.0 | Minimum CWT scale (drops high-freq noise) |
| `k_scales_min` | 2 | Min scales that must be significant |
| `m_on` | 3 | Debounce to enter ON state |
| `r_off` | 3 | Debounce to exit ON state |
| `min_event_len` | 20 | Minimum event duration (samples) |
| `tau_mode` | "onset" | Change point estimation mode |

### Parameter Tuning Tips

**Too many false positives?**
- Increase `k_scales_min` (e.g., 3 or 4)
- Decrease `alpha` (e.g., 0.02)
- Increase `min_event_len`

**Missing true changes?**
- Decrease `k_scales_min` (e.g., 0 or 1)
- Increase `alpha` (e.g., 0.10)
- Decrease `min_event_len`

**Noisy edges?**
- Increase `m_on` and `r_off`

## Usage Examples

### Basic Detection

```python
from detectors.detectors_wavelets import detect_nonstationarity

# X: [T, N] multivariate time series
# baseline_idx: indices of known stationary segment

result = detect_nonstationarity(
    X, baseline_idx,
    alpha=0.05,
    k_scales_min=3,
    min_event_len=40,
    tau_mode="onset",
)

print(f"Change points: {result.change_points}")
print(f"Events: {len(result.events)}")
```

### With Piecewise Recalibration (for deviation jumps)

```python
from detectors.detectors_wavelets import detect_with_recalibration

result = detect_with_recalibration(
    X, baseline_idx,
    n_iterations=2,
    quiet_quantile=0.7,
    min_segment_len=100,
    # Detection params
    alpha=0.05,
    k_scales_min=3,
    min_event_len=40,
)
```

## Output Structure

```python
@dataclass
class DetectionResult:
    events: List[Event]           # Detected events with metadata
    E_total: np.ndarray           # Global excess energy time series [T]
    E_ch: np.ndarray              # Per-channel excess energy [T, N]
    calibration: CalibrationResult
    is_nonstationary: bool
    change_points: List[int]      # Estimated change point indices
    edge_margin: int

@dataclass
class Event:
    t_start: int                  # Event onset
    t_end: int                    # Event offset
    tau: int                      # Estimated change point within event
    channels_ranked: List[int]    # Channels by contribution
    peak: float                   # Peak excess energy
    area: float                   # Total excess energy (severity)
```

## Limitations

1. **Variance vs Structure**: The detector responds primarily to spectral/variance changes, not purely structural (edge) changes. Small structural changes in small graphs may not produce detectable signatures.

2. **Baseline Dependency**: Requires a known stationary baseline segment for calibration.

3. **Computational Cost**: Surrogate generation and CWT are O(n_surrogates × T × M) where M is number of scales.

4. **Offline Processing**: The piecewise recalibration approach requires the full recording (not suitable for real-time).

## References

- Morlet wavelet: Grossmann & Morlet (1984)
- Fourier surrogates: Theiler et al. (1992)
- Max-stat thresholds: Nichols & Holmes (2002)
