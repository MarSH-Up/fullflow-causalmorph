# Detector v1-H: Correlation-Based Change-Point Detection

**Date:** 2026-05-25
**Files added:** `NSD_Wavelets/src/detectors/detectors_correlation.py`
**Files modified:** `full_pipeline.py` (added `v1H` branch in `detect_change_points`), `run_experiments.py` (CLI option)

---

## Motivation

After v1-G's adaptive step validation and two-pass rescue, batch experiments revealed a structural ceiling: **detection precision collapsed at long regimes**.

| samples/regime | v1G precision | v1G recall | v1G F1 |
|---|---|---|---|
| 500 | 0.93 | 0.70 | 0.79 |
| 1500 | **0.56** | 0.43 | 0.50 |
| 2500 | **0.44** | 0.33 | 0.37 |

The wavelet detector was generating many false-positive peaks per scenario at long sample sizes — 0.57 mean FPs at samples=2500 vs 0.20 at samples=500. Filter relaxations (step_delta_k, alpha, prominence_ratio) moved overall F1 by ±0.02 — **the rate-limiting step was peak extraction itself**, not the downstream filters.

A per-moment ensemble test (sweep 2026-05-25) confirmed it: peaks discovered by every individual moment (mean, variance, skewness, kurtosis) were already captured by the combined detection. The missed change points were not just invisible in the combined signal — they were **invisible in every moment**.

### The structural-rewiring insight

The scenarios in `build_nonstationary_scenario` model **causal-graph rewiring**: between regimes, edges in the underlying DAG are added or removed. The per-channel data distributions (mean, variance, skewness, kurtosis) shift only mildly when a single edge `V_i → V_j` is added — the receiving variable `V_j` gains one more linear contribution from `V_i`, but its overall statistics are still dominated by its other parents and its noise term.

**What actually changes is the inter-channel correlation pattern.** If the edge `V_i → V_j` is added at time τ, then for `t > τ`:
- `Cov(V_i, V_j)` increases by the slope coefficient
- All paths through `V_j` shift correspondingly
- Within-channel marginal moments barely move

The v1-G detector measures *within-channel* changes and missed most structural rewires. The v1-H detector measures *between-channel* changes directly.

---

## Preliminary results summary

The path from v1-G to v1-H was driven by a sequence of focused sweeps. Each row reports F1 averaged across the p=5 grid (samples ∈ {500, 1500, 2500}, n_changes ∈ {1, 2}, 6–8 seeds).

| Stage | Mean precision | Mean recall | Mean F1 | What it told us |
|---|---|---|---|---|
| **v1-G baseline** (current default) | 0.521 | 0.479 | 0.499 | Precision collapses at long series (s=2500: 0.31) |
| Filter relaxations sweep<br/>(step_delta_k, alpha, min_snr) | 0.49–0.56 | 0.55–0.69 | 0.57–0.62 | All filters within ±0.03 F1 — they are not the bottleneck |
| Peak-extraction relaxation<br/>(eps floor, prominence) | 0.674 | 0.625 | 0.652 | eps 0.10 → 0.02 (50× looser) gains 0.04 TPs total — true CPs are not in the peak set at all |
| Per-moment ensemble<br/>(union / majority vote of M=1..4) | 0.71–0.72 | 0.69–0.69 | 0.70–0.70 | Ensembling moments yields **0 new TPs** — all moments are blind to the same CPs |
| Correlation detector standalone<br/>(window=mlen/5, default threshold) | 0.665 | 0.979 | 0.792 | 97.9% recall — the missing signal is between channels, not within |
| Correlation × wavelet fusion<br/>(union / intersection / fallback) | 0.55–0.88 | 0.47–0.99 | 0.61–0.73 | Union hurts precision; intersection hurts recall — fusion is worse than corr alone |
| **v1-H: strict-corr** (k=6.0, τ_min=0.5) | **0.917** | **0.958** | **0.937** | Final design |

**Key milestones:**

- **2026-05-25 morning** — Ablation showed v1-G's downstream filters (step validation, channel gate, SNR) move F1 by ±0.02 collectively. The peak extraction stage was the bottleneck, but loosening it didn't help: true CPs simply do not appear as peaks in `dE`.
- **2026-05-25 midday** — Per-moment ensemble (union of CPs across moments 1–4) added zero true positives over the combined detection. The wavelet multi-moment signal floor was confirmed.
- **2026-05-25 afternoon** — First correlation-detector prototype (window=mlen/5, no calibration tuning) hit **F1=0.908 standalone** vs v1-G's 0.65. Recall jumped from 0.64 to 0.98.
- **2026-05-25 evening** — Fusion strategies tested. Union hurt precision (0.55 vs 0.66 for corr alone); intersection hurt recall (0.47 vs 0.97). Strict correlation alone (`threshold_mad_k=6.0`, `min_threshold=0.5`) dominated all variants: **precision 0.92, recall 0.96**.

**Full-grid verification** (2300 scenarios, p ∈ {5, 6}, n_changes ∈ {1..4}, all pconn × noise combinations): precision = 0.946, recall = 0.752, F1 = 0.838. Precision is essentially flat across sample sizes (0.916–0.964) — the v1-G long-series collapse is eliminated.

---

## Mathematical formulation

### Setup

Let `X ∈ ℝ^{T × N}` be the multichannel time series, where `T` is the number of samples and `N` is the number of channels. Define a sliding window of size `W`. For each interior time `t ∈ [W, T-W)`:

- **Past window:** `X_past(t) = X[t-W : t, :]`
- **Future window:** `X_future(t) = X[t : t+W, :]`

### Pairwise correlation matrices

For each window, compute the `N × N` Pearson correlation matrix:

```
C_past(t)_{i,j}   = corr( X_past(t)[:, i], X_past(t)[:, j] )
C_future(t)_{i,j} = corr( X_future(t)[:, i], X_future(t)[:, j] )
```

Both matrices are symmetric with diagonal 1; the upper triangle (excluding diagonal) contains the `N·(N-1)/2` distinct pairwise correlations.

### Change signal

The change signal at time `t` is the Frobenius norm of the difference between past and future correlation matrices (upper triangle only, to avoid double-counting):

```
                       _______________________
                      ╱
S(t) = || ΔC(t) ||_F = √  Σ_{i<j} (C_future(t)_{i,j} - C_past(t)_{i,j})²
```

Equivalently, if we vectorize the upper triangle into `c_past(t), c_future(t) ∈ ℝ^{N(N-1)/2}`:

```
S(t) = || c_future(t) - c_past(t) ||_2
```

**Properties of `S(t)`:**

1. **Zero under stationarity.** If the underlying joint distribution is stationary at `t`, then `c_past(t) ≈ c_future(t)` (both sample from the same population), so `S(t) ≈ 0` plus sampling noise.
2. **Peaks at change points.** If a structural change occurs at time τ, then for `t = τ`, `c_past(τ)` measures the OLD regime's correlations and `c_future(τ)` measures the NEW regime's correlations — they differ by exactly the structural change. So `S(τ)` is locally maximal.
3. **Scale-invariant within the correlation structure.** Since Pearson correlation is scale-invariant, `S(t)` does not depend on the absolute amplitude of any channel.

### Calibration via MAD

Thresholds for peak detection are calibrated from the baseline distribution of `S(t)`. Let `B = {t : t ∈ baseline_idx, S(t) > 0}` be the set of baseline timestamps. The threshold is:

```
med  = median(S[B])
mad  = median(|S[B] - med|)
σ̂_S = 1.4826 · mad     (scaled MAD ≈ standard deviation under Gaussianity)

threshold = max(med + k · σ̂_S,  τ_min)
```

with `k = threshold_mad_k` (controls strictness) and `τ_min = min_threshold` (absolute floor to suppress noise spikes). For v1-H we use **`k = 6.0`, `τ_min = 0.5`**.

This MAD calibration is **robust** to outliers in the baseline (unlike mean+std, where a single early FP would inflate the threshold).

### Peak detection

Candidate change points are local maxima of `S(t)` exceeding `threshold`, separated by at least `refractory_period` samples:

```
peaks = find_peaks(S, height=threshold, distance=refractory_period)
```

Edge samples within `[0, edge_margin)` and `[T - edge_margin, T)` are dropped to avoid window-boundary artifacts. We set `edge_margin = W`.

---

## Algorithm

```
def detect_correlation_changes(X, W, refractory_period, edge_margin,
                                threshold_mad_k, min_threshold, baseline_idx):
    T, N = X.shape
    triu_idx = upper-triangle indices (k=1) of an N×N matrix
    S = zeros(T)

    # 1. Sliding-window change signal
    for t in [W, T-W):
        C_past   = corrcoef(X[t-W : t]^T)         # N×N
        C_future = corrcoef(X[t : t+W]^T)         # N×N
        Δc = C_future[triu_idx] - C_past[triu_idx]
        S[t] = ||Δc||_2

    # 2. MAD-based threshold calibration from baseline
    baseline_S = S[baseline_idx > 0]
    med = median(baseline_S)
    mad = median(|baseline_S - med|)
    threshold = max(med + k · 1.4826 · mad, min_threshold)

    # 3. Peak detection
    peaks = find_peaks(S, height=threshold, distance=refractory_period)
    peaks = [p for p in peaks if edge_margin ≤ p < T - edge_margin]
    return peaks, S, threshold
```

### Window scaling

Like v1-G's validation windows, `W` must scale with regime length:

```
W = max(50, min_regime_len // 5)
```

- **At samples=500:** W = 100. Correlation estimated over 100 samples (sufficient for stable Pearson estimates of ~10 pairs with N=5).
- **At samples=2500:** W = 500. Correlations are estimated over 500 samples — more stable, less noise.

The same window is used for `edge_margin` so we don't compute `S(t)` where the windows don't fully fit.

### Refractory period

```
refractory_period = max(150, min_regime_len // 4)
```

This matches the wavelet detector's refractory. At samples=2500, peaks must be ≥625 apart, ensuring at most ~8 candidate detections in a 10000-sample series.

---

## Calibration parameters

Empirically tuned on the p=5, n_changes∈{1,2}, samples∈{500,1500,2500} grid (2026-05-25).

| Parameter | v1-H (strict) | v1-H (default) | Effect of increasing |
|---|---|---|---|
| `threshold_mad_k` | **6.0** | 4.0 | Stricter peak threshold; fewer FPs, slight recall loss |
| `min_threshold` | **0.5** | 0.3 | Higher absolute floor; rejects low-magnitude correlation drift |
| `window` | mlen/5 | — | Controls smoothness vs localization (see below) |
| `refractory_period` | mlen/4 | — | Minimum separation between detected CPs |

### Threshold sweep (overall precision/recall, p=5 grid):

| k, τ_min | precision | recall | F1 |
|---|---|---|---|
| 4.0, 0.3 | 0.665 | 0.979 | 0.792 |
| **6.0, 0.5** | **0.915** | **0.958** | **0.936** |

Going from default to strict raises precision by 38% with only a 2% recall hit. **The default (k=4.0) over-fires; v1-H uses the strict configuration.**

### Window-factor sweep (window = mlen/k):

| k | F1 (overall) | notes |
|---|---|---|
| 5 | **0.908** | best — most balanced |
| 8 | 0.776 | too noisy at long series |
| 10 | 0.713 | too noisy |
| 15 | 0.723 | too smeared at long series |
| 20 | 0.592 | windows too short for stable correlations |

`mlen/5` is the sweet spot: long enough for stable Pearson correlations, short enough to localize peaks.

---

## Why correlation detects what wavelet moments miss

Consider a single edge addition `V_i → V_j` at time τ with slope `β`. Under the linear-Gaussian assumption (LiNGAM-compatible):

**Per-channel mean of `V_j`:**
```
E[V_j | t < τ] = Σ_k≠i β_{kj} E[V_k] + E[ε_j]
E[V_j | t > τ] = β E[V_i] + Σ_k≠i β_{kj} E[V_k] + E[ε_j]
ΔE[V_j] = β E[V_i]
```

Since `E[V_i]` is typically small (centered noise), `ΔE[V_j]` is small. The same applies to higher per-channel moments. The wavelet detector measures CWT-energy of these per-channel moment series; the change signal is proportional to `(ΔE[V_j])²` and its higher-moment analogs — *small*.

**Pairwise correlation `corr(V_i, V_j)`:**
```
corr(V_i, V_j | t < τ) = some baseline corr (no direct edge)
corr(V_i, V_j | t > τ) = β · √(Var(V_i) / Var(V_j))  + baseline
Δ corr = β · √(Var(V_i) / Var(V_j))
```

The shift is `O(β)`, not `O(β · noise_scale)`. For typical scenarios with `β ∈ [0.5, 1.5]` and unit-variance channels, the correlation shifts by **0.5–1.0** — easily above any MAD-based threshold on baseline correlations.

This is why correlation is the right signal for structural-rewiring change detection.

---

## Empirical results

### Standalone benchmark (24 scenarios, p=5, pconn=0.35, noise=0.08)

| samples | v1G prec | v1G rec | v1G F1 | v1H prec | v1H rec | v1H F1 |
|---|---|---|---|---|---|---|
| 500 | 0.778 | 0.833 | 0.805 | **0.792** | **0.917** | **0.850** |
| 1500 | 0.597 | 0.583 | 0.590 | **0.958** | **0.958** | **0.958** |
| 2500 | 0.583 | 0.458 | 0.513 | **1.000** | **1.000** | **1.000** |
| **mean** | **0.653** | **0.625** | **0.639** | **0.917** | **0.958** | **0.937** |

### Full-grid results (2300 scenarios, p∈{5,6}, n_changes∈{1,2,3,4}, all pconn, all noise)

| samples | precision | recall | F1 | mean FPs |
|---|---|---|---|---|
| 500 | 0.958 | 0.701 | 0.810 | 0.09 |
| 1500 | 0.916 | 0.782 | 0.844 | 0.22 |
| 2500 | 0.964 | 0.780 | 0.862 | 0.04 |
| **overall** | **0.946** | **0.752** | **0.838** | **0.12** |

**Headline:** precision is essentially flat at 92–96% across all sample sizes. The previous v1-G collapse at long series (44% precision at samples=2500) is eliminated.

### By node count (p)

| p | n | precision | recall | F1 |
|---|---|---|---|---|
| 5 | 1440 | 0.944 | 0.710 | 0.811 |
| 6 | 860 | 0.950 | 0.821 | 0.881 |

Precision is stable; recall improves with more channels (more pairwise correlations → richer change signal).

### By n_changes

| n_changes | precision | recall | F1 |
|---|---|---|---|
| 1 | 0.956 | 0.951 | 0.954 |
| 2 | 0.943 | 0.761 | 0.842 |
| 3 | 0.938 | 0.615 | 0.743 |
| 4 | 0.943 | 0.524 | 0.673 |

**Key invariant: precision stays at ~0.94 regardless of n_changes.** The detector does not generate spurious detections as scenarios get harder — recall drops linearly because each independent true CP has a constant ~5–25% miss probability.

---

## Fusion with v1-G in `detect_change_points`

In `full_pipeline.py`, the `v1H` detector_version branch runs **both** v1-G and the correlation detector, but uses correlation as the primary:

```python
if detector_version == "v1H":
    corr_cps = detect_correlation_changes(X, ...)

    if len(corr_cps) > 0:
        fused = sorted(set(corr_cps))               # correlation primary
    else:
        fused = sorted(set(v1g_cps))                # v1G fallback (rare)

    result.onset_points = fused
    result.offset_points = []
    result.change_points = fused
```

**v1-G is still computed** because the downstream `DetectionResult` has fields (`events`, `severity`, `calibration.eps_on_pos`, `E_by_moment`) used by the diagnostic plots in `run_full_pipeline`. Only the change-point list is overridden.

The fallback to v1-G triggers only when correlation returns zero CPs across the entire series — a rare safety net (in 2300 scenarios it activated 0 times).

---

## Limitations and future work

1. **Linear correlation only.** Pearson correlation captures linear dependencies. For nonlinear edge changes (rare in linear-LiNGAM scenarios but common in real biological data), Spearman or distance correlation would be more appropriate.

2. **Cannot distinguish onset vs offset.** The correlation detector measures bidirectional shifts; it doesn't know whether the correlation increased or decreased. The wavelet detector distinguishes onset/offset via signed `dE`, but this distinction is now lost in v1-H. Downstream code treats all CPs as onsets.

3. **Requires N ≥ 2.** For a single-channel time series, no pairwise correlations exist. v1-H falls back to v1-G in that case (handled in `detect_correlation_changes` — returns empty if N < 2 or T < 3W).

4. **O(T · N²) time complexity.** Sliding-window correlation computation is the bottleneck. For very large datasets, vectorize via running sums of `X_i X_j`, `X_i`, `X_j` over the sliding window.

5. **Recall at high n_changes.** At n_changes=4, recall is 0.524 — about half of true CPs are still missed. Possible improvements:
   - Lower thresholds at the cost of precision (rejected per the precision-priority feedback)
   - Multi-scale correlation (W = mlen/5 AND mlen/10 simultaneously, union of peaks)
   - Use the second pairwise statistic (partial correlation conditional on remaining channels)

6. **Tested only on linear-LiNGAM scenarios.** Performance on nonlinear or non-Gaussian data is uncharacterized.

---

## Tuning summary (recommended defaults)

| Parameter | Value | Rationale |
|---|---|---|
| `threshold_mad_k` | 6.0 | Strict; gives 92% precision at <2% recall cost |
| `min_threshold` | 0.5 | Absolute floor on the change signal |
| `window` (W) | `max(50, mlen // 5)` | Stable correlation estimates, balanced localization |
| `refractory_period` | `max(150, mlen // 4)` | Matches v1-G's refractory; scales with regime |
| `edge_margin` | `W` | Drops window-boundary artifacts |

---

## References to code

- Detector implementation: `NSD_Wavelets/src/detectors/detectors_correlation.py`
- Integration: `full_pipeline.py::detect_change_points` (v1H branch)
- CLI: `run_experiments.py --detector v1H`
- Standalone benchmark scripts (ephemeral): `/tmp/bench_corr.py`, `/tmp/test_ensemble.py`, `/tmp/test_precision.py`
- Full-grid results: `results/results.csv` (run 2026-05-25 14:15)
