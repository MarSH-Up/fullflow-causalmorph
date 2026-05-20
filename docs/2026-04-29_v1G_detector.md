# Detector v1-G: Adaptive Step Validation + Two-Pass Rescue

**Date:** 2026-04-29  
**Files changed:** `NSD_Wavelets/src/detectors/detectors_wavelets.py`, `full_pipeline.py`, `batch_experiments.py`, `bilinear_example.py`

---

## Motivation

Batch ablation (800 experiments, p=3-10) confirmed CausalMorph wins 98% of the time vs DirectLiNGAM/ICA-LiNGAM on nSHD. The bottleneck was **detection recall** (~0.35 with v1-F). Stage-by-stage analysis showed `filter_peaks_by_local_step` killed 88% of raw peaks — the primary recall killer.

## What changed in v1-G

Three new functions added to `detectors_wavelets.py` (before the Piecewise Recalibration section):

### 1. `filter_peaks_by_local_step_adaptive()`
Replaces the fixed-threshold step validation. Scales `delta_k` inversely with peak strength:

```
peak_strength = |dE[tau]| / eps_threshold
discount = min(max_discount, 0.2 * (peak_strength - 1))
delta_k_eff = delta_k * (1 - discount)
```

- Peak 4x threshold: delta_k reduced by 60% (1.2 -> 0.48)
- Peak 2x threshold: delta_k reduced by 20% (1.2 -> 0.96)
- Peak barely above threshold: no reduction

Uses shorter pre/post windows (100 samples vs v1-F's 160) to reduce edge dead zones.

### 2. `_second_pass_peaks()`
After the first pass, gaps between consecutive CPs longer than `min_gap_ratio * mean_regime_len` likely contain a missed CP. Re-scans those gaps with:
- 0.6x eps thresholds
- 0.6x delta_k
- Relaxed channel gate (k-1, 0.8x delta_ch_k)
- Only keeps the single strongest peak per gap (avoids FP floods)

### 3. `detect_nonstationarity_v1G()`
Full detection pipeline. Steps 1-6 identical to v1-F. Changes:
- **Step 6b**: Adaptive step validation (replaces fixed)
- **Step 6c**: Channel consistency gate uses 160-sample windows (not the shorter step windows — per-channel estimates need longer windows for stable mean/std)
- **Step 8 (new)**: Two-pass rescue

## Tuning decisions

| Parameter | v1-F | v1-G | Rationale |
|-----------|------|------|-----------|
| `step_delta_k` | 1.5 | 1.2 | Lower base threshold; adaptive discount compensates for FPs |
| `step_pre_win` | 160 | 100 | Reduces edge dead zones from 234 to 174 samples |
| `ch_win` (channel gate) | 160 | 160 | Shorter windows made channel estimates too noisy — reverted after testing |
| `adaptive_max_discount` | — | 0.6 | Strong peaks get up to 60% reduction |
| `two_pass` | — | True | Rescues missed CPs in long gaps |
| `k_channels_min` | 2 | 2 | Lowering to 1 crashed precision at low p |

### Tested and rejected
- **ch_win=240**: Best precision (0.882) but recall dropped too much (0.525 vs 0.575)
- **step_delta_k=1.0**: Too many FPs that poisoned onset/offset proximity filter
- **k_channels_min=1**: Precision crashed at p=3 (0.469) and p=5 (0.531)
- **Strong-peak bypass (dE > 4x eps skips step test)**: Same precision crash
- **Removing onset/offset proximity filter**: Precision dropped without recall gain

## Integration

- `full_pipeline.py`: `detect_change_points()` defaults to `detector_version="v1G"`. Pass `"v1F"` for original. `run_full_pipeline()` exposes `detector_version` param.
- `batch_experiments.py`: `--detector v1F|v1G` CLI flag. RuntimeWarnings suppressed globally.
- `bilinear_example.py`: Switched to `detect_nonstationarity_v1G`.
- **Evaluation tolerance**: Changed from ±125 to ±200 samples throughout (matching the user's acceptance window requirement).

## Benchmark results

### 50-seed comparison at p=5, 5 regimes (±125 tolerance)

| Metric | v1-F | v1-G | Delta |
|--------|------|------|-------|
| F1 | 0.538 | **0.622** | +0.084 |
| Precision | **0.792** | 0.713 | -0.079 |
| Recall | 0.450 | **0.575** | +0.125 |
| Micro-F1 | 0.564 | **0.634** | +0.070 |
| Total TP | 90 | **115** | +25 |
| Total FP | **29** | 48 | +19 |

Win rate: v1-G wins 22/50, v1-F wins 11, 17 ties (67% excl. ties).

### 30-seed comparison across p values (±200 tolerance)

| p | F1 | Precision | Recall |
|---|-----|-----------|--------|
| 3 | 0.632 | 0.632 | 0.642 |
| 5 | 0.720 | 0.833 | 0.658 |
| 7 | 0.691 | 0.853 | 0.600 |

## Version history

| Version | Key feature | Default in pipeline since |
|---------|-------------|--------------------------|
| v1-F | Multi-moment CWT + k-of-channels gate + fixed step validation | 2026-03-25 |
| v1-G | Adaptive step + two-pass rescue | 2026-04-29 |
