# FullFlow Session Summary — 2026-03-25

## Goal

Build `full_pipeline.py`: an end-to-end script that integrates two existing
research projects into a single non-stationary causal discovery pipeline.

| Project | Location | Purpose |
|---|---|---|
| **NSD_Wavelets** | `NSD_Wavelets/src/` | Wavelet-based non-stationarity detection (Gatekeeper v1-E) |
| **CausalMorph** | `causalmorph/` | Linearising transformation for better causal structure learning |

---

## What Was Built

**File:** `FullFlow/full_pipeline.py`

A pipeline that:

1. **Generates** a multi-regime non-stationary causal time series (linear SCM, normal noise, chained DAGs with controlled structural changes).
2. **Detects** change points using the wavelet-based multi-moment detector (v1-E).
3. **Extracts** the causal structure from each detected data window using CausalMorph + DirectLiNGAM, iteratively warm-starting each window from the previous one's output.
4. **Evaluates** every extracted structure against ground truth using SHD, F1, Precision, Recall.
5. **Plots** 6 figures: detection diagnostics, time series, true structures, learned structures, adjacency heatmaps, SHD bar charts.

---

## Pipeline Architecture

```
                        Non-Stationary Signal (multi-regime)
                                    |
                        [1] build_nonstationary_scenario()
                                    |
                          numpy array X [T, p]
                                    |
                        [2] detect_change_points()
                        (v1-E multimoment wavelet detector)
                                    |
                          DetectionResult
                          .change_points = [cp1, cp2, ...]
                                    |
                        [3] extract_causal_structures()
                                    |
                 +------------------+------------------+
                 |                  |                  |
           Window 0            Window 1           Window k
          [0 : cp1)          [cp1 : cp2)        [cp_{k-1} : T)
               |                  |                  |
     initial_adj/order    prev regime's adj    prev regime's adj
        (ground truth)     (from window 0)     (from window k-1)
               |                  |                  |
          CausalMorph        CausalMorph        CausalMorph
               |                  |                  |
          DirectLiNGAM       DirectLiNGAM       DirectLiNGAM
               |                  |                  |
         RegimeStructure    RegimeStructure    RegimeStructure
               |                  |                  |
                 +------------------+------------------+
                                    |
                        List[RegimeStructure]
                            (all stored)
                                    |
                        [4] compute_shd() per window
                        [5] print_summary()
                        [6] 6 plots
```

---

## Key Data Structures

### RegimeStructure (dataclass)

```python
@dataclass
class RegimeStructure:
    regime_idx: int                          # 0-based window index
    window_start: int                        # first sample index
    window_end: int                          # last sample index (exclusive)
    n_samples: int                           # window_end - window_start
    causal_order: List[int]                  # topological order (col indices)
    adjacency_matrix: pd.DataFrame           # weighted adj from DirectLiNGAM
    used_prior: bool                         # True if warm-started from prev window
    true_regime_idx: int                     # which true regime has max overlap
    true_adjacency_matrix: pd.DataFrame      # binary ground-truth adj
    shd_metrics: Dict[str, Any]              # SHD, F1, Precision, Recall, TP/FP/FN...
```

### Pipeline return dict

```python
{
    "structures":              List[RegimeStructure],
    "detected_change_points":  List[int],
    "true_change_points":      List[int],
    "X":                       np.ndarray,          # [T, p]
    "variable_names":          List[str],            # ["V1", ..., "V5"]
    "detection_result":        DetectionResult,      # full v1-E result
}
```

---

## Scenario Configuration

Mirrors `run_realistic_scenario` from `NSD_Wavelets/src/evaluation/run_experiment.py`.

| Parameter | Value |
|---|---|
| p (variables) | 5 |
| n_regimes | 5 (4 change points) |
| samples per regime | random in [600, 800] |
| mode | `"linear"` (Pearl's linear SCM) |
| noise distribution | `["normal"] * p` |
| deviation | `0.5 + rng.uniform(0, 1)` per regime |
| signal_strength | `1.2 + rng.uniform(0, 0.6)` per regime |
| base_pconn | 0.35 |
| change_pcts | `[0, 30, 25, 35, 30]` (% structural change per transition) |
| DAG chaining | each regime's graph = previous graph + change_pct% mixed edge changes |

---

## Detector Configuration

Uses `detect_nonstationarity_multimoment` (Gatekeeper v1-E).

| Parameter | Value | Why |
|---|---|---|
| min_scale | 3.0 | standard |
| moments | [1, 2, 3, 4] | mean + variance + skewness + kurtosis |
| moment_window | 50 | rolling window for moment computation |
| n_surrogates | 100 | good calibration accuracy |
| alpha | 0.40 | permissive (40th percentile) for multi-regime |
| k_scales_min | 1 | minimum scales required (low = more sensitive) |
| smooth_window | 12 | moderate smoothing before differentiation |
| min_snr | 0.3 | low SNR threshold for subsequent changes |
| refractory_period | min(150, min_regime_len // 4) | prevents double-counting |
| baseline | 50% of first regime | calibration window |

---

## CausalMorph Integration

### Initial prior (Window 0)

The first window is bootstrapped with the **ground-truth** adjacency matrix and
topological order from true regime 0. This simulates having domain knowledge
or a known baseline structure. Without this, cold DirectLiNGAM on raw
nonlinear/mixed data produces poor priors that cascade into all subsequent
windows.

```python
first_graph = scenario["regimes"][0]["graph"]
initial_order = [variable_names.index(v) for v in nx.topological_sort(first_graph)]
initial_adj   = (first_regime["adj_matrix"] != 0).astype(float)
```

### Iterative warm-start (Window k, k >= 1)

Each subsequent window receives `(causal_order, adjacency_matrix)` from the
**previous window's** DirectLiNGAM fit on CausalMorph-transformed data.
CausalMorph uses these to:
- Set the linearisation order (Taylor expansion follows causal order)
- Know parent sets for residual computation
- Generate better non-Gaussian synthetic residuals

This means the quality of structure learning can **improve** over windows as
the prior gets refined, or **degrade** if a window's structure is badly
estimated. The ground-truth bootstrap for window 0 helps anchor the chain.

### Per-window flow

```
window_df (raw data) + prior (adj, order)
        |
    causalMorph(window_df, causal_order=prior_order, adjacency_matrix=prior_adj)
        |
    transformed_df (linearised data, better for LiNGAM)
        |
    DirectLiNGAM.fit(transformed_df)
        |
    extracted: causal_order_, adjacency_matrix_
        |
    -> stored in RegimeStructure
    -> becomes prior for next window
```

---

## Evaluation

### SHD computation

Each detected window is mapped to the true regime with maximum sample overlap
(`assign_true_regime`). Then `mycomparegraphs` (from `causalmorph/utils/metrics.py`)
computes:

- **SHD** (Structural Hamming Distance): FP + FN + reversed edges
- **normalized_shd**: SHD / (p * (p-1))
- **F1, Precision, Recall**: edge-level classification metrics
- **TP, FP, FN, TN**: confusion matrix components
- **MCC**: Matthews correlation coefficient

The predicted adjacency is binarised at threshold 0.05 before comparison.

### Detection accuracy

Change point matching uses a 200-sample tolerance window. Per-GT errors are
printed with OK (<=100), LARGE (<=200), MISSED (>200) tags.

---

## Plots (6 figures)

1. **Detection diagnostics** (8-panel, from run_experiment.py):
   - Signal with GT/onset/offset markers
   - dE(t) derivative with eps thresholds
   - E_signed(t) = E_pos - E_neg
   - Per-moment energy (mean, variance, skewness, kurtosis)
   - Forgetting severity trace

2. **Time series**: all variables stacked, true CPs (red dashed) vs detected (green dotted), coloured regime backgrounds.

3. **True structures**: grid of ground-truth DAGs per regime.

4. **Learned structures**: grid of extracted graphs (green nodes, weighted edge labels).

5. **Adjacency heatmaps**: RdBu colourmap with annotated values per extracted window.

6. **SHD metrics**: 2-panel bar chart — raw SHD (left) + grouped bars for norm-SHD / F1 / Precision / Recall (right).

---

## Iteration History (what changed during the session)

### v1 — Initial skeleton
- Created `full_pipeline.py` with `build_nonstationary_scenario`, `detect_change_points`, `extract_causal_structures`, `run_full_pipeline`.
- Used `detect_nonstationarity` (v1-D, single energy).
- Used `mode="nonlinear"`, `dist=["laplace"]`.
- Cold DirectLiNGAM bootstrap for first window.

### v2 — Fix scenario key
- `scenario["X"]` -> `scenario["combined_data"].values` (the actual dict key).
- Variable names: `V0`... -> `V1`... (matching 1-indexed convention).
- Proper DAG chaining via `base_graph=dags[-1]`.

### v3 — Add plots + SHD
- Added 5 plot functions: timeseries, true structures, learned structures, adjacency heatmaps, SHD bar chart.
- Added `RegimeStructure.shd_metrics`, `assign_true_regime`, `compute_shd`.
- Added `print_summary` with per-regime edge list and SHD breakdown.

### v4 — Tune detector sensitivity
- Increased alpha, lowered k_scales_min and min_snr.
- Used 70% of first regime as baseline.
- Result: detector found fewer changes (went from 1 to 0 — wrong direction).

### v5 — Switch to v1-E multimoment detector
- Changed from `detect_nonstationarity` to `detect_nonstationarity_multimoment`.
- Matched params from `run_multi_regime_experiment_multimoment` in run_experiment.py.
- Still not detecting well because scenario used nonlinear+laplace data.

### v6 — Match run_realistic_scenario exactly
- Scenario: `mode="linear"`, `dist=["normal"]`, varying deviation/signal_strength.
- Detector: `alpha=0.40`, `k_scales_min=1`, `min_snr=0.3`, `min_scale=3.0`, `smooth_window=12`, `n_surrogates=100`.
- Added `plot_detection_diagnostics` (8-panel diagnostic from run_experiment.py).
- Returns full `DetectionResult` for plotting.
- Default params: `p=5`, `n_regimes=5`, `min_samples=600`, `max_samples=800`.
- **Detection now works**: 3/4 change points detected.

### v7 — Ground-truth bootstrap for CausalMorph
- Cold DirectLiNGAM was producing poor initial priors (SHD 0.5, F1 0.0).
- Changed `extract_causal_structures` to accept `initial_adj` and `initial_order`.
- Pipeline passes regime 0's ground-truth graph as the initial prior.
- Subsequent windows still use previous window's extracted structure (iterative).

---

## Open Items / Next Steps

1. **First change point missed**: GT=617 is not detected (error=808). The detector
   calibrates on the first regime, so the transition at the end of that regime is
   absorbed into the baseline. Consider using a shorter baseline or a different
   calibration strategy.

2. **CausalMorph quality degrades over windows**: SHD tends to increase in later
   windows. Investigate whether feeding back the _binarised_ adj (instead of raw
   weighted) helps, or whether re-running cold LiNGAM as a secondary check would
   improve robustness.

3. **Nonlinear scenarios**: The current config uses `mode="linear"` because the
   detector was calibrated for that. To handle nonlinear SCMs, the detector may
   need recalibration or the CausalMorph transformation should be applied _before_
   detection (pre-linearise, then detect).

4. **Real-world prior**: Currently uses ground-truth adj for window 0. In practice,
   this would come from domain knowledge, a separate observational study, or a
   preliminary LiNGAM run on a known-stable baseline period.

5. **Evaluation on more seeds**: Run across multiple seeds/configs to get aggregate
   SHD/F1 statistics instead of single-seed results.

6. **Smooth transitions**: All current experiments use `transition_type="abrupt"`.
   The scenario generator supports `"smooth"` — test whether the detector handles
   gradual transitions.

---

## File Layout

```
FullFlow/
  full_pipeline.py          <- main script (everything in one file)
  NSD_Wavelets/
    src/
      detectors/
        detectors_wavelets.py      <- detect_nonstationarity_multimoment (v1-E)
      scenarios/
        NonStationaryCausalScenarios.py  <- scenario generation
      evaluation/
        run_experiment.py           <- reference experiment configs
        metrics.py                  <- evaluate_detection, simplify_transitions
  causalmorph/
    core/
      causalmorph_algorithm.py     <- causalMorph function
    utils/
      metrics.py                   <- mycomparegraphs, normalized_shd
    data_generation/
      synthetic_scenarios.py       <- causal_graph_synthetic_scenarios
    basic_usage.py                 <- reference usage example
  docs/
    2026-03-25_session_summary.md  <- this file
```

---

## How to Run

```bash
cd /Users/mariodelossantos/Desktop/Research/PhD/Code/FullFlow
python full_pipeline.py
```

Default config: p=5, 5 regimes, 600-800 samples/regime, seed=42, all plots enabled.

To customise:

```python
from full_pipeline import run_full_pipeline

results = run_full_pipeline(
    p=5,
    n_regimes=5,
    min_samples=600,
    max_samples=800,
    base_pconn=0.35,
    change_pcts=[0, 30, 25, 35, 30],
    seed=42,
    verbose=True,
    show_plots=True,
)

# Access all extracted structures
for s in results["structures"]:
    print(s.regime_idx, s.shd_metrics["SHD"], s.shd_metrics["F1"])
```

---

## Key Imports / Dependencies

- `lingam` (DirectLiNGAM)
- `numpy`, `pandas`, `matplotlib`, `networkx`
- `scipy` (used internally by both projects)
- `numba` (used by mycomparegraphs for fast graph metrics)
- `sklearn` (Matthews correlation coefficient)
