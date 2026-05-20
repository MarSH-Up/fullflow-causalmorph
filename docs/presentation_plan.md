# Presentation Plan
## Causal Discovery in Non-Stationary Causal Graphical Models: Applications to Brain Effective Connectivity

> **Context for CoWork / Canva**
>
> - Target length: **25–30 minutes** talk + 10 questions (~40 min total)
> - Slide count: **15 content slides + 1 title** = 16 slides
> - Pacing: ~1.5–2 min per slide (detector section runs 2 min/slide)
> - All figures are in `presentation_figures/` at 150 DPI
> - Each slide entry has: title · bullets · `FIGURE:` citation · `POINT AT:` note · speaker cue

---

## Timing Budget

| Section | Slides | Minutes |
| --- | --- | --- |
| Brain context + PhD framework | 1–3 | 5 min |
| Problem + scenario | 4–5 | 4 min |
| Detector — Gatekeeper v1-F (novel) | 6–12 | 14 min |
| Causal recovery — CausalMorph | 13 | 3 min |
| Results + roadmap | 14–16 | 5 min |
| **Total** | **16** | **~31 min** |

---

## Figure Reference

| File | Dimensions | Orientation | Used on slide |
| --- | --- | --- | --- |
| `fig_02_rolling_moments.png` | 1820×1518 | tall | 7 |
| `fig_03_cwt_scalogram.png` | 1800×1366 | tall | 8 |
| `fig_04_surrogate_calibration.png` | 2079×738 | wide | 9 |
| `fig_06_multimoment_aggregation.png` | 1777×1450 | tall | 10 |
| `fig_07_derivative_detection.png` | 1776×1366 | tall | 11 |
| `fig_08_channel_gate.png` | 2076×885 | wide | 12 |
| `fig_09_detection_diagnostic.png` | 2085×2382 | very tall | 13 |
| `fig_10_timeseries.png` | 2385×2061 | very tall | 2 |
| `fig_11_structures_comparison.png` | 3360×1330 | very wide | 5, 14 |
| `fig_13_shd_metrics.png` | 2085×740 | wide | 15 |
| `fig_14_pipeline_architecture.png` | 1876×1112 | landscape | 4 |

> **Not used in main deck** (available as backup):
> `fig_01_morlet_wavelet.png` (replaced by fig_03),
> `fig_05_signed_energy_computation.png` (replaced by fig_06),
> `fig_12_consensus.png` (Bayesian aggregation omitted — not yet robust enough to present).

---

## Slide 01 — Title

**Title:** Causal Discovery in Non-Stationary Causal Graphical Models

**Subtitle:** Applications to Brain Effective Connectivity

**Sub-subtitle:** Progress report: Gatekeeper v1-F · CausalMorph integration

**[Name / Institution / Date]**

> *Speaker:* Leave on screen while people settle. No figure needed.

---

## Slide 02 — Why Brain Effective Connectivity?

**Title:** The Brain Changes — Causally

- **Effective connectivity** = the directed causal influence one brain region exerts on another
- Brain plasticity: connectivity rewires continuously during learning, recovery, ageing
- **fNIRS** (functional Near-Infrared Spectroscopy): measures hemodynamic responses in cortical regions
  — each channel = one brain region, the time series = oxygenation proxy for neural activity
- Non-stationarity is inherent: *"statistical properties of the signals change over time"* (Kim 2010)
- Foundational causal models (LiNGAM, PCMCI, PC) assume a **single fixed graph** — they cannot
  track a brain whose connectivity evolves over a session

**The core challenge:** detect when the connectivity changes AND recover what it becomes — from
the observed hemodynamic signals alone, without knowing the change times in advance.

**FIGURE:** `fig_10_timeseries.png`

> **Placement:** Full slide, crop to 3 variables. Relabel mentally: V1 = prefrontal, V2 = motor, V3 = parietal.

**POINT AT:**

- Coloured background bands — "each colour is a different connectivity regime, e.g. rest vs. task"
- Red dashed lines — "structural transitions: the graph changes here"
- Visible amplitude changes between bands — "the causal dependency structure is different in each regime"

> *Speaker:* "Think of each variable as a brain region and the signal as its fNIRS channel.
> The red lines are moments when the effective connectivity changes — perhaps the subject starts
> a new task, or a motor skill consolidates. Classical methods would estimate one graph and miss all of these."

---

## Slide 03 — PhD Framework and Today's Scope

**Title:** A Three-Pillar Framework for Non-Stationary Causal Discovery

**The unified goal (General Objective):** develop and validate a causal discovery framework for
non-stationary Dynamic Causal Bayesian Networks (nsDCBNs), applied to brain hemodynamic signals.

Three pillars, seven specific objectives:

| Pillar | Component | Objectives | Status |
| --- | --- | --- | --- |
| **Preconditioning** | CausalMorph — linearising transform that makes LiNGAM robust | O4 | ✓ integrated |
| **Detection** | Gatekeeper v1-F — wavelet multi-moment change-point detector | O1, O3 | ✓ built + tested |
| **Learning** | nsDCBN — causal model that adapts across detected regimes | O4, O5 | partial (LiNGAM per window) |

Remaining ahead: O2 (fNIRS neuro-dynamic simulator), O6 (noise/non-stationarity validation), O7 (real fNIRS data).

**Today's presentation:** O1 (synthetic generation) + O3 (Gatekeeper v1-F) + O4/O5 (CausalMorph + evaluation).

> **Placement:** Table-heavy slide, no figure. Use coloured rows or icons for the three pillars.

> *Speaker:* "My PhD has three components. Today I'll focus on the middle one — the change-point detector —
> because that is the novel algorithmic contribution. I'll also show the CausalMorph integration
> and give you quantitative results. The fNIRS-specific work and full nsDCBN are the next milestones."

---

## Slide 04 — Problem Statement

**Title:** Formal Problem: Non-Stationary Linear Causal Model

The observed multivariate time series X ∈ ℝ^{T×p} (p brain regions, T samples) is generated by
R sequential structural causal models with abrupt transitions at unknown times τ₁ < … < τ_{R-1}:

```
X_i(t) = Σ_{j ∈ PA_i^(r)}  w_ji^(r) · X_j(t)  +  ε_i^(r)(t)
```

- PA_i^(r) = causal parents of region i in regime r (the connectivity graph)
- w_ji^(r) = connection strength (changes across regimes)
- ε_i^(r) ~ non-Gaussian, i.i.d. — **required for LiNGAM identifiability** (Shimizu 2006)

Two simultaneous goals:

1. Detect change points {τ̂_k} from the signal alone
2. Recover causal graph Ĝ_r for each detected window [τ̂_k, τ̂_{k+1})

**FIGURE:** `fig_14_pipeline_architecture.png`

> **Placement:** Right 55% of slide. Equation + goals on the left.

**POINT AT:**

- The two pipeline stages (detect → recover)
- The "window" annotations in the figure — "each detected interval gets its own graph"

> *Speaker:* "The key difficulty is that we don't know when the changes happen.
> We have to figure out the segmentation and the causal structure simultaneously."

---

## Slide 05 — Synthetic Scenario (O1): Learning Trajectory

**Title:** O1 — Synthetic Generation: Brain-Inspired Learning Trajectory

- p = 5 variables (brain regions) · R = 5 regimes · 600–800 samples per regime (~3 500 total)
- Graph evolves from **G_init** (sparse, few connections) → **G_target** (denser, more connections)
  modelling how connectivity strengthens and reorganises during skill learning
- Edge changes scheduled uniformly: every consecutive pair of regimes differs by at least one edge
- Within each regime: edge weights + noise scale re-sampled independently
  → two sources of non-stationarity: **topological** (graph changes) + **parametric** (weights/noise)

**FIGURE:** `fig_11_structures_comparison.png`

> **Placement:** Use the **top row only** (true DAGs, blue nodes) cropped from the figure.
> The bottom row (learned) is used again on Slide 14.

**POINT AT:**

- Edges appearing/disappearing from left to right across the top row
- Node positions are fixed — "same brain regions, connectivity rewires"
- The progression from sparser to denser graph — "brain learning: more connections form over time"

> *Speaker:* "This is our synthetic ground truth. Each panel is one causal regime.
> In the brain analogy, this could be a motor task where the prefrontal-motor connectivity
> strengthens progressively across a training session."

---

## Slide 06 — Gatekeeper v1-F: Seven-Step Pipeline

**Title:** O3 — Structural Change Detection: Gatekeeper v1-F

The detector analyses the raw signal and outputs a list of change-point times:

| Step | Operation | Key idea |
| --- | --- | --- |
| 0 | Artifact rejection (Median + MAD) | Remove outliers before analysis |
| 1 | Rolling moments (mean, var, skew, kurt) | Capture distributional — not just mean — changes |
| 2 | CWT Morlet multi-scale | Localise changes in time *and* frequency |
| 3 | Fourier surrogate calibration | FWER-controlled thresholds under stationarity null |
| 4 | Signed log-ratio Z = log(𝒲/θ) | Symmetric: detects both up-shifts and down-shifts |
| 5 | dE derivative + find_peaks | Localise *transitions*, not sustained high-energy states |
| 6 | Step validation + K-of-channels gate (v1-F) | Two-stage false-positive rejection |

> **Placement:** Table fills the slide. No figure — let the audience read the table.
> Highlight rows 4 and 6 as the key innovations vs. prior versions.

> *Speaker:* "This is the algorithmic heart of today's talk. Each step addresses a specific
> failure mode of earlier detectors. I'll take you through the interesting ones in detail."

---

## Slide 07 — Step 1: Rolling Moments

**Title:** Why Track Four Moments?

- A structural change can shift **any** statistical moment, not just the mean
- In brain signals: adding an effective connection increases **variance** — the mean may not move
- In fNIRS: task onset changes **skewness** of the hemodynamic response before changing the mean
- We compute a causal rolling window W = 50 per channel:

| m | Statistic | When it responds to a causal change |
| --- | --- | --- |
| 1 | Mean | Shifts if edge adds a direct input |
| 2 | Variance | Increases when an edge is added (always) |
| 3 | Skewness | Sensitive to asymmetric noise distribution changes |
| 4 | Kurtosis | Sensitive to tail behaviour / outlier changes |

**FIGURE:** `fig_02_rolling_moments.png`

> **Placement:** Full slide or right 60%. If using text+figure split, show the table on the left.

**POINT AT:**

- Panel 1 (raw signal): "the change at t=350 looks like an amplitude increase"
- Panel 2 (M1, mean, red): "mean barely moves — a mean-only detector would miss this"
- Panel 3 (M2, variance, blue): "variance doubles sharply — this is the dominant signal"
- Panels 4–5 (M3/M4): "skewness and kurtosis also react, with smaller amplitude"
- The red dashed vertical line at t=350 across all panels

> *Speaker:* "If we only tracked the mean we would miss this change entirely.
> This matters in brain signals where connectivity changes often manifest first as
> variance changes in the hemodynamic response before any mean shift occurs."

---

## Slide 08 — Step 2: CWT — Multi-Scale Localisation

**Title:** Why Wavelets? Simultaneous Time and Frequency Resolution

- STFT uses a fixed window — no scale adaptivity for transient changes
- The Morlet CWT resolves a change point at **every scale simultaneously**
- A structural change = energy concentration at one time across **many scales** simultaneously
- Random noise artefacts = energy at one or two isolated scales only

Morlet wavelet (ω₀ = 6, compatible with hemodynamic signal frequencies):

```
ψ(t) = π^{-1/4} · exp(i·6·t) · exp(−t²/2)
```

Scalogram power: 𝒲(t, s) = |CWT(t, s)|²

**FIGURE:** `fig_03_cwt_scalogram.png`

> **Placement:** Full slide, three-panel figure displayed full-screen, walk top to bottom.

**POINT AT:**

- Top panel (raw signal): "variance step at t=400"
- Middle panel (rolling variance): "M² doubles at the change point"
- Bottom panel (scalogram): "the bright vertical band spans ALL scales at t=400 — a genuine change"
- The dark/quiet region before the change — "uniformly low energy under stationarity"
- A contrast: "noise would produce scattered bright pixels, not a band"

> *Speaker:* "The bright vertical band is the key visual signature of a structural change.
> Multi-scale consistency is what distinguishes a real connectivity shift from sensor noise,
> which is especially important in fNIRS where motion artefacts are common."

---

## Slide 09 — Step 3: Fourier Surrogate Calibration

**Title:** When Is Energy 'Too Large'? FWER-Controlled Thresholds

**Problem:** We need a per-scale null distribution — what does wavelet energy look like under stationarity?

**Solution:** Generate K = 100 Fourier surrogates from the baseline segment:

```
X̃^(k) = IFFT [ |FFT(X)| · exp(i·φ^(k)) ]    φ^(k) ~ Uniform(0, 2π)
```

- Surrogates preserve the **power spectrum** (autocorrelation structure) but randomise phase
- They are stationary by construction — no change point, same spectral content as the brain signal
- Threshold per scale: θ_s = Quantile_{1−α}{ time-quantile(𝒲_surrogate, 0.95) }
- α = 0.40 — permissive for multi-regime sensitivity; strict α would miss weak connectivity shifts

**FIGURE:** `fig_04_surrogate_calibration.png`

> **Placement:** Full slide, wide (2:1) figure across full width.

**POINT AT:**

- Left panel: "dark line = original baseline; coloured = three surrogates"
- Left panel: "same amplitude envelope, different temporal fluctuation — stationary by construction"
- Right panel (step plot): "one threshold value per CWT scale"
- Right panel: "small scales (high frequency) have higher thresholds — more noise at short timescales"

> *Speaker:* "The surrogates give us a data-driven null distribution that respects the autocorrelation
> structure of the brain signal. This is important because fNIRS has strong 1/f-like noise —
> a flat threshold would generate constant false alarms."

---

## Slide 10 — Step 4: Signed Log-Ratio Energy + Moment Weighting

**Title:** Combining Four Moments with Factorial-Inverse Weights

Signed log-ratio deviation for moment m, channel n, scale s:

```
Z_n^(m)(t,s) = log( 𝒲_n^(m)(t,s) / θ_{n,s}^(m) )
```

- Z > 0 → energy above baseline → **up-shift** (e.g. new connection forms)
- Z < 0 → energy below baseline → **down-shift** (e.g. connection weakens)

Aggregate with **factorial-inverse weights** — more stable moments weighted higher:

```
E_signed(t) = Σ_n  Σ_m (1/m!)  Σ_s  [ Z_n^(m+) − Z_n^(m−) ]
```

Weights: w₁ = 1.0 (mean) · w₂ = 0.5 (variance) · w₃ ≈ 0.17 (skewness) · w₄ ≈ 0.04 (kurtosis)

**FIGURE:** `fig_06_multimoment_aggregation.png`

> **Placement:** Full slide, tall figure displayed full-screen.

**POINT AT:**

- Top-left (bar chart): "factorial weights — mean is most trusted, kurtosis least"
- Top-right: "the four moment energies at the change point — variance (blue) leads"
- Middle/bottom panels: "each moment's signed energy — they all spike near t=350"
- Why combining helps: "if mean doesn't react but variance does, the combined signal still fires"

> *Speaker:* "The symmetric log-ratio means we can detect both increases AND decreases in connectivity.
> In a learning paradigm, some connections strengthen while others weaken simultaneously —
> a one-sided detector would miss half the picture."

---

## Slide 11 — Step 5: Derivative-Based Detection

**Title:** Detect Transitions, Not Sustained States

**The problem with thresholding E_signed directly:**

In a sustained high-connectivity regime E_signed stays elevated → continuous false alarms every sample.

**Solution:** differentiate and detect peaks in the derivative:

```
ΔE(t) = E_signed(t) − E_signed(t−1)    [after smoothing, window = 12]
```

- ΔE spikes **up** at regime onsets (connectivity increase)
- ΔE spikes **down** at regime offsets (connectivity decrease)
- Peak detection via `scipy.signal.find_peaks`: height = ε (calibrated on surrogates), refractory = 150

**FIGURE:** `fig_07_derivative_detection.png`

> **Placement:** Full slide, three-panel figure displayed full-screen.

**POINT AT:**

- Top (E_signed): "energy stays high after the change — thresholding here re-triggers every sample"
- Middle (ΔE): "the derivative is large only AT the transition — perfect temporal localisation"
- Threshold lines (dotted): "calibrated from surrogate peak heights — less conservative than max-stat"
- Bottom panel: "find_peaks identifies the spike; red triangle = detected change point"
- Agreement between green (true CP) and red triangle

> *Speaker:* "This design choice — differentiating before thresholding — prevents the
> cascading false positives that plagued earlier detector versions on our synthetic data.
> It is also consistent with how electrophysiological event detectors work in neuroscience."

---

## Slide 12 — Step 6b: K-of-Channels Gate (v1-F — New)

**Title:** v1-F: Rejecting Single-Channel Artefacts

fNIRS-specific motivation: motion artefacts, optode coupling noise, and hair occlusion
affect individual channels. We want a change that is **distributed across brain regions**.

For each candidate peak τ, compute per-channel step z-score:

```
z_n(τ) = ( median(E_n[post]) − median(E_n[pre]) ) / σ̂_n
```

**Gate:** accept peak only if ≥ k_ch = 2 channels satisfy |z_n| ≥ 1.5

Also reports **concentration ratio** R(τ) = max|Δ_n| / Σ|Δ_n|:

- R ≈ 1/N → distributed change (genuine connectivity shift)
- R ≈ 1 → single-channel artefact → **rejected**

**FIGURE:** `fig_08_channel_gate.png`

> **Placement:** Full slide, wide (2:1) figure across full width.

**POINT AT:**

- Left: "green channels step up clearly; grey channels are flat — noise or artefact"
- Left: "3 of 5 channels are active — this peak PASSES the gate"
- Right (z-scores bar): "dashed line = threshold k_z = 1.5"
- Concentration ratio in the subtitle — "R = 0.31, close to 1/5 = 0.20 — distributed change"

> *Speaker:* "This gate was specifically designed with fNIRS in mind.
> A single optode picking up a heartbeat artefact can look like a strong change in the aggregate.
> Requiring consensus across channels makes the detector much more robust to sensor-level noise."

---

## Slide 13 — Full Detection Result (O3)

**Title:** Gatekeeper v1-F on a 5-Regime Signal

**FIGURE:** `fig_09_detection_diagnostic.png`

> **Placement:** Full slide, 8-panel very tall figure (2085×2382 px) displayed full-screen.
> Guide the audience panel by panel — they cannot read everything at once.

**POINT AT (top to bottom):**

- Panel 1 (signals): "five variables; green dotted = detected CPs, red solid = ground truth"
- Panel 2 (ΔE): "the detection signal; spikes align with regime transitions"
- Panel 3 (E_signed): "aggregate signed energy — clear level steps at each transition"
- Panels 4–7 (per-moment): "variance (blue) dominates; skewness and kurtosis add secondary evidence"
- Panel 8 (severity): "leaky-integrator severity — accumulated evidence of non-stationarity"
- Detection outcome: "3 of 4 change points detected (~75%) when running Gatekeeper v1-F; oracle baseline: 3/3 (100%)"

> *Speaker:* "The overall picture: the detector fires correctly at the first and last transition.
> It misses the middle ones because they are weaker structural changes — only one edge swaps.
> This sets up the open problem we'll discuss at the end."

---

## Slide 14 — O4 — CausalMorph: Iterative Warm-Start

**Title:** O4 — Non-Stationary Causal Learning: CausalMorph Integration

**Challenge:** cold DirectLiNGAM on each detected window gives SHD ≈ 0.5, F1 ≈ 0.0.

**CausalMorph solution:** pre-whiten data using the previous regime's structure:

```
X' = (I − B̂^(r-1)) · X
```

X' is closer to the noise residuals — LiNGAM identifies the causal order more reliably even
when the new regime's graph differs only partially from the prior.

Warm-start chain across detected windows:

```
(π⁽⁰⁾, B̂⁽⁰⁾) →CM→ (π⁽¹⁾, B̂⁽¹⁾) →CM→ (π⁽²⁾, B̂⁽²⁾) →CM→ …
```

- Window 0 prior: **ground-truth regime 0** (simulates baseline connectivity known from a resting-state scan)
- Window k ≥ 1 prior: previous window's DirectLiNGAM output

**FIGURE:** `fig_11_structures_comparison.png`

> **Placement:** Full slide, very wide figure across full width (both rows).

**POINT AT:**

- Top row (blue): "true causal graphs — ground-truth connectivity per regime"
- Bottom row (green): "learned graphs — what CausalMorph + LiNGAM recovers"
- Edge labels on the bottom row — "coefficient values; sign and magnitude"
- nSHD values in the subtitles — "normalised SHD, lower is better"
- Note where topology roughly matches vs. where it diverges

> *Speaker:* "In a real fNIRS setting, the window-0 prior would come from a resting-state scan
> at the beginning of the session — a standard protocol in neuroimaging.
> Subsequent windows are then warm-started from the previous window's estimate,
> so the model tracks the evolving connectivity without refitting from scratch."

---

## Slide 15 — O5 — Results: SHD and Detection

**Title:** O5 — Performance Evaluation

**Change-point detection (tolerance ±125 samples):**

- Oracle (ground-truth CPs): **3 / 3 detected (100%)** — upper bound on causal recovery quality
- Gatekeeper v1-F (real detector): **~75%** of change points detected in earlier runs

**Causal structure recovery — oracle segmentation (p=4, 4 regimes):**

| Regime | Window | nSHD |
| --- | --- | --- |
| 0 | [0 : 880] | **0.000** |
| 1 | [880 : 1763] | 0.167 |
| 2 | [1763 : 2740] | 0.333 |
| 3 | [2740 : 3687] | 0.083 |
| **Mean** | | **0.146** |

nSHD ∈ [0, 1] — 0 = perfect recovery, 1 = worst possible.

**FIGURE:** `fig_13_shd_metrics.png`

> **Placement:** Full slide, wide figure across full width.

**POINT AT:**

- SHD bars per regime: "lower is better; regime 0 is perfect (warm-started from ground truth)"
- Regime 2 highest SHD (0.333): "most complex window — 3 edges, prior from regime 1 which itself had error"
- Regime 3 recovers to 0.083 despite 4 edges — "warm-start chain stabilises as more data accumulates"
- Mean nSHD = 0.146: "under oracle segmentation, the causal learning step is the binding constraint"

> *Speaker:* "These results use oracle change points — the detector is bypassed and the true
> boundaries are fed directly to CausalMorph. This isolates causal recovery quality from detection error.
> Even with perfect segmentation, regime 2 shows the highest error: three active edges and a
> propagated prior from a noisier earlier window. The good news is regime 3, the most connected
> regime, recovers well — the warm-start chain benefits from accumulating structure over time."

---

## Slide 16 — Summary and PhD Roadmap

**Title:** Summary and Next Steps

**What was built and validated (today):**

- **O1 ✓** Generic DAG engine: multi-regime learning trajectory with controllable ground truth
- **O3 ✓** Gatekeeper v1-F: wavelet multi-moment detector (surrogate calibration, signed energy,
  derivative peaks, K-of-channels consistency gate)
- **O4 ✓ (partial)** CausalMorph + DirectLiNGAM iterative warm-start per detected window
- **O5 ✓** Evaluation on synthetic data: nSHD per regime + % change points detected

**Open limitations (current synthetic setting):**

- Detector: ~75% of CPs detected; weak single-edge transitions are sometimes missed
- CausalMorph (oracle segmentation): mean nSHD = 0.146; regime 2 worst at 0.333 (prior propagation error)
- Warm-start chain is sensitive to one poorly estimated window
- Only abrupt transitions tested

**PhD roadmap — what comes next:**

- **O2** Neuro-dynamic fNIRS simulator: hemodynamic response function, realistic noise model
- **O3 continued** Improve detector sensitivity for weak changes; test on smooth transitions
- **O6** Systematic validation: noise levels, non-stationarity intensity, edge density
- **O7** Real fNIRS data: observational assessment of brain effective connectivity

> **No figure on this slide.**

> *Speaker:* "The pipeline works end-to-end on synthetic data.
> The immediate next milestone is O2 — building an fNIRS-specific simulator
> so we can test on signals that match the spectral and noise characteristics of real hemodynamic data.
> That will also let us validate the surrogate calibration, which currently uses a generic Fourier model."

---

## Anticipated Q&A

**Q1: How does this relate to CD-NOD and other non-stationary causal discovery methods?**

CD-NOD detects non-stationarity by testing whether residuals from a causal model depend on an
auxiliary time variable. Our approach is complementary: we first segment the time series using
wavelet energy, then fit a separate causal model per segment. CD-NOD detects the *existence* of
non-stationarity; we additionally *localise* it and *recover the structure per regime*.
A quantitative comparison on shared benchmarks is an important open item (O6).

**Q2: Why fNIRS rather than fMRI or EEG?**

fNIRS is portable, tolerates movement, and is cheaper — making it suitable for naturalistic learning
paradigms (e.g., recording a subject practising a motor skill at a desk). The hemodynamic signal
is slower than EEG but more spatially resolved than scalp EEG, and it naturally produces the
multi-channel regime-changing signals our framework targets (O7).

**Q3: Why LiNGAM and not a Bayesian or score-based method?**

LiNGAM exploits non-Gaussianity to identify a unique causal graph without requiring a search over
DAG space. It is O(p³) per window and gives a point estimate — fast enough to refit at every
detected change point. Score-based methods (GES, NOTEARS) would be more flexible but are slower
and require a different identifiability argument.

**Q4: How sensitive is the detector to α = 0.40?**

α controls the surrogate quantile used for thresholding. Lower α is stricter: reduces false positives
but also misses weaker changes. 0.40 was tuned for the multi-regime synthetic scenario in
run_experiment.py. For real fNIRS data, α should be calibrated on a known-stationary resting-state
baseline from each participant (O6).

**Q5: What is the CausalMorph transformation doing geometrically?**

X' = (I − B̂)X subtracts the predicted contribution of each node's parents from its own signal.
In the ideal case, X' ≈ noise residuals: i.i.d., non-Gaussian, zero-mean.
LiNGAM operates best on residuals, so this pre-whitening makes the causal order recovery easier
even when the new regime's graph differs only slightly from the prior (O4).

**Q6: Why does the detector miss the first change point?**

The baseline is the first 50% of regime 0. The first transition occurs at the end of regime 0,
close to the end of the baseline window. The surrogate thresholds are calibrated on the *full*
baseline, so local energy increases near the boundary have lower power. A held-out pre-session
resting-state baseline (standard in fNIRS protocols) would fix this (O3 improvement, O7).

**Q7: Does the K-of-channels gate break down for low channel counts?**

The gate is auto-disabled for N = 1 (single-channel case) and clamped to max(1, N//2) for small N.
For N = 2 the effective threshold becomes 1 channel — equivalent to no gate. In practice fNIRS
devices have 16–64 channels in the regions of interest, so the gate operates at full strength (O7).

**Q8: Why factorial-inverse weights rather than learned weights?**

They are a principled non-parametric choice: lower moments are more statistically stable at small
window sizes, so they naturally deserve higher weight. Learning the weights would require labelled
regime-change data, which defeats the purpose of an unsupervised detector. The 1/m! schedule is
also interpretable and reproducible across datasets (O3).

**Q9: What does the Window-0 prior represent in a real fNIRS session?**

In a real application, Window-0 prior = connectivity estimated from a resting-state scan before
the task begins — a standard protocol in neuroimaging. The warm-start chain then tracks how
connectivity evolves from that baseline across the session (O4, O7).

**Q10: What metric will you use when there is no ground truth on real data?**

Without ground truth we shift to: (a) test-retest reliability across repeated sessions,
(b) spatial consistency with known anatomy (e.g., does prefrontal → motor edge appear during
motor tasks?), and (c) comparison with fMRI functional connectivity as an indirect validation.
This is the focus of O7.
