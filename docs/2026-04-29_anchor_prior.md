# 2026-04-29 — Experiment log: anchor prior + hybrid warm/cold

Tracking doc for the change motivated by the 2026-04-29 batch results
(`batch_results.csv`, 50 runs, p=3..7, 10 seeds each).

---

## Motivation

Per-p means from the batch run:

| p | det_F1 | struct_F1 | consensus_F1 | mean_nSHD |
|---|--------|-----------|--------------|-----------|
| 3 | 0.50   | 0.39      | 0.22         | 0.23      |
| 4 | 0.58   | 0.40      | 0.19         | 0.18      |
| 5 | 0.50   | 0.49      | 0.49         | 0.20      |
| 6 | 0.71   | 0.39      | 0.40         | 0.21      |
| 7 | 0.68   | 0.38      | 0.44         | 0.23      |
| **all** | **0.59** | **0.41** | **0.35** | **0.21** |

**Smoking gun — perfect detection still produces poor structure:**

| seed     | p | det_F1 | struct_F1 | consensus_F1 |
|----------|---|--------|-----------|--------------|
| ...104   | 5 | 1.00   | 0.64      | 0.25         |
| ...118   | 6 | 1.00   | 0.19      | 0.17         |
| ...125   | 7 | 1.00   | 0.41      | 0.59         |
| ...129   | 7 | 1.00   | 0.31      | 0.12         |

Mean struct_F1 across the 4 perfect-detection seeds ≈ 0.39 — **identical to the
overall mean of 0.41**. So fixing detection alone wouldn't move the structural
result; there's a separate ceiling imposed by the warm-start chain.

This was already flagged in `NSD_Wavelets/docs/session-2026-04-09.md`:
> "CausalMorph warm-start may be hurt by accumulated errors in the prior chain
> when earlier regimes have different topology."

## Hypothesis

The chain `(π⁰, B̂⁰) → CausalMorph → (π¹, B̂¹) → CausalMorph → …` propagates
any error in an early regime through every subsequent window. The strong
`true_regime_idx=0 → SHD=0` pattern observed when the GT prior is used for
window 0 stops the moment the prior gets replaced by a noisy estimate.

## Change

Added a `prior_mode` parameter to `extract_causal_structures`:

- **`"anchor"` (new default)** — every window receives the same prior
  `(initial_order, initial_adj)` (the regime-0 ground truth, or the cold
  bootstrap if no GT is given). No information flows between windows.
- **`"chain"`** — previous behaviour preserved for comparison: window k uses
  the previous window's extracted `(causal_order, adj_matrix)`.

Threaded the flag through:

1. `extract_causal_structures(..., prior_mode="anchor")` — `full_pipeline.py`
2. `run_full_pipeline(..., prior_mode="anchor")` — `full_pipeline.py`
3. `run_one(..., prior_mode="anchor")` and `run_batch(..., prior_mode="anchor")`
   — `batch_experiments.py`
4. `--prior_mode {anchor,chain}` CLI flag on `batch_experiments.py`

`__main__` of `full_pipeline.py` now passes `prior_mode="anchor"` explicitly.

## Files modified

| File | Change |
|------|--------|
| `full_pipeline.py` | `extract_causal_structures` gains `prior_mode` (anchor/chain); anchor mode reuses initial prior for every window; if cold-started, the cold result becomes the anchor. `run_full_pipeline` exposes the flag. `__main__` defaults to `"anchor"`. |
| `batch_experiments.py` | `run_one` and `run_batch` accept `prior_mode`; new `--prior_mode` CLI argument; banner prints active mode. |
| `docs/2026-04-29_anchor_prior.md` | This file. |

## Paired comparison results (50 rows, batch_results.csv vs batch_anchor.csv)

| metric | chain | anchor | Δ(a−c) | anchor wins | ties | chain wins |
|--------|-------|--------|--------|-------------|------|------------|
| det_F1 | 0.594 | 0.593 | −0.001 | 5 | 39 | 6 |
| mean_norm_shd | 0.208 | **0.201** | −0.007 | 21 | 10 | 19 |
| mean_struct_F1 | 0.409 | **0.421** | +0.012 | 18 | 14 | 18 |
| consensus_norm_shd | 0.412 | 0.412 | −0.001 | 18 | 14 | 18 |
| **consensus_F1** | **0.348** | 0.303 | **−0.045** | 10 | 17 | **23** |

Per-p mean_norm_shd: anchor wins p=3 (−0.052), p=4 (−0.030), p=7 (−0.006);
loses p=5 (+0.036), p=6 (+0.017).

**Conclusion:** anchor is not a free win. It modestly improves per-window nSHD
but hurts consensus_F1 by −0.045. The chain's diversity of per-regime votes is
useful for consensus aggregation — anchor collapses all windows onto the
regime-0 prior, starving the aggregation of the signal about how the structure
evolved. **Default reverted to `"chain"`; `"anchor"` retained as opt-in.**

---

## Experiment 2: hybrid warm/cold selection (option B) — FAILED

### Design
Per window: fit both warm (CausalMorph + LiNGAM on transformed data) and cold
(DirectLiNGAM on raw data). Score both by `Σ|corr(eᵢ, eⱼ)|` (residual
independence, lower = better). Keep cold only if it beats warm by > 0.02.

### Result (8 seeds, p=5, n_regimes=5)

| mode   | mean_nSHD | struct_F1 | cons_F1 | cold% |
|--------|-----------|-----------|---------|-------|
| chain  | 0.262     | 0.399     | 0.383   | n/a   |
| hybrid | 0.433     | 0.137     | 0.320   | ~90%  |

Hybrid is strongly WORSE. Cold wins ~90% of windows, but picking cold degrades
all metrics substantially.

### Root cause
The residual independence criterion is fundamentally biased against the warm
fit: DirectLiNGAM literally minimizes residual dependence as its objective on
whatever data it receives. The cold fit was trained on raw data → its B
minimizes residual corr in raw space by construction. The warm fit's B was
estimated on CausalMorph-transformed data → when scored in raw space it looks
worse, even if it captures a more accurate structure. Scoring both in raw space
is not a fair comparison.

**Hybrid disabled (`hybrid=False` default).** Code retained for reference.

### What a working hybrid would need
A score that is data-space-agnostic: e.g., cross-validated reconstruction on a
held-out time slice, or a structural-disagreement penalty (if warm and cold
agree, use warm; if they disagree strongly and the window is long, use cold).
Both require more design work than their expected payoff.

---

## Files modified

| File | Change |
|------|--------|
| `full_pipeline.py` | `prior_mode` (anchor/chain), `hybrid` flag, `_residual_indep_score` helper, `RegimeStructure.{chosen,score_warm,score_cold}`. Defaults: `prior_mode="chain"`, `hybrid=False`. |
| `batch_experiments.py` | `prior_mode`, `hybrid`, `--hybrid` CLI flag. Defaults match pipeline. |
| `batch_anchor.csv` | 50-run anchor batch for paired comparison. |
| `docs/2026-04-29_anchor_prior.md` | This file. |

---

## Experiment 3: k_channels_min=2 + hybrid onset policy — POSITIVE on SHD

### Changes
- `k_channels_min`: 1 → 2 (restore original v1-F default; require ≥2 channels to agree)
- CP assembly: onset-only + two exceptions (isolated offsets with no nearby onset;
  offsets that precede all onsets — edge-removal transitions)

### Result (50 runs, same seeds as baseline)

| p | old nSHD | new nSHD | Δ | old struct_F1 | new struct_F1 | Δ |
|---|----------|----------|---|---------------|---------------|---|
| 3 | 0.226 | 0.226 | 0 | 0.390 | 0.390 | 0 |
| 4 | 0.177 | **0.132** | **−0.045** | 0.396 | **0.615** | **+0.219** |
| 5 | 0.203 | **0.168** | **−0.035** | 0.490 | **0.518** | **+0.028** |
| 6 | 0.209 | 0.191 | −0.018 | 0.390 | 0.362 | −0.028 |
| 7 | 0.228 | 0.215 | −0.013 | 0.379 | 0.338 | −0.041 |
| **ALL** | 0.209 | **0.187** | **−0.022** | 0.409 | **0.444** | **+0.035** |

det_F1 all: 0.594 → 0.461 (recall 0.670 → 0.370 — stricter gate misses weak transitions,
especially at p=6–7 where k_ch=2 is harsh on 6+ channels).

### Interpretation
Stricter channel gate (k_ch=2) rejects single-channel artefact detections → fewer but
cleaner windows → better per-window LiNGAM fit via cleaner cascade chain. The improvement
is largest at p=4–5 (the fNIRS-relevant range). At p=6–7 the gate is too restrictive and
recall collapses enough to hurt struct_F1.

### Current defaults in code
`k_channels_min=2`, hybrid onset policy active. Best suited for p=4–5 scenarios.

---

## What's left to try

1. **p-adaptive k_channels_min**: use `k_ch = min(2, p-1)` so small graphs (p=3) require
   1 channel and larger graphs (p≥4) require 2. Avoids the p=3 edge case where k_ch=2
   equals requiring all channels.

2. **Option (i) — trim transients from window edges.** Drop first/last N samples before
   LiNGAM to skip post-CP settling. Orthogonal to all changes above.

3. **Recency-weighted consensus.** Weight later regimes more heavily in Beta-Bernoulli
   (they are closer to G_target). Flagged in session-2026-04-09.
