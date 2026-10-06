# Full-grid results: review of the numbers, the code, and a proposal

Scope: `results/results_october_5/` (18,000 scenarios, baseline and winning arms),
`run_experiments.py`, `full_pipeline.py`. Every number below was recomputed from
the stored CSVs; every code claim has a file:line.

The one-paragraph version: the ablation result (bootstrap + overlap helps, on
every metric, everywhere) is real and survives scrutiny. The *comparison*
table does not, for three reasons that are independent of each other — the
consensus aggregator returns the complete graph for half the grid, the
pipeline is seeded with the ground-truth structure of regime 0, and none of the
four "methods" is an external competitor. Each is fixable, and the fixes are
cheap in code and expensive only in compute.

---

## 1. What the four columns in the table actually are

This is the part that reframes everything else, so it comes first.

All four "methods" run **inside the same loop, on the same detected windows**
(`run_experiments.py:363–372`, `full_pipeline.py:875–1013`). None of them sees
the series unsegmented. Concretely:

| column prefix | what runs per window | prior on window 0 | chain prior | bootstrap |
|---|---|---|---|---|
| *(pipeline)* | CausalMorph → DirectLiNGAM | **true regime-0 graph** | yes | yes (20 × 0.60) |
| `causalmorph_*` | CausalMorph → DirectLiNGAM | **true regime-0 graph** | yes | no |
| `directlingam_*` | DirectLiNGAM on raw window | none | none | no |
| `icalingam_*` | ICA-LiNGAM on raw window | none | none | no |

So:

- **"Pipeline vs CausalMorph" is the bootstrap ablation computed a second
  time.** `causalmorph_*` is the pipeline with `bootstrap_n=0` — same transform,
  same chain, same anchor. It is not CausalMorph as a competitor; it is the
  baseline arm, re-evaluated inside the winning arm's run. The +5.0% nSHD it
  shows is the same effect as the headline +5.8% windowed nSHD, measured with
  less care. It should not appear in a comparison table at all.
- **"Pipeline vs DirectLiNGAM / ICA-LiNGAM" is confounded four ways.** Those two
  columns lack the CausalMorph transform *and* the chain prior *and* the
  ground-truth anchor *and* bootstrap. The 99.4% / 99.99% win-rates bundle a
  legitimate method effect with a ground-truth leak (§2.2). Nothing in the
  current data separates them.
- **There is no "ignore non-stationarity" control.** The obvious baseline for a
  non-stationary method — fit one graph to the whole series — was never run.
  Nor was any method built for non-stationarity (CD-NOD). The comparison the
  paper needs is exactly the one the table omits.

---

## 2. The results, re-read

### 2.1 The consensus aggregator returns the complete graph for half the grid

`aggregate_structures_bayesian` (`full_pipeline.py:117–175`) uses a Beta(1,1)
prior, scales the votes by the number of windows `n`, and includes an edge when
the posterior mean exceeds **0.20**. For an edge that *no* window voted for,
the posterior mean is `1 / (n + 2)`:

| windows `n` | posterior mean of an absent edge | > 0.20? |
|---|---|---|
| 1 | 0.333 | **included** |
| 2 | 0.250 | **included** |
| 3 | 0.200 | excluded (boundary) |
| 4 | 0.167 | excluded |

With one or two windows, every off-diagonal edge enters the consensus — the
consensus graph is complete. The data confirm it exactly:

| `n_structures` | scenarios | share | consensus recall | consensus precision | consensus nSHD |
|---|---|---|---|---|---|
| 1 | 2,917 | 16.2% | **1.000** | 0.337 | 0.663 |
| 2 | 6,273 | 34.8% | **1.000** | 0.344 | 0.656 |
| 3 | 3,631 | 20.2% | 0.442 | 0.450 | 0.334 |
| ≥ 4 | 5,179 | 28.8% | 0.451 | 0.332 | 0.370 |

Recall of exactly 1.000 across 9,190 scenarios is not a result; it is the
signature of a complete graph. **51% of every consensus metric in the paper is
measuring that.** It also explains two things that looked like findings:

- the Pipeline-vs-CausalMorph "tie" on consensus F1 (p = 0.37): in half the
  grid both produce the identical complete graph, so they tie by construction;
- the 10,673 ties on consensus precision in the ablation.

The fix is one number. With Beta(1,1), `post_mean > 0.5` is equivalent to
*weighted vote share > 0.5* for every `n`, including `n = 1` (0.667 vs 0.333).
That turns the aggregator into a weighted majority vote, which is also much
easier to state in a paper than a Beta posterior with an unexplained 0.20.
Per-window adjacencies are not stored in the CSV, so this needs a re-run.

### 2.2 The pipeline is seeded with the ground-truth structure of regime 0

`run_experiments.py:320–334` builds `initial_adj` from `true_adjs[0]` and
`initial_order` from a topological sort of it, and passes both to
`extract_causal_structures` and to the ablation. In `chain` mode window 0 uses
that prior directly (`full_pipeline.py:713–719`), and every later window's
prior is the previous window's estimate — so the leak propagates down the chain.

How much it is worth, from the per-window nSHD column:

| method | window 0 | windows ≥ 1 | ratio |
|---|---|---|---|
| Pipeline (anchor = truth) | 0.136 | 0.258 | **0.53** |
| `causalmorph_*` (anchor = truth) | 0.147 | 0.270 | **0.54** |
| DirectLiNGAM (no anchor) | 0.338 | 0.425 | 0.79 |
| ICA-LiNGAM (no anchor) | 0.378 | 0.506 | 0.75 |

Window 0 is easier for everyone (ratio ≈ 0.77 with no anchor — longer, cleaner,
no mis-segmentation upstream). The anchored methods get an extra ~1.5× on top of
that. That extra is the leak, and it is baked into every structure metric the
pipeline reports, most heavily into regime 0 and into the chain that follows it.

A reviewer who reads `run_experiments.py` will find this in under a minute.

The cold-start path already exists: when `initial_adj is None`, window 0 is
fit cold with DirectLiNGAM and that estimate becomes the anchor
(`full_pipeline.py:721–734`). The code change is passing `None`. The
experimental consequence is a full re-run.

### 2.3 Detection is the bottleneck, and it shapes every structure metric

| | value |
|---|---|
| detection precision | 0.849 |
| detection recall | **0.518** |
| true change points per scenario | 3.20 |
| detected | 2.01 |
| scenarios under-segmented (`n_structures < n_regimes`) | **59.3%** |
| over-segmented | 11.3% |
| exact | 29.4% |

In three scenarios out of five a detected window spans more than one true
regime. `assign_true_regime` (`full_pipeline.py:71–83`) then scores that window
against whichever regime overlaps it most, so the "per-regime" metrics are, for
most of the grid, "per-window-against-its-plurality-regime" metrics. The
structure numbers are as much a measurement of detection recall as of
extraction quality, and nothing in the current design separates the two.

The experiment that separates them is cheap to specify: **run extraction on the
true change points.** That gives the ceiling of the extraction layer and puts a
number on what the detector's 0.518 recall is costing downstream.

### 2.4 Everything degrades with `p`, including the pipeline

Mean nSHD 0.128 → 0.258 and F1 0.74 → 0.30 from `p = 3` to `p = 12`. The lead
over the LiNGAM baselines compresses (nSHD gap to DirectLiNGAM 0.370 → 0.179)
while the lead over the no-bootstrap variant grows (0.007 → 0.016). The paper
cannot claim scalability from this; it can claim that the relative advantage
of bootstrap grows as the problem hardens.

### 2.5 What survives

The extraction ablation — baseline vs bootstrap + overlap, paired, 18,000
scenarios — is sound as a *within-pipeline* result: all five metrics move the
right way, win-rates 0.67–0.72, p ≈ 0, direction consistent with the subset,
insensitive to which duplicate copy is kept. The leak and the consensus bug
affect both arms identically, so the *difference* is uncontaminated. The
absolute levels are not.

---

## 3. Code review

Ordered by consequence.

1. **`full_pipeline.py:117–175` — consensus threshold.** `edge_threshold=0.20`
   with Beta(1,1) admits every edge for `n ≤ 2` (§2.1). Change to 0.5 or
   threshold the weighted vote directly. The docstring describes the posterior
   correctly and never mentions what happens at small `n`.

2. **`run_experiments.py:320–334` — ground-truth anchor.** `initial_adj` and
   `initial_order` come from `true_adjs[0]`. Pass `None` for the honest
   condition; keep the oracle condition as a reported upper bound.

3. **`run_experiments.py:337–338` and `full_pipeline.py:403, 414–415, 471` —
   oracle regime length.** `min_regime_len = min(regime_sizes)` is taken from
   the ground truth and sets `baseline_end`, the refractory period,
   `step_win`, `ch_win` and `corr_window`. The detector is tuned per scenario
   with information it would not have on real data. Replace with a fixed value
   (500, the grid floor) or an estimate, and report the sensitivity.

4. **`run_experiments.py:345` — `evaluate_detection(..., tolerance=200)`.**
   200 samples is 40% of the shortest regime. Report precision/recall at 50,
   100 and 200.

5. **`full_pipeline.py:71–83` — `assign_true_regime`.** Plurality overlap is a
   reasonable choice, but with 59% under-segmentation it should be stated
   explicitly in the metric definitions, and the windowed/transition metrics
   from `metrics.py` should be the headline for structure-over-time rather than
   per-window SHD.

6. **`full_pipeline.py:85–115` — `compute_shd` denominator (D-01).**
   `shd / (d·(d−1))` normalises against every possible directed edge. For a
   density-0.20 graph most of that denominator is true negatives, which SHD
   does not count, so nSHD reads flatteringly small. Either report raw SHD and
   F1 as headline, or normalise by the number of true edges, and say which.

7. **Non-determinism, source unidentified.** Duplicate rows from the two
   concurrent runs differ in 195 of 200 sampled scenarios. The usual suspect —
   `nx.topological_sort` under hash randomisation — was tested and is *not* it
   (identical order under four `PYTHONHASHSEED` values). `_bootstrap_fit` seeds
   its RNG; DirectLiNGAM is deterministic. Remaining candidates: `ICALiNGAM`
   has no `random_state` (affects only `icalingam_*`), and BLAS reduction order
   under different thread counts (would affect everything, at near-ties in
   DirectLiNGAM's ordering). Needs a controlled test before the paper claims
   reproducibility: same scenario, two processes, pinned threads, diff.

8. **`full_pipeline.py:953` — ablation binarises the chain prior at a
   hard-coded 0.05** while the pipeline takes `chain_prior_edge_threshold` as a
   parameter (0.05 in production, so no numerical difference today, but the two
   paths will silently diverge if the parameter is ever changed).

9. **`run_experiments.py:363–372` — the ablation re-runs CausalMorph.** Since
   `causalmorph_*` equals the baseline arm, it is 18,000 extra CausalMorph +
   DirectLiNGAM fits per run that duplicate `v1h_full_baseline`. Drop it from
   `ABLATION_METHODS` once the ladder in §4 exists.

---

## 4. Proposal

The goal is a comparison table in which every row differs from its neighbour
by exactly one thing, no row sees the ground truth, and at least one row is a
method someone else built.

### 4.1 Two code changes before any re-run

- `aggregate_structures_bayesian`: `edge_threshold=0.5`.
- `run_one`: `initial_adj=None, initial_order=None` by default; add
  `--oracle_anchor` to restore the current behaviour for the upper-bound row.

Both are a handful of lines. Add `--oracle_segmentation` (use `true_cps` as
`detected_cps`) at the same time, and `--no_segmentation` (one window,
`detected_cps=[]`), so the whole design below runs from flags.

### 4.2 The ablation ladder (detected windows, cold start)

| row | transform | prior | bootstrap + overlap | what it isolates |
|---|---|---|---|---|
| L0 | — | — | — | DirectLiNGAM per window: the floor |
| L1 | CausalMorph | cold (window-0 DL fit) | — | the transform |
| L2 | CausalMorph | chain | — | the chain prior |
| L3 | CausalMorph | chain | yes | the full pipeline — this is the row the paper reports |

L0 exists today as `directlingam_*`. L2 is the current baseline arm minus the
leak. L3 is the winning arm minus the leak. Each step up the ladder is a paired
comparison on the same scenarios.

### 4.3 Three control rows

| row | what | why |
|---|---|---|
| C1 | L3 with **oracle anchor** | upper bound; quantifies the leak the current results carry |
| C2 | L3 on **true change points** | ceiling of the extraction layer; isolates detection cost |
| C3 | DirectLiNGAM on the **whole series** | the "ignore non-stationarity" baseline the paper currently lacks |

C3 is the single most persuasive row for a reader who doubts the premise, and
it is the cheapest run in this document.

### 4.4 One external method

CD-NOD is in `causal-learn` and is the method every reviewer will name. Run it
on the same 18,000 series (it needs no segmentation) and score its single
output against the per-regime truth the same way as C3. If it is slow at
`p = 12`, run it on the 180-scenario stratified subset first and promote it if
the direction is clear. The current Q&A document already promises this
comparison as open item O6.

### 4.5 Reporting

- Headline structure-over-time metrics: windowed nSHD, transition SHD, dynamic
  score. Per-window SHD moves to the appendix with the plurality-regime caveat.
- Detection reported at three tolerances.
- Consensus metrics reported only after 4.1, and with `n_structures`
  distribution alongside, so a reader can see how many scenarios had one window.
- Drop `causalmorph_*` from every table. The bootstrap ablation is already the
  baseline-vs-winning comparison; a third copy adds nothing and invites the
  "vs CausalMorph" misreading.
- Determinism: a short section stating what is seeded and a reproducibility
  check (same scenario, two processes, identical rows), once item 3.7 is closed.

### 4.6 Cost and order

Per run at 12 workers on the M4 Max, with the bootstrap rows ~9 h and the
no-bootstrap rows ~2 h:

| phase | runs | ≈ hours | unlocks |
|---|---|---|---|
| 1 | L3 cold, C1 oracle-anchor | 18 | the honest headline + the size of the leak |
| 2 | C3 whole-series, L1, L2 | 6 | the ladder and the premise baseline |
| 3 | C2 oracle-segmentation | 9 | detection-vs-extraction attribution |
| 4 | CD-NOD | unknown | the external comparison |

Phase 1 alone replaces every number in the current comparison table with one
that survives a code read. Phases 2–3 turn the table into an argument. Phase 4
turns it into a paper.

The existing `v1h_full_baseline` and `v1h_full_winning` remain valid as the
*oracle-anchor* versions of L2 and L3; nothing is wasted, it is relabelled.
