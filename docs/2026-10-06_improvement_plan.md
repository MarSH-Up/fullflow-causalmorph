# Plan: hardening the nsDCBN pipeline

Companion to `2026-10-06_full_grid_review_and_proposal.md`. That document says
what is wrong; this one says what to change, states each change as a
modification of a specific estimator, derives what it should do, and fixes the
protocol that decides whether it did.

Every quantity quoted as "measured" comes from `results/results_october_5/` or
from the two small re-runs of 2026-10-06 (96 and 24 scenarios). Every quantity
quoted as "predicted" is derived here and is falsifiable by the protocol in §3.

---

## 0. Notation

| symbol | meaning |
|---|---|
| $X \in \mathbb{R}^{T \times p}$ | one scenario; $T$ samples, $p$ variables |
| $\mathcal{C}^\* = \{c^\*_1 < \dots < c^\*_K\}$ | true change points; $R = K+1$ regimes |
| $G^\*_k$, $A^\*_k \in \{0,1\}^{p\times p}$ | true DAG and adjacency of regime $k$ |
| $\hat{\mathcal{C}} = \{\hat c_1 < \dots < \hat c_{\hat K}\}$ | detected change points |
| $w_j = [s_j, e_j)$ | window $j$ induced by $\hat{\mathcal{C}}$ (after overlap); $n_j = e_j - s_j$ |
| $\hat A_j$ | estimated adjacency on window $j$ |
| $m = p(p-1)/2$ | number of unordered variable pairs |
| $W$ | half-width of the detector's sliding window |
| $\rho_{ij}$, $r_{ij}$ | population and sample correlation of pair $(i,j)$ |
| $\sigma(\cdot)$ | the plurality map $w_j \mapsto k$ of `assign_true_regime` |

Edge-set metrics on $(\hat A, A^\*)$ with diagonals removed:
$\mathrm{TP}, \mathrm{FP}, \mathrm{FN}$, $\mathrm{SHD} = \mathrm{FP}+\mathrm{FN}$,
$\mathrm{nSHD} = \mathrm{SHD}/(p(p-1))$, precision, recall, $F_1$.

---

## 1. The pipeline as implemented

Four stages. The defects live at precise points in three of them.

### P1 — Detection statistic (`detectors_correlation.py:55–65`)

$$
S(t) \;=\; \bigl\| \operatorname{vec}_u\!\bigl(\hat C_{[t,\,t+W)}\bigr) - \operatorname{vec}_u\!\bigl(\hat C_{[t-W,\,t)}\bigr) \bigr\|_2,
\qquad t \in [W,\, T-W),
$$

where $\hat C_I$ is the sample correlation matrix on index set $I$ and
$\operatorname{vec}_u$ stacks the $m$ upper-triangular entries. (This settles
D-07: upper triangle, not full matrix, so no double counting.)

### P1 — Threshold (`detectors_correlation.py:68–80`)

With $B = [0, b)$, $b = \max(150, \lfloor \min_k |w^\*_k| / 4 \rfloor)$ the first
samples of the series, and $S_B = \{S(t) : t \in B,\ S(t) > 0\}$,

$$
\theta \;=\; \max\Bigl\{\operatorname{med}(S_B) + k \cdot 1.4826 \cdot \operatorname{MAD}(S_B),\ \theta_{\min}\Bigr\},
\qquad k = 6,\ \theta_{\min} = 0.5 .
$$

$\theta$ is computed **once**, from regime 0, and never updated. Peaks of $S$
above $\theta$ separated by at least the refractory distance are the raw
detections $\hat{\mathcal C}_{\text{raw}}$.

**Defect D1.** The null scale of $S$ depends on $m$ and $W$ (§2.3), so a fixed
absolute floor $\theta_{\min}$ is not scale-aware; and a threshold calibrated on
regime 0 is applied to transitions whose signal is, by construction of the
generator, smaller (§2.6). Measured: on $R=7$, $n=2500$ scenarios where nothing
is pruned, detection of the $k$-th true change point equals
$\mathbb{1}[\max_{|t-c^\*_k|<W} S(t) \ge \theta]$ in every cell of the table, and
that indicator falls from 1.00 at $k=0$ to 0.50–0.58 at $k=5$.

### P2 — Post-processing (`run_experiments.py:245–270`)

$\hat{\mathcal C}_{\text{raw}} \to \hat{\mathcal C}$ by (i) merging detections
closer than 160 samples, then (ii) repeatedly deleting whichever detection
bounds the shortest window until every window satisfies

$$
n_j \;\ge\; n_{\min}(p) \;=\; \max\bigl(250,\ 10\,p^2\bigr).
$$

**Defect D2.** $n_{\min}(p)$ is an *extraction* constraint (DirectLiNGAM needs
samples) applied inside the *detection* output, and $10p^2$ has no derivation
anywhere in the code. For $p \ge 8$ it exceeds the grid's 500-sample regimes, so
every true change point adjacent to such a regime is deleted regardless of
whether P1 found it. Measured on 96 scenarios: recall 0.232 → 0.646 at $n=500$,
0.573 → 0.774 at $n=1500$ when step (ii) is removed; and across the full grid,
recall restricted to non-prunable change points is flat in $p$ (0.50–0.67).
The reported $p$-collapse of the detector is this rule.

### P3 — Extraction (`full_pipeline.py:629–872`)

Windows are $w_j$ with overlap $\lfloor \omega\, n_{j-1} \rfloor$, $\omega = 0.15$.
On window $j$ with prior $\Pi_j = (\pi_j, A^{\Pi}_j)$ (a causal order and a
binary adjacency), the estimate is a bootstrap aggregate:

$$
\hat A_j \;=\; \mathbb{1}\Bigl[\tfrac{1}{B}\textstyle\sum_{b=1}^{B} \mathbb{1}\bigl[\,|\hat A^{(b)}_j| > 0.05\,\bigr] \;\ge\; \tau_{\text{boot}}\Bigr],
\qquad B = 20,\ \tau_{\text{boot}} = 0.60,
$$

where $\hat A^{(b)}_j = \mathrm{DirectLiNGAM}\bigl(\mathrm{CausalMorph}(X^{(b)}_{w_j};\,\Pi_j)\bigr)$
on a with-replacement resample $X^{(b)}$. The chain prior is
$\Pi_{j} = (\hat\pi_{j-1}, \hat A_{j-1})$ for $j \ge 1$.

**Defect D3.** $\Pi_0 = \bigl(\operatorname{topo}(G^\*_0),\ A^\*_0\bigr)$
(`run_experiments.py:320–334`): window 0 is seeded with the ground truth, and
the chain carries it forward. Measured: window-0 nSHD / later-window nSHD is
0.53 for the anchored methods and 0.77–0.79 for the un-anchored ones.

### P4 — Consensus (`full_pipeline.py:117–175`)

With weights $\lambda_j = n_j / \sum_l n_l$ and vote share
$\bar v_{ik} = \sum_j \lambda_j\, \hat A_{j,ik}$, under a $\mathrm{Beta}(\alpha,\beta)$
prior scaled by $\hat K + 1 =: n$ windows,

$$
\mathbb{E}[\,q_{ik} \mid \text{votes}\,] \;=\; \frac{\alpha + n\,\bar v_{ik}}{\alpha + \beta + n},
\qquad \text{include } (i,k) \iff \mathbb{E}[q_{ik}] > \tau_c ,
$$

with $\alpha = \beta = 1$, $\tau_c = 0.20$.

**Defect D4.** For an edge with $\bar v_{ik} = 0$ the posterior mean is
$1/(n+2)$, which exceeds $0.20$ for $n \in \{1, 2\}$. The consensus graph is
complete whenever at most two windows exist. Measured: 51.0% of scenarios;
consensus recall exactly 1.000 on all 9,190 of them.

---

## 2. Proposals

Each: current form → proposed form → what it should do → how we know.

### 2.1 Consensus: a decision rule with a loss, and a prior with the right mass

**Minimal fix.** Under 0–1 loss the Bayes decision is to include an edge iff its
posterior probability exceeds $\tfrac12$. With $\alpha=\beta=1$,

$$
\frac{1 + n\bar v}{2 + n} > \frac12 \iff \bar v > \frac12 \quad\text{for every } n \ge 1,
$$

so $\tau_c = 0.5$ turns P4 into a length-weighted majority vote, exactly, with
no small-$n$ pathology. One constant.

**Principled fix.** The uniform prior puts half its mass on "edge present",
which for a graph of expected density $\rho_0 \approx 0.2$ is wrong by a factor
of 2.5. Use a sparsity-encoding prior with pseudo-count $\kappa$:

$$
\alpha = \kappa\,\rho_0,\qquad \beta = \kappa\,(1-\rho_0),\qquad
\mathbb{E}[q_{ik}] = \frac{\kappa\rho_0 + n\bar v_{ik}}{\kappa + n},
$$

decision $\mathbb{E}[q_{ik}] > \tfrac12$. For $\kappa=1$, $\rho_0 = 0.2$ an absent
edge has posterior $0.2/(1+n) \le 0.1$ for all $n$, and inclusion requires
$\bar v > \tfrac12 + \tfrac{0.3}{n}$ — slightly conservative for one window,
majority in the limit. $\rho_0$ can be set from the mean density of $\{\hat A_j\}$
(empirical Bayes) so nothing is tuned to the grid.

**Prediction.** Consensus recall on $n \le 2$ scenarios drops from 1.000 to the
level of the single-window estimate (≈ 0.41, the per-regime recall); consensus
precision on those scenarios rises from 0.34 to ≈ 0.48; the Pipeline-vs-
no-bootstrap "tie" on consensus $F_1$ ($p = 0.37$) disappears because the two
arms no longer share a degenerate output.

**Verification.** Recompute P4 only. Per-window adjacencies are not stored, so
this needs a re-run; it rides along with 2.4 at no extra cost.

### 2.2 Decouple detection from extraction

Replace the deletion step (ii) of P2 by two separate objects:

1. $\hat{\mathcal C}$: the detector's output after step (i) only. **This is what
   the detection metrics score.**
2. $\mathcal W(\hat{\mathcal C}, n_{\min})$: the extraction windows, obtained by
   merging each segment shorter than $n_{\min}$ into its *shorter* neighbour
   (so the merged window's plurality regime is least disturbed), and recording
   for every window the set of detected change points it absorbed.

The structure metrics score $\mathcal W$; the detection metrics score
$\hat{\mathcal C}$. A merged change point counts as *detected* and as
*not extracted*, and the gap between the two is reported as a number —
the extraction layer's minimum-sample cost.

**$n_{\min}(p)$.** Replace $10p^2$ by a calibrated linear rule. DirectLiNGAM's
ordering is stable once $n \gtrsim c\,p$; estimate $c$ once by fitting
DirectLiNGAM on stationary windows of length $n \in \{10p, 20p, 40p, 80p\}$
for $p \in \{4, 8, 12\}$ and reading off the smallest $n/p$ at which nSHD is
within 10% of its $n = 80p$ value. Then

$$
n_{\min}(p) = \max\bigl(250,\ \lceil c\,p \rceil\bigr).
$$

If $c \approx 40$, $n_{\min}(12) = 480 < 500$ and the grid floor is reachable
at every $p$. If the calibration says $c > 42$, the grid floor itself is too
low for $p=12$ and that becomes a stated limitation rather than a silent one.

**Prediction.** Full-grid detection recall from 0.465 (micro) to 0.65–0.75
(the 96-scenario re-run gave 0.646 / 0.774 / 0.776 by regime length).
Under-segmentation from 59% to roughly 35%. Precision falls from 0.74 to
≈ 0.65–0.70 — step (ii) was also removing false positives, and that loss is
real and should be reported, not hidden.

### 2.3 The detection statistic: standardise, then aggregate for sparse change

**Why the current $S$ is scale-blind.** For a pair with population correlation
$\rho_{ij}$ and a window of $W$ samples, the sample correlation satisfies
$\operatorname{Var}(r_{ij}) \approx (1-\rho_{ij}^2)^2 / W$. Two disjoint windows
give independent $r$'s, so under $H_0$

$$
\mathbb{E}\bigl[S(t)^2\bigr] \;\approx\; \frac{2}{W}\sum_{i<j}\bigl(1-\rho_{ij}^2\bigr)^2 \;\le\; \frac{2m}{W}.
$$

The null scale of $S$ grows like $p/\sqrt W$. The MAD calibration tracks this
for the *median and spread*, so false positives are controlled; but the floor
$\theta_{\min} = 0.5$ is absolute, and the whole construction treats a change in
3 of 66 pairs the same as a change in 60 of 66. Under $H_1$ with a changed
pair-set $D$ and shifts $\delta_{ij}$,

$$
\mathbb{E}\bigl[S^2\bigr] \approx \frac{2m}{W} + \sum_{(i,j)\in D}\delta_{ij}^2
\quad\Longrightarrow\quad
\text{SNR} \;\propto\; \frac{W \,\|\delta_D\|_2^2}{m},
$$

which is why $W = 100$ (regimes of 500) is 2.2× worse than $W = 500$, and why
a sparse change on a large $p$ is the hard case.

**Proposed statistic.** Standardise each pair on the Fisher scale and aggregate
in a way that does not dilute a sparse change:

$$
z_{ij}(I) = \operatorname{artanh}\bigl(r_{ij}(I)\bigr),\qquad
\zeta_{ij}(t) = \sqrt{\tfrac{W-3}{2}}\;\bigl(z_{ij}([t,t+W)) - z_{ij}([t-W,t))\bigr) \;\overset{H_0}{\approx}\; \mathcal N(0,1),
$$

$$
S_{\max}(t) = \max_{i<j} |\zeta_{ij}(t)|,
\qquad\text{or}\qquad
S_{(k)}(t) = \sum_{l=1}^{k} \zeta^2_{(l)}(t)\ \ (\text{top-}k \text{ squared}).
$$

Under $H_0$ and approximate independence across pairs,
$\mathbb{E}[S_{\max}] \approx \sqrt{2\ln m}$ with $O(1/\sqrt{\ln m})$
concentration, so a threshold

$$
\theta_{\max} = \sqrt{2\ln m} + \gamma,\qquad \gamma \in [1.5, 2.5],
$$

is **calibration-free in both $p$ and $W$** and needs no baseline segment at
all. The top-$k$ form with $k \approx$ the expected number of rewired pairs
(≈ $0.4\,p$ in this generator) is the compromise when changes touch a handful
of pairs rather than one.

Two caveats to carry into the paper. Pairs sharing a variable are not
independent, so $\sqrt{2\ln m}$ is a slight overestimate of the null mean (safe
direction). The noise is Laplace, not Gaussian, so $\operatorname{Var}(z)$ carries
a kurtosis factor; $\gamma$ absorbs it and is chosen on a *held-out* $p=5$ slice
exactly as $k=6$ was.

**Adaptive null, regardless of statistic.** Whatever $S$ is used, re-estimate
its null location/scale from the most recent accepted segment with a guard band
of $W$ on each side, instead of from regime 0 only:

$$
\theta_t = \operatorname{med}\bigl(S\,|\,[\hat c_{\text{last}} + W,\ t - W)\bigr) + k\cdot 1.4826\cdot\operatorname{MAD}(\cdot).
$$

This removes the dependence on regime 0 being representative.

**Prediction.** The positional decay on long regimes (0.77 → 0.35) is almost
entirely "signal below a regime-0 threshold"; an adaptive null should flatten
it to within the generator's own asymmetry (§2.6), i.e. to ≈ 0.6–0.7 at every
position. The $S_{\max}$ statistic should move the $n=500$ column most.

**Verification.** Detection-only runs are cheap (no extraction): the full grid
in well under an hour at 12 workers. Sweep $\gamma$ on a $p = 5$ slice, lock it,
then run the grid once per statistic. Report $(\text{P}, \text{R}, F_1)$ micro,
and $R(\tau)$ for $\tau \in \{25, 50, 100, 200\}$.

### 2.4 Window-0 prior: cold start by default, oracle as a labelled upper bound

$$
\Pi_0^{\text{cold}} = \bigl(\hat\pi^{\,\mathrm{DL}}_0,\ \mathbb{1}[\,|\hat A^{\,\mathrm{DL}}_0| > 0.05\,]\bigr),
\qquad
\Pi_0^{\text{oracle}} = \bigl(\operatorname{topo}(G^\*_0),\ A^\*_0\bigr).
$$

The cold path exists (`full_pipeline.py:721–734`); the change is
`initial_adj=None, initial_order=None` as the default in `run_one`, with
`--oracle_anchor` restoring today's behaviour for the C1 row of §3.3.

**Prediction.** From the measured ratios, window-0 nSHD rises from 0.136 to
$\approx 0.258 \times 0.78 \approx 0.20$. The propagated effect on windows
$j \ge 1$ is the quantity we do not know and C1 exists to measure. The
*ablation* difference (bootstrap vs none) is predicted unchanged to within
noise, because both arms lose the same anchor.

### 2.5 Chain prior as a soft Markov prior (phase 2, optional)

The chain passes a *binarised* $\hat A_{j-1}$ forward and discards the bootstrap
frequencies $\hat q_{j-1,ik} \in [0,1]$ it already computed. A first-order
Markov prior on the edge marginals keeps them:

$$
q^{\Pi}_{j,ik} \;=\; \lambda\,\hat q_{j-1,ik} + (1-\lambda)\,\rho_0,\qquad \lambda \in [0.5, 0.9],
$$

and CausalMorph is given $A^{\Pi}_j = \mathbb{1}[q^{\Pi}_j > \tfrac12]$ while the
bootstrap threshold for window $j$ is applied to the *posterior*
$\tfrac12\bigl(q^{\Pi}_j + \hat q_j\bigr)$ rather than to $\hat q_j$ alone. This
is the principled version of what the disabled "hybrid" selector tried to do
by score comparison: it lets a weak prior be overridden by data and a strong
one persist, with $\lambda$ as the single forgetting parameter. It also makes
`chain_prior_edge_threshold` unnecessary. Untested; belongs after §2.1–2.4
are in.

### 2.6 Generator: a density-neutral, uniformly loaded schedule

`_schedule` (`full_pipeline.py:288`) assigns change $k$ of *each* list to
regime $1 + \lfloor k\,(R-1)/n \rfloor$. Both lists fire their first element at
regime 1, so transition 0 carries $\approx 2\times$ the edge changes of later
ones (measured: 4.7 vs 3.3–4.1 at $p=10$), and because losses front-load and
gains accumulate, density rises monotonically along the series (measured
0.136 → 0.268). Both inflate detectability of the first change point relative
to the rest.

Interleave the two lists on a single schedule of length $n_L + n_G$:

$$
r_L(k) = 1 + \Bigl\lfloor \frac{2k\,(R-1)}{n_L+n_G} \Bigr\rfloor,\qquad
r_G(k) = 1 + \Bigl\lfloor \frac{(2k+1)(R-1)}{n_L+n_G} \Bigr\rfloor,
$$

and pair every gain with a loss at the same transition (density-neutral) by
taking $n_L = n_G$. Then each transition changes $\approx (n_L+n_G)/(R-1)$
edges and $|E(G^\*_k)|$ is constant in $k$.

This changes the benchmark, so it is a *new* dataset version (v2), run
alongside v1 for one phase so the paper can say what the asymmetry was worth.

---

## 3. Evaluation protocol

### 3.1 Metrics, stated once

Detection, over the whole grid, **micro-averaged**:

$$
\mathrm{P} = \frac{\sum_s \mathrm{TP}_s}{\sum_s (\mathrm{TP}_s + \mathrm{FP}_s)},\qquad
\mathrm{R} = \frac{\sum_s \mathrm{TP}_s}{\sum_s (\mathrm{TP}_s + \mathrm{FN}_s)},
$$

with matching at tolerance $\tau$ (one-to-one, nearest first) and the curve
$\mathrm R(\tau)$ for $\tau \in \{25,50,100,200\}$. Macro averages, if shown,
exclude scenarios with $\hat K = 0$ from the precision mean and say so. (Today:
macro 0.849/0.518 vs micro 0.739/0.465; the difference is the
$\text{precision}=1$ convention on 16.2% of scenarios.)

Structure, per window, scored against $A^\*_{\sigma(w_j)}$; headline
structure-over-time metrics are `windowed_nshd_weighted`, `transition_shd`,
`dynamic_score` from `metrics.py`, which already weight by window length and
penalise mis-segmentation. Per-window nSHD moves to the appendix with the
plurality caveat. Report $\mathrm{SHD}$ and $F_1$ alongside nSHD; if nSHD is
kept as headline, normalise by $|E(A^\*)|$ as well as by $p(p-1)$ and show both
once (D-01).

### 3.2 Error decomposition

With $\mathcal E(\cdot)$ any structure metric averaged over the grid, the three
oracle rows of §3.3 give a telescoping attribution:

$$
\underbrace{\mathcal E(\hat{\mathcal C},\, \Pi^{\text{cold}})}_{\text{what we ship}}
\;=\;
\underbrace{\mathcal E(\mathcal C^\*,\, \Pi^{\text{cold}})}_{\text{extraction ceiling}}
\;+\;
\underbrace{\bigl[\mathcal E(\hat{\mathcal C},\Pi^{\text{cold}}) - \mathcal E(\mathcal C^\*,\Pi^{\text{cold}})\bigr]}_{\Delta_{\text{det}}:\ \text{cost of detection}}
$$

and separately
$\Delta_{\text{anchor}} = \mathcal E(\hat{\mathcal C},\Pi^{\text{cold}}) - \mathcal E(\hat{\mathcal C},\Pi^{\text{oracle}})$,
the size of the leak the October results carry. Each term is a paired
difference over the same 18,000 scenarios.

### 3.3 Rows

All rows: v1H detection, $\omega = 0.15$, chain mode, scored identically.

| row | windows | $\Pi_0$ | transform | chain | bootstrap | isolates |
|---|---|---|---|---|---|---|
| L0 | $\hat{\mathcal C}$ | — | — | — | — | DirectLiNGAM per window (floor) |
| L1 | $\hat{\mathcal C}$ | cold | CM | — | — | the transform |
| L2 | $\hat{\mathcal C}$ | cold | CM | yes | — | the chain prior |
| **L3** | $\hat{\mathcal C}$ | cold | CM | yes | yes | **the pipeline** |
| C1 | $\hat{\mathcal C}$ | oracle | CM | yes | yes | $\Delta_{\text{anchor}}$ |
| C2 | $\mathcal C^\*$ | cold | CM | yes | yes | $\Delta_{\text{det}}$ |
| C3 | $[0,T)$ | — | — | — | — | ignore non-stationarity |
| X1 | — | — | — | — | — | CD-NOD, single output scored as C3 |

`v1h_full_baseline` and `v1h_full_winning` are the oracle-anchor versions of
L2 and L3 and are kept, relabelled.

### 3.4 Statistics

Every comparison is paired by `scenario_id`. Report, per metric:
mean ± SD per arm; paired mean difference with a 95% bootstrap CI
(10,000 resamples of scenarios); Wilcoxon signed-rank $p$; the matched-pairs
rank-biserial effect size
$r_{rb} = (\#\text{wins} - \#\text{losses}) / (\#\text{wins} + \#\text{losses})$;
win/loss/tie counts. Holm correction across the 5 headline metrics within each
row comparison. With $N = 18{,}000$, $p$-values underflow; the CI and $r_{rb}$
carry the information.

### 3.5 Determinism

Before any row is final: the same scenario, two separate processes,
`OMP_NUM_THREADS=1`, `PYTHONHASHSEED=0`, `ICALiNGAM(random_state=seed)`; assert
byte-equal rows. If it fails, bisect by stage (detection only, extraction only)
to find the first stage whose output differs. The October duplicates differ in
195/200 scenarios and the cause is not yet located; `nx.topological_sort` was
tested and excluded.

---

## 4. Predicted outcomes, so the plan can be wrong

| quantity | today | predicted after §2.1–2.4 | basis |
|---|---|---|---|
| detection recall (micro) | 0.465 | 0.65–0.75 | 96-scenario re-run without step (ii) |
| detection precision (micro) | 0.739 | 0.65–0.70 | same re-run |
| under-segmented scenarios | 59% | ≈ 35% | recall gain at $n \le 1500$ |
| positional recall, pos 5, $R=7$, $n=2500$ | 0.35 | 0.6–0.7 | adaptive null (§2.3) |
| consensus recall, $n \le 2$ windows | 1.000 | ≈ 0.41 | §2.1 |
| consensus precision, $n \le 2$ | 0.34 | ≈ 0.48 | §2.1 |
| window-0 nSHD | 0.136 | ≈ 0.20 | §2.4 ratio argument |
| bootstrap ablation, per-regime precision | +9.4% | +8 to +11% | leak is common to both arms |
| L3 vs DirectLiNGAM, nSHD win-rate | 99.4% | lower; 85–95% | §1 D3 removed from one side only |

The last row is the one that matters for the paper: a drop *there* is the
honest number replacing a leaked one, not a regression.

---

## 5. Order and cost

At 12 workers on the M4 Max; bootstrap rows ≈ 9 h, non-bootstrap ≈ 2 h,
detection-only ≈ 0.5 h.

| phase | work | hours | unlocks |
|---|---|---|---|
| 0 | §2.1, §2.2 (code), §2.4 (flag), §3.5 test | — | everything below |
| 1 | detection-only grid: current $S$ with and without step (ii); $S_{\max}$; adaptive null | ≈ 2 | the detection section, final |
| 2 | L3, C1 | 18 | honest headline; $\Delta_{\text{anchor}}$ |
| 3 | C3, L1, L2 | 6 | ladder; the premise baseline |
| 4 | C2 | 9 | $\Delta_{\text{det}}$ |
| 5 | X1 (CD-NOD) on the 180-subset, promote if clear | ? | external comparison |
| 6 | generator v2 (§2.6), rerun phase 1 only | 2 | asymmetry quantified |
| 7 | §2.5 soft chain prior, as a new L4 row | 9 | optional |

Phase 1 is independent of extraction and can run tonight. Phase 2 replaces
every structure number currently in the report. Phases 3–4 turn the table into
an argument; 5 makes it a paper.

---

## 6. Out of scope here

CD-NOD runtime at $p=12$ (unknown until tried); the real-data track (SO6); the
nSHD-vs-$|E|$ normalisation decision, which is editorial once both are shown.

---

## 7. Validation log — detection track (2026-10-06, 600-scenario stratified sample)

Sample: 60 scenarios per $p$, 20 per samples-per-regime within each $p$, seed
20261006 (`results/hardening_validation/sample_ids.txt`). "Before" is the
October-5 full-grid run restricted to the same scenarios. Micro-averaged,
tolerance 200 unless stated. Branch `feature/pipeline-hardening`.

### 7.1 What §2.2 predicted, and what happened

| row | P | R | $F_1$ | R@25 | under-seg | over-seg |
|---|---|---|---|---|---|---|
| before (v1H, old post-processing) | 0.735 | 0.481 | 0.581 | 0.395 | 53.5% | 11.7% |
| v1H, merge instead of delete (§2.2) | 0.588 | 0.685 | 0.633 | 0.569 | 38.3% | 23.0% |

Predicted R 0.65–0.75, P 0.65–0.70, under-seg ≈ 35%: recall and
under-segmentation as predicted, precision 6 pp below the band — the
deleted rule was also removing false positives, and the loss concentrates at
$n = 1500$ (P 0.525). Zero-detection scenarios 12.2% → 0.0%: that figure was
the pruning, not the detector.

### 7.2 Two predictions of §2.3 that were wrong

**Adaptive null does not flatten the positional decay.** v1H + adaptive null:
P 0.797, R 0.565, position profile 0.86 → 0.18 (unchanged in shape). What
decays at later transitions is the *signal* (§2.6), and re-estimating the null
cannot recover signal. It is a precision tool (over-seg 23% → 7.8%), kept as an
option, not the fix.

**A fixed detector window is not a free replacement for the oracle regime
length.** v1H with $W$ from a fixed 500: P 0.354. The oracle $\min_k |w^*_k|$
was doing real work. v1L tolerates it far better (§7.4) but still pays ≈ 0.05
$F_1$; a data-driven $W$ remains open.

### 7.3 The statistic was the problem — and the Fisher scale was the next one

| detector | statistic | P | R | $F_1$ | under | over | recall by position (0…5) |
|---|---|---|---|---|---|---|---|
| v1J, $\gamma$ 2.0 | Fisher-standardised max | 0.615 | **0.957** | 0.749 | **5.5%** | 47.3% | 0.98 0.93 0.96 0.94 0.95 0.97 |
| v1J, $\gamma$ 2.5 | | 0.741 | 0.943 | 0.830 | 9.0% | 29.2% | 0.98 0.91 0.94 0.93 0.93 0.95 |
| v1K, $\gamma$ 2.5 | + per-pair robust scale | **0.944** | 0.826 | 0.881 | 31.2% | **4.0%** | 0.97 0.78 0.77 0.70 0.74 0.77 |
| v1L, $\gamma$ 2.0 | + scale re-estimated after pass 1 | 0.845 | 0.901 | 0.872 | 15.7% | 20.8% | 0.97 0.86 0.89 0.82 0.89 0.90 |
| v1L, $\gamma$ 2.5 | | 0.912 | 0.867 | **0.889** | 23.0% | 9.0% | 0.97 0.83 0.84 0.78 0.80 0.81 |
| v1L, $\gamma$ 3.0 | | 0.931 | 0.826 | 0.876 | 30.3% | 5.5% | — |

v1J confirmed the mechanism of §2.3 outright: the max-pair statistic with the
analytic threshold removes the positional decay entirely and lifts recall to
0.96. Its false alarms were not a $\gamma$ problem. Measured on real grid
regimes with no change point, the pooled null sd of $\zeta$ is
**1.22 / 1.20 / 1.30 / 1.66 at $p$ = 3 / 6 / 9 / 12** — the Gaussian
i.i.d. variance $1/(W-3)$ is too small and the error grows with $p$ (Laplace
innovations, strong $\rho$, pink noise). Per-step false-alarm rate at $p=12$:
0.32 at $\gamma=2$ against 6.5e-5 predicted; 0.035 even at $\gamma=5$. No
global $\gamma$ serves every $p$.

v1K studentises each pair by $1.4826 \cdot \operatorname{MAD}_t \zeta_{ij}$
over the series: null sd 0.87 / 0.87 / 0.83 / 0.91, false-alarm rate 0.000
at every $p$ for $\gamma = 3$. Precision 0.94, flat in $p$. But it lost
recall at every position after the first: the change-point bumps (width $2W$
each, up to a third of a short-regime series) inflate the MAD.

v1L re-estimates the scale after a first pass, masking $\pm W$ around its
change points. It recovered 4.1 pp of recall for 3.2 pp of precision and has
the best $F_1$. The step after position 0 narrowed but did not close: one
pass only masks the change points it found. The predicted "P ≈ 0.93 with
R ≈ 0.90+" was optimistic; R 0.867 at that precision, 0.901 at P 0.845.

### 7.4 Robustness to a non-oracle window ($W$ from a fixed 500)

| | oracle $W$ | fixed $W$ | $\Delta F_1$ |
|---|---|---|---|
| v1H | 0.633 | 0.454 | −0.18 |
| v1K $\gamma$ 3.0 | 0.861 | 0.789 | −0.07 |
| v1L $\gamma$ 2.5 | 0.889 | 0.840 | −0.05 |

### 7.5 Where this leaves the choice

v1L and v1J lie on one precision–recall frontier; v1L extends it on the
precision side, v1J holds the recall end. Detection $F_1$ cannot choose
between a missed change point (a window spanning two regimes, scored against
one) and a spurious one (a regime split into two windows of the same graph);
the structure metrics can. L3 is being run with v1J $\gamma$ 2.5, v1L
$\gamma$ 2.0 and v1L $\gamma$ 2.5 on the same 600 scenarios
(`run_l3_head_to_head.sh`); the winner takes the full ladder.

---

## 8. The adjacency convention (found 2026-10-06, after §7)

This supersedes every structure number in §1, §2.4, §4 and in the October-5
results. It was found because the cold-start pipeline (§2.4) collapsed to
per-regime recall 0.05 on the validation sample, which no legitimate removal
of a window-0 prior could explain.

### 8.1 Three conventions, no conversion

| object | layout | where |
|---|---|---|
| ground truth $A^\*$, every metric | $A[\text{from}, \text{to}]$ | generator, `compute_shd`, `metrics.py` |
| `lingam` output | $B[\text{to}, \text{from}]$ | `DirectLiNGAM.adjacency_matrix_`, `ICALiNGAM` |
| CausalMorph prior | parents of $i$ read from **row** $i$, i.e. $[\text{to}, \text{from}]$ | `causalmorph_algorithm.py:202` |

Measured on a 2-variable SEM $x_1 \to x_2$: `lingam` returns $B_{10} = 0.84$.
Measured on grid scenarios, recall of window 0 scored as-is / scored on the
transpose:

| path | p=6 | p=9 |
|---|---|---|
| oracle-seeded (the October configuration) | 1.00 / 0.00 | 0.75 / 0.00 |
| cold start | 0.00 / 1.00 | 0.00 / 0.75 |
| cold start with its prior transposed | 1.00 / 0.00 | 0.75 / 0.00 |

### 8.2 What happened in the October pipeline

1. `compute_shd(B_{\text{lingam}}, A^\*)` compared $[\text{to},\text{from}]$
   with $[\text{from},\text{to}]$ elementwise: **a reversed edge scored as a
   true positive, a correct edge as a false positive plus a false negative.**
2. The oracle prior $A^\*_0$ was handed to CausalMorph in $[\text{from},\text{to}]$;
   CausalMorph read it as the reversed graph. Its transform imposes the
   prior's orientation, so DirectLiNGAM on the transformed window returned
   the reversed graph — which (1) then scored as correct. The two errors
   cancelled for the seeded pipeline and compounded for everything else.
3. The chain prior, $\mathbb{1}[|B_{\text{lingam}}| > \tau]$, carried the
   reversal forward window by window.
4. The prior-free baselines (DirectLiNGAM, ICA-LiNGAM per window) produced
   correctly oriented graphs and were scored against the transpose. Their
   reported $F_1$ of 0.136 and 0.014, and the 99.4% / 99.99% win-rates
   against them, are artefacts of (1).

So the October tables measured the pipeline's ability to recover the
**reverse** of the true DAG, seeded with the reverse of the true prior. The
"anchor leak" of §2.4 was not a 1.5× effect on window 0; it was the
orientation of every edge.

### 8.3 The fix

One convention inside `full_pipeline.py`: $A[\text{from},\text{to}]$.
`_from_lingam(B) = B^\top$ at all five `lingam` outputs; `_to_causalmorph(A)
= A^\top$ at all three CausalMorph calls; `_residual_indep_score` takes
$A[\text{from},\text{to}]$ ($E = X - XA$). Scoring unchanged. Tests on a
$V_1 \to V_2 \to V_3$ SEM: cold pipeline, bootstrap pipeline and both baseline
learners now score recall $\ge 0.5$ against the truth (each was 0.0 with
SHD 4 — both edges missing, both reversals present).

### 8.4 First corrected numbers (600-scenario sample, same scenarios)

L2 = v1H windows, cold start, **no bootstrap**, consensus 0.5, merge-for-extraction.

| metric | Oct-5 "winning" as reported | L2, corrected |
|---|---|---|
| per-regime precision ↑ | 0.469 | **0.681** |
| per-regime recall ↑ | 0.433 | **0.593** |
| mean nSHD ↓ | 0.231 | **0.154** |
| windowed nSHD ↓ | 0.245 | **0.163** |
| transition SHD ↓ | 0.186 | **0.130** |
| dynamic score ↓ | 0.222 | **0.167** |
| consensus precision ↑ | 0.338 | **0.693** |
| consensus recall ↑ | 0.724 (complete-graph bug, §2.1) | 0.396 |
| consensus $F_1$ ↑ | 0.426 | **0.488** |

The pipeline without bootstrap and without any oracle beats every October
figure. It was always better than it reported.

### 8.5 What this changes in the plan

- §4's predictions for structure metrics are void; §4's detection
  predictions stand (detection never touches adjacencies).
- The honest "before" for any before/after comparison is the old
  configuration **re-scored under the fixed convention**
  (`--oracle_anchor --consensus_threshold 0.2 --legacy_postprocessing`, row
  `C0_legacy`), not the October CSVs. The October CSVs stay as the record of
  what was reported.
- The baseline comparison (C3, X1) must be redone; the LiNGAM baselines are
  far stronger than reported and the pipeline's margin over them is now a
  real, measurable quantity rather than a metric artefact.
- `causalmorph/utils/metrics.py` and any notebook that scores `lingam`
  output against `nx`-generated truth should be audited for the same
  comparison. The KBS paper's evaluation code is a separate codebase and is
  not covered by this audit.

---

## 9. `transition_shd` rewards missing transitions (found 2026-10-06)

`transition_shd` aligns each true regime to its best-overlap window and
charges $|\delta_{\text{true}} \oplus \delta_{\text{pred}}| / p(p-1)$ per
transition. When a transition is **missed**, regimes $k$ and $k{+}1$ align to
the same window, $\delta_{\text{pred}} = \emptyset$, and the charge is
$|\delta_{\text{true}}|$ — one to four edges on this grid. When it is
**detected**, $\delta_{\text{pred}}$ is the symmetric difference of two noisy
fits and carries both fits' estimation noise. Measured on the 600-scenario
sample, by segmentation outcome:

| row | exact | over | **under** |
|---|---|---|---|
| C0 (old config, corrected) | 0.217 | 0.124 | **0.114** |
| L2 (cold, no bootstrap) | 0.162 | 0.126 | **0.099** |
| L3 v1J | 0.193 | 0.177 | **0.085** |

Under-segmentation scores best in every configuration: the metric is
minimised by predicting no change. `windowed_nshd_weighted` has the opposite
blind spot — a regime split into two windows keeps only the larger overlap,
so spurious change points cost nothing beyond shorter fits.

**Replacement: `transition_f1`** — $\delta_{\text{true}}$ and
$\delta_{\text{pred}}$ as edge sets, $F_1$ per transition, averaged. A missed
transition scores 0; a learned transition with one spurious change scores
$2/3$; no true change and none predicted scores 1. Added to the pipeline and
both baseline arms (`run_experiments.py`), tested first. Rows produced before
2a6048e do not carry it; the final ladder will. `transition_shd` stays in the
CSV for continuity and should not be used to rank configurations.

---

## 10. Under the corrected convention, bootstrap + overlap hurt (600-scenario sample)

All rows cold start, consensus 0.5, merge-for-extraction, corrected
convention. Reference: L2 = v1H windows, **no bootstrap, no overlap**.

| paired vs L2 | C0 (old config, corrected) | L3 v1J $\gamma$2.5 | L3 v1L $\gamma$2.0 | L3 v1L $\gamma$2.5 |
|---|---|---|---|---|
| per-regime precision | −11.7% | −9.9% | −8.4% | −8.5% |
| per-regime recall | −8.8% | −15.9% | −13.0% | −12.2% |
| mean nSHD | −20.3% | −26.5% | −23.4% | −22.1% |
| windowed nSHD | −19.0% | −18.2% | −16.2% | −15.7% |
| consensus $F_1$ | −0.7% | −19.0% | −15.5% | −14.4% |
| dynamic score | −20.3% | +25.5% | +31.2% | +36.1% |

L3 rows = bootstrap 20 at threshold 0.60 + overlap 0.15 on the new detectors'
windows. They lose to L2 on every structure metric and win only dynamic
score, which carries detection $F_1$. On **exactly segmented** scenarios the
gap persists (windowed nSHD 0.140 for L2 vs 0.183–0.190 for the L3 rows), so
it is the extraction, not the windows. The three detectors differ by at most
two points among themselves, ordered by less over-segmentation.

Reading: in the reversed orientation of §8, the bootstrap's 60% consensus
suppressed noisy reversed edges and looked like a +9.4% precision gain. In
the correct orientation the same threshold cuts true edges (recall −12 to
−16%). The "winning configuration" of the report was tuned on the reversed
problem. `run_bootstrap_sweep.sh` separates overlap from bootstrap and
lowers the threshold (0.30 / 0.40 / 0.50) on v1L $\gamma$2.5 windows; phase 2
adds L3 on v1H windows (bootstrap at equal windows) and L2 on v1J / v1L
windows (detector at no bootstrap).

Detector choice: second-order next to extraction. Among the three, v1L
$\gamma$2.5 is marginally best on structure and has 9% over-segmentation
versus 29% for v1J; it is the provisional choice for the ladder, to be
confirmed by the no-bootstrap rows.

---

## 11. CausalMorph on data that already satisfies LiNGAM's assumptions

Phase 2 isolated the two mechanisms at equal windows (600-scenario sample,
corrected convention, cold start):

- **Bootstrap 20 @ 0.60 + overlap 0.15, same v1H windows:** −6.8% per-regime
  precision, −8.7% recall, −10.1% nSHD, −12.0% consensus $F_1$ vs no bootstrap.
- **Detector at no bootstrap:** v1J / v1L vs v1H: +5.7% / +3.9% precision,
  recall flat, windowed nSHD +2.9% / +1.5%; v1J ≈ v1L on structure, v1L with
  9% over-segmentation vs 29%. **v1L $\gamma$2.5 chosen.**
- **Baselines on the pipeline's own v1H windows, correctly scored**
  (`L3_cold_v2`): pipeline $F_1$ 0.566 / nSHD 0.169; DirectLiNGAM per window
  0.743 / 0.147; **ICA-LiNGAM per window 0.848 / 0.077.**

To tell design from a residual bug, one stationary experiment (linear SEM,
Laplace noise, $p=6$, $n=1500$, density 0.3, 8 seeds):

| | precision | recall |
|---|---|---|
| DirectLiNGAM raw | 1.000 | 1.000 |
| ICA-LiNGAM raw | 1.000 | 1.000 |
| DirectLiNGAM ∘ CausalMorph, prior = **truth** | 0.858 | 0.835 |
| DirectLiNGAM ∘ CausalMorph, prior = DirectLiNGAM estimate (cold path) | 0.876 | 0.853 |
| DirectLiNGAM ∘ CausalMorph, prior = truth **untransposed** (the old bug) | 0.030 | 0.036 |
| DirectLiNGAM ∘ CausalMorph, prior = empty | 1.000 | 1.000 |

With the true graph as prior, CausalMorph's transform turns a perfect
recovery into ~85%; with an empty prior it is the identity. The chain prior
then re-applies this loss window after window — that is the whole gap
between L2 (recall 0.59) and DirectLiNGAM on the same windows (0.84).

This is consistent with what CausalMorph is: a preconditioner for data that
**violate** LiNGAM's assumptions (KBS 2026: nonlinear mechanisms, Gaussian-
leaning noise). This grid is linear with Laplace innovations — LiNGAM's own
model — where a preconditioner has nothing to fix and can only remove
signal.

### Consequences

1. On this benchmark the honest extraction is **LiNGAM per window without
   CausalMorph** (ICA-LiNGAM best). The pipeline's defensible contribution is
   the segmentation — detection recall 0.48 → 0.87 at precision 0.91 (v1L)
   — and the per-window / consensus machinery around it.
2. CausalMorph's contribution needs a grid that violates the assumptions it
   was built for: nonlinear mechanisms and/or Gaussian noise, as in the KBS
   evaluation. That is a benchmark-design decision for the advisors, not a
   tuning problem.
3. The chain prior, as implemented (CausalMorph imposes the previous window's
   orientation), is harmful on this grid; whether it helps on an
   assumption-violating grid is open.
4. The October "99.4% / 99.99% win-rates against the baselines" invert: the
   baselines win.

Final rows for the figures: `run_final_rows.sh` — v1L $\gamma$2.5 windows, no
bootstrap, with the per-window learners (LB), the oracle-segmentation
ceiling (C2) and the whole-series control with every learner (C3).

---

## 12. Extraction sweep: overlap alone and bootstrap thresholds (2026-10-07)

All rows: v1L $\gamma$2.5 windows, cold start, consensus 0.5, corrected
convention. Reference: no bootstrap, no overlap (`L2_cold_v1L25_v2`).
Relative change vs the reference, positive = better.

| | overlap 0.15 alone | boot @0.30 + ov | boot @0.40 + ov | boot @0.50 + ov | boot @0.60 + ov |
|---|---|---|---|---|---|
| per-regime precision | −15.8% | −21.1% | −18.0% | −15.2% | −11.9% |
| per-regime recall | −9.4% | −4.7% | −7.8% | −10.3% | −11.5% |
| windowed nSHD | −23.4% | −36.4% | −28.8% | −23.2% | −17.4% |
| transition $F_1$ | −31.9% | −34.5% | −32.8% | −32.8% | — |
| consensus $F_1$ | −13.9% | −11.3% | −12.7% | −14.6% | −14.8% |

**Overlap alone hurts** on every metric. It prepends 15% of the previous
window to each window, i.e. samples that belong to the previous regime by
construction; in the reversed orientation of §8 that smoothed the chain, in
the correct one it contaminates every fit.

**No bootstrap threshold recovers it.** Lowering the threshold trades
precision for recall monotonically (0.30: recall −4.7%, precision −21%);
nothing approaches the plain per-window fit. A clean bootstrap-without-
overlap row (`X_boot_t060_noov_v1L25_v2`) is queued to separate the two for
the figure; the ranking above does not depend on it.

**Extraction chosen for the figures:** v1L $\gamma$2.5 windows, cold start,
no bootstrap, no overlap, consensus 0.5 — and, per §11, the per-window
learner without CausalMorph as the "after" structure, with the pipeline's
own columns shown alongside.

## 13. Benchmark v2: the regimes CausalMorph was built for (2026-10-07)

Same 600 scenario ids as §7–§12 (same $p$, $R$, samples, density, seed),
regenerated under three assumption-violating settings, by
`run_benchmark_v2.sh`. Mechanism `nonlinear` means
$X_i = 0.3\,(X_{pa}w) + 0.7\,(X_{pa}w)^3 + \varepsilon_i$ (the cube; `tanh`,
`sigmoid` and `softplus` are linear to first order at the grid's signal
scale and reproduce the linear cell). Everything else as in §12: v1L
$\gamma$2.5 windows, cold start, no bootstrap, no overlap, consensus 0.5,
corrected convention. Per cell, the three learners see the **same windows**.

| cell | detection P / R | pipeline (CM + chain) F1 | DirectLiNGAM / window F1 | ICA-LiNGAM / window F1 | pipeline vs ICA, F1 win/loss |
|---|---|---|---|---|---|
| linear / Laplace (the grid) | 0.912 / 0.867 | 0.621 | 0.902 | **0.920** | 8 / 555 |
| nonlinear / Laplace | 0.398 / 0.626 | 0.409 | 0.631 | **0.810** | 14 / 563 |
| linear / Gaussian | 0.939 / 0.902 | 0.303 | 0.427 | **0.867** | 4 / 584 |
| nonlinear / Gaussian | 0.661 / 0.633 | 0.248 | 0.374 | **0.737** | 27 / 566 |

Three readings.

1. **Gaussian noise does not break ICA-LiNGAM on this generator** (0.867 on
   the linear/Gaussian cell). Non-identifiability of LiNGAM is a statement
   about the population model with *all* noises Gaussian; the generator's
   normalised weights $w/\|w\|_2 \in [1,3]^k$ plus the deviation-scaled noise
   leave enough asymmetry in the finite sample for ICA to orient edges.
   DirectLiNGAM, which relies on pairwise independence measures, does
   degrade (0.427). The cell where CausalMorph's premise holds most clearly
   is therefore the one where the comparison learner is least hurt.
2. **The pipeline is below both per-window learners in every cell**, and
   the gap is largest exactly where CausalMorph is supposed to help
   (linear/Gaussian: 0.303 vs 0.867, 4 wins in 588 pairs). On the
   assumption-satisfying grid, §11 attributed the loss to CausalMorph
   removing recoverable structure; the v2 cells show the loss is not
   specific to that regime.
3. **Detection, not extraction, is the binding constraint on the nonlinear
   cells.** v1L's precision falls to 0.40 on nonlinear/Laplace: the cube
   produces heavy-tailed marginals and the Fisher-standardised statistic's
   Gaussian null is miscalibrated. A per-cell $\gamma$ sweep is the next
   detection item; it does not change the extraction ranking, which is
   computed on identical windows.

### 13.0 Detection on the nonlinear cell: $\gamma$ does not fix it

v1L on nonlinear / Laplace, micro P / R at tolerance 200, 600 scenarios:

| $\gamma$ | P | R | F1 | under-segmented | over-segmented |
|---|---|---|---|---|---|
| 2.5 | 0.398 | 0.626 | 0.487 | 33.7% | 40.8% |
| 3.0 | 0.401 | 0.564 | 0.469 | 38.0% | 34.8% |
| 3.5 | 0.408 | 0.522 | 0.458 | 40.8% | 30.0% |
| 4.0 | 0.408 | 0.475 | 0.439 | 44.3% | 26.8% |

Precision is flat at 0.40 while recall falls: the false alarms are not
threshold noise, they are real shifts of the correlation statistic inside a
regime. With a cubic mechanism the pairwise correlation of $(X_i, X_j)$
depends on the realised range of the parent in each block, so blocks drawn
from the same regime differ in correlation by more than the Gaussian null
allows. The fix is a statistic that is invariant to monotone mechanisms
(a rank-correlation change statistic — Spearman or Kendall — with the same
Fisher standardisation) rather than a larger margin. Queued as a detector
item; it does not affect the extraction comparison, which uses identical
windows per cell.

### 13.1 CausalMorph is non-deterministic

`causalmorph_algorithm.py` synthesises the non-Gaussian residuals with
`np.random.laplace` and `np.random.shuffle` on numpy's **global** RNG,
unseeded (lines 54–57, 270, 457). Two calls on identical input differ
(measured max $|\Delta|$ = 7.06 on a three-variable SEM); the structure
fitted afterwards therefore differs between runs. This is the source of the
cross-run non-determinism in the October grid (identical scenarios, different
rows) and of the failing equality test for the cold mode. Fix in
`_causalmorph_seeded`: save the global state, seed from the window's stream,
restore. Every pipeline fit is now reproducible from `(seed, regime_idx)`.

### 13.2 The same convention mismatch exists in the CausalMorph repository's own benchmark

`causalmorph/experiments/benchmarks/run_synthetic_experiments.py` generates
with `causal_graph_simple_scenario_v25`, whose adjacency is
`nx.to_pandas_adjacency(G)`, i.e. $A[\text{from},\text{to}]$ (verified in
the IYCC copy, lines 58/100). It then

- passes `pd.DataFrame(adj_true)` straight to `causalMorphing`, which reads
  parents from **rows** (expects $[\text{to},\text{from}]$) — the §8
  orientation error, so the transform is applied to the *parents* of each
  edge rather than to the children; and
- scores `model.adjacency_matrix_` (lingam's $B[\text{to},\text{from}]$)
  against `adj_true` with `mycomparegraphs` **without a transpose**. On a
  perfectly recovered three-variable chain this returns
  TP = 0, FP = 2, FN = 2, reversed = 2, $F_1$ = 0, SHD = 6. With the
  transpose it returns TP = 2, FP = 0, FN = 0 — and reversed = −2, SHD = −2,
  because `reversed_edges = Σ[(Ê = 1) ∧ (Aᵀ = 1)] − TP` subtracts TP
  unconditionally.

Both are verified on the copy of the code vendored in this repository
(`Full-Flow/causalmorph`). Whether the KBS paper's reported numbers were
produced by this exact script is not something the code alone can tell; it
needs to be checked against the paper's experiment scripts before anything
is concluded about those results. What the code does show is that, as
written, a perfect recovery scores 0 and a fully reversed recovery scores
1, so "original vs transformed" comparisons made with it measure the
opposite of what they claim.

---

## 14. CausalMorph v2 (2026-10-07)

CausalMorph v2 keeps the three-stage architecture of the published algorithm
and changes what Stage II does to the residual. The published Stage II is
called **v1** below; the v1 numbers are our replication under the corrected
adjacency convention of §8, used as the ablation. The learner of record per
window is ICA-LiNGAM, and v2 is a way to get more out of it: no Stage II
change where the residual is already non-Gaussian, a gain where the noise is
Gaussian. Design and success criterion:
`docs/superpowers/specs/2026-10-07-causalmorph-v2-design.md`.

### 14.1 What Stage II did

On a chain $V_1 \to V_2 \to V_3$, v1 replaces the residual $\varepsilon$ of
each non-root node by an i.i.d. non-Gaussian draw, shuffled and recoloured to
the same variance. That synthetic residual is independent of the real one
(measured correlation −0.002), so the transformed variable keeps only the
part its parents explain: $\mathrm{corr}(V_2^{\text{new}}, V_2) = R^2 = 0.50$.
The consequence is that every edge leaving a non-root node is erased: the
direct $V_2 \to V_3$ signal given $V_1$ falls from 0.695 to 0.016. Root nodes
are untouched, so root → child edges survive, which is why the loss grows
with graph depth. Stage I (MDL-selected polynomial, Taylor linearisation) is
kept unchanged because the spec keeps it, not because it helps: on this
branch's evidence it costs 0.03 to 0.04 on nonlinear/Laplace, and lite is at
or below the learner alone in every benchmark cell (§14.3).

### 14.2 v2

For each non-root node $i$, in causal order:

1. The parents are taken from the **transformed** data $X_{\text{new}}$, so the
   structural equations the learner sees are consistent (v1 regresses on the
   original parents).
2. Stage I is unchanged: standardise the parents, select the polynomial
   degree by MDL, linearise around the coordinate-wise median, giving
   $y_{\text{lin}}$ and $\varepsilon = y - y_{\text{lin}}$.
3. Gate: if the Jarque-Bera $p$-value of $\varepsilon$ is below 0.01, the
   residual is already non-Gaussian and $\varepsilon' = \varepsilon$.
4. Otherwise $\varepsilon' = s \, F^{-1}_{\text{Laplace}}(r/(n+1))$, a monotone
   rank map of $\varepsilon$ to the Laplace quantiles, with $s$ chosen so
   that $\mathrm{sd}(\varepsilon') = \mathrm{sd}(\varepsilon)$.
5. Stage III: $\varepsilon'$ is orthogonalised against $[1, X_{\text{new}}[:, P]]$
   by a QR decomposition, rescaled to $\mathrm{sd}(\varepsilon)$, and added back
   to $y_{\text{lin}}$.

No random number is drawn anywhere: $\varepsilon'$ is a deterministic, strictly
monotone function of $\varepsilon$, so it keeps the sample pairing with every
descendant. The **lite** ablation switches Stage II off ($\varepsilon' = \varepsilon$
always) and keeps Stage I and III. As a probe, on the unit-test chain with
0.9 weights the direct $V_2 \to V_3$ partial signal given $V_1$ is 0.655 in the
original data and 0.655 after v2 (0.671 when the gate opens). (The 0.016 of
§14.1 is a different chain, with unit weights, so the two are not compared
here.)

### 14.3 Benchmark v2, four arms, same windows

Same 600 scenario ids and cells as §13; v1L $\gamma$2.5 windows, cold start,
no bootstrap, no overlap, consensus 0.5, corrected convention. Mean structure
$F_1$ per window. *none* is the learner alone on the same windows; *v1*,
*lite* and *v2* are the learner after the corresponding transform. The last
column is the paired per-scenario comparison of v2 against *none*, as
wins / losses (ties are not counted).

| cell | learner | none | v1 | lite | v2 | v2 vs none (win/loss) |
|---|---|---|---|---|---|---|
| linear / Laplace | DirectLiNGAM | 0.902 | 0.616 | 0.901 | 0.901 | 120 / 135 |
| linear / Laplace | ICA-LiNGAM | 0.920 | 0.717 | 0.920 | 0.921 | 142 / 128 |
| nonlinear / Laplace | DirectLiNGAM | 0.631 | 0.445 | 0.601 | 0.605 | 134 / 357 |
| nonlinear / Laplace | ICA-LiNGAM | 0.810 | 0.578 | 0.771 | 0.771 | 140 / 370 |
| linear / Gaussian | DirectLiNGAM | 0.427 | 0.327 | 0.424 | 0.419 | 182 / 268 |
| linear / Gaussian | ICA-LiNGAM | 0.867 | 0.701 | 0.863 | **0.878** | **283 / 201** |
| nonlinear / Gaussian | DirectLiNGAM | 0.374 | 0.275 | 0.360 | 0.369 | 182 / 220 |
| nonlinear / Gaussian | ICA-LiNGAM | 0.737 | 0.582 | 0.727 | **0.778** | **326 / 198** |

Reading the table by cell, for ICA-LiNGAM:

- **linear / Laplace:** a tie (0.921 vs 0.920); v2 makes no Stage II change
  where LiNGAM's own assumptions hold, against 0.717 for v1. It is not the
  identity: lite and v2 still alter 259 to 270 of the 600 linear/Laplace
  scenarios through Stages I/III (the non-tied pairs of §14.6).
- **nonlinear / Laplace:** v2 is 0.039 below the learner alone (0.771 vs
  0.810). The gate rarely fires here (0.07, below), so v2 is within 0.004 of
  lite (0.771 vs 0.771 for ICA-LiNGAM, 0.605 vs 0.601 for DirectLiNGAM): the
  loss is not Stage II but Stage I/III (linearisation, orthogonalisation and
  the transformed-parent regression), which lite shares, and which also costs
  DirectLiNGAM 0.026 (0.605 vs 0.631).
- **linear / Gaussian:** +0.011 (283 wins, 201 losses).
- **nonlinear / Gaussian:** +0.041 (326 wins, 198 losses).

With DirectLiNGAM, v2 matches the learner alone on linear/Laplace (0.901 vs
0.902, 120 / 135, within noise), is slightly below it on linear/Gaussian
(0.419 vs 0.427, 182 / 268), and does not rescue it on Gaussian noise: it
stays at 0.419 and 0.369 against ICA-LiNGAM's 0.878 and 0.778. Lite
sits at or below *none* in every cell, so linearisation without the gated
Stage II does not produce the ICA gain on Gaussian noise.
`n_transform_failed` was 0 on all 16 new v2 and lite rows (600 rows each): no
window fell back to the raw data. That column counts windows, not replicas
(with bootstrap, a window counts if any replica failed), and it does not count
the per-node skips inside v2 (a node left as is for too few rows, a constant
parent, a Stage I error or a degenerate residual).

Gate activation rate, by cell (`tools/v2_gate_rate.py`; K = 20 scenarios,
about 72 to 78 nodes per cell, so a coarse estimate; windows cut at the true
regimes, order and graph from a DirectLiNGAM fit on each window, non-root
nodes counted). A node is *gated* when its residual is mapped to Laplace
(Jarque-Bera $p \geq 0.01$):

| cell | non-root nodes | gated | rate |
|---|---|---|---|
| linear / Laplace | 72 | 2 | 0.03 |
| nonlinear / Laplace | 72 | 5 | 0.07 |
| linear / Gaussian | 77 | 75 | 0.97 |
| nonlinear / Gaussian | 78 | 76 | 0.97 |

The gate does what it is for: it is nearly closed on the Laplace cells and
nearly always open on the Gaussian ones.

### 14.4 Stationary replication, true prior and estimated prior

Stationary experiment (corrected convention, $W = 4$,
`tools/kbs_replication.py`), 69 configurations: 45 linear/Laplace,
12 linear/Gaussian, 12 nonlinear/Laplace. The prior passed to the transform
is either the true graph, walked in a topological order of the true graph,
or each learner's own estimate from the raw data: the graph and causal order
of ICA-LiNGAM fitted to the raw data for the ICA-LiNGAM rows, of DirectLiNGAM
for the DirectLiNGAM rows, which is what the pipeline does. This is a
deviation from spec §5, which names the DirectLiNGAM estimate as the prior for
both learners (controller ruling). An earlier version of this table passed the
true graph together with the learner's estimated order; since v2 reads each
node's parents from the already-transformed data, a child listed before its
parent was regressed on the parent's raw values (42 of 100 true edges were
child-first for DirectLiNGAM on linear/Gaussian). The true-prior rows below
replace it; the estimated-prior rows are unchanged. Mean structure $F_1$; no
run failed or returned NaN.

| cell | learner | prior | none | v1 | lite | v2 |
|---|---|---|---|---|---|---|
| linear / Laplace (45) | ICA-LiNGAM | true | 0.991 | 0.635 | 0.986 | 0.984 |
| linear / Laplace (45) | ICA-LiNGAM | estimated | 0.991 | 0.642 | 0.992 | 0.992 |
| linear / Laplace (45) | DirectLiNGAM | true | 0.993 | 0.540 | 0.987 | 0.986 |
| linear / Laplace (45) | DirectLiNGAM | estimated | 0.993 | 0.535 | 0.992 | 0.992 |
| linear / Gaussian (12) | ICA-LiNGAM | true | 0.934 | 0.767 | 0.947 | **0.958** |
| linear / Gaussian (12) | ICA-LiNGAM | estimated | 0.934 | 0.720 | 0.916 | 0.919 |
| linear / Gaussian (12) | DirectLiNGAM | true | 0.482 | 0.663 | 0.482 | **0.790** |
| linear / Gaussian (12) | DirectLiNGAM | estimated | 0.482 | 0.412 | 0.482 | 0.459 |
| nonlinear / Laplace (12) | ICA-LiNGAM | true | 0.832 | 0.674 | 0.749 | 0.749 |
| nonlinear / Laplace (12) | ICA-LiNGAM | estimated | 0.832 | 0.652 | 0.757 | 0.757 |
| nonlinear / Laplace (12) | DirectLiNGAM | true | 0.520 | 0.651 | 0.567 | 0.567 |
| nonlinear / Laplace (12) | DirectLiNGAM | estimated | 0.520 | 0.352 | 0.487 | 0.487 |

On linear/Laplace with the true prior, v1 reproduces the degradation of
§11 and §13.2 (ICA-LiNGAM 0.991 → 0.635), and v2 is within 0.007 of the
learner alone (0.984). With Gaussian noise v2 is the best arm under the true
prior (ICA-LiNGAM 0.958 vs 0.934; DirectLiNGAM 0.790 vs 0.482), but with the
estimated prior the gain mostly disappears (0.919 and 0.459), so it depends
on how good the prior is. On nonlinear/Laplace v2 is below the learner alone
for ICA-LiNGAM (0.749 vs 0.832), the same Stage I/III cost seen in §14.3; v1
is also below it. Two results a sceptic would look for go the other way:
with the true prior on nonlinear/Laplace, DirectLiNGAM is best under v1
(0.651), above v2 (0.567) and none (0.520), and on stationary
linear/Laplace DirectLiNGAM v2 is 0.986 against 0.993 for none. The 12-run
blocks are small, DirectLiNGAM on linear/Gaussian is noisy, so differences of
a few hundredths there are within noise, and the nonlinear/Gaussian cell was
not run on the stationary side.

**Gate threshold.** The threshold was swept once, on the true-prior runs:

| cell | learner | $\alpha = 0.001$ | $\alpha = 0.01$ | $\alpha = 0.05$ |
|---|---|---|---|---|
| linear / Laplace | ICA-LiNGAM | 0.984 | 0.984 | 0.984 |
| linear / Laplace | DirectLiNGAM | 0.986 | 0.986 | 0.986 |
| linear / Gaussian | ICA-LiNGAM | 0.963 | 0.958 | 0.962 |
| linear / Gaussian | DirectLiNGAM | 0.831 | 0.790 | 0.800 |
| nonlinear / Laplace | ICA-LiNGAM | 0.749 | 0.749 | 0.749 |
| nonlinear / Laplace | DirectLiNGAM | 0.567 | 0.567 | 0.567 |

The rule was to move off 0.01 only if another value were better on every block
for both learners. None is: 0.001 is better than 0.01 only on linear/Gaussian,
for both learners (0.963 vs 0.958, one of 12 runs differs; 0.831 vs 0.790,
two runs), and ties exactly on the other four blocks; 0.05 is also better only
on linear/Gaussian (0.962 vs 0.958; 0.800 vs 0.790, the latter two wins and
two losses). **0.01 is kept.** 0.001 is never worse than 0.01 on these
blocks, but its edge rests on three of 24 runs.

### 14.5 Verdict against the spec's success criterion

The criterion (design §2) is, per learner, that v2 + learner is at least as
good as the learner alone on linear/Laplace, and better in at least one
assumption-violating cell. **It is met for ICA-LiNGAM and not met for
DirectLiNGAM.** For ICA-LiNGAM v2 ties the learner alone on linear/Laplace
(0.921 vs 0.920) and is better on both Gaussian cells (+0.011 on
linear/Gaussian, 283 / 201; +0.041 on nonlinear/Gaussian, 326 / 198). For
DirectLiNGAM v2 matches the learner alone on the linear/Laplace cell
(0.901 vs 0.902) but is not better in any violating cell: it is below on
linear/Gaussian (0.419 vs 0.427), nonlinear/Gaussian (0.369 vs 0.374) and
nonlinear/Laplace (0.605 vs 0.631). Lite alone does not meet the criterion for
either learner (ICA-LiNGAM: 0.863 vs 0.867 and 0.727 vs 0.737 on the Gaussian
cells), so the contribution is the gated Stage II, not the linearisation.

Two limits go with this. The gain is modest where it exists (+0.011 and
+0.041 in mean $F_1$), and v2 loses about 0.04 on nonlinear/Laplace, a cost
that comes from Stage I/III (linearisation, orthogonalisation and the
transformed-parent regression), which lite shares, not from the gate.

A third limit is the Stage I floor. v2 skips a node only below
`_MIN_ROWS = 20` rows, an absolute floor that does not scale with the number
of parents. With 5 parents and fewer than about 320 rows, MDL still selects a
degree above 1, so the output drifts from the identity even with the gate
closed: the median correlation of a transformed node with its original values
is 0.44 at n = 20, 0.72 at n = 40 to 80, 0.95 at n = 160 and 1.00 at n = 320.
The gate itself has little power below about 100 rows. The sampled
true-regime windows (507 to 674 rows) do not reach this range, but short
detected windows can. The queued fix is to cap `max_degree` so that the number
of polynomial features stays at or below n/10; it is not made here, so that
the numbers of §14.3 stay tied to the code that produced them.

**Next.** The Stage I/III cost is about 0.03 to 0.04 on nonlinear/Laplace
(0.026 for DirectLiNGAM, 0.039 for ICA-LiNGAM). What Stage I does there is
narrow: when the residual is kept, the output is
$X_{\text{new}}[:, i] = y_{\text{lin}} + c\,(y - \Pi y)$, with $\Pi$ the
projection on $[1, X_{\text{new}}[:, P]]$, so Stage I's only effect is to
replace the OLS slope on the parents by the Taylor slope at the median (on a
cubic toy, OLS slope 8.21 against Taylor slope 0.235, correlation 0.76). The
first step is therefore an ablation of the OLS slope against the Taylor slope
on nonlinear/Laplace, before adopting any Stage I gate. The candidate fix, if
the Taylor slope is the cost, is to gate Stage I as well: apply the
polynomial linearisation only when the MDL-selected degree above 1 improves
the fit beyond a threshold. Flagged, not done.

### 14.6 Paired statistics

Paired per-scenario comparisons on benchmark v2 (600 scenarios per cell, the
rows of §14.3), from `tools/v2_paired_stats.py`, which writes
`results/hardening_validation/v2_paired_stats.csv`. $\Delta F_1$ is the mean
per-scenario difference in mean structure $F_1$ (first arm minus second), with
a 95% bootstrap CI (2000 resamples of the differences); $p$ is the Wilcoxon
signed-rank test on the non-zero differences; $r_{rb}$ is the rank-biserial
effect on the non-ties, (win − loss) / (win + loss).

| cell | comparison | ΔF1 | CI 95% | p | win/loss/tie | r_rb |
|---|---|---|---|---|---|---|
| lin_laplace | v2+ICA vs ICA alone | +0.001 | [−0.001, +0.002] | 0.49 | 142/128/330 | +0.05 |
| lin_laplace | v2+DL vs DL alone | −0.001 | [−0.003, +0.000] | 0.052 | 120/135/345 | −0.06 |
| lin_laplace | lite+ICA vs ICA alone | −0.000 | [−0.002, +0.001] | 0.39 | 124/135/341 | −0.04 |
| lin_laplace | v2+ICA vs v1+ICA | +0.204 | [+0.194, +0.214] | 1.8e−89 | 533/13/54 | +0.95 |
| lin_laplace | v2+ICA vs lite+ICA | +0.001 | [+0.000, +0.002] | 0.021 | 68/47/485 | +0.18 |
| nonlin_laplace | v2+ICA vs ICA alone | −0.039 | [−0.046, −0.031] | 3.6e−31 | 140/370/90 | −0.45 |
| nonlin_laplace | v2+DL vs DL alone | −0.026 | [−0.032, −0.020] | 2.2e−25 | 134/357/109 | −0.45 |
| nonlin_laplace | lite+ICA vs ICA alone | −0.038 | [−0.046, −0.031] | 7.9e−31 | 139/372/89 | −0.46 |
| nonlin_laplace | v2+ICA vs v1+ICA | +0.193 | [+0.182, +0.205] | 2.1e−81 | 522/27/51 | +0.90 |
| nonlin_laplace | v2+ICA vs lite+ICA | −0.000 | [−0.001, +0.000] | 0.69 | 60/49/491 | +0.10 |
| lin_normal | v2+ICA vs ICA alone | +0.011 | [+0.003, +0.018] | 1.9e−5 | 283/201/116 | +0.17 |
| lin_normal | v2+DL vs DL alone | −0.008 | [−0.011, −0.005] | 2.6e−8 | 182/268/150 | −0.19 |
| lin_normal | lite+ICA vs ICA alone | −0.004 | [−0.011, +0.002] | 0.30 | 209/234/157 | −0.06 |
| lin_normal | v2+ICA vs v1+ICA | +0.177 | [+0.167, +0.186] | 1.5e−86 | 529/23/48 | +0.92 |
| lin_normal | v2+ICA vs lite+ICA | +0.015 | [+0.009, +0.021] | 7.5e−7 | 280/187/133 | +0.20 |
| nonlin_normal | v2+ICA vs ICA alone | +0.042 | [+0.028, +0.056] | 1.3e−10 | 326/198/76 | +0.24 |
| nonlin_normal | v2+DL vs DL alone | −0.004 | [−0.008, −0.000] | 0.027 | 182/220/198 | −0.09 |
| nonlin_normal | lite+ICA vs ICA alone | −0.010 | [−0.021, +0.002] | 0.013 | 218/288/94 | −0.14 |
| nonlin_normal | v2+ICA vs v1+ICA | +0.196 | [+0.183, +0.209] | 8.7e−74 | 508/51/41 | +0.82 |
| nonlin_normal | v2+ICA vs lite+ICA | +0.051 | [+0.039, +0.064] | 1.1e−17 | 343/167/90 | +0.35 |

On linear/Laplace the CI of v2 against the learner alone lies inside ±0.003
for both learners, which is equivalence at that margin, not just a failure to
find a difference. On the Gaussian cells the ICA-LiNGAM gains are significant
with small-to-moderate effect sizes ($r_{rb}$ = 0.17 and 0.24), which is the
honest description, since with n = 600 the p-values alone say little. On
nonlinear/Laplace the loss is significant ($r_{rb}$ = −0.45) and the same for
lite (−0.46), i.e. a Stage I/III cost, not a Stage II one.
