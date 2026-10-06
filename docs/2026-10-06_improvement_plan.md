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
