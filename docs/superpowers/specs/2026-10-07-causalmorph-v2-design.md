# CausalMorph v2 — design

Date: 2026-10-07. Branch: `feature/pipeline-hardening`. Status: approved in
conversation, awaiting written review.

## 1. Why

With the adjacency convention corrected (plan §8), CausalMorph lowers the
structure learner's accuracy in every regime of benchmark v2 and in a
stationary replication of the KBS experiment with the true graph as prior
(linear/Laplace: ICA-LiNGAM 0.99 → 0.64 F1). The cause is Stage II as
implemented: the residual ε of each non-root node is replaced by an i.i.d.
draw from a non-Gaussian family, shuffled, and recoloured to the same
variance. The synthetic residual is independent of the real one
(measured corr −0.002). Each transformed variable therefore keeps only the
part explained by its parents (corr(V2_new, V2) = R² = 0.50 on a chain), and
every edge leaving a non-root node is erased: the direct V2 → V3 signal given
V1 falls from 0.695 to 0.016. Root nodes are untouched, so root → child
edges survive, which is why the loss grows with graph depth.

Stage I (MDL-selected polynomial, Taylor linearisation) is the part that
helps, and only DirectLiNGAM on nonlinear data benefits from it today.

## 2. Goal and success criterion

Produce CausalMorph v2 as an improvement over the published algorithm, for
use in the nonstationary pipeline with an estimated per-window prior
(what governs), and as a standalone transform (the stationary setting).

Success, paired by scenario on the same windows:

- linear / Laplace (assumptions hold): v2 + learner ≥ learner alone;
- at least one assumption-violating cell of benchmark v2: v2 + learner >
  learner alone;
- both for ICA-LiNGAM and for DirectLiNGAM.

If the lite ablation (Stage II off) already meets this, the contribution is
the linearisation and is reported as such.

## 3. Algorithm

Inputs: X (n × p), causal order O, adjacency A in the repository convention
A[from, to] (converted at the boundary exactly as for v1), `stage2 ∈
{"rankmap", "off"}`, `gate_alpha` (default 0.01).

For each i in O with parent set P = {j : A[j, i] ≠ 0}; roots are skipped
(X_new[:, i] = X[:, i]).

Parents are taken from the **transformed** data X_new[:, P], so the
structural equations the learner sees are consistent (v1 regresses on the
original parents).

Stage I (unchanged from v1): standardise X_new[:, P], select the polynomial
degree by MDL, Taylor-linearise around the coordinate-wise median →
y_lin; ε = y − y_lin.

Stage II (new): compute the Jarque-Bera statistic of ε.

- If p < gate_alpha, ε is already non-Gaussian: ε' = ε.
- Otherwise ε is near-Gaussian: ε' = s · F⁻¹_Laplace(r / (n + 1)), where r
  are average ranks of ε (ties averaged), F⁻¹_Laplace is the standard Laplace
  quantile function, and s is chosen so that sd(ε') = sd(ε).

ε' is a deterministic, strictly monotone function of ε: it preserves the
sample pairing with every descendant, the ordering of observations, and the
variance. No random number is drawn anywhere in v2.

`stage2 = "off"` sets ε' = ε unconditionally (the lite ablation).

Stage III (unchanged in substance): Q from a QR decomposition of
[1, X_new[:, P]]; ε'⊥ = ε' − Q Qᵀ ε'; rescale ε'⊥ to sd(ε); X_new[:, i] =
y_lin + ε'⊥.

Failure handling per node, as in v1: if Stage I fails or sd(ε) < 1e-8 or
sd(ε'⊥) < 1e-8, X_new[:, i] = X[:, i] and the node is reported in the
details dict when requested.

Why Jarque-Bera rather than a fixed kurtosis threshold: its power grows with
n. At n = 5000 the gate opens only for genuinely Gaussian residuals; at
n = 250 a Laplace residual is still recognised as non-Gaussian. The
threshold is a parameter and is swept once (0.001, 0.01, 0.05) in the
evaluation.

## 4. Interfaces

New module `causalmorph/core/causalmorph_v2.py`; the published algorithm is
not modified.

```python
def causalmorph_v2(X: pd.DataFrame, causal_order: list[int],
                   adjacency_matrix: pd.DataFrame,   # [to, from], as v1 reads it
                   stage2: str = "rankmap", gate_alpha: float = 0.01,
                   return_details: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]
```

The adjacency argument keeps v1's layout so the existing boundary converter
`_to_causalmorph` applies unchanged. `return_details` returns, per node,
`gated` (bool), `jb_p`, `residual_before`, `residual_after`.

Pipeline (`full_pipeline.py`): `extract_causal_structures(..., transform="v1")`
with `transform ∈ {"v1", "v2", "lite", "none"}`; `"v1"` is the seeded
published algorithm (current behaviour), `"none"` skips the transform and
fits the learner on the window. `_bootstrap_fit` receives the same argument.
`RegimeStructure.transform` records it. `run_experiments.py`: `--transform`,
column `transform` in results.

## 5. Evaluation plan

Arms: `v1` (paper, seeded), `lite`, `v2`, `none`; each with
`learner ∈ {directlingam, icalingam}`, `prior_mode = cold`, no bootstrap, no
overlap, v1L γ2.5 windows, on the four benchmark v2 cells (same 600
scenario ids, `run_cold_learner_rows.sh` extended with `--transform`).
Existing rows already cover `v1` (P_cold_*, P_cold_ica_*) and `none`
(the LB_* ablation columns).

Stationary replication (`kbs_replication.py`, moved into `tools/`): the
same four arms with the true graph as prior and with the DirectLiNGAM
estimate as prior, corrected convention, 45 linear + nonlinear runs.

Report: a §14 in the improvement plan with the paired table per cell and
arm, and the gate's activation rate per cell.

## 6. Tests (written first)

In `tests/test_causalmorph_v2.py`:

1. non-gated residual is kept sample-wise: corr(ε', ε) = 1 on a Laplace
   residual;
2. gated residual is a monotone map: Spearman(ε', ε) = 1, sd(ε') = sd(ε),
   excess kurtosis of ε' > 1 on a Gaussian residual;
3. gate: Laplace residual (n = 250) not gated; Gaussian residual (n = 250)
   gated;
4. ε'⊥ is orthogonal to [1, parents] (|corr| < 1e-8);
5. determinism: two calls on the same input are identical;
6. chain probe: on V1 → V2 → V3 linear Laplace, the direct V2 → V3 partial
   signal given V1 after v2 is > 0.6 (v1: 0.016);
7. adjacency convention: with A[to, from] a single edge V1 → V2 transforms
   V2 and leaves V1 untouched;
8. `stage2="off"` equals v2 with the gate never open (Laplace data);
9. end to end: v2 + DirectLiNGAM on the chain recovers the same adjacency
   as DirectLiNGAM alone;
10. pipeline plumbing: `transform` recorded on `RegimeStructure`, CLI
    accepts `--transform`, `"none"` equals the learner's ablation column.

## 7. Out of scope

Transforming root nodes; target distributions other than Laplace; any
change to the detector or the consensus; modifying the published v1 code.
