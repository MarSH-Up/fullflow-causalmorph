# CausalMorph v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CausalMorph v2 (sample-wise, gated, deterministic Stage II; parents taken from transformed data), wire it into the pipeline as `--transform v1|v2|lite|none`, and evaluate the four arms on benchmark v2 and the stationary replication.

**Architecture:** A new module `causalmorph/core/causalmorph_v2.py` reuses v1's Stage I (`taylor_linearize`) and replaces Stage II by a Jarque-Bera-gated monotone rank map to Laplace; Stage III (QR orthogonalisation, variance matching) is reimplemented locally so v1 stays untouched. `full_pipeline.py` gets one dispatcher `_apply_transform(transform, X, order, adj, seed)` used at every point where `_causalmorph_seeded` is called today. `run_experiments.py` exposes `--transform` and records it. Evaluation reuses `run_cold_learner_rows.sh` and `tools/kbs_replication.py`.

**Tech Stack:** Python 3.11 (`/Users/mdlsh/miniconda3/envs/IYCC-env/bin/python`), numpy, pandas, scipy 1.11 (`jarque_bera`, `rankdata`, `laplace.ppf`), scikit-learn (via v1's `taylor_linearize`), `lingam`, `unittest` (no pytest in the env).

**Spec:** `docs/superpowers/specs/2026-10-07-causalmorph-v2-design.md`

## Global Constraints

- Run every test with `cd Full-Flow && /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_causalmorph_v2 -v` (and `tests.test_hardening` before each commit; 48 tests must stay green).
- `OMP_NUM_THREADS=1` for every Python invocation (BLAS pinned; matches the experiment drivers).
- `causalmorph/core/causalmorph_algorithm.py` is not modified (spec §7).
- The adjacency passed to `causalmorph_v2` is in v1's layout: parents of node i are the non-zeros of **row** i (`[to, from]`). The repository's truth convention is `A[from, to]`; convert at the boundary with the existing `_to_causalmorph(A) = A.T`.
- No random number is drawn in v2 (spec §3). `np.random` must not be touched.
- Commits: conventional prefixes (`feat:`, `test:`, `chore:`, `docs:`); no Co-Authored-By or AI attribution lines (user rule); never push.
- Default `gate_alpha = 0.01`; default `stage2 = "rankmap"`.

## Review Focus

Inputs the spec implies but the task tests below did not originally exercise; each now has a test in the owning task.

1. A residual with many ties (discrete or clipped data) — the rank map must still be monotone and finite; average ranks keep `r/(n+1)` strictly inside (0, 1). → Task 2, test `test_rankmap_handles_ties`.
2. A node whose parent set is empty after the convention conversion (a root) — must be returned untouched and never gated. → Task 3, test `test_roots_untouched`.
3. A window with n < 8 samples per node (Jarque-Bera needs n ≥ 2 but is meaningless; `taylor_linearize` with a degree-4 polynomial on 5 rows overfits to zero residual) — v2 must fall back to the original column instead of raising. → Task 3, test `test_tiny_window_falls_back`.
4. A constant parent column (variance < 1e-8) — v1 skips the node; v2 must do the same rather than let `StandardScaler` divide by zero. → Task 3, test `test_constant_parent_skipped`.
5. `transform="none"` must give **exactly** the learner's ablation result on the same window (same seed), otherwise the four-arm comparison is not paired. → Task 5, test `test_transform_none_matches_ablation`.

---

### Task 1: Module skeleton, convention, determinism

**Files:**
- Create: `causalmorph/core/causalmorph_v2.py`
- Create: `tests/test_causalmorph_v2.py`

**Interfaces:**
- Produces: `causalmorph_v2(X: pd.DataFrame, causal_order: list[int], adjacency_matrix: pd.DataFrame, stage2: str = "rankmap", gate_alpha: float = 0.01, return_details: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]`. Returned DataFrame has the same index, columns and dtype (float64) as `X`. `details[var_name]` is a dict with keys `gated: bool`, `jb_p: float`, `residual_before: np.ndarray`, `residual_after: np.ndarray`, `skipped: str | None`.

- [ ] **Step 1: Write the failing tests (convention, determinism, roots untouched)**

```python
# tests/test_causalmorph_v2.py
"""
CausalMorph v2: sample-wise, gated, deterministic Stage II.
Run: /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_causalmorph_v2 -v
"""
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "causalmorph"))
warnings.filterwarnings("ignore")


def _chain(n=3000, seed=21, noise="laplace", nonlinear=False):
    """V1 -> V2 -> V3. Returns X (DataFrame), A_truth [from, to], noise matrix e."""
    rng = np.random.default_rng(seed)
    e = rng.laplace(size=(n, 3)) if noise == "laplace" else rng.normal(size=(n, 3))
    v1 = e[:, 0]
    v2 = (0.3 * v1 + 0.7 * v1 ** 3 if nonlinear else 0.9 * v1) + e[:, 1]
    v3 = (0.3 * v2 + 0.7 * v2 ** 3 if nonlinear else 0.9 * v2) + e[:, 2]
    X = pd.DataFrame(np.c_[v1, v2, v3], columns=["V1", "V2", "V3"])
    A = pd.DataFrame(0.0, index=X.columns, columns=X.columns)
    A.loc["V1", "V2"] = 1.0; A.loc["V2", "V3"] = 1.0
    return X, A, e


def _cm_adj(A):
    """truth [from, to] -> the [to, from] layout v1/v2 read parents from."""
    return A.T


class Convention(unittest.TestCase):
    def test_single_edge_transforms_child_only(self):
        from core.causalmorph_v2 import causalmorph_v2
        rng = np.random.default_rng(0); n = 2000
        v1 = rng.normal(size=n); v2 = 0.8 * v1 + rng.normal(size=n)   # Gaussian: gate opens
        X = pd.DataFrame({"V1": v1, "V2": v2})
        A = pd.DataFrame([[0.0, 1.0], [0.0, 0.0]], index=X.columns, columns=X.columns)  # V1 -> V2
        out = causalmorph_v2(X, [0, 1], _cm_adj(A))
        np.testing.assert_array_equal(out["V1"].values, X["V1"].values)
        self.assertFalse(np.allclose(out["V2"].values, X["V2"].values))

    def test_roots_untouched(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain(noise="normal")
        out, det = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), return_details=True)
        np.testing.assert_array_equal(out["V1"].values, X["V1"].values)
        self.assertNotIn("V1", det)


class Determinism(unittest.TestCase):
    def test_two_calls_identical_and_global_rng_untouched(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain(noise="normal")
        np.random.seed(5); before = np.random.get_state()[1].copy()
        a = causalmorph_v2(X, [0, 1, 2], _cm_adj(A))
        after = np.random.get_state()[1].copy()
        b = causalmorph_v2(X, [0, 1, 2], _cm_adj(A))
        np.testing.assert_array_equal(a.values, b.values)
        np.testing.assert_array_equal(before, after)   # no draw from the global RNG

    def test_output_shape_and_columns(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain()
        out = causalmorph_v2(X, [0, 1, 2], _cm_adj(A))
        self.assertEqual(list(out.columns), list(X.columns))
        self.assertEqual(out.shape, X.shape)
        self.assertTrue(np.isfinite(out.values).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_causalmorph_v2 -v`
Expected: 4 errors, `ModuleNotFoundError: No module named 'core.causalmorph_v2'`.

- [ ] **Step 3: Write the module with Stage I + III and an identity Stage II**

```python
# causalmorph/core/causalmorph_v2.py
"""
CausalMorph v2.

Same three stages as the published algorithm, with Stage II replaced:

  I.   MDL-selected polynomial, Taylor linearisation (reused from v1).
  II.  Gated, sample-wise, deterministic non-Gaussianisation of the residual:
       keep the residual if Jarque-Bera rejects normality at gate_alpha,
       otherwise map it monotonically (by ranks) onto a Laplace of the same
       standard deviation.  (v1 replaced the residual by an independent,
       shuffled synthetic draw, which erased every edge leaving a non-root
       node.)
  III. Orthogonalise against [1, parents], match the residual variance,
       reconstruct.

Parents are read from the *transformed* data, so the structural equations
the downstream learner sees are mutually consistent.  Nothing random is
drawn.  Adjacency layout is the one v1 reads: parents of node i are the
non-zeros of row i.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, laplace, rankdata
from sklearn.preprocessing import StandardScaler

from .causalmorph_algorithm import taylor_linearize

_EPS_VAR = 1e-8
_MIN_ROWS = 20   # below this Stage I over-fits and the gate is meaningless


def _rank_map_to_laplace(eps: np.ndarray) -> np.ndarray:
    """Monotone, deterministic map of eps onto a Laplace with sd(eps)."""
    n = len(eps)
    u = rankdata(eps, method="average") / (n + 1.0)       # strictly in (0, 1)
    z = laplace.ppf(u)                                     # standard Laplace quantiles
    z = z - z.mean()
    sd_z = z.std()
    if sd_z < _EPS_VAR:
        return eps.copy()
    return z * (eps.std() / sd_z)


def _stage2(eps: np.ndarray, stage2: str, gate_alpha: float) -> Tuple[np.ndarray, bool, float]:
    """Returns (eps_prime, gated, jb_p)."""
    if stage2 == "off":
        return eps.copy(), False, float("nan")
    if stage2 != "rankmap":
        raise ValueError(f"stage2 must be 'rankmap' or 'off', got {stage2!r}")
    jb_p = float(jarque_bera(eps).pvalue)
    if jb_p < gate_alpha:                                  # already non-Gaussian: keep
        return eps.copy(), False, jb_p
    return _rank_map_to_laplace(eps), True, jb_p


def _stage3(eps_prime: np.ndarray, parents: np.ndarray, target_sd: float) -> Optional[np.ndarray]:
    """Project eps_prime off span{1, parents}; rescale to target_sd. None if degenerate."""
    design = np.column_stack([np.ones(len(eps_prime)), parents])
    Q, _ = np.linalg.qr(design)
    e_orth = eps_prime - Q @ (Q.T @ eps_prime)
    sd = e_orth.std()
    if sd < _EPS_VAR:
        return None
    return e_orth * (target_sd / sd)


def causalmorph_v2(
    X: pd.DataFrame,
    causal_order: List[int],
    adjacency_matrix: pd.DataFrame,
    stage2: str = "rankmap",
    gate_alpha: float = 0.01,
    return_details: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, dict]]]:
    if stage2 not in ("rankmap", "off"):
        raise ValueError(f"stage2 must be 'rankmap' or 'off', got {stage2!r}")
    var_names = list(X.columns)
    X_vals = X.to_numpy(dtype=float)
    X_new = X_vals.copy()
    adj = np.asarray(adjacency_matrix, dtype=float)
    n = X_vals.shape[0]
    details: Dict[str, dict] = {}

    for idx in causal_order:
        name = var_names[idx]
        parents = np.where(adj[idx, :] != 0)[0]
        if len(parents) == 0:
            continue                                        # root: untouched, not reported
        rec = {"gated": False, "jb_p": float("nan"), "residual_before": None,
               "residual_after": None, "skipped": None}
        details[name] = rec
        y = X_vals[:, idx]
        P = X_new[:, parents]                                # transformed parents
        if n < _MIN_ROWS:
            rec["skipped"] = "too_few_rows"; continue
        if np.any(np.var(P, axis=0) < _EPS_VAR):
            rec["skipped"] = "constant_parent"; continue
        try:
            P_scaled = StandardScaler().fit_transform(P)
            y_lin = taylor_linearize(P_scaled, y)
        except Exception as exc:                             # noqa: BLE001 — v1 behaviour
            rec["skipped"] = f"stage1:{type(exc).__name__}"; continue
        eps = y - y_lin
        sd_eps = eps.std()
        rec["residual_before"] = eps.copy()
        if sd_eps < _EPS_VAR:
            rec["skipped"] = "zero_residual"; continue
        eps_prime, gated, jb_p = _stage2(eps, stage2, gate_alpha)
        rec["gated"], rec["jb_p"] = gated, jb_p
        e_final = _stage3(eps_prime, P, sd_eps)
        if e_final is None:
            rec["skipped"] = "degenerate_after_orthogonalisation"; continue
        X_new[:, idx] = y_lin + e_final
        rec["residual_after"] = e_final

    out = pd.DataFrame(X_new, index=X.index, columns=var_names)
    return (out, details) if return_details else out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_causalmorph_v2 -v`
Expected: 4 tests, OK.

- [ ] **Step 5: Commit**

```bash
git add causalmorph/core/causalmorph_v2.py tests/test_causalmorph_v2.py
git commit -m "feat: CausalMorph v2 skeleton (stage I reused, deterministic stage II/III)"
```

---

### Task 2: Stage II behaviour — gate, sample-wise preservation, rank map

**Files:**
- Modify: `causalmorph/core/causalmorph_v2.py` (only if a test exposes a defect; the Task 1 code is expected to pass)
- Test: `tests/test_causalmorph_v2.py`

**Interfaces:**
- Consumes: `causalmorph_v2(...)` and the private helpers `_rank_map_to_laplace(eps)`, `_stage2(eps, stage2, gate_alpha)` from Task 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_causalmorph_v2.py` (before `if __name__ == "__main__":`):

```python
from scipy.stats import kurtosis, spearmanr


class StageTwo(unittest.TestCase):
    def test_laplace_residual_not_gated_and_kept_samplewise(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, e = _chain(noise="laplace")
        out, det = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), return_details=True)
        self.assertFalse(det["V2"]["gated"]); self.assertFalse(det["V3"]["gated"])
        r = np.corrcoef(det["V2"]["residual_before"], det["V2"]["residual_after"])[0, 1]
        self.assertGreater(r, 0.99)          # orthogonalisation changes it only marginally

    def test_gaussian_residual_gated_monotone_variance_kept(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, e = _chain(noise="normal")
        out, det = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), return_details=True)
        self.assertTrue(det["V2"]["gated"])
        b, a = det["V2"]["residual_before"], det["V2"]["residual_after"]
        self.assertGreater(spearmanr(b, a).correlation, 0.99)
        self.assertAlmostEqual(a.std() / b.std(), 1.0, places=6)
        self.assertGreater(kurtosis(a), 1.0)   # excess kurtosis: Laplace is 3, Gaussian 0

    def test_gate_at_small_n(self):
        from core.causalmorph_v2 import _stage2
        rng = np.random.default_rng(3)
        _, gated_l, _ = _stage2(rng.laplace(size=250), "rankmap", 0.01)
        _, gated_g, _ = _stage2(rng.normal(size=250), "rankmap", 0.01)
        self.assertFalse(gated_l); self.assertTrue(gated_g)

    def test_rankmap_handles_ties(self):
        from core.causalmorph_v2 import _rank_map_to_laplace
        eps = np.repeat(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 50)   # heavy ties
        z = _rank_map_to_laplace(eps)
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(z.std(), eps.std(), places=9)
        order = np.argsort(eps, kind="stable")
        self.assertTrue(np.all(np.diff(z[order]) >= -1e-12))              # monotone

    def test_stage2_off_is_identity(self):
        from core.causalmorph_v2 import _stage2
        eps = np.random.default_rng(1).normal(size=500)
        z, gated, p = _stage2(eps, "off", 0.01)
        np.testing.assert_array_equal(z, eps); self.assertFalse(gated); self.assertTrue(np.isnan(p))

    def test_lite_equals_v2_on_laplace(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain(noise="laplace")
        a = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), stage2="rankmap")
        b = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), stage2="off")
        np.testing.assert_allclose(a.values, b.values)   # gate never opens on Laplace
```

- [ ] **Step 2: Run the tests**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_causalmorph_v2 -v`
Expected: all 10 pass with the Task 1 implementation. If `test_laplace_residual_not_gated_and_kept_samplewise` fails on the 0.99 threshold, the cause is the orthogonalisation removing the parent-correlated part of a residual that Stage I left in; lower nothing — inspect `det["V2"]["jb_p"]` first (must be < 0.01), then check that `_stage3` is given `P` (transformed parents), not `P_scaled`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_causalmorph_v2.py causalmorph/core/causalmorph_v2.py
git commit -m "test: CausalMorph v2 stage II gate, monotone rank map, lite equivalence"
```

---

### Task 3: Stage III, chain probe, fallbacks

**Files:**
- Modify: `causalmorph/core/causalmorph_v2.py` (only if a test exposes a defect)
- Test: `tests/test_causalmorph_v2.py`

**Interfaces:**
- Consumes: `causalmorph_v2` and `_stage3(eps_prime, parents, target_sd)` from Task 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_causalmorph_v2.py`:

```python
def _partial_signal(a, b, given):
    """corr of the residuals of a and b after regressing each on `given`."""
    def resid(v):
        c = np.polyfit(given, v, 1); return v - np.polyval(c, given)
    return np.corrcoef(resid(a), resid(b))[0, 1]


class StageThreeAndFallbacks(unittest.TestCase):
    def test_final_residual_orthogonal_to_parents(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain(noise="normal")
        out, det = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), return_details=True)
        # V2's parent is V1 (untouched root); V3's parent is the transformed V2
        self.assertLess(abs(np.corrcoef(det["V2"]["residual_after"], out["V1"])[0, 1]), 1e-8)
        self.assertLess(abs(np.corrcoef(det["V3"]["residual_after"], out["V2"])[0, 1]), 1e-8)

    def test_chain_probe_direct_edge_survives(self):
        # v1 drops the V2->V3 partial signal given V1 from 0.695 to 0.016 on this chain
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain(n=5000, seed=0, noise="laplace")
        out = causalmorph_v2(X, [0, 1, 2], _cm_adj(A))
        self.assertGreater(_partial_signal(out["V2"].values, out["V3"].values, out["V1"].values), 0.6)

    def test_chain_probe_survives_when_gated(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain(n=5000, seed=0, noise="normal")
        out, det = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), return_details=True)
        self.assertTrue(det["V2"]["gated"])
        self.assertGreater(_partial_signal(out["V2"].values, out["V3"].values, out["V1"].values), 0.5)

    def test_tiny_window_falls_back(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain(n=10)
        out, det = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), return_details=True)
        np.testing.assert_array_equal(out.values, X.values)
        self.assertEqual(det["V2"]["skipped"], "too_few_rows")

    def test_constant_parent_skipped(self):
        from core.causalmorph_v2 import causalmorph_v2
        X, A, _ = _chain(noise="normal")
        X = X.copy(); X["V1"] = 1.0                    # constant root
        out, det = causalmorph_v2(X, [0, 1, 2], _cm_adj(A), return_details=True)
        np.testing.assert_array_equal(out["V2"].values, X["V2"].values)
        self.assertEqual(det["V2"]["skipped"], "constant_parent")
        self.assertIsNone(det["V3"]["skipped"])        # V3's parent V2 is fine
```

- [ ] **Step 2: Run the tests**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_causalmorph_v2 -v`
Expected: all 15 pass. If `test_chain_probe_direct_edge_survives` fails, print `_partial_signal` for the original `X` (expect ≈ 0.69) and for `out`; a value near the R² of V2 on V1 (≈ 0.45) means the residual was replaced rather than kept — check `_stage2` returned `eps.copy()` for `gated == False`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_causalmorph_v2.py causalmorph/core/causalmorph_v2.py
git commit -m "test: CausalMorph v2 orthogonality, chain probe, fallbacks"
```

---

### Task 4: End-to-end with a learner, and export

**Files:**
- Modify: `causalmorph/core/__init__.py`
- Test: `tests/test_causalmorph_v2.py`

**Interfaces:**
- Produces: `from core import causalmorph_v2` (and `from causalmorph.core import causalmorph_v2`) both work.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_causalmorph_v2.py`:

```python
class EndToEnd(unittest.TestCase):
    def _fit(self, df, seed=0):
        from lingam import DirectLiNGAM
        m = DirectLiNGAM(); m.fit(df)
        return (np.abs(m.adjacency_matrix_.T) > 0.05).astype(int)   # -> [from, to]

    def test_v2_plus_directlingam_equals_directlingam_on_linear_laplace(self):
        from core import causalmorph_v2
        X, A, _ = _chain(noise="laplace")
        est_raw = self._fit(X)
        est_v2 = self._fit(causalmorph_v2(X, [0, 1, 2], _cm_adj(A)))
        np.testing.assert_array_equal(est_raw, est_v2)
        np.testing.assert_array_equal(est_v2, A.values.astype(int))

    def test_v2_recovers_chain_with_gaussian_noise(self):
        # the regime CausalMorph is for: DirectLiNGAM alone is not reliable here
        from core import causalmorph_v2
        X, A, _ = _chain(n=4000, seed=7, noise="normal")
        est_v2 = self._fit(causalmorph_v2(X, [0, 1, 2], _cm_adj(A)))
        tp = int(((est_v2 == 1) & (A.values == 1)).sum())
        self.assertGreaterEqual(tp, 1)                 # at least one true edge, oriented
        rev = int(((est_v2 == 1) & (A.values.T == 1)).sum())
        self.assertLessEqual(rev, 1)
```

- [ ] **Step 2: Run to verify the import fails**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_causalmorph_v2.EndToEnd -v`
Expected: `ImportError: cannot import name 'causalmorph_v2' from 'core'`.

- [ ] **Step 3: Export from the package**

Edit `causalmorph/core/__init__.py` to:

```python
"""
CausalMorph Core Algorithm Module

Contains the main CausalMorph algorithm implementation including:
- Stage I: MDL-guided linearization
- Stage II: Non-Gaussian synthesis
- Stage III: Orthogonalization

causalmorph_v2: same stages, Stage II gated, sample-wise and deterministic.
"""

from .causalmorph_algorithm import causalMorph
from .causalmorph_v2 import causalmorph_v2

__all__ = ["causalMorph", "causalmorph_v2"]
```

- [ ] **Step 4: Run the tests**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_causalmorph_v2 -v`
Expected: 17 tests, OK. If `test_v2_recovers_chain_with_gaussian_noise` fails on `tp >= 1`, record the three estimated matrices in the commit message and keep the test; it is the spec's success criterion at unit scale and is allowed to be red until Task 7's evaluation says whether the gate threshold must change (do not weaken the assertion silently).

- [ ] **Step 5: Commit**

```bash
git add causalmorph/core/__init__.py tests/test_causalmorph_v2.py
git commit -m "feat: export causalmorph_v2; end-to-end tests with DirectLiNGAM"
```

---

### Task 5: Pipeline plumbing — `transform` argument

**Files:**
- Modify: `full_pipeline.py` (`RegimeStructure` ~line 51; `_causalmorph_seeded` ~line 113; `_bootstrap_fit` ~lines 665–740; `extract_causal_structures` ~lines 756–1020)
- Test: `tests/test_hardening.py` (new class `TransformOption`)

**Interfaces:**
- Consumes: `causalmorph_v2` (Task 4), existing `_causalmorph_seeded(X, causal_order, adj_truth, seed)`, `_to_causalmorph`, `_make_learner`.
- Produces: `_apply_transform(transform: str, X: pd.DataFrame, causal_order, adj_truth: pd.DataFrame, seed: int) -> pd.DataFrame` where `adj_truth` is `A[from, to]`; `extract_causal_structures(..., transform: str = "v1")`; `_bootstrap_fit(..., transform: str = "v1")`; `RegimeStructure.transform: str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hardening.py` before `if __name__ == "__main__":`:

```python
class TransformOption(unittest.TestCase):
    """extract_causal_structures(transform=...) selects the preconditioner."""

    def test_apply_transform_dispatch(self):
        from full_pipeline import _apply_transform
        X, names, A = AdjacencyConvention._chain_sem()
        df = pd.DataFrame(X, columns=names)
        none = _apply_transform("none", df, [0, 1, 2], A, 0)
        np.testing.assert_array_equal(none.values, df.values)
        v2 = _apply_transform("v2", df, [0, 1, 2], A, 0)
        lite = _apply_transform("lite", df, [0, 1, 2], A, 0)
        np.testing.assert_allclose(v2.values, lite.values)      # Laplace: gate closed
        v1 = _apply_transform("v1", df, [0, 1, 2], A, 0)
        self.assertFalse(np.allclose(v1.values, v2.values))
        with self.assertRaises(ValueError):
            _apply_transform("v3", df, [0, 1, 2], A, 0)

    def test_transform_recorded_and_v2_keeps_chain(self):
        from full_pipeline import extract_causal_structures, compute_shd
        X, names, A = AdjacencyConvention._chain_sem()
        s = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=0,
                                      prior_mode="cold", transform="v2")[0]
        self.assertEqual(s.transform, "v2")
        self.assertGreaterEqual(compute_shd(s.adjacency_matrix, A)["Recall"], 0.99)

    def test_transform_none_matches_ablation(self):
        from full_pipeline import extract_causal_structures, extract_causal_structures_ablation
        X, names, true_cps, *_ = _tiny_scenario()
        p = extract_causal_structures(X, true_cps, names, verbose=False, bootstrap_n=0,
                                      prior_mode="cold", transform="none", window_overlap=0.0, seed=0)
        a = extract_causal_structures_ablation(X, true_cps, names, verbose=False,
                                               window_overlap=0.0, seed=0)["directlingam"]
        for s, t in zip(p, a):
            np.testing.assert_array_equal(
                (np.abs(s.adjacency_matrix.values) > 0).astype(int),
                (np.abs(t.adjacency_matrix.values) > 0).astype(int))

    def test_bootstrap_accepts_transform(self):
        from full_pipeline import extract_causal_structures
        X, names, A = AdjacencyConvention._chain_sem()
        s = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=4,
                                      bootstrap_threshold=0.5, prior_mode="cold", transform="v2")[0]
        self.assertEqual(s.transform, "v2")
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_hardening.TransformOption -v`
Expected: `ImportError: cannot import name '_apply_transform'` / `TypeError: unexpected keyword argument 'transform'`.

- [ ] **Step 3: Add the dispatcher next to `_causalmorph_seeded`**

In `full_pipeline.py`, directly after `_causalmorph_seeded`:

```python
TRANSFORMS = ("v1", "v2", "lite", "none")


def _apply_transform(transform: str, X: pd.DataFrame, causal_order, adj_truth: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Preconditioner applied to a window before the structure learner.
      v1   : the published CausalMorph (seeded; see _causalmorph_seeded)
      v2   : CausalMorph v2 — gated, sample-wise, deterministic Stage II
      lite : v2 with Stage II off (linearise + orthogonalise only)
      none : the raw window
    adj_truth is A[from, to]; conversion to the layout the transforms read
    happens here and nowhere else.
    """
    if transform == "none":
        return X
    if transform == "v1":
        return _causalmorph_seeded(X, causal_order, adj_truth, seed)
    if transform in ("v2", "lite"):
        from core.causalmorph_v2 import causalmorph_v2
        return causalmorph_v2(X, causal_order=list(causal_order), adjacency_matrix=_to_causalmorph(adj_truth),
                              stage2=("rankmap" if transform == "v2" else "off"))
    raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
```

Then add the import at module level next to the existing one (`full_pipeline.py` line 45 is `from core.causalmorph_algorithm import causalMorph`; `causalmorph/` is already on `sys.path` at line 34):

```python
from core.causalmorph_algorithm import causalMorph
from core.causalmorph_v2 import causalmorph_v2
```

and drop the local `from core.causalmorph_v2 import causalmorph_v2` line inside `_apply_transform`.

- [ ] **Step 4: Add `transform` to `RegimeStructure`, `_bootstrap_fit`, `extract_causal_structures`**

`RegimeStructure` (after `learner: str = "directlingam"`):

```python
    transform: str = "v1"                           # preconditioner: v1 | v2 | lite | none
```

`_bootstrap_fit` signature: add `transform: str = "v1"` after `learner`. Replace both `_causalmorph_seeded(sample_df, prior_order, prior_adj, seed_base + b * 7919)` and `_causalmorph_seeded(window_df, prior_order, prior_adj, seed_base)` calls by `_apply_transform(transform, sample_df, prior_order, prior_adj, seed_base + b * 7919)` and `_apply_transform(transform, window_df, prior_order, prior_adj, seed_base)`. Keep `use_cm`: when `use_cm` is False the raw data is used regardless of `transform`.

`extract_causal_structures` signature: add `transform: str = "v1",` after `seed: int = 0,` and document it in the docstring:

```python
    transform : preconditioner before the learner — "v1" (published CausalMorph,
      seeded), "v2" (gated sample-wise Stage II), "lite" (v2, Stage II off),
      "none" (raw window). See _apply_transform.
```

Inside: pass `transform=transform` to both `_bootstrap_fit` calls; replace `transformed = _causalmorph_seeded(window_df, causal_order_prior, adj_prior, wseed)` by `transformed = _apply_transform(transform, window_df, causal_order_prior, adj_prior, wseed)`; add `transform=transform,` to the `RegimeStructure(` construction at ~line 1001 (next to `learner=learner,`). Validate once at the top of the function:

```python
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
```

- [ ] **Step 5: Run the new tests and the whole suite**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_hardening -v 2>&1 | tail -5`
Expected: 52 tests, OK. If `test_transform_none_matches_ablation` fails, the two paths differ in the learner's seed or in the window boundaries: compare `s.window_start/window_end` with `t.window_start/window_end` first, then the `_make_learner` seed (`wseed` vs the ablation's seed) — `DirectLiNGAM` is deterministic, so a binary-pattern mismatch means different windows or different data.

- [ ] **Step 6: Commit**

```bash
git add full_pipeline.py tests/test_hardening.py
git commit -m "feat: --transform plumbing in extract_causal_structures (v1|v2|lite|none)"
```

---

### Task 6: CLI flag and results column

**Files:**
- Modify: `run_experiments.py` (`run_loaded` signature ~line 353–372; the `extract_causal_structures(` call ~line 450; the row dict ~line 505–512; argparse ~line 886; `run_loaded(` call in `main` ~line 917)
- Test: `tests/test_hardening.py` (`TransformOption`)

**Interfaces:**
- Consumes: `extract_causal_structures(..., transform=...)` from Task 5.
- Produces: `run_loaded(..., transform: str = "v1")`; CLI `--transform {v1,v2,lite,none}`; results column `transform`.

- [ ] **Step 1: Write the failing tests**

Append to `TransformOption` in `tests/test_hardening.py`:

```python
    def test_cli_accepts_transform(self):
        import subprocess, sys
        out = subprocess.run([sys.executable, "run_experiments.py", "--help"], capture_output=True, text=True, cwd=ROOT).stdout
        self.assertIn("--transform", out)
        self.assertIn("lite", out.split("--transform")[1][:80])

    def test_row_records_transform(self):
        from run_experiments import run_loaded
        X, names, true_cps, true_adjs, sizes = _tiny_scenario()
        meta = dict(scenario_id="t", p=3, n_regimes=2, n_changes=1, samples_regime=500,
                    base_pconn=0.5, noise_fraction=0.02, seed=11)
        row = run_loaded(X, names, true_cps, true_adjs, sizes, min_samples=500, meta=meta,
                         window_overlap=0.0, detector_version="v1L", corr_gamma=2.5, prior_mode="cold",
                         hybrid=False, bootstrap_n=0, bootstrap_threshold=0.6, chain_prior_threshold=0.05,
                         run_ablation=False, transform="lite")
        self.assertEqual(row["status"], "ok", row.get("error_msg"))
        self.assertEqual(row["transform"], "lite")
```

(This is the same call `RunOneFlags._row` in the same file makes, plus `transform="lite"`.)

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_hardening.TransformOption -v`
Expected: `AssertionError: '--transform' not found` and `TypeError: unexpected keyword argument 'transform'`.

- [ ] **Step 3: Thread the flag**

In `run_experiments.py`:

1. `run_loaded(...)`: add `transform: str = "v1",` after `learner: str = "directlingam",`.
2. In the `extract_causal_structures(` call that already passes `learner=learner, seed=seed,`, add `transform=transform,`.
3. In the row dict, after `"prior_mode": prior_mode,`, add `"transform": transform,`.
4. argparse, after the `--learner` argument:

```python
    exp.add_argument("--transform",            type=str, default="v1", choices=["v1", "v2", "lite", "none"],
                     help="preconditioner before the learner: v1 (published CausalMorph), v2 (gated sample-wise), lite (v2 without stage II), none")
```

5. In `main`, where `learner=args.learner,` is passed to `run_loaded`/`run_one`, add `transform=args.transform,`. If `run_one(sid, index_row, dataset_dir, **cfg)` forwards `**cfg`, the extra key flows through; confirm by grepping `def run_one` and the dict that builds `cfg`.

- [ ] **Step 4: Run the suite**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_hardening 2>&1 | tail -3`
Expected: 54 tests, OK.

- [ ] **Step 5: Smoke-run one scenario end to end**

Run:
```bash
cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow
head -3 results/hardening_validation/sample_ids.txt > /tmp/claude-501/smoke_ids.txt 2>/dev/null || head -3 results/hardening_validation/sample_ids.txt > "$TMPDIR/smoke_ids.txt"
OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python run_experiments.py --dataset_dir datasets --detector v1L --corr_gamma 2.5 --no-hybrid --bootstrap_n 0 --window_overlap 0.0 --prior_mode cold --transform v2 --no_ablation --workers 1 --batch_size 10 --scenario_ids_file "$TMPDIR/smoke_ids.txt" --output_dir "$TMPDIR/smoke_v2" | tail -3
OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -c "import pandas as pd; d=pd.read_csv('$TMPDIR/smoke_v2/results.csv'); print(d[['scenario_id','transform','learner','mean_struct_f1']])"
```
Expected: 3 rows, `transform == v2`, finite F1.

- [ ] **Step 6: Commit**

```bash
git add run_experiments.py tests/test_hardening.py
git commit -m "feat: --transform CLI flag; results record the preconditioner"
```

---

### Task 7: Evaluation rows on benchmark v2

**Files:**
- Modify: `run_cold_learner_rows.sh` (add the `v2` and `lite` arms)
- Modify: `analyze_hardening_validation.py` (`benchmark_v2_summary`, the `variants` list)

**Interfaces:**
- Consumes: `--transform` (Task 6); existing rows `LB_*` (learner-alone columns), `P_cold_*` / `P_cold_ica_*` (= `v1` arm, cold).
- Produces: rows `P_v2_<cell>`, `P_v2_ica_<cell>`, `P_lite_<cell>`, `P_lite_ica_<cell>` for `cell ∈ {lin_laplace, nonlin_laplace, lin_normal, nonlin_normal}`; the `--v2` report prints them.

- [ ] **Step 1: Extend the driver**

Replace the `for c in $CELLS; do ... done` loop in `run_cold_learner_rows.sh` with:

```bash
ARMS="${ARMS:-v1 v2 lite}"          # v1 rows are the existing P_cold_* / P_cold_ica_*
for c in $CELLS; do
  ds="$(dataset_for "$c")"
  for arm in $ARMS; do
    case "$arm" in
      v1)   run_row "$ds" "P_cold_$c"      --prior_mode cold
            run_row "$ds" "P_cold_ica_$c"  --prior_mode cold --learner icalingam
            run_row "$ds" "P_chain_ica_$c" --prior_mode chain --learner icalingam ;;
      v2)   run_row "$ds" "P_v2_$c"        --prior_mode cold --transform v2
            run_row "$ds" "P_v2_ica_$c"    --prior_mode cold --transform v2 --learner icalingam ;;
      lite) run_row "$ds" "P_lite_$c"      --prior_mode cold --transform lite
            run_row "$ds" "P_lite_ica_$c"  --prior_mode cold --transform lite --learner icalingam ;;
      *) echo "unknown arm $arm" >&2; exit 1 ;;
    esac
  done
done
```

`--resume` in `run_row` makes the `v1` rows no-ops when already complete.

- [ ] **Step 2: Extend the report**

In `analyze_hardening_validation.py`, `benchmark_v2_summary`, replace the `variants` list with:

```python
    variants = [("v1 + DL, cold (no chain)",  "P_cold_"),  ("v1 + ICA, cold (no chain)",  "P_cold_ica_"),
                ("v1 + ICA, chain prior",      "P_chain_ica_"),
                ("lite + DL, cold",            "P_lite_"),  ("lite + ICA, cold",            "P_lite_ica_"),
                ("v2 + DL, cold",              "P_v2_"),    ("v2 + ICA, cold",              "P_v2_ica_")]
```

and, in the per-variant paired line, add the comparison against the matching learner-alone column: for names containing `"ICA"` compare with `lb["icalingam_mean_f1"]`, otherwise with `lb["directlingam_mean_f1"]`:

```python
        for vname, v in extra:
            f = v.mean_struct_f1.reindex(lb.index)
            own = lb["icalingam_mean_f1"] if "ICA" in vname else lb["directlingam_mean_f1"]
            w1, l1 = wl(f, lb["mean_struct_f1"]); w2, l2 = wl(f, own)
            print(f"    {vname:<34} vs pipeline {w1}/{l1}   vs own learner alone {w2}/{l2}   (F1 win/loss)")
```

- [ ] **Step 3: Launch (background, 12 workers; ~4 min per row, 16 rows)**

Run:
```bash
cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow
PYTHON=/Users/mdlsh/miniconda3/envs/IYCC-env/bin/python ARMS="v2 lite" bash run_cold_learner_rows.sh > results/hardening_validation/v2_lite.log 2>&1 &
```
Wait for `COLD/LEARNER ROWS DONE` in the log (poll the log file, not `pgrep`).

- [ ] **Step 4: Report**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python analyze_hardening_validation.py --v2`
Expected: four cells, each with seven variant rows and their paired win/loss lines. Also compute the gate activation rate per cell with this script:

```python
# tools/v2_gate_rate.py
"""Share of non-root nodes whose residual the v2 gate maps (Jarque-Bera p >= alpha),
on the true regime windows of the first K sampled scenarios of each benchmark cell.
Run: cd Full-Flow && OMP_NUM_THREADS=1 python tools/v2_gate_rate.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph"))
from run_experiments import load_scenario
from full_pipeline import _make_learner, _from_lingam, _to_causalmorph
from core.causalmorph_v2 import causalmorph_v2

CELLS = {"lin_laplace": "datasets", "nonlin_laplace": "datasets_v2_nonlin_laplace",
         "lin_normal": "datasets_v2_lin_normal", "nonlin_normal": "datasets_v2_nonlin_normal"}
K = int(os.environ.get("K", "20"))
ids = [l.strip() for l in open(os.path.join(ROOT, "results/hardening_validation/sample_ids.txt")) if l.strip()][:K]

print(f"{'cell':<16}{'nodes':>7}{'gated':>7}{'rate':>7}")
for cell, ds in CELLS.items():
    index = pd.read_csv(os.path.join(ROOT, ds, "index.csv"), dtype=str).set_index("scenario_id")
    nodes = gated = 0
    for sid in ids:
        X, names, cps, adjs, sizes, _ = load_scenario(sid, os.path.join(ROOT, ds), index.loc[sid])
        bounds = [0] + list(cps) + [len(X)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            W = pd.DataFrame(X[a:b], columns=names)
            m = _make_learner("directlingam", 0).fit(W)
            est = pd.DataFrame((np.abs(_from_lingam(m.adjacency_matrix_)) > 0.05).astype(float), index=names, columns=names)
            _, det = causalmorph_v2(W, list(m.causal_order_), _to_causalmorph(est), return_details=True)
            for rec in det.values():
                if rec["skipped"] is None:
                    nodes += 1; gated += int(rec["gated"])
    print(f"{cell:<16}{nodes:>7}{gated:>7}{(gated / nodes if nodes else float('nan')):>7.2f}")
```

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && OMP_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python tools/v2_gate_rate.py`
Expected: four rows; the Laplace cells near 0.0, the Gaussian cells well above 0.5. A Laplace cell above 0.2 means the gate is too loose at the grid's window sizes and the Task 8 sweep decides the default.

- [ ] **Step 5: Commit**

```bash
git add run_cold_learner_rows.sh analyze_hardening_validation.py tools/v2_gate_rate.py
git commit -m "chore: v2/lite arms on benchmark v2; gate activation rate"
```

---

### Task 8: Stationary replication with the four arms

**Files:**
- Create: `tools/kbs_replication.py` (move from the session scratchpad and extend)

**Interfaces:**
- Consumes: `causalmorph_v2` (Task 4), `_causalmorph_seeded` and `compute_shd` from `full_pipeline`, `causal_graph_simple_scenario_v25` from `/Users/mdlsh/Desktop/Research/PhD/Code/IYCC/IYCC/src/components/scenarios/simple_scenario.py`.
- Produces: `results/kbs_replication/kbs_replication_arms.csv` and a printed table.

- [ ] **Step 1: Write the script**

```python
# tools/kbs_replication.py
"""
Stationary replication of the KBS experiment (corrected convention) with
four preconditioning arms: none, v1 (published, seeded), lite, v2 — each
with ICA-LiNGAM and DirectLiNGAM, with the TRUE graph as prior and with the
learner's own estimate as prior.

Run: cd Full-Flow && W=4 OMP_NUM_THREADS=1 python tools/kbs_replication.py
"""
import itertools, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "causalmorph"))
sys.path.insert(0, "/Users/mdlsh/Desktop/Research/PhD/Code/IYCC/IYCC/src")
from components.scenarios.simple_scenario import causal_graph_simple_scenario_v25
from full_pipeline import _apply_transform, _from_lingam, compute_shd, _make_learner
from concurrent.futures import ProcessPoolExecutor, as_completed

ARMS = ("none", "v1", "lite", "v2")


def binar(B, thr=0.05):
    return (np.abs(np.asarray(B)) > thr).astype(float)


def one(cfg):
    A, X = causal_graph_simple_scenario_v25(p=cfg["p"], pconn=0.3, dist=[cfg["dist"]] * cfg["p"], deviation=0.5,
            signal_strength=2.0, nsamples=cfg["n"], seed=cfg["seed"], mode=cfg["mode"], nonlin_func=cfg["func"],
            nonlinearity=cfg["nl"], max_attempts=20, min_std=1e-6, min_edges=1)
    X = X.astype(float); A = A.astype(float)
    out = dict(cfg)
    for alg in ("icalingam", "directlingam"):
        m0 = _make_learner(alg, cfg["seed"]).fit(X)
        est0 = pd.DataFrame(binar(_from_lingam(m0.adjacency_matrix_)), index=A.index, columns=A.columns)
        order0 = list(m0.causal_order_)
        for prior_name, prior_adj in (("true", A), ("est", est0)):
            for arm in ARMS:
                try:
                    T = _apply_transform(arm, X, order0, prior_adj, cfg["seed"])
                    m = _make_learner(alg, cfg["seed"]).fit(T)
                    est = pd.DataFrame(binar(_from_lingam(m.adjacency_matrix_)), index=A.index, columns=A.columns)
                    out[f"{alg}_{prior_name}_{arm}"] = compute_shd(est, A)["F1"]
                except Exception as exc:  # noqa: BLE001
                    out[f"{alg}_{prior_name}_{arm}"] = float("nan"); out["error"] = repr(exc)[:120]
    return out


def main():
    cfgs = []
    for p, n, r in itertools.product([5, 10, 20], [500, 1000, 2000], range(5)):
        cfgs.append(dict(p=p, n=n, seed=1000 * p + n + r, mode="linear", func="tanh", nl=0.0, dist="laplace"))
    for p, func, r in itertools.product([5, 10], ["tanh", "square"], range(3)):
        cfgs.append(dict(p=p, n=1000, seed=7000 + 100 * p + r, mode="nonlinear", func=func, nl=0.5, dist="laplace"))
    for p, n, r in itertools.product([5, 10], [1000, 2000], range(3)):
        cfgs.append(dict(p=p, n=n, seed=9000 + 100 * p + n + r, mode="linear", func="tanh", nl=0.0, dist="normal"))
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("W", "4"))) as ex:
        futs = [ex.submit(one, c) for c in cfgs]
        for f in as_completed(futs):
            rows.append(f.result())
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results", "kbs_replication"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "kbs_replication", "kbs_replication_arms.csv"), index=False)
    for (mode, dist), d in df.groupby(["mode", "dist"]):
        print(f"\n{mode} / {dist}   n runs={len(d)}   mean F1")
        print(f"  {'':24}" + "".join(f"{a:>8}" for a in ARMS))
        for alg in ("icalingam", "directlingam"):
            for prior in ("true", "est"):
                print(f"  {alg + ' / prior ' + prior:<24}" + "".join(f"{d[f'{alg}_{prior}_{a}'].mean():>8.3f}" for a in ARMS))


if __name__ == "__main__":
    main()
```

Note `log1p` is excluded: it produced NaNs in the session's probe (negative inputs); the paper's list keeps it but the generator does not guard it.

- [ ] **Step 2: Run**

Run: `cd /Users/mdlsh/Desktop/Research/PhD/ThirdYear/Full-Flow && W=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python tools/kbs_replication.py 2>&1 | grep -vE "Warning|warn|OMP"`
Expected: three blocks (linear/laplace, nonlinear/laplace, linear/normal), each 4 rows × 4 arms; `none` ≈ 0.99 on linear/laplace for ICA; `v1` well below `none` on linear/laplace (reproduces §13.2); `v2` and `lite` ≥ `none` − 0.02 on linear/laplace. Anything else is a finding, not a bug, unless `error` is non-empty.

- [ ] **Step 3: Sweep the gate threshold (spec §3)**

Add to `tools/kbs_replication.py`, inside `one(cfg)` after the arms loop, a direct sweep for the v2 arm with the true prior:

```python
        from core.causalmorph_v2 import causalmorph_v2
        from full_pipeline import _to_causalmorph
        for alpha in (0.001, 0.01, 0.05):
            T = causalmorph_v2(X, order0, _to_causalmorph(A), gate_alpha=alpha)
            m = _make_learner(alg, cfg["seed"]).fit(T)
            est = pd.DataFrame(binar(_from_lingam(m.adjacency_matrix_)), index=A.index, columns=A.columns)
            out[f"{alg}_true_v2_a{alpha}"] = compute_shd(est, A)["F1"]
```

and print, per (mode, dist) block, the three means per learner. Keep 0.01 as the default unless 0.001 or 0.05 is better on **every** block for both learners; record the decision in §14.

- [ ] **Step 4: Commit**

```bash
git add tools/kbs_replication.py
git commit -m "chore: stationary replication with none/v1/lite/v2 arms and gate sweep"
```

---

### Task 9: Write up §14 and refresh the summary tables

**Files:**
- Modify: `docs/2026-10-06_improvement_plan.md` (append §14)

- [ ] **Step 1: Append §14**

Fill from the Task 7 `--v2` output and the Task 8 table. Structure (write the real numbers; no placeholders):

```markdown
## 14. CausalMorph v2 (2026-10-07)

### 14.1 What Stage II did
<one paragraph: synthetic residual independent of the real one (corr −0.002), V2 keeps R² of itself, V2→V3 partial signal 0.695 → 0.016; v1 erases every edge leaving a non-root node>

### 14.2 v2
<the algorithm in five lines: transformed parents, Stage I unchanged, Jarque-Bera gate at 0.01, monotone rank map to Laplace with matched sd, QR orthogonalisation; no RNG>

### 14.3 Benchmark v2, four arms, same windows
| cell | learner | none | v1 | lite | v2 | v2 vs none (win/loss) |
<8 rows: 4 cells × {DL, ICA}>

Gate activation rate per cell: <4 numbers>

### 14.4 Stationary replication, true prior and estimated prior
<the Task 8 table>

### 14.5 Verdict against the spec's success criterion
<one paragraph: does v2 ≥ none on linear/Laplace for both learners; does v2 > none in at least one violating cell for both; is lite enough>
```

- [ ] **Step 2: Commit**

```bash
git add docs/2026-10-06_improvement_plan.md
git commit -m "docs: §14 CausalMorph v2 results"
```
