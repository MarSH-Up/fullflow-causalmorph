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
from scipy.stats import kurtosis, spearmanr

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
        # 0.695 -> 0.016 under v1 is spec §1's unit-weight probe; this chain (0.9 weights) measures 0.655
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


if __name__ == "__main__":
    unittest.main()
