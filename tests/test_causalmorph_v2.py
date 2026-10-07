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
