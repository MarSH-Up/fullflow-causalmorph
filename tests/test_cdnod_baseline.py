"""
CD-NOD baseline (Huang et al., JMLR 2020) via causal-learn, evaluated with the
pipeline's own metric functions.
Run: /Users/mdlsh/miniconda3/envs/IYCC-env/bin/python -m unittest tests.test_cdnod_baseline -v
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
sys.path.insert(0, os.path.join(ROOT, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(ROOT, "causalmorph"))
warnings.filterwarnings("ignore")


class Conversion(unittest.TestCase):
    """causal-learn encoding: graph[j, i] == 1 and graph[i, j] == -1 means i -> j;
    -1/-1 undirected; 1/1 bidirected. The last row/column is the context index C."""

    def test_directed_undirected_bidirected_and_context_dropped(self):
        from cdnod_baseline import cdnod_to_adjacency
        g = np.zeros((4, 4), dtype=int)          # nodes 0,1,2 + C at index 3
        g[0, 1], g[1, 0] = -1, 1                 # 0 -> 1
        g[1, 2], g[2, 1] = -1, -1                # 1 -- 2
        g[0, 2], g[2, 0] = 1, 1                  # 0 <-> 2
        g[3, 2], g[2, 3] = -1, 1                 # C -> 2 (dropped)
        A, n_und = cdnod_to_adjacency(g, 3)
        want = np.array([[0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 0]])
        np.testing.assert_array_equal(A, want)
        self.assertEqual(n_und, 2)               # the undirected and the bidirected pair

    def test_reversed_direction(self):
        from cdnod_baseline import cdnod_to_adjacency
        g = np.zeros((3, 3), dtype=int)
        g[1, 0], g[0, 1] = -1, 1                 # 1 -> 0
        A, n_und = cdnod_to_adjacency(g, 2)
        np.testing.assert_array_equal(A, [[0, 0], [1, 0]])
        self.assertEqual(n_und, 0)


class Skeleton(unittest.TestCase):
    def test_orientation_is_ignored(self):
        from cdnod_baseline import skeleton_prf
        truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        est = truth.T.copy()                     # every edge reversed
        r = skeleton_prf(est, truth)
        self.assertEqual((r["Precision"], r["Recall"], r["F1"]), (1.0, 1.0, 1.0))

    def test_counts(self):
        from cdnod_baseline import skeleton_prf
        truth = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])   # pairs {0,1} {0,2}
        est = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])     # pairs {0,1} {1,2}
        r = skeleton_prf(est, truth)
        self.assertAlmostEqual(r["Precision"], 0.5)
        self.assertAlmostEqual(r["Recall"], 0.5)


class Subsample(unittest.TestCase):
    def test_time_uniform_sorted_unique_and_capped(self):
        from cdnod_baseline import time_uniform_subsample
        idx = time_uniform_subsample(10_000, 500)
        self.assertEqual(len(idx), 500)
        self.assertTrue(np.all(np.diff(idx) > 0))
        self.assertEqual(idx[0], 0); self.assertEqual(idx[-1], 9_999)
        np.testing.assert_array_equal(time_uniform_subsample(300, 500), np.arange(300))


def _two_regime_chain(T=1200, seed=0):
    """V1 -> V2 in regime 0; V1 -> V2 -> V3 in regime 1 (edge gained), Laplace noise."""
    rng = np.random.default_rng(seed)
    half = T // 2
    e = rng.laplace(size=(T, 3))
    v1 = e[:, 0]; v2 = 1.2 * v1 + e[:, 1]
    v3 = np.where(np.arange(T) < half, 0.0, 1.5) * v2 + e[:, 2]
    names = ["V1", "V2", "V3"]
    A0 = pd.DataFrame(0.0, index=names, columns=names); A0.loc["V1", "V2"] = 1
    A1 = A0.copy(); A1.loc["V2", "V3"] = 1
    return np.c_[v1, v2, v3], names, [half], [A0, A1]


class Evaluation(unittest.TestCase):
    def test_single_graph_uses_the_pipeline_metrics(self):
        from cdnod_baseline import evaluate_single_graph
        from full_pipeline import compute_shd
        X, names, cps, adjs = _two_regime_chain()
        A = adjs[1].values                        # predict regime 1's graph everywhere
        m = evaluate_single_graph(A, adjs, cps, len(X), names)
        per = [compute_shd(pd.DataFrame(A, index=names, columns=names), t) for t in adjs]
        self.assertAlmostEqual(m["mean_norm_shd"], np.mean([r["normalized_shd"] for r in per]), places=6)
        self.assertAlmostEqual(m["mean_struct_f1"], np.mean([r["F1"] for r in per]), places=6)
        # regime 0 differs by one edge out of 6 slots, regime 1 by none, equal lengths
        self.assertAlmostEqual(m["windowed_nshd_weighted"], (1 / 6) / 2, places=6)
        self.assertEqual(m["transition_f1"], 0.0)  # a single graph predicts no change
        # skeleton: regime 0 has 1 true pair, 2 predicted (P 0.5, R 1, F1 2/3); regime 1 exact
        self.assertAlmostEqual(m["mean_skeleton_f1"], np.mean([2 / 3, 1.0]), places=3)

    def test_run_cdnod_fisherz_recovers_the_union_skeleton(self):
        from cdnod_baseline import run_cdnod, skeleton_prf
        X, names, cps, adjs = _two_regime_chain(T=2000)
        A, info = run_cdnod(X, test="fisherz", n_sub=None)
        union = ((adjs[0].values + adjs[1].values) > 0).astype(int)
        self.assertEqual(skeleton_prf(A, union)["Recall"], 1.0)
        self.assertEqual(info["n_rows_used"], 2000)
        self.assertIn("runtime_s", info)

    def test_run_cdnod_segments_index(self):
        from cdnod_baseline import run_cdnod
        X, names, cps, adjs = _two_regime_chain(T=800)
        A, info = run_cdnod(X, test="fisherz", n_sub=None, c_index="segments", change_points=cps)
        self.assertEqual(A.shape, (3, 3))
        with self.assertRaises(ValueError):
            run_cdnod(X, test="fisherz", c_index="segments")       # change points required


class Runner(unittest.TestCase):
    def test_runner_row(self):
        from tools.run_cdnod import run_scenario
        X, names, cps, adjs = _two_regime_chain(T=600)
        meta = dict(scenario_id="t", p=3, n_regimes=2, samples_regime=300)
        row = run_scenario(X, names, cps, adjs, meta, test="fisherz", n_sub=None, c_index="time")
        self.assertEqual(row["status"], "ok", row.get("error_msg"))
        for k in ("windowed_nshd_weighted", "mean_struct_f1", "mean_skeleton_f1", "consensus_f1",
                  "transition_f1", "n_undirected", "runtime_s", "cdnod_test", "cdnod_c_index"):
            self.assertIn(k, row)
        self.assertEqual(row["cdnod_test"], "fisherz")


if __name__ == "__main__":
    unittest.main()
