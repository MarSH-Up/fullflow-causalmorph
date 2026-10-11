"""
Tests for the pipeline-hardening changes of 2026-10-06.

Each test names the production change that makes it pass. Run with:

    python -m unittest tests.test_hardening -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(ROOT, "causalmorph"))

import warnings
warnings.filterwarnings("ignore")


def _adj(names, edges):
    A = pd.DataFrame(0.0, index=names, columns=names)
    for a, b in edges:
        A.loc[a, b] = 1.0
    return A


def _struct(names, edges, n_samples, idx=0):
    from full_pipeline import RegimeStructure
    return RegimeStructure(
        regime_idx=idx, window_start=0, window_end=n_samples, n_samples=n_samples,
        causal_order=list(range(len(names))), adjacency_matrix=_adj(names, edges),
        used_prior=False,
    )


# ── 2.1 consensus ────────────────────────────────────────────────────────────

class ConsensusDecisionRule(unittest.TestCase):
    """aggregate_structures_bayesian: 0-1 loss => include iff posterior > 1/2."""

    names = ["V1", "V2", "V3", "V4"]

    def test_single_window_consensus_equals_that_window(self):
        from full_pipeline import aggregate_structures_bayesian
        s = _struct(self.names, [("V1", "V2"), ("V3", "V4")], 500)
        cons, _ = aggregate_structures_bayesian([s], self.names)
        self.assertTrue(np.array_equal(cons.values, s.adjacency_matrix.values),
                        "with one window the consensus must be that window, not the complete graph")

    def test_two_windows_absent_edge_is_excluded(self):
        from full_pipeline import aggregate_structures_bayesian
        s0 = _struct(self.names, [("V1", "V2")], 600, 0)
        s1 = _struct(self.names, [("V1", "V2")], 400, 1)
        cons, _ = aggregate_structures_bayesian([s0, s1], self.names)
        self.assertEqual(cons.loc["V3", "V4"], 0.0)
        self.assertEqual(cons.loc["V1", "V2"], 1.0)

    def test_length_weighted_majority(self):
        from full_pipeline import aggregate_structures_bayesian
        # edge only in the longer window (vote share 0.6) -> in; only in the shorter (0.4) -> out
        s0 = _struct(self.names, [("V1", "V2")], 600, 0)
        s1 = _struct(self.names, [("V3", "V4")], 400, 1)
        cons, _ = aggregate_structures_bayesian([s0, s1], self.names)
        self.assertEqual(cons.loc["V1", "V2"], 1.0)
        self.assertEqual(cons.loc["V3", "V4"], 0.0)

    def test_sparsity_prior_keeps_unanimous_edge_and_drops_absent(self):
        from full_pipeline import aggregate_structures_bayesian
        s = _struct(self.names, [("V1", "V2")], 500)
        cons, probs = aggregate_structures_bayesian([s], self.names,
                                                    prior_density=0.2, prior_strength=1.0)
        self.assertEqual(cons.loc["V1", "V2"], 1.0)
        self.assertEqual(cons.loc["V3", "V4"], 0.0)
        # absent edge posterior = kappa*rho0/(kappa+n) = 0.2/2
        self.assertAlmostEqual(probs.loc["V3", "V4"], 0.1, places=6)


# ── 2.2 detection / extraction decoupling ────────────────────────────────────

class ExtractionWindows(unittest.TestCase):
    """extraction_windows(cps, T, n_min) merges short segments into the shorter neighbour."""

    def test_short_segment_merges_into_shorter_neighbour(self):
        from run_experiments import extraction_windows
        # segments: 400 | 1000 | 300 | 700 ; n_min=500
        cps_ext, absorbed = extraction_windows([400, 1400, 1700], T=2400, n_min=500)
        self.assertEqual(cps_ext, [1400])
        self.assertEqual(sorted(absorbed), [400, 1700])

    def test_nothing_merged_when_all_long_enough(self):
        from run_experiments import extraction_windows
        cps_ext, absorbed = extraction_windows([1000, 2000], T=3000, n_min=500)
        self.assertEqual(cps_ext, [1000, 2000])
        self.assertEqual(absorbed, [])

    def test_single_short_series_yields_one_window(self):
        from run_experiments import extraction_windows
        cps_ext, absorbed = extraction_windows([200], T=400, n_min=500)
        self.assertEqual(cps_ext, [])
        self.assertEqual(absorbed, [200])


class MinWindowRule(unittest.TestCase):
    def test_linear_rule_with_floor(self):
        from run_experiments import min_window_for
        self.assertEqual(min_window_for(p=12, samples_per_node=40), 480)
        self.assertEqual(min_window_for(p=3, samples_per_node=40), 250)


# ── 2.3 detector statistic ───────────────────────────────────────────────────

class StandardisedCorrelationChange(unittest.TestCase):

    def test_fisher_z_change_has_unit_null_scale(self):
        from detectors.detectors_correlation import standardized_corr_change
        rng = np.random.default_rng(0)
        X = rng.standard_normal((12000, 6))
        zeta = standardized_corr_change(X, window=200)          # [T, m]
        inner = zeta[400:-400]
        inner = inner[np.isfinite(inner).all(axis=1)]
        # zeta(t) overlaps zeta(t+1) in W-1 samples, so each pair has only
        # ~T/W effective draws; the pooled scale over all pairs is the testable
        # quantity (n_eff ~ 900 here, SE ~ 0.025).
        pooled_sd = inner.std()
        self.assertTrue(0.9 < pooled_sd < 1.1, f"pooled null sd {pooled_sd:.3f}")

    def test_max_statistic_threshold_is_analytic(self):
        from detectors.detectors_correlation import detect_correlation_changes
        rng = np.random.default_rng(1)
        X = rng.standard_normal((3000, 5))
        _, _, thr = detect_correlation_changes(X, window=150, statistic="max", gamma=2.0)
        m = 5 * 4 / 2
        self.assertAlmostEqual(thr, np.sqrt(2 * np.log(m)) + 2.0, places=9)

    def test_max_statistic_finds_single_pair_change(self):
        from detectors.detectors_correlation import detect_correlation_changes
        rng = np.random.default_rng(2)
        T, p, W = 4000, 8, 150
        X = rng.standard_normal((T, p))
        # after T/2, make V2 strongly depend on V1; every other pair unchanged
        X[T // 2:, 1] = 0.9 * X[T // 2:, 0] + 0.3 * rng.standard_normal(T - T // 2)
        cps, _, _ = detect_correlation_changes(X, window=W, refractory_period=300,
                                               statistic="max", gamma=2.0)
        self.assertEqual(len(cps), 1, f"expected one change point, got {cps}")
        self.assertLess(abs(cps[0] - T // 2), W)

    def test_adaptive_null_tolerates_estimator_noise_drift(self):
        from detectors.detectors_correlation import detect_correlation_changes
        rng = np.random.default_rng(3)
        T, p, W = 6000, 6, 150
        # no correlation change anywhere: channels stay independent. The second
        # half is temporally autocorrelated (AR(1), phi=0.85), which cuts the
        # effective sample size per window ~4.5x and inflates the variance of
        # every sample correlation — a nuisance real recordings have and a
        # regime-0 threshold does not anticipate. (Scale or heavy tails alone
        # would not do it: r is scale-free and its variance ~1/W regardless.)
        white = rng.standard_normal((T, p))
        X = white.copy()
        phi = 0.85
        for t in range(T // 2 + 1, T):
            X[t] = phi * X[t - 1] + np.sqrt(1 - phi ** 2) * white[t]
        base = np.arange(0, 400)
        fixed, _, _ = detect_correlation_changes(X, window=W, refractory_period=300,
                                                 baseline_idx=base, threshold_mad_k=6.0,
                                                 min_threshold=0.5)
        adapt, _, _ = detect_correlation_changes(X, window=W, refractory_period=300,
                                                 baseline_idx=base, threshold_mad_k=6.0,
                                                 min_threshold=0.5, adaptive_null=True)
        self.assertGreaterEqual(len(fixed), 2, "premise: a regime-0 threshold must over-fire here")
        self.assertLessEqual(len(adapt), 1)


class StudentisedStatistic(unittest.TestCase):
    """Per-pair robust studentisation: zeta / (1.4826 * MAD_t zeta) restores unit
    null scale when the Fisher approximation does not (heavy tails, strong
    correlation, temporal autocorrelation)."""

    @staticmethod
    def _hard_null(T=12000, p=6, seed=5):
        # Laplace SEM with strong edges whose innovations are temporally
        # autocorrelated (AR(1), phi=0.85 — the grid's pink observation noise
        # has the same effect): no change point anywhere, but each window holds
        # far fewer effective samples than W, so Var(artanh r) >> 1/(W-3).
        rng = np.random.default_rng(seed)
        phi = 0.85
        E = rng.laplace(size=(T, p))
        for t in range(1, T):
            E[t] = phi * E[t - 1] + np.sqrt(1 - phi ** 2) * E[t]
        X = np.zeros((T, p))
        X[:, 0] = E[:, 0]
        for j in range(1, p):
            X[:, j] = 0.9 * X[:, j - 1] + E[:, j]
        return X

    def test_plain_fisher_scale_is_inflated_on_hard_null(self):
        from detectors.detectors_correlation import standardized_corr_change
        z = standardized_corr_change(self._hard_null(), window=200)
        z = z[np.isfinite(z).all(axis=1)]
        self.assertGreater(z.std(), 1.3, "premise: the Fisher scale must be badly off here")

    def test_studentised_scale_is_unit_on_hard_null(self):
        from detectors.detectors_correlation import standardized_corr_change
        z = standardized_corr_change(self._hard_null(), window=200, studentize=True)
        z = z[np.isfinite(z).all(axis=1)]
        self.assertTrue(0.8 < z.std() < 1.15, f"studentised pooled sd {z.std():.3f}")

    def test_studentised_max_keeps_analytic_threshold_and_finds_change(self):
        from detectors.detectors_correlation import detect_correlation_changes
        rng = np.random.default_rng(2)
        T, p, W = 4000, 8, 150
        X = rng.standard_normal((T, p))
        X[T // 2:, 1] = 0.9 * X[T // 2:, 0] + 0.3 * rng.standard_normal(T - T // 2)
        cps, _, thr = detect_correlation_changes(X, window=W, refractory_period=300,
                                                 statistic="max", gamma=3.0, studentize=True)
        self.assertAlmostEqual(thr, np.sqrt(2 * np.log(28)) + 3.0, places=9)
        self.assertEqual(len(cps), 1, f"expected one change point, got {cps}")
        self.assertLess(abs(cps[0] - T // 2), W)


class RefinedScale(unittest.TestCase):
    """One-step refinement of the per-pair scale: re-estimate the MAD excluding
    +-W around the change points of a first pass. With short regimes the
    change-point bumps occupy a large share of the series and inflate the
    single-pass MAD, which costs recall on the later, smaller changes."""

    @staticmethod
    def _short_regimes(seed=8, p=6, n_reg=6, L=600):
        # p channels, independent except one pair whose correlation flips sign
        # at every regime boundary: 5 changes of width 2W in 3600 samples.
        rng = np.random.default_rng(seed)
        T = n_reg * L
        X = rng.laplace(size=(T, p))
        for k in range(n_reg):
            a, b = k * L, (k + 1) * L
            rho = 0.7 if k % 2 == 0 else -0.7
            X[a:b, 1] = rho * X[a:b, 0] + np.sqrt(1 - rho ** 2) * X[a:b, 1]
        cps = [k * L for k in range(1, n_reg)]
        return X, cps

    def test_refined_scale_is_below_single_pass_on_the_changing_pair(self):
        from detectors.detectors_correlation import standardized_corr_change
        X, cps = self._short_regimes()
        W = 120
        _, s0 = standardized_corr_change(X, W, studentize=True, return_scale=True)
        mask = np.zeros(len(X), bool)
        for c in cps:
            mask[max(0, c - W):c + W] = True
        _, s1 = standardized_corr_change(X, W, studentize=True, scale_mask=mask, return_scale=True)
        self.assertLess(s1[0], 0.8 * s0[0], f"pair (0,1): single-pass {s0[0]:.2f}, refined {s1[0]:.2f}")

    def test_refined_null_scale_is_unit_inside_regimes(self):
        from detectors.detectors_correlation import standardized_corr_change
        X, cps = self._short_regimes()
        W = 120
        mask = np.zeros(len(X), bool)
        for c in cps:
            mask[max(0, c - W):c + W] = True
        z0 = standardized_corr_change(X, W, studentize=True)
        z1 = standardized_corr_change(X, W, studentize=True, scale_mask=mask)
        inside = ~mask
        inside[:W] = False; inside[-W:] = False
        robust = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
        # the studentisation fixes the ROBUST scale to one; with Laplace noise
        # and |rho| = 0.7 the tails are heavy, so the plain sd sits above it
        # by construction and is only bounded, not pinned
        r0, r1 = robust(z0[inside][:, 0]), robust(z1[inside][:, 0])
        self.assertLess(r0, 0.85, f"premise: single-pass scale must be inflated here (robust {r0:.2f})")
        self.assertTrue(0.85 < r1 < 1.15, f"refined within-regime robust scale {r1:.3f}")
        self.assertLess(z1[inside][:, 0].std(), 1.3)

    def test_refine_scale_recovers_changes_the_single_pass_misses(self):
        from detectors.detectors_correlation import detect_correlation_changes
        X, cps = self._short_regimes()
        W = 120
        single, _, _ = detect_correlation_changes(X, window=W, refractory_period=300,
                                                  statistic="max", gamma=3.0, studentize=True)
        refined, _, _ = detect_correlation_changes(X, window=W, refractory_period=300,
                                                   statistic="max", gamma=3.0, studentize=True,
                                                   refine_scale=True)
        rec = lambda det: np.mean([any(abs(d - c) <= W for d in det) for c in cps])
        self.assertGreaterEqual(rec(refined), rec(single))
        self.assertGreaterEqual(rec(refined), 0.8, f"refined recall {rec(refined):.2f}, found {refined}")
        fp = sum(all(abs(d - c) > W for c in cps) for d in refined)
        self.assertLessEqual(fp, 1, f"refined false positives {fp}: {refined}")


class AdjacencyConvention(unittest.TestCase):
    """One convention everywhere: adjacency[from, to], the generator's.
    lingam returns B[to, from] and CausalMorph reads parents from rows, i.e.
    also [to, from]; both must be converted at the boundary, never scored raw."""

    @staticmethod
    def _chain_sem(T=3000, seed=21):
        rng = np.random.default_rng(seed)
        e = rng.laplace(size=(T, 3))
        x1 = e[:, 0]; x2 = 0.9 * x1 + e[:, 1]; x3 = 0.9 * x2 + e[:, 2]
        X = np.c_[x1, x2, x3]
        names = ["V1", "V2", "V3"]
        A = _adj(names, [("V1", "V2"), ("V2", "V3")])      # [from, to]
        return X, names, A

    def test_cold_pipeline_output_is_scored_in_truth_convention(self):
        from full_pipeline import extract_causal_structures, compute_shd
        X, names, A = self._chain_sem()
        s = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=0)[0]
        m = compute_shd(s.adjacency_matrix, A)
        self.assertGreaterEqual(m["Recall"], 0.5, f"cold output scored against truth: {m}")
        self.assertNotEqual(s.adjacency_matrix.loc["V1", "V2"], 0.0, "V1 -> V2 must sit at [from, to]")
        self.assertEqual(s.adjacency_matrix.loc["V2", "V1"], 0.0)

    def test_baseline_learners_are_scored_in_truth_convention(self):
        from full_pipeline import extract_causal_structures_ablation, compute_shd
        X, names, A = self._chain_sem()
        res = extract_causal_structures_ablation(X, [], names, verbose=False, seed=1)
        for method in ("directlingam", "icalingam"):
            m = compute_shd(res[method][0].adjacency_matrix, A)
            self.assertGreaterEqual(m["Recall"], 0.5, f"{method} scored against truth: {m}")

    def test_bootstrap_output_is_in_truth_convention(self):
        from full_pipeline import extract_causal_structures, compute_shd
        X, names, A = self._chain_sem()
        s = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=5, bootstrap_threshold=0.5)[0]
        m = compute_shd(s.adjacency_matrix, A)
        self.assertGreaterEqual(m["Recall"], 0.5, f"bootstrap output scored against truth: {m}")
        self.assertEqual(s.edge_probs.shape, (3, 3))
        self.assertGreater(s.edge_probs[0, 1], s.edge_probs[1, 0], "edge_probs must be [from, to] too")

    def test_true_prior_in_truth_convention_is_not_reversed(self):
        # regression lock (passes today by a double reversal; must keep passing
        # for the right reason once the boundary conversion is in place)
        from full_pipeline import extract_causal_structures, compute_shd
        X, names, A = self._chain_sem()
        s = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=0,
                                      initial_adj=A, initial_order=[0, 1, 2])[0]
        m = compute_shd(s.adjacency_matrix, A)
        self.assertGreaterEqual(m["Recall"], 0.5, f"oracle-prior output: {m}")


class BenchmarkV2Generator(unittest.TestCase):
    """build_nonstationary_scenario exposes the mechanism and the noise family,
    so a grid that violates LiNGAM's assumptions can be generated: Gaussian
    noise (not identifiable by LiNGAM) and/or nonlinear mechanisms (which
    CausalMorph's Stage I linearises)."""

    @staticmethod
    def _gen(**kw):
        from full_pipeline import build_nonstationary_scenario
        out = build_nonstationary_scenario(p=4, n_regimes=2, min_samples=3000, max_samples=3000,
                                           base_pconn=0.5, change_pcts=[0.5], seed=3, noise_fraction=0.0, **kw)
        X, true_adjs = out[0], out[6]
        return X, true_adjs

    @staticmethod
    def _root(X, A):
        # a variable with no parents: its marginal is the structural noise itself
        A0 = A[0].values
        roots = [j for j in range(A0.shape[0]) if not (A0[:, j] != 0).any()]
        return X[:3000, roots[0]]

    def test_default_noise_is_laplace(self):
        from scipy.stats import kurtosis
        X, A = self._gen()
        self.assertGreater(kurtosis(self._root(X, A)), 1.5, "Laplace excess kurtosis is 3")

    def test_gaussian_noise_option(self):
        from scipy.stats import kurtosis
        X, A = self._gen(noise_dist="normal")
        self.assertLess(abs(kurtosis(self._root(X, A))), 0.6, "Gaussian excess kurtosis is 0")

    def test_nonlinear_mechanism_option(self):
        # A nonlinear mechanism leaves structure in the residual of a linear
        # fit. The function matters: with the generator's weight scale (unit
        # parents x 1.2-1.8) tanh / sigmoid / softplus are nearly linear in
        # range (degree-5 fit gains < 0.01 R^2 over linear); cube is genuinely
        # nonlinear (gain 0.07 at w=0.7, 0.16 at w=1.0). Odd functions are
        # uncorrelated with x^2 by symmetry, so the test looks at x^3.
        from full_pipeline import build_nonstationary_scenario
        for kw, expect_nonlinear in ((dict(), False),
                                     (dict(mechanism="nonlinear", nonlinearity=0.7, nonlin_func="cube"), True)):
            out = build_nonstationary_scenario(p=2, n_regimes=2, min_samples=4000, max_samples=4000,
                                               base_pconn=1.0, change_pcts=[0.5], seed=5, noise_fraction=0.0, **kw)
            X, A0 = out[0][:4000], out[6][0].values
            src, dst = (0, 1) if A0[0, 1] != 0 else (1, 0)
            x, y = X[:, src], X[:, dst]
            x = (x - x.mean()) / x.std()
            beta = np.polyfit(x, y, 1); resid = y - np.polyval(beta, x)
            curvature = abs(np.corrcoef(resid, x ** 3)[0, 1])
            if expect_nonlinear:
                self.assertGreater(curvature, 0.08, f"cube mechanism must leave curvature in the linear residual (got {curvature:.3f})")
            else:
                self.assertLess(curvature, 0.08, f"linear mechanism must not (got {curvature:.3f})")


# ── 2.6 generator schedule ───────────────────────────────────────────────────

class ChangeSchedule(unittest.TestCase):

    def test_v1_reproduces_current_floor_binning(self):
        from full_pipeline import schedule_changes
        lose, gain = schedule_changes(["a", "b"], ["c", "d"], n_regimes=4, version="v1")
        self.assertEqual(lose, {"a": 1, "b": 2})
        self.assertEqual(gain, {"c": 1, "d": 2})

    def test_v2_loads_every_transition_within_one(self):
        from full_pipeline import schedule_changes
        lose, gain = schedule_changes(["a", "b"], ["c", "d"], n_regimes=4, version="v2", seed=0)
        counts = {r: 0 for r in (1, 2, 3)}
        for r in list(lose.values()) + list(gain.values()):
            counts[r] += 1
        self.assertEqual(sum(counts.values()), 4)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_v2_extra_changes_are_not_always_on_the_first_transition(self):
        from full_pipeline import schedule_changes
        # 4 changes over 3 transitions: one transition gets 2. P(it is transition 1) should be ~1/3.
        hits = 0
        n_seeds = 300
        for s in range(n_seeds):
            lose, gain = schedule_changes(["a", "b"], ["c", "d"], n_regimes=4, version="v2", seed=s)
            fired = list(lose.values()) + list(gain.values())
            hits += fired.count(1) == 2
        frac = hits / n_seeds
        self.assertTrue(abs(frac - 1 / 3) < 0.12, f"transition 1 got the extra change in {frac:.2f} of seeds")


# ── determinism / ablation list ──────────────────────────────────────────────

class Determinism(unittest.TestCase):

    def test_ica_lingam_ablation_is_reproducible(self):
        from full_pipeline import extract_causal_structures_ablation
        rng = np.random.default_rng(4)
        T, p = 600, 4
        X = rng.laplace(size=(T, p))
        X[:, 1] += 0.8 * X[:, 0]
        X[:, 2] += 0.6 * X[:, 1]
        names = [f"V{i+1}" for i in range(p)]
        a = extract_causal_structures_ablation(X, [], names, verbose=False, seed=7)
        b = extract_causal_structures_ablation(X, [], names, verbose=False, seed=7)
        np.testing.assert_array_equal(a["icalingam"][0].adjacency_matrix.values,
                                      b["icalingam"][0].adjacency_matrix.values)

    def test_causalmorph_is_not_an_ablation_method(self):
        from full_pipeline import ABLATION_METHODS
        self.assertNotIn("causalmorph", ABLATION_METHODS)


# ── 2.4 / flags: anchor, oracle segmentation, no segmentation ────────────────

def _tiny_scenario(seed=11):
    from full_pipeline import build_nonstationary_scenario
    out = build_nonstationary_scenario(
        p=3, n_regimes=2, min_samples=500, max_samples=500,
        base_pconn=0.5, change_pcts=[0.5], seed=seed, noise_fraction=0.02,
    )
    X, true_cps, names, regime_sizes = out[0], out[1], out[2], out[3]
    true_adjs = out[6]
    return X, names, true_cps, true_adjs, regime_sizes


class RunOneFlags(unittest.TestCase):

    def _row(self, **kw):
        from run_experiments import run_loaded
        X, names, true_cps, true_adjs, sizes = _tiny_scenario()
        meta = dict(scenario_id="t", p=3, n_regimes=2, n_changes=1, samples_regime=500,
                    base_pconn=0.5, noise_fraction=0.02, seed=11)
        base = dict(window_overlap=0.0, detector_version="v1H", prior_mode="chain",
                    hybrid=False, bootstrap_n=0, bootstrap_threshold=0.6,
                    chain_prior_threshold=0.05, run_ablation=False)
        base.update(kw)
        return run_loaded(X, names, true_cps, true_adjs, sizes, min_samples=500,
                          meta=meta, **base)

    def test_default_window0_prior_is_cold(self):
        row = self._row()
        self.assertEqual(row["status"], "ok", row.get("error_msg"))
        self.assertEqual(row["w0_prior_source"], "cold")

    def test_oracle_anchor_flag_restores_ground_truth_prior(self):
        row = self._row(oracle_anchor=True)
        self.assertEqual(row["w0_prior_source"], "anchor")

    def test_oracle_segmentation_uses_true_change_points(self):
        row = self._row(oracle_segmentation=True)
        X, names, true_cps, *_ = _tiny_scenario()
        self.assertEqual(row["detected_cps"], ";".join(str(c) for c in true_cps))
        self.assertEqual(row["det_recall"], 1.0)

    def test_no_segmentation_gives_one_window(self):
        row = self._row(no_segmentation=True)
        self.assertEqual(row["n_structures"], 1)
        self.assertEqual(row["n_detected_cps"], 0)

    def test_detection_scored_before_extraction_merge(self):
        # a true cp that extraction must merge (n_min huge) still counts as detected
        row = self._row(oracle_segmentation=True, samples_per_node=100000)
        self.assertEqual(row["det_recall"], 1.0)
        self.assertEqual(row["n_structures"], 1)
        self.assertEqual(row["n_cps_merged"], 1)


class LegacyPostprocessing(unittest.TestCase):
    """--legacy_postprocessing reproduces the pre-2026-10-06 post-processing:
    change points bounding a window shorter than max(min_samples/2, 10 p^2)
    are deleted, and detection is scored AFTER that deletion. It exists so
    the old configuration can be re-scored under the corrected adjacency
    convention as an honest 'before'."""

    def test_legacy_deletes_and_scores_after_deletion(self):
        from run_experiments import run_loaded
        X, names, true_cps, true_adjs, sizes = _tiny_scenario()
        meta = dict(scenario_id="t", p=3, n_regimes=2, n_changes=1, samples_regime=500,
                    base_pconn=0.5, noise_fraction=0.02, seed=11)
        base = dict(window_overlap=0.0, detector_version="v1H", prior_mode="chain", hybrid=False,
                    bootstrap_n=0, bootstrap_threshold=0.6, chain_prior_threshold=0.05,
                    run_ablation=False, oracle_segmentation=True, samples_per_node=100000)
        # Every window is "too short" under both rules (samples_per_node drives the
        # new n_min, min_samples drives the legacy one): the new path keeps the
        # change point for detection and merges for extraction; legacy deletes it
        # before scoring, so recall drops to 0 and no change point is reported.
        new = run_loaded(X, names, true_cps, true_adjs, sizes, min_samples=100000, meta=meta, **base)
        old = run_loaded(X, names, true_cps, true_adjs, sizes, min_samples=100000, meta=meta,
                         legacy_postprocessing=True, **base)
        self.assertEqual(new["det_recall"], 1.0)
        self.assertEqual(new["n_cps_merged"], 1)
        self.assertEqual(old["det_recall"], 0.0)
        self.assertEqual(old["n_detected_cps"], 0)
        self.assertEqual(old["n_structures"], 1)

    def test_legacy_keeps_change_point_when_windows_are_long_enough(self):
        from run_experiments import run_loaded
        X, names, true_cps, true_adjs, sizes = _tiny_scenario()
        meta = dict(scenario_id="t", p=3, n_regimes=2, n_changes=1, samples_regime=500,
                    base_pconn=0.5, noise_fraction=0.02, seed=11)
        old = run_loaded(X, names, true_cps, true_adjs, sizes, min_samples=500, meta=meta,
                         window_overlap=0.0, detector_version="v1H", prior_mode="chain", hybrid=False,
                         bootstrap_n=0, bootstrap_threshold=0.6, chain_prior_threshold=0.05,
                         run_ablation=False, oracle_segmentation=True, legacy_postprocessing=True)
        # p=3: 10 p^2 = 90 < 500-sample regimes, nothing deleted
        self.assertEqual(old["det_recall"], 1.0)
        self.assertEqual(old["n_structures"], 2)


class TransitionF1(unittest.TestCase):
    """transition_f1: the edge-set F1 of the learned changes against the true
    changes, per true transition, averaged. A missed transition (both regimes
    aligned to one window) has an empty predicted change set and scores 0;
    transition_shd charges that case only |delta_true| / max_edges, which for
    sparse changes is below the estimation noise of two correct fits, so it
    rewards under-segmentation."""

    names = ["V1", "V2", "V3", "V4"]

    def _true(self):
        A0 = _adj(self.names, [("V1", "V2")])
        A1 = _adj(self.names, [("V1", "V2"), ("V2", "V3")])          # +V2->V3
        return [A0, A1], [500], 1000

    def test_missed_transition_scores_zero(self):
        from evaluation.metrics import transition_f1
        adjs, cps, T = self._true()
        one_window = [_struct(self.names, [("V1", "V2"), ("V2", "V3")], T)]   # spans both regimes
        self.assertEqual(transition_f1(one_window, adjs, cps, T, 4), 0.0)

    def test_perfectly_learned_transition_scores_one(self):
        from evaluation.metrics import transition_f1
        adjs, cps, T = self._true()
        s0 = _struct(self.names, [("V1", "V2")], 500, 0)
        s1 = _struct(self.names, [("V1", "V2"), ("V2", "V3")], 500, 1); s1.window_start, s1.window_end = 500, 1000
        self.assertEqual(transition_f1([s0, s1], adjs, cps, T, 4), 1.0)

    def test_noisy_but_detected_transition_scores_between(self):
        from evaluation.metrics import transition_f1
        adjs, cps, T = self._true()
        s0 = _struct(self.names, [("V1", "V2")], 500, 0)
        # learned the true change plus one spurious change (V3->V4 appears)
        s1 = _struct(self.names, [("V1", "V2"), ("V2", "V3"), ("V3", "V4")], 500, 1)
        s1.window_start, s1.window_end = 500, 1000
        f1 = transition_f1([s0, s1], adjs, cps, T, 4)
        self.assertAlmostEqual(f1, 2 / 3, places=6)     # precision 1/2, recall 1

    def test_no_true_change_and_no_predicted_change_is_perfect(self):
        from evaluation.metrics import transition_f1
        A = _adj(self.names, [("V1", "V2")])
        s0 = _struct(self.names, [("V1", "V2")], 500, 0)
        s1 = _struct(self.names, [("V1", "V2")], 500, 1); s1.window_start, s1.window_end = 500, 1000
        self.assertEqual(transition_f1([s0, s1], [A, A], [500], 1000, 4), 1.0)


class ColdPriorMode(unittest.TestCase):
    """prior_mode="cold": every window gets its own DirectLiNGAM prior for
    CausalMorph, nothing is inherited from the previous window. The chain is
    what costs the pipeline most (stationary: cold 0.88/0.85 vs chained
    pipeline 0.70/0.59), so this is the fair configuration for CausalMorph."""

    def test_every_window_is_cold_and_inherits_nothing(self):
        from full_pipeline import extract_causal_structures
        X, names, true_cps, *_ = _tiny_scenario()
        structs = extract_causal_structures(X, true_cps, names, verbose=False, bootstrap_n=0, prior_mode="cold")
        self.assertEqual(len(structs), 2)
        self.assertTrue(all(s.prior_source == "cold" for s in structs), [s.prior_source for s in structs])
        self.assertTrue(all(not s.used_prior for s in structs))

    def test_cold_mode_matches_window0_cold_path_on_each_window(self):
        # window 1 under "cold" must equal what the window-0 cold path gives on
        # window 1's data alone — i.e. no dependence on window 0
        from full_pipeline import extract_causal_structures
        X, names, true_cps, *_ = _tiny_scenario()
        cold = extract_causal_structures(X, true_cps, names, verbose=False, bootstrap_n=0, prior_mode="cold",
                                         window_overlap=0.0)
        alone = extract_causal_structures(X[true_cps[0]:], [], names, verbose=False, bootstrap_n=0,
                                          window_overlap=0.0)
        np.testing.assert_array_equal(cold[1].adjacency_matrix.values, alone[0].adjacency_matrix.values)

    def test_cli_accepts_cold(self):
        import subprocess, sys
        out = subprocess.run([sys.executable, "run_experiments.py", "--help"], capture_output=True, text=True, cwd=ROOT).stdout
        self.assertIn("cold", out.split("--prior_mode")[1][:80])


class LearnerOption(unittest.TestCase):
    """learner="icalingam": ICA-LiNGAM fits the CausalMorph-transformed window
    (and provides the cold prior) instead of DirectLiNGAM."""

    def test_learner_is_recorded_and_reproducible(self):
        from full_pipeline import extract_causal_structures
        X, names, true_cps, *_ = _tiny_scenario()
        a = extract_causal_structures(X, true_cps, names, verbose=False, bootstrap_n=0, prior_mode="cold",
                                      window_overlap=0.0, learner="icalingam", seed=3)
        b = extract_causal_structures(X, true_cps, names, verbose=False, bootstrap_n=0, prior_mode="cold",
                                      window_overlap=0.0, learner="icalingam", seed=3)
        self.assertTrue(all(s.learner == "icalingam" for s in a))
        np.testing.assert_array_equal(a[0].adjacency_matrix.values, b[0].adjacency_matrix.values)

    def test_learner_changes_the_fit(self):
        from full_pipeline import extract_causal_structures
        X, names, A = AdjacencyConvention._chain_sem()
        d = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=0, prior_mode="cold")
        i = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=0, prior_mode="cold", learner="icalingam", seed=1)
        self.assertEqual(d[0].learner, "directlingam")
        # both must still recover the chain in the truth convention
        from full_pipeline import compute_shd
        self.assertGreaterEqual(compute_shd(i[0].adjacency_matrix, A)["Recall"], 0.5)


# ── 2.5 soft chain prior ─────────────────────────────────────────────────────

class SoftChainPrior(unittest.TestCase):

    def test_bootstrap_edge_probs_are_stored(self):
        from full_pipeline import extract_causal_structures
        X, names, true_cps, *_ = _tiny_scenario()
        structs = extract_causal_structures(X, true_cps, names, verbose=False,
                                            bootstrap_n=3, bootstrap_threshold=0.5)
        self.assertIsNotNone(structs[0].edge_probs)
        self.assertEqual(structs[0].edge_probs.shape, (3, 3))

    def test_soft_prior_is_convex_combination(self):
        from full_pipeline import extract_causal_structures
        X, names, true_cps, *_ = _tiny_scenario()
        lam, rho0 = 0.7, 0.2
        structs = extract_causal_structures(X, true_cps, names, verbose=False,
                                            bootstrap_n=3, bootstrap_threshold=0.5,
                                            chain_soft_lambda=lam, prior_density=rho0)
        q_prev = structs[0].edge_probs
        expected = lam * q_prev + (1 - lam) * rho0
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(structs[1].prior_edge_probs, expected, atol=1e-9)


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

    def test_v2_dispatch_orientation_on_gaussian_chain(self):
        # Gaussian noise opens the gate, so a reversed prior (the [from, to]
        # truth handed over without _to_causalmorph) would transform the root.
        from full_pipeline import _apply_transform
        rng = np.random.default_rng(21)
        e = rng.normal(size=(3000, 3))
        x1 = e[:, 0]; x2 = 0.9 * x1 + e[:, 1]; x3 = 0.9 * x2 + e[:, 2]
        names = ["V1", "V2", "V3"]
        df = pd.DataFrame(np.c_[x1, x2, x3], columns=names)
        A = _adj(names, [("V1", "V2"), ("V2", "V3")])
        v2 = _apply_transform("v2", df, [0, 1, 2], A, 0)
        lite = _apply_transform("lite", df, [0, 1, 2], A, 0)
        np.testing.assert_array_equal(v2["V1"].values, df["V1"].values)
        self.assertFalse(np.allclose(v2["V2"].values, df["V2"].values))
        self.assertFalse(np.allclose(v2["V3"].values, df["V3"].values))
        self.assertFalse(np.allclose(v2.values, lite.values))

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
        self.assertEqual(len(p), len(a))
        for s, t in zip(p, a):
            self.assertEqual((s.window_start, s.window_end), (t.window_start, t.window_end))
            np.testing.assert_array_equal(
                (np.abs(s.adjacency_matrix.values) > 0).astype(int),
                (np.abs(t.adjacency_matrix.values) > 0).astype(int))

    def test_bootstrap_accepts_transform(self):
        from full_pipeline import extract_causal_structures
        X, names, A = AdjacencyConvention._chain_sem()
        s = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=4,
                                      bootstrap_threshold=0.5, prior_mode="cold", transform="v2")[0]
        self.assertEqual(s.transform, "v2")


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
        self.assertEqual(row["n_transform_failed"], 0)

    def test_resume_refuses_a_different_transform(self):
        import tempfile
        from run_experiments import run_experiments
        with tempfile.TemporaryDirectory() as d:
            data, out = os.path.join(d, "data"), os.path.join(d, "out")
            os.makedirs(data); os.makedirs(out)
            pd.DataFrame({"scenario_id": ["s0", "s1"], "p": ["3", "3"]}).to_csv(
                os.path.join(data, "index.csv"), index=False)
            pd.DataFrame({"scenario_id": ["s0"], "status": ["ok"], "w0_prior_source": ["cold"],
                          "learner": ["directlingam"], "transform": ["v1"]}).to_csv(
                os.path.join(out, "results.csv"), index=False)
            with self.assertRaises(SystemExit) as cm:
                run_experiments(dataset_dir=data, output_dir=out, resume=True, transform="v2")
            self.assertIn("transform", str(cm.exception.code))

    def test_resume_refuses_missing_learner_columns(self):
        # a file started without dagma would silently drop dagma_* columns on append (header lock)
        import tempfile
        from run_experiments import run_experiments
        with tempfile.TemporaryDirectory() as d:
            data, out = os.path.join(d, "data"), os.path.join(d, "out")
            os.makedirs(data); os.makedirs(out)
            pd.DataFrame({"scenario_id": ["s0", "s1"], "p": ["3", "3"]}).to_csv(
                os.path.join(data, "index.csv"), index=False)
            pd.DataFrame({"scenario_id": ["s0"], "status": ["ok"], "w0_prior_source": ["cold"],
                          "learner": ["icalingam"], "transform": ["v2cap"],
                          "directlingam_mean_f1": [0.9], "icalingam_mean_f1": [0.9]}).to_csv(
                os.path.join(out, "results.csv"), index=False)
            with self.assertRaises(SystemExit) as cm:
                run_experiments(dataset_dir=data, output_dir=out, resume=True, transform="v2cap",
                                learner="icalingam", ablation_methods=("directlingam", "icalingam", "dagma"))
            self.assertIn("dagma", str(cm.exception.code))

    def test_transform_failure_is_recorded(self):
        from unittest import mock
        import full_pipeline
        from full_pipeline import extract_causal_structures
        X, names, A = AdjacencyConvention._chain_sem()
        def boom(*a, **k):
            raise RuntimeError("synthetic transform failure")
        with mock.patch.object(full_pipeline, "causalmorph_v2", boom):
            s = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=0,
                                          prior_mode="cold", transform="v2")[0]
        self.assertTrue(s.transform_failed)
        ok = extract_causal_structures(X, [], names, verbose=False, bootstrap_n=0,
                                       prior_mode="cold", transform="v2")[0]
        self.assertFalse(ok.transform_failed)



class StageOneCapTransforms(unittest.TestCase):
    """v2cap / litecap: v2 / lite with the Stage I degree cap."""

    def test_dispatch(self):
        from full_pipeline import _apply_transform, TRANSFORMS
        from core.causalmorph_v2 import causalmorph_v2
        from full_pipeline import _to_causalmorph
        self.assertIn("v2cap", TRANSFORMS); self.assertIn("litecap", TRANSFORMS)
        X, names, A = AdjacencyConvention._chain_sem()
        df = pd.DataFrame(X[:120], columns=names)
        got = _apply_transform("v2cap", df, [0, 1, 2], A, 0)
        want = causalmorph_v2(df, [0, 1, 2], _to_causalmorph(A), stage2="rankmap", stage1_cap=True)
        np.testing.assert_array_equal(got.values, want.values)
        got = _apply_transform("litecap", df, [0, 1, 2], A, 0)
        want = causalmorph_v2(df, [0, 1, 2], _to_causalmorph(A), stage2="off", stage1_cap=True)
        np.testing.assert_array_equal(got.values, want.values)

    def test_cli_and_row(self):
        import subprocess
        out = subprocess.run([sys.executable, "run_experiments.py", "--help"], capture_output=True, text=True, cwd=ROOT).stdout
        self.assertIn("v2cap", out.split("--transform")[1][:120])
        from run_experiments import run_loaded
        X, names, true_cps, true_adjs, sizes = _tiny_scenario()
        meta = dict(scenario_id="t", p=3, n_regimes=2, n_changes=1, samples_regime=500,
                    base_pconn=0.5, noise_fraction=0.02, seed=11)
        row = run_loaded(X, names, true_cps, true_adjs, sizes, min_samples=500, meta=meta,
                         window_overlap=0.0, detector_version="v1L", corr_gamma=2.5, prior_mode="cold",
                         hybrid=False, bootstrap_n=0, bootstrap_threshold=0.6, chain_prior_threshold=0.05,
                         run_ablation=False, transform="v2cap", learner="icalingam")
        self.assertEqual(row["status"], "ok", row.get("error_msg"))
        self.assertEqual(row["transform"], "v2cap")
        self.assertEqual(row["n_transform_failed"], 0)


class SkeletonColumns(unittest.TestCase):
    def test_rows_carry_skeleton_f1(self):
        from run_experiments import run_loaded
        X, names, true_cps, true_adjs, sizes = _tiny_scenario()
        meta = dict(scenario_id="t", p=3, n_regimes=2, n_changes=1, samples_regime=500,
                    base_pconn=0.5, noise_fraction=0.02, seed=11)
        row = run_loaded(X, names, true_cps, true_adjs, sizes, min_samples=500, meta=meta,
                         window_overlap=0.0, detector_version="v1L", corr_gamma=2.5, prior_mode="cold",
                         hybrid=False, bootstrap_n=0, bootstrap_threshold=0.6, chain_prior_threshold=0.05,
                         run_ablation=True, transform="v2cap", learner="icalingam")
        self.assertEqual(row["status"], "ok", row.get("error_msg"))
        for k in ("mean_skeleton_f1", "icalingam_mean_skeleton_f1", "directlingam_mean_skeleton_f1"):
            self.assertTrue(0.0 <= row[k] <= 1.0, (k, row[k]))
        # orientation-free F1 can never be below the directed one on the same windows
        self.assertGreaterEqual(row["mean_skeleton_f1"] + 1e-9, row["mean_struct_f1"] - 0.0015)


class DagmaPerWindow(unittest.TestCase):
    """DAGMA (linear, Bello et al. 2022) as a per-window learner on the detected windows."""

    def test_default_methods_unchanged(self):
        from full_pipeline import extract_causal_structures_ablation, ABLATION_METHODS
        X, names, true_cps, *_ = _tiny_scenario()
        r = extract_causal_structures_ablation(X, true_cps, names, verbose=False, window_overlap=0.0)
        self.assertEqual(tuple(r), ABLATION_METHODS)

    def test_dagma_recovers_chain_in_truth_convention(self):
        from full_pipeline import extract_causal_structures_ablation, compute_shd
        X, names, A = AdjacencyConvention._chain_sem()
        r = extract_causal_structures_ablation(X, [], names, verbose=False, window_overlap=0.0, methods=("dagma",))
        self.assertEqual(list(r), ["dagma"])
        s = r["dagma"][0]
        self.assertEqual(compute_shd(s.adjacency_matrix, A)["F1"], 1.0)     # A is [from, to]
        self.assertEqual(sorted(s.causal_order), [0, 1, 2])

    def test_dagma_is_scale_equivariant(self):
        # the default lambda1 is calibrated for unit-scale data; the window is rescaled by one
        # global factor (relative variances and linear weights unchanged) before fitting
        from full_pipeline import extract_causal_structures_ablation, compute_shd
        X, names, A = AdjacencyConvention._chain_sem()
        a = extract_causal_structures_ablation(X * 0.05, [], names, verbose=False, window_overlap=0.0, methods=("dagma",))
        b = extract_causal_structures_ablation(X, [], names, verbose=False, window_overlap=0.0, methods=("dagma",))
        self.assertEqual(compute_shd(a["dagma"][0].adjacency_matrix, A)["F1"], 1.0)
        np.testing.assert_allclose(a["dagma"][0].adjacency_matrix.values, b["dagma"][0].adjacency_matrix.values, atol=1e-6)

    def test_unknown_method_rejected(self):
        from full_pipeline import extract_causal_structures_ablation
        X, names, A = AdjacencyConvention._chain_sem()
        with self.assertRaises(ValueError):
            extract_causal_structures_ablation(X, [], names, verbose=False, methods=("notears",))

    def test_row_and_cli(self):
        import subprocess
        out = subprocess.run([sys.executable, "run_experiments.py", "--help"], capture_output=True, text=True, cwd=ROOT).stdout
        self.assertIn("--ablation_methods", out)
        from run_experiments import run_loaded
        X, names, true_cps, true_adjs, sizes = _tiny_scenario()
        meta = dict(scenario_id="t", p=3, n_regimes=2, n_changes=1, samples_regime=500,
                    base_pconn=0.5, noise_fraction=0.02, seed=11)
        row = run_loaded(X, names, true_cps, true_adjs, sizes, min_samples=500, meta=meta,
                         window_overlap=0.0, detector_version="v1L", corr_gamma=2.5, prior_mode="cold",
                         hybrid=False, bootstrap_n=0, bootstrap_threshold=0.6, chain_prior_threshold=0.05,
                         run_ablation=True, transform="v2cap", learner="icalingam", ablation_methods=("dagma",))
        self.assertEqual(row["status"], "ok", row.get("error_msg"))
        for k in ("dagma_windowed_nshd_weighted", "dagma_mean_f1", "dagma_mean_skeleton_f1", "dagma_transition_f1"):
            self.assertIn(k, row)
        self.assertNotIn("icalingam_mean_f1", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
