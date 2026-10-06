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


if __name__ == "__main__":
    unittest.main(verbosity=2)
