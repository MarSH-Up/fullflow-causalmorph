"""
Full Pipeline: Non-Stationary Causal Discovery
===============================================
Integrates NSD_Wavelets non-stationarity detection with iterative CausalMorph
structure learning.

Flow:
  1. Generate a multi-regime non-stationary causal scenario
       (mirrors run_realistic_scenario from NSD_Wavelets/src/evaluation/run_experiment.py)
  2. Run wavelet-based multi-moment detector (v1-G default, v1-F fallback) on the full time series.
  3. For each detected change point, extract the data window and apply CausalMorph:
       - First window: cold DirectLiNGAM bootstraps the prior.
       - Subsequent windows: previous regime's (causal_order, adj_matrix) is used
         as the warm-start prior, enabling CausalMorph to improve iteratively.
  4. Store every extracted structure in a single list.
  5. Plot: detection diagnostics, time series, true/learned structures, adjacency
     heatmaps, SHD metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from lingam import DirectLiNGAM, ICALiNGAM

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(_HERE, "causalmorph"))

from scenarios.NonStationaryCausalScenarios import NonStationaryCausalScenario
from detectors.detectors_wavelets import (
    detect_nonstationarity_multimoment,
    detect_nonstationarity_v1G,
    DetectionResult,
    get_moment_weights,
)
from core.causalmorph_algorithm import causalMorph
from evaluation.metrics import evaluate_detection, simplify_transitions


# ── Data structure ────────────────────────────────────────────────────────────
@dataclass
class RegimeStructure:
    """Causal structure extracted from one detected regime window."""

    regime_idx: int
    window_start: int
    window_end: int
    n_samples: int
    causal_order: List[int]
    adjacency_matrix: pd.DataFrame
    used_prior: bool
    prior_source: str = "cold"   # "cold" | "anchor" | "chain"
    chosen: str = "n/a"          # "warm" | "cold" | "n/a" (hybrid mode)
    score_warm: float = float("nan")
    score_cold: float = float("nan")
    true_regime_idx: int = -1
    true_adjacency_matrix: Optional[pd.DataFrame] = None
    shd_metrics: Dict[str, Any] = field(default_factory=dict)


# ── SHD helpers ──────────────────────────────────────────────────────────────
def assign_true_regime(
    window_start: int, window_end: int, true_cps: List[int], T: int
) -> int:
    """Return the index of the true regime with the most overlap in this window."""
    boundaries = [0] + list(true_cps) + [T]
    best_regime, best_overlap = 0, 0
    for k, (rs, re) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        overlap = max(0, min(window_end, re) - max(window_start, rs))
        if overlap > best_overlap:
            best_overlap = overlap
            best_regime = k
    return best_regime


def compute_shd(
    pred_adj: pd.DataFrame,
    true_adj: pd.DataFrame,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """Compute SHD, normalized SHD, F1, Precision, Recall for edge sets."""
    gt = np.asarray(true_adj.values if isinstance(true_adj, pd.DataFrame) else true_adj).copy()
    est = np.abs(np.asarray(pred_adj.values if isinstance(pred_adj, pd.DataFrame) else pred_adj))
    gt_bin = (gt != 0)
    est_bin = (est > threshold)
    np.fill_diagonal(gt_bin, False)
    np.fill_diagonal(est_bin, False)

    tp = int(np.sum(gt_bin & est_bin))
    fp = int(np.sum(~gt_bin & est_bin))
    fn = int(np.sum(gt_bin & ~est_bin))
    shd = fp + fn  # missing + spurious edges
    d = gt_bin.shape[0]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "SHD": shd,
        "normalized_shd": round(shd / (d * (d - 1)), 3),
        "F1": round(f1, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
    }


def aggregate_structures_bayesian(
    structures: List[RegimeStructure],
    variable_names: List[str],
    threshold: float = 0.05,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    edge_threshold: float = 0.20,
) -> tuple:
    """
    Collapse all learned regime structures into one consensus graph via
    Beta-Bernoulli Bayesian aggregation.

    Model (per edge e_ij):
      - Prior     : Beta(alpha_prior, beta_prior)  — uniform by default
      - Votes     : each regime casts a binary vote (edge present/absent),
                    weighted by its fraction of total samples
      - Likelihood: effective count k_eff = n * Σ w_r * x_r,  n_eff = n regimes
      - Posterior  : Beta(alpha + k_eff,  beta + n_eff - k_eff)
      - Post. mean : (alpha + k_eff) / (alpha + beta + n_eff)
      - Include    : if posterior mean > edge_threshold (default 0.20)

    Returns
    -------
    consensus_adj : pd.DataFrame  binary {0,1} consensus adjacency
    edge_probs    : pd.DataFrame  posterior edge probabilities in [0,1]
    """
    p = len(variable_names)
    n = len(structures)
    if n == 0:
        empty = pd.DataFrame(np.zeros((p, p)), index=variable_names, columns=variable_names)
        return empty, empty

    # Regime weights ∝ window length
    lengths = np.array([s.n_samples for s in structures], dtype=float)
    weights = lengths / lengths.sum()

    # Weighted binary votes per edge
    weighted_votes = np.zeros((p, p))
    for s, w in zip(structures, weights):
        adj = s.adjacency_matrix.values if isinstance(s.adjacency_matrix, pd.DataFrame) else np.asarray(s.adjacency_matrix)
        bin_adj = (np.abs(adj) > threshold).astype(float)
        np.fill_diagonal(bin_adj, 0.0)
        weighted_votes += w * bin_adj  # value in [0,1] per edge

    # Posterior Beta parameters (scale to n regimes for meaningful concentration)
    n_eff = float(n)
    post_alpha = alpha_prior + n_eff * weighted_votes
    post_beta  = beta_prior  + n_eff * (1.0 - weighted_votes)
    post_mean  = post_alpha / (post_alpha + post_beta)
    np.fill_diagonal(post_mean, 0.0)

    consensus_bin = (post_mean > edge_threshold).astype(float)
    edge_probs    = pd.DataFrame(post_mean,    index=variable_names, columns=variable_names)
    consensus_adj = pd.DataFrame(consensus_bin, index=variable_names, columns=variable_names)
    return consensus_adj, edge_probs


# ── Hybrid model selection helper ────────────────────────────────────────────
_HYBRID_MARGIN = 0.02  # cold must beat warm by this margin to override


def _residual_indep_score(adj_values: np.ndarray, X_raw: np.ndarray) -> float:
    """
    Mean |Pearson corr| of LiNGAM residuals across all variable pairs.
    Lower = better: LiNGAM assumes mutually independent residuals, so a fit
    that leaves correlated residuals is worse regardless of which prior was used.

    Residuals: E = X - X @ B.T   (from x = Bx + e  =>  e = (I-B)x)
    Score:     s = mean_{i<j} |corr(E[:,i], E[:,j])|
    """
    E = X_raw - X_raw @ adj_values.T
    p = E.shape[1]
    total, n_pairs = 0.0, 0
    for i in range(p):
        for j in range(i + 1, p):
            r = np.corrcoef(E[:, i], E[:, j])[0, 1]
            if not np.isnan(r):
                total += abs(r)
                n_pairs += 1
    return total / n_pairs if n_pairs else 0.0


# ── Pink noise helper ─────────────────────────────────────────────────────────
def _pink_noise_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance pink (1/f) noise via FFT spectral shaping."""
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]          # avoid /0 at the DC bin
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    std = pink.std()
    return pink / std if std > 1e-12 else pink


# ── Step 1: Signal generation ─────────────────────────────────────────────────
def build_nonstationary_scenario(
    p: int,
    n_regimes: int,
    min_samples: int,
    max_samples: int,
    base_pconn: float,
    change_pcts: List[float],
    seed: int,
    noise_fraction: float = 0.08,
):
    """
    Build a multi-regime non-stationary causal time series.

    Structural noise: Laplace at low deviation (0.05–0.10) — non-Gaussian,
    which is required for LiNGAM identifiability, but kept small so the causal
    signal dominates (high SNR).

    Observation noise: pink (1/f) noise added on top, scaled to
    noise_fraction × std(Xⱼ) per variable.  Pink noise varies naturally with
    the local signal amplitude and is visually easier to read than white noise.

    Returns
    -------
    X, true_cps, variable_names, regime_sizes, scenario, scenario_gen, true_adjs, change_infos
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    scenario_gen = NonStationaryCausalScenario(
        p=p,
        mode="linear",
        seed=seed,
    )

    # Learning trajectory: same nodes, edges rewire toward a final structure ───
    # G_initial  — sparse starting graph (brain before learning)
    # G_target   — denser final graph    (brain after learning)
    # Intermediate regimes step monotonically from G_initial → G_target:
    #   edges only in G_initial are removed progressively
    #   edges only in G_target   are added   progressively
    #   edges in both are always present
    nodes = [f"V{i+1}" for i in range(p)]

    np.random.seed(seed)
    G_initial = nx.DiGraph()
    G_initial.add_nodes_from(nodes)
    for i in range(p):
        for j in range(i + 1, p):
            if np.random.rand() < base_pconn * 0.6:   # start sparse
                G_initial.add_edge(nodes[i], nodes[j])
    if G_initial.number_of_edges() == 0:
        G_initial.add_edge(nodes[0], nodes[1])

    np.random.seed(seed + 997)
    G_target = nx.DiGraph()
    G_target.add_nodes_from(nodes)
    for i in range(p):
        for j in range(i + 1, p):
            if np.random.rand() < base_pconn * 1.4:   # end denser
                G_target.add_edge(nodes[i], nodes[j])
    if G_target.number_of_edges() == 0:
        G_target.add_edge(nodes[0], nodes[-1])

    initial_edges = set(G_initial.edges())
    target_edges  = set(G_target.edges())
    stable_edges  = initial_edges & target_edges
    edges_to_lose = sorted(initial_edges - target_edges)   # removed over time
    edges_to_gain = sorted(target_edges  - initial_edges)  # added   over time

    # Assign each edge change to a specific regime transition so that
    # every consecutive pair of regimes always differs by at least one edge.
    # Schedule: change k fires at regime  1 + int(k * (n_regimes-1) / n_changes).
    # This spreads changes as evenly as possible, with the first always at regime 1.
    n_trans = max(n_regimes - 1, 1)

    def _schedule(edges):
        """edge -> regime index at which it becomes active (gained) or inactive (lost)."""
        n = len(edges)
        if n == 0:
            return {}
        return {e: 1 + int(k * n_trans / n) for k, e in enumerate(edges)}

    lose_at = _schedule(edges_to_lose)   # edge removed starting at this regime
    gain_at = _schedule(edges_to_gain)   # edge added   starting at this regime

    dags = []
    for i in range(n_regimes):
        current_edges = (
            stable_edges
            | {e for e, r in lose_at.items() if i < r}   # not yet removed
            | {e for e, r in gain_at.items() if i >= r}  # already gained
        )
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(current_edges)
        dags.append(G)

    # Regime configs: each regime gets its trajectory DAG + varying noise ──────
    regime_lengths = [
        int(rng.integers(min_samples, max_samples + 1)) for _ in range(n_regimes)
    ]
    change_infos = []
    regime_configs = []
    for i, dag in enumerate(dags):
        deviation       = 0.05 + rng.uniform(0.0, 0.05)   # low — Laplace keeps non-Gaussianity
        signal_strength = 1.2 + rng.uniform(0.0, 0.6)
        regime_configs.append({
            "fixed_graph":    dag,
            "deviation":      deviation,
            "signal_strength": signal_strength,
            "nsamples":       regime_lengths[i],
            "dist":           ["laplace"] * p,             # non-Gaussian → LiNGAM identifiable
            "regime_seed":    seed + i * 50,
        })
        if i > 0:
            e_prev = set(dags[i - 1].edges())
            e_curr = set(dag.edges())
            change_infos.append({
                "type":          "structural",
                "edges_added":   sorted(e_curr - e_prev),
                "edges_removed": sorted(e_prev - e_curr),
                "n_added":       len(e_curr - e_prev),
                "n_removed":     len(e_prev - e_curr),
            })

    scenario = scenario_gen.create_nonstationary_scenario(
        regime_configs=regime_configs,
        transition_type="abrupt",
    )

    combined: pd.DataFrame = scenario["combined_data"]
    X = combined.values.copy()
    variable_names = list(combined.columns)
    true_cps = scenario["change_points"]

    # Inject pink (1/f) noise scaled to each variable's signal amplitude
    if noise_fraction > 0:
        for j in range(p):
            sig_std = float(np.std(X[:, j])) or 1.0
            X[:, j] += noise_fraction * sig_std * _pink_noise_unit(len(X), rng)

    true_adjs = [
        (regime["adj_matrix"] != 0).astype(float) for regime in scenario["regimes"]
    ]

    return (
        X,
        true_cps,
        variable_names,
        regime_lengths,
        scenario,
        scenario_gen,
        true_adjs,
        change_infos,
    )


# ── Step 2: Change-point detection ───────────────────────────────────────────
# Mirrors run_realistic_scenario detector config
def detect_change_points(
    X: np.ndarray,
    baseline_end: int,
    min_regime_len: int,
    seed: int = 0,
    detector_version: str = "v1G",
) -> DetectionResult:
    """
    Run the wavelet-based multi-moment detector.

    detector_version :
      "v1F" — original v1-F with fixed step_delta_k=1.5
      "v1G" — (default) adaptive step validation + two-pass rescue

    Returns the full DetectionResult (used for the diagnostic plot).
    """
    baseline_idx = np.arange(0, baseline_end)
    refractory = min(150, min_regime_len // 4)

    if detector_version == "v1G":
        result = detect_nonstationarity_v1G(
            X,
            baseline_idx,
            min_scale=3.0,
            moments=[1, 2, 3, 4],
            moment_window=50,
            n_surrogates=100,
            alpha=0.40,
            k_scales_min=1,
            smooth_window=12,
            refractory_period=refractory,
            min_snr=0.3,
            k_channels_min=2,
            step_delta_k=1.2,
            adaptive_max_discount=0.6,
            step_pre_win=100,
            step_post_win=100,
            two_pass=True,
            seed=seed,
        )
    else:
        result = detect_nonstationarity_multimoment(
            X,
            baseline_idx,
            min_scale=3.0,
            moments=[1, 2, 3, 4],
            moment_window=50,
            n_surrogates=100,
            alpha=0.40,
            k_scales_min=1,
            smooth_window=12,
            refractory_period=refractory,
            min_snr=0.3,
            k_channels_min=2,
            step_delta_k=1.5,
            seed=seed,
        )
    return result


# ── Step 3: Iterative CausalMorph structure extraction ───────────────────────
def extract_causal_structures(
    X: np.ndarray,
    detected_change_points: List[int],
    variable_names: List[str],
    initial_adj: Optional[pd.DataFrame] = None,
    initial_order: Optional[List[int]] = None,
    window_overlap: float = 0.25,
    verbose: bool = True,
    prior_mode: str = "chain",
    hybrid: bool = False,
) -> List[RegimeStructure]:
    """
    Iteratively extract causal structures from every detected regime window.

    prior_mode :
      "chain"  — (default) window 0 uses initial_adj/initial_order; each
                 subsequent window uses the previous regime's extracted
                 (causal_order, adj_matrix) as the CausalMorph prior.
      "anchor" — every window uses the same initial_adj/initial_order prior.
                 Stops cascade but hurts consensus diversity (see 2026-04-29 doc).

    hybrid : bool
      If True, each window additionally runs a cold DirectLiNGAM on the raw
      data. Both fits are scored by mean |Pearson corr| of residuals (lower =
      better independence). Cold wins only if it beats warm by > _HYBRID_MARGIN.
      The chosen fit, both scores, and which was picked are stored in
      RegimeStructure (.chosen, .score_warm, .score_cold).

    window_overlap : fraction of the previous window to prepend to each window.
    """
    if prior_mode not in ("anchor", "chain"):
        raise ValueError(f"prior_mode must be 'anchor' or 'chain', got {prior_mode!r}")
    T = len(X)
    boundaries = [0] + list(detected_change_points) + [T]
    raw_windows = list(zip(boundaries[:-1], boundaries[1:]))

    windows = [raw_windows[0]]
    for i in range(1, len(raw_windows)):
        prev_start, prev_end = raw_windows[i - 1]
        prev_len = prev_end - prev_start
        overlap = int(prev_len * window_overlap)
        new_start = max(raw_windows[i][0] - overlap, prev_start)
        windows.append((new_start, raw_windows[i][1]))

    structures: List[RegimeStructure] = []
    prev_causal_order: Optional[List[int]] = None
    prev_adj_matrix: Optional[pd.DataFrame] = None
    # In anchor mode the prior used for window 0 is reused for every window.
    anchor_order: Optional[List[int]] = initial_order
    anchor_adj: Optional[pd.DataFrame] = initial_adj

    for regime_idx, (start, end) in enumerate(windows):
        n_samples = end - start

        if verbose:
            print(
                f"\n  [Regime {regime_idx}]  window [{start} : {end}]  "
                f"({n_samples} samples)"
            )

        window_df = pd.DataFrame(X[start:end], columns=variable_names)

        # ── Determine prior ──────────────────────────────────────────────────
        prior_source = "cold"
        if prev_causal_order is None:
            if initial_adj is not None and initial_order is not None:
                if verbose:
                    print("    Using ground-truth adj/order as initial prior.")
                causal_order_prior = initial_order
                adj_prior = initial_adj
                prior_source = "anchor"
            else:
                if verbose:
                    print("    Cold start — initialising prior with DirectLiNGAM...")
                model_cold = DirectLiNGAM()
                model_cold.fit(window_df)
                causal_order_prior = model_cold.causal_order_
                adj_cold = model_cold.adjacency_matrix_
                adj_prior = pd.DataFrame(
                    (np.abs(adj_cold) > 0.05).astype(float),
                    columns=variable_names,
                    index=variable_names,
                )
                # Cold result becomes the anchor for subsequent windows.
                anchor_order = causal_order_prior
                anchor_adj = adj_prior
            used_prior = False
        elif prior_mode == "anchor":
            causal_order_prior = anchor_order
            adj_prior = anchor_adj
            used_prior = True
            prior_source = "anchor"
            if verbose:
                order_names = [variable_names[i] for i in anchor_order]
                print(f"    Using anchored prior (regime 0):  order={' -> '.join(order_names)}")
        else:  # chain
            causal_order_prior = prev_causal_order
            adj_prior = prev_adj_matrix
            used_prior = True
            prior_source = "chain"
            if verbose:
                order_names = [variable_names[i] for i in prev_causal_order]
                print(f"    Using chained prior from regime {regime_idx - 1}:  order={' -> '.join(order_names)}")

        # ── Apply CausalMorph ────────────────────────────────────────────────
        if verbose:
            print("    Applying CausalMorph...")
        try:
            transformed = causalMorph(
                window_df,
                causal_order=causal_order_prior,
                adjacency_matrix=adj_prior,
                verbose=False,
            )
        except Exception as exc:
            if verbose:
                print(f"    CausalMorph raised {type(exc).__name__}: {exc}")
                print("    Falling back to raw window data.")
            transformed = window_df

        # ── Fit LiNGAM on transformed (warm) data ───────────────────────────
        if verbose:
            print("    Fitting DirectLiNGAM on transformed data (warm)...")
        model_warm = DirectLiNGAM()
        model_warm.fit(transformed)
        adj_warm = pd.DataFrame(
            model_warm.adjacency_matrix_,
            columns=variable_names,
            index=variable_names,
        )

        # ── Hybrid model selection ───────────────────────────────────────────
        X_raw = window_df.values
        if hybrid:
            if verbose:
                print("    Fitting DirectLiNGAM on raw data (cold)...")
            model_cold_h = DirectLiNGAM()
            model_cold_h.fit(window_df)
            adj_cold_h = pd.DataFrame(
                model_cold_h.adjacency_matrix_,
                columns=variable_names,
                index=variable_names,
            )
            s_warm = _residual_indep_score(adj_warm.values, X_raw)
            s_cold = _residual_indep_score(adj_cold_h.values, X_raw)
            if s_cold < s_warm - _HYBRID_MARGIN:
                chosen_fit = "cold"
                adj_final = adj_cold_h
                final_order = model_cold_h.causal_order_
            else:
                chosen_fit = "warm"
                adj_final = adj_warm
                final_order = model_warm.causal_order_
            if verbose:
                tag = f"→ chose {chosen_fit}  (s_warm={s_warm:.3f}, s_cold={s_cold:.3f})"
                print(f"    Hybrid selection: {tag}")
        else:
            s_warm = float("nan")
            s_cold = float("nan")
            chosen_fit = "n/a"
            adj_final = adj_warm
            final_order = model_warm.causal_order_

        struct = RegimeStructure(
            regime_idx=regime_idx,
            window_start=start,
            window_end=end,
            n_samples=n_samples,
            causal_order=final_order,
            adjacency_matrix=adj_final,
            used_prior=used_prior,
            prior_source=prior_source,
            chosen=chosen_fit,
            score_warm=s_warm,
            score_cold=s_cold,
        )
        structures.append(struct)

        prev_causal_order = final_order
        # Binarize before using as CausalMorph prior — CausalMorph uses != 0
        # to determine parents, so raw float weights would make every variable
        # a parent of every other (same approach as basic_usage.py: binary adj)
        prev_adj_matrix = (np.abs(adj_final.values) > 0.05).astype(float)
        prev_adj_matrix = pd.DataFrame(prev_adj_matrix, columns=variable_names, index=variable_names)

        if verbose:
            n_edges_raw = int((adj_final.values != 0).sum())
            n_edges_bin = int(prev_adj_matrix.values.sum())
            order_names = [variable_names[i] for i in final_order]
            print(f"    Learned order: {' -> '.join(order_names)}")
            print(f"    Edges found  : {n_edges_raw} raw, {n_edges_bin} after threshold (>0.05)")
            if regime_idx < len(windows) - 1:
                print(f"    => {n_edges_bin} edges + order above will be prior for regime {regime_idx + 1}")

    return structures


# ── Multi-algorithm ablation ─────────────────────────────────────────────────
ABLATION_METHODS = ("causalmorph", "directlingam", "icalingam")


def extract_causal_structures_ablation(
    X: np.ndarray,
    detected_change_points: List[int],
    variable_names: List[str],
    initial_adj: Optional[pd.DataFrame] = None,
    initial_order: Optional[List[int]] = None,
    window_overlap: float = 0.25,
    verbose: bool = True,
    prior_mode: str = "chain",
) -> Dict[str, List[RegimeStructure]]:
    """
    Run three causal discovery methods on each detected window and return
    separate structure lists for side-by-side comparison.

    Methods
    -------
    causalmorph   : CausalMorph transform → DirectLiNGAM  (existing pipeline)
    directlingam  : DirectLiNGAM on raw window data        (no transform)
    icalingam     : ICA-LiNGAM on raw window data          (alternative algo)

    Returns dict  {method_name: List[RegimeStructure]}
    """
    T = len(X)
    boundaries = [0] + list(detected_change_points) + [T]
    raw_windows = list(zip(boundaries[:-1], boundaries[1:]))

    windows = [raw_windows[0]]
    for i in range(1, len(raw_windows)):
        prev_start, prev_end = raw_windows[i - 1]
        prev_len = prev_end - prev_start
        overlap = int(prev_len * window_overlap)
        new_start = max(raw_windows[i][0] - overlap, prev_start)
        windows.append((new_start, raw_windows[i][1]))

    results: Dict[str, List[RegimeStructure]] = {m: [] for m in ABLATION_METHODS}

    # CausalMorph chain state
    prev_causal_order: Optional[List[int]] = None
    prev_adj_matrix: Optional[pd.DataFrame] = None
    anchor_order: Optional[List[int]] = initial_order
    anchor_adj: Optional[pd.DataFrame] = initial_adj

    for regime_idx, (start, end) in enumerate(windows):
        n_samples = end - start
        if verbose:
            print(f"\n  [Regime {regime_idx}]  window [{start} : {end}]  ({n_samples} samples)")

        window_df = pd.DataFrame(X[start:end], columns=variable_names)

        # ── 1. CausalMorph + DirectLiNGAM (existing logic) ──────────────
        prior_source = "cold"
        if prev_causal_order is None:
            if initial_adj is not None and initial_order is not None:
                causal_order_prior = initial_order
                adj_prior = initial_adj
                prior_source = "anchor"
            else:
                model_cold = DirectLiNGAM()
                model_cold.fit(window_df)
                causal_order_prior = model_cold.causal_order_
                adj_cold = model_cold.adjacency_matrix_
                adj_prior = pd.DataFrame(
                    (np.abs(adj_cold) > 0.05).astype(float),
                    columns=variable_names, index=variable_names,
                )
                anchor_order = causal_order_prior
                anchor_adj = adj_prior
            used_prior = False
        elif prior_mode == "anchor":
            causal_order_prior = anchor_order
            adj_prior = anchor_adj
            used_prior = True
            prior_source = "anchor"
        else:
            causal_order_prior = prev_causal_order
            adj_prior = prev_adj_matrix
            used_prior = True
            prior_source = "chain"

        try:
            transformed = causalMorph(
                window_df,
                causal_order=causal_order_prior,
                adjacency_matrix=adj_prior,
                verbose=False,
            )
        except Exception:
            transformed = window_df

        model_cm = DirectLiNGAM()
        model_cm.fit(transformed)
        adj_cm = pd.DataFrame(model_cm.adjacency_matrix_, columns=variable_names, index=variable_names)

        results["causalmorph"].append(RegimeStructure(
            regime_idx=regime_idx, window_start=start, window_end=end,
            n_samples=n_samples, causal_order=model_cm.causal_order_,
            adjacency_matrix=adj_cm, used_prior=used_prior, prior_source=prior_source,
        ))

        prev_causal_order = model_cm.causal_order_
        prev_adj_matrix = pd.DataFrame(
            (np.abs(adj_cm.values) > 0.05).astype(float),
            columns=variable_names, index=variable_names,
        )

        # ── 2. DirectLiNGAM only (no CausalMorph) ──────────────────────
        model_dl = DirectLiNGAM()
        model_dl.fit(window_df)
        adj_dl = pd.DataFrame(model_dl.adjacency_matrix_, columns=variable_names, index=variable_names)

        results["directlingam"].append(RegimeStructure(
            regime_idx=regime_idx, window_start=start, window_end=end,
            n_samples=n_samples, causal_order=model_dl.causal_order_,
            adjacency_matrix=adj_dl, used_prior=False, prior_source="cold",
        ))

        # ── 3. ICA-LiNGAM (no CausalMorph) ─────────────────────────────
        model_ica = ICALiNGAM(max_iter=3000)
        model_ica.fit(window_df)
        adj_ica = pd.DataFrame(model_ica.adjacency_matrix_, columns=variable_names, index=variable_names)

        results["icalingam"].append(RegimeStructure(
            regime_idx=regime_idx, window_start=start, window_end=end,
            n_samples=n_samples, causal_order=model_ica.causal_order_,
            adjacency_matrix=adj_ica, used_prior=False, prior_source="cold",
        ))

        if verbose:
            for method in ABLATION_METHODS:
                s = results[method][-1]
                n_edges = int((np.abs(s.adjacency_matrix.values) > 0.05).sum())
                order_names = [variable_names[i] for i in s.causal_order]
                print(f"    {method:<14}  edges={n_edges:>2}  order={' -> '.join(order_names)}")

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_detection_diagnostics(
    data: np.ndarray,
    result: DetectionResult,
    ground_truth: List[int],
    detected: List[int],
    title: str = "Multi-Moment Non-Stationarity Detection (v1-F)",
    figsize: tuple = (14, 16),
    tolerance: int = 0,
):
    """
    Multi-panel detection diagnostic — mirrors plot_multimoment_results in run_experiment.py.

    Panels:
      1. Original signals with change points
      2. Energy derivative dE(t) — key detection signal
      3. Signed energy E_signed(t) = E_pos - E_neg
      4-7. Per-moment energies (mean, var, skew, kurt) if available
      8. Forgetting severity trace

    Parameters
    ----------
    tolerance : int
        If > 0, shade ±tolerance window around each ground-truth change point.
    """
    T, N = data.shape

    moments_available = result.moments_used if result.moments_used else [2]
    n_moment_panels = len(moments_available)
    n_panels = 4 + n_moment_panels

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

    gt_style = dict(color="#2ecc71", linestyle="-", linewidth=3, alpha=0.9)
    onset_style = dict(color="#e74c3c", linestyle="--", linewidth=2.5, alpha=0.9)
    offset_style = dict(color="#3498db", linestyle=":", linewidth=2.5, alpha=0.9)
    det_style = dict(color="#e74c3c", linestyle="--", linewidth=2.5, alpha=0.9)

    moment_names = {1: "Mean", 2: "Variance", 3: "Skewness", 4: "Kurtosis"}
    moment_colors = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71", 4: "#9b59b6"}

    def _draw_gt_bands(ax):
        """Draw ground-truth vertical lines and optional tolerance shading."""
        for i, cp in enumerate(ground_truth):
            if tolerance > 0:
                ax.axvspan(cp - tolerance, cp + tolerance,
                           alpha=0.18, color='#2ecc71', zorder=0,
                           label=f'±{tolerance} window' if i == 0 else '')
            ax.axvline(cp, **gt_style, label='Ground Truth' if i == 0 else '', zorder=10)

    panel_idx = 0

    # Panel 1: Original signals ──────────────────────────────────────────────
    ax = axes[panel_idx]
    for ch in range(min(3, N)):
        ax.plot(data[:, ch], alpha=0.6, linewidth=0.8, label=f"V{ch+1}")
    _draw_gt_bands(ax)
    for i, cp in enumerate(result.onset_points):
        ax.axvline(cp, **onset_style, label="Onset" if i == 0 else "", zorder=11)
    for i, cp in enumerate(result.offset_points):
        ax.axvline(cp, **offset_style, label="Offset" if i == 0 else "", zorder=11)
    ax.set_ylabel("Signal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    panel_idx += 1

    # Panel 2: Energy derivative dE(t) ──────────────────────────────────────
    ax = axes[panel_idx]
    ax.fill_between(
        range(T),
        result.dE,
        0,
        where=(result.dE > 0),
        alpha=0.4,
        color="red",
        label="dE > 0 (onset)",
    )
    ax.fill_between(
        range(T),
        result.dE,
        0,
        where=(result.dE < 0),
        alpha=0.4,
        color="blue",
        label="dE < 0 (offset)",
    )
    ax.plot(result.dE, color="black", linewidth=0.8)
    ax.axhline(
        result.calibration.eps_on_pos,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"eps_pos={result.calibration.eps_on_pos:.1f}",
    )
    ax.axhline(
        -result.calibration.eps_on_neg,
        color="blue",
        linestyle=":",
        alpha=0.7,
        label=f"eps_neg={result.calibration.eps_on_neg:.1f}",
    )
    _draw_gt_bands(ax)
    for event in result.events:
        marker = "^" if event.event_type == "onset" else "v"
        color = "red" if event.event_type == "onset" else "blue"
        ax.plot(
            event.tau,
            result.dE[event.tau],
            marker,
            color=color,
            markersize=10,
            zorder=12,
        )
    ax.set_ylabel("dE (derivative)")
    ax.set_title("Energy Derivative dE(t) - Detection Signal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    panel_idx += 1

    # Panel 3: Signed energy E_signed ───────────────────────────────────────
    ax = axes[panel_idx]
    ax.fill_between(range(T), result.E_signed, alpha=0.4, color="steelblue")
    ax.plot(result.E_signed, color="navy", linewidth=0.8)
    ax.plot(result.E_pos, color="red", linewidth=0.5, alpha=0.5, label="E_pos")
    ax.plot(-result.E_neg, color="blue", linewidth=0.5, alpha=0.5, label="-E_neg")
    _draw_gt_bands(ax)
    for i, cp in enumerate(detected):
        ax.axvline(cp, **det_style, label="Detected" if i == 0 else "", zorder=11)
    ax.set_ylabel("E_signed")
    ax.set_title("Total Weighted Energy E_signed(t) = E_pos - E_neg")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    panel_idx += 1

    # Panels 4+: Per-moment energies ────────────────────────────────────────
    if result.E_by_moment:
        weights = result.moment_weights or get_moment_weights(moments_available)
        for m in moments_available:
            if m in result.E_by_moment:
                E_m_pos, E_m_neg = result.E_by_moment[m]
                E_m_signed = E_m_pos - E_m_neg
                weight = weights.get(m, 1.0)

                ax = axes[panel_idx]
                color = moment_colors.get(m, "gray")
                ax.fill_between(range(T), E_m_signed, alpha=0.3, color=color)
                ax.plot(
                    E_m_signed,
                    color=color,
                    linewidth=0.8,
                    label=f'{moment_names.get(m, f"M{m}")} (w=1/{m}!={weight:.3f})',
                )
                _draw_gt_bands(ax)
                ax.set_ylabel(f"E_{moment_names.get(m, f'M{m}')}")
                ax.set_title(
                    f"{moment_names.get(m, f'Moment {m}')} Energy (weight = 1/{m}! = {weight:.3f})"
                )
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.2)
                panel_idx += 1

    # Final panel: Forgetting severity ──────────────────────────────────────
    ax = axes[panel_idx]
    ax.plot(result.severity, color="purple", linewidth=1.2)
    ax.fill_between(range(T), result.severity, alpha=0.2, color="purple")
    _draw_gt_bands(ax)
    for i, cp in enumerate(detected):
        ax.axvline(cp, **det_style, label="Detected" if i == 0 else "", zorder=11)
    ax.set_ylabel("Severity")
    ax.set_xlabel("Sample Index")
    ax.set_title("Forgetting Severity (Leaky Integrator)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show(block=False)


def plot_timeseries(
    X: np.ndarray,
    variable_names: List[str],
    true_cps: List[int],
    detected_cps: List[int],
):
    """Time series for each variable with true (red) and detected (green) change points."""
    T, p = X.shape
    fig, axes = plt.subplots(p, 1, figsize=(16, 2.8 * p), sharex=True)
    if p == 1:
        axes = [axes]

    boundaries = [0] + true_cps + [T]
    regime_colors = plt.cm.Set3(np.linspace(0, 1, len(true_cps) + 1))

    for i, (ax, name) in enumerate(zip(axes, variable_names)):
        for r, (rs, re) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            ax.axvspan(rs, re, alpha=0.12, color=regime_colors[r], zorder=0)

        ax.plot(X[:, i], linewidth=0.8, color="#1A3A5C", alpha=0.85)

        for j, cp in enumerate(true_cps):
            ax.axvline(
                cp,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="True CP" if (i == 0 and j == 0) else None,
            )

        for j, cp in enumerate(detected_cps):
            ax.axvline(
                cp,
                color="#27AE60",
                linestyle=":",
                linewidth=2.0,
                alpha=0.9,
                label="Detected CP" if (i == 0 and j == 0) else None,
            )

        ax.set_ylabel(name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle="--")

        if i == 0 and (true_cps or detected_cps):
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Sample Index", fontsize=12)
    fig.suptitle(
        "Non-Stationary Time Series  —  True (red dashed) vs Detected (green dotted) Change Points",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show(block=False)


def _draw_graph_on_ax(
    ax: plt.Axes,
    adj: np.ndarray,
    variable_names: List[str],
    pos: dict,
    node_color: str,
    show_edge_labels: bool = True,
):
    """Build a DiGraph from adj and draw it on ax using shared pos."""
    G = nx.DiGraph()
    G.add_nodes_from(variable_names)
    for i, src in enumerate(variable_names):
        for j, dst in enumerate(variable_names):
            if abs(adj[i, j]) > 0.01:
                G.add_edge(src, dst, weight=float(adj[i, j]))
    nx.draw_networkx(
        G, pos, ax=ax,
        with_labels=True,
        node_color=node_color,
        node_size=1400,
        edge_color="#2C3E50",
        arrowsize=18,
        font_size=10,
        font_weight="bold",
    )
    if show_edge_labels:
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)
    ax.axis("off")
    return G


def plot_structures_comparison(
    scenario: dict,
    structures: List[RegimeStructure],
    variable_names: List[str],
):
    """
    True (top row, blue) vs Learned (bottom row, green) causal structures.

    Columns = max(n_true_regimes, n_detected_windows).
    Consistent node layout across both rows for easy comparison.
    """
    true_regimes = scenario["regimes"]
    n_true = len(true_regimes)
    n_learned = len(structures)
    n_cols = max(n_true, n_learned)

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Shared node layout — circular is edge-independent, so positions are
    # identical across all panels (true and learned) regardless of graph structure.
    ref_G = nx.DiGraph()
    ref_G.add_nodes_from(variable_names)
    shared_pos = nx.circular_layout(ref_G)

    # ── Row 0: True structures ────────────────────────────────────────────────
    for col in range(n_cols):
        ax = axes[0, col]
        if col < n_true:
            G = true_regimes[col]["graph"]
            adj = nx.to_numpy_array(G, nodelist=variable_names)
            _draw_graph_on_ax(ax, adj, variable_names, shared_pos,
                              node_color="#AED6F1", show_edge_labels=False)
            ax.set_title(
                f"True Regime {col}  ({G.number_of_edges()} edges)",
                fontsize=11, fontweight="bold", color="#1A3A5C",
            )
        else:
            ax.axis("off")

    # ── Row 1: Learned structures ─────────────────────────────────────────────
    for col in range(n_cols):
        ax = axes[1, col]
        if col < n_learned:
            s = structures[col]
            G = _draw_graph_on_ax(ax, s.adjacency_matrix.values, variable_names,
                                  shared_pos, node_color="#A9DFBF", show_edge_labels=True)
            prior_tag = f"prior r{s.regime_idx - 1}" if s.used_prior else "cold"
            shd_tag = (
                f"  nSHD={s.shd_metrics['normalized_shd']:.3f}"
                if s.shd_metrics else ""
            )
            ax.set_title(
                f"Learned Regime {s.regime_idx}  [{s.window_start}:{s.window_end}]\n"
                f"{G.number_of_edges()} edges  ({prior_tag}){shd_tag}",
                fontsize=10, fontweight="bold", color="#1A5226",
            )
        else:
            ax.axis("off")

    fig.suptitle(
        "Causal Structures — True (top, blue)  vs  Learned (bottom, green)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show(block=False)


def plot_adjacency_heatmaps(
    structures: List[RegimeStructure], variable_names: List[str]
):
    """Adjacency matrix heatmaps for all extracted structures."""
    n = len(structures)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.8))
    if n == 1:
        axes = [axes]

    all_vals = np.concatenate([s.adjacency_matrix.values.ravel() for s in structures])
    vmax = float(np.abs(all_vals).max()) or 1.0

    for s, ax in zip(structures, axes):
        adj = s.adjacency_matrix.values
        im = ax.imshow(adj, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(variable_names)))
        ax.set_yticks(range(len(variable_names)))
        ax.set_xticklabels(variable_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(variable_names, fontsize=9)
        ax.set_xlabel("Effect", fontsize=9)
        ax.set_ylabel("Cause", fontsize=9)
        ax.set_title(
            f"Regime {s.regime_idx}  ({s.n_samples} samples)",
            fontsize=10,
            fontweight="bold",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(len(variable_names)):
            for j in range(len(variable_names)):
                val = adj[i, j]
                if abs(val) > 0.01:
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

    fig.suptitle(
        "Adjacency Matrices — Learned Structures", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show(block=False)


def plot_shd_metrics(structures: List[RegimeStructure]):
    """Bar chart comparing SHD, F1, Precision and Recall across all detected windows."""
    structs_with_metrics = [s for s in structures if s.shd_metrics]
    if not structs_with_metrics:
        return

    labels = [
        f"R{s.regime_idx}\n(true {s.true_regime_idx})" for s in structs_with_metrics
    ]
    shd_vals = [s.shd_metrics["SHD"] for s in structs_with_metrics]
    norm_shd_vals = [s.shd_metrics["normalized_shd"] for s in structs_with_metrics]
    f1_vals = [s.shd_metrics["F1"] for s in structs_with_metrics]
    prec_vals = [s.shd_metrics["Precision"] for s in structs_with_metrics]
    rec_vals = [s.shd_metrics["Recall"] for s in structs_with_metrics]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(
        x, shd_vals, color="#E74C3C", alpha=0.8, edgecolor="white", linewidth=0.8
    )
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("SHD (lower is better)", fontsize=11)
    ax.set_title(
        "Structural Hamming Distance per Regime", fontsize=12, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(shd_vals) * 1.3 + 1)

    ax = axes[1]
    width = 0.2
    ax.bar(
        x - 1.5 * width,
        norm_shd_vals,
        width,
        label="norm SHD",
        color="#E74C3C",
        alpha=0.8,
    )
    ax.bar(x - 0.5 * width, f1_vals, width, label="F1", color="#2ECC71", alpha=0.8)
    ax.bar(
        x + 0.5 * width, prec_vals, width, label="Precision", color="#3498DB", alpha=0.8
    )
    ax.bar(x + 1.5 * width, rec_vals, width, label="Recall", color="#9B59B6", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "norm SHD  /  F1  /  Precision  /  Recall", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "SHD Metrics — Learned vs True Structures per Detected Regime",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show(block=False)


def plot_consensus_structure(
    consensus_adj: pd.DataFrame,
    edge_probs: pd.DataFrame,
    variable_names: List[str],
    n_regimes: int,
    true_adjs: Optional[List[pd.DataFrame]] = None,
):
    """
    Three-panel figure:
      Left   — Bayesian consensus (learned, edges coloured by posterior P)
      Middle — Global ground truth (final target structure, G_target)
      Right  — Posterior heatmap; GT edges marked with green border
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    pos = nx.circular_layout(nx.DiGraph([(v, v) for v in variable_names]))  # fixed layout

    # ── helpers ──────────────────────────────────────────────────────────────
    def _draw_graph(ax, adj_df, color, title, *, edge_probs_df=None):
        G = nx.DiGraph()
        G.add_nodes_from(variable_names)
        for src in variable_names:
            for dst in variable_names:
                if adj_df.loc[src, dst] > 0:
                    p = float(edge_probs_df.loc[src, dst]) if edge_probs_df is not None else 1.0
                    G.add_edge(src, dst, prob=p)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900, node_color=color,
                               edgecolors="#555", linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
        if G.number_of_edges() > 0:
            probs = [G[u][v]["prob"] for u, v in G.edges()]
            cmap  = plt.cm.Blues if edge_probs_df is not None else None
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=probs if cmap else "#555",
                edge_cmap=cmap, edge_vmin=0.0, edge_vmax=1.0,
                width=2.5, connectionstyle="arc3,rad=0.08",
                arrows=True, arrowsize=20,
            )
            lbl = {(u, v): f"{G[u][v]['prob']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, lbl, ax=ax, font_size=7, label_pos=0.35)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    # ── Left: Bayesian consensus (learned) ───────────────────────────────────
    _draw_graph(axes[0], consensus_adj, "#85C1E9",
                f"Bayesian Consensus (learned)\n{n_regimes} regimes · threshold=0.20",
                edge_probs_df=edge_probs)

    # ── Middle: global ground truth (final target = last regime) ─────────────
    if true_adjs is not None and len(true_adjs) > 0:
        gt_values = true_adjs[-1]
        if isinstance(gt_values, np.ndarray):
            gt_df = pd.DataFrame(gt_values, index=variable_names, columns=variable_names)
        else:
            gt_df = gt_values.copy()
            gt_df.index   = variable_names
            gt_df.columns = variable_names
        gt_df = (gt_df != 0).astype(float)

        # SHD between consensus and ground truth
        shd_metrics = compute_shd(consensus_adj, gt_df)
        _draw_graph(axes[1], gt_df, "#A9DFBF",
                    f"Global Ground Truth (target)\n"
                    f"nSHD={shd_metrics['normalized_shd']:.3f}  "
                    f"F1={shd_metrics['F1']:.2f}")
    else:
        axes[1].set_title("Ground truth not available", fontsize=11)
        axes[1].axis("off")
        gt_df = None

    # ── Right: posterior heatmap with GT edges marked ─────────────────────────
    ax = axes[2]
    prob_matrix = edge_probs.values.copy()
    np.fill_diagonal(prob_matrix, np.nan)
    im = ax.imshow(prob_matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(variable_names)))
    ax.set_yticks(range(len(variable_names)))
    ax.set_xticklabels(variable_names, fontsize=9)
    ax.set_yticklabels(variable_names, fontsize=9)
    ax.set_xlabel("To (child)", fontsize=10)
    ax.set_ylabel("From (parent)", fontsize=10)
    plt.colorbar(im, ax=ax, label="Posterior P(edge exists)")
    for i, src in enumerate(variable_names):
        for j, dst in enumerate(variable_names):
            if i != j:
                val = prob_matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.6 else "black")
                # Green border = GT edge present
                if gt_df is not None and gt_df.loc[src, dst] > 0:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="#27AE60", linewidth=2.5,
                    ))
    ax.set_title(
        "Posterior P(edge)\n[green border = ground truth edge]",
        fontsize=11, fontweight="bold",
    )

    fig.suptitle("Bayesian Structure Aggregation vs Global Ground Truth",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)


# ── Summary print ─────────────────────────────────────────────────────────────
def print_summary(
    structures: List[RegimeStructure],
    variable_names: List[str],
    true_cps: List[int],
    detected_cps: List[int],
):
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  True change points      : {true_cps}")
    print(f"  Detected change points  : {detected_cps}")
    n_matched = sum(any(abs(t - d) <= 200 for d in detected_cps) for t in true_cps)
    print(f"  Matched (within +/-200) : {n_matched} / {len(true_cps)}")
    print(f"\n  Extracted structures    : {len(structures)}")
    print()

    for s in structures:
        if not s.used_prior:
            prior_tag = "cold start"
        elif s.prior_source == "anchor":
            prior_tag = "anchored to regime-0 prior"
        elif s.prior_source == "chain":
            prior_tag = f"prior from regime {s.regime_idx - 1}"
        else:
            prior_tag = s.prior_source
        print(
            f"  -- Regime {s.regime_idx}  "
            f"[{s.window_start} : {s.window_end}]  "
            f"n={s.n_samples}  ({prior_tag})"
        )

        order_names = [variable_names[i] for i in s.causal_order]
        print(f"     Causal order  : {' -> '.join(order_names)}")

        adj = s.adjacency_matrix
        edges = []
        for src in variable_names:
            for dst in variable_names:
                w = adj.loc[src, dst]
                if abs(w) > 0.01:
                    edges.append(f"{src}->{dst} ({w:+.3f})")
        print(f"     Edges         : {', '.join(edges) if edges else '(none)'}")

        if s.shd_metrics:
            print(f"     vs true regime {s.true_regime_idx}:  "
                  f"norm SHD = {s.shd_metrics['normalized_shd']:.3f}")
        print()

    # ── Mean norm SHD across all regimes ──────────────────────────────────────
    iter_metrics = [s.shd_metrics for s in structures if s.shd_metrics]
    if iter_metrics:
        mean_nshd = np.mean([m["normalized_shd"] for m in iter_metrics])
        print(f"  Mean norm SHD : {mean_nshd:.3f}")

    print("=" * 70)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_full_pipeline(
    p: int = 5,
    n_regimes: int = 5,
    min_samples: int = 600,
    max_samples: int = 800,
    base_pconn: float = 0.35,
    change_pcts: Optional[List[float]] = None,
    window_overlap: float = 0.25,
    noise_fraction: float = 0.08,
    seed: int = 42,
    verbose: bool = True,
    show_plots: bool = True,
    use_true_change_points: bool = False,
    prior_mode: str = "chain",
    hybrid: bool = False,
    ablation: bool = False,
    detector_version: str = "v1G",
) -> dict:
    """
    End-to-end pipeline.

    Defaults match run_realistic_scenario from NSD_Wavelets/src/evaluation/run_experiment.py:
      p=5, 5 regimes (4 changes), 600-800 samples/regime,
      base_pconn=0.35, change_pcts=[0, 30, 25, 35, 30].

    Parameters
    ----------
    use_true_change_points : bool, optional
        If True, skip the wavelet detector and pass the ground-truth change
        points (generated in step 1) directly to CausalMorph.  Useful for
        testing the causal discovery step in isolation.
    ablation : bool, optional
        If True, also run DirectLiNGAM-only and ICA-LiNGAM on each window
        for side-by-side comparison. Results stored under 'ablation' key.
    detector_version : str, optional
        "v1G" (default) for adaptive step + two-pass, "v1F" for original.

    Returns
    -------
    dict with keys:
        'structures', 'detected_change_points', 'true_change_points',
        'X', 'variable_names', 'detection_result'
        If ablation=True, also 'ablation': {method: List[RegimeStructure]}
    """
    if change_pcts is None:
        change_pcts = [0, 30, 25, 35, 30]

    n_changes = n_regimes - 1

    print("=" * 70)
    print("Full Pipeline: Non-Stationary Causal Discovery")
    print("=" * 70)

    # ── 1. Generate scenario ─────────────────────────────────────────────────
    print(
        f"\n[1] Generating scenario  "
        f"(p={p}, {n_regimes} regimes / {n_changes} changes, "
        f"[{min_samples}-{max_samples}] samples/regime, "
        f"noise_fraction={noise_fraction})"
    )
    (
        X,
        true_cps,
        variable_names,
        regime_sizes,
        scenario,
        scenario_gen,
        true_adjs,
        change_infos,
    ) = build_nonstationary_scenario(
        p=p,
        n_regimes=n_regimes,
        min_samples=min_samples,
        max_samples=max_samples,
        base_pconn=base_pconn,
        change_pcts=change_pcts,
        seed=seed,
        noise_fraction=noise_fraction,
    )
    print(f"    Total samples      : {len(X)}")
    print(f"    Regime sizes       : {regime_sizes}")
    print(f"    True change points : {true_cps}")
    edge_counts = [scenario["regimes"][i]["graph"].number_of_edges() for i in range(n_regimes)]
    print(f"    Edges per regime   : {edge_counts}  (trajectory: {edge_counts[0]} → {edge_counts[-1]})")
    for k, info in enumerate(change_infos):
        added   = ", ".join(f"{u}→{v}" for u, v in info["edges_added"])   or "—"
        removed = ", ".join(f"{u}→{v}" for u, v in info["edges_removed"]) or "—"
        print(
            f"    Regime {k} → {k+1}: "
            f"+{info['n_added']} edge(s) [{added}]  "
            f"-{info['n_removed']} edge(s) [{removed}]"
        )

    # ── 2. Detect change points ──────────────────────────────────────────────
    skip_detection = use_true_change_points
    if skip_detection:
        print(f"\n[2] Skipping detector — using ground-truth change points: {true_cps}")
        detected_cps = true_cps
        det_result = None
    else:
        print(f"\n[2] Running wavelet-based multi-moment detector ({detector_version})...")
        baseline_end = min(150, regime_sizes[0] // 4)
        min_regime_len = min(regime_sizes)
        refractory = min(150, min_regime_len // 4)
        det_result = detect_change_points(
            X,
            baseline_end=baseline_end,
            min_regime_len=min_regime_len,
            seed=seed,
            detector_version=detector_version,
        )
        onset_set = set(det_result.onset_points)
        first_onset = min(onset_set) if onset_set else float("inf")
        kept_offsets = [
            cp for cp in det_result.offset_points
            if (cp < first_onset)
            or all(abs(cp - on) > refractory * 2 for on in onset_set)
        ]
        raw_detected_cps = sorted(onset_set | set(kept_offsets))
        print(f"    Raw detected           : {raw_detected_cps}")
        print(f"    Onset points           : {det_result.onset_points}")
        print(f"    Offset points          : {det_result.offset_points}")
        print(f"    Kept offsets           : {kept_offsets}")
        print(f"    Events                 : {len(det_result.events)}")
        for ev in det_result.events:
            ch_info = ""
            if ev.n_active_channels is not None:
                ch_info = (
                    f", ch_active={ev.n_active_channels}/{X.shape[1]}, "
                    f"conc={ev.concentration_ratio:.2f}"
                )
            print(f"      type={ev.event_type}, tau={ev.tau}, peak_dE={ev.peak_dE:.2f}{ch_info}")

        # Merge nearby detections — same as run_realistic_scenario
        detected_cps = simplify_transitions(raw_detected_cps, min_distance=160)
        print(f"    After simplify (min_dist=160): {detected_cps}")

        # Enforce minimum window size iteratively:
        #   - tiny window at the start/end  → drop the adjacent edge CP
        #   - tiny window in the interior   → merge its two flanking CPs into midpoint
        # Repeat until every window is ≥ min_window or no CPs remain.
        # Floor is scaled by p so larger graphs require proportionally more samples
        # (DirectLiNGAM needs ~10p² observations for stable estimation).
        min_window = max(min_samples // 2, 10 * p * p)
        _cps = sorted(detected_cps)
        while _cps:
            _bounds = [0] + _cps + [len(X)]
            _sizes  = [_bounds[i + 1] - _bounds[i] for i in range(len(_bounds) - 1)]
            if min(_sizes) >= min_window:
                break
            min_idx = _sizes.index(min(_sizes))
            if min_idx == 0:
                _cps.pop(0)
            elif min_idx == len(_sizes) - 1:
                _cps.pop(-1)
            else:
                # Prefer the onset (earlier) detection — it marks when the change
                # started, not when the signal settled back.
                onset_set = set(det_result.onset_points)
                left_cp  = _bounds[min_idx]
                right_cp = _bounds[min_idx + 1]
                keep = left_cp if (left_cp in onset_set or right_cp not in onset_set) else right_cp
                _cps.pop(min_idx)
                _cps[min_idx - 1] = keep
        detected_cps = _cps
        print(f"    After min-window filter ({min_window} samples): {detected_cps}")

    # Calibration thresholds
    if not skip_detection:
        print(f"    Calibration: eps_on_pos={det_result.calibration.eps_on_pos:.2f}, "
              f"eps_on_neg={det_result.calibration.eps_on_neg:.2f}")

    # Per-moment contributions at each GT change point
    if not skip_detection and det_result.E_by_moment:
        moment_names_map = {1: 'Mean', 2: 'Variance', 3: 'Skewness', 4: 'Kurtosis'}
        moments_used = det_result.moments_used or [1, 2, 3, 4]
        T_tmp = len(X)
        print(f"\n    Per-moment contributions at ground-truth change points:")
        for gt in true_cps:
            print(f"      At t={gt}:")
            for m in moments_used:
                if m in det_result.E_by_moment:
                    E_m_pos, E_m_neg = det_result.E_by_moment[m]
                    win = 50
                    pre = np.mean(
                        E_m_pos[max(0, gt - win):gt] - E_m_neg[max(0, gt - win):gt]
                    )
                    post = np.mean(
                        E_m_pos[gt:min(T_tmp, gt + win)] - E_m_neg[gt:min(T_tmp, gt + win)]
                    )
                    print(f"        {moment_names_map.get(m, f'M{m}')}: delta={post - pre:+.2f}")

    # Detection errors (±200 tolerance)
    det_tolerance = 200
    print(f"\n    Detection errors (±{det_tolerance} tolerance):")
    for gt in true_cps:
        if detected_cps:
            closest = min(detected_cps, key=lambda x: abs(x - gt))
            err = abs(gt - closest)
            status = "OK" if err <= det_tolerance else "LARGE ERROR"
            print(f"      GT={gt}, detected={closest}, error={err} [{status}]")
        else:
            print(f"      GT={gt}, MISSED")

    # False positives
    fps = [d for d in detected_cps if min((abs(d - gt) for gt in true_cps), default=999) > det_tolerance]
    if fps:
        print(f"    Potential false positives: {fps}")

    # F1 / Precision / Recall
    eval_result = evaluate_detection(true_cps, detected_cps, tolerance=det_tolerance)
    print(f"\n    Evaluation (tolerance={det_tolerance}):")
    print(f"      F1={eval_result.f1_score:.2f}  Precision={eval_result.precision:.2f}  "
          f"Recall={eval_result.recall:.2f}")
    print(f"      TP={eval_result.true_positives}, FP={eval_result.false_positives}, "
          f"FN={eval_result.false_negatives}")

    # ── 3. Iterative CausalMorph structure extraction ────────────────────────
    # Bootstrap CausalMorph with the first regime's ground-truth structure.
    # In a real application this would come from domain knowledge or a known
    # baseline graph.  CausalMorph then iterates from this good starting point.
    first_regime = scenario["regimes"][0]
    first_graph = first_regime["graph"]
    gt_order_names = list(nx.topological_sort(first_graph))
    initial_order = [variable_names.index(v) for v in gt_order_names]
    initial_adj = (first_regime["adj_matrix"] != 0).astype(float)

    print(f"\n[3] Extracting causal structures via iterative CausalMorph...")
    print(
        f"    Initial prior from true regime 0: order={gt_order_names}, "
        f"edges={first_graph.number_of_edges()}"
    )
    print(f"    Prior mode: {prior_mode!r}  "
          f"({'anchored to regime-0 GT for every window' if prior_mode == 'anchor' else 'chained from previous window'})"
          f"{'  +hybrid(cold fallback)' if hybrid else ''}")
    structures = extract_causal_structures(
        X=X,
        detected_change_points=detected_cps,
        variable_names=variable_names,
        initial_adj=initial_adj,
        initial_order=initial_order,
        window_overlap=window_overlap,
        verbose=verbose,
        prior_mode=prior_mode,
        hybrid=hybrid,
    )

    # ── 3b. Ablation: run alternative algorithms on same windows ──────────
    ablation_results = {}
    if ablation:
        print(f"\n[3b] Running ablation: DirectLiNGAM-only + ICA-LiNGAM on same windows...")
        ablation_results = extract_causal_structures_ablation(
            X=X,
            detected_change_points=detected_cps,
            variable_names=variable_names,
            initial_adj=initial_adj,
            initial_order=initial_order,
            window_overlap=window_overlap,
            verbose=verbose,
            prior_mode=prior_mode,
        )

    # ── 4. Attach SHD metrics (iterative) ───────────────────────────────────
    T = len(X)
    for s in structures:
        true_r = assign_true_regime(s.window_start, s.window_end, true_cps, T)
        true_adj = true_adjs[true_r]
        s.true_regime_idx = true_r
        s.true_adjacency_matrix = true_adj
        s.shd_metrics = compute_shd(s.adjacency_matrix, true_adj)

    # Attach SHD for ablation methods too
    if ablation_results:
        for method, structs_m in ablation_results.items():
            for s in structs_m:
                true_r = assign_true_regime(s.window_start, s.window_end, true_cps, T)
                true_adj = true_adjs[true_r]
                s.true_regime_idx = true_r
                s.true_adjacency_matrix = true_adj
                s.shd_metrics = compute_shd(s.adjacency_matrix, true_adj)

    # ── 4.5. Bayesian consensus structure ────────────────────────────────────
    consensus_adj, edge_probs = aggregate_structures_bayesian(
        structures, variable_names
    )
    print("\n[4.5] Bayesian consensus structure:")
    consensus_edges = [
        (src, dst, float(edge_probs.loc[src, dst]))
        for src in variable_names
        for dst in variable_names
        if consensus_adj.loc[src, dst] > 0
    ]
    if consensus_edges:
        for src, dst, prob in sorted(consensus_edges, key=lambda x: -x[2]):
            print(f"      {src} → {dst}   P={prob:.3f}")
    else:
        print("      (no consensus edges at threshold 0.5)")

    # Ablation consensus per method
    ablation_consensus = {}
    if ablation_results:
        for method, structs_m in ablation_results.items():
            c_adj, c_probs = aggregate_structures_bayesian(structs_m, variable_names)
            ablation_consensus[method] = {"consensus_adj": c_adj, "edge_probs": c_probs}

    # ── 5. Print full summary ────────────────────────────────────────────────
    print_summary(structures, variable_names, true_cps, detected_cps)

    # Ablation comparison summary
    if ablation_results:
        print("\n" + "=" * 70)
        print("ABLATION COMPARISON  (same windows, different algorithms)")
        print("=" * 70)
        for method in ABLATION_METHODS:
            structs_m = ablation_results.get(method, [])
            metrics_m = [s.shd_metrics for s in structs_m if s.shd_metrics]
            if metrics_m:
                mn_shd = np.mean([m["normalized_shd"] for m in metrics_m])
                mn_f1  = np.mean([m["F1"]             for m in metrics_m])
                mn_pre = np.mean([m["Precision"]      for m in metrics_m])
                mn_rec = np.mean([m["Recall"]          for m in metrics_m])
            else:
                mn_shd = mn_f1 = mn_pre = mn_rec = float("nan")
            # Consensus vs GT
            ac = ablation_consensus.get(method)
            if ac and true_adjs:
                gt_raw = true_adjs[-1]
                gt_df = pd.DataFrame(
                    (np.asarray(gt_raw.values if hasattr(gt_raw, "values") else gt_raw) != 0).astype(float),
                    index=variable_names, columns=variable_names,
                )
                cm = compute_shd(ac["consensus_adj"], gt_df)
                cons_nshd = cm["normalized_shd"]
                cons_f1   = cm["F1"]
            else:
                cons_nshd = cons_f1 = float("nan")
            print(f"  {method:<14}  mean_nSHD={mn_shd:.3f}  F1={mn_f1:.3f}  "
                  f"Prec={mn_pre:.3f}  Rec={mn_rec:.3f}  "
                  f"| consensus  nSHD={cons_nshd:.3f}  F1={cons_f1:.3f}")
        print("=" * 70)

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    if show_plots:
        if not skip_detection:
            print("[Plots] Detection diagnostics (v1-F multi-moment)...")
            plot_detection_diagnostics(
                X,
                det_result,
                true_cps,
                detected_cps,
                title=f"Full Pipeline v1-G: {n_regimes} regimes, p={p}",
                tolerance=200,
            )

        print("[Plots] Time series with change points...")
        plot_timeseries(X, variable_names, true_cps, detected_cps)

        print("[Plots] True vs Learned causal structures...")
        plot_structures_comparison(scenario, structures, variable_names)

        print("[Plots] Bayesian consensus structure...")
        plot_consensus_structure(consensus_adj, edge_probs, variable_names, len(structures),
                                 true_adjs=true_adjs)

        print("[Plots] SHD metrics per regime...")
        plot_shd_metrics(structures)

        # Press 'q' or Escape in any figure window to close all plots
        def _close_all(event):
            if event.key in ('q', 'escape'):
                plt.close('all')
        for fig_num in plt.get_fignums():
            plt.figure(fig_num).canvas.mpl_connect('key_press_event', _close_all)
        print("[Press 'q' or Escape in any plot window to close all]")
        plt.show()

    out = {
        "structures": structures,
        "detected_change_points": detected_cps,
        "true_change_points": true_cps,
        "X": X,
        "variable_names": variable_names,
        "detection_result": det_result,
        "eval_result": eval_result,
        "consensus_adj": consensus_adj,
        "edge_probs": edge_probs,
        "true_adjs": true_adjs,
    }
    if ablation:
        out["ablation"] = ablation_results
        out["ablation_consensus"] = ablation_consensus
    return out


if __name__ == "__main__":
    import time

    # ── Toggle here ───────────────────────────────────────────────────────────
    USE_TRUE_CHANGE_POINTS = False  # True → skip detector, use ground-truth CPs
    # ─────────────────────────────────────────────────────────────────────────

    _seed = int(time.time() * 1000) % (2**31)
    print(f"[Seed] {_seed}")
    results = run_full_pipeline(
        p=4,
        n_regimes=3,
        min_samples=900,       # 90 s at 10 Hz
        max_samples=1200,      # 120 s at 10 Hz
        base_pconn=0.35,
        change_pcts=[0, 30, 25, 35, 30],
        noise_fraction=0.08,
        seed=_seed,
        verbose=True,
        show_plots=True,
        window_overlap=0,
        use_true_change_points=USE_TRUE_CHANGE_POINTS,
        prior_mode="chain",
        hybrid=False,
    )
    structures: List[RegimeStructure] = results["structures"]
    er = results["eval_result"]
    print(f"\nTotal regime structures stored : {len(structures)}")
    print(f"Detection — F1={er.f1_score:.2f}  Precision={er.precision:.2f}  Recall={er.recall:.2f}")
