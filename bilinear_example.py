"""
BilinearModel Integration Example
==================================
Generates multi-regime fNIRS-like data using the BilinearModel neurodynamics
(Z matrix driven by known A connectivity), where causal structure switches
across regimes.  Feeds the resulting time series through the full pipeline
to test detection + structure learning on physiologically realistic data.

The key idea: Z[t] evolves via  dZ/dt = A @ Z + C @ U + ...
so A encodes direct causal relationships.  By switching A across regimes
we create non-stationary causal data with known ground truth.

Usage:
    python bilinear_example.py                    # default 3-region, 3 regimes
    python bilinear_example.py --p 4 --n_regimes 4
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "BilinearModel_code-version", "src"))
sys.path.insert(0, os.path.join(_HERE, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(_HERE, "causalmorph"))

from BilinearModel_Neurodynamics import Neurodynamics
from BilinearModel_StimulusGenerator import bilinear_model_stimulus_train_generator

from full_pipeline import (
    extract_causal_structures,
    extract_causal_structures_ablation,
    assign_true_regime,
    compute_shd,
    aggregate_structures_bayesian,
    plot_timeseries,
    plot_structures_comparison,
    plot_shd_metrics,
    plot_adjacency_heatmaps,
    plot_detection_diagnostics,
    plot_consensus_structure,
    RegimeStructure,
    ABLATION_METHODS,
)
from detectors.detectors_wavelets import detect_nonstationarity_v1G
from evaluation.metrics import evaluate_detection, simplify_transitions


# ── A-matrix generation ──────────────────────────────────────────────────────

def _generate_dag_A(p: int, density: float, strength_range: tuple,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random lower-triangular A matrix (DAG, no self-loops).
    A[i, j] != 0 means j -> i  (row = effect, col = cause).
    """
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            if rng.random() < density:
                lo, hi = strength_range
                w = rng.uniform(lo, hi)
                if rng.random() < 0.5:
                    w = -w
                A[i, j] = w
    return A


def _A_to_binary_adj(A: np.ndarray) -> np.ndarray:
    """Convert A connectivity matrix to binary adjacency (cause -> effect, no self-loops)."""
    adj = (A != 0).astype(float)
    np.fill_diagonal(adj, 0.0)
    return adj


def _A_to_dag(A: np.ndarray, variable_names: List[str]) -> nx.DiGraph:
    """Build a DiGraph from A matrix (A[i,j] != 0 means j -> i). Skips self-loops."""
    G = nx.DiGraph()
    G.add_nodes_from(variable_names)
    p = A.shape[0]
    for i in range(p):
        for j in range(p):
            if i != j and abs(A[i, j]) > 1e-10:
                G.add_edge(variable_names[j], variable_names[i], weight=float(A[i, j]))
    return G


# ── Multi-regime neurodynamics data generation ───────────────────────────────

def generate_bilinear_regimes(
    p: int = 3,
    n_regimes: int = 3,
    samples_per_regime: int = 1000,
    freq: float = 10.0,
    density: float = 0.4,
    noise_fraction: float = 0.05,
    seed: int = 42,
):
    """
    Generate multi-regime neurodynamics data with switching A matrices.

    Each regime uses a different A matrix (causal structure), simulating
    plasticity or learning.  The neurodynamics ODE is integrated per regime
    and segments are concatenated.

    Returns
    -------
    Z_full     : np.ndarray [T_total, p]  — concatenated neurodynamics
    true_cps   : List[int]                — change point indices
    true_As    : List[np.ndarray]         — ground truth A per regime
    true_adjs  : List[pd.DataFrame]       — binary adjacency per regime
    variable_names : List[str]
    scenario   : dict                     — compatible with plot_structures_comparison
    """
    rng = np.random.default_rng(seed)
    variable_names = [f"V{i+1}" for i in range(p)]

    # Learning-task regime generation:
    #   Regime 0 — "novice": very sparse graph, weak weights (few mechanics discovered)
    #   Middle regimes — "exploration": big structural rewiring, new edges appear and
    #                     old wrong ones disappear (trial-and-error learning)
    #   Final regime — "proficient": denser graph, stronger weights (mechanics mastered)
    #
    # Each transition changes 40-60% of possible edges — drastic, like discovering
    # a new game mechanic that reframes how everything connects.

    max_edges = p * (p - 1) // 2  # lower-triangular slots

    A_list = []

    def _add_self_damping(A_mat):
        """Add negative diagonal (self-decay) so the ODE is stable."""
        for i in range(A_mat.shape[0]):
            A_mat[i, i] = -rng.uniform(0.05, 0.15)
        return A_mat

    # Regime 0: novice — sparse, moderate coupling
    novice_density = max(0.25, density * 0.5)
    A_base = _generate_dag_A(p, novice_density, (0.20, 0.45), rng)
    if np.count_nonzero(A_base - np.diag(np.diag(A_base))) == 0:
        A_base[1, 0] = rng.uniform(0.20, 0.45)
    _add_self_damping(A_base)
    A_list.append(A_base.copy())

    for r in range(1, n_regimes):
        A_new = A_list[-1].copy()
        progress = r / (n_regimes - 1) if n_regimes > 1 else 1.0

        # Number of edge slots to change: 40-60% of all possible edges
        n_changes = max(2, int(max_edges * rng.uniform(0.4, 0.6)))

        # Pick random lower-triangular positions to rewire
        all_slots = [(i, j) for i in range(1, p) for j in range(i)]
        rng.shuffle(all_slots)
        slots_to_change = all_slots[:n_changes]

        for i, j in slots_to_change:
            if abs(A_new[i, j]) > 1e-10:
                A_new[i, j] = 0.0
            else:
                lo = 0.20 + 0.20 * progress
                hi = 0.45 + 0.25 * progress
                w = rng.uniform(lo, hi)
                if rng.random() < 0.5:
                    w = -w
                A_new[i, j] = w

        # Surviving edges get reinforced
        for i in range(1, p):
            for j in range(i):
                if abs(A_new[i, j]) > 1e-10:
                    scale = 1.0 + progress * rng.uniform(0.1, 0.3)
                    A_new[i, j] *= scale

        # Re-draw self-damping (can vary across regimes)
        _add_self_damping(A_new)

        # Guarantee minimum connectivity
        n_off_diag = np.count_nonzero(A_new - np.diag(np.diag(A_new)))
        min_edges = max(2, int(max_edges * (0.2 + 0.4 * progress)))
        while n_off_diag < min_edges:
            i = rng.integers(1, p)
            j = rng.integers(0, i)
            if abs(A_new[i, j]) < 1e-10:
                lo = 0.20 + 0.20 * progress
                hi = 0.45 + 0.25 * progress
                A_new[i, j] = rng.uniform(lo, hi) * (1 if rng.random() > 0.5 else -1)
                n_off_diag += 1

        A_list.append(A_new)

    B = np.zeros((p, p, p))
    C = np.eye(p) * 0.06

    # Stimulus: block design (5s on, 25s rest, multiple cycles)
    action_times = [5] * p
    rest_times = [25] * p
    n_cycles_per_regime = max(2, samples_per_regime // int(freq * 30))
    cycles = [n_cycles_per_regime] * p

    segments = []
    true_cps = []
    offset = 0

    for r, A in enumerate(A_list):
        U_stim, timestamps = bilinear_model_stimulus_train_generator(
            freq, action_times, rest_times, cycles, p,
        )
        T_regime = len(timestamps)
        Z0 = rng.standard_normal(p) * 0.01 if r == 0 else segments[-1][-1, :]

        Z = Neurodynamics(Z0, timestamps, A.copy(), B.copy(), C.copy(), U_stim)

        # Add small noise for LiNGAM identifiability (Laplace)
        if noise_fraction > 0:
            for j in range(p):
                sig = float(np.std(Z[:, j])) or 1.0
                Z[:, j] += noise_fraction * sig * rng.laplace(0, 1, T_regime)

        segments.append(Z)
        if r > 0:
            true_cps.append(offset)
        offset += T_regime

    Z_full = np.vstack(segments)

    # Build true adjacency DataFrames and scenario dict (for plotting compatibility)
    true_adjs = []
    regimes_for_scenario = []
    for r, A in enumerate(A_list):
        adj_bin = _A_to_binary_adj(A)
        adj_df = pd.DataFrame(adj_bin, index=variable_names, columns=variable_names)
        true_adjs.append(adj_df)
        G = _A_to_dag(A, variable_names)
        regimes_for_scenario.append({
            "graph": G,
            "adj_matrix": pd.DataFrame(A, index=variable_names, columns=variable_names),
        })

    scenario = {
        "regimes": regimes_for_scenario,
        "change_points": true_cps,
        "combined_data": pd.DataFrame(Z_full, columns=variable_names),
    }

    return Z_full, true_cps, A_list, true_adjs, variable_names, scenario


# ── Main ─────────────────────────────────────────────────────────────────────

def run_bilinear_example(
    p: int = 5,
    n_regimes: int = 3,
    samples_per_regime: int = 1000,
    freq: float = 10.0,
    density: float = 0.4,
    noise_fraction: float = 0.12,
    seed: int = 42,
    use_true_cps: bool = False,
    ablation: bool = True,
    show_plots: bool = True,
    verbose: bool = True,
) -> dict:
    """
    End-to-end example: BilinearModel data → detection → structure learning.
    """
    print("=" * 70)
    print("BilinearModel Integration Example")
    print("=" * 70)

    # ── 1. Generate data ─────────────────────────────────────────────────────
    print(f"\n[1] Generating bilinear neurodynamics data  "
          f"(p={p}, {n_regimes} regimes, {samples_per_regime} samples/regime)")
    Z, true_cps, A_list, true_adjs, variable_names, scenario = generate_bilinear_regimes(
        p=p, n_regimes=n_regimes, samples_per_regime=samples_per_regime,
        freq=freq, density=density, noise_fraction=noise_fraction, seed=seed,
    )
    T_total = len(Z)
    print(f"    Total samples: {T_total}")
    print(f"    True change points: {true_cps}")
    for r, A in enumerate(A_list):
        n_edges = int(np.count_nonzero(A))
        print(f"    Regime {r}: {n_edges} edges  A_diag_off={np.count_nonzero(A - np.diag(np.diag(A)))}")

    # ── 2. Detect change points ──────────────────────────────────────────────
    if use_true_cps:
        print(f"\n[2] Using ground-truth change points: {true_cps}")
        detected_cps = true_cps
        det_result = None
    else:
        print(f"\n[2] Running wavelet detector (bilinear-tuned)...")
        Z_det = Z.copy()
        for ch in range(Z_det.shape[1]):
            mu, sigma = Z_det[:, ch].mean(), Z_det[:, ch].std()
            if sigma > 1e-12:
                Z_det[:, ch] = (Z_det[:, ch] - mu) / sigma
        regime_len = samples_per_regime
        baseline_end = min(150, regime_len // 4)
        refractory = min(150, regime_len // 4)
        det_result = detect_nonstationarity_v1G(
            Z_det,
            baseline_idx=np.arange(0, baseline_end),
            min_scale=3.0,
            moments=[1, 2, 3, 4],
            moment_window=50,
            n_surrogates=100,
            alpha=0.45,
            k_scales_min=1,
            smooth_window=12,
            refractory_period=refractory,
            min_snr=0.25,
            k_channels_min=1,
            delta_ch_k=1.2,
            step_delta_k=1.2,
            adaptive_max_discount=0.6,
            step_pre_win=100,
            step_post_win=100,
            two_pass=True,
            seed=seed,
        )
        onset_set = set(det_result.onset_points)
        first_onset = min(onset_set) if onset_set else float("inf")
        kept_offsets = [
            cp for cp in det_result.offset_points
            if (cp < first_onset) or all(abs(cp - on) > refractory * 2 for on in onset_set)
        ]
        raw_cps = sorted(onset_set | set(kept_offsets))
        detected_cps = simplify_transitions(raw_cps, min_distance=160)

        min_window = max(regime_len // 2, 10 * p * p)
        _cps = sorted(detected_cps)
        while _cps:
            _bounds = [0] + _cps + [T_total]
            _sizes = [_bounds[i+1] - _bounds[i] for i in range(len(_bounds) - 1)]
            if min(_sizes) >= min_window:
                break
            min_idx = _sizes.index(min(_sizes))
            if min_idx == 0:
                _cps.pop(0)
            elif min_idx == len(_sizes) - 1:
                _cps.pop(-1)
            else:
                _cps.pop(min_idx)
        detected_cps = _cps
        print(f"    Detected: {detected_cps}")

    # Detection evaluation
    eval_result = evaluate_detection(true_cps, detected_cps, tolerance=125)
    print(f"    F1={eval_result.f1_score:.2f}  Precision={eval_result.precision:.2f}  "
          f"Recall={eval_result.recall:.2f}")

    # ── 3. Structure learning ────────────────────────────────────────────────
    # Use first regime's GT as initial prior (same as full_pipeline)
    G0 = scenario["regimes"][0]["graph"]
    gt_order_names = list(nx.topological_sort(G0))
    initial_order = [variable_names.index(v) for v in gt_order_names]
    initial_adj = true_adjs[0].copy()

    print(f"\n[3] CausalMorph + DirectLiNGAM on each window...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*overflow.*|.*divide by zero.*|.*invalid value.*",
                                RuntimeWarning)
        warnings.filterwarnings("ignore", ".*FastICA did not converge.*")
        structures = extract_causal_structures(
            X=Z, detected_change_points=detected_cps,
            variable_names=variable_names,
            initial_adj=initial_adj, initial_order=initial_order,
            window_overlap=0.0, verbose=verbose, prior_mode="chain",
        )

    # Attach SHD
    for s in structures:
        true_r = assign_true_regime(s.window_start, s.window_end, true_cps, T_total)
        s.true_regime_idx = true_r
        s.true_adjacency_matrix = true_adjs[true_r]
        s.shd_metrics = compute_shd(s.adjacency_matrix, true_adjs[true_r])

    # ── 3b. Ablation ─────────────────────────────────────────────────────────
    ablation_results = {}
    if ablation:
        print(f"\n[3b] Ablation: DirectLiNGAM-only + ICA-LiNGAM...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*overflow.*|.*divide by zero.*|.*invalid value.*",
                                    RuntimeWarning)
            warnings.filterwarnings("ignore", ".*FastICA did not converge.*")
            ablation_results = extract_causal_structures_ablation(
                X=Z, detected_change_points=detected_cps,
                variable_names=variable_names,
                initial_adj=initial_adj, initial_order=initial_order,
                window_overlap=0.0, verbose=verbose, prior_mode="chain",
            )
        for method, structs_m in ablation_results.items():
            for s in structs_m:
                true_r = assign_true_regime(s.window_start, s.window_end, true_cps, T_total)
                s.true_regime_idx = true_r
                s.true_adjacency_matrix = true_adjs[true_r]
                s.shd_metrics = compute_shd(s.adjacency_matrix, true_adjs[true_r])

    # ── 4. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  True CPs: {true_cps}   Detected CPs: {detected_cps}")
    print(f"  Detection F1={eval_result.f1_score:.2f}")

    def _method_summary(label, structs):
        metrics = [s.shd_metrics for s in structs if s.shd_metrics]
        if metrics:
            mn = np.mean([m["normalized_shd"] for m in metrics])
            mf = np.mean([m["F1"] for m in metrics])
            mp = np.mean([m["Precision"] for m in metrics])
            mr = np.mean([m["Recall"] for m in metrics])
        else:
            mn = mf = mp = mr = float("nan")
        # consensus
        c_adj, c_probs = aggregate_structures_bayesian(structs, variable_names)
        gt_df = true_adjs[-1]
        cm = compute_shd(c_adj, gt_df)
        print(f"  {label:<14}  mean_nSHD={mn:.3f}  F1={mf:.3f}  Prec={mp:.3f}  Rec={mr:.3f}"
              f"  | consensus nSHD={cm['normalized_shd']:.3f}  F1={cm['F1']:.3f}")

    _method_summary("causalmorph", structures)
    if ablation_results:
        for method in ABLATION_METHODS:
            if method == "causalmorph":
                continue
            _method_summary(method, ablation_results.get(method, []))
    print("=" * 70)

    # ── 5. Consensus ────────────────────────────────────────────────────────
    consensus_adj, edge_probs = aggregate_structures_bayesian(structures, variable_names)

    ablation_consensus = {}
    if ablation_results:
        for method, structs_m in ablation_results.items():
            c_adj, c_probs = aggregate_structures_bayesian(structs_m, variable_names)
            ablation_consensus[method] = {"consensus_adj": c_adj, "edge_probs": c_probs}

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    if show_plots:
        if det_result is not None:
            print("[Plots] Detection diagnostics (v1-F multi-moment)...")
            plot_detection_diagnostics(
                Z, det_result, true_cps, detected_cps,
                title=f"BilinearModel v1-F: {n_regimes} regimes, p={p}",
                tolerance=125,
            )

        print("[Plots] Time series with change points...")
        plot_timeseries(Z, variable_names, true_cps, detected_cps)

        print("[Plots] True vs Learned causal structures...")
        plot_structures_comparison(scenario, structures, variable_names)

        print("[Plots] Bayesian consensus structure...")
        plot_consensus_structure(consensus_adj, edge_probs, variable_names,
                                 len(structures), true_adjs=true_adjs)

        print("[Plots] Adjacency heatmaps...")
        plot_adjacency_heatmaps(structures, variable_names)

        print("[Plots] SHD metrics per regime...")
        plot_shd_metrics(structures)

        if ablation_results:
            methods_to_plot = ["causalmorph"] + [m for m in ABLATION_METHODS if m != "causalmorph"]
            all_structs = {"causalmorph": structures}
            all_structs.update(ablation_results)

            fig, ax = plt.subplots(figsize=(10, 5))
            n_methods = len(methods_to_plot)
            n_windows = len(structures)
            width = 0.8 / n_methods
            x = np.arange(n_windows)
            colors = {"causalmorph": "#2ECC71", "directlingam": "#3498DB", "icalingam": "#E74C3C"}

            for i, method in enumerate(methods_to_plot):
                vals = [s.shd_metrics.get("normalized_shd", float("nan"))
                        for s in all_structs.get(method, [])]
                if len(vals) < n_windows:
                    vals += [float("nan")] * (n_windows - len(vals))
                ax.bar(x + i * width, vals, width, label=method,
                       color=colors.get(method, "gray"), alpha=0.8)

            ax.set_xticks(x + width * (n_methods - 1) / 2)
            ax.set_xticklabels([f"R{i}" for i in range(n_windows)])
            ax.set_ylabel("Normalized SHD (lower = better)")
            ax.set_title("Ablation: nSHD per Regime Window")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show(block=False)

        def _close_all(event):
            if event.key in ('q', 'escape'):
                plt.close('all')
        for fig_num in plt.get_fignums():
            plt.figure(fig_num).canvas.mpl_connect('key_press_event', _close_all)
        print("[Press 'q' or Escape in any plot window to close all]")
        plt.show()

    return {
        "structures": structures,
        "ablation": ablation_results,
        "ablation_consensus": ablation_consensus,
        "consensus_adj": consensus_adj,
        "edge_probs": edge_probs,
        "true_cps": true_cps,
        "detected_cps": detected_cps,
        "eval_result": eval_result,
        "Z": Z,
        "variable_names": variable_names,
        "true_adjs": true_adjs,
        "scenario": scenario,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BilinearModel causal discovery example")
    parser.add_argument("--p",                type=int,   default=5)
    parser.add_argument("--n_regimes",        type=int,   default=3)
    parser.add_argument("--samples_per_regime", type=int, default=1000)
    parser.add_argument("--freq",             type=float, default=10.0)
    parser.add_argument("--density",          type=float, default=0.4)
    parser.add_argument("--noise_fraction",   type=float, default=0.12)
    parser.add_argument("--seed",             type=int,   default=int(time.time()) % (2**31))
    parser.add_argument("--use_true_cps",     action="store_true")
    parser.add_argument("--no_ablation",      action="store_true")
    parser.add_argument("--no_plots",         action="store_true")
    args = parser.parse_args()

    print(f"[Seed] {args.seed}")
    run_bilinear_example(
        p=args.p,
        n_regimes=args.n_regimes,
        samples_per_regime=args.samples_per_regime,
        freq=args.freq,
        density=args.density,
        noise_fraction=args.noise_fraction,
        seed=args.seed,
        use_true_cps=args.use_true_cps,
        ablation=not args.no_ablation,
        show_plots=not args.no_plots,
    )
