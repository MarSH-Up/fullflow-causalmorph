"""
Experiment runner: Test wavelet detector (v1-F) on NonStationaryCausalScenarios.

v1-F adds (over v1-E):
- K-of-channels consistency gate: rejects peaks where < k_channels_min
  channels show a significant per-channel step (reduces single-channel FPs)
- Per-event multichannel diagnostics (n_active_channels, concentration_ratio)

v1-E features (preserved):
- Multi-moment analysis (mean, variance, skewness, kurtosis)
- Factorial-inverse weighting: 1/m! for moment m
- Wavelet analysis on rolling moment time series
- Signed deviations, differential dE, peak-based detection, step validation
"""

import os
import sys

# Pin to single thread to avoid non-deterministic floating-point from multi-threaded BLAS/FFT
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from scenarios.NonStationaryCausalScenarios import NonStationaryCausalScenario
from detectors.detectors_wavelets import (
    detect_nonstationarity_multimoment,
    DetectionResult,
    get_moment_weights,
)
from evaluation.metrics import evaluate_detection, simplify_transitions


# =============================================================================
# Visualization (Updated for v1-D)
# =============================================================================

def plot_structure_changes(scenario: dict, figsize: tuple = (14, 4)):
    """
    Plot the DAG structures for each regime.
    """
    import networkx as nx

    regimes = scenario["regimes"]
    n_regimes = len(regimes)

    fig, axes = plt.subplots(1, n_regimes, figsize=figsize)
    if n_regimes == 1:
        axes = [axes]

    for idx, (regime, ax) in enumerate(zip(regimes, axes)):
        G = regime["graph"]

        # Consistent layout across regimes
        pos = nx.spring_layout(G, seed=42)

        # Draw graph
        nx.draw(
            G, pos, ax=ax,
            with_labels=True,
            node_color='lightblue',
            node_size=1500,
            edge_color='gray',
            arrowsize=15,
            font_size=10,
            font_weight='bold',
        )

        n_edges = G.number_of_edges()
        ax.set_title(f"Regime {idx + 1}\n({n_edges} edges)", fontsize=11)

    plt.suptitle("Causal Structure Changes Across Regimes", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig


def plot_multimoment_results(
    data: np.ndarray,
    result: DetectionResult,
    ground_truth: List[int],
    detected: List[int],
    title: str = "Multi-Moment Non-Stationarity Detection (v1-F)",
    figsize: tuple = (14, 16),
    tolerance: int = 0,
):
    """
    Plot multi-moment detection results showing per-moment contributions.

    Panels:
    1. Original signals with change points
    2. Energy derivative dE(t) - key detection signal
    3. Signed energy E_signed(t) = E_pos - E_neg
    4-7. Per-moment energies (mean, var, skew, kurt) if available
    8. Forgetting severity trace

    Parameters
    ----------
    tolerance : int
        If > 0, shade ±tolerance window around each ground-truth change point
        to visualize the acceptance band. Default: 0 (no shading).
    """
    T, N = data.shape

    # Determine number of panels based on available moments
    moments_available = result.moments_used if result.moments_used else [2]
    n_moment_panels = len(moments_available)
    n_panels = 4 + n_moment_panels  # 3 base panels + moment panels + severity

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

    # Common style
    gt_style = dict(color='#2ecc71', linestyle='-', linewidth=3, alpha=0.9)
    det_style = dict(color='#e74c3c', linestyle='--', linewidth=2.5, alpha=0.9)
    onset_style = dict(color='#e74c3c', linestyle='--', linewidth=2.5, alpha=0.9)
    offset_style = dict(color='#3498db', linestyle=':', linewidth=2.5, alpha=0.9)

    def _draw_gt_bands(ax):
        """Draw ground-truth vertical lines and optional tolerance shading."""
        for i, cp in enumerate(ground_truth):
            if tolerance > 0:
                ax.axvspan(cp - tolerance, cp + tolerance,
                           alpha=0.18, color='#2ecc71', zorder=0,
                           label=f'±{tolerance} window' if i == 0 else '')
            ax.axvline(cp, **gt_style, label='Ground Truth' if i == 0 else '', zorder=10)

    moment_names = {1: 'Mean', 2: 'Variance', 3: 'Skewness', 4: 'Kurtosis'}
    moment_colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71', 4: '#9b59b6'}

    panel_idx = 0

    # Panel 1: Original signals
    ax = axes[panel_idx]
    for ch in range(min(3, N)):
        ax.plot(data[:, ch], alpha=0.6, linewidth=0.8, label=f"V{ch+1}")
    _draw_gt_bands(ax)
    for i, cp in enumerate(result.onset_points):
        ax.axvline(cp, **onset_style, label='Onset' if i == 0 else '', zorder=11)
    for i, cp in enumerate(result.offset_points):
        ax.axvline(cp, **offset_style, label='Offset' if i == 0 else '', zorder=11)
    ax.set_ylabel("Signal")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    panel_idx += 1

    # Panel 2: Energy Derivative dE(t)
    ax = axes[panel_idx]
    ax.fill_between(range(T), result.dE, 0, where=(result.dE > 0),
                    alpha=0.4, color='red', label='dE > 0 (onset)')
    ax.fill_between(range(T), result.dE, 0, where=(result.dE < 0),
                    alpha=0.4, color='blue', label='dE < 0 (offset)')
    ax.plot(result.dE, color='black', linewidth=0.8)
    ax.axhline(result.calibration.eps_on_pos, color='red', linestyle=':', alpha=0.7,
               label=f'eps_pos={result.calibration.eps_on_pos:.1f}')
    ax.axhline(-result.calibration.eps_on_neg, color='blue', linestyle=':', alpha=0.7,
               label=f'eps_neg={result.calibration.eps_on_neg:.1f}')
    _draw_gt_bands(ax)
    for event in result.events:
        marker = '^' if event.event_type == "onset" else 'v'
        color = 'red' if event.event_type == "onset" else 'blue'
        ax.plot(event.tau, result.dE[event.tau], marker, color=color, markersize=10, zorder=12)
    ax.set_ylabel("dE (derivative)")
    ax.set_title("Energy Derivative dE(t) - Detection Signal")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)
    panel_idx += 1

    # Panel 3: Signed energy E_signed
    ax = axes[panel_idx]
    ax.fill_between(range(T), result.E_signed, alpha=0.4, color='steelblue')
    ax.plot(result.E_signed, color='navy', linewidth=0.8)
    ax.plot(result.E_pos, color='red', linewidth=0.5, alpha=0.5, label='E_pos')
    ax.plot(-result.E_neg, color='blue', linewidth=0.5, alpha=0.5, label='-E_neg')
    _draw_gt_bands(ax)
    for i, cp in enumerate(detected):
        ax.axvline(cp, **det_style, label='Detected' if i == 0 else '', zorder=11)
    ax.set_ylabel("E_signed")
    ax.set_title("Total Weighted Energy E_signed(t) = E_pos - E_neg")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)
    panel_idx += 1

    # Panels 4+: Per-moment energies
    if result.E_by_moment:
        weights = result.moment_weights or get_moment_weights(moments_available)
        for m in moments_available:
            if m in result.E_by_moment:
                E_m_pos, E_m_neg = result.E_by_moment[m]
                E_m_signed = E_m_pos - E_m_neg
                weight = weights.get(m, 1.0)

                ax = axes[panel_idx]
                color = moment_colors.get(m, 'gray')
                ax.fill_between(range(T), E_m_signed, alpha=0.3, color=color)
                ax.plot(E_m_signed, color=color, linewidth=0.8,
                       label=f'{moment_names.get(m, f"M{m}")} (w=1/{m}!={weight:.3f})')
                _draw_gt_bands(ax)
                ax.set_ylabel(f"E_{moment_names.get(m, f'M{m}')}")
                ax.set_title(f"{moment_names.get(m, f'Moment {m}')} Energy (weight = 1/{m}! = {weight:.3f})")
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.2)
                panel_idx += 1

    # Final panel: Forgetting severity
    ax = axes[panel_idx]
    ax.plot(result.severity, color='purple', linewidth=1.2)
    ax.fill_between(range(T), result.severity, alpha=0.2, color='purple')
    _draw_gt_bands(ax)
    for i, cp in enumerate(detected):
        ax.axvline(cp, **det_style, label='Detected' if i == 0 else '', zorder=11)
    ax.set_ylabel("Severity")
    ax.set_xlabel("Sample Index")
    ax.set_title("Forgetting Severity (Leaky Integrator)")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    return fig


# =============================================================================
# Experiment with Multiple Regimes
# =============================================================================

def run_multi_regime_experiment_multimoment(
    p: int = 5,
    n_regimes: int = 4,
    n_samples_per_regime: int = 300,
    change_pct: float = 25.0,
    moments: List[int] = [1, 2, 3, 4],
    moment_window: int = 50,
    seed: int = 42,
    show_plots: bool = True,
):
    """
    Run multi-moment detection (v1-F) on a scenario with multiple regime changes.

    Parameters
    ----------
    p : int
        Number of variables
    n_regimes : int
        Number of regimes (n_regimes - 1 change points)
    n_samples_per_regime : int
        Samples per regime
    change_pct : float
        Structural change percentage between consecutive regimes
    moments : List[int]
        Which moments to use (1=mean, 2=var, 3=skew, 4=kurt)
    moment_window : int
        Window size for rolling moment computation
    seed : int
        Random seed
    show_plots : bool
        Whether to display plots
    """
    print("=" * 70)
    print(f"Gatekeeper v1-F: {n_regimes}-Regime Multi-Moment Detection")
    print("=" * 70)

    np.random.seed(seed)
    scenario_gen = NonStationaryCausalScenario(p=p, mode="linear", seed=seed)

    # Generate chain of DAGs with progressive changes
    dags = []
    change_infos = []

    # First DAG (random base graph)
    G0, _, _ = NonStationaryCausalScenario.create_regime_pair_with_change(
        p=p, base_pconn=0.3, change_pct=0, seed=seed
    )
    dags.append(G0)

    # Each subsequent DAG is a change FROM THE PREVIOUS DAG
    for i in range(1, n_regimes):
        _, Gi, info = NonStationaryCausalScenario.create_regime_pair_with_change(
            p=p,
            change_pct=change_pct,
            change_type="mixed",
            seed=seed + i * 100,
            base_graph=dags[-1],
        )
        dags.append(Gi)
        change_infos.append(info)

    # Print structural changes
    print(f"\nScenario: {p} variables, {n_regimes} regimes, {n_samples_per_regime} samples/regime")
    print(f"\nStructural changes between regimes:")
    for k, info in enumerate(change_infos, start=1):
        print(
            f"  Regime {k} -> {k+1}: {info['actual_change_pct']:.1f}% change "
            f"(+{info['n_edges_added']} edges, -{info['n_edges_removed']} edges, "
            f"Jaccard={info['jaccard_edges']:.2f})"
        )

    # Print moment weights
    weights = get_moment_weights(moments)
    print(f"\nMoment weights (factorial-inverse):")
    moment_names = {1: 'Mean', 2: 'Variance', 3: 'Skewness', 4: 'Kurtosis'}
    for m in moments:
        print(f"  {moment_names.get(m, f'M{m}')}: 1/{m}! = {weights[m]:.4f}")

    # Build regime configs
    regime_configs = []
    for i, dag in enumerate(dags):
        deviation = np.random.uniform(0.5, 1.5)
        regime_configs.append({
            "fixed_graph": dag,
            "deviation": deviation,
            "signal_strength": 1.5,
            "nsamples": n_samples_per_regime,
            "dist": ["normal"] * p,
            "regime_seed": seed + i * 50,
        })

    # Generate scenario
    scenario = scenario_gen.create_nonstationary_scenario(
        regime_configs=regime_configs,
        transition_type="abrupt",
    )

    data = scenario["combined_data"].values
    ground_truth = scenario["change_points"]
    T = data.shape[0]

    # Run multi-moment detector (v1-F)
    # Use shorter baseline for multi-regime (first 60% of first regime)
    baseline_idx = np.arange(int(n_samples_per_regime * 0.6))

    # Adjust parameters for multi-regime detection:
    # - Shorter refractory period to allow detecting changes closer together
    # - Lower min_snr since later changes may have different baseline
    refractory = min(150, n_samples_per_regime // 5)

    result = detect_nonstationarity_multimoment(
        data, baseline_idx,
        min_scale=4.0,
        moments=moments,
        moment_window=moment_window,
        n_surrogates=75,
        alpha=0.02,  # Slightly less strict for multi-regime
        k_scales_min=2,  # Lower threshold for multi-regime
        smooth_window=15,
        refractory_period=refractory,
        min_snr=1.5,  # Lower SNR threshold for subsequent changes
        k_channels_min=2,  # v1-F: require 2+ channels
        seed=seed,
    )

    # Print ground truth
    print(f"\nGround truth ({len(ground_truth)} change points):")
    for i, cp in enumerate(ground_truth):
        print(f"  t={cp}: regime {i+1} -> regime {i+2}")

    # Print detection results
    print(f"\nDetected change points: {result.change_points}")
    print(f"  Onset points: {result.onset_points}")
    print(f"  Offset points: {result.offset_points}")

    # Print events
    print(f"\nEvents detected: {len(result.events)}")
    for i, event in enumerate(result.events):
        ch_info = ""
        if event.n_active_channels is not None:
            ch_info = f", ch_active={event.n_active_channels}/{data.shape[1]}, conc={event.concentration_ratio:.2f}"
        print(f"  Event {i+1}: type={event.event_type}, tau={event.tau}, "
              f"peak_dE={event.peak_dE:.2f}{ch_info}")

    # Print calibration thresholds
    print(f"\nCalibration thresholds:")
    print(f"  eps_on_pos (onset): {result.calibration.eps_on_pos:.2f}")
    print(f"  eps_on_neg (offset): {result.calibration.eps_on_neg:.2f}")

    # Print severity stats
    print(f"\nSeverity trace (forgetting): max={result.severity.max():.2f}, "
          f"decays to {result.severity[-1]:.2f} at end")

    # Print per-moment contribution at each change point
    if result.E_by_moment:
        print(f"\nPer-moment contributions at change points:")
        for gt in ground_truth:
            print(f"  At t={gt}:")
            window = 50
            for m in moments:
                if m in result.E_by_moment:
                    E_m_pos, E_m_neg = result.E_by_moment[m]
                    pre_mean = np.mean(E_m_pos[max(0, gt-window):gt] - E_m_neg[max(0, gt-window):gt])
                    post_mean = np.mean(E_m_pos[gt:min(T, gt+window)] - E_m_neg[gt:min(T, gt+window)])
                    delta = post_mean - pre_mean
                    print(f"    {moment_names.get(m, f'M{m}')}: delta={delta:+.2f}")

    # Print detection errors
    detected = simplify_transitions(result.change_points, min_distance=50)
    print(f"\nDetection errors:")
    for gt in ground_truth:
        closest = min(detected, key=lambda x: abs(x - gt)) if detected else None
        if closest is not None:
            error = abs(gt - closest)
            print(f"  GT={gt}, detected={closest}, error={error} samples")
        else:
            print(f"  GT={gt}, MISSED")

    # Evaluate
    eval_result = evaluate_detection(ground_truth, detected, tolerance=125)

    print(f"\nEvaluation (tolerance=125 for causal data):")
    print(f"  F1={eval_result.f1_score:.2f}  Precision={eval_result.precision:.2f}  Recall={eval_result.recall:.2f}")
    print(f"  TP={eval_result.true_positives}, FP={eval_result.false_positives}, FN={eval_result.false_negatives}")

    # Visualize
    if show_plots:
        plot_structure_changes(scenario)
        plot_multimoment_results(
            data, result, ground_truth, detected,
            title=f"Multi-Regime v1-E ({n_regimes} regimes, {change_pct}% change)",
            tolerance=125,
        )

    print("=" * 70)
    return {
        "scenario": scenario,
        "result": result,
        "eval": eval_result,
        "ground_truth": ground_truth,
        "detected": detected,
    }


def run_realistic_scenario(
    p: int = 5,
    seed: int = 42,
    moments: List[int] = [1, 2, 3, 4],
    moment_window: int = 50,
    show_plots: bool = True,
):
    """
    Run a more realistic scenario with:
    - 5 nodes (variables)
    - 5 regimes (4 change points)
    - Variable regime lengths (200-600 samples each)
    - Different structural change percentages between regimes

    This simulates a more realistic non-stationary time series where
    regime durations and change magnitudes vary.
    """
    print("=" * 70)
    print("Gatekeeper v1-F: REALISTIC Multi-Regime Scenario")
    print("=" * 70)

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Define regime configurations with variable lengths and changes
    # Format: (n_samples, change_pct_from_previous)
    # Random samples per regime with minimum 600
    min_samples = 600
    max_samples = 800

    # Randomize regime lengths
    regime_lengths = [rng.integers(min_samples, max_samples + 1) for _ in range(5)]
    change_pcts = [0, 30, 25, 35, 30]  # Structural change percentages

    regime_specs = list(zip(regime_lengths, change_pcts))

    n_regimes = len(regime_specs)
    scenario_gen = NonStationaryCausalScenario(p=p, mode="linear", seed=seed)

    # Generate chain of DAGs with progressive changes
    dags = []
    change_infos = []

    # First DAG (random base graph with moderate connectivity)
    G0, _, _ = NonStationaryCausalScenario.create_regime_pair_with_change(
        p=p, base_pconn=0.35, change_pct=0, seed=seed
    )
    dags.append(G0)

    # Each subsequent DAG is a change from the previous
    for i in range(1, n_regimes):
        _, Gi, info = NonStationaryCausalScenario.create_regime_pair_with_change(
            p=p,
            change_pct=regime_specs[i][1],
            change_type="mixed",
            seed=seed + i * 100,
            base_graph=dags[-1],
        )
        dags.append(Gi)
        change_infos.append(info)

    # Print scenario summary
    total_samples = sum(spec[0] for spec in regime_specs)
    print(f"\nScenario: {p} variables, {n_regimes} regimes, {total_samples} total samples")
    print(f"\nRegime structure:")
    cumulative = 0
    for i, (n_samples, change_pct) in enumerate(regime_specs):
        if i == 0:
            print(f"  Regime {i+1}: t=[0, {cumulative + n_samples}) - {n_samples} samples (baseline)")
        else:
            print(f"  Regime {i+1}: t=[{cumulative}, {cumulative + n_samples}) - {n_samples} samples")
        cumulative += n_samples

    print(f"\nStructural changes between regimes:")
    for k, info in enumerate(change_infos, start=1):
        print(
            f"  Regime {k} -> {k+1}: {info['actual_change_pct']:.1f}% change "
            f"(+{info['n_edges_added']} edges, -{info['n_edges_removed']} edges, "
            f"Jaccard={info['jaccard_edges']:.2f})"
        )

    # Print moment weights
    weights = get_moment_weights(moments)
    print(f"\nMoment weights (factorial-inverse):")
    moment_names = {1: 'Mean', 2: 'Variance', 3: 'Skewness', 4: 'Kurtosis'}
    for m in moments:
        print(f"  {moment_names.get(m, f'M{m}')}: 1/{m}! = {weights[m]:.4f}")

    # Build regime configs with variable lengths and different noise deviations
    regime_configs = []
    for i, (dag, (n_samples, _)) in enumerate(zip(dags, regime_specs)):
        # Vary the noise deviation between regimes
        deviation = 0.5 + rng.uniform(0.0, 1.0)
        signal_strength = 1.2 + rng.uniform(0.0, 0.6)
        regime_configs.append({
            "fixed_graph": dag,
            "deviation": deviation,
            "signal_strength": signal_strength,
            "nsamples": n_samples,
            "dist": ["normal"] * p,
            "regime_seed": seed + i * 50,
        })

    # Generate scenario
    scenario = scenario_gen.create_nonstationary_scenario(
        regime_configs=regime_configs,
        transition_type="abrupt",
    )

    data = scenario["combined_data"].values
    ground_truth = scenario["change_points"]
    T = data.shape[0]

    print(f"\nGround truth ({len(ground_truth)} change points):")
    for i, cp in enumerate(ground_truth):
        print(f"  t={cp}: regime {i+1} -> regime {i+2}")

    # Run multi-moment detector (v1-F)
    # Use first 50% of first regime as baseline (shorter for sensitivity)
    first_regime_len = regime_specs[0][0]
    baseline_idx = np.arange(int(first_regime_len * 0.5))

    # Calculate minimum regime length for refractory period
    min_regime_len = min(spec[0] for spec in regime_specs)
    refractory = min(150, min_regime_len // 4)

    result = detect_nonstationarity_multimoment(
        data, baseline_idx,
        min_scale=3.0,          # Standard min_scale
        moments=moments,
        moment_window=moment_window,
        n_surrogates=100,       # More surrogates for better calibration
        alpha=0.40,             # Permissive threshold (40th percentile)
        k_scales_min=1,         # Minimum scales required
        smooth_window=12,       # Moderate smoothing
        refractory_period=refractory,
        min_snr=0.3,            # Low SNR threshold
        k_channels_min=2,       # v1-F: require 2+ channels
        step_delta_k=1.5,       # Relaxed step threshold for multi-regime sensitivity
        seed=seed,
    )

    # Print detection results
    print(f"\nDetected change points: {result.change_points}")
    print(f"  Onset points: {result.onset_points}")
    print(f"  Offset points: {result.offset_points}")

    # Print events
    print(f"\nEvents detected: {len(result.events)}")
    for i, event in enumerate(result.events):
        ch_info = ""
        if event.n_active_channels is not None:
            ch_info = f", ch_active={event.n_active_channels}/{data.shape[1]}, conc={event.concentration_ratio:.2f}"
        print(f"  Event {i+1}: type={event.event_type}, tau={event.tau}, "
              f"peak_dE={event.peak_dE:.2f}{ch_info}")

    # Print calibration thresholds
    print(f"\nCalibration thresholds:")
    print(f"  eps_on_pos (onset): {result.calibration.eps_on_pos:.2f}")
    print(f"  eps_on_neg (offset): {result.calibration.eps_on_neg:.2f}")

    # Print per-moment contribution at each ground truth change point
    if result.E_by_moment:
        print(f"\nPer-moment contributions at change points:")
        for gt in ground_truth:
            print(f"  At t={gt}:")
            window = 50
            for m in moments:
                if m in result.E_by_moment:
                    E_m_pos, E_m_neg = result.E_by_moment[m]
                    pre_mean = np.mean(E_m_pos[max(0, gt-window):gt] - E_m_neg[max(0, gt-window):gt])
                    post_mean = np.mean(E_m_pos[gt:min(T, gt+window)] - E_m_neg[gt:min(T, gt+window)])
                    delta = post_mean - pre_mean
                    print(f"    {moment_names.get(m, f'M{m}')}: delta={delta:+.2f}")

    # Merge nearby detections — use tolerance window as min_distance to avoid double-counting
    # consecutive onsets/offsets near the same GT (e.g., two detections both within ±125 of GT)
    detected = simplify_transitions(result.change_points, min_distance=160)

    print(f"\nDetection errors (±125 tolerance):")
    for gt in ground_truth:
        closest = min(detected, key=lambda x: abs(x - gt)) if detected else None
        if closest is not None:
            error = abs(gt - closest)
            status = "OK" if error <= 125 else "LARGE ERROR"
            print(f"  GT={gt}, detected={closest}, error={error} samples [{status}]")
        else:
            print(f"  GT={gt}, MISSED")

    # Check for false positives (detections far from any ground truth)
    fps = []
    for det in detected:
        min_dist = min(abs(det - gt) for gt in ground_truth)
        if min_dist > 125:
            fps.append(det)
    if fps:
        print(f"\nPotential false positives: {fps}")

    # Evaluate
    eval_result = evaluate_detection(ground_truth, detected, tolerance=125)

    print(f"\nEvaluation (tolerance=125):")
    print(f"  F1={eval_result.f1_score:.2f}  Precision={eval_result.precision:.2f}  Recall={eval_result.recall:.2f}")
    print(f"  TP={eval_result.true_positives}, FP={eval_result.false_positives}, FN={eval_result.false_negatives}")

    # Visualize
    if show_plots:
        plot_structure_changes(scenario)
        plot_multimoment_results(
            data, result, ground_truth, detected,
            title=f"Realistic Scenario: {n_regimes} regimes, variable lengths",
            tolerance=125,
        )

    print("=" * 70)
    return {
        "scenario": scenario,
        "result": result,
        "eval": eval_result,
        "ground_truth": ground_truth,
        "detected": detected,
    }


def run_multimoment_experiment(
    p: int = 5,
    n_samples: int = 800,
    change_pct: float = 25.0,
    seed: int = 42,
    moments: List[int] = [1, 2, 3, 4],
    moment_window: int = 50,
    show_plots: bool = True,
):
    """
    Run detection using only the multi-moment detector (v1-F).

    Parameters
    ----------
    p : int
        Number of variables
    n_samples : int
        Samples per regime
    change_pct : float
        Structural change percentage
    seed : int
        Random seed
    moments : List[int]
        Which moments to use (1=mean, 2=var, 3=skew, 4=kurt)
    moment_window : int
        Window size for rolling moment computation
    show_plots : bool
        Whether to display plots
    """
    print("=" * 70)
    print("Gatekeeper v1-F: Multi-Moment Non-Stationarity Detection")
    print("=" * 70)

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate scenario
    dag1, dag2, change_info = NonStationaryCausalScenario.create_regime_pair_with_change(
        p=p, base_pconn=0.3, change_pct=change_pct, change_type="mixed", seed=seed,
    )

    scenario_gen = NonStationaryCausalScenario(p=p, mode="linear", seed=seed)

    n1 = int(rng.uniform(0.5, 1.5) * n_samples)
    n2 = int(rng.uniform(0.5, 1.5) * n_samples)

    scenario = scenario_gen.create_nonstationary_scenario(
        regime_configs=[
            {"fixed_graph": dag1, "deviation": 0.6, "signal_strength": 1.5,
             "nsamples": n1, "dist": ["normal"] * p, "regime_seed": seed},
            {"fixed_graph": dag2, "deviation": 0.9, "signal_strength": 1.5,
             "nsamples": n2, "dist": ["normal"] * p, "regime_seed": seed + 100},
        ],
        transition_type="abrupt",
    )

    data = scenario["combined_data"].values
    ground_truth = scenario["change_points"]
    T = data.shape[0]

    print(f"\nScenario: {p} variables, {T} samples ({n1} + {n2} per regime)")
    print(f"Structural change: {change_info['actual_change_pct']:.1f}% "
          f"(+{change_info['n_edges_added']} edges, -{change_info['n_edges_removed']} edges)")

    baseline_idx = np.arange(int(T * 0.20))

    # Print moment weights
    weights = get_moment_weights(moments)
    print(f"\nMoment weights (factorial-inverse):")
    moment_names = {1: 'Mean', 2: 'Variance', 3: 'Skewness', 4: 'Kurtosis'}
    for m in moments:
        print(f"  {moment_names.get(m, f'M{m}')}: 1/{m}! = {weights[m]:.4f}")

    # Run v1-E
    result = detect_nonstationarity_multimoment(
        data, baseline_idx,
        min_scale=4.0,
        moments=moments,
        moment_window=moment_window,
        n_surrogates=75,
        alpha=0.01,
        k_scales_min=3,
        smooth_window=15,
        refractory_period=200,
        min_snr=2.0,
        k_channels_min=2,       # v1-F: require 2+ channels
        seed=seed,
    )

    # Print results
    print(f"\nGround truth: change point at t={ground_truth[0]}")
    print(f"\nDetected change points: {result.change_points}")
    print(f"  Onset: {result.onset_points}, Offset: {result.offset_points}")

    print(f"\nEvents detected: {len(result.events)}")
    for i, event in enumerate(result.events):
        ch_info = ""
        if event.n_active_channels is not None:
            ch_info = f", ch_active={event.n_active_channels}/{data.shape[1]}, conc={event.concentration_ratio:.2f}"
        print(f"  Event {i+1}: type={event.event_type}, tau={event.tau}, "
              f"peak_dE={event.peak_dE:.2f}{ch_info}")

    print(f"\nCalibration thresholds:")
    print(f"  eps_on_pos: {result.calibration.eps_on_pos:.2f}")
    print(f"  eps_on_neg: {result.calibration.eps_on_neg:.2f}")

    # Per-moment analysis
    if result.E_by_moment:
        print(f"\nPer-moment contribution at change point (t={ground_truth[0]}):")
        cp = ground_truth[0]
        window = 50
        for m in moments:
            if m in result.E_by_moment:
                E_m_pos, E_m_neg = result.E_by_moment[m]
                pre_mean = np.mean(E_m_pos[max(0, cp-window):cp] - E_m_neg[max(0, cp-window):cp])
                post_mean = np.mean(E_m_pos[cp:min(T, cp+window)] - E_m_neg[cp:min(T, cp+window)])
                delta = post_mean - pre_mean
                print(f"  {moment_names.get(m, f'M{m}')}: pre={pre_mean:.2f}, post={post_mean:.2f}, delta={delta:.2f}")

    # Evaluate
    detected = simplify_transitions(result.change_points, min_distance=50)
    eval_result = evaluate_detection(ground_truth, detected, tolerance=125)

    print(f"\nEvaluation (tolerance=125):")
    print(f"  F1={eval_result.f1_score:.2f}, Precision={eval_result.precision:.2f}, Recall={eval_result.recall:.2f}")
    print(f"  TP={eval_result.true_positives}, FP={eval_result.false_positives}, FN={eval_result.false_negatives}")

    # Visualize
    if show_plots:
        plot_structure_changes(scenario)
        plot_multimoment_results(
            data, result, ground_truth, detected,
            title=f"v1-E: Multi-Moment Detection ({change_pct}% change)",
            tolerance=125,
        )

    print("=" * 70)
    return {"scenario": scenario, "result": result, "eval": eval_result, "ground_truth": ground_truth, "detected": detected}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run wavelet detector experiments (v1-E Multi-Moment)")
    parser.add_argument("--test", type=str, default="all",
                       choices=["2regime", "multiregime", "realistic", "all"],
                       help="Which test to run (default: all)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots")
    args = parser.parse_args()

    show_plots = not args.no_plots

    if args.test in ["2regime", "all"]:
        # Test 1: 2-regime Multi-Moment (v1-F)
        print("\n" + "=" * 70)
        print("TEST 1: 2-Regime Multi-Moment Detection (v1-F)")
        print("=" * 70 + "\n")
        run_multimoment_experiment(
            p=4, n_samples=800, change_pct=25.0,
            moments=[1, 2, 3, 4], moment_window=50, show_plots=show_plots
        )

    if args.test in ["multiregime", "all"]:
        # Test 2: Multi-regime Multi-Moment (v1-F) - 3 regimes = 2 change points
        print("\n" + "=" * 70)
        print("TEST 2: 3-Regime Multi-Moment Detection (v1-F)")
        print("=" * 70 + "\n")
        run_multi_regime_experiment_multimoment(
            p=4, n_regimes=3, n_samples_per_regime=800, change_pct=25.0,
            moments=[1, 2, 3, 4], moment_window=50, show_plots=show_plots
        )

    if args.test in ["realistic", "all"]:
        # Test 3: Realistic scenario - 5 nodes, 5 regimes, variable lengths
        print("\n" + "=" * 70)
        print("TEST 3: REALISTIC Scenario (5 nodes, 5 regimes, variable lengths)")
        print("=" * 70 + "\n")
        run_realistic_scenario(
            p=5, seed=42,
            moments=[1, 2, 3, 4], moment_window=50, show_plots=show_plots
        )
