"""
Synthetic Benchmark Experiments for CausalMorph

This script runs comprehensive benchmarks on synthetic data to evaluate
CausalMorph's performance across various configurations.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from lingam import ICALiNGAM, DirectLiNGAM
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

from causalmorph import causalMorphing
from causalmorph.utils.metrics import normalized_shd, mycomparegraphs
from causalmorph.data_generation.synthetic_scenarios import causal_graph_simple_scenario_v25


def process_single_experiment(config):
    """
    Run a single experiment with given configuration.

    Parameters:
        config: Dictionary with experiment parameters

    Returns:
        Dictionary with results
    """
    try:
        # Generate synthetic data
        adj_true, data = causal_graph_simple_scenario_v25(
            p=config['p'],
            pconn=config['pconn'],
            dist=config['dist'],
            deviation=config['deviation'],
            signal_strength=config['signal_strength'],
            nsamples=config['nsamples'],
            seed=config['seed'],
            mode=config['mode'],
            nonlin_func=config.get('nonlin_func', 'tanh'),
            nonlinearity=config.get('nonlinearity', 0.0),
            max_attempts=20,
            min_std=1e-6,
            min_edges=config.get('min_edges', 1),
        )

        adj_true = adj_true.values.astype(np.int8)

        # Select LiNGAM algorithm
        if config.get('algorithm', 'ICALiNGAM') == 'ICALiNGAM':
            model = ICALiNGAM(max_iter=20000)
        else:
            model = DirectLiNGAM()

        # Fit on original data
        model.fit(data)
        pred_orig = model.adjacency_matrix_
        metrics_orig = mycomparegraphs(pred_orig, adj_true)
        shd_original = normalized_shd(adj_true, pred_orig)

        # Apply CausalMorph
        transformed = causalMorphing(
            data,
            causal_order=model.causal_order_,
            adjacency_matrix=pd.DataFrame(adj_true),
            verbose=False,
        )

        # Fit on transformed data
        model_trans = ICALiNGAM(max_iter=20000) if config.get('algorithm', 'ICALiNGAM') == 'ICALiNGAM' else DirectLiNGAM()
        model_trans.fit(transformed)
        pred_trans = model_trans.adjacency_matrix_
        metrics_trans = mycomparegraphs(pred_trans, adj_true)
        shd_transformed = normalized_shd(adj_true, pred_trans)

        # Extract values
        shd_orig_value = shd_original["normalized_shd"] if isinstance(shd_original, dict) else shd_original
        shd_trans_value = shd_transformed["normalized_shd"] if isinstance(shd_transformed, dict) else shd_transformed

        # Return results
        return {
            **config,
            "shd_original": shd_orig_value,
            "shd_transformed": shd_trans_value,
            "shd_absolute_original": metrics_orig["SHD"],
            "shd_absolute_transformed": metrics_trans["SHD"],
            "improvement": shd_orig_value - shd_trans_value,
            "f1_original": metrics_orig["F1"],
            "f1_transformed": metrics_trans["F1"],
            "f1_improvement": metrics_trans["F1"] - metrics_orig["F1"],
            "precision_original": metrics_orig["Precision"],
            "precision_transformed": metrics_trans["Precision"],
            "recall_original": metrics_orig["Recall"],
            "recall_transformed": metrics_trans["Recall"],
            "mcc_original": metrics_orig["MCC"],
            "mcc_transformed": metrics_trans["MCC"],
            "num_edges_true": int(np.sum(adj_true)),
            "num_edges_pred_original": int(np.sum(pred_orig)),
            "num_edges_pred_transformed": int(np.sum(pred_trans)),
        }
    except Exception as e:
        print(f"❌ Error in experiment: {e}")
        return None


def parallel_process_experiments(experiments, num_workers=None):
    """
    Process experiments in parallel.

    Parameters:
        experiments: List of experiment configurations
        num_workers: Number of parallel workers (None = auto)

    Returns:
        List of results
    """
    if num_workers is None:
        num_workers = max(1, psutil.cpu_count(logical=False) - 2)

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_experiment, exp): exp for exp in experiments}
        for future in tqdm(as_completed(futures), total=len(experiments), desc="Running experiments"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error in future: {e}")
    return results


def generate_experiment_grid():
    """
    Generate a grid of experiment configurations.

    Returns:
        List of experiment dictionaries
    """
    experiments = []

    # Vary parameters
    p_values = [5, 10, 20]
    nsamples_values = [500, 1000, 2000]
    modes = ["linear", "nonlinear"]
    nonlinearity_values = [0.0, 0.5, 0.75]
    nonlin_funcs = ["tanh", "square", "log1p"]

    for p in p_values:
        for nsamples in nsamples_values:
            for mode in modes:
                for repetition in range(5):  # 5 repetitions per configuration
                    if mode == "linear":
                        config = {
                            "p": p,
                            "pconn": 0.3,
                            "dist": ["laplace"] * p,
                            "deviation": 0.5,
                            "signal_strength": 2.0,
                            "nsamples": nsamples,
                            "seed": 42 + repetition,
                            "mode": mode,
                            "nonlinearity": 0.0,
                            "algorithm": "ICALiNGAM",
                            "repetition": repetition,
                        }
                    else:
                        for nonlinearity in nonlinearity_values:
                            for nonlin_func in nonlin_funcs:
                                config = {
                                    "p": p,
                                    "pconn": 0.3,
                                    "dist": ["laplace"] * p,
                                    "deviation": 0.5,
                                    "signal_strength": 2.0,
                                    "nsamples": nsamples,
                                    "seed": 42 + repetition,
                                    "mode": mode,
                                    "nonlinearity": nonlinearity,
                                    "nonlin_func": nonlin_func,
                                    "algorithm": "ICALiNGAM",
                                    "repetition": repetition,
                                }
                                experiments.append(config)

                    if mode == "linear":
                        experiments.append(config)

    return experiments


def main(output_dir=".", num_workers=None):
    """
    Run complete benchmark experiments.

    Parameters:
        output_dir: Directory to save results
        num_workers: Number of parallel workers (None = auto)
    """
    print("=" * 80)
    print("CausalMorph Synthetic Benchmark Experiments")
    print("=" * 80)

    # Generate experiment configurations
    print("\nGenerating experiment configurations...")
    experiments = generate_experiment_grid()
    print(f"Total experiments: {len(experiments)}")

    # Display system information
    mem_info = psutil.virtual_memory()
    print(f"\nSystem Information:")
    print(f"  Available memory: {mem_info.available / (1024**3):.2f} GB")
    print(f"  CPU cores: {psutil.cpu_count(logical=False)}")

    # Run experiments
    print("\nRunning experiments...")
    results = parallel_process_experiments(experiments, num_workers=num_workers)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"causalmorph_results_{timestamp}.csv")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)

    print(f"\n✅ Experiments completed!")
    print(f"Results saved to: {output_file}")
    print(f"Total successful experiments: {len(results)}")

    # Display summary statistics
    if len(results) > 0:
        print("\nSummary Statistics:")
        print(f"  Mean SHD improvement: {df_results['improvement'].mean():.4f}")
        print(f"  Mean F1 improvement: {df_results['f1_improvement'].mean():.4f}")
        print(f"  % with improved SHD: {(df_results['improvement'] > 0).mean() * 100:.2f}%")
        print(f"  % with improved F1: {(df_results['f1_improvement'] > 0).mean() * 100:.2f}%")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Run experiments
    main(output_dir=output_dir, num_workers=num_workers)
