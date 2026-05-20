"""
Basic Usage Example for CausalMorph

This example demonstrates how to:
1. Generate synthetic nonlinear data
2. Apply CausalMorph transformation
3. Compare LiNGAM performance before and after transformation
"""



import numpy as np
import pandas as pd
import random
from lingam import DirectLiNGAM

from utils.metrics import mycomparegraphs, normalized_shd
from data_generation.synthetic_scenarios import causal_graph_synthetic_scenarios
from core.causalmorph_algorithm import causalMorph, plot_bijective_demonstration
from utils.statistical_perservation import plot_statistical_preservation


def run_causalmorph_example(p=10, nsamples=1000, seed=42, verbose=True, debug=False, show_bijective=True):
    """
    Run a complete CausalMorph experiment on synthetic nonlinear data.

    Parameters:
        p: Number of variables
        nsamples: Number of samples
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output
        debug: Enable debug mode (generates and saves diagnostic plots)
        show_bijective: Generate bijective transformation demonstration plots

    Returns:
        dict: Results comparing original vs transformed data
    """
    print("=" * 80)
    print(f"CausalMorph Example: Synthetic Nonlinear Data")
    print("=" * 80)

    # ========================================
    # Step 1: Generate synthetic nonlinear data
    # ========================================
    print(f"\nGenerating synthetic data:")
    print(f"  Variables (p): {p}")
    print(f"  Samples (n): {nsamples}")
    print(f"  Seed: {seed}")
    print(f"  Mode: NONLINEAR")

    # Systematically cycle through all available nonlinear functions
    possible_funcs = [ "relu", "sin", "cos", "sigmoid", "tanh", "log1p"]
    # Use seed to deterministically select function (cycles through all functions)
    chosen_func = possible_funcs[seed % len(possible_funcs)]
    print(f"  Nonlinear function: {chosen_func} (function {seed % len(possible_funcs) + 1}/{len(possible_funcs)})")

    adj_true, data = causal_graph_synthetic_scenarios(
        p=p,
        pconn=0.3,
        dist=["laplace"] * p,
        deviation=0.5,
        signal_strength=2.0,
        nsamples=nsamples,
        seed=seed,
        mode="nonlinear",
        nonlin_func=chosen_func,
        nonlinearity=0.75,
        max_attempts=20,
        min_std=1e-6,
        min_edges=5,
    )

    adj_true = adj_true.values.astype(np.int8)
    print(f"\nData shape: {data.shape}")
    print(f"True number of edges: {int(np.sum(adj_true))}")

    # ========================================
    # Step 2: Fit DirectLiNGAM on ORIGINAL data
    # ========================================
    model_orig = DirectLiNGAM()
    model_orig.fit(data)
    pred_orig = model_orig.adjacency_matrix_

    metrics_orig = mycomparegraphs(pred_orig, adj_true)
    shd_original = normalized_shd(adj_true, pred_orig)

    print(f"\nOriginal Results:")
    print(f"  SHD: {metrics_orig['SHD']}")
    print(f"  Normalized SHD: {shd_original['normalized_shd']:.4f}")

    # ========================================
    # Step 3: Apply CausalMorph transformation
    # ========================================
    result = causalMorph(
        data,
        causal_order=model_orig.causal_order_,
        adjacency_matrix=pd.DataFrame(adj_true),
        verbose=verbose,
        validate=False,
        debug=debug,
        return_details=show_bijective or debug,
    )

    # Handle tuple return when return_details=True
    if isinstance(result, tuple):
        transformed, details = result
        print(f"\nTransformation details collected for {len(details)} variables")

        # Generate bijective demonstration plots
        if show_bijective:
            print("\n" + "-" * 40)
            print("Generating bijective transformation plots...")
            print("-" * 40)
            for var_name, var_details in details.items():
                if var_details['residual_before'] is not None and len(var_details['residual_before']) > 0:
                    plot_bijective_demonstration(
                        var_details['residual_before'],
                        var_name=var_name,
                        n_samples=nsamples
                    )
                    # Also generate statistical preservation plot
                    plot_statistical_preservation(
                        var_details['residual_before'],
                        var_name=var_name
                    )
    else:
        transformed = result

    print(f"\nTransformed data shape: {transformed.shape}")

    # ========================================
    # Step 4: Fit DirectLiNGAM on TRANSFORMED data
    # ========================================
    model_trans = DirectLiNGAM()
    model_trans.fit(transformed)
    pred_trans = model_trans.adjacency_matrix_

    metrics_trans = mycomparegraphs(pred_trans, adj_true)
    shd_transformed = normalized_shd(adj_true, pred_trans)

    print(f"\nTransformed Results:")
    print(f"  SHD: {metrics_trans['SHD']}")
    print(f"  Normalized SHD: {shd_transformed['normalized_shd']:.4f}")

    # ========================================
    # Step 5: Compare results
    # ========================================
    print("\n" + "=" * 80)
    print("COMPARISON: Original vs Transformed (using CausalMorph)")
    print("=" * 80)

    shd_improvement = shd_original['normalized_shd'] - shd_transformed['normalized_shd']

    print(f"\nSHD Improvement: {shd_improvement:.4f}")
    print(f"  Original SHD: {shd_original['normalized_shd']:.4f}")
    print(f"  Transformed SHD: {shd_transformed['normalized_shd']:.4f}")
    print(f"  {'✓ IMPROVED' if shd_improvement > 0 else '✗ WORSE'}")

    # Return results
    return {
        "p": p,
        "nsamples": nsamples,
        "seed": seed,
        "nonlinearity": 0.75,
        "mode": "nonlinear",
        "num_edges": int(np.sum(adj_true)),
        "shd_original": shd_original['normalized_shd'],
        "shd_transformed": shd_transformed['normalized_shd'],
        "shd_improvement": shd_improvement,
    }


if __name__ == "__main__":
    # Randomly choose seed and number of variables
    random_seed = random.randint(1, 1000)
    random_p = random.randint(5, 10)
    
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed+12)

    print("\n" + "=" * 80)
    print("CAUSALMORPH BASIC USAGE EXAMPLE")
    print("=" * 80)
    print(f"Random seed: {random_seed}")
    print(f"Number of variables: {random_p}")

    # Run the example
    results = run_causalmorph_example(
        p=random_p,
        nsamples=1000,
        seed=random_seed,
        verbose=False,
        debug=False,  # Set to True to enable debug mode and generate diagnostic plots
        show_bijective=True  # Generate bijective transformation demonstration plots
    )

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
