"""
Main example: Percentage-based structural changes for benchmarking non-stationarity detection.

This example demonstrates:
- Controlled structural changes based on percentage
- Automatic rounding to nearest achievable change
- Detailed tracking of what changed
"""

from scenarios.NonStationaryCausalScenarios import NonStationaryCausalScenario


def demo_percentage_changes():
    """
    Demonstrate how percentage-based structural changes work for different numbers of nodes.
    """
    print("=" * 80)
    print("Percentage-Based Structural Changes - Examples")
    print("=" * 80)
    print("\nShowing how target percentages map to actual achievable changes:\n")

    # Test different node counts and target percentages
    test_cases = [
        (4, 25.0),  # 4 nodes, 25% change
        (5, 30.0),  # 5 nodes, 30% change
        (6, 20.0),  # 6 nodes, 20% change
        (10, 15.0), # 10 nodes, 15% change
    ]

    for p, target_pct in test_cases:
        max_edges = p * (p - 1) // 2
        target_n_changes = round((target_pct / 100.0) * max_edges)
        actual_pct = (target_n_changes / max_edges) * 100.0

        print(f"p={p} nodes (max {max_edges} edges):")
        print(f"  Target: {target_pct:.1f}% → {target_pct/100 * max_edges:.2f} edges")
        print(f"  Rounded: {target_n_changes} edges → Actual: {actual_pct:.2f}%")
        print()


def main():
    print("\n" + "=" * 80)
    print("Non-Stationary Causal Scenario: Controlled Percentage Change")
    print("=" * 80)

    # Parameters
    p = 4  # Number of nodes
    target_change_pct = 25.0  # Target 30% structural change
    change_type = "mixed"  # Add and remove edges

    print(f"\nSetup:")
    print(f"  - {p} variables (V1, ..., V{p})")
    print(f"  - Target structural change: {target_change_pct}%")
    print(f"  - Change type: {change_type} (add + remove edges)")
    print(f"  - Linear SCM (Pearl's framework)")
    print(f"  - Constant parameters across regimes\n")

    # Generate base and changed graphs using the helper function
    dag1, dag2, change_info = NonStationaryCausalScenario.create_regime_pair_with_change(
        p=p,
        base_pconn=0.3,
        change_pct=target_change_pct,
        change_type=change_type,
        seed=42,
    )

    # Print detailed change information
    print("Structural Change Details:")
    print(f"  Max possible edges: {change_info['max_possible_edges']}")
    print(f"  Target change: {change_info['target_change_pct']:.1f}%")
    print(f"  Actual change: {change_info['actual_change_pct']:.2f}%")
    print(f"  Total changes: {change_info['total_changes']} edges")
    print(f"    - Edges added: {change_info['n_edges_added']}")
    print(f"    - Edges removed: {change_info['n_edges_removed']}")
    print()

    print("Base graph (Regime 1):")
    print(f"  Edges: {sorted(dag1.edges())}")
    print(f"  Count: {change_info['base_n_edges']}")
    print()

    print("Changed graph (Regime 2):")
    print(f"  Edges: {sorted(dag2.edges())}")
    print(f"  Count: {change_info['new_n_edges']}")
    print()

    if change_info['edges_added']:
        print(f"Added edges: {sorted(change_info['edges_added'])}")
    if change_info['edges_removed']:
        print(f"Removed edges: {sorted(change_info['edges_removed'])}")
    print()

    # Initialize scenario generator
    scenario_gen = NonStationaryCausalScenario(
        p=p,
        mode="linear",
        seed=42,
    )

    # Create regime configurations
    regime_configs = [
        {
            "fixed_graph": dag1,
            "deviation": 0.5,
            "signal_strength": 1.5,
            "nsamples": 500,
            "dist": ["normal"] * p,
            "regime_seed": 100,
        },
        {
            "fixed_graph": dag2,
            "deviation": 0.5,
            "signal_strength": 1.5,
            "nsamples": 500,
            "dist": ["normal"] * p,
            "regime_seed": 200,
        },
    ]

    # Generate scenario
    print("Generating scenario...")
    scenario = scenario_gen.create_nonstationary_scenario(
        regime_configs=regime_configs,
        transition_type="abrupt",
    )

    print(f"Structural change occurs at sample index: {scenario['change_points'][0]}")
    print("\nPlotting results...\n")

    # Plot scenario
    scenario_gen.plot_scenario(
        scenario,
        structures_figsize=(12, 5),
        timeseries_figsize=(14, 8),
    )

    return scenario, change_info


def benchmark_different_change_levels():
    """
    Create scenarios with different change levels for benchmarking.
    This demonstrates how you can systematically test different change magnitudes.
    """
    print("\n" + "=" * 80)
    print("Benchmark: Testing Multiple Change Levels")
    print("=" * 80)

    p = 6  # 6 nodes → max 15 edges
    change_levels = [10.0, 20.0, 30.0, 40.0, 50.0]  # Different change percentages

    print(f"\nGenerating scenarios with p={p} nodes (max {p*(p-1)//2} edges)")
    print(f"Testing change levels: {change_levels}%\n")

    results = []

    for target_pct in change_levels:
        dag1, dag2, change_info = NonStationaryCausalScenario.create_regime_pair_with_change(
            p=p,
            base_pconn=0.3,
            change_pct=target_pct,
            change_type="mixed",
            seed=42 + int(target_pct),  # Different seed for each
        )

        print(f"Change level {target_pct}%:")
        print(f"  Target: {target_pct}% → Actual: {change_info['actual_change_pct']:.2f}%")
        print(f"  Changes: +{change_info['n_edges_added']} -{change_info['n_edges_removed']} "
              f"= {change_info['total_changes']} total")
        print(f"  Edges: {change_info['base_n_edges']} → {change_info['new_n_edges']}")

        results.append({
            'target_pct': target_pct,
            'change_info': change_info,
            'dag1': dag1,
            'dag2': dag2,
        })

    print("\nThese scenarios can be used to benchmark non-stationarity detection methods.")
    print("The controlled change percentages allow you to test detection sensitivity.\n")

    return results


if __name__ == "__main__":
    # First, show how percentage changes are rounded
    demo_percentage_changes()

    # Run main example with specific percentage change
    scenario, change_info = main()

    # Optionally: demonstrate benchmark use case
    # Uncomment to see multiple change levels:
    # benchmark_results = benchmark_different_change_levels()
