# Non-Stationary Causal Scenarios for Benchmarking

This module provides tools for generating controlled non-stationary causal scenarios following Pearl's Structural Causal Model (SCM) framework. It's designed for benchmarking non-stationarity detection algorithms.

## Key Features

### 1. Pearl's SCM Framework
Each variable follows: **X_i = f_i(PA_i, U_i)**
- PA_i: Parents of X_i (determined by DAG structure)
- U_i: Exogenous noise
- f_i: Structural equation (linear or nonlinear)

### 2. Percentage-Based Structural Changes
Control structural changes as a percentage of maximum possible edges:
- For p nodes, max edges = p*(p-1)/2
- Specify target percentage (e.g., 20%, 30%, 40%)
- System automatically rounds to nearest achievable change
- Track exact changes: which edges added/removed

### 3. Consistent Functional Form
- Mode (linear/nonlinear) set at initialization
- Remains constant across ALL regimes
- Only structure and parameters change

## Usage Examples

### Example 1: Simple Percentage-Based Change

```python
from scenarios.NonStationaryCausalScenarios import NonStationaryCausalScenario

# Create 5-node scenario with 30% structural change
dag1, dag2, change_info = NonStationaryCausalScenario.create_regime_pair_with_change(
    p=5,                      # 5 nodes → max 10 edges
    base_pconn=0.3,          # Connection probability for base graph
    change_pct=30.0,         # Target 30% change
    change_type="mixed",     # Add and remove edges
    seed=42
)

print(f"Target: {change_info['target_change_pct']}%")
print(f"Actual: {change_info['actual_change_pct']:.2f}%")
print(f"Changes: +{change_info['n_edges_added']} -{change_info['n_edges_removed']}")
```

### Example 2: Full Scenario with Time Series

```python
# Initialize with linear mode (consistent across regimes)
scenario_gen = NonStationaryCausalScenario(p=5, mode="linear", seed=42)

# Generate DAGs with 30% change
dag1, dag2, change_info = NonStationaryCausalScenario.create_regime_pair_with_change(
    p=5, change_pct=30.0, seed=42
)

# Create regimes with same parameters, different structures
regime_configs = [
    {
        "fixed_graph": dag1,
        "deviation": 0.5,
        "signal_strength": 1.5,
        "nsamples": 500,
        "dist": ["normal"] * 5,
    },
    {
        "fixed_graph": dag2,
        "deviation": 0.5,        # Same
        "signal_strength": 1.5,  # Same
        "nsamples": 500,         # Same
        "dist": ["normal"] * 5,  # Same
    },
]

# Generate scenario
scenario = scenario_gen.create_nonstationary_scenario(regime_configs)

# Plot structures and time series
scenario_gen.plot_scenario(scenario)
```

### Example 3: Benchmarking Multiple Change Levels

```python
# Test detection sensitivity across different change magnitudes
change_levels = [10.0, 20.0, 30.0, 40.0, 50.0]
p = 6  # 6 nodes → max 15 edges

for target_pct in change_levels:
    dag1, dag2, info = NonStationaryCausalScenario.create_regime_pair_with_change(
        p=p,
        change_pct=target_pct,
        seed=42 + int(target_pct)
    )

    print(f"Level {target_pct}%: {info['total_changes']} edges changed")
    # Use dag1, dag2 to create scenario and test detection algorithm
```

## Percentage Rounding Examples

The system automatically rounds to the nearest achievable change:

| Nodes | Max Edges | Target % | Target Edges | Rounded | Actual % |
|-------|-----------|----------|--------------|---------|----------|
| 4     | 6         | 25%      | 1.50         | 2       | 33.33%   |
| 5     | 10        | 30%      | 3.00         | 3       | 30.00%   |
| 6     | 15        | 20%      | 3.00         | 3       | 20.00%   |
| 10    | 45        | 15%      | 6.75         | 7       | 15.56%   |

## Change Types

Three types of structural changes available:

1. **"add"**: Only add new edges
2. **"remove"**: Only remove existing edges
3. **"mixed"**: Both add and remove (50/50 split, default)

## Nonlinear Functions

For nonlinear mode, supported functions include:
- `sigmoid`: σ(x) = 1/(1 + exp(-x))
- `tanh`: tanh(x)
- `cube`: x³
- `softplus`: log(1 + exp(x))
- `elu`: Exponential Linear Unit
- `leaky_relu`: Leaky ReLU

## Files

- `src/scenarios/NonStationaryCausalScenarios.py`: Main class implementation
- `src/scenarios/CausalSyntheticScenarios.py`: Original stationary scenario generator
- `src/scenarios/SyntheticDAGExperiment.py`: DAG experiment utilities
- `src/main.py`: Example usage with percentage-based changes

## For Benchmarking

This framework is ideal for benchmarking because:

1. **Controlled changes**: Know exactly what changed and when
2. **Percentage-based**: Easy to test different magnitudes
3. **Ground truth**: Complete knowledge of DAG structures
4. **Reproducible**: Seeded random generation
5. **Detailed tracking**: Full change metadata available

## Citation

If using this for research, please cite the appropriate causal modeling frameworks:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
