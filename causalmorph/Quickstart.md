# CausalMorph Quick Start Guide

Get started with CausalMorph in 5 minutes! CausalMorph is a data preconditioning algorithm that transforms observational data to make it more compatible with Linear Non-Gaussian Acyclic Model (LiNGAM) assumptions, improving causal discovery accuracy.

## Installation

Clone the repository and install CausalMorph in development mode:

```bash
git clone https://github.com/MarSH-Up/CausalMorph.git
cd CausalMorph
pip install -e .
```

## Basic Usage

This example demonstrates the complete CausalMorph workflow: generating nonlinear data, applying the transformation, and comparing causal discovery results.

```python
import numpy as np
import pandas as pd
from lingam import DirectLiNGAM

# Import CausalMorph components
from causalmorph import causalMorph  # Main transformation algorithm
from causalmorph.data_generation.synthetic_scenarios import causal_graph_synthetic_scenarios
from causalmorph.utils.metrics import normalized_shd, mycomparegraphs

# 1. Generate synthetic nonlinear data
adj_true, data = causal_graph_synthetic_scenarios(
    p=10,                    # Number of variables in the causal graph
    pconn=0.3,              # Probability of connection between variables (30%)
    dist=["laplace"] * 10,  # Noise distribution for each variable (Laplace)
    deviation=0.5,          # Standard deviation of the noise distribution
    signal_strength=2.0,    # Strength of causal effects between variables
    nsamples=1000,          # Number of samples to generate
    seed=42,                # Random seed for reproducibility
    mode="nonlinear",       # Type of causal relationships (linear/nonlinear)
    nonlin_func="tanh",     # Nonlinear function to apply (tanh, sin, cos, etc.)
    nonlinearity=0.75,      # Degree of nonlinearity (0.0=linear, 1.0=fully nonlinear)
)

# 2. Fit DirectLiNGAM on ORIGINAL (untransformed) data
model = DirectLiNGAM()
model.fit(data)
pred_orig = model.adjacency_matrix_  # Predicted adjacency matrix

# Evaluate original results (before CausalMorph transformation)
metrics_orig = mycomparegraphs(pred_orig, adj_true.values)  # Compare predicted vs true graph
shd_original = normalized_shd(adj_true.values, pred_orig)   # Calculate normalized SHD
print(f"Original SHD: {metrics_orig['SHD']}")
print(f"Original Normalized SHD: {shd_original['normalized_shd']:.4f}")

# 3. Apply CausalMorph transformation to precondition the data
transformed = causalMorph(
    data,                                               # Original data to transform
    causal_order=model.causal_order_,                  # Variable ordering from DirectLiNGAM
    adjacency_matrix=pd.DataFrame(adj_true.values),    # True causal graph structure
    verbose=False,                                      # Silent mode (no detailed output)
)

# 4. Fit DirectLiNGAM on TRANSFORMED (preconditioned) data
model_trans = DirectLiNGAM()
model_trans.fit(transformed)
pred_trans = model_trans.adjacency_matrix_  # New predicted adjacency matrix

# Evaluate transformed results (after CausalMorph transformation)
metrics_trans = mycomparegraphs(pred_trans, adj_true.values)  # Compare predicted vs true graph
shd_transformed = normalized_shd(adj_true.values, pred_trans) # Calculate normalized SHD
print(f"\nTransformed SHD: {metrics_trans['SHD']}")
print(f"Transformed Normalized SHD: {shd_transformed['normalized_shd']:.4f}")

# 5. Compare improvement (lower SHD is better)
shd_improvement = shd_original['normalized_shd'] - shd_transformed['normalized_shd']
print(f"\nSHD Improvement: {shd_improvement:.4f}")
print(f"  Original SHD: {shd_original['normalized_shd']:.4f}")
print(f"  Transformed SHD: {shd_transformed['normalized_shd']:.4f}")
print(f"  {'✓ IMPROVED' if shd_improvement > 0 else '✗ WORSE'}")
```

## Run the Example Script

Run the provided example to see CausalMorph in action:

```bash
cd causalmorph
python basic_usage.py
```

This will:
- Generate synthetic nonlinear data with random seed and variables
- Fit DirectLiNGAM on original (untransformed) data
- Apply CausalMorph transformation to precondition the data
- Fit DirectLiNGAM on transformed data
- Compare SHD metrics before and after transformation
- (Optionally) Generate diagnostic plots with `debug=True`

## Next Steps

1. **Explore Notebooks**: See `notebooks/` for interactive Jupyter tutorials and visualizations
2. **Run Benchmarks**: Check `experiments/benchmarks/` for comprehensive performance tests
3. **Read the Paper**: See the manuscript `KNOSYS-D-25-17892.pdf` for theoretical background
4. **Customize**: Experiment with different parameters (p, nonlinearity, nonlin_func) to test various scenarios
5. **Debug Mode**: Set `debug=True` to generate diagnostic plots showing transformation stages

## Key Parameters

### CausalMorph Main Function

```python
causalMorph(
    X,                      # Input DataFrame containing the observational data
    causal_order,          # Variable ordering from causal discovery algorithm (e.g., DirectLiNGAM)
    adjacency_matrix,      # True or estimated causal adjacency matrix (DataFrame)
    verbose=False,         # If True, print detailed progress for each variable transformation
    validate=False,        # If True, show validation metrics (e.g., best non-Gaussian fit)
    debug=False,           # If True, generate and save diagnostic plots for analysis
)
```

### Synthetic Data Generation

```python
causal_graph_synthetic_scenarios(
    p=10,                  # Number of variables in the causal graph
    pconn=0.3,            # Connection probability between pairs of variables (30%)
    nsamples=1000,        # Number of samples (observations) to generate
    mode="nonlinear",     # Type of causal relationships: "linear" or "nonlinear"
    nonlinearity=0.75,    # Degree of nonlinearity: 0.0 (fully linear) to 1.0 (fully nonlinear)
    nonlin_func="tanh",   # Nonlinear function: tanh, log1p, relu, sin, cos, sigmoid
    dist=["laplace"]*p,   # List of noise distributions for each variable (laplace, uniform, etc.)
    deviation=0.5,        # Standard deviation of noise for each variable
    signal_strength=2.0,  # Multiplicative strength of causal effects between variables
    seed=42,              # Random seed for reproducibility
)
```

## Common Issues

### Import Error: "ModuleNotFoundError: No module named 'causalmorph'"
Make sure you're in the correct directory and have installed the package in development mode:
```bash
cd CausalMorph
pip install -e .
```

### Missing Dependencies
If you encounter missing package errors, install all required dependencies:
```bash
pip install -r requirements.txt
```

Common required packages include: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lingam`, `matplotlib`

## Support

- **GitHub**: https://github.com/MarSH-Up/CausalMorph
- **Email**: madlsh3517@gmail.com
- **Paper**: KNOSYS-D-25-17892

## Citation

```bibtex
@article{delossantos2025causalmorph,
  title={CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models},
  author={De Los Santos-Hern{\'a}ndez, Mario and Montero-Hern{\'a}ndez, Samuel and Orihuela-Espina, Felipe and Sucar, L. Enrique},
  journal={Knowledge-Based Systems},
  year={2025},
  note={Manuscript ID: KNOSYS-D-25-17892, Under Review}
}
```
