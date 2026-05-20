---
title: "CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models"
---

# CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models

> **_This README uses mathjax for rendering equations. For best results, view on GitHub with math support, or paste into a markdown viewer with math support (e.g., Jupyter, Typora)._**

[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/Mdls35/causal_synthetic_scenarios)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of CausalMorph, a data preconditioning algorithm that projects observational datasets toward the Linear Non-Gaussian Acyclic Model (LiNGAM) compatible regime.

**Paper**: CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models  
**Authors**: Mario De Los Santos-Hernández, Samuel Montero-Hernández, Felipe Orihuela-Espina, L. Enrique Sucar  
**Journal**: Knowledge-Based Systems (Under Review)  
**Manuscript ID**: KNOSYS-D-25-17892  

## Overview

The Linear Non-Gaussian Acyclic Model (LiNGAM) family provides a unique advantage in causal discovery: unlike most methods, LiNGAM can identify a single, fully directed causal graph from purely observational data rather than an equivalence class of possible structures. However, LiNGAM's strict assumptions of **linearity** and **non-Gaussian noise** are often violated in practice, limiting its applicability.

CausalMorph addresses this challenge through a three-stage data preconditioning process:

- **Stage I: Local Linearization** — MDL-guided polynomial approximation with Taylor expansion
- **Stage II: Non-Gaussian Synthesis** — Whitening-recoloring with explicit non-Gaussian residuals
- **Stage III: Orthogonalization** — Enforcement of statistical uncorrelatedness between noise and parents

### Key Results

Across 17,280 unique synthetic configurations (34,560 total runs), CausalMorph achieves:

- 37.7% relative reduction in Structural Hamming Distance (SHD) for DirectLiNGAM ($p < 0.001$)
- 16.4% relative reduction in SHD for ICALiNGAM ($p < 0.001$)
- Regularization effect: Improved performance even when LiNGAM conditions are met

## Dataset

We provide and recommend the use of a comprehensive set of synthetic data scenarios for causal discovery benchmarks. You can access pretrained synthetic datasets used in our experiments via the following Hugging Face dataset:

- [Synthetic Causal Scenarios @ Hugging Face](https://huggingface.co/datasets/Mdls35/causal_synthetic_scenarios)

To use this dataset in Python, you can do:
```python
from datasets import load_dataset

# Load all synthetic scenarios
ds = load_dataset("Mdls35/causal_synthetic_scenarios", split="train")
print(ds)
```
This resource includes realistic and diverse linear/nonlinear, (non-)Gaussian, and (non-)acyclic ground-truth scenarios.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Basic Installation

```bash
git clone https://github.com/MarSH-Up/CausalMorph.git
cd CausalMorph
pip install -r requirements.txt
```

### Development Installation

```bash
git clone https://github.com/MarSH-Up/CausalMorph.git
cd CausalMorph
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from causalmorph import causalMorph
from lingam import DirectLiNGAM

# Load your observational data (you can also use the Hugging Face dataset above)
data = pd.read_csv('your_data.csv')

# Step 1: Get initial causal order estimate
# (can be from domain knowledge, heuristics, or initial LiNGAM run)
model_initial = DirectLiNGAM()
model_initial.fit(data)
tentative_order = model_initial.causal_order_

# Step 2: Apply CausalMorph transformation
data_transformed = causalMorph(
    data,
    causal_order=tentative_order,
    verbose=True
)

# Step 3: Run LiNGAM on the transformed data
model_final = DirectLiNGAM()
model_final.fit(data_transformed)

# Get the improved adjacency matrix
adjacency_matrix = model_final.adjacency_matrix_
```

For more detailed examples, see `basic_usage.py` or `Quickstart.md`.

## Repository Structure

```
causalmorph/                       # Main Python package (root)
├── core/                          # Core algorithm implementation
│   ├── __init__.py
│   └── causalmorph_algorithm.py   # Complete CausalMorph algorithm
│                                  # (Stage I, II, III)
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── metrics.py                 # SHD, F1, Precision, Recall, MCC
│   └── non_gaussian.py            # Non-Gaussianity testing
│
├── data_generation/               # Synthetic data generation
│   ├── __init__.py
│   └── synthetic_scenarios.py     # Scenario generators
│
├── tests/                         # Unit tests
│   └── __init__.py
│
├── experiments/                   # Experimental code
│   ├── benchmarks/                # Benchmark experiments
│   │   └── run_synthetic_experiments.py
│   ├── synthetic_data/            # Data generation scripts
│   └── analysis/                  # Result analysis scripts
│
├── examples/                      # Usage examples
│   └── README.md
│
├── __init__.py                    # Package initialization
├── basic_usage.py                 # Basic usage demonstration
├── setup.py                       # Package installation
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
├── Readme.md                      # This file
├── Install.md                     # Installation guide
├── Quickstart.md                  # Quick start guide
├── Repository_Structure.md        # Detailed structure documentation
└── .gitignore                     # Git ignore patterns
```

For detailed file descriptions and usage information, see `Repository_Structure.md`.

## Methodology

> All equations below are rendered using MathJax. Inline math is written as `$...$`, and display math with `$$...$$`.

### Stage I: MDL-Guided Local Linearization

For each variable $X_i$ with tentative parent set $pa(X_i)$:

- Fit polynomial $\hat{p}(\cdot)$ to standardized parent data $X_{pa(i)}^{\text{scaled}}$
- Select optimal degree $d^*$ using the Minimum Description Length (MDL) criterion:

  $$
  \mathrm{MDL}(d) = n \log(MSE_d + \epsilon_{\log}) + \lambda k_d
  $$

  where:

    - $n$ = sample size  
    - $MSE_d$ = mean squared error for degree $d$  
    - $k_d$ = number of polynomial terms  
    - $\lambda$ = penalty parameter (e.g., $\lambda=2.0$)  
    - $\epsilon_{\log}$ = small constant (e.g., $10^{-10}$) for stability

- Compute local linear approximation via first-order Taylor expansion at anchor point $x_0$ (coordinate-wise median of the standardized parents):

$$
x_{i,\mathrm{lin}} = \hat{p}(x_0) + \big(X_{pa(i)}^{\text{scaled}} - x_0\big)\nabla\hat{p}(x_0)
$$

  The gradient $\nabla\hat{p}(x_0)$ is estimated numerically (e.g., via finite differences).

### Stage II: Synthetic Non-Gaussian Residuals

- Extract original residual: $\epsilon_{\mathrm{orig}} = x_i - x_{i,\mathrm{lin}}$
- Whiten residual and obtain coloring matrix $C$
- Sample from non-Gaussian distributions (Laplace, Uniform, Exponential, Student’s $t$)
- Recolor: $\epsilon_{\mathrm{synth}} = C Z_{\mathrm{cand}}$
- Select the candidate with the minimum p-value (most non-Gaussian) in a normality test

### Stage III: Orthogonalization and Variance Matching

- Compute orthonormal basis $Q$ for the parent space
- Orthogonalize: $\epsilon_{\mathrm{ortho}} = \epsilon_{\mathrm{synth}} - QQ^{\top}\epsilon_{\mathrm{synth}}$
- Scale to match original variance:

$$
x_{i,\mathrm{new}} = x_{i,\mathrm{lin}} + \epsilon_{\mathrm{ortho}} \cdot \frac{\sigma(\epsilon_{\mathrm{orig}})}{\sigma(\epsilon_{\mathrm{ortho}})}
$$

## Citation

If you use CausalMorph in your research, please cite:

```bibtex
@article{delossantos2025causalmorph,
  title={CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models},
  author={De Los Santos-Hern{\'a}ndez, Mario and Montero-Hern{\'a}ndez, Samuel and Orihuela-Espina, Felipe and Sucar, L. Enrique},
  journal={Knowledge-Based Systems},
  year={2025},
  note={Manuscript ID: KNOSYS-D-25-17892, Under Review}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

- Mario De Los Santos-Hernández: madlsh3517@gmail.com
- Project Repository: https://github.com/MarSH-Up/CausalMorph

## Acknowledgments

- Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE), Puebla, México
- School of Computer Science, University of Birmingham, United Kingdom