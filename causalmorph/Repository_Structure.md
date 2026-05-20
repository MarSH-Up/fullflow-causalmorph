# CausalMorph Repository Structure

This document provides an overview of the CausalMorph repository organization.

## Directory Tree

```
causalmorph/                       # Main Python package (root)
├── core/                          # Core algorithm implementation
│   ├── __init__.py
│   └── causalmorph_algorithm.py   # Main CausalMorph algorithm
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── metrics.py                 # SHD, F1, Precision, Recall, MCC metrics
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
│   ├── README.md
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
├── Readme.md                      # Main README
├── Install.md                     # Installation guide
├── Quickstart.md                  # Quick start guide
├── Repository_Structure.md        # This file
└── .gitignore                     # Git ignore patterns

```

## Key Files

### Core Package (`causalmorph/`)

1. **`core/causalmorph_algorithm.py`**
   - Main CausalMorph algorithm (from IYCC `Coloring.py`)
   - Functions: `causalMorph()`, `causalMorphing()`
   - Stage I: MDL-guided linearization
   - Stage II: Non-Gaussian synthesis
   - Stage III: Orthogonalization

2. **`utils/metrics.py`**
   - Graph comparison metrics (from IYCC `SHD.py`)
   - Functions: `normalized_shd()`, `mycomparegraphs()`
   - Comprehensive metrics: SHD, F1, Precision, Recall, MCC

3. **`utils/non_gaussian.py`**
   - Non-Gaussianity testing (from IYCC `check_non_gaussian.py`)
   - Function: `check_non_gaussian()`
   - Multiple statistical tests

4. **`data_generation/synthetic_scenarios.py`**
   - Synthetic data generation (from IYCC `simple_scenario.py`)
   - Function: `causal_graph_simple_scenario_v25()`
   - Supports linear and nonlinear modes

### Examples

1. **`basic_usage.py`** (root level)
   - Complete working example (based on IYCC `test_scenario.py`)
   - Demonstrates:
     - Data generation
     - CausalMorph application
     - Result comparison
     - Optional plotting

2. **`examples/README.md`**
   - Documentation for examples directory

### Experiments (`experiments/`)

1. **`benchmarks/run_synthetic_experiments.py`**
   - Comprehensive benchmark suite (based on IYCC `synthetic_pipeline_mac.py`)
   - Parallel processing
   - Configurable experiment grid
   - CSV result output

### Documentation

1. **`Readme.md`** - Overview, installation, quick start, citation
2. **`Install.md`** - Detailed installation instructions
3. **`Quickstart.md`** - Quick start guide
4. **`Repository_Structure.md`** - This file - repository structure documentation
5. **`LICENSE`** - MIT License
6. **`.gitignore`** - Git ignore patterns

### Configuration

1. **`setup.py`** - Package setup and installation
2. **`requirements.txt`** - Python dependencies

## Package Dependencies

### Core Dependencies
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.11.0
- scikit-learn >= 1.3.0
- lingam >= 1.8.0
- statsmodels >= 0.14.0

### Visualization
- matplotlib >= 3.7.0
- seaborn >= 0.13.0
- plotly >= 5.14.0

### Other
- networkx >= 3.0
- tqdm >= 4.65.0
- numba >= 0.57.0

## Code Organization from IYCC

### Mapping from IYCC to CausalMorph

| IYCC Source | CausalMorph Destination | Purpose |
|-------------|------------------------|---------|
| `src/components/Coloring.py` | `core/causalmorph_algorithm.py` | Main algorithm |
| `src/components/SHD.py` | `utils/metrics.py` | Metrics computation |
| `src/components/check_non_gaussian.py` | `utils/non_gaussian.py` | Non-Gaussian tests |
| `src/components/scenarios/simple_scenario.py` | `data_generation/synthetic_scenarios.py` | Data generation |
| `src/test_scenario.py` | `basic_usage.py` | Usage example |
| `src/synthetic_pipeline_mac.py` | `experiments/benchmarks/run_synthetic_experiments.py` | Benchmarks |

## Installation

See `Install.md` for detailed installation instructions.

Quick install:
```bash
git clone https://github.com/MarSH-Up/CausalMorph.git
cd CausalMorph
pip install -e .
```

## Usage

See `Quickstart.md` for a quick start guide.

Basic usage:
```python
from causalmorph import causalMorph
from causalmorph.utils.metrics import normalized_shd, mycomparegraphs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see `LICENSE` file for details.

## Citation

See `Readme.md` for citation information.

## Contact

- **Author**: Mario De Los Santos-Hernández
- **Email**: madlsh3517@gmail.com
- **GitHub**: https://github.com/MarSH-Up/CausalMorph
