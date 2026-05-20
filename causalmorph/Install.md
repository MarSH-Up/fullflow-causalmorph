# Installation Guide for CausalMorph

This guide provides detailed installation instructions for CausalMorph.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- (Optional) Conda for environment management

## Installation Methods

### Method 1: Install from Source (Recommended for Development)

1. Clone the repository:
```bash
git clone https://github.com/MarSH-Up/CausalMorph.git
cd CausalMorph
```

2. Create a virtual environment (recommended):

**Using venv:**
```bash
python -m venv causalmorph_env
source causalmorph_env/bin/activate  # On Windows: causalmorph_env\Scripts\activate
```

**Using conda:**
```bash
conda create -n causalmorph python=3.10
conda activate causalmorph
```

3. Install the package in development mode:
```bash
pip install -e .
```

This will install CausalMorph and all its dependencies.

### Method 2: Install with Development Tools

For developers who want to contribute:

```bash
pip install -e ".[dev]"
```

This installs additional development tools:
- pytest (testing)
- pytest-cov (coverage)
- black (code formatting)
- flake8 (linting)
- jupyter (notebooks)

### Method 3: Manual Installation

If you prefer to install dependencies manually:

1. Install core dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package:
```bash
pip install -e .
```

## Verify Installation

Test that CausalMorph is installed correctly:

```python
import causalmorph
print(causalmorph.__version__)
```

Or run a quick test:

```python
from causalmorph import causalMorph
from causalmorph.utils.metrics import normalized_shd
from causalmorph.data_generation.synthetic_scenarios import causal_graph_simple_scenario_v25

print("✓ CausalMorph installed successfully!")
```

## Platform-Specific Notes

### macOS with Apple Silicon (M1/M2/M3/M4)

For GPU acceleration on Apple Silicon:

```bash
pip install torch>=2.0.0
```

CausalMorph will automatically detect and use MPS (Metal Performance Shaders) for GPU acceleration when available.

### Windows

No special configuration needed. Standard installation works on Windows.

### Linux

No special configuration needed. Standard installation works on Linux.

## Troubleshooting

### ImportError: No module named 'causalmorph'

Make sure you're in the correct virtual environment and have installed the package:
```bash
pip list | grep causalmorph
```

### Dependencies Not Found

Re-install dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Permission Errors on macOS/Linux

Use `--user` flag:
```bash
pip install --user -e .
```

## Next Steps

After installation:

1. **Run Basic Example:**
   ```bash
   python basic_usage.py
   ```

2. **Run Benchmarks:**
   ```bash
   python experiments/benchmarks/run_synthetic_experiments.py
   ```

3. **Explore Documentation:**
   - See `Quickstart.md` for a quick start guide
   - See `Readme.md` for detailed usage information
   - See `Repository_Structure.md` for repository organization

## Updating

To update to the latest version:

```bash
cd CausalMorph
git pull origin main
pip install -e . --upgrade
```

## Uninstall

To uninstall CausalMorph:

```bash
pip uninstall causalmorph
```

## Support

If you encounter issues:
- Check the [GitHub Issues](https://github.com/MarSH-Up/CausalMorph/issues)
- Email: madlsh3517@gmail.com
