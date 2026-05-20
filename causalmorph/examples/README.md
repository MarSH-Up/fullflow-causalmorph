# CausalMorph Examples

This directory contains example scripts demonstrating how to use CausalMorph.

## Basic Usage

Run the basic usage example:

```bash
python basic_usage.py
```

This example demonstrates:
1. Generating synthetic nonlinear data
2. Fitting DirectLiNGAM on original data
3. Applying CausalMorph transformation
4. Fitting DirectLiNGAM on transformed data
5. Comparing results

## Expected Output

The script will:
- Generate synthetic data with 10 variables and 1000 samples
- Show metrics (SHD, F1, Precision, Recall, MCC) for both original and transformed data
- Display improvement statistics
- Optionally generate diagnostic plots (set `plot_stages=True`)

## Customization

You can modify the parameters in `basic_usage.py`:
- `p`: Number of variables
- `nsamples`: Number of samples
- `seed`: Random seed
- `plot_stages`: Whether to generate diagnostic plots
