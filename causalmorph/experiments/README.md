# CausalMorph Experiments

This directory contains scripts for running comprehensive experiments.

## Benchmark Experiments

### Synthetic Data Benchmarks

Run synthetic data benchmarks:

```bash
cd benchmarks
python run_synthetic_experiments.py [output_dir] [num_workers]
```

Parameters:
- `output_dir`: Directory to save results (default: current directory)
- `num_workers`: Number of parallel workers (default: auto-detect)

Example:
```bash
python run_synthetic_experiments.py ../data/results 8
```

This will:
- Generate multiple synthetic datasets with varying configurations
- Run CausalMorph on each dataset
- Compare with baseline DirectLiNGAM/ICALiNGAM
- Save results to CSV file

### Results

Results are saved as `causalmorph_results_YYYYMMDD_HHMMSS.csv` containing:
- Configuration parameters (p, nsamples, mode, nonlinearity, etc.)
- Original metrics (SHD, F1, Precision, Recall, MCC)
- Transformed metrics
- Improvement statistics

## Analysis

Use the `analysis/` directory for analyzing experimental results:
- Statistical tests
- Visualization scripts
- Summary statistics
