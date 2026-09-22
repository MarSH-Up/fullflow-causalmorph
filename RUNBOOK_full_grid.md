# Runbook — full-grid extraction ablation

Closes `Third_Year_Report/results/pending_full_grid_extraction.md`: the detection
side (v1G vs v1H) is already validated on all 18,000 scenarios, the extraction
side (bootstrap + overlap) was only measured on the stratified 180-scenario
subset. This run produces the paired full-grid numbers.

The configuration is frozen. Nothing here is a configuration search.

| | baseline arm | winning arm |
|---|---|---|
| detector | v1H | v1H |
| hybrid | off | off |
| `bootstrap_n` | 0 | 20 |
| `bootstrap_threshold` | — | 0.60 |
| `window_overlap` | 0.0 | 0.15 |
| `chain_prior_threshold` | 0.05 | 0.05 |

---

## 1. Environment

Python 3.11 (the reference env is 3.11.4).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-fullgrid.txt
```

Versions are pinned to the local `IYCC-env` conda environment, the one the
180-scenario subset was produced with. `numpy` is held below 2.0 because
`lingam==1.9.1` and `pandas==2.0.3` were built against the 1.x ABI.

`cdt` and `numba` appear in `causalmorph/` imports but are **not** needed: they
live in `causalmorph/utils/metrics.py` and `synthetic_pipeline_mac.py`, neither
of which the `run_experiments.py` path touches. Verified by inspecting
`sys.modules` after a real scenario.

> The figure-F note in the Presentation repo says the subset was reproduced with
> `bilineal-model-env` (numpy 1.25.0, lingam 1.8.3), not `IYCC-env` (numpy
> 1.26.4, lingam 1.9.1). Both run correctly. If the new numbers must be bit-
> comparable with the stored subset results, confirm which one was actually used
> before pinning.

## 2. Datasets

The run needs `datasets/` with 18,000 scenarios (36,001 files, **19 GB**).

**Option A — copy from the main PC** (preferred, avoids any regeneration risk):

```bash
rsync -a --info=progress2 <main-pc>:/path/to/Full-Flow/datasets/ ./datasets/
```

**Option B — regenerate on the server.** All defaults reproduce the grid:

```bash
python generate_datasets.py --output_dir datasets
```

The grid is `p ∈ 3..12`, `R ∈ 2..7`, samples `{500, 1500, 2500}`, density
`{0.20, 0.35, 0.50, 0.75}`, noise `{0.02, 0.08, 0.20}`, 10 seeds. Note it is
**triangular** in the low-`p` corner — `n_changes` is capped at `p-1`, so `p=3`
only has `R ∈ {2,3}`, `p=4` up to `R=4`, and so on. 150 populated `(p,R,n)`
cells of 120 scenarios each.

Either way, `run_full_grid.sh` checks the digest of `datasets/index.csv` against
the reference grid:

```
sha256  ff34488a472798a2cede676821c01c0afdb7cb006c2ebd0b3e97a4a03b797466
```

A mismatch does not invalidate the run, but it means the scenarios are not the
same draws the subset came from.

Disk: 19 GB for the datasets, ~15 MB for both output CSVs.

## 3. Launch

```bash
mkdir -p results/logs
nohup ./run_full_grid.sh > results/logs/driver.log 2>&1 &
```

Defaults to 12 workers. Overrides:

```bash
WORKERS=24 ./run_full_grid.sh        # more parallelism
ARMS=baseline ./run_full_grid.sh     # one arm only (baseline|winning|both)
DATASET_DIR=/mnt/data/datasets ./run_full_grid.sh
PYTHON=/path/to/python ./run_full_grid.sh
```

The script runs the baseline arm, then the winning arm, then
`compare_full_grid.py`. Outputs:

```
results/v1h_full_baseline/results.csv
results/v1h_full_winning/results.csv
results/full_grid_comparison/{headline,by_p,by_samples,by_regimes}.csv
results/full_grid_comparison/plot_F_values.txt
results/logs/{baseline,winning,comparison,driver}.log
```

## 4. Expected runtime

Cost model fitted on 18 scenarios timed individually on an M4 Max
(10 performance + 4 efficiency cores, 36 GB), median in-sample error 26%:

```
seconds ≈ 0.00086 · p^1.74 · R^1.13 · n^0.65
```

Summed over the grid, for the winning configuration:

| | core-hours | at 12 workers |
|---|---|---|
| baseline (`bootstrap_n=0`) | ~24 | ~2 h |
| winning (`bootstrap_n=20`) | ~109 | ~9 h |
| **total** | **~133** | **~11 h** |

The baseline is 4–6× cheaper because it skips the 20 resamples per window
(measured: 9.5 s vs 40.6 s at p9/R4/n2500; 32 s vs 135 s at p12/R7/n2500).

**This is calibrated on an M4 Max, not the server.** Scale it by the server's
per-core speed. As a healthy-throughput reference on that machine: 8 workers,
winning configuration, 120 scenarios of p=3/R=2/n=500 run in 27 s (~16,000
scenarios/hour at that cost point). If a machine is far below that, stop and
diagnose before committing to the long run — see below.

`run_full_grid.sh` pins the BLAS thread count to 1 per worker. Without it the
same 120 scenarios take 35 s (1.2x), because every worker process otherwise
spawns one BLAS thread per core.

If throughput is an order of magnitude below the reference, the cause is not
thread pinning. Check, in this order:

```bash
sysctl hw.memsize hw.ncpu hw.perflevel0.logicalcpu   # RAM and performance cores
vm_stat | grep -E "Pageouts|Swapins"                  # non-zero and growing = swapping
```

Each worker holds its own numpy/sklearn/lingam interpreter, so 8 workers on a
machine with little RAM will swap and throughput collapses while the CPU still
reads as busy. Use at most one worker per performance core, and fewer if memory
is tight. To separate machine speed from parallelism, time one scenario alone:

```bash
python run_experiments.py --dataset_dir datasets --output_dir /tmp/one \
    --detector v1H --no-hybrid --bootstrap_n 20 --bootstrap_threshold 0.60 \
    --window_overlap 0.15 --chain_prior_threshold 0.05 --workers 1 --p_filter 3
```

A p=3 scenario costs ~1.3 s on the M4 Max. If a single worker is near that, the
machine is fine and the problem is parallelism or memory.

Cost is concentrated in `p ≥ 10` and `R ≥ 6`: `p=12` alone is 24 of the 109
core-hours. The index is ordered by `p` ascending, so the run starts on its
cheapest scenarios — early throughput is **not** representative of the whole
grid and will drop sharply as `p` grows.

To calibrate on the server before committing to the long run, time one arm on a
single node count:

```bash
python run_experiments.py --dataset_dir datasets --output_dir /tmp/cal \
    --detector v1H --no-hybrid --bootstrap_n 20 --bootstrap_threshold 0.60 \
    --window_overlap 0.15 --chain_prior_threshold 0.05 \
    --workers 12 --p_filter 6
```

## 5. Interruptions

Both arms pass `--resume`. `run_experiments.py` **appends** to `results.csv`
every `--batch_size` scenarios (10 here) and drops the batch, so memory holds
one batch rather than the whole result set and the file is written once rather
than rewritten at every checkpoint. On resume it reads back only the
`scenario_id` column, skips those ids and keeps appending to the same file.
Re-running a finished arm is a no-op.

At 18,000 scenarios of ~459 bytes the final file is ~8 MB. Rewriting it whole at
every checkpoint — what the runner did before — would have meant 7.4 GB of
cumulative writes at `batch_size=10` (0.38 GB at 200); appending writes 8 MB.

Error rows carry only the 10 identifying keys while ok rows carry all 78, so
every row is reindexed onto the header before being appended. The header is
locked to the first ok row, and rows stay buffered until one appears.

So after any crash, reboot or OOM, the recovery is simply:

```bash
./run_full_grid.sh
```

Losses are bounded by one batch — at most 10 scenarios.

Note that resume skips a scenario by id regardless of its `status`, so a
scenario that already failed is *not* retried. That is deliberate — these
failures are deterministic, and retrying them on every restart would never
converge. To force a retry, delete those rows from `results.csv` first:

```bash
python - <<'PY'
import pandas as pd
d = pd.read_csv("results/v1h_full_winning/results.csv")
d[d.status == "ok"].to_csv("results/v1h_full_winning/results.csv", index=False)
PY
```

## 6. Monitoring

```bash
tail -f results/logs/winning.log
grep -c , results/v1h_full_winning/results.csv          # rows done (+1 header)
grep "checkpoint" results/logs/winning.log | tail -3
grep -c "ERROR" results/logs/winning.log                 # failed scenarios
```

Failed scenarios are written with `status != "ok"` rather than aborting the run.
`compare_full_grid.py` drops them and reports how many it dropped.

## 7. After the run

1. **Read `results/full_grid_comparison/headline.csv`.** It carries, per metric:
   mean ± SD per arm, paired delta, relative change, per-scenario win/loss
   counts and a Wilcoxon signed-rank p-value — the format acceptance criterion 1
   asks for.
2. **Check the direction sanity check.** The script compares every metric
   against the 180-scenario subset result and prints a `DIRECTION REVERSAL`
   block if any metric flipped. Acceptance criterion 3 says a reversal must be
   investigated, not reported.
3. **Update figure F.** `plot_F_values.txt` holds the `PRECISION` / `ERROR`
   literals ready to paste over lines 18–25 of
   `Third_Year_Report/Presentation/plot_scripts/plot_F_bootstrap_ablation.py`,
   which hard-codes the subset values. Then:
   ```bash
   cd Third_Year_Report/Presentation/plot_scripts && python3 plot_F_bootstrap_ablation.py
   ```
4. **Update Table `tab:winning_config`** in the report from `headline.csv`, and
   the per-dimension curves from `by_p.csv`, `by_samples.csv`, `by_regimes.csv`.
5. **Super-additivity (acceptance criterion 5)** is *not* covered by this run.
   It needs the two isolated arms — bootstrap-only and overlap-only — on the
   full grid as well, i.e. two more runs:
   ```bash
   python run_experiments.py --dataset_dir datasets --output_dir results/v1h_full_2bootstrap \
       --detector v1H --no-hybrid --bootstrap_n 20 --bootstrap_threshold 0.60 \
       --window_overlap 0.0 --chain_prior_threshold 0.05 --workers 12 --batch_size 200 --resume
   python run_experiments.py --dataset_dir datasets --output_dir results/v1h_full_2overlap \
       --detector v1H --no-hybrid --bootstrap_n 0 \
       --window_overlap 0.15 --chain_prior_threshold 0.05 --workers 12 --batch_size 200 --resume
   ```
   Budget roughly another ~11 h at 12 workers for the pair. Decide whether the
   manuscript needs it at full scale or whether the subset evidence suffices.

## 8. Errata found while preparing this run

Both surfaced by running `compare_full_grid.py` against the stored 180-scenario
results, which it reproduces exactly for 3 of the 5 metrics.

- **Report Table 3 / figure F, transition SHD: `−7.4%` does not reconcile.**
  Stored values are 0.1161 → 0.1080, i.e. **−7.0%**. No aggregation convention
  (ratio of means, mean of ratios, medians) yields 7.4%.
- **Dynamic score baseline is `0.217`, not `0.218`** (the known D-02 item), and
  the relative change is **−3.3%**, not −3.4%.

The other three metrics match the report to two decimals, including the
per-scenario win-rates 106/31, 113/34 and 115/33.

## 9. File map

| Path | Role |
|---|---|
| `run_full_grid.sh` | driver — preflight, both arms, comparison |
| `compare_full_grid.py` | paired Δ, win-rates, Wilcoxon, stratifications, figure-F literals |
| `run_experiments.py` | per-scenario runner (`--resume` added for this run) |
| `requirements-fullgrid.txt` | pinned runtime dependencies |
| `results/v1h_phase1/results.csv` | 180-scenario **baseline** reference |
| `results/v1h_winning/results.csv` | 180-scenario **winning** reference |

> `results/v1h_subset/` is an *older* baseline (62 columns, no dynamic metrics,
> per-regime precision 0.324). The compendium's file map calls it the baseline;
> the one that reproduces Table 3 is `v1h_phase1` (78 columns, 0.328).
