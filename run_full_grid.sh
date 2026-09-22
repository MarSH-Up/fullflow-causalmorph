#!/usr/bin/env bash
#
# Full-grid extraction ablation: baseline vs bootstrap+overlap over the
# 18,000-scenario grid, then the paired comparison.
#
# Safe to re-run: both arms pass --resume, so an interrupted run continues from
# its own results.csv instead of recomputing. Re-running after completion is a
# no-op for that arm.
#
#   ./run_full_grid.sh                 # 12 workers, datasets/
#   WORKERS=24 ./run_full_grid.sh      # override parallelism
#   ARMS=baseline ./run_full_grid.sh   # run one arm only (baseline|winning|both)
#
# Detached:
#   nohup ./run_full_grid.sh > results/logs/driver.log 2>&1 &
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-10}"
DATASET_DIR="${DATASET_DIR:-datasets}"
ARMS="${ARMS:-both}"
LOG_DIR="results/logs"

# Frozen winning configuration — do not change without re-running both arms.
DETECTOR="v1H"
BOOTSTRAP_N=20
BOOTSTRAP_THRESHOLD=0.60
WINDOW_OVERLAP=0.15
CHAIN_PRIOR_THRESHOLD=0.05

# Each worker is its own process, so an unpinned BLAS spawns one thread per core
# inside every one of them and they fight over the same cores. Measured on an
# M4 Max, 8 workers, 120 scenarios: 35 s unpinned vs 27 s pinned (1.2x).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_DIR"

# ── preflight ─────────────────────────────────────────────────────────────────

echo "== preflight =="

if [ ! -f "$DATASET_DIR/index.csv" ]; then
    echo "ERROR: $DATASET_DIR/index.csv not found."
    echo "  Generate the grid first (defaults reproduce the 18,000-scenario grid):"
    echo "    $PYTHON generate_datasets.py --output_dir $DATASET_DIR"
    exit 1
fi

N_SCENARIOS=$(($(wc -l < "$DATASET_DIR/index.csv") - 1))
echo "  scenarios in index : $N_SCENARIOS"
if [ "$N_SCENARIOS" -ne 18000 ]; then
    echo "  WARNING: expected 18000. This is not the full grid."
fi

# The grid the 180-scenario subset was drawn from. A different digest means the
# datasets were regenerated; results stay valid but are not bit-comparable.
EXPECTED_SHA="ff34488a472798a2cede676821c01c0afdb7cb006c2ebd0b3e97a4a03b797466"
ACTUAL_SHA=$($PYTHON - "$DATASET_DIR/index.csv" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)
if [ "$ACTUAL_SHA" = "$EXPECTED_SHA" ]; then
    echo "  index.csv digest   : matches the reference grid"
else
    echo "  index.csv digest   : DIFFERS from the reference grid"
    echo "    expected $EXPECTED_SHA"
    echo "    actual   $ACTUAL_SHA"
fi

$PYTHON - <<'PY'
import importlib, sys
missing = [m for m in ("numpy", "pandas", "scipy", "sklearn", "networkx", "lingam", "matplotlib")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"ERROR: missing packages {missing}. pip install -r requirements-fullgrid.txt")
import numpy, pandas
print(f"  python             : {sys.version.split()[0]}")
print(f"  numpy/pandas       : {numpy.__version__} / {pandas.__version__}")
PY

echo "  workers            : $WORKERS"
echo "  batch size         : $BATCH_SIZE"
echo

# ── arms ──────────────────────────────────────────────────────────────────────

run_arm () {
    local name="$1"; shift
    local out="results/v1h_full_${name}"
    local log="$LOG_DIR/${name}.log"

    echo "== $name =="
    echo "  output : $out/results.csv"
    echo "  log    : $log"
    local t0=$SECONDS

    $PYTHON -u run_experiments.py \
        --dataset_dir "$DATASET_DIR" \
        --output_dir "$out" \
        --detector "$DETECTOR" \
        --no-hybrid \
        --chain_prior_threshold "$CHAIN_PRIOR_THRESHOLD" \
        --workers "$WORKERS" \
        --batch_size "$BATCH_SIZE" \
        --resume \
        "$@" 2>&1 | tee -a "$log"

    local rows
    rows=$(($(wc -l < "$out/results.csv") - 1))
    echo "  -> $rows rows in $(( (SECONDS - t0) / 60 )) min"
    echo
}

if [ "$ARMS" = "both" ] || [ "$ARMS" = "baseline" ]; then
    run_arm baseline --bootstrap_n 0 --window_overlap 0.0
fi

if [ "$ARMS" = "both" ] || [ "$ARMS" = "winning" ]; then
    run_arm winning \
        --bootstrap_n "$BOOTSTRAP_N" \
        --bootstrap_threshold "$BOOTSTRAP_THRESHOLD" \
        --window_overlap "$WINDOW_OVERLAP"
fi

# ── comparison ────────────────────────────────────────────────────────────────

if [ "$ARMS" = "both" ]; then
    echo "== paired comparison =="
    $PYTHON compare_full_grid.py \
        --baseline results/v1h_full_baseline/results.csv \
        --winning  results/v1h_full_winning/results.csv \
        --out      results/full_grid_comparison 2>&1 | tee "$LOG_DIR/comparison.log"
fi

echo "Done."
