#!/usr/bin/env bash
#
# Extraction sweep on the 600-scenario sample, v1L gamma 2.5 windows, cold
# start, corrected adjacency convention. Separates the two extraction
# mechanisms that the old "winning configuration" bundled:
#
#   X_overlap_only     bootstrap 0,  overlap 0.15
#   X_boot_t030        bootstrap 20, threshold 0.30, overlap 0.15
#   X_boot_t040        bootstrap 20, threshold 0.40, overlap 0.15
#   X_boot_t050        bootstrap 20, threshold 0.50, overlap 0.15
#
# Reference rows already on disk: L2_cold_v1L25_v2 (bootstrap 0, overlap 0)
# and L3_cold_v1L25_v2 (bootstrap 20, threshold 0.60, overlap 0.15).
#
#   ./run_bootstrap_sweep.sh
#   THRESHOLDS="0.4" ./run_bootstrap_sweep.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PYTHON="${PYTHON:-python}"; WORKERS="${WORKERS:-12}"
DETECTOR="${DETECTOR:-v1L}"; GAMMA="${GAMMA:-2.5}"; SUFFIX="${SUFFIX:-_v1L25_v2}"
THRESHOLDS="${THRESHOLDS:-0.30 0.40 0.50}"
OUT="results/hardening_validation"; LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
COMMON="--dataset_dir datasets --detector $DETECTOR --corr_gamma $GAMMA --no-hybrid \
        --chain_prior_threshold 0.05 --no_ablation \
        --workers $WORKERS --batch_size 10 --resume --scenario_ids_file $OUT/sample_ids.txt"

run_row () {
  local name="$1"; shift
  echo "== $name  $(date +%H:%M) =="; local t0=$SECONDS
  # shellcheck disable=SC2086
  $PYTHON -u run_experiments.py $COMMON "$@" --output_dir "$OUT/$name" 2>&1 \
     | grep -vE "^  \[" | tee -a "$LOG/$name.log" | grep -E "Final save|succeeded|ERROR|Traceback|error:" || true
  echo "  -> $(( $(wc -l < "$OUT/$name/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
  echo "$name" >> "$OUT/rows_done.txt"
}

run_row "X_overlap_only${SUFFIX}" --bootstrap_n 0 --window_overlap 0.15
for t in $THRESHOLDS; do
  tag="$(printf '%03d' "$(echo "$t * 100 / 1" | bc)")"
  run_row "X_boot_t${tag}${SUFFIX}" --bootstrap_n 20 --bootstrap_threshold "$t" --window_overlap 0.15
done
echo "XSWEEP DONE"
