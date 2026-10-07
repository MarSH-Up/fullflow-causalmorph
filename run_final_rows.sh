#!/usr/bin/env bash
#
# Final rows for the before/after figures, on the 600-scenario sample, v1L
# gamma 2.5 windows, cold start, no bootstrap, no overlap, corrected
# adjacency convention (the best extraction found in §10-§11 of the plan).
#
#   LB_v1L25_v2            pipeline + DirectLiNGAM / ICA-LiNGAM per window (ablation on)
#   C2_oracle_seg_v1L25_v2 extraction on the true change points (ceiling)
#   C3_no_seg_v1L25_v2     one window over the whole series, ablation on
#                          (the "ignore non-stationarity" control, every learner)
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PYTHON="${PYTHON:-python}"; WORKERS="${WORKERS:-12}"
DETECTOR="${DETECTOR:-v1L}"; GAMMA="${GAMMA:-2.5}"; SUFFIX="${SUFFIX:-_v1L25_v2}"
OUT="results/hardening_validation"; LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
COMMON="--dataset_dir datasets --detector $DETECTOR --corr_gamma $GAMMA --no-hybrid \
        --chain_prior_threshold 0.05 --bootstrap_n 0 --window_overlap 0.0 \
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

run_row "LB${SUFFIX}"
run_row "C2_oracle_seg${SUFFIX}" --oracle_segmentation --no_ablation
run_row "C3_no_seg${SUFFIX}" --no_segmentation
echo "FINAL ROWS DONE"
