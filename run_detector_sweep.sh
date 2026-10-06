#!/usr/bin/env bash
#
# Detector sweep on the 600-scenario validation sample (detection-only rows:
# no bootstrap, no ablation, ~6 min each at 12 workers).
#
# v1J: Fisher-standardised max-pair statistic, threshold sqrt(2 ln m) + gamma.
# gamma=2.0 gave R=0.92 / P=0.65 / 46% over-segmentation on the first 350
# scenarios; this sweeps gamma upward for the precision-recall curve, and adds
# v1J with the detector window sized from a fixed 500 instead of the true
# minimum regime length (does the analytic threshold tolerate a non-oracle W?).
#
#   ./run_detector_sweep.sh
#   GAMMAS="3.0 4.0" ./run_detector_sweep.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-12}"
GAMMAS="${GAMMAS:-2.5 3.0 3.5 4.0}"
FIXEDW_GAMMA="${FIXEDW_GAMMA:-}"      # set to run v1J + fixed W at that gamma
OUT="results/hardening_validation"
IDS="$OUT/sample_ids.txt"
LOG="$OUT/logs"; mkdir -p "$LOG"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

COMMON="--dataset_dir datasets --detector v1J --no-hybrid --chain_prior_threshold 0.05 \
        --bootstrap_n 0 --window_overlap 0.0 --no_ablation \
        --workers $WORKERS --batch_size 10 --resume --scenario_ids_file $IDS"

run_row () {
  local name="$1"; shift
  echo "== $name =="; local t0=$SECONDS
  # shellcheck disable=SC2086
  $PYTHON -u run_experiments.py $COMMON "$@" --output_dir "$OUT/$name" 2>&1 \
     | grep -vE "^  \[" | tee -a "$LOG/$name.log"
  echo "  -> $(( $(wc -l < "$OUT/$name/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
  echo "$name" >> "$OUT/rows_done.txt"
}

for g in $GAMMAS; do
  run_row "D_v1J_g${g}" --corr_gamma "$g"
done
if [ -n "$FIXEDW_GAMMA" ]; then
  run_row "D_v1J_g${FIXEDW_GAMMA}_fixedW500" --corr_gamma "$FIXEDW_GAMMA" --fixed_min_regime_len 500
fi
echo "SWEEP DONE"
