#!/usr/bin/env bash
#
# L3 (bootstrap 20, cold start, no ablation) on the 600-scenario sample with
# several detectors, to pick the detector by the structure metrics it serves
# rather than by detection F1. Rows are named L3_cold_<suffix>.
#
#   ./run_l3_head_to_head.sh
#   CANDIDATES="v1L:2.0:v1L20 v1L:3.0:v1L30" ./run_l3_head_to_head.sh
#
# Written as bash on purpose: the first attempt ran under zsh, where an
# unquoted `set -- $spec` does not word-split, and silently launched nothing.
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PYTHON="${PYTHON:-python}"; WORKERS="${WORKERS:-12}"
CANDIDATES="${CANDIDATES:-v1J:2.5:v1J25 v1L:2.0:v1L20 v1L:2.5:v1L25}"
OUT="results/hardening_validation"; LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
COMMON="--dataset_dir datasets --no-hybrid --chain_prior_threshold 0.05 \
        --bootstrap_n 20 --bootstrap_threshold 0.60 --window_overlap 0.15 --no_ablation \
        --workers $WORKERS --batch_size 10 --resume --scenario_ids_file $OUT/sample_ids.txt"

for cand in $CANDIDATES; do
  IFS=: read -r det gamma sfx <<<"$cand"
  name="L3_cold_${sfx}"
  echo "== $name  (detector $det, gamma $gamma)  $(date +%H:%M) =="; t0=$SECONDS
  # shellcheck disable=SC2086
  $PYTHON -u run_experiments.py $COMMON --detector "$det" --corr_gamma "$gamma" --output_dir "$OUT/$name" 2>&1 \
     | grep -vE "^  \[" | tee -a "$LOG/$name.log" | grep -E "Final save|succeeded|ERROR|Traceback|error:" || true
  echo "  -> $(( $(wc -l < "$OUT/$name/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
  echo "$name" >> "$OUT/rows_done.txt"
done
echo "H2H DONE"
