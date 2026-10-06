#!/usr/bin/env bash
#
# Early validation of docs/2026-10-06_improvement_plan.md on a stratified
# 600-scenario sample (60 per p, 20 per samples-per-regime within each p;
# results/hardening_validation/sample_ids.txt, seed 20261006).
#
# Rows follow §3.3 of the plan. Cheap rows run first so the detection answer
# arrives before the bootstrap rows finish. Each row resumes.
#
#   ./run_hardening_validation.sh            # all rows, 12 workers
#   ROWS="L3_cold C1_oracle_anchor" ./run_hardening_validation.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-12}"
OUT="results/hardening_validation"
IDS="$OUT/sample_ids.txt"
LOG="$OUT/logs"; mkdir -p "$LOG"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

COMMON="--dataset_dir datasets --no-hybrid --chain_prior_threshold 0.05 \
        --workers $WORKERS --batch_size 10 --resume --scenario_ids_file $IDS"
BOOT="--bootstrap_n 20 --bootstrap_threshold 0.60 --window_overlap 0.15"
NOBOOT="--bootstrap_n 0 --window_overlap 0.0"

# name | flags
ROW_DEFS=(
  # detection variants (cheap: no bootstrap, no ablation)
  "D_v1H            --detector v1H $NOBOOT --no_ablation"
  "D_v1H_adaptive   --detector v1H $NOBOOT --no_ablation --adaptive_null"
  "D_v1H_fixedW500  --detector v1H $NOBOOT --no_ablation --fixed_min_regime_len 500"
  "D_v1J            --detector v1J $NOBOOT --no_ablation"
  # extraction ladder and controls (v1H windows throughout)
  "L2_cold          --detector v1H $NOBOOT --no_ablation"
  "C3_no_seg        --detector v1H $NOBOOT --no_segmentation"
  "L3_cold          --detector v1H $BOOT"
  "C1_oracle_anchor --detector v1H $BOOT --oracle_anchor --no_ablation"
  "C2_oracle_seg    --detector v1H $BOOT --oracle_segmentation --no_ablation"
  "L4_soft_chain    --detector v1H $BOOT --chain_soft_lambda 0.7 --consensus_prior_density 0.2 --no_ablation"
)

want="${ROWS:-}"
for def in "${ROW_DEFS[@]}"; do
  name="${def%% *}"; flags="${def#* }"
  if [ -n "$want" ] && ! grep -qw "$name" <<<"$want"; then continue; fi
  echo "== $name =="; t0=$SECONDS
  # shellcheck disable=SC2086
  $PYTHON -u run_experiments.py $COMMON $flags --output_dir "$OUT/$name" 2>&1 \
     | grep -vE "^  \[" | tee -a "$LOG/$name.log"
  echo "  -> $(( $(wc -l < "$OUT/$name/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
  echo "$name" >> "$OUT/rows_done.txt"
done
echo "ALL DONE"
