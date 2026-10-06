#!/usr/bin/env bash
#
# Early validation of docs/2026-10-06_improvement_plan.md on a stratified
# 600-scenario sample (60 per p, 20 per samples-per-regime within each p;
# results/hardening_validation/sample_ids.txt, seed 20261006).
#
# Rows follow §3.3 of the plan. Cheap rows run first so the detection answer
# arrives before the bootstrap rows finish. Each row resumes.
#
#   ./run_hardening_validation.sh            # all rows, 12 workers, v1H windows
#   ROWS="L3_cold C1_oracle_anchor" ./run_hardening_validation.sh
#   DETECTOR=v1K GAMMA=3.0 SUFFIX=_v1K ROWS="L2_cold C3_no_seg L3_cold C1_oracle_anchor C2_oracle_seg L4_soft_chain" \
#       ./run_hardening_validation.sh        # the ladder on v1K windows
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-12}"
DETECTOR="${DETECTOR:-v1H}"       # detector for every row (v1H | v1J | v1K)
GAMMA="${GAMMA:-2.0}"             # corr_gamma for v1J / v1K
SUFFIX="${SUFFIX:-}"              # appended to row names, e.g. _v1K
OUT="results/hardening_validation"
IDS="$OUT/sample_ids.txt"
LOG="$OUT/logs"; mkdir -p "$LOG"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

COMMON="--dataset_dir datasets --no-hybrid --chain_prior_threshold 0.05 --corr_gamma $GAMMA \
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
  "L2_cold          --detector $DETECTOR $NOBOOT --no_ablation"
  "C3_no_seg        --detector $DETECTOR $NOBOOT --no_segmentation"
  "L3_cold          --detector $DETECTOR $BOOT"
  "C1_oracle_anchor --detector $DETECTOR $BOOT --oracle_anchor --no_ablation"
  "C2_oracle_seg    --detector $DETECTOR $BOOT --oracle_segmentation --no_ablation"
  "L4_soft_chain    --detector $DETECTOR $BOOT --chain_soft_lambda 0.7 --consensus_prior_density 0.2 --no_ablation"
)

want="${ROWS:-}"
for def in "${ROW_DEFS[@]}"; do
  name="${def%% *}"; flags="${def#* }"
  if [ -n "$want" ] && ! grep -qw "$name" <<<"$want"; then continue; fi
  out_name="${name}${SUFFIX}"
  echo "== $out_name =="; t0=$SECONDS
  # shellcheck disable=SC2086
  $PYTHON -u run_experiments.py $COMMON $flags --output_dir "$OUT/$out_name" 2>&1 \
     | grep -vE "^  \[" | tee -a "$LOG/$out_name.log"
  echo "  -> $(( $(wc -l < "$OUT/$out_name/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
  echo "$out_name" >> "$OUT/rows_done.txt"
done
echo "ALL DONE"
