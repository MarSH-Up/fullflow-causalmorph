#!/usr/bin/env bash
#
# Benchmark v2: the same 600 validation scenarios (same ids, same p / R /
# samples / density / seed) regenerated under the regimes CausalMorph was
# built for, then detection and extraction on each.
#
#   variant            mechanism   noise     where it sits
#   lin_laplace        linear      laplace   the current grid (already run: D_v1L_g2.5, LB_v1L25_v2)
#   nonlin_laplace     nonlinear   laplace   nonlinear mechanisms, LiNGAM-identifiable noise
#   lin_normal         linear      normal    Gaussian noise: LiNGAM not identifiable
#   nonlin_normal      nonlinear   normal    both violations
#
# Per variant: D_v1L (detection only) and LB (pipeline with CausalMorph + the
# per-window learners on the same windows), cold start, no bootstrap.
#
#   ./run_benchmark_v2.sh
#   VARIANTS="nonlin_normal" ./run_benchmark_v2.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PYTHON="${PYTHON:-python}"; WORKERS="${WORKERS:-12}"
VARIANTS="${VARIANTS:-nonlin_laplace lin_normal nonlin_normal}"
NONLIN="${NONLIN:-0.7}"; NONLIN_FUNC="${NONLIN_FUNC:-cube}"   # cube: the only generator function that is nonlinear at the grid's signal scale
OUT="results/hardening_validation"; IDS="$OUT/sample_ids.txt"; LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

gen_flags () {
  case "$1" in
    nonlin_laplace) echo "--mechanism nonlinear --nonlinearity $NONLIN --nonlin_func $NONLIN_FUNC --noise_dist laplace" ;;
    lin_normal)     echo "--mechanism linear --noise_dist normal" ;;
    nonlin_normal)  echo "--mechanism nonlinear --nonlinearity $NONLIN --nonlin_func $NONLIN_FUNC --noise_dist normal" ;;
    *) echo "unknown variant $1" >&2; exit 1 ;;
  esac
}

run_row () {
  local ds="$1" name="$2"; shift 2
  echo "== $name  ($ds)  $(date +%H:%M) =="; local t0=$SECONDS
  # shellcheck disable=SC2086
  $PYTHON -u run_experiments.py --dataset_dir "$ds" --detector v1L --corr_gamma 2.5 --no-hybrid \
      --chain_prior_threshold 0.05 --bootstrap_n 0 --window_overlap 0.0 \
      --workers "$WORKERS" --batch_size 10 --resume --scenario_ids_file "$IDS" "$@" --output_dir "$OUT/$name" 2>&1 \
     | grep -vE "^  \[" | tee -a "$LOG/$name.log" | grep -E "Final save|succeeded|ERROR|Traceback|error:" || true
  echo "  -> $(( $(wc -l < "$OUT/$name/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
  echo "$name" >> "$OUT/rows_done.txt"
}

for v in $VARIANTS; do
  ds="datasets_v2_$v"
  if [ ! -f "$ds/index.csv" ] || [ "$(( $(wc -l < "$ds/index.csv") - 1 ))" -lt 600 ]; then
    echo "== generating $ds  $(date +%H:%M) =="
    # shellcheck disable=SC2046
    $PYTHON generate_datasets.py --output_dir "$ds" --only_ids "$IDS" $(gen_flags "$v") 2>&1 | tail -2
  fi
  run_row "$ds" "D_v1L_$v" --no_ablation
  run_row "$ds" "LB_$v"
done
echo "V2 DONE"
