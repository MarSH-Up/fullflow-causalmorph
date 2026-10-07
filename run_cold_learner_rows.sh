#!/usr/bin/env bash
#
# Chain-free pipeline and CausalMorph + ICA-LiNGAM, on the four benchmark
# cells (same 600 scenario ids, v1L gamma 2.5 windows, no bootstrap, no
# overlap, seeded CausalMorph). Ablation baselines for the same windows are
# already in the LB_* rows, so these run with --no_ablation.
#
#   P_cold_<cell>       CausalMorph + DirectLiNGAM, cold prior every window (no chain)
#   P_cold_ica_<cell>   CausalMorph + ICA-LiNGAM,   cold prior every window
#   P_chain_ica_<cell>  CausalMorph + ICA-LiNGAM,   chain prior (isolates the learner)
#   P_v2_<cell>, P_v2_ica_<cell>      same as P_cold_*, with --transform v2
#   P_lite_<cell>, P_lite_ica_<cell>  same as P_cold_*, with --transform lite
#   ARMS="v1 v2 lite" (default) selects which arms to run.
#
#   ./run_cold_learner_rows.sh
#   CELLS="lin_laplace" ./run_cold_learner_rows.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PYTHON="${PYTHON:-python}"; WORKERS="${WORKERS:-12}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
OUT="results/hardening_validation"; IDS="$OUT/sample_ids.txt"; LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "datasets_v2_$1"; fi; }

run_row () {
  local ds="$1" name="$2"; shift 2
  echo "== $name  ($ds)  $(date +%H:%M) =="; local t0=$SECONDS
  # shellcheck disable=SC2086
  $PYTHON -u run_experiments.py --dataset_dir "$ds" --detector v1L --corr_gamma 2.5 --no-hybrid \
      --chain_prior_threshold 0.05 --bootstrap_n 0 --window_overlap 0.0 --no_ablation \
      --workers "$WORKERS" --batch_size 10 --resume --scenario_ids_file "$IDS" "$@" --output_dir "$OUT/$name" 2>&1 \
     | grep -vE "^  \[" | tee -a "$LOG/$name.log" | grep -E "Final save|succeeded|ERROR|Traceback|error:" || true
  echo "  -> $(( $(wc -l < "$OUT/$name/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
  echo "$name" >> "$OUT/rows_done.txt"
}

ARMS="${ARMS:-v1 v2 lite}"          # v1 rows are the existing P_cold_* / P_cold_ica_*
for c in $CELLS; do
  ds="$(dataset_for "$c")"
  for arm in $ARMS; do
    case "$arm" in
      v1)   run_row "$ds" "P_cold_$c"      --prior_mode cold
            run_row "$ds" "P_cold_ica_$c"  --prior_mode cold --learner icalingam
            run_row "$ds" "P_chain_ica_$c" --prior_mode chain --learner icalingam ;;
      v2)   run_row "$ds" "P_v2_$c"        --prior_mode cold --transform v2
            run_row "$ds" "P_v2_ica_$c"    --prior_mode cold --transform v2 --learner icalingam ;;
      lite) run_row "$ds" "P_lite_$c"      --prior_mode cold --transform lite
            run_row "$ds" "P_lite_ica_$c"  --prior_mode cold --transform lite --learner icalingam ;;
      *) echo "unknown arm $arm" >&2; exit 1 ;;
    esac
  done
done
echo "COLD/LEARNER ROWS DONE"
