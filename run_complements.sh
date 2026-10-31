#!/usr/bin/env bash
#
# Complements of run_v3.sh, light (about an hour with 40 workers):
#   diagnostics   benchmark diagnostics per scenario (varsortability, R2-sortability, cubic R2 gain,
#                 density, empty-graph and single-oracle nSHD, edge flips) on the main ids of every cell
#                 and on the external benchmark          -> $OUT/<cell>/diagnostics, $OUT/external/diagnostics
#   dagma_ext     DAGMA with the setting chosen on the calibration sample, on the external windows of main
#                 -> $OUT/external/dagma_tuned
#   L_large       the method with L = 2000 and L = 2500 on the 600 held-out ids (where a bound above the
#                 true shortest regime breaks the segmentation)  -> $OUT/<cell>/L2000, L2500
#
#   PYTHON=.venv/bin/python WORKERS=40 nohup ./run_complements.sh > results/v3_complements.log 2>&1 &
#   STEPS="diagnostics" ./run_complements.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-40}"
OUT="${OUT:-results/v3}"
REF="${REF:-results/full_20261012_1845}"
EXTRAS="${EXTRAS:-results/review_extras}"
EXT="${EXT:-datasets_dagma_v2}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
STEPS="${STEPS:-diagnostics dagma_ext L_large}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"
DET="${DET:-union}"
IDS="$REF/kci_ids.txt"
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "${DATA_PREFIX}$1"; fi; }

COMMON=(--corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05 --bootstrap_n 0
        --window_overlap 0.0 --prior_mode cold --learner directlingam --standardize_windows --permute_variables
        --transform v2cap --workers "$WORKERS" --resume)
METHOD=(--stage0 auto --stage0_output stage0 --curvature_threshold 0.05 --stage0_threshold 0.5)

run () {   # name command...
  local name="$1"; shift; local t0=$SECONDS status
  echo "== $name  $(date '+%F %H:%M') =="
  set +e; "$@" 2>&1 | grep -vE "^  \[" >> "$LOG/${name//\//_}.log"; status=${PIPESTATUS[0]}; set -e
  [ "$status" -eq 0 ] || { echo "ERROR: $name failed (exit $status); see $LOG/${name//\//_}.log" >&2; tail -3 "$LOG/${name//\//_}.log" >&2; exit 1; }
  echo "  -> done in $(( (SECONDS - t0) / 60 )) min"
}

for step in $STEPS; do
  case "$step" in
    diagnostics)
      for c in $CELLS; do
        run "$c/diagnostics" "$PYTHON" -u tools/benchmark_diagnostics.py --dataset_dir "$(dataset_for "$c")" --output_dir "$OUT/$c/diagnostics" \
            --scenario_ids_file "$OUT/main_ids.txt" --workers "$WORKERS" --resume
      done
      run "external/diagnostics" "$PYTHON" -u tools/benchmark_diagnostics.py --dataset_dir "$EXT" --output_dir "$OUT/external/diagnostics" --workers "$WORKERS" --resume ;;
    dagma_ext)
      choice="$(tail -1 "$EXTRAS/dagma_choice.txt")"
      run "external/dagma_tuned" "$PYTHON" -u run_experiments.py --dataset_dir "$EXT" --output_dir "$OUT/external/dagma_tuned" "${COMMON[@]}" \
          --cps_from "$OUT/external/main/results.csv" --ablation_methods "$choice" --batch_size 64 ;;
    L_large)
      for L in 2000 2500; do for c in $CELLS; do
        run "$c/L$L" "$PYTHON" -u run_experiments.py --dataset_dir "$(dataset_for "$c")" --output_dir "$OUT/$c/L$L" "${COMMON[@]}" "${METHOD[@]}" \
            --detector "$DET" --cut_filter rank --fixed_min_regime_len "$L" --no_ablation --scenario_ids_file "$IDS" --batch_size 64
      done; done ;;
    *) echo "unknown step '$step'" >&2; exit 1 ;;
  esac
done
echo "COMPLEMENTS DONE  $(date '+%F %H:%M')"
