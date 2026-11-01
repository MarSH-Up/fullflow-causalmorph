#!/usr/bin/env bash
#
# Complements of run_v3.sh (results go under the same $OUT, rows appended to $OUT/PLAN.tsv so that
# tools/run_status.py --out results/v3 follows them). Roughly 2-3 h with 40 workers, CASTOR apart:
#   diagnostics   benchmark diagnostics per scenario (varsortability, R2-sortability, cubic R2 gain,
#                 density, empty-graph and single-oracle nSHD, edge flips) on the main ids of every cell
#                 and on the external benchmark          -> $OUT/<cell>/diagnostics, $OUT/external/diagnostics
#   dagma_ext     DAGMA with the setting chosen on the calibration sample, on the external windows of main
#                 -> $OUT/external/dagma_tuned
#   L_large       the method with L = 2000 and L = 2500 on the 600 held-out ids (where a bound above the
#                 true shortest regime breaks the segmentation)  -> $OUT/<cell>/L2000, L2500
#   stage0_sens   Stage 0 thresholds moved around the chosen (0.05, 0.5) on the 600 held-out ids, on the
#                 segmentation of main600 (only the per-window stage reruns): (kappa0, g0) in
#                 {(0.02,0.5), (0.10,0.5), (0.05,0.4), (0.05,0.67)}   -> $OUT/<cell>/sens_k<kappa0>_g<g0>
#   castor        what run_v3.sh's castor step has not finished (Gaussian cells, external): waits until no
#                 run_castor.py of this user is running, then resumes every cell and the external benchmark
#                 with the same ids, window and time-out        -> $OUT/<cell>/castor, $OUT/external/castor
#
#   PYTHON=.venv/bin/python WORKERS=40 nohup ./run_complements.sh > results/v3_complements.log 2>&1 &
#   STEPS="diagnostics" ./run_complements.sh
#   ./relaunch.sh --complements        # pull, check the dependencies and start it (does not stop CASTOR)
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
STEPS="${STEPS:-diagnostics dagma_ext L_large castor}"
SENS="${SENS:-0.02,0.5 0.10,0.5 0.05,0.4 0.05,0.67}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"
DET="${DET:-union}"
FIXED_L="${FIXED_L:-500}"
CASTOR_TIMEOUT="${CASTOR_TIMEOUT:-1800}"
IDS="$REF/kci_ids.txt"
CASTOR_IDS="$OUT/castor_ids.txt"
[ -f "$IDS" ] || { echo "missing $IDS" >&2; exit 1; }
[ -f "$OUT/main_ids.txt" ] || { echo "missing $OUT/main_ids.txt (run run_v3.sh first)" >&2; exit 1; }
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "${DATA_PREFIX}$1"; fi; }
n_ext=$(( $(wc -l < "$EXT/index.csv") - 1 )); n_main=$(grep -c . "$OUT/main_ids.txt"); n_held=$(grep -c . "$IDS")

COMMON=(--corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05 --bootstrap_n 0
        --window_overlap 0.0 --prior_mode cold --learner directlingam --standardize_windows --permute_variables
        --transform v2cap --workers "$WORKERS" --resume)
METHOD=(--stage0 auto --stage0_output stage0 --curvature_threshold 0.05 --stage0_threshold 0.5)

plan () {   # step data want: one row per (step, data) in $OUT/PLAN.tsv, read by tools/run_status.py
  awk -F'\t' -v s="$1" -v d="$2" '$1 == s && $2 == d { f = 1 } END { exit !f }' "$OUT/PLAN.tsv" 2>/dev/null || printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$OUT/PLAN.tsv"
}
for step in $STEPS; do
  case "$step" in
    diagnostics) for c in $CELLS; do plan diagnostics "$c" "$n_main"; done; plan diagnostics external "$n_ext" ;;
    dagma_ext)   plan dagma_tuned external "$n_ext" ;;
    L_large)     for L in 2000 2500; do for c in $CELLS; do plan "L$L" "$c" "$n_held"; done; done ;;
    stage0_sens) for kg in $SENS; do for c in $CELLS; do plan "sens_k${kg%,*}_g${kg#*,}" "$c" "$n_held"; done; done ;;
    castor)      [ -f "$CASTOR_IDS" ] || { echo "missing $CASTOR_IDS" >&2; exit 1; }
                 for c in $CELLS; do plan castor "$c" "$(grep -c . "$CASTOR_IDS")"; done; plan castor external "$n_ext" ;;
    *) echo "unknown step '$step'" >&2; exit 1 ;;
  esac
done
echo "complements: git $(git rev-parse --short HEAD 2>/dev/null || echo unknown) | steps: $STEPS | workers: $WORKERS | $(date '+%F %H:%M')" | tee -a "$OUT/RUN_INFO.txt"

run () {   # name dir want command...
  local name="$1" dir="$2" want="$3"; shift 3; local t0=$SECONDS status n
  echo "== $name  $(date '+%F %H:%M') =="
  set +e; "$@" 2>&1 | grep -vE "^  \[" >> "$LOG/${name//\//_}.log"; status=${PIPESTATUS[0]}; set -e
  [ "$status" -eq 0 ] || { echo "ERROR: $name failed (exit $status); see $LOG/${name//\//_}.log" >&2; tail -3 "$LOG/${name//\//_}.log" >&2; exit 1; }
  n=$(( $(wc -l < "$dir/results.csv") - 1 ))
  [ "$n" -ge "$want" ] || { echo "ERROR: $name has $n of $want rows" >&2; exit 1; }
  echo "  -> $n rows in $(( (SECONDS - t0) / 60 )) min"
}

for step in $STEPS; do
  case "$step" in
    diagnostics)
      for c in $CELLS; do
        run "$c/diagnostics" "$OUT/$c/diagnostics" "$n_main" "$PYTHON" -u tools/benchmark_diagnostics.py --dataset_dir "$(dataset_for "$c")" \
            --output_dir "$OUT/$c/diagnostics" --scenario_ids_file "$OUT/main_ids.txt" --workers "$WORKERS" --resume
      done
      run "external/diagnostics" "$OUT/external/diagnostics" "$n_ext" "$PYTHON" -u tools/benchmark_diagnostics.py --dataset_dir "$EXT" \
          --output_dir "$OUT/external/diagnostics" --workers "$WORKERS" --resume ;;
    dagma_ext)
      choice="$(tail -1 "$EXTRAS/dagma_choice.txt")"
      run "external/dagma_tuned" "$OUT/external/dagma_tuned" "$n_ext" "$PYTHON" -u run_experiments.py --dataset_dir "$EXT" --output_dir "$OUT/external/dagma_tuned" \
          "${COMMON[@]}" --cps_from "$OUT/external/main/results.csv" --ablation_methods "$choice" --batch_size 64 ;;
    L_large)
      for L in 2000 2500; do for c in $CELLS; do
        run "$c/L$L" "$OUT/$c/L$L" "$n_held" "$PYTHON" -u run_experiments.py --dataset_dir "$(dataset_for "$c")" --output_dir "$OUT/$c/L$L" "${COMMON[@]}" "${METHOD[@]}" \
            --detector "$DET" --cut_filter rank --fixed_min_regime_len "$L" --no_ablation --scenario_ids_file "$IDS" --batch_size 64
      done; done ;;
    stage0_sens)
      for kg in $SENS; do k0="${kg%,*}"; g0="${kg#*,}"; for c in $CELLS; do
        run "$c/sens_k${k0}_g${g0}" "$OUT/$c/sens_k${k0}_g${g0}" "$n_held" "$PYTHON" -u run_experiments.py --dataset_dir "$(dataset_for "$c")" --output_dir "$OUT/$c/sens_k${k0}_g${g0}" \
            "${COMMON[@]}" --stage0 auto --stage0_output stage0 --curvature_threshold "$k0" --stage0_threshold "$g0" \
            --cps_from "$OUT/$c/main600/results.csv" --no_ablation --scenario_ids_file "$IDS" --batch_size 64
      done; done ;;
    castor)
      while pgrep -u "$(whoami)" -f "run_castor.py" > /dev/null; do
        echo "   castor: run_castor.py of $(whoami) still running, waiting  $(date '+%F %H:%M')"; sleep 600
      done
      [ -d third_party/CASTOR ] || { echo "missing third_party/CASTOR (run_v3.sh clones it)" >&2; exit 1; }
      for c in $CELLS; do
        run "$c/castor" "$OUT/$c/castor" "$(grep -c . "$CASTOR_IDS")" "$PYTHON" -u tools/run_castor.py --dataset_dir "$(dataset_for "$c")" --output_dir "$OUT/$c/castor" \
            --scenario_ids_file "$CASTOR_IDS" --window "$FIXED_L" --timeout "$CASTOR_TIMEOUT" --permute_variables --workers "$WORKERS" --resume
      done
      run "external/castor" "$OUT/external/castor" "$n_ext" "$PYTHON" -u tools/run_castor.py --dataset_dir "$EXT" --output_dir "$OUT/external/castor" \
          --window "$FIXED_L" --timeout "$CASTOR_TIMEOUT" --permute_variables --workers "$WORKERS" --resume ;;
  esac
done
echo "COMPLEMENTS DONE  $(date '+%F %H:%M')"
