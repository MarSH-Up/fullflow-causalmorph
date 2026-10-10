#!/usr/bin/env bash
#
# Full experiments for CausalMorph v2: the 18,000-scenario grid under the four
# benchmark cells (mechanism x noise), three arms each, then the analysis.
#
#   cell             mechanism   noise     dataset dir
#   lin_laplace      linear      laplace   datasets/                (the existing grid)
#   nonlin_laplace   nonlinear   laplace   datasets_full_nonlin_laplace/
#   lin_normal       linear      normal    datasets_full_lin_normal/
#   nonlin_normal    nonlinear   normal    datasets_full_nonlin_normal/
#
#   arm    configuration                                          what it gives
#   main   v2 with the Stage I cap (v2cap) + ICA-LiNGAM, cold     the proposed pipeline with every
#          prior, ablation on                                     change, plus DirectLiNGAM and
#                                                                 ICA-LiNGAM alone on the same windows
#   lite   v2cap with Stage II off (litecap) + ICA-LiNGAM         isolates the new Stage II
#   v1     published CausalMorph + chain prior + DirectLiNGAM     the October pipeline, corrected
#                                                                 convention, seeded ("before")
#   v2     v2 without the Stage I cap + ICA-LiNGAM (optional,     isolates the Stage I cap
#          not in the default ARMS)
#
# Shared by every arm: v1L detector, gamma 2.5, no hybrid, no bootstrap, no
# window overlap. Arms run in the order main → lite → v1 across all cells, so
# the proposed pipeline is complete first if the run is stopped early.
#
# Safe to re-run: every arm passes --resume (which also refuses a results.csv
# written with another --transform or --learner), and dataset generation is
# skipped when the dataset directory is already complete.
#
#   ./run_full_experiments.sh
#   WORKERS=24 ./run_full_experiments.sh
#   CELLS="lin_laplace lin_normal" ARMS="main" ./run_full_experiments.sh
#   ARMS="main lite v1 v2" ./run_full_experiments.sh                      # also measure the Stage I cap
#   SCENARIO_IDS_FILE=ids.txt OUT=/tmp/smoke DATA_PREFIX=/tmp/smoke/ds_ ./run_full_experiments.sh   # restricted run
#
# Detached (server):
#   nohup ./run_full_experiments.sh > results/logs/full_v2_driver.log 2>&1 &
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-50}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
ARMS="${ARMS:-main lite v1}"
OUT="${OUT:-results/full_v2}"
SCENARIO_IDS_FILE="${SCENARIO_IDS_FILE:-}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"   # generated cells go to ${DATA_PREFIX}<cell>/
NONLIN="${NONLIN:-0.7}"; NONLIN_FUNC="${NONLIN_FUNC:-cube}"   # same mechanism as the 600-scenario benchmark
EXPECTED_SCENARIOS=18000
LOG="$OUT/logs"; mkdir -p "$LOG"

# One BLAS thread per worker process (oversubscription costs ~1.2x, see RUNBOOK_full_grid.md).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

dataset_for () {
  case "$1" in
    lin_laplace) echo "datasets" ;;
    nonlin_laplace|lin_normal|nonlin_normal) echo "${DATA_PREFIX}$1" ;;
    *) echo "unknown cell '$1' (use lin_laplace nonlin_laplace lin_normal nonlin_normal)" >&2; exit 1 ;;
  esac
}

gen_flags () {
  case "$1" in
    nonlin_laplace) echo "--mechanism nonlinear --nonlinearity $NONLIN --nonlin_func $NONLIN_FUNC --noise_dist laplace" ;;
    lin_normal)     echo "--mechanism linear --noise_dist normal" ;;
    nonlin_normal)  echo "--mechanism nonlinear --nonlinearity $NONLIN --nonlin_func $NONLIN_FUNC --noise_dist normal" ;;
  esac
}

arm_flags () {
  case "$1" in
    main) echo "--transform v2cap   --learner icalingam    --prior_mode cold" ;;
    lite) echo "--transform litecap --learner icalingam    --prior_mode cold  --no_ablation" ;;
    v1)   echo "--transform v1      --learner directlingam --prior_mode chain --no_ablation" ;;
    v2)   echo "--transform v2      --learner icalingam    --prior_mode cold  --no_ablation" ;;
    *) echo "unknown arm '$1' (use main lite v1 v2)" >&2; exit 1 ;;
  esac
}

n_rows () { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }

ensure_dataset () {
  local cell="$1" ds; ds="$(dataset_for "$cell")"
  [ "$cell" = "lin_laplace" ] && { [ -f "$ds/index.csv" ] || { echo "missing $ds/index.csv (see RUNBOOK_full_grid.md)" >&2; exit 1; }; return; }
  local want=$EXPECTED_SCENARIOS only=""
  if [ -n "$SCENARIO_IDS_FILE" ]; then want=$(grep -c . "$SCENARIO_IDS_FILE"); only="--only_ids $SCENARIO_IDS_FILE"; fi
  if [ "$(n_rows "$ds/index.csv")" -ge "$want" ]; then return; fi
  echo "== generating $ds  $(date '+%F %H:%M') =="
  # shellcheck disable=SC2046,SC2086
  $PYTHON generate_datasets.py --output_dir "$ds" $only $(gen_flags "$cell") 2>&1 | tail -2
  local got; got=$(n_rows "$ds/index.csv")
  [ "$got" -ge "$want" ] || { echo "dataset $ds has $got scenarios, expected $want" >&2; exit 1; }
}

run_arm () {
  local cell="$1" arm="$2" ds dir t0; ds="$(dataset_for "$cell")"; dir="$OUT/$cell/$arm"; t0=$SECONDS
  echo "== $cell / $arm  $(date '+%F %H:%M')  (done so far: $(n_rows "$dir/results.csv")) =="
  local ids=(); [ -n "$SCENARIO_IDS_FILE" ] && ids=(--scenario_ids_file "$SCENARIO_IDS_FILE")
  # The runner's exit status is checked explicitly: a failed or refused run
  # (e.g. --resume over a file written with another --transform) stops the
  # driver instead of being masked by the log filters.
  local status
  set +e
  # shellcheck disable=SC2046
  $PYTHON -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" \
      --detector v1L --corr_gamma 2.5 --no-hybrid --chain_prior_threshold 0.05 \
      --bootstrap_n 0 --window_overlap 0.0 \
      $(arm_flags "$arm") "${ids[@]}" \
      --workers "$WORKERS" --batch_size "$BATCH_SIZE" --resume 2>&1 \
    | grep -vE "^  \[" | tee -a "$LOG/${cell}_${arm}.log" | grep -E "Final save|succeeded|ERROR|Traceback|error:"
  status=${PIPESTATUS[0]}
  set -e
  if [ "$status" -ne 0 ]; then
    echo "ERROR: $cell / $arm stopped (run_experiments exit $status). Last lines of $LOG/${cell}_${arm}.log:" >&2
    tail -3 "$LOG/${cell}_${arm}.log" >&2
    exit 1
  fi
  echo "  -> $(n_rows "$dir/results.csv") rows, $(( (SECONDS - t0) / 60 )) min"
}

echo "full experiments  cells: $CELLS  arms: $ARMS  workers: $WORKERS  out: $OUT  $(date '+%F %H:%M')"
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee "$OUT/RUN_INFO.txt"
echo "cells: $CELLS | arms: $ARMS | ids: ${SCENARIO_IDS_FILE:-all}" >> "$OUT/RUN_INFO.txt"

for cell in $CELLS; do ensure_dataset "$cell"; done
for arm in $ARMS; do
  for cell in $CELLS; do run_arm "$cell" "$arm"; done
done

echo "== analysis  $(date '+%F %H:%M') =="
$PYTHON tools/analyze_full_v2.py --root "$OUT" 2>&1 | tee "$OUT/summary.txt"
echo "FULL EXPERIMENTS DONE  $(date '+%F %H:%M')"
