#!/usr/bin/env bash
#
# Extras of the second internal review, in the configuration of run_standardised.sh
# (standardised windows, DirectLiNGAM, detector bound L = 500):
#
#   dagma_sweep  the 600 calibration scenarios per cell: DAGMA (paper schedule) over a
#                grid of l1 weights and weight thresholds, the same tuning budget the
#                method's constants had; tools/pick_dagma.py then picks ONE setting for
#                all cells (as for the method) and writes $OUT/dagma_choice.txt
#   pc           the 600 held-out scenarios per cell: PC with Fisher-z per window, a
#                CPDAG learner that does not need non-Gaussianity (skeleton F1 matters)
#   dagma_tuned  every scenario of the four cells, on the windows the main run of
#                run_standardised.sh detected (--cps_from): DAGMA with the chosen
#                setting. Needs that main step finished; run it last.
#
# Output: $OUT/<cell>/<step>/results.csv (each step resumes).
#
#   nohup ./run_review_extras.sh > results/review_extras.log 2>&1 &
#   STEPS="dagma_tuned" ./run_review_extras.sh          # after run_standardised.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-24}"
REF="${REF:-results/full_20261012_1845}"
STD="${STD:-results/standardised}"
OUT="${OUT:-results/review_extras}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
STEPS="${STEPS:-dagma_sweep pc}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"
FIXED_L="${FIXED_L:-500}"
PERMUTE="${PERMUTE:-1}"
CAL_IDS="${CAL_IDS:-results/hardening_validation/sample_ids.txt}"
IDS="$REF/kci_ids.txt"
LAMBDAS="${LAMBDAS:-0.005 0.01 0.02 0.05 0.1}"
THRESHOLDS="${THRESHOLDS:-0.1 0.2 0.3}"
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "${DATA_PREFIX}$1"; fi; }
BASE=(--detector v1O --corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05
      --bootstrap_n 0 --window_overlap 0.0 --prior_mode cold --learner directlingam --standardize_windows
      --transform v2cap --fixed_min_regime_len "$FIXED_L")
[ "$PERMUTE" = 1 ] && BASE+=(--permute_variables)
SWEEP=""; for l in $LAMBDAS; do for t in $THRESHOLDS; do SWEEP="$SWEEP,dagma_l${l}_t${t}"; done; done; SWEEP="${SWEEP#,}"

want_for () {
  case "$1" in
    dagma_sweep) grep -c . "$CAL_IDS" ;;
    pc)          grep -c . "$IDS" ;;
    dagma_tuned) echo $(( $(wc -l < "$STD/$2/main/results.csv") - 1 )) ;;   # every scenario main ran
  esac
}

{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cells: $CELLS | steps: $STEPS | sweep: $SWEEP | workers: $WORKERS"; } | tee "$OUT/RUN_INFO.txt"
: > "$OUT/PLAN.tsv"   # read by tools/run_status.py
for step in $STEPS; do for c in $CELLS; do printf '%s\t%s\t%s\n' "$step" "$c" "$(want_for "$step" "$c")" >> "$OUT/PLAN.tsv"; done; done

run () {   # name dir want command...
  local name="$1" dir="$2" want="$3"; shift 3; local t0=$SECONDS status n
  echo "== $name  $(date '+%F %H:%M') =="
  set +e
  "$@" 2>&1 | grep -vE "^  \[" >> "$LOG/${name//\//_}.log"
  status=${PIPESTATUS[0]}
  set -e
  [ "$status" -eq 0 ] || { echo "ERROR: $name failed (exit $status); see $LOG/${name//\//_}.log" >&2; tail -3 "$LOG/${name//\//_}.log" >&2; exit 1; }
  n=$(( $(wc -l < "$dir/results.csv") - 1 ))
  [ "$n" -ge "$want" ] || { echo "ERROR: $name has $n of $want rows" >&2; exit 1; }
  echo "  -> $n rows in $(( (SECONDS - t0) / 60 )) min"
}

for step in $STEPS; do
  if [ "$step" = dagma_tuned ]; then
    [ -f "$OUT/dagma_choice.txt" ] || "$PYTHON" tools/pick_dagma.py "$OUT" $CELLS > "$OUT/dagma_choice.txt" 2> "$OUT/dagma_sweep_table.txt"
    choice="$(tail -1 "$OUT/dagma_choice.txt")"
    echo "   DAGMA setting chosen on the calibration sample: $choice"
  fi
  for c in $CELLS; do
    ds="$(dataset_for "$c")"; dir="$OUT/$c/$step"; want="$(want_for "$step" "$c")"
    common=("$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${BASE[@]}" --workers "$WORKERS" --resume --batch_size 64)
    case "$step" in
      dagma_sweep) run "$c/$step" "$dir" "$want" "${common[@]}" --scenario_ids_file "$CAL_IDS" --ablation_methods "$SWEEP" ;;
      pc)          run "$c/$step" "$dir" "$want" "${common[@]}" --scenario_ids_file "$IDS" --ablation_methods "pc,directlingam" ;;
      dagma_tuned)
        main="$STD/$c/main/results.csv"
        [ "$(( $(wc -l < "$main") - 1 ))" -ge "$want" ] || { echo "ERROR: $main is not complete; run dagma_tuned after run_standardised.sh" >&2; exit 1; }
        run "$c/$step" "$dir" "$want" "${common[@]}" --cps_from "$main" --ablation_methods "$choice" ;;
      *) echo "unknown step '$step' (use dagma_sweep pc dagma_tuned)" >&2; exit 1 ;;
    esac
  done
done
echo "REVIEW EXTRAS DONE  $(date '+%F %H:%M')"
