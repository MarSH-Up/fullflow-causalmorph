#!/usr/bin/env bash
#
# The paper's runs in the scale-free configuration: every extraction window is
# standardised per column (--standardize_windows), the per-window learner is
# DirectLiNGAM, whose causal-order search does not depend on scale, and the
# detector is sized from the fixed bound L = 500 (the shortest regime the grid
# can generate), never from the scenario's true shortest regime.
#
#   on the 600 held-out KCI scenarios per cell (REF/kci_ids.txt), first:
#     oracle_seg  true change points as segmentation (detector cost)
#     oracle_L    detector sized from the true shortest regime (old headline)
#     L1000       detector sized from L = 1000
#     lite        Stage II of the preconditioning removed (litecap)
#     no_seg      one window over the whole series (ignore the non-stationarity)
#     pelt        PELT with a Gaussian mean-and-covariance cost (ruptures) as the
#                 detector, same bound, same extraction: is the detector needed?
#     cdnod_kci   CD-NOD with KCI on a 500-row subsample, scored per regime and
#                 against the union of the regime graphs
#   on every scenario of the four cells (the analysis drops the calibration sample):
#     cdnod_fz    CD-NOD with Fisher-z and C = time, scored per regime and against
#                 the union of the regime graphs (its own target)
#     main        the method, plus DirectLiNGAM, ICA-LiNGAM, DAGMA (configuration
#                 of its paper), sortnregress, random-order regression and the
#                 empty graph on the same standardised windows
# CD-NOD runs on the raw series: its tests (Fisher-z, KCI with median-heuristic
# widths) do not depend on the scale of the variables.
#
# Output: $OUT/<cell>/<step>/results.csv (each step resumes).
#
#   nohup ./run_standardised.sh > results/standardised.log 2>&1 &
#   MAIN_N=3000 ... runs main and cdnod_fz on a stratified held-out sample of 3,000 per cell
#   STEPS="main" CELLS="lin_laplace" WORKERS=40 ./run_standardised.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-40}"
REF="${REF:-results/full_20261012_1845}"
OUT="${OUT:-results/standardised}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
STEPS="${STEPS:-oracle_seg oracle_L L1000 lite no_seg pelt cdnod_fz cdnod_kci main}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"
FIXED_L="${FIXED_L:-500}"
PERMUTE="${PERMUTE:-1}"   # 1: permute the variables of every scenario (the generator writes the causal order as the column order)
PERM=(); [ "$PERMUTE" = 1 ] && PERM=(--permute_variables)
MAIN_N="${MAIN_N:-0}"   # 0: every scenario in main and cdnod_fz; N: a stratified held-out sample of N per cell
EXCLUDE_IDS="${EXCLUDE_IDS:-results/hardening_validation/sample_ids.txt}"
IDS="$REF/kci_ids.txt"
[ -f "$IDS" ] || { echo "missing $IDS" >&2; exit 1; }
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "${DATA_PREFIX}$1"; fi; }

BASE=(--detector v1O --corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05
      --bootstrap_n 0 --window_overlap 0.0 --prior_mode cold --learner directlingam --standardize_windows)
LEARNERS="directlingam,icalingam,dagma_paper,sortnregress,randomregress,empty"

MAIN_IDS=()
if [ "$MAIN_N" -gt 0 ]; then
  [ -f "$OUT/main_ids.txt" ] || "$PYTHON" tools/stratified_ids.py --index datasets/index.csv --n "$MAIN_N" \
      --exclude "$EXCLUDE_IDS" --out "$OUT/main_ids.txt" >&2
  MAIN_IDS=(--scenario_ids_file "$OUT/main_ids.txt")
fi
want_for () {
  if [ "$1" = main ] || [ "$1" = cdnod_fz ]; then
    if [ "$MAIN_N" -gt 0 ]; then grep -c . "$OUT/main_ids.txt"; else echo $(( $(wc -l < "$(dataset_for "$2")/index.csv") - 1 )); fi
  else grep -c . "$IDS"; fi
}

{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cells: $CELLS | ids: $IDS | fixed L: $FIXED_L | steps: $STEPS | workers: $WORKERS"; } | tee "$OUT/RUN_INFO.txt"
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
  for c in $CELLS; do
    ds="$(dataset_for "$c")"; dir="$OUT/$c/$step"; want="$(want_for "$step" "$c")"
    common=("$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${BASE[@]}" ${PERM[@]+"${PERM[@]}"} --workers "$WORKERS" --resume)
    case "$step" in
      oracle_seg) run "$c/$step" "$dir" "$want" "${common[@]}" --transform v2cap --oracle_segmentation --no_ablation --scenario_ids_file "$IDS" --batch_size 64 ;;
      oracle_L)   run "$c/$step" "$dir" "$want" "${common[@]}" --transform v2cap --no_ablation --scenario_ids_file "$IDS" --batch_size 64 ;;
      L1000)      run "$c/$step" "$dir" "$want" "${common[@]}" --transform v2cap --fixed_min_regime_len 1000 --no_ablation --scenario_ids_file "$IDS" --batch_size 64 ;;
      lite)       run "$c/$step" "$dir" "$want" "${common[@]}" --transform litecap --fixed_min_regime_len "$FIXED_L" --no_ablation --scenario_ids_file "$IDS" --batch_size 64 ;;
      no_seg)     run "$c/$step" "$dir" "$want" "${common[@]}" --transform v2cap --no_segmentation --no_ablation --scenario_ids_file "$IDS" --batch_size 64 ;;
      pelt)       run "$c/$step" "$dir" "$want" "${common[@]}" --transform v2cap --detector pelt --fixed_min_regime_len "$FIXED_L" --no_ablation --scenario_ids_file "$IDS" --batch_size 64 ;;
      cdnod_fz)   run "$c/$step" "$dir" "$want" "$PYTHON" -u tools/run_cdnod.py --dataset_dir "$ds" --output_dir "$dir" \
                    --test fisherz --n_sub 0 --c_index time ${MAIN_IDS[@]+"${MAIN_IDS[@]}"} ${PERM[@]+"${PERM[@]}"} --workers "$WORKERS" --resume ;;
      cdnod_kci)  run "$c/$step" "$dir" "$want" "$PYTHON" -u tools/run_cdnod.py --dataset_dir "$ds" --output_dir "$dir" \
                    --test kci --n_sub 500 --c_index time --scenario_ids_file "$IDS" ${PERM[@]+"${PERM[@]}"} --workers "$WORKERS" --resume ;;
      main)       run "$c/$step" "$dir" "$want" "${common[@]}" --transform v2cap --fixed_min_regime_len "$FIXED_L" --ablation_methods "$LEARNERS" ${MAIN_IDS[@]+"${MAIN_IDS[@]}"} --batch_size 128 ;;
      *) echo "unknown step '$step' (use oracle_seg oracle_L L1000 lite no_seg pelt cdnod_fz cdnod_kci main)" >&2; exit 1 ;;
    esac
  done
done
echo "STANDARDISED DONE  $(date '+%F %H:%M')"
