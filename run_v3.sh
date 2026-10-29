#!/usr/bin/env bash
#
# The method after the 2026-10-29 revision, on the grid and the external benchmark:
#   segmentation  DET (default union: v1O ∪ PELT candidates) validated by the rank-based
#                 segment-level criterion (--cut_filter rank), bound L = 500
#   per window    CausalMorph v2 with Stage 0 routing (--stage0 auto: curvature -> Stage 0,
#                 linear Gaussian -> PC's CPDAG, else the LiNGAM path), DirectLiNGAM,
#                 standardised windows, variables permuted
#
# Steps, on the 600 held-out KCI ids per cell (REF/kci_ids.txt):
#   main600      the method + the per-window baselines on its windows (ablation columns)
#   nostage0     CausalMorph v2 + DirectLiNGAM without Stage 0 (the previous method) on the same windows
#   oracle_seg   true change points as segmentation
#   no_seg       one window over the whole series
#   L1000        bound L = 1000
#   det_v1O      the previous detector alone (no cut filter)        } detector table, same method
#   det_v1O_rank v1O + rank validation                              }
#   det_pelt     PELT alone                                         }
#   det_pelt_rank PELT + rank validation                            }
#   castor       CASTOR (linear EM, window L) on the whole series
# on the MAIN_N stratified held-out ids per cell (the previous run's main_ids.txt minus the 150
# local tuning scenarios):
#   main         the method + baselines; cdnod_fz is copied from $STD (it does not depend on windows)
#   dagma_tuned  DAGMA with the setting chosen on the calibration sample, on main's windows
# on the external benchmark (DAGMA simulator, $EXT):
#   external     the method + baselines; cdnod_fz copied from $ROB/external; castor
#
# Output: $OUT/<cell>/<step>/results.csv (each step resumes); PLAN.tsv for tools/run_status.py.
#
#   MAIN_N=3000 WORKERS=40 nohup ./run_v3.sh > results/v3.log 2>&1 &
#   DET=pelt ./run_v3.sh            # PELT candidates instead of the union
#   STEPS="main600 castor" CELLS="lin_laplace" ./run_v3.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-40}"
REF="${REF:-results/full_20261012_1845}"
STD="${STD:-results/standardised}"
ROB="${ROB:-results/robustness}"
EXTRAS="${EXTRAS:-results/review_extras}"
OUT="${OUT:-results/v3}"
EXT="${EXT:-datasets_dagma_v2}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
STEPS="${STEPS:-main600 nostage0 oracle_seg no_seg L1000 det_v1O det_v1O_rank det_pelt det_pelt_rank main dagma_tuned external castor}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"
FIXED_L="${FIXED_L:-500}"
DET="${DET:-union}"
MAIN_N="${MAIN_N:-3000}"
LOCAL_IDS="${LOCAL_IDS:-results/robustness/grid_ids.txt}"   # scenarios used to fix Stage 0's thresholds locally: excluded from main
IDS="$REF/kci_ids.txt"
[ -f "$IDS" ] || { echo "missing $IDS" >&2; exit 1; }
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "${DATA_PREFIX}$1"; fi; }

COMMON=(--corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05 --bootstrap_n 0
        --window_overlap 0.0 --prior_mode cold --learner directlingam --standardize_windows --permute_variables
        --transform v2cap --workers "$WORKERS" --resume)
METHOD=(--stage0 auto --stage0_output stage0 --curvature_threshold 0.05 --stage0_threshold 0.5)
SEG=(--detector "$DET" --cut_filter rank --fixed_min_regime_len "$FIXED_L")
LEARNERS="directlingam,icalingam,dagma_paper,sortnregress,randomregress,empty,pc"

# main ids: the previous run's stratified sample minus the local tuning scenarios
if [ "$MAIN_N" -gt 0 ]; then
  if [ ! -f "$OUT/main_ids.txt" ]; then
    [ -f "$STD/main_ids.txt" ] || { echo "missing $STD/main_ids.txt (run run_standardised.sh with MAIN_N first)" >&2; exit 1; }
    grep -vxFf "$LOCAL_IDS" "$STD/main_ids.txt" > "$OUT/main_ids.txt"
  fi
  MAIN_IDS=(--scenario_ids_file "$OUT/main_ids.txt")
else
  MAIN_IDS=()
fi
want_for () {
  case "$1" in
    main|dagma_tuned) if [ "$MAIN_N" -gt 0 ]; then grep -c . "$OUT/main_ids.txt"; else echo $(( $(wc -l < "$(dataset_for "$2")/index.csv") - 1 )); fi ;;
    external) echo $(( $(wc -l < "$EXT/index.csv") - 1 )) ;;
    castor) if [ "$2" = external ]; then echo $(( $(wc -l < "$EXT/index.csv") - 1 )); else grep -c . "$IDS"; fi ;;
    *) grep -c . "$IDS" ;;
  esac
}

{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cells: $CELLS | ids: $IDS | main: $MAIN_N | detector: $DET + rank | L: $FIXED_L | external: $EXT | steps: $STEPS | workers: $WORKERS"; } | tee "$OUT/RUN_INFO.txt"
: > "$OUT/PLAN.tsv"   # read by tools/run_status.py
for step in $STEPS; do
  case "$step" in
    external) printf '%s\t%s\t%s\n' "$step" external "$(want_for external x)" >> "$OUT/PLAN.tsv" ;;
    castor)   for c in $CELLS external; do printf '%s\t%s\t%s\n' "$step" "$c" "$(want_for castor "$c")" >> "$OUT/PLAN.tsv"; done ;;
    *)        for c in $CELLS; do printf '%s\t%s\t%s\n' "$step" "$c" "$(want_for "$step" "$c")" >> "$OUT/PLAN.tsv"; done ;;
  esac
done

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
copy_if_missing () {   # src_dir dst_dir: results that do not depend on the windows are reused
  if [ ! -f "$2/results.csv" ] && [ -f "$1/results.csv" ]; then mkdir -p "$2"; cp "$1/results.csv" "$2/results.csv"; echo "   copied $1 -> $2"; fi
}

for step in $STEPS; do
  if [ "$step" = external ]; then
    dir="$OUT/external/main"; want="$(want_for external x)"
    run "external/main" "$dir" "$want" "$PYTHON" -u run_experiments.py --dataset_dir "$EXT" --output_dir "$dir" "${COMMON[@]}" "${METHOD[@]}" "${SEG[@]}" --ablation_methods "$LEARNERS" --batch_size 64
    copy_if_missing "$ROB/external/cdnod_fz" "$OUT/external/cdnod_fz"
    continue
  fi
  if [ "$step" = castor ]; then
    if [ ! -d third_party/CASTOR ]; then
      git clone -q https://github.com/arahmani1/CASTOR third_party/CASTOR && git -C third_party/CASTOR checkout -q b425ebcf3beba4259a6af6a0d11e0739c2cfbcf2
    fi
    echo "   CASTOR at $(git -C third_party/CASTOR rev-parse --short HEAD)" | tee -a "$OUT/RUN_INFO.txt"
    for c in $CELLS; do
      run "$c/castor" "$OUT/$c/castor" "$(want_for castor "$c")" "$PYTHON" -u tools/run_castor.py --dataset_dir "$(dataset_for "$c")" --output_dir "$OUT/$c/castor" \
          --scenario_ids_file "$IDS" --window "$FIXED_L" --permute_variables --workers "$WORKERS" --resume
    done
    run "external/castor" "$OUT/external/castor" "$(want_for castor external)" "$PYTHON" -u tools/run_castor.py --dataset_dir "$EXT" --output_dir "$OUT/external/castor" \
        --window "$FIXED_L" --permute_variables --workers "$WORKERS" --resume
    continue
  fi
  for c in $CELLS; do
    ds="$(dataset_for "$c")"; dir="$OUT/$c/$step"; want="$(want_for "$step" "$c")"
    base=("$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${COMMON[@]}")
    held=(--scenario_ids_file "$IDS" --batch_size 64)
    case "$step" in
      main600)       run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" "${SEG[@]}" --ablation_methods "$LEARNERS" "${held[@]}" ;;
      nostage0)      run "$c/$step" "$dir" "$want" "${base[@]}" --stage0 off --cps_from "$OUT/$c/main600/results.csv" --no_ablation "${held[@]}" ;;
      oracle_seg)    run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" --oracle_segmentation --no_ablation "${held[@]}" ;;
      no_seg)        run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" --no_segmentation --no_ablation "${held[@]}" ;;
      L1000)         run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" --detector "$DET" --cut_filter rank --fixed_min_regime_len 1000 --no_ablation "${held[@]}" ;;
      det_v1O)       run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" --detector v1O --fixed_min_regime_len "$FIXED_L" --no_ablation "${held[@]}" ;;
      det_v1O_rank)  run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" --detector v1O --cut_filter rank --fixed_min_regime_len "$FIXED_L" --no_ablation "${held[@]}" ;;
      det_pelt)      run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" --detector pelt --fixed_min_regime_len "$FIXED_L" --no_ablation "${held[@]}" ;;
      det_pelt_rank) run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" --detector pelt --cut_filter rank --fixed_min_regime_len "$FIXED_L" --no_ablation "${held[@]}" ;;
      main)          run "$c/$step" "$dir" "$want" "${base[@]}" "${METHOD[@]}" "${SEG[@]}" --ablation_methods "$LEARNERS" ${MAIN_IDS[@]+"${MAIN_IDS[@]}"} --batch_size 128
                     copy_if_missing "$STD/$c/cdnod_fz" "$OUT/$c/cdnod_fz" ;;
      dagma_tuned)   choice="$(tail -1 "$EXTRAS/dagma_choice.txt")"
                     run "$c/$step" "$dir" "$want" "${base[@]}" --cps_from "$OUT/$c/main/results.csv" --ablation_methods "$choice" --batch_size 128 ;;
      *) echo "unknown step '$step'" >&2; exit 1 ;;
    esac
  done
done
echo "V3 DONE  $(date '+%F %H:%M')"
