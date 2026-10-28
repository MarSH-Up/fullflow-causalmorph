#!/usr/bin/env bash
#
# What the preconditioning (CausalMorph v2) contributes, on standardised windows with
# the detector bound L = 500, for both per-window learners:
#
#   <learner>_v2       full preconditioning (v2cap)
#   <learner>_lite     without Stage II, the noise alignment (Jarque-Bera gate + rank map)
#   <learner>_none     no preconditioning
#   <learner>_shuffle  full preconditioning fed the tentative skeleton oriented along a
#                      random order: if the gain survives, it does not come from the
#                      tentative orientation
#
# The detector runs once per cell (directlingam_v2); every other configuration reuses
# those windows (--cps_from), so all of them are scored on the same segmentation.
#
#   PYTHON=... IDS=results/robustness/grid_ids.txt DATA_PREFIX=datasets_rob150_ ./run_causalmorph_check.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-12}"
OUT="${OUT:-results/causalmorph_check}"
IDS="${IDS:-results/full_20261012_1845/kci_ids.txt}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"
FIXED_L="${FIXED_L:-500}"
PERMUTE="${PERMUTE:-1}"
CONFIGS="${CONFIGS:-directlingam_v2 directlingam_lite directlingam_none directlingam_shuffle icalingam_v2 icalingam_lite icalingam_none icalingam_shuffle}"
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "${DATA_PREFIX}$1"; fi; }
BASE=(--detector v1O --corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05
      --bootstrap_n 0 --window_overlap 0.0 --prior_mode cold --standardize_windows
      --fixed_min_regime_len "$FIXED_L" --no_ablation --scenario_ids_file "$IDS")
[ "$PERMUTE" = 1 ] && BASE+=(--permute_variables)
WANT=$(grep -c . "$IDS")

{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cells: $CELLS | ids: $IDS | configs: $CONFIGS | workers: $WORKERS"; } | tee "$OUT/RUN_INFO.txt"
: > "$OUT/PLAN.tsv"
for cfg in $CONFIGS; do for c in $CELLS; do printf '%s\t%s\t%s\n' "$cfg" "$c" "$WANT" >> "$OUT/PLAN.tsv"; done; done

run () {   # name dir command...
  local name="$1" dir="$2"; shift 2; local t0=$SECONDS status
  echo "== $name  $(date '+%F %H:%M') =="
  set +e
  "$@" 2>&1 | grep -vE "^  \[" >> "$LOG/${name//\//_}.log"
  status=${PIPESTATUS[0]}
  set -e
  [ "$status" -eq 0 ] || { echo "ERROR: $name failed (exit $status); see $LOG/${name//\//_}.log" >&2; tail -3 "$LOG/${name//\//_}.log" >&2; exit 1; }
  echo "  -> $(( $(wc -l < "$dir/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
}

for cfg in $CONFIGS; do
  learner="${cfg%_*}"; variant="${cfg##*_}"
  case "$variant" in
    v2) extra=(--transform v2cap) ;;
    lite) extra=(--transform litecap) ;;
    none) extra=(--transform none) ;;
    shuffle) extra=(--transform v2cap --prior_shuffle) ;;
    *) echo "unknown variant in '$cfg'" >&2; exit 1 ;;
  esac
  for c in $CELLS; do
    dir="$OUT/$c/$cfg"
    reuse=(); [ "$cfg" = directlingam_v2 ] || reuse=(--cps_from "$OUT/$c/directlingam_v2/results.csv")
    run "$c/$cfg" "$dir" "$PYTHON" -u run_experiments.py --dataset_dir "$(dataset_for "$c")" --output_dir "$dir" \
        "${BASE[@]}" --learner "$learner" "${extra[@]}" ${reuse[@]+"${reuse[@]}"} --workers "$WORKERS" --batch_size 32 --resume
  done
done
echo "CAUSALMORPH CHECK DONE  $(date '+%F %H:%M')"
