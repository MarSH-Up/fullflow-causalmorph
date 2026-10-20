#!/usr/bin/env bash
#
# Two held-out complements to the full run, on the 600 KCI scenarios
# (REF/kci_ids.txt, disjoint from the 600 calibration scenarios), four cells:
#
#   cdnod_fzseg  CD-NOD, Fisher-z, C = the segments the pipeline detected in
#                the reference run (same windows as the pipeline and DAGMA)
#   oracle_seg   the pipeline with the true change points as segmentation
#                (no detector), to price missed and spurious change points
#
# Output: results/heldout_extras/<cell>/<step>/results.csv (each step resumes).
#
#   nohup ./run_heldout_extras.sh > results/heldout_extras.log 2>&1 &
#   STEPS="cdnod_fzseg" CELLS="lin_laplace" WORKERS=12 ./run_heldout_extras.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-24}"
REF="${REF:-results/full_20261012_1845}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
STEPS="${STEPS:-cdnod_fzseg oracle_seg}"
OUT="${OUT:-results/heldout_extras}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"
IDS="$REF/kci_ids.txt"
[ -f "$IDS" ] || { echo "missing $IDS" >&2; exit 1; }
WANT=$(grep -c . "$IDS")
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "${DATA_PREFIX}$1"; fi; }

# the pipeline configuration of the reference run (run_full_experiments.sh PIPE_COMMON + main arm)
PIPE=(--detector v1O --corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05
      --bootstrap_n 0 --window_overlap 0.0 --transform v2cap --learner icalingam --prior_mode cold)

checked () {   # name, command...: run, keep the log, verify the row count
  local name="$1"; shift; local t0=$SECONDS status f
  echo "== $name  $(date '+%F %H:%M') =="
  set +e
  "$@" 2>&1 | grep -vE "^  \[" | tee -a "$LOG/$name.log" | grep -E "Final save|succeeded|ERROR|Traceback|error|errors:"
  status=${PIPESTATUS[0]}
  set -e
  [ "$status" -eq 0 ] || { echo "ERROR: $name failed (exit $status); see $LOG/$name.log" >&2; tail -3 "$LOG/$name.log" >&2; exit 1; }
  f="$OUT/${name%%:*}/${name##*:}/results.csv"
  local n; n=$(( $(wc -l < "$f") - 1 ))
  [ "$n" -ge "$WANT" ] || { echo "ERROR: $name has $n of $WANT rows" >&2; exit 1; }
  echo "  -> $n rows in $(( (SECONDS - t0) / 60 )) min"
}

{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "ref: $REF | ids: $IDS ($WANT) | cells: $CELLS | steps: $STEPS"; } | tee "$OUT/RUN_INFO.txt"

for step in $STEPS; do
  for c in $CELLS; do
    ds="$(dataset_for "$c")"; dir="$OUT/$c/$step"
    case "$step" in
      cdnod_fzseg)
        seg="$REF/$c/main/results.csv"; [ -f "$seg" ] || { echo "missing $seg" >&2; exit 1; }
        checked "$c:$step" "$PYTHON" -u tools/run_cdnod.py --dataset_dir "$ds" --output_dir "$dir" \
            --test fisherz --n_sub 0 --c_index segments --segments_from "$seg" \
            --scenario_ids_file "$IDS" --workers "$WORKERS" --resume ;;
      oracle_seg)
        checked "$c:$step" "$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE[@]}" \
            --oracle_segmentation --no_ablation \
            --scenario_ids_file "$IDS" --workers "$WORKERS" --batch_size 64 --resume ;;
      *) echo "unknown step '$step' (use cdnod_fzseg oracle_seg)" >&2; exit 1 ;;
    esac
  done
done
echo "HELDOUT EXTRAS DONE  $(date '+%F %H:%M')"
