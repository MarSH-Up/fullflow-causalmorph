#!/usr/bin/env bash
#
# External benchmark: regimes drawn with the DAGMA simulator
# (tools/generate_dagma_benchmark.py), the pipeline with the constants of the
# reference run frozen (no recalibration), DAGMA on the same windows, and
# CD-NOD with Fisher-z on the whole series.
#
# Output: results/dagma_benchmark/{main,cdnod_fz}/results.csv (each step resumes).
#
#   nohup ./run_dagma_benchmark.sh > results/dagma_benchmark.log 2>&1 &
#   WORKERS=8 STEPS="main" ./run_dagma_benchmark.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-12}"
DATA="${DATA:-datasets_dagma}"
OUT="${OUT:-results/dagma_benchmark}"
STEPS="${STEPS:-main cdnod_fz}"
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

# the pipeline configuration of the reference run (run_full_experiments.sh PIPE_COMMON + main arm)
PIPE=(--detector v1O --corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05
      --bootstrap_n 0 --window_overlap 0.0 --transform v2cap --learner icalingam --prior_mode cold)

[ -f "$DATA/index.csv" ] || "$PYTHON" tools/generate_dagma_benchmark.py --output_dir "$DATA"
WANT=$(( $(wc -l < "$DATA/index.csv") - 1 ))

{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "data: $DATA ($WANT scenarios) | steps: $STEPS | workers: $WORKERS"; } | tee "$OUT/RUN_INFO.txt"

for step in $STEPS; do
  t0=$SECONDS; dir="$OUT/$step"
  echo "== $step  $(date '+%F %H:%M') =="
  case "$step" in
    main)     CMD=("$PYTHON" -u run_experiments.py --dataset_dir "$DATA" --output_dir "$dir" "${PIPE[@]}"
                   --ablation_methods dagma --workers "$WORKERS" --batch_size 32 --resume) ;;
    cdnod_fz) CMD=("$PYTHON" -u tools/run_cdnod.py --dataset_dir "$DATA" --output_dir "$dir"
                   --test fisherz --n_sub 0 --c_index time --workers "$WORKERS" --resume) ;;
    *) echo "unknown step '$step' (use main cdnod_fz)" >&2; exit 1 ;;
  esac
  set +e
  "${CMD[@]}" 2>&1 | grep -vE "^  \[" | tee -a "$LOG/$step.log" | grep -E "Final save|succeeded|ERROR|Traceback|error|errors:"
  status=${PIPESTATUS[0]}
  set -e
  [ "$status" -eq 0 ] || { echo "ERROR: $step failed (exit $status); see $LOG/$step.log" >&2; tail -3 "$LOG/$step.log" >&2; exit 1; }
  n=$(( $(wc -l < "$dir/results.csv") - 1 ))
  [ "$n" -ge "$WANT" ] || { echo "ERROR: $step has $n of $WANT rows" >&2; exit 1; }
  echo "  -> $n rows in $(( (SECONDS - t0) / 60 )) min"
done
echo "DAGMA BENCHMARK DONE  $(date '+%F %H:%M')"
