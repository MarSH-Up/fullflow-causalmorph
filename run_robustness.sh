#!/usr/bin/env bash
#
# Robustness checks of the internal review, on a held-out subsample of the grid
# and on the external benchmark built from the DAGMA simulator:
#
#   variance leak   every arm twice: raw windows, and windows standardised per
#                   column (--standardize_windows); sortnregress (Reisach et al.
#                   2021) and the empty graph as references
#   oracle bound    the detector is sized from a fixed bound (FIXED_L), never
#                   from the true shortest regime; the external benchmark draws
#                   its regime lengths at random (generate_dagma_benchmark --jitter)
#   DAGMA config    DAGMA with the per-window rescaling (dagma) and with the
#                   configuration of its paper on the raw window (dagma_paper)
#   learner         the pipeline with ICA-LiNGAM (raw, std) and with DirectLiNGAM
#                   (raw_dl, std_dl), whose causal-order search does not use scale
#   segmentation    on the external benchmark, the method (DirectLiNGAM, standardised)
#                   without segmentation (no_seg) and with PELT as the detector (pelt)
#   CD-NOD          Fisher-z, C = time, on the external benchmark, scored per regime
#                   and against the union of the regime graphs
#
# Output: $OUT/<data>/<arm>/results.csv (each step resumes).
#
#   local test:  N_GRID=40 DAGMA_SEEDS=3 WORKERS=12 PYTHON=... ./run_robustness.sh
#   server:      nohup ./run_robustness.sh > results/robustness.log 2>&1 &
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-24}"
OUT="${OUT:-results/robustness}"
N_GRID="${N_GRID:-150}"            # held-out grid scenarios per cell
DAGMA_SEEDS="${DAGMA_SEEDS:-10}"   # repetitions per setting of the external benchmark
FIXED_L="${FIXED_L:-500}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"   # existing full cells; missing ones are generated for the sampled ids
DAGMA_DATA="${DAGMA_DATA:-datasets_dagma_v2}"
STEPS="${STEPS:-raw std raw_dl std_dl no_seg pelt cdnod_fz}"
EXCLUDE_IDS="${EXCLUDE_IDS:-results/hardening_validation/sample_ids.txt}"   # calibration sample
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

PIPE=(--detector v1O --corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05
      --bootstrap_n 0 --window_overlap 0.0 --transform v2cap --prior_mode cold
      --fixed_min_regime_len "$FIXED_L")
RAW_LEARNERS="icalingam,directlingam,dagma,dagma_paper,sortnregress,randomregress,empty"
STD_LEARNERS="icalingam,directlingam,dagma,dagma_paper,sortnregress,randomregress"

dataset_for () { if [ "$1" = "lin_laplace" ]; then echo datasets; else echo "${DATA_PREFIX}$1"; fi; }
gen_flags () {
  case "$1" in
    nonlin_laplace) echo "--mechanism nonlinear --nonlinearity 0.7 --nonlin_func cube --noise_dist laplace" ;;
    lin_normal)     echo "--mechanism linear --noise_dist normal" ;;
    nonlin_normal)  echo "--mechanism nonlinear --nonlinearity 0.7 --nonlin_func cube --noise_dist normal" ;;
  esac
}

# ── data ──────────────────────────────────────────────────────────────────────
IDS="$OUT/grid_ids.txt"
if [ ! -f "$IDS" ]; then
  "$PYTHON" - "$EXCLUDE_IDS" "$N_GRID" "$IDS" <<'EOF'
import sys, pandas as pd
excl = {l.strip() for l in open(sys.argv[1]) if l.strip()}
idx = pd.read_csv("datasets/index.csv")
idx = idx[~idx.scenario_id.isin(excl)]
idx.sample(int(sys.argv[2]), random_state=2027).scenario_id.to_csv(sys.argv[3], index=False, header=False)
EOF
fi
for c in $CELLS; do
  ds="$(dataset_for "$c")"
  if [ "$c" != "lin_laplace" ] && [ ! -f "$ds/index.csv" ]; then
    echo "== generating $ds for $(grep -c . "$IDS") ids  $(date '+%F %H:%M') =="
    # shellcheck disable=SC2046
    "$PYTHON" generate_datasets.py --output_dir "$ds" --only_ids "$IDS" $(gen_flags "$c") > "$LOG/generate_$c.log" 2>&1
  fi
done
[ -f "$DAGMA_DATA/index.csv" ] || "$PYTHON" tools/generate_dagma_benchmark.py --output_dir "$DAGMA_DATA" --n_seeds "$DAGMA_SEEDS"

{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cells: $CELLS ($(grep -c . "$IDS") ids) | external: $DAGMA_DATA | fixed L: $FIXED_L | steps: $STEPS | workers: $WORKERS"; } | tee "$OUT/RUN_INFO.txt"
# the plan read by tools/run_status.py: step, data, expected rows, in run order
targets="$CELLS external"
: > "$OUT/PLAN.tsv"
for step in $STEPS; do for t in $targets; do
  case "$step" in no_seg|pelt|cdnod_fz) [ "$t" != external ] && continue ;; esac
  if [ "$t" = external ]; then n=$(( $(wc -l < "$DAGMA_DATA/index.csv") - 1 )); else n=$(grep -c . "$IDS"); fi
  printf '%s\t%s\t%s\n' "$step" "$t" "$n" >> "$OUT/PLAN.tsv"
done; done

# ── arms ──────────────────────────────────────────────────────────────────────
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

for step in $STEPS; do
  for t in $targets; do
    if [ "$t" = "external" ]; then ds="$DAGMA_DATA"; ids=(); else ds="$(dataset_for "$t")"; ids=(--scenario_ids_file "$IDS"); fi
    dir="$OUT/$t/$step"
    case "$step" in
      raw) run "$t/$step" "$dir" "$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE[@]}" \
               --learner icalingam --ablation_methods "$RAW_LEARNERS" ${ids[@]+"${ids[@]}"} --workers "$WORKERS" --batch_size 32 --resume ;;
      std) run "$t/$step" "$dir" "$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE[@]}" \
               --learner icalingam --standardize_windows --ablation_methods "$STD_LEARNERS" ${ids[@]+"${ids[@]}"} --workers "$WORKERS" --batch_size 32 --resume ;;
      raw_dl) run "$t/$step" "$dir" "$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE[@]}" \
               --learner directlingam --no_ablation ${ids[@]+"${ids[@]}"} --workers "$WORKERS" --batch_size 32 --resume ;;
      std_dl) run "$t/$step" "$dir" "$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE[@]}" \
               --learner directlingam --standardize_windows --no_ablation ${ids[@]+"${ids[@]}"} --workers "$WORKERS" --batch_size 32 --resume ;;
      no_seg) [ "$t" = "external" ] || continue   # the grid cells: run_standardised.sh
             run "$t/$step" "$dir" "$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE[@]}" \
               --learner directlingam --standardize_windows --no_segmentation --no_ablation --workers "$WORKERS" --batch_size 32 --resume ;;
      pelt) [ "$t" = "external" ] || continue
             run "$t/$step" "$dir" "$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE[@]}" \
               --learner directlingam --standardize_windows --detector pelt --no_ablation --workers "$WORKERS" --batch_size 32 --resume ;;
      cdnod_fz) [ "$t" = "external" ] || continue   # the grid cells: run_standardised.sh
             run "$t/$step" "$dir" "$PYTHON" -u tools/run_cdnod.py --dataset_dir "$ds" --output_dir "$dir" \
               --test fisherz --n_sub 0 --c_index time --workers "$WORKERS" --resume ;;
      *) echo "unknown step '$step' (use raw std raw_dl std_dl no_seg pelt cdnod_fz)" >&2; exit 1 ;;
    esac
  done
done
echo "ROBUSTNESS DONE  $(date '+%F %H:%M')"
