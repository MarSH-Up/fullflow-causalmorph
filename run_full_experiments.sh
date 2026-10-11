#!/usr/bin/env bash
#
# Full experiments: the 18,000-scenario grid under four benchmark cells
# (mechanism x noise), the proposed pipeline, its ablations and every external
# method, then the analysis on the held-out scenarios.
#
#   cell             mechanism   noise     dataset dir
#   lin_laplace      linear      laplace   datasets/                (the existing grid)
#   nonlin_laplace   nonlinear   laplace   datasets_full_nonlin_laplace/
#   lin_normal       linear      normal    datasets_full_lin_normal/
#   nonlin_normal    nonlinear   normal    datasets_full_nonlin_normal/
#
#   arm        what runs                                              scenarios
#   main       pipeline with every change: v1L detection, v2 with     all
#              the Stage I cap (v2cap), ICA-LiNGAM, cold prior; plus
#              on the SAME detected windows: DirectLiNGAM, ICA-LiNGAM
#              and DAGMA (linear) alone (MAIN_ABLATION)
#   cdnod_fz   CD-NOD (Huang et al. 2020), Fisher-z, C = time index,  all
#              whole series, one graph
#   lite       v2cap with Stage II off (litecap) + ICA-LiNGAM         all
#   v1         published CausalMorph + chain prior + DirectLiNGAM     all
#              (the October pipeline, corrected convention: "before")
#   cdnod_kci  CD-NOD with KCI (the paper's test) on a 500-row        KCI_N held-out scenarios,
#              time-uniform subsample                                 stratified by p, regimes, samples
#   v2         optional: v2 without the Stage I cap + ICA-LiNGAM      all
#
# Not included: RPCMCI (tigramite) was tried and does not apply here. It
# assigns regimes from lagged prediction errors and failed to converge on every
# probe with contemporaneous structure (2026-10-10).
#
# Shared by the pipeline arms: v1N detector (Pearson gamma 2.0 and Spearman gamma 1.5 must agree), no hybrid, no
# bootstrap, no window overlap. Arms run in the order of ARMS across all
# cells, so the proposed pipeline and the cheap baselines finish first.
#
# The analysis excludes the 600 scenarios used to tune the method
# (results/hardening_validation/sample_ids.txt; EXCLUDE_IDS) so the reported
# numbers are on held-out scenarios. CD-NOD with KCI runs on held-out ids only.
#
# Safe to re-run: every arm resumes; --resume refuses a results file written
# with another --transform, --learner, per-window learner set or CD-NOD
# setting; dataset generation is skipped when a directory is complete.
#
#   ./run_full_experiments.sh
#   WORKERS=32 BATCH_SIZE=128 ./run_full_experiments.sh
#   CELLS="lin_laplace" ARMS="main cdnod_fz" ./run_full_experiments.sh
#   ARMS="main cdnod_fz lite v1 cdnod_kci v2" ./run_full_experiments.sh        # also measure the Stage I cap
#   SCENARIO_IDS_FILE=ids.txt OUT=/tmp/smoke DATA_PREFIX=/tmp/smoke/ds_ ./run_full_experiments.sh   # smoke test
#
# Detached (server):
#   nohup ./run_full_experiments.sh > results/logs/full_driver.log 2>&1 &
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"   # scenarios per batch; results.csv is appended after every batch
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
ARMS="${ARMS:-main cdnod_fz lite v1 cdnod_kci}"
OUT="${OUT:-results/full_$(date +%Y%m%d_%H%M)}"   # a new folder per run; set OUT=<folder> to resume one
SCENARIO_IDS_FILE="${SCENARIO_IDS_FILE:-}"
DATA_PREFIX="${DATA_PREFIX:-datasets_full_}"   # generated cells go to ${DATA_PREFIX}<cell>/
NONLIN="${NONLIN:-0.7}"; NONLIN_FUNC="${NONLIN_FUNC:-cube}"   # same mechanism as the 600-scenario benchmark
MAIN_ABLATION="${MAIN_ABLATION:-directlingam,icalingam,dagma}"
EXCLUDE_IDS="${EXCLUDE_IDS:-results/hardening_validation/sample_ids.txt}"
KCI_N="${KCI_N:-600}"; KCI_NSUB="${KCI_NSUB:-500}"
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

PIPE_COMMON=(--detector v1N --corr_gamma 2.0 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05 --bootstrap_n 0 --window_overlap 0.0)

arm_cmd () {   # prints nothing; fills the global CMD array for arm $1 on dataset $2, output $3
  local arm="$1" ds="$2" dir="$3"
  case "$arm" in
    main) CMD=("$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE_COMMON[@]}"
               --transform v2cap --learner icalingam --prior_mode cold --ablation_methods "$MAIN_ABLATION") ;;
    lite) CMD=("$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE_COMMON[@]}"
               --transform litecap --learner icalingam --prior_mode cold --no_ablation) ;;
    v1)   CMD=("$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE_COMMON[@]}"
               --transform v1 --learner directlingam --prior_mode chain --no_ablation) ;;
    v2)   CMD=("$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${PIPE_COMMON[@]}"
               --transform v2 --learner icalingam --prior_mode cold --no_ablation) ;;
    cdnod_fz)  CMD=("$PYTHON" -u tools/run_cdnod.py --dataset_dir "$ds" --output_dir "$dir"
                    --test fisherz --n_sub 0 --c_index time) ;;
    cdnod_kci) CMD=("$PYTHON" -u tools/run_cdnod.py --dataset_dir "$ds" --output_dir "$dir"
                    --test kci --n_sub "$KCI_NSUB" --c_index time) ;;
    *) echo "unknown arm '$arm' (use main cdnod_fz lite v1 cdnod_kci v2)" >&2; exit 1 ;;
  esac
}

n_rows () { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }

want_rows () { if [ -n "$SCENARIO_IDS_FILE" ]; then grep -c . "$SCENARIO_IDS_FILE"; else echo "$EXPECTED_SCENARIOS"; fi; }

# Dataset generation runs in the background, one process per missing cell, so
# the arms on cells that are already complete (lin_laplace) start at once; each
# arm waits only for the cell it needs.
# (one variable GEN_PID_<cell> per cell: portable to bash 3.2, which has no associative arrays)
start_generation () {
  local cell="$1" ds; ds="$(dataset_for "$cell")"
  if [ "$cell" = "lin_laplace" ]; then
    [ -f "$ds/index.csv" ] || { echo "missing $ds/index.csv (see RUNBOOK_full_grid.md)" >&2; exit 1; }
    return
  fi
  [ "$(n_rows "$ds/index.csv")" -ge "$(want_rows)" ] && return
  local only=""; [ -n "$SCENARIO_IDS_FILE" ] && only="--only_ids $SCENARIO_IDS_FILE"
  echo "== generating $ds in the background (log: $LOG/generate_$cell.log)  $(date '+%F %H:%M') =="
  # shellcheck disable=SC2046,SC2086
  $PYTHON generate_datasets.py --output_dir "$ds" $only $(gen_flags "$cell") > "$LOG/generate_$cell.log" 2>&1 &
  printf -v "GEN_PID_$cell" '%s' "$!"
}

wait_dataset () {
  local cell="$1" ds; ds="$(dataset_for "$cell")"
  local var="GEN_PID_$cell"
  if [ -n "${!var:-}" ]; then
    echo "   waiting for $ds to finish generating  $(date '+%F %H:%M')"
    wait "${!var}" || { echo "ERROR: generating $ds failed; see $LOG/generate_$cell.log" >&2; tail -3 "$LOG/generate_$cell.log" >&2; exit 1; }
    unset "$var"
    echo "   $ds ready  $(date '+%F %H:%M')"
  fi
  local got; got=$(n_rows "$ds/index.csv")
  [ "$got" -ge "$(want_rows)" ] || { echo "dataset $ds has $got scenarios, expected $(want_rows)" >&2; exit 1; }
}

kci_ids () {   # stratified held-out ids for CD-NOD with KCI (same file for every cell)
  local f="$OUT/kci_ids.txt"
  if [ ! -f "$f" ]; then
    $PYTHON tools/stratified_ids.py --index datasets/index.csv --n "$KCI_N" --exclude "$EXCLUDE_IDS" \
        ${SCENARIO_IDS_FILE:+--within "$SCENARIO_IDS_FILE"} --out "$f" >&2
  fi
  echo "$f"
}

run_arm () {
  local cell="$1" arm="$2" ds dir t0 status; ds="$(dataset_for "$cell")"; dir="$OUT/$cell/$arm"; t0=$SECONDS
  echo "== $cell / $arm  $(date '+%F %H:%M')  (done so far: $(n_rows "$dir/results.csv")) =="
  arm_cmd "$arm" "$ds" "$dir"
  if [ "$arm" = "cdnod_kci" ]; then CMD+=(--scenario_ids_file "$(kci_ids)")
  elif [ -n "$SCENARIO_IDS_FILE" ]; then CMD+=(--scenario_ids_file "$SCENARIO_IDS_FILE"); fi
  CMD+=(--workers "$WORKERS" --batch_size "$BATCH_SIZE" --resume)
  # The runner's exit status is checked explicitly: a failed or refused run
  # (e.g. --resume over a file written with other settings) stops the driver
  # instead of being masked by the log filters.
  set +e
  "${CMD[@]}" 2>&1 | grep -vE "^  \[" | tee -a "$LOG/${cell}_${arm}.log" \
    | grep -E "Final save|succeeded|ERROR|Traceback|error|errors:"
  status=${PIPESTATUS[0]}
  set -e
  if [ "$status" -ne 0 ]; then
    echo "ERROR: $cell / $arm stopped (exit $status). Last lines of $LOG/${cell}_${arm}.log:" >&2
    tail -3 "$LOG/${cell}_${arm}.log" >&2
    exit 1
  fi
  echo "  -> $(n_rows "$dir/results.csv") rows, $(( (SECONDS - t0) / 60 )) min"
}

echo "full experiments  cells: $CELLS  arms: $ARMS  workers: $WORKERS  out: $OUT  $(date '+%F %H:%M')"
{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cells: $CELLS | arms: $ARMS | ids: ${SCENARIO_IDS_FILE:-all} | per-window learners: $MAIN_ABLATION"
  echo "kci: $KCI_N held-out scenarios, $KCI_NSUB-row subsample | analysis excludes: $EXCLUDE_IDS"; } | tee "$OUT/RUN_INFO.txt"

for cell in $CELLS; do start_generation "$cell"; done
for arm in $ARMS; do
  for cell in $CELLS; do wait_dataset "$cell"; run_arm "$cell" "$arm"; done
done

echo "== analysis  $(date '+%F %H:%M') =="
$PYTHON tools/analyze_full_v2.py --root "$OUT" --exclude_ids "$EXCLUDE_IDS" 2>&1 | tee "$OUT/summary.txt"
echo "FULL EXPERIMENTS DONE  $(date '+%F %H:%M')"
