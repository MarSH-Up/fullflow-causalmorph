#!/usr/bin/env bash
#
# Local validation of CausalMorph v3 and Stage 0 inside the full pipeline (detector v1O,
# standardised windows, L = 500, variables permuted), on small cells given as name=dir:
#
#   v2cap          the paper's preconditioning (reference; also runs PC, the empty graph and
#                  DirectLiNGAM on the same windows and provides the windows to the rest)
#   v3             polynomial residual in Stage I, no Stage 0
#   v3_gauss       + Stage 0 in the windows the gate calls Gaussian, regime graph = learner's fit
#   v3_gauss_s0    same, regime graph = Stage 0's DAG
#   v3_always_s0   Stage 0 in every window, regime graph = Stage 0's DAG
#   v3_auto        routing by identifiability source: curved windows (CURV, default 0.05) to Stage 0,
#                  linear Gaussian windows (GATE, default 0.5) to PC's CPDAG, the rest to the LiNGAM path
#   v2_auto        the same routing with the paper's v2cap transform on the LiNGAM path
#   baselines      DAGMA (paper configuration) and ICA-LiNGAM on the same windows (ablation columns)
#
#   PYTHON=... WORKERS=10 ./run_stage0_check.sh lin_laplace=path/ds_lin_laplace tanh_unit_normal=path/ds_tanh_unit_normal
#   python tools/analyze_stage0_check.py results/stage0_check
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-10}"
OUT="${OUT:-results/stage0_check}"
FIXED_L="${FIXED_L:-500}"
CONFIGS="${CONFIGS:-v2cap v3 v3_gauss v3_gauss_s0 v3_always_s0 v3_auto v2_auto}"
CURV="${CURV:-0.05}"; GATE="${GATE:-0.5}"
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
BASE=(--corr_gamma 2.5 --corr_gamma_rank 1.5 --no-hybrid --chain_prior_threshold 0.05
      --bootstrap_n 0 --window_overlap 0.0 --prior_mode cold --learner directlingam --standardize_windows
      --fixed_min_regime_len "$FIXED_L" --permute_variables --workers "$WORKERS" --batch_size 16 --resume)
echo "git: $(git rev-parse --short HEAD) | cells: $* | configs: $CONFIGS" | tee "$OUT/RUN_INFO.txt"

run () {   # name dir command...
  local name="$1" dir="$2"; shift 2; local t0=$SECONDS status
  echo "== $name  $(date '+%F %H:%M') =="
  set +e; "$@" 2>&1 | grep -vE "^  \[" >> "$LOG/${name//\//_}.log"; status=${PIPESTATUS[0]}; set -e
  [ "$status" -eq 0 ] || { echo "ERROR: $name failed (exit $status); see $LOG/${name//\//_}.log" >&2; tail -3 "$LOG/${name//\//_}.log" >&2; exit 1; }
  echo "  -> $(( $(wc -l < "$dir/results.csv") - 1 )) rows in $(( (SECONDS - t0) / 60 )) min"
}

for cell in "$@"; do
  name="${cell%%=*}"; ds="${cell#*=}"
  for cfg in $CONFIGS; do
    dir="$OUT/$name/$cfg"
    case "$cfg" in
      v2cap)        extra=(--detector v1O --transform v2cap --ablation_methods pc,empty,directlingam) ;;
      v3)           extra=(--detector v1O --transform v3 --no_ablation) ;;
      v3_gauss)     extra=(--detector v1O --transform v3 --stage0 gauss --no_ablation) ;;
      v3_gauss_s0)  extra=(--detector v1O --transform v3 --stage0 gauss --stage0_output stage0 --no_ablation) ;;
      v3_always_s0) extra=(--detector v1O --transform v3 --stage0 always --stage0_output stage0 --no_ablation) ;;
      v3_auto)      extra=(--detector v1O --transform v3 --stage0 auto --stage0_output stage0 --curvature_threshold "$CURV" --stage0_threshold "$GATE" --no_ablation) ;;
      v2_auto)      extra=(--detector v1O --transform v2cap --stage0 auto --stage0_output stage0 --curvature_threshold "$CURV" --stage0_threshold "$GATE" --no_ablation) ;;
      baselines)    extra=(--detector v1O --transform v2cap --ablation_methods dagma_paper,icalingam) ;;
      filt_bic)     extra=(--detector v1O --cut_filter bic --transform v2cap --stage0 auto --stage0_output stage0 --no_ablation) ;;
      filt_rank)    extra=(--detector v1O --cut_filter rank --transform v2cap --stage0 auto --stage0_output stage0 --no_ablation) ;;
      det_v1P)      extra=(--detector v1P --transform v2cap --stage0 auto --stage0_output stage0 --no_ablation) ;;
      det_v1X)      extra=(--detector v1X --transform v2cap --stage0 auto --stage0_output stage0 --no_ablation) ;;
      det_v1P_g*)   extra=(--detector v1P --corr_gamma_xi "${cfg#det_v1P_g}" --transform v2cap --stage0 auto --stage0_output stage0 --no_ablation) ;;
      det_v1X_g*)   extra=(--detector v1X --corr_gamma_xi "${cfg#det_v1X_g}" --transform v2cap --stage0 auto --stage0_output stage0 --no_ablation) ;;
      *) echo "unknown config '$cfg'" >&2; exit 1 ;;
    esac
    reuse=(); case "$cfg" in v2cap|det_*|filt_*) ;; *) reuse=(--cps_from "$OUT/$name/v2cap/results.csv") ;; esac
    run "$name/$cfg" "$dir" "$PYTHON" -u run_experiments.py --dataset_dir "$ds" --output_dir "$dir" "${BASE[@]}" "${extra[@]}" ${reuse[@]+"${reuse[@]}"}
  done
done
echo "STAGE0 CHECK DONE  $(date '+%F %H:%M')"
