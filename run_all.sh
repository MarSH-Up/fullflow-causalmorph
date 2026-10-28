#!/usr/bin/env bash
#
# Every run of the internal review in one command, in parallel, with the workers
# split between them:
#
#   run_robustness.sh      raw vs standardised windows, all baselines, external benchmark
#   run_standardised.sh    the paper's configuration: 600-scenario steps, then main
#   run_review_extras.sh   DAGMA calibration sweep and PC per window; once both the
#                          sweep and main are done, DAGMA with the chosen setting on
#                          the windows main detected (dagma_tuned)
#
# Each runner keeps its own log (results/<run>.log) and resumes where it stopped.
# One view of everything:   watch -n 30 .venv/bin/python tools/run_status.py
#
#   nohup ./run_all.sh > results/run_all.log 2>&1 &
#   MAIN_N=3000 nohup ./run_all.sh > results/run_all.log 2>&1 &     # main on 3,000 held-out per cell
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
W_ROB="${W_ROB:-16}"; W_STD="${W_STD:-32}"; W_EXTRA="${W_EXTRA:-16}"
MAIN_N="${MAIN_N:-0}"
export PYTHON MAIN_N

"$PYTHON" -c "import ruptures, dagma, lingam, causallearn" || {
  echo "ERROR: $PYTHON is missing ruptures, dagma, lingam or causal-learn (see requirements-fullgrid.txt)" >&2; exit 1; }
echo "run_all  $(date '+%F %H:%M')  git $(git rev-parse --short HEAD)  workers rob/std/extras $W_ROB/$W_STD/$W_EXTRA  MAIN_N $MAIN_N"

WORKERS="$W_ROB" ./run_robustness.sh > results/robustness.log 2>&1 &
P_ROB=$!
WORKERS="$W_STD" ./run_standardised.sh > results/standardised.log 2>&1 &
P_STD=$!
WORKERS="$W_EXTRA" STEPS="dagma_sweep pc" ./run_review_extras.sh > results/review_extras.log 2>&1 &
P_EXT=$!

status=0
wait "$P_EXT" || { echo "ERROR: run_review_extras.sh (sweep, pc) failed; see results/review_extras.log" >&2; status=1; }
wait "$P_STD" || { echo "ERROR: run_standardised.sh failed; see results/standardised.log" >&2; status=1; }
if [ "$status" -eq 0 ]; then
  # the completed sweep and pc steps resume instantly; only dagma_tuned does work
  WORKERS="$W_STD" STEPS="dagma_sweep pc dagma_tuned" ./run_review_extras.sh >> results/review_extras.log 2>&1 \
    || { echo "ERROR: dagma_tuned failed; see results/review_extras.log" >&2; status=1; }
fi
wait "$P_ROB" || { echo "ERROR: run_robustness.sh failed; see results/robustness.log" >&2; status=1; }
echo "run_all finished  $(date '+%F %H:%M')  status $status"
exit "$status"
