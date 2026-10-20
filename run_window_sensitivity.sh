#!/usr/bin/env bash
#
# Detector-window sensitivity: the main arm (pipeline + per-window learners)
# on the 600 held-out scenarios of the CD-NOD/KCI subsample, with the detector
# windows sized from a fixed assumed lower bound on regime length
# (--fixed_min_regime_len W) instead of each scenario's true shortest regime.
# One run per W under results/window_sensitivity/W<W>/, paired against the
# reference run's main arm on the same scenario ids.
#
#   nohup ./run_window_sensitivity.sh > results/window_sensitivity.log 2>&1 &
#   WINDOWS="500" WORKERS=24 ./run_window_sensitivity.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REF="${REF:-results/full_20261012_1845}"     # run whose kci_ids.txt gives the 600 held-out ids
WINDOWS="${WINDOWS:-500 1000}"
WORKERS="${WORKERS:-36}"
CELLS="${CELLS:-lin_laplace nonlin_laplace lin_normal nonlin_normal}"
OUT_ROOT="${OUT_ROOT:-results/window_sensitivity}"
IDS="$REF/kci_ids.txt"
[ -f "$IDS" ] || { echo "missing $IDS" >&2; exit 1; }
WANT=$(grep -c . "$IDS")

for W in $WINDOWS; do
  out="$OUT_ROOT/W$W"
  echo "== window sensitivity W=$W -> $out  $(date '+%F %H:%M') =="
  # The driver's closing analysis step may fail on a partial run; the arm
  # itself is checked below by row count, so a non-zero driver exit is not fatal.
  if ! WORKERS="$WORKERS" CELLS="$CELLS" ARMS=main FIXED_MIN_REGIME_LEN="$W" \
       SCENARIO_IDS_FILE="$IDS" OUT="$out" ./run_full_experiments.sh; then
    echo "   driver exited non-zero for W=$W; checking the main arm rows" >&2
  fi
  for cell in $CELLS; do
    f="$out/$cell/main/results.csv"
    n=$([ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0)
    [ "$n" -ge "$WANT" ] || { echo "ERROR: W=$W $cell has $n of $WANT rows" >&2; exit 1; }
  done
  echo "   W=$W complete: $WANT rows in every cell"
done
echo "WINDOW SENSITIVITY DONE  $(date '+%F %H:%M')"
