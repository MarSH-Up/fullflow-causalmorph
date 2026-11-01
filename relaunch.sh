#!/usr/bin/env bash
#
# One command to (re)start every review run on the server:
#   1. stop this user's runs (run_all, the three runners and their workers) and wait
#   2. optionally move the current results aside (--archive SUFFIX)
#   3. git pull (only once nothing is running)
#   4. check the dependencies of .venv
#   5. start run_all.sh in the background
#
#   ./relaunch.sh                      # stop, pull, resume where every step stopped
#   ./relaunch.sh --archive noperm     # same, but start from scratch: results/<run> -> results/<run>_noperm
#   ./relaunch.sh --v3                 # the same steps for run_v3.sh (results/v3, log results/v3.log)
#   ./relaunch.sh --complements        # run_complements.sh next to whatever is running (nothing is stopped;
#                                      # pull refused if it would rewrite a running bash script)
#   DRY_RUN=1 ./relaunch.sh ...        # print what would happen
#
# MAIN_N (default 3000) and the worker split W_ROB / W_STD / W_EXTRA pass through to run_all.sh;
# MAIN_N and WORKERS (default 40) to run_v3.sh.
# The body is one function so that `git pull` can rewrite this file while it runs.

main () {
  set -euo pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")"
  local archive="" dry="${DRY_RUN:-0}" v3=0 comp=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --archive) archive="${2:?--archive needs a suffix, e.g. noperm}"; shift 2 ;;
      --v3) v3=1; shift ;;
      --complements) comp=1; shift ;;
      *) echo "unknown option $1" >&2; exit 1 ;;
    esac
  done
  do_ () { echo "+ $*"; [ "$dry" = 1 ] || "$@"; }
  local runs="robustness standardised review_extras"
  [ "$v3" = 1 ] && runs="v3"
  local pat="run_all.sh|run_v3.sh|run_robustness.sh|run_standardised.sh|run_review_extras.sh|run_experiments.py|run_cdnod.py|run_castor.py|generate_datasets.py|generate_dagma_benchmark.py"

  if [ "$comp" = 1 ]; then
    [ -n "$archive" ] && { echo "--complements keeps every result; drop --archive" >&2; exit 1; }
    local busy=0
    echo "== 1. $(whoami)'s runs (kept running)"
    if pgrep -u "$(whoami)" -f "$pat" > /dev/null; then
      busy=1; pgrep -u "$(whoami)" -fl "run_v3.sh|run_all.sh|run_castor.py|run_experiments.py|run_cdnod.py" | cut -c1-120 | head -8
      echo "   ($(pgrep -u "$(whoami)" -fc "$pat") processes; the complements start with fewer workers)"
    else
      echo "   nothing running"
    fi
    if pgrep -u "$(whoami)" -f "run_complements.sh" > /dev/null; then echo "ERROR: run_complements.sh is already running" >&2; exit 1; fi
    echo "== 2. git pull"
    do_ git fetch -q origin dev
    if [ "$busy" = 1 ] && git diff --name-only HEAD origin/dev | grep -qxE "run_v3.sh|run_all.sh|run_robustness.sh|run_standardised.sh|run_review_extras.sh"; then
      echo "ERROR: the pull would rewrite a bash script that is running (bash reads it incrementally); wait or stop it first:" >&2
      git diff --name-only HEAD origin/dev | grep -xE "run_v3.sh|run_all.sh|run_robustness.sh|run_standardised.sh|run_review_extras.sh" >&2; exit 1
    fi
    do_ git pull --ff-only origin dev
    echo "   at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"
    echo "== 3. dependencies"
    export PYTHON="${PYTHON:-.venv/bin/python}"
    "$PYTHON" -c "import ruptures, dagma, lingam, causallearn, torch, sklearn" || { echo "ERROR: $PYTHON is missing a dependency" >&2; exit 1; }
    echo "   ok ($PYTHON)"
    export WORKERS="${WORKERS:-$([ "$busy" = 1 ] && echo 20 || echo 40)}"
    echo "== 4. starting run_complements.sh with $WORKERS workers (steps: ${STEPS:-diagnostics dagma_ext L_large castor})"
    if [ "$dry" = 1 ]; then
      echo "+ nohup ./run_complements.sh > results/v3_complements.log 2>&1 &"
    else
      nohup ./run_complements.sh > results/v3_complements.log 2>&1 &
      sleep 5
      head -3 results/v3_complements.log
    fi
    echo
    echo "follow it with:"
    echo "   watch -n 60 $PYTHON tools/run_status.py --out results/v3"
    exit 0
  fi

  echo "== 1. stopping $(whoami)'s runs"
  if pgrep -u "$(whoami)" -f "$pat" > /dev/null; then
    do_ pkill -u "$(whoami)" -f "run_all.sh|run_v3.sh|run_robustness.sh|run_standardised.sh|run_review_extras.sh" || true
    do_ pkill -u "$(whoami)" -f "run_experiments.py|run_cdnod.py|run_castor.py|generate_datasets.py|generate_dagma_benchmark.py" || true
    if [ "$dry" != 1 ]; then
      for _ in $(seq 1 30); do pgrep -u "$(whoami)" -f "$pat" > /dev/null || break; sleep 2; done
      if pgrep -u "$(whoami)" -f "$pat" > /dev/null; then
        echo "   still running after 60 s, forcing"; pkill -9 -u "$(whoami)" -f "$pat" || true; sleep 2
      fi
      if pgrep -u "$(whoami)" -f "$pat" > /dev/null; then
        echo "ERROR: could not stop:"; pgrep -u "$(whoami)" -fl "$pat"; exit 1
      fi
    fi
  fi
  echo "   nothing running"

  if [ -n "$archive" ]; then
    echo "== 2. moving the current results aside as *_$archive"
    for r in $runs; do
      if [ -e "results/${r}_$archive" ]; then echo "ERROR: results/${r}_$archive already exists; pick another suffix"; exit 1; fi
    done
    for r in $runs; do
      if [ -d "results/$r" ]; then do_ mv "results/$r" "results/${r}_$archive"; fi
      if [ -f "results/$r.log" ]; then do_ mv "results/$r.log" "results/${r}_$archive.log"; fi
    done
  else
    echo "== 2. keeping the current results (every step resumes)"
  fi

  echo "== 3. git pull"
  do_ git pull --ff-only origin dev
  echo "   at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"

  echo "== 4. dependencies"
  export PYTHON="${PYTHON:-.venv/bin/python}"
  "$PYTHON" -c "import ruptures, dagma, lingam, causallearn, torch, sklearn" || {
    echo "ERROR: $PYTHON is missing a dependency; install it into .venv first, e.g."
    echo "   uv pip install --python .venv/bin/python 'ruptures==1.1.9'"; exit 1; }
  echo "   ok ($PYTHON)"

  export MAIN_N="${MAIN_N:-3000}"
  if [ "$v3" = 1 ]; then
    echo "== 5. starting run_v3.sh"
    export WORKERS="${WORKERS:-40}"
    if [ "$dry" = 1 ]; then
      echo "+ nohup ./run_v3.sh > results/v3.log 2>&1 &"
    else
      nohup ./run_v3.sh > results/v3.log 2>&1 &
      sleep 5
      head -3 results/v3.log
    fi
    echo
    echo "follow it with:"
    echo "   watch -n 60 $PYTHON tools/run_status.py --out results/v3"
    exit 0
  fi
  echo "== 5. starting run_all.sh"
  if [ "$dry" = 1 ]; then
    echo "+ nohup ./run_all.sh > results/run_all.log 2>&1 &"
  else
    nohup ./run_all.sh > results/run_all.log 2>&1 &
    sleep 5
    head -3 results/run_all.log
  fi
  echo
  echo "follow everything with:"
  echo "   watch -n 30 $PYTHON tools/run_status.py"
  exit 0
}
main "$@"
