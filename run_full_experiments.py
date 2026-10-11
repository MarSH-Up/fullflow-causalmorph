#!/usr/bin/env python3
"""
One-command entry point for run_full_experiments.sh (every arm, every cell, then
the analysis). Runs the driver with THIS interpreter, so the active venv is used.

    python run_full_experiments.py                       # 12 workers, all arms and cells
    python run_full_experiments.py --workers 60          # the 64-core Threadripper server
    python run_full_experiments.py --workers 24 --batch_size 128
    python run_full_experiments.py --cells lin_laplace --arms main cdnod_fz
    python run_full_experiments.py --out results/full_20261011_0300   # resume that run

Every new run writes to its own folder, results/full_<YYYYmmdd_HHMM>, so an
earlier run is never overwritten. The driver's output goes to the terminal and
to <out>/logs/driver.log. Pass --out with an existing folder to resume it.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128,
                    help="scenarios per batch; every runner appends its results.csv after each batch")
    ap.add_argument("--cells", nargs="+", help="default: lin_laplace nonlin_laplace lin_normal nonlin_normal")
    ap.add_argument("--arms", nargs="+", default=["main", "cdnod_fz", "lite", "cdnod_kci"],
                    help="default: main cdnod_fz lite cdnod_kci (v1 = the October pipeline and v2 = the uncapped "
                         "ablation are available but no longer run by default)")
    ap.add_argument("--out", help="output folder; default: a new results/full_<YYYYmmdd_HHMM>")
    ap.add_argument("--kci_n", type=int, default=600, help="held-out scenarios for CD-NOD with KCI")
    a = ap.parse_args()
    a.out = a.out or f"results/full_{dt.datetime.now():%Y%m%d_%H%M}"
    print(f"output folder: {a.out}", flush=True)

    env = dict(os.environ, PYTHON=sys.executable, WORKERS=str(a.workers), BATCH_SIZE=str(a.batch_size), OUT=a.out,
               KCI_N=str(a.kci_n))
    if a.cells:
        env["CELLS"] = " ".join(a.cells)
    env["ARMS"] = " ".join(a.arms)
    log_dir = os.path.join(HERE, a.out, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "driver.log")
    with open(log_path, "a") as log:
        proc = subprocess.Popen(["bash", os.path.join(HERE, "run_full_experiments.sh")], cwd=HERE, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            log.write(line); log.flush()
        code = proc.wait()
    sys.exit(code)


if __name__ == "__main__":
    main()
