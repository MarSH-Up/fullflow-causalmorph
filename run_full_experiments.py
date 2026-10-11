#!/usr/bin/env python3
"""
One-command entry point for run_full_experiments.sh (every arm, every cell, then
the analysis). Runs the driver with THIS interpreter, so the active venv is used.

    python run_full_experiments.py                       # 12 workers, all arms and cells
    python run_full_experiments.py --workers 24 --batch_size 128
    python run_full_experiments.py --cells lin_laplace --arms main cdnod_fz

The driver's output goes to the terminal and to results/full_v2/logs/driver.log.
Re-running resumes where it stopped.
"""
import argparse
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
    ap.add_argument("--arms", nargs="+", help="default: main cdnod_fz lite v1 cdnod_kci (add v2 for the cap ablation)")
    ap.add_argument("--out", default="results/full_v2")
    ap.add_argument("--kci_n", type=int, default=600, help="held-out scenarios for CD-NOD with KCI")
    a = ap.parse_args()

    env = dict(os.environ, PYTHON=sys.executable, WORKERS=str(a.workers), BATCH_SIZE=str(a.batch_size), OUT=a.out,
               KCI_N=str(a.kci_n))
    if a.cells:
        env["CELLS"] = " ".join(a.cells)
    if a.arms:
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
