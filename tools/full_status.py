#!/usr/bin/env python3
"""
Progress of run_full_experiments.sh: one row per (arm, cell) in run order, with
rows done, percentage, errors, rate since that step started, and time left;
plus dataset generation and disk usage.

    python tools/full_status.py                  # the newest results/full_* run
    python tools/full_status.py --out /tmp/smoke
    watch -n 30 .venv/bin/python tools/full_status.py     # live, refreshed every 30 s
"""
import argparse
import datetime as dt
import os
import re
import shutil

ARMS = ["main", "cdnod_fz", "lite", "v1", "cdnod_kci", "v2"]
CELLS = ["lin_laplace", "nonlin_laplace", "lin_normal", "nonlin_normal"]
LABEL = {"main": "pipeline + DL/ICA/DAGMA", "cdnod_fz": "CD-NOD Fisher-z", "lite": "ablation lite",
         "v1": "October pipeline", "cdnod_kci": "CD-NOD KCI", "v2": "v2 without cap"}


def latest_run(root="results"):
    """The most recently started full-grid run: newest results/full_* folder with a RUN_INFO.txt."""
    runs = [os.path.join(root, d) for d in os.listdir(root) if d.startswith("full_")] if os.path.isdir(root) else []
    runs = [d for d in runs if os.path.exists(os.path.join(d, "RUN_INFO.txt"))]
    return max(runs, key=lambda d: os.path.getmtime(os.path.join(d, "RUN_INFO.txt"))) if runs else None


def count_rows(path):
    if not os.path.exists(path):
        return 0, 0
    n = err = 0
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
        si = header.index("status") if "status" in header else None
        for line in f:
            n += 1
            if si is not None and ",error," in f",{line.strip()},":
                err += 1
    return n, err


def step_starts(log_path):
    """{(cell, arm): start datetime} from the driver log's '== cell / arm  YYYY-MM-DD HH:MM' lines."""
    starts = {}
    if os.path.exists(log_path):
        for line in open(log_path):
            m = re.match(r"== (\w+) / (\w+)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", line)
            if m:
                starts.setdefault((m.group(1), m.group(2)), dt.datetime.strptime(m.group(3), "%Y-%m-%d %H:%M"))
    return starts


def fmt_h(hours):
    if hours is None:
        return "-"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="run folder; default: the newest results/full_* run")
    ap.add_argument("--expected", type=int, default=18000)
    a = ap.parse_args()
    a.out = a.out or latest_run()
    if not a.out:
        raise SystemExit("no results/full_* run found; pass --out")
    info = open(os.path.join(a.out, "RUN_INFO.txt")).read() if os.path.exists(os.path.join(a.out, "RUN_INFO.txt")) else ""
    m = re.search(r"arms: ([\w ]+?) \|", info); arms = m.group(1).split() if m else ARMS[:5]
    m = re.search(r"cells: ([\w ]+?) \|", info); cells = m.group(1).split() if m else CELLS
    kci_file = os.path.join(a.out, "kci_ids.txt")
    kci_n = sum(1 for l in open(kci_file) if l.strip()) if os.path.exists(kci_file) else 600
    starts = step_starts(os.path.join(a.out, "logs", "driver.log"))
    now = dt.datetime.now()

    print(f"\n{a.out}   {now:%Y-%m-%d %H:%M}\n")
    print("datasets")
    for c in cells:
        ds = "datasets" if c == "lin_laplace" else f"datasets_full_{c}"
        idx = os.path.join(ds, "index.csv")
        if os.path.exists(idx):
            n = sum(1 for _ in open(idx)) - 1
            state = "ready" if n >= a.expected else "incomplete"
        else:   # the generator writes index.csv last: count the scenario files written so far
            n = sum(1 for e in os.scandir(ds) if e.name.endswith("-dat.csv")) if os.path.isdir(ds) else 0
            state = "generating" if n else "pending"
        print(f"  {c:<16} {ds:<30} {n:>6} / {a.expected} scenarios {state}")

    head = f"{'#':>2}  {'step':<26}{'cell':<16}{'done':>14}{'%':>7}{'errors':>8}{'rate/h':>9}{'left':>10}  status"
    print("\n" + head + "\n" + "-" * len(head))
    total_left_h, k = 0.0, 0
    for arm in arms:
        for c in cells:
            k += 1
            want = kci_n if arm == "cdnod_kci" else a.expected
            n, err = count_rows(os.path.join(a.out, c, arm, "results.csv"))
            started = starts.get((c, arm))
            rate = left = None
            if started and n:
                hrs = max((now - started).total_seconds() / 3600, 1e-6)
                rate = n / hrs
                left = (want - n) / rate if n < want else 0.0
            status = "done" if n >= want else ("running" if started else "pending")
            if status == "running" and left is not None:
                total_left_h += left
            print(f"{k:>2}  {LABEL.get(arm, arm):<26}{c:<16}{n:>6} / {want:<6}{100 * n / want:>6.1f}%{err:>8}"
                  f"{(f'{rate:,.0f}' if rate else '-'):>9}{fmt_h(left):>10}  {status}")
    du = shutil.disk_usage(".")
    print(f"\nrunning step, time left at its current rate: {fmt_h(total_left_h) if total_left_h else '-'}"
          f"   (pending steps not included)\ndisk free: {du.free / 1e9:.0f} GB of {du.total / 1e9:.0f} GB\n")


if __name__ == "__main__":
    main()
