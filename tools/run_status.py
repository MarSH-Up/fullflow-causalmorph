#!/usr/bin/env python3
"""
Progress of run_robustness.sh or run_standardised.sh: one row per (step, data)
of the run's PLAN.tsv, in run order, with rows done, percentage, errors, rate
since that step started and time left; plus a rough estimate for the whole run.

    python tools/run_status.py                                   # results/robustness
    python tools/run_status.py --out results/standardised --log results/standardised.log
    watch -n 30 .venv/bin/python tools/run_status.py             # live, refreshed every 30 s
"""
import argparse
import datetime as dt
import os
import re

from full_status import count_rows, fmt_h


def step_starts(log_path):
    """{(data, step): start datetime} from the runner's '== data/step  YYYY-MM-DD HH:MM ==' lines."""
    starts = {}
    if os.path.exists(log_path):
        for line in open(log_path):
            m = re.match(r"== (\w+)/(\w+)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", line)
            if m:
                starts.setdefault((m.group(1), m.group(2)), dt.datetime.strptime(m.group(3), "%Y-%m-%d %H:%M"))
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/robustness")
    ap.add_argument("--log", default=None, help="the runner's stdout log; default: <out>.log")
    a = ap.parse_args()
    plan_path = os.path.join(a.out, "PLAN.tsv")
    if not os.path.exists(plan_path):
        raise SystemExit(f"no {plan_path}: the run has not started (or pass --out)")
    plan = [l.rstrip("\n").split("\t") for l in open(plan_path) if l.strip()]
    starts = step_starts(a.log or a.out.rstrip("/") + ".log")
    now = dt.datetime.now()

    head = f"{'#':>2}  {'step':<14}{'data':<16}{'done':>15}{'%':>7}{'errors':>8}{'rate/h':>9}{'left':>10}  status"
    print(f"\n{a.out}   {now:%Y-%m-%d %H:%M}\n\n{head}\n{'-' * len(head)}")
    done_rows = total_rows = 0
    for k, (step, t, want) in enumerate(plan, 1):
        want = int(want)
        n, err = count_rows(os.path.join(a.out, t, step, "results.csv"))
        started = starts.get((t, step))
        rate = left = None
        if started and n:
            hrs = max((now - started).total_seconds() / 3600, 1e-6)
            rate = n / hrs
            left = (want - n) / rate if n < want else 0.0
        status = "done" if n >= want else ("running" if started else "pending")
        done_rows += min(n, want); total_rows += want
        print(f"{k:>2}  {step:<14}{t:<16}{n:>7} / {want:<6}{100 * n / max(want, 1):>6.1f}%{err:>8}"
              f"{(f'{rate:,.0f}' if rate else '-'):>9}{fmt_h(left):>10}  {status}")

    first = min(starts.values()) if starts else None
    eta = (now - first).total_seconds() / 3600 * (total_rows - done_rows) / done_rows if first and done_rows else None
    print(f"\noverall: {done_rows} / {total_rows} rows ({100 * done_rows / max(total_rows, 1):.1f}%)"
          f"   time left at the overall rate: {fmt_h(eta) if eta else '-'}   (steps differ in cost: rough)\n")


if __name__ == "__main__":
    main()
