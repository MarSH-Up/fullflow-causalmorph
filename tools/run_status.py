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

# preliminary result shown for every step with rows: mean nSHD of the step's own output
# and of the baselines it carries on the same windows (rows finished so far)
PRELIM = [("mean_norm_shd", "nSHD"), ("directlingam_mean_norm_shd", "DL"), ("icalingam_mean_norm_shd", "ICA"),
          ("dagma_paper_mean_norm_shd", "DAGMA"), ("sortnregress_mean_norm_shd", "sortnr"),
          ("randomregress_mean_norm_shd", "rand"), ("empty_mean_norm_shd", "empty"), ("union_norm_shd", "vs union")]


def prelim(path):
    """'nSHD .071 | DAGMA .271 | ... | F1 .87' from the rows finished so far; '' when there are none."""
    import pandas as pd
    try:
        d = pd.read_csv(path)
    except (OSError, ValueError):
        return ""
    d = d[d.status == "ok"] if "status" in d else d
    if d.empty:
        return ""
    parts = [f"{lab} {d[c].mean():.3f}" for c, lab in PRELIM if c in d and d[c].notna().any()]
    if {"chow0001_tp", "chow0001_fp", "chow0001_fn"} <= set(d.columns) and d.chow0001_tp.notna().any():
        tp, fp, fn = d.chow0001_tp.sum(), d.chow0001_fp.sum(), d.chow0001_fn.sum()
        if tp + fp and tp + fn:
            P, R = tp / (tp + fp), tp / (tp + fn)
            parts.append(f"conf F1 {2 * P * R / (P + R):.2f}")
    return " | ".join(parts)


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
    results = []
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
        if n:
            results.append((k, step, t, status, prelim(os.path.join(a.out, t, step, "results.csv"))))

    first = min(starts.values()) if starts else None
    eta = (now - first).total_seconds() / 3600 * (total_rows - done_rows) / done_rows if first and done_rows else None
    print(f"\noverall: {done_rows} / {total_rows} rows ({100 * done_rows / max(total_rows, 1):.1f}%)"
          f"   time left at the overall rate: {fmt_h(eta) if eta else '-'}   (steps differ in cost: rough)\n")
    if results:
        print("preliminary results (mean nSHD over the rows finished so far; lower is better)")
        for k, step, t, status, line in results:
            print(f"{k:>2}  {step:<14}{t:<16}{'' if status == 'done' else '(partial) '}{line}")
        print()


if __name__ == "__main__":
    main()
