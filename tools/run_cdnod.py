#!/usr/bin/env python3
"""
Run the CD-NOD baseline (cdnod_baseline.py) over a dataset directory and write
one results.csv row per scenario, with the same metric columns as the pipeline
rows (windowed_nshd_weighted, mean_struct_f1, mean_norm_shd, transition_f1,
consensus_f1, mean_skeleton_f1) so the two can be paired by scenario_id.

    python tools/run_cdnod.py --dataset_dir datasets --output_dir results/x/cdnod_kci \
        --test kci --n_sub 500 --scenario_ids_file ids.txt --workers 12 --resume

    # CD-NOD given the pipeline's detected segments as the context variable:
    python tools/run_cdnod.py ... --c_index segments --segments_from results/x/main/results.csv

Appends every --batch_size results (workers never wait for a batch to finish); --resume skips scenarios already in results.csv and refuses
a file written with another --test / --c_index / --n_sub.
"""
import argparse
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "NSD_Wavelets", "src"), os.path.join(ROOT, "causalmorph")):
    if path not in sys.path:
        sys.path.insert(0, path)

from cdnod_baseline import evaluate_single_graph, run_cdnod  # noqa: E402


def run_scenario(X, names, true_cps, true_adjs, meta, test="kci", alpha=0.05, n_sub=500, c_index="time",
                 change_points=None) -> dict:
    row = {**meta, "cdnod_test": test, "cdnod_c_index": c_index, "cdnod_n_sub": n_sub if n_sub else "all",
           "cdnod_alpha": alpha}
    try:
        A, info = run_cdnod(X, test=test, alpha=alpha, n_sub=n_sub, c_index=c_index, change_points=change_points)
        row.update(info)
        row.update(evaluate_single_graph(A, true_adjs, true_cps, len(X), names))
        row["status"] = "ok"
    except Exception:  # noqa: BLE001 — recorded per scenario, the run continues
        row.update(status="error", error_msg=traceback.format_exc(limit=3).strip())
    return row


def _job(sid, index_row, dataset_dir, cfg, cps_detected, permute_variables=False):
    import warnings as _w
    _w.filterwarnings("ignore")
    from run_experiments import load_scenario
    meta = dict(scenario_id=sid, p=int(index_row["p"]), n_regimes=int(index_row["n_regimes"]),
                samples_regime=int(index_row.get("samples_regime") or 0))
    try:
        X, names, true_cps, true_adjs, _, _ = load_scenario(sid, dataset_dir, index_row, permute_variables=permute_variables)
    except Exception:  # noqa: BLE001
        return {**meta, "status": "error", "error_msg": traceback.format_exc(limit=3).strip()}
    return run_scenario(X, names, true_cps, true_adjs, meta, change_points=cps_detected, **cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--test", choices=["kci", "fisherz"], default="kci")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_sub", type=int, default=500, help="time-uniform subsample size; 0 = all rows")
    ap.add_argument("--c_index", choices=["time", "segments"], default="time")
    ap.add_argument("--segments_from", help="pipeline results.csv whose detected_cps define the segments")
    ap.add_argument("--scenario_ids_file")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--permute_variables", action="store_true",
                    help="permute the variables of every scenario at load (see run_experiments.load_scenario)")
    a = ap.parse_args()
    n_sub = a.n_sub or None
    cfg = dict(test=a.test, alpha=a.alpha, n_sub=n_sub, c_index=a.c_index)

    index = pd.read_csv(os.path.join(a.dataset_dir, "index.csv"), dtype=str)
    if a.scenario_ids_file:
        keep = {ln.strip() for ln in open(a.scenario_ids_file) if ln.strip()}
        index = index[index.scenario_id.isin(keep)]
    detected = {}
    if a.c_index == "segments":
        if not a.segments_from:
            raise SystemExit("--c_index segments needs --segments_from <pipeline results.csv>")
        seg = pd.read_csv(a.segments_from, dtype=str)
        seg = seg[seg.status == "ok"].drop_duplicates("scenario_id")
        detected = {r.scenario_id: [int(x) for x in str(r.detected_cps).split(";") if x.strip() and x != "nan"]
                    for r in seg.itertuples()}
        index = index[index.scenario_id.isin(detected)]

    os.makedirs(a.output_dir, exist_ok=True)
    out = os.path.join(a.output_dir, "results.csv")
    if a.resume and os.path.exists(out):
        prev = pd.read_csv(out, dtype=str)
        want = {"cdnod_test": a.test, "cdnod_c_index": a.c_index, "cdnod_n_sub": str(n_sub) if n_sub else "all"}
        for k, v in want.items():
            found = set(prev[k].dropna()) if k in prev else set()
            if found - {v}:
                raise SystemExit(f"{out} was written with {k}={sorted(found)}, this run asks for {v!r}; "
                                 f"use a fresh --output_dir.")
        index = index[~index.scenario_id.isin(set(prev.scenario_id))]
    print(f"CD-NOD  test={a.test} n_sub={n_sub or 'all'} C={a.c_index}  scenarios to run: {len(index)}  "
          f"workers={a.workers}", flush=True)

    t0, done, header = time.time(), 0, not os.path.exists(out)
    columns = None if header else list(pd.read_csv(out, nrows=0).columns)
    rows = list(index.iterrows())

    def flush(buf):
        nonlocal columns, header
        df = pd.DataFrame(buf)
        if columns is None:
            columns = list(df.columns) + [c for c in ("error_msg",) if c not in df.columns]
        df = df.reindex(columns=columns)
        df.to_csv(out, mode="a", header=header, index=False); header = False
        n_err = int((df.status != "ok").sum())
        print(f"  {done}/{len(rows)}  ({(time.time() - t0) / 60:.1f} min){'  errors: ' + str(n_err) if n_err else ''}",
              flush=True)

    # Every scenario is queued at once so no worker waits for the slowest scenario of a batch
    # (KCI time varies by an order of magnitude with p); results are appended every batch_size.
    buf = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_job, r.scenario_id, r, a.dataset_dir, cfg, detected.get(r.scenario_id), a.permute_variables) for _, r in rows]
        for f in as_completed(futs):
            buf.append(f.result()); done += 1
            if len(buf) >= a.batch_size:
                flush(buf); buf = []
        if buf:
            flush(buf)
    print(f"Final save: {done} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
