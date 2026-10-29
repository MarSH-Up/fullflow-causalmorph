#!/usr/bin/env python3
"""
CASTOR baseline (Rahmani & Frossard, AISTATS 2025; third_party/CASTOR, linear EM) over a
dataset directory: one results.csv row per scenario with the pipeline's metric names so the
rows pair with the method's by scenario_id.

CASTOR learns the number of regimes, the regime of every sample and one linear SEM (with one
lag) per regime by EM, initialised from consecutive blocks of `window` samples. The series is
standardised globally; the lagged design is the series shifted by one sample. Scoring: every
true regime is matched to the CASTOR regime that holds most of its samples and that regime's
instantaneous graph (|W| > w_threshold) is scored against the true graph (mean over regimes:
mean_norm_shd, mean_skeleton_f1); the changes of the sample labels, merged within 160
samples, are scored as change points (det_tp / det_fp / det_fn at the 200-sample tolerance).

    python tools/run_castor.py --dataset_dir datasets --output_dir results/x/castor \
        --scenario_ids_file ids.txt --window 500 --workers 12 --resume --permute_variables
"""
import argparse
import contextlib
import io
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "NSD_Wavelets", "src"), os.path.join(ROOT, "causalmorph"), os.path.join(ROOT, "third_party", "CASTOR")):
    if path not in sys.path:
        sys.path.insert(0, path)

DEFAULTS = dict(window=500, w_threshold=0.4, max_it=5, loss_thres=1.0, zeta=100)   # the tutorial's settings, window = the detector's bound L


def castor_fit(X: np.ndarray, window: int, w_threshold: float, max_it: int, loss_thres: float, zeta: int):
    """(labels [T], graphs {label: A[from, to] binary}) from CASTOR's linear EM on the standardised series."""
    import torch
    torch.set_num_threads(1)
    # CASTOR's modules import each other by bare name (utils, linear_utils, ...); causalmorph has a
    # `utils` package too, so CASTOR's directory goes first and a foreign `utils` already imported is dropped
    castor_dir = os.path.join(ROOT, "third_party", "CASTOR")
    if sys.path[0] != castor_dir:
        sys.path.insert(0, castor_dir)
    for name in ("utils", "linear_utils", "time_varying_weights", "dynotears", "causal_structure"):
        m = sys.modules.get(name)
        if m is not None and not str(getattr(m, "__file__", "")).startswith(castor_dir):
            del sys.modules[name]
    from CASTOR import CASTOR
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xc, Xlag = Z[1:], Z[:-1]
    n = Xc.shape[1]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, _, gamma, L = CASTOR(pd.DataFrame(Xc), Xc, Xlag, lags=2).run_linear(max_it, loss_thres, w_threshold, window, zeta)
    lab = np.argmax(gamma, axis=1)
    labels = np.concatenate([[lab[0]], lab])            # the dropped first sample takes its neighbour's label
    graphs = {k: (np.abs(L[:n, :n, k]) > w_threshold).astype(float) for k in range(gamma.shape[1])}
    return labels, graphs


def score(labels, graphs, names, true_cps, true_adjs, T):
    from full_pipeline import compute_shd
    from cdnod_baseline import skeleton_prf
    from evaluation.metrics import evaluate_detection
    from evaluation.metrics import simplify_transitions
    bounds = [0] + list(true_cps) + [T]
    per, sk, matched = [], [], []
    for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg = labels[a:b]
        k = int(np.bincount(seg).argmax())
        A = pd.DataFrame(graphs[k], index=names, columns=names)
        per.append(compute_shd(A, true_adjs[r])); sk.append(skeleton_prf(A, true_adjs[r])["F1"]); matched.append(k)
    raw = [int(t) for t in range(1, T) if labels[t] != labels[t - 1]]
    cps = simplify_transitions(raw, min_distance=160)
    ev = evaluate_detection(true_cps, cps, tolerance=200)
    return {
        "mean_norm_shd": float(np.mean([x["normalized_shd"] for x in per])),
        "mean_struct_f1": float(np.mean([x["F1"] for x in per])),
        "mean_skeleton_f1": float(np.mean(sk)),
        "n_regimes_found": int(len(graphs)), "n_regimes_matched": int(len(set(matched))),
        "n_detected_cps": len(cps), "detected_cps": ";".join(map(str, cps)), "n_raw_label_changes": len(raw),
        "det_tp": ev.true_positives, "det_fp": ev.false_positives, "det_fn": ev.false_negatives,
        "det_precision": ev.precision, "det_recall": ev.recall,
    }


def _job(sid, index_row, dataset_dir, cfg, permute_variables):
    import warnings as _w
    _w.filterwarnings("ignore")
    from run_experiments import load_scenario
    meta = dict(scenario_id=sid, p=int(index_row["p"]), n_regimes=int(index_row["n_regimes"]),
                samples_regime=int(index_row.get("samples_regime") or 0), castor_window=cfg["window"], castor_w_threshold=cfg["w_threshold"])
    t0 = time.time()
    try:
        X, names, true_cps, true_adjs, _, _ = load_scenario(sid, dataset_dir, index_row, permute_variables=permute_variables)
        labels, graphs = castor_fit(X, **cfg)
        row = {**meta, **score(labels, graphs, names, true_cps, true_adjs, len(X)), "status": "ok"}
    except Exception:  # noqa: BLE001 — recorded per scenario, the run continues
        row = {**meta, "status": "error", "error_msg": traceback.format_exc(limit=3).strip()}
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True); ap.add_argument("--output_dir", required=True)
    ap.add_argument("--scenario_ids_file")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--permute_variables", action="store_true")
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}

    index = pd.read_csv(os.path.join(a.dataset_dir, "index.csv"), dtype=str)
    if a.scenario_ids_file:
        keep = {ln.strip() for ln in open(a.scenario_ids_file) if ln.strip()}
        index = index[index.scenario_id.isin(keep)]
    os.makedirs(a.output_dir, exist_ok=True)
    out = os.path.join(a.output_dir, "results.csv")
    if a.resume and os.path.exists(out):
        prev = pd.read_csv(out, dtype=str)
        index = index[~index.scenario_id.isin(set(prev.scenario_id))]
    print(f"CASTOR  {cfg}  scenarios to run: {len(index)}  workers={a.workers}", flush=True)

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
        print(f"  {done}/{len(rows)}  ({(time.time() - t0) / 60:.1f} min){'  errors: ' + str(n_err) if n_err else ''}", flush=True)

    buf = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_job, r.scenario_id, r, a.dataset_dir, cfg, a.permute_variables) for _, r in rows]
        for f in as_completed(futs):
            buf.append(f.result()); done += 1
            if len(buf) >= a.batch_size:
                flush(buf); buf = []
        if buf:
            flush(buf)
    print(f"Final save: {done} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
