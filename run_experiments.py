"""
Experiment runner for Full Pipeline — batched edition.

Loads pre-generated scenarios from datasets/ (produced by generate_datasets.py),
runs detection + CausalMorph + DirectLiNGAM + ICA-LiNGAM on each, and saves
results in batches of --batch_size scenarios.

Output layout
-------------
  {output_dir}/batch_0000.csv   — scenarios 0..batch_size-1
  {output_dir}/batch_0001.csv   — scenarios batch_size..2*batch_size-1
  ...

Each row includes all three algorithms (CausalMorph, DirectLiNGAM, ICA-LiNGAM)
side-by-side for direct comparison.

Resuming
--------
Batches whose CSV already exists are skipped entirely.  Within an in-progress
batch, completed scenario_ids are skipped.  It is safe to Ctrl-C and re-run.

Usage
-----
  python run_experiments.py                              # all datasets, batch=100
  python run_experiments.py --batch_size 50              # smaller batches
  python run_experiments.py --p_filter 5                 # only p=5 scenarios
  python run_experiments.py --dataset_dir my_data --output_dir my_results
"""

import os
import sys
import time
import argparse
import traceback
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(_HERE, "causalmorph"))

from full_pipeline import (
    detect_change_points,
    extract_causal_structures,
    extract_causal_structures_ablation,
    compute_shd,
    assign_true_regime,
    aggregate_structures_bayesian,
    ABLATION_METHODS,
)
from evaluation.metrics import evaluate_detection, simplify_transitions


# ── Load one scenario ─────────────────────────────────────────────────────────

def load_scenario(scenario_id: str, dataset_dir: str, index_row: pd.Series):
    """
    Load a scenario from its CSV pair.

    Returns
    -------
    X              : ndarray (T, p)
    variable_names : List[str]
    true_cps       : List[int]
    true_adjs      : List[DataFrame]  — binary per-regime adjacency matrices
    regime_sizes   : List[int]
    min_samples    : int  — samples per regime, used for min_window calc
    """
    dat_df = pd.read_csv(os.path.join(dataset_dir, f"{scenario_id}-dat.csv"))
    variable_names = list(dat_df.columns)
    X = dat_df.values.astype(float)

    true_cps    = [int(x) for x in str(index_row["change_points"]).split(";") if x.strip()]
    regime_sizes = [int(x) for x in str(index_row.get("regime_sizes", "")).split(";") if x.strip()]
    min_samples  = int(index_row.get("samples_regime") or index_row.get("min_samples_regime") or 600)

    am_df = pd.read_csv(os.path.join(dataset_dir, f"{scenario_id}-am.csv"), index_col="from_var")
    true_adjs = []
    for r_idx in sorted(am_df["regime_idx"].unique()):
        block = am_df[am_df["regime_idx"] == r_idx].drop(columns=["regime_idx"])
        block.index   = variable_names
        block.columns = variable_names
        true_adjs.append(block.astype(float))

    return X, variable_names, true_cps, true_adjs, regime_sizes, min_samples


# ── Detection post-processing (mirrors run_full_pipeline) ─────────────────────

def _postprocess_detections(det_result, X, p, min_regime_len, min_samples):
    refractory  = min(150, min_regime_len // 4)
    onset_set   = set(det_result.onset_points)
    first_onset = min(onset_set) if onset_set else float("inf")

    kept_offsets = [
        cp for cp in det_result.offset_points
        if (cp < first_onset) or all(abs(cp - on) > refractory * 2 for on in onset_set)
    ]
    raw_cps      = sorted(onset_set | set(kept_offsets))
    detected_cps = simplify_transitions(raw_cps, min_distance=160)

    min_window = max(min_samples // 2, 10 * p * p)
    _cps = sorted(detected_cps)
    while _cps:
        _bounds = [0] + _cps + [len(X)]
        _sizes  = [_bounds[i + 1] - _bounds[i] for i in range(len(_bounds) - 1)]
        if min(_sizes) >= min_window:
            break
        min_idx = _sizes.index(min(_sizes))
        if min_idx == 0:
            _cps.pop(0)
        elif min_idx == len(_sizes) - 1:
            _cps.pop(-1)
        else:
            left_cp  = _bounds[min_idx]
            right_cp = _bounds[min_idx + 1]
            keep = left_cp if (left_cp in onset_set or right_cp not in onset_set) else right_cp
            _cps.pop(min_idx)
            _cps[min_idx - 1] = keep
    return _cps


# ── Metric extraction ─────────────────────────────────────────────────────────

def _extract_metrics(
    scenario_id: str,
    p: int,
    n_regimes: int,
    n_changes: int,
    samples_regime: int,
    base_pconn: float,
    noise_fraction: float,
    seed: int,
    result: dict,
) -> dict:
    er             = result["eval_result"]
    structs        = result["structures"]
    true_adjs      = result["true_adjs"]
    consensus_adj  = result["consensus_adj"]
    variable_names = result["variable_names"]

    # Ground-truth final-target adjacency (shared across all method comparisons)
    gt_raw = true_adjs[-1]
    gt_arr = np.asarray(gt_raw.values if hasattr(gt_raw, "values") else gt_raw)
    gt_df  = pd.DataFrame(
        (gt_arr != 0).astype(float), index=variable_names, columns=variable_names
    )

    # ── Detection ─────────────────────────────────────────────────────────────
    det = {
        "det_f1":         round(er.f1_score, 4),
        "det_precision":  round(er.precision, 4),
        "det_recall":     round(er.recall, 4),
        "det_tp":         er.true_positives,
        "det_fp":         er.false_positives,
        "det_fn":         er.false_negatives,
        "n_true_cps":     len(result["true_change_points"]),
        "n_detected_cps": len(result["detected_change_points"]),
        "total_samples":  len(result["X"]),
    }

    # ── CausalMorph per-regime and consensus ──────────────────────────────────
    metrics_list = [s.shd_metrics for s in structs if s.shd_metrics]
    if metrics_list:
        cm_struct = {
            "n_structures":          len(structs),
            "mean_shd":              round(np.mean([m["SHD"]            for m in metrics_list]), 3),
            "mean_norm_shd":         round(np.mean([m["normalized_shd"] for m in metrics_list]), 3),
            "mean_struct_f1":        round(np.mean([m["F1"]             for m in metrics_list]), 3),
            "mean_struct_precision": round(np.mean([m["Precision"]      for m in metrics_list]), 3),
            "mean_struct_recall":    round(np.mean([m["Recall"]         for m in metrics_list]), 3),
        }
    else:
        cm_struct = {k: np.nan for k in (
            "n_structures", "mean_shd", "mean_norm_shd",
            "mean_struct_f1", "mean_struct_precision", "mean_struct_recall",
        )}
        cm_struct["n_structures"] = len(structs)

    cm_shd = compute_shd(consensus_adj, gt_df)
    cm_consensus = {
        "consensus_shd":       cm_shd["SHD"],
        "consensus_norm_shd":  cm_shd["normalized_shd"],
        "consensus_f1":        cm_shd["F1"],
        "consensus_precision": cm_shd["Precision"],
        "consensus_recall":    cm_shd["Recall"],
    }

    # ── Ablation: DirectLiNGAM and ICA-LiNGAM ────────────────────────────────
    ablation_out = {}
    if "ablation" in result and result["ablation"]:
        for method in ABLATION_METHODS:
            structs_m = result["ablation"].get(method, [])
            metrics_m = [s.shd_metrics for s in structs_m if s.shd_metrics]
            if metrics_m:
                ablation_out[f"{method}_mean_shd"]       = round(np.mean([m["SHD"]            for m in metrics_m]), 3)
                ablation_out[f"{method}_mean_norm_shd"]  = round(np.mean([m["normalized_shd"] for m in metrics_m]), 3)
                ablation_out[f"{method}_mean_f1"]        = round(np.mean([m["F1"]             for m in metrics_m]), 3)
                ablation_out[f"{method}_mean_precision"] = round(np.mean([m["Precision"]      for m in metrics_m]), 3)
                ablation_out[f"{method}_mean_recall"]    = round(np.mean([m["Recall"]         for m in metrics_m]), 3)
            else:
                for sfx in ("mean_shd", "mean_norm_shd", "mean_f1", "mean_precision", "mean_recall"):
                    ablation_out[f"{method}_{sfx}"] = np.nan

            ac = result.get("ablation_consensus", {}).get(method)
            if ac:
                ab_shd = compute_shd(ac["consensus_adj"], gt_df)
                ablation_out[f"{method}_cons_norm_shd"] = ab_shd["normalized_shd"]
                ablation_out[f"{method}_cons_f1"]       = ab_shd["F1"]
                ablation_out[f"{method}_cons_precision"] = ab_shd["Precision"]
                ablation_out[f"{method}_cons_recall"]   = ab_shd["Recall"]
            else:
                for sfx in ("cons_norm_shd", "cons_f1", "cons_precision", "cons_recall"):
                    ablation_out[f"{method}_{sfx}"] = np.nan

    return {
        # ── identification ───────────────────────────────────────────────────
        "scenario_id":    scenario_id,
        "p":              p,
        "n_regimes":      n_regimes,
        "n_changes":      n_changes,
        "samples_regime": samples_regime,
        "base_pconn":     base_pconn,
        "noise_fraction": noise_fraction,
        "seed":           seed,
        # ── detection ────────────────────────────────────────────────────────
        **det,
        # ── CausalMorph per-regime ────────────────────────────────────────────
        **cm_struct,
        # ── CausalMorph consensus ─────────────────────────────────────────────
        **cm_consensus,
        # ── ablation (DirectLiNGAM, ICA-LiNGAM) ──────────────────────────────
        **ablation_out,
    }


# ── Single experiment ─────────────────────────────────────────────────────────

def run_one(
    scenario_id: str,
    index_row: pd.Series,
    dataset_dir: str,
    window_overlap: float,
    detector_version: str,
    prior_mode: str,
    suppress_output: bool,
) -> dict:
    """Run one scenario (always with ablation). Returns a flat metrics dict."""
    ctx = contextlib.redirect_stdout(io.StringIO()) if suppress_output else contextlib.nullcontext()

    p              = int(index_row["p"])
    n_regimes      = int(index_row["n_regimes"])
    seed           = int(index_row["seed"])
    n_changes      = int(index_row.get("n_changes") or n_regimes - 1)
    samples_regime = int(index_row.get("samples_regime") or index_row.get("min_samples_regime") or 600)
    base_pconn     = float(index_row.get("base_pconn") or 0.35)
    noise_fraction = float(index_row.get("noise_fraction") or 0.08)

    try:
        with ctx:
            X, variable_names, true_cps, true_adjs, regime_sizes, min_samples = load_scenario(
                scenario_id, dataset_dir, index_row
            )

            # Reconstruct initial prior from regime-0 adjacency
            adj0 = true_adjs[0].values
            G0   = nx.DiGraph()
            G0.add_nodes_from(variable_names)
            for i, src in enumerate(variable_names):
                for j, dst in enumerate(variable_names):
                    if adj0[i, j] != 0:
                        G0.add_edge(src, dst)
            try:
                gt_order_names = list(nx.topological_sort(G0))
            except nx.NetworkXUnfeasible:
                gt_order_names = variable_names
            initial_order = [variable_names.index(v) for v in gt_order_names]
            initial_adj   = pd.DataFrame(adj0, columns=variable_names, index=variable_names)

            # Detection
            min_regime_len = min(regime_sizes) if regime_sizes else len(X) // max(n_regimes, 1)
            baseline_end   = min(150, min_regime_len // 4)

            det_result   = detect_change_points(
                X, baseline_end=baseline_end, min_regime_len=min_regime_len,
                seed=seed, detector_version=detector_version,
            )
            detected_cps = _postprocess_detections(det_result, X, p, min_regime_len, min_samples)
            eval_result  = evaluate_detection(true_cps, detected_cps, tolerance=200)

            # CausalMorph structure extraction
            structures = extract_causal_structures(
                X=X,
                detected_change_points=detected_cps,
                variable_names=variable_names,
                initial_adj=initial_adj,
                initial_order=initial_order,
                window_overlap=window_overlap,
                verbose=False,
                prior_mode=prior_mode,
            )
            T = len(X)
            for s in structures:
                tr = assign_true_regime(s.window_start, s.window_end, true_cps, T)
                s.true_regime_idx       = tr
                s.true_adjacency_matrix = true_adjs[tr]
                s.shd_metrics           = compute_shd(s.adjacency_matrix, true_adjs[tr])

            # Ablation: DirectLiNGAM + ICA-LiNGAM on the same windows
            ablation_results = extract_causal_structures_ablation(
                X=X,
                detected_change_points=detected_cps,
                variable_names=variable_names,
                initial_adj=initial_adj,
                initial_order=initial_order,
                window_overlap=window_overlap,
                verbose=False,
                prior_mode=prior_mode,
            )
            for structs_m in ablation_results.values():
                for s in structs_m:
                    tr = assign_true_regime(s.window_start, s.window_end, true_cps, T)
                    s.true_regime_idx       = tr
                    s.true_adjacency_matrix = true_adjs[tr]
                    s.shd_metrics           = compute_shd(s.adjacency_matrix, true_adjs[tr])

            ablation_consensus = {}
            for method, structs_m in ablation_results.items():
                c_adj, c_probs = aggregate_structures_bayesian(structs_m, variable_names)
                ablation_consensus[method] = {"consensus_adj": c_adj, "edge_probs": c_probs}

            consensus_adj, edge_probs = aggregate_structures_bayesian(structures, variable_names)

            result = {
                "structures":             structures,
                "detected_change_points": detected_cps,
                "true_change_points":     true_cps,
                "X":                      X,
                "variable_names":         variable_names,
                "detection_result":       det_result,
                "eval_result":            eval_result,
                "consensus_adj":          consensus_adj,
                "edge_probs":             edge_probs,
                "true_adjs":              true_adjs,
                "ablation":               ablation_results,
                "ablation_consensus":     ablation_consensus,
            }

        row = _extract_metrics(
            scenario_id, p, n_regimes, n_changes, samples_regime, base_pconn,
            noise_fraction, seed, result,
        )
        row["status"]    = "ok"
        row["error_msg"] = ""

    except Exception:
        row = {
            "scenario_id":    scenario_id,
            "p":              p,
            "n_regimes":      n_regimes,
            "n_changes":      n_changes,
            "samples_regime": samples_regime,
            "base_pconn":     base_pconn,
            "noise_fraction": noise_fraction,
            "seed":           seed,
            "status":         "error",
            "error_msg":      traceback.format_exc(limit=3).strip(),
        }

    return row


# ── Batch summary helper ──────────────────────────────────────────────────────

def _print_batch_summary(rows: list, batch_idx: int):
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        print(f"  Batch {batch_idx:04d} — no successful rows")
        return
    df = pd.DataFrame(ok_rows)
    det_f1   = df["det_f1"].mean()   if "det_f1"   in df else float("nan")
    nshd     = df["mean_norm_shd"].mean() if "mean_norm_shd" in df else float("nan")
    cons_f1  = df["consensus_f1"].mean()  if "consensus_f1"  in df else float("nan")
    dl_nshd  = df["directlingam_mean_norm_shd"].mean() if "directlingam_mean_norm_shd" in df else float("nan")
    ica_nshd = df["icalingam_mean_norm_shd"].mean()    if "icalingam_mean_norm_shd"    in df else float("nan")
    n_ok  = len(ok_rows)
    n_err = len(rows) - n_ok
    print(
        f"\n  ── Batch {batch_idx:04d} ({n_ok} ok / {n_err} err) ──"
        f"  det_f1={det_f1:.3f}"
        f"  CM_nSHD={nshd:.3f}  CM_cons_f1={cons_f1:.3f}"
        f"  DL_nSHD={dl_nshd:.3f}"
        f"  ICA_nSHD={ica_nshd:.3f}"
    )


# ── Global summary helper ─────────────────────────────────────────────────────

def _print_global_summary(output_dir: str):
    batch_files = sorted(
        f for f in os.listdir(output_dir) if f.startswith("batch_") and f.endswith(".csv")
    )
    if not batch_files:
        return

    dfs = [pd.read_csv(os.path.join(output_dir, f)) for f in batch_files]
    df_all = pd.concat(dfs, ignore_index=True)
    ok     = df_all[df_all["status"] == "ok"]

    print(f"\n{'='*70}")
    print(f"GLOBAL SUMMARY  ({len(ok)}/{len(df_all)} succeeded  across {len(batch_files)} batches)")
    print(f"{'='*70}")

    cols = [
        ("det_f1",                     "Detection F1"),
        ("det_precision",              "Detection Precision"),
        ("det_recall",                 "Detection Recall"),
        ("mean_norm_shd",              "CausalMorph  mean nSHD"),
        ("mean_struct_f1",             "CausalMorph  mean F1"),
        ("consensus_norm_shd",         "CausalMorph  consensus nSHD"),
        ("consensus_f1",               "CausalMorph  consensus F1"),
        ("directlingam_mean_norm_shd", "DirectLiNGAM mean nSHD"),
        ("directlingam_mean_f1",       "DirectLiNGAM mean F1"),
        ("directlingam_cons_f1",       "DirectLiNGAM consensus F1"),
        ("icalingam_mean_norm_shd",    "ICA-LiNGAM   mean nSHD"),
        ("icalingam_mean_f1",          "ICA-LiNGAM   mean F1"),
        ("icalingam_cons_f1",          "ICA-LiNGAM   consensus F1"),
    ]
    for col, label in cols:
        if col in ok.columns and ok[col].notna().any():
            print(f"  {label:<30}  mean={ok[col].mean():.3f}  std={ok[col].std():.3f}")

    # Per-p table
    if "p" in ok.columns:
        print(f"\n  {'p':>3}   n   det_f1  CM_nSHD  DL_nSHD  ICA_nSHD  CM_F1  DL_F1  ICA_F1")
        print(f"  {'─'*70}")
        for pval in sorted(ok["p"].astype(str).unique(), key=int):
            sub = ok[ok["p"].astype(str) == pval]
            def m(c): return sub[c].mean() if c in sub else float("nan")
            print(
                f"  {pval:>3}  {len(sub):>4}"
                f"  {m('det_f1'):.3f}"
                f"   {m('mean_norm_shd'):.3f}"
                f"    {m('directlingam_mean_norm_shd'):.3f}"
                f"     {m('icalingam_mean_norm_shd'):.3f}"
                f"    {m('consensus_f1'):.3f}"
                f"  {m('directlingam_cons_f1'):.3f}"
                f"  {m('icalingam_cons_f1'):.3f}"
            )

    print(f"{'='*70}")


# ── Main batch runner ─────────────────────────────────────────────────────────

def run_experiments(
    dataset_dir: str      = "datasets",
    output_dir: str       = "results",
    batch_size: int       = 100,
    detector_version: str = "v1G",
    prior_mode: str       = "chain",
    window_overlap: float = 0.0,
    suppress_output: bool = True,
    p_filter: int         = None,
):
    """
    Run all scenarios in dataset_dir/index.csv in batches of batch_size.

    Each batch is saved to {output_dir}/batch_{idx:04d}.csv.
    Completed batches (file exists + all scenarios present) are skipped.
    """
    index_path = os.path.join(dataset_dir, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No index.csv found in {dataset_dir!r}. Run generate_datasets.py first."
        )

    index_df = pd.read_csv(index_path, dtype=str)
    if p_filter is not None:
        index_df = index_df[index_df["p"] == str(p_filter)]

    os.makedirs(output_dir, exist_ok=True)

    scenarios = list(index_df.iterrows())
    total     = len(scenarios)
    n_batches = (total + batch_size - 1) // batch_size

    print("=" * 70)
    print("Experiment Runner — Full Pipeline (batched)")
    print("=" * 70)
    print(f"  dataset_dir : {dataset_dir}/")
    print(f"  output_dir  : {output_dir}/")
    print(f"  detector    : {detector_version}")
    print(f"  prior_mode  : {prior_mode}")
    print(f"  methods     : CausalMorph + DirectLiNGAM + ICA-LiNGAM")
    print(f"  scenarios   : {total}  ({n_batches} batches of {batch_size})")
    if p_filter:
        print(f"  p_filter    : {p_filter}")
    print()

    t_global = time.time()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end   = min(batch_start + batch_size, total)
        batch_rows  = scenarios[batch_start:batch_end]

        batch_path  = os.path.join(output_dir, f"batch_{batch_idx:04d}.csv")

        # Check how many are already done in this batch
        done_ids = set()
        if os.path.exists(batch_path):
            try:
                done_df  = pd.read_csv(batch_path, dtype=str)
                done_ids = set(done_df["scenario_id"].tolist())
            except Exception:
                done_ids = set()

        remaining_in_batch = [
            (_, row) for _, row in batch_rows
            if str(row["scenario_id"]) not in done_ids
        ]

        if not remaining_in_batch:
            print(f"  Batch {batch_idx:04d}/{n_batches-1}  [{batch_start}–{batch_end-1}]  COMPLETE — skip")
            continue

        print(
            f"\n  Batch {batch_idx:04d}/{n_batches-1}"
            f"  [{batch_start}–{batch_end-1}]"
            f"  ({len(remaining_in_batch)} to run, {len(done_ids)} already done)"
        )

        t_batch  = time.time()
        new_rows = []

        for local_idx, (_, index_row) in enumerate(remaining_in_batch, start=1):
            sid     = str(index_row["scenario_id"])
            elapsed = time.time() - t_batch
            global_pos = batch_start + batch_size - len(remaining_in_batch) + local_idx
            print(
                f"    [{local_idx:>3}/{len(remaining_in_batch)}]"
                f"  (global {global_pos}/{total})"
                f"  {sid}"
                f"  {elapsed:.0f}s",
                end="  ", flush=True,
            )

            row = run_one(
                scenario_id=sid,
                index_row=index_row,
                dataset_dir=dataset_dir,
                window_overlap=window_overlap,
                detector_version=detector_version,
                prior_mode=prior_mode,
                suppress_output=suppress_output,
            )
            new_rows.append(row)

            # Append to batch CSV immediately (crash-safe)
            df_row = pd.DataFrame([row])
            write_header = not os.path.exists(batch_path) or os.path.getsize(batch_path) == 0
            df_row.to_csv(batch_path, mode="a", header=write_header, index=False)

            if row["status"] == "ok":
                cm_nshd  = row.get("mean_norm_shd",              float("nan"))
                dl_nshd  = row.get("directlingam_mean_norm_shd", float("nan"))
                ica_nshd = row.get("icalingam_mean_norm_shd",    float("nan"))
                print(
                    f"CM={cm_nshd:.3f}"
                    f"  DL={dl_nshd:.3f}"
                    f"  ICA={ica_nshd:.3f}"
                    f"  det_f1={row.get('det_f1', float('nan')):.2f}"
                )
            else:
                print(f"ERROR: {row['error_msg'][:80]}")

        _print_batch_summary(new_rows, batch_idx)

    _print_global_summary(output_dir)
    print(f"\nTotal elapsed: {time.time() - t_global:.0f}s")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run causal pipeline experiments on pre-generated datasets.\n"
            "Always runs CausalMorph, DirectLiNGAM, and ICA-LiNGAM per window."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_dir",    type=str,   default="datasets",
                        help="Directory with index.csv and scenario CSVs (default: datasets/)")
    parser.add_argument("--output_dir",     type=str,   default="results",
                        help="Directory to write batch CSVs (default: results/)")
    parser.add_argument("--batch_size",     type=int,   default=100,
                        help="Scenarios per batch CSV (default: 100)")
    parser.add_argument("--detector",       type=str,   default="v1G", choices=["v1F", "v1G"],
                        help="Detector version (default: v1G)")
    parser.add_argument("--prior_mode",     type=str,   default="chain", choices=["anchor", "chain"],
                        help="CausalMorph prior mode (default: chain)")
    parser.add_argument("--window_overlap", type=float, default=0.0,
                        help="CausalMorph window overlap fraction (default: 0.0)")
    parser.add_argument("--verbose",        action="store_true",
                        help="Show per-run pipeline output (very noisy)")
    parser.add_argument("--p_filter",       type=int,   default=None,
                        help="Only run scenarios with this node count")
    args = parser.parse_args()

    run_experiments(
        dataset_dir     = args.dataset_dir,
        output_dir      = args.output_dir,
        batch_size      = args.batch_size,
        detector_version= args.detector,
        prior_mode      = args.prior_mode,
        window_overlap  = args.window_overlap,
        suppress_output = not args.verbose,
        p_filter        = args.p_filter,
    )
