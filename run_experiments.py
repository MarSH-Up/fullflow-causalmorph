"""
Experiment runner for Full Pipeline.

Loads pre-generated scenarios from datasets/ (produced by generate_datasets.py),
runs detection + CausalMorph + DirectLiNGAM + ICA-LiNGAM on each, and appends
to results.csv in output_dir every --batch_size scenarios. Rows are flushed and
dropped, so memory holds one batch rather than the whole result set.

Output
------
  {output_dir}/results.csv   — one row per scenario, all methods side-by-side

Each row includes all three algorithms (CausalMorph, DirectLiNGAM, ICA-LiNGAM)
plus per-regime SHD columns (cm_shd_0, cm_shd_1, … / directlingam_shd_0, …).

Usage
-----
  python run_experiments.py                              # all datasets, auto worker count
  python run_experiments.py --workers 64                 # explicit parallelism
  python run_experiments.py --batch_size 50              # smaller progress groups
  python run_experiments.py --p_filter 5                 # only p=5 scenarios
  python run_experiments.py --resume                     # continue an interrupted run
  python run_experiments.py --dataset_dir my_data --output_dir my_results
"""

import os
import sys
import math
import time
import argparse
import traceback
import contextlib
import io
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import List, Optional
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(_HERE, "causalmorph"))


def _init_worker(here_path: str):
    """Initializer for worker processes: ensures submodule paths are on sys.path."""
    sys.path.insert(0, os.path.join(here_path, "NSD_Wavelets", "src"))
    sys.path.insert(0, os.path.join(here_path, "causalmorph"))
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

from full_pipeline import (
    detect_change_points,
    extract_causal_structures,
    extract_causal_structures_ablation,
    compute_shd,
    assign_true_regime,
    aggregate_structures_bayesian,
    ABLATION_METHODS,
)
from evaluation.metrics import (
    evaluate_detection,
    simplify_transitions,
    windowed_nshd_weighted,
    coverage_score,
    transition_shd,
    dynamic_score,
)


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
    min_samples  = int(index_row.get("min_samples_regime") or index_row.get("samples_regime") or 600)

    am_df = pd.read_csv(os.path.join(dataset_dir, f"{scenario_id}-am.csv"), index_col="from_var")
    true_adjs = []
    for r_idx in sorted(am_df["regime_idx"].unique()):
        block = am_df[am_df["regime_idx"] == r_idx].drop(columns=["regime_idx"])
        block.index   = variable_names
        block.columns = variable_names
        true_adjs.append(block.astype(float))

    return X, variable_names, true_cps, true_adjs, regime_sizes, min_samples


# ── Detection post-processing (mirrors run_full_pipeline) ─────────────────────

def detection_cps(det_result) -> List[int]:
    """
    The detector's change points, scored as detection: union of onset and
    offset points, with detections closer than 160 samples merged. No
    extraction constraint is applied here (see extraction_windows).
    """
    raw = sorted(set(det_result.onset_points) | set(det_result.offset_points))
    return simplify_transitions(raw, min_distance=160)


def legacy_prune(cps: List[int], onset_set: set, T: int, p: int, min_samples: int) -> List[int]:
    """
    The pre-2026-10-06 post-processing, kept verbatim for --legacy_postprocessing:
    repeatedly delete the change point bounding the shortest window until every
    window has at least max(min_samples // 2, 10 * p * p) samples, preferring to
    keep onsets. The result was used both for scoring detection and for
    extraction, so a change point the detector found next to a short regime
    counted as missed. For p >= 8 the rule exceeds the grid's 500-sample
    regimes and deletes every change point adjacent to one.
    """
    min_window = max(int(min_samples) // 2, 10 * int(p) * int(p))
    _cps = sorted(int(c) for c in cps)
    while _cps:
        _bounds = [0] + _cps + [T]
        _sizes = [_bounds[i + 1] - _bounds[i] for i in range(len(_bounds) - 1)]
        if min(_sizes) >= min_window:
            break
        min_idx = _sizes.index(min(_sizes))
        if min_idx == 0:
            _cps.pop(0)
        elif min_idx == len(_sizes) - 1:
            _cps.pop(-1)
        else:
            left_cp, right_cp = _bounds[min_idx], _bounds[min_idx + 1]
            keep = left_cp if (left_cp in onset_set or right_cp not in onset_set) else right_cp
            _cps.pop(min_idx)
            _cps[min_idx - 1] = keep
    return _cps


def min_window_for(p: int, samples_per_node: float, floor: int = 250) -> int:
    """
    Smallest window the extraction step will fit a model on: max(floor, c*p).
    Linear in p because DirectLiNGAM's ordering stabilises once n is a modest
    multiple of p; the previous 10*p^2 rule exceeded the grid's 500-sample
    regimes from p=8 and silently deleted every change point next to one.
    """
    return max(int(floor), int(math.ceil(samples_per_node * p)))


def extraction_windows(cps: List[int], T: int, n_min: int):
    """
    Windows for the extraction step: segments shorter than n_min are merged
    into their shorter neighbour until every window has at least n_min
    samples. Returns (cps_for_extraction, absorbed_cps). An absorbed change
    point still counts as detected; it is reported as not extracted.
    """
    cps = sorted(int(c) for c in cps)
    absorbed: List[int] = []
    while cps:
        bounds = [0] + cps + [T]
        sizes = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        k = int(np.argmin(sizes))
        if sizes[k] >= n_min:
            break
        if k == 0:
            drop = cps[0]
        elif k == len(sizes) - 1:
            drop = cps[-1]
        else:
            left, right = sizes[k - 1], sizes[k + 1]
            drop = bounds[k] if left <= right else bounds[k + 1]   # boundary shared with the shorter neighbour
        cps.remove(drop)
        absorbed.append(drop)
    if not cps and T < n_min and absorbed == []:
        pass
    return cps, absorbed


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
        "det_precision":  round(er.precision, 4),
        "det_recall":     round(er.recall, 4),
        "det_tp":         er.true_positives,
        "det_fp":         er.false_positives,
        "det_fn":         er.false_negatives,
        "n_true_cps":     len(result["true_change_points"]),
        "true_cps":       ";".join(str(x) for x in result["true_change_points"]),
        "n_detected_cps": len(result["detected_change_points"]),
        "detected_cps":   ";".join(str(x) for x in result["detected_change_points"]),
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
        cm_struct["cm_norm_shd_per_regime"] = ";".join(str(round(m["normalized_shd"], 3)) for m in metrics_list)
    else:
        cm_struct = {k: np.nan for k in (
            "n_structures", "mean_shd", "mean_norm_shd",
            "mean_struct_f1", "mean_struct_precision", "mean_struct_recall",
        )}
        cm_struct["n_structures"]           = len(structs)
        cm_struct["cm_norm_shd_per_regime"] = ""

    cm_shd = compute_shd(consensus_adj, gt_df)
    cm_consensus = {
        "consensus_shd":       cm_shd["SHD"],
        "consensus_norm_shd":  cm_shd["normalized_shd"],
        "consensus_f1":        cm_shd["F1"],
        "consensus_precision": cm_shd["Precision"],
        "consensus_recall":    cm_shd["Recall"],
    }

    # ── Dynamic / non-stationary metrics (CausalMorph) ───────────────────────
    # Replace unweighted mean nSHD with length-weighted alignment-aware metrics.
    # Captures: coverage of missed regimes, transition tracking, composite score.
    T_full       = len(result["X"])
    det_f1       = er.f1_score
    true_cps_lst = result["true_change_points"]
    cm_dyn = {
        "windowed_nshd_weighted": round(windowed_nshd_weighted(structs, true_adjs, true_cps_lst, T_full, p), 4),
        "coverage_score":         round(coverage_score(structs, true_cps_lst, T_full), 4),
        "transition_shd":         round(transition_shd(structs, true_adjs, true_cps_lst, T_full, p), 4),
    }
    cm_dyn["dynamic_score"] = round(
        dynamic_score(cm_dyn["windowed_nshd_weighted"], det_f1, cm_dyn["coverage_score"]), 4
    )

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
                ablation_out[f"{method}_norm_shd_per_regime"] = ";".join(str(round(m["normalized_shd"], 3)) for m in metrics_m)
            else:
                for sfx in ("mean_shd", "mean_norm_shd", "mean_f1", "mean_precision", "mean_recall"):
                    ablation_out[f"{method}_{sfx}"] = np.nan
                ablation_out[f"{method}_norm_shd_per_regime"] = ""

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

            # Dynamic metrics per ablation method
            w_a = windowed_nshd_weighted(structs_m, true_adjs, true_cps_lst, T_full, p)
            c_a = coverage_score(structs_m, true_cps_lst, T_full)
            t_a = transition_shd(structs_m, true_adjs, true_cps_lst, T_full, p)
            ablation_out[f"{method}_windowed_nshd_weighted"] = round(w_a, 4)
            ablation_out[f"{method}_coverage_score"]         = round(c_a, 4)
            ablation_out[f"{method}_transition_shd"]         = round(t_a, 4)
            ablation_out[f"{method}_dynamic_score"]          = round(dynamic_score(w_a, det_f1, c_a), 4)

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
        # ── CausalMorph dynamic / non-stationary metrics ─────────────────────
        **cm_dyn,
        # ── CausalMorph consensus ─────────────────────────────────────────────
        **cm_consensus,
        # ── ablation (DirectLiNGAM, ICA-LiNGAM) ──────────────────────────────
        **ablation_out,
    }


# ── Single experiment ─────────────────────────────────────────────────────────

def run_loaded(
    X: np.ndarray,
    variable_names: List[str],
    true_cps: List[int],
    true_adjs: List[pd.DataFrame],
    regime_sizes: List[int],
    min_samples: int,
    meta: dict,
    window_overlap: float,
    detector_version: str,
    prior_mode: str,
    suppress_output: bool = True,
    hybrid: bool = False,
    bootstrap_n: int = 20,
    bootstrap_threshold: float = 0.60,
    chain_prior_threshold: float = 0.05,
    oracle_anchor: bool = False,
    oracle_segmentation: bool = False,
    no_segmentation: bool = False,
    samples_per_node: float = 40.0,
    fixed_min_regime_len: Optional[int] = None,
    consensus_threshold: float = 0.5,
    consensus_prior_density: Optional[float] = None,
    chain_soft_lambda: Optional[float] = None,
    corr_gamma: float = 2.0,
    adaptive_null: bool = False,
    det_tolerance: int = 200,
    run_ablation: bool = True,
    legacy_postprocessing: bool = False,
) -> dict:
    """
    Run the pipeline on an already-loaded scenario. Returns a flat metrics dict.

    Experimental conditions (all off by default = the honest configuration):
      oracle_anchor        window 0 is seeded with the true regime-0 graph —
                           the configuration of every result before 2026-10-06;
                           kept as a labelled upper bound.
      oracle_segmentation  extraction runs on the true change points: ceiling
                           of the extraction layer, isolates detection cost.
      no_segmentation      one window over the whole series: the
                           "ignore non-stationarity" control.
      fixed_min_regime_len use this instead of min(true regime sizes) to size
                           the detector's windows; None keeps the oracle value.
      legacy_postprocessing
                           the pre-2026-10-06 deletion rule (see legacy_prune),
                           applied before scoring detection; no merging. With
                           --oracle_anchor and --consensus_threshold 0.2 this
                           reproduces the old configuration exactly, so it can
                           be re-scored under the corrected adjacency convention.
    """
    ctx = contextlib.redirect_stdout(io.StringIO()) if suppress_output else contextlib.nullcontext()
    p = int(meta["p"]); n_regimes = int(meta["n_regimes"]); seed = int(meta["seed"])
    T = len(X)

    try:
        with ctx:
            if oracle_anchor:
                adj0 = true_adjs[0].values
                G0 = nx.DiGraph(); G0.add_nodes_from(variable_names)
                for i, src in enumerate(variable_names):
                    for j, dst in enumerate(variable_names):
                        if adj0[i, j] != 0:
                            G0.add_edge(src, dst)
                try:
                    gt_order_names = list(nx.topological_sort(G0))
                except nx.NetworkXUnfeasible:
                    gt_order_names = variable_names
                initial_order = [variable_names.index(v) for v in gt_order_names]
                initial_adj = pd.DataFrame(adj0, columns=variable_names, index=variable_names)
            else:
                initial_order, initial_adj = None, None

            # ── Detection ────────────────────────────────────────────────────
            if fixed_min_regime_len is not None:
                min_regime_len = int(fixed_min_regime_len)
            else:
                min_regime_len = min(regime_sizes) if regime_sizes else T // max(n_regimes, 1)
            baseline_end = max(150, min_regime_len // 4)

            if no_segmentation:
                det_result, cps_det = None, []
            elif oracle_segmentation:
                det_result, cps_det = None, list(true_cps)
            else:
                det_result = detect_change_points(
                    X, baseline_end=baseline_end, min_regime_len=min_regime_len,
                    seed=seed, detector_version=detector_version,
                    corr_gamma=corr_gamma, adaptive_null=adaptive_null,
                )
                cps_det = detection_cps(det_result)
            if legacy_postprocessing:
                onset_set = set(det_result.onset_points) if det_result is not None else set(cps_det)
                cps_det = legacy_prune(cps_det, onset_set, T, p, min_samples)
            eval_result = evaluate_detection(true_cps, cps_det, tolerance=det_tolerance)

            # ── Extraction windows (merge, never delete) ─────────────────────
            if legacy_postprocessing:
                n_min = max(int(min_samples) // 2, 10 * p * p)
                cps_ext, absorbed = list(cps_det), []
            else:
                n_min = min_window_for(p, samples_per_node)
                cps_ext, absorbed = extraction_windows(cps_det, T, n_min)

            structures = extract_causal_structures(
                X=X, detected_change_points=cps_ext, variable_names=variable_names,
                initial_adj=initial_adj, initial_order=initial_order,
                window_overlap=window_overlap, verbose=False, prior_mode=prior_mode,
                hybrid=hybrid, bootstrap_n=bootstrap_n, bootstrap_threshold=bootstrap_threshold,
                chain_prior_edge_threshold=chain_prior_threshold,
                chain_soft_lambda=chain_soft_lambda,
                prior_density=(consensus_prior_density if consensus_prior_density is not None else 0.2),
            )
            for s_ in structures:
                tr = assign_true_regime(s_.window_start, s_.window_end, true_cps, T)
                s_.true_regime_idx = tr
                s_.true_adjacency_matrix = true_adjs[tr]
                s_.shd_metrics = compute_shd(s_.adjacency_matrix, true_adjs[tr])

            ablation_results, ablation_consensus = {}, {}
            if run_ablation:
                ablation_results = extract_causal_structures_ablation(
                    X=X, detected_change_points=cps_ext, variable_names=variable_names,
                    window_overlap=window_overlap, verbose=False, seed=seed,
                )
                for structs_m in ablation_results.values():
                    for s_ in structs_m:
                        tr = assign_true_regime(s_.window_start, s_.window_end, true_cps, T)
                        s_.true_regime_idx = tr
                        s_.true_adjacency_matrix = true_adjs[tr]
                        s_.shd_metrics = compute_shd(s_.adjacency_matrix, true_adjs[tr])
                for method, structs_m in ablation_results.items():
                    c_adj, c_probs = aggregate_structures_bayesian(
                        structs_m, variable_names, edge_threshold=consensus_threshold,
                        prior_density=consensus_prior_density,
                    )
                    ablation_consensus[method] = {"consensus_adj": c_adj, "edge_probs": c_probs}

            consensus_adj, edge_probs = aggregate_structures_bayesian(
                structures, variable_names, edge_threshold=consensus_threshold,
                prior_density=consensus_prior_density,
            )

            result = {
                "structures": structures, "detected_change_points": cps_det,
                "true_change_points": true_cps, "X": X, "variable_names": variable_names,
                "detection_result": det_result, "eval_result": eval_result,
                "consensus_adj": consensus_adj, "edge_probs": edge_probs, "true_adjs": true_adjs,
                "ablation": ablation_results, "ablation_consensus": ablation_consensus,
            }

        row = _extract_metrics(
            meta["scenario_id"], p, n_regimes, int(meta["n_changes"]), int(meta["samples_regime"]),
            float(meta["base_pconn"]), float(meta["noise_fraction"]), seed, result,
        )
        row.update({
            "n_cps_merged":       len(absorbed),
            "cps_merged":         ";".join(str(c) for c in sorted(absorbed)),
            "n_windows_ext":      len(cps_ext) + 1,
            "n_min_window":       n_min,
            "w0_prior_source":    structures[0].prior_source if structures else "",
            "anchor_mode":        "oracle" if oracle_anchor else "cold",
            "segmentation_mode":  ("none" if no_segmentation else
                                   "oracle" if oracle_segmentation else detector_version),
            "min_regime_len_used": min_regime_len,
            "postprocessing_mode": "legacy" if legacy_postprocessing else "merge",
            "status":             "ok",
            "error_msg":          "",
        })

    except Exception:
        row = {
            "scenario_id":    meta["scenario_id"],
            "p":              p,
            "n_regimes":      n_regimes,
            "n_changes":      int(meta["n_changes"]),
            "samples_regime": int(meta["samples_regime"]),
            "base_pconn":     float(meta["base_pconn"]),
            "noise_fraction": float(meta["noise_fraction"]),
            "seed":           seed,
            "status":         "error",
            "error_msg":      traceback.format_exc(limit=3).strip(),
        }

    return row


def run_one(scenario_id: str, index_row: pd.Series, dataset_dir: str, **cfg) -> dict:
    """Load one scenario from disk and run it (see run_loaded for the options)."""
    meta = dict(
        scenario_id=scenario_id,
        p=int(index_row["p"]),
        n_regimes=int(index_row["n_regimes"]),
        seed=int(index_row["seed"]),
        n_changes=int(index_row.get("n_changes") or int(index_row["n_regimes"]) - 1),
        samples_regime=int(index_row.get("samples_regime") or index_row.get("min_samples_regime") or 600),
        base_pconn=float(index_row.get("base_pconn") or 0.35),
        noise_fraction=float(index_row.get("noise_fraction") or 0.08),
    )
    try:
        X, variable_names, true_cps, true_adjs, regime_sizes, min_samples = load_scenario(
            scenario_id, dataset_dir, index_row
        )
    except Exception:
        return {**meta, "status": "error", "error_msg": traceback.format_exc(limit=3).strip()}
    return run_loaded(X, variable_names, true_cps, true_adjs, regime_sizes, min_samples, meta, **cfg)


# ── Batch summary helper ──────────────────────────────────────────────────────

def _print_batch_summary(rows: list, batch_idx: int):
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        print(f"  Batch {batch_idx:04d} — no successful rows")
        return
    df = pd.DataFrame(ok_rows)
    nshd     = df["mean_norm_shd"].mean() if "mean_norm_shd" in df else float("nan")
    cons_f1  = df["consensus_f1"].mean()  if "consensus_f1"  in df else float("nan")
    dl_nshd  = df["directlingam_mean_norm_shd"].mean() if "directlingam_mean_norm_shd" in df else float("nan")
    ica_nshd = df["icalingam_mean_norm_shd"].mean()    if "icalingam_mean_norm_shd"    in df else float("nan")
    n_ok  = len(ok_rows)
    n_err = len(rows) - n_ok
    print(
        f"\n  ── Batch {batch_idx:04d} ({n_ok} ok / {n_err} err) ──"
        f"  CM_nSHD={nshd:.3f}  CM_cons_f1={cons_f1:.3f}"
        f"  DL_nSHD={dl_nshd:.3f}"
        f"  ICA_nSHD={ica_nshd:.3f}"
    )


# ── Global summary helper ─────────────────────────────────────────────────────

def _print_global_summary(output_dir: str):
    results_path = os.path.join(output_dir, "results.csv")
    if not os.path.exists(results_path):
        return

    df_all = pd.read_csv(results_path)
    ok     = df_all[df_all["status"] == "ok"]

    print(f"\n{'='*70}")
    print(f"GLOBAL SUMMARY  ({len(ok)}/{len(df_all)} succeeded)")
    print(f"{'='*70}")

    cols = [
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
        print(f"\n  {'p':>3}   n   CM_nSHD  DL_nSHD  ICA_nSHD  CM_F1  DL_F1  ICA_F1")
        print(f"  {'─'*65}")
        for pval in sorted(ok["p"].astype(str).unique(), key=int):
            sub = ok[ok["p"].astype(str) == pval]
            def m(c): return sub[c].mean() if c in sub else float("nan")
            print(
                f"  {pval:>3}  {len(sub):>4}"
                f"   {m('mean_norm_shd'):.3f}"
                f"    {m('directlingam_mean_norm_shd'):.3f}"
                f"     {m('icalingam_mean_norm_shd'):.3f}"
                f"    {m('consensus_f1'):.3f}"
                f"  {m('directlingam_cons_f1'):.3f}"
                f"  {m('icalingam_cons_f1'):.3f}"
            )

    print(f"{'='*70}")


def _append_rows(rows: list, results_path: str, columns):
    """
    Append rows to results.csv, creating it with a header on first write.

    Returns the canonical column list. Error rows carry only the 10 identifying
    keys while ok rows carry all 78, so every row is reindexed onto `columns`
    before writing and short rows become empty cells rather than shifted ones.
    The header is locked to the first ok row seen; until one appears the rows
    stay buffered, since an error row alone would fix a 10-column header for the
    rest of the run.
    """
    if columns is None:
        first_ok = next((r for r in rows if r.get("status") == "ok"), None)
        if first_ok is None:
            return None, rows          # keep buffering, schema still unknown
        columns = list(first_ok.keys())

    write_header = not os.path.exists(results_path)
    (pd.DataFrame(rows)
       .reindex(columns=columns)
       .to_csv(results_path, mode="a", header=write_header, index=False))
    return columns, []


# ── Main batch runner ─────────────────────────────────────────────────────────

def run_experiments(
    dataset_dir: str      = "datasets",
    output_dir: str       = "results",
    batch_size: int       = 10,
    n_workers: int        = None,
    detector_version: str = "v1G",
    prior_mode: str       = "chain",
    window_overlap: float = 0.15,
    hybrid: bool          = False,
    bootstrap_n: int      = 20,
    bootstrap_threshold: float = 0.60,
    chain_prior_threshold: float = 0.05,
    suppress_output: bool = True,
    p_filter: int         = None,
    resume: bool          = False,
    scenario_ids_file: Optional[str] = None,
    **run_cfg,
):
    """
    Run all scenarios in dataset_dir/index.csv in parallel, checkpointing
    results.csv to output_dir every batch_size scenarios.

    With resume=True, scenarios already present in an existing results.csv are
    skipped and their rows are carried into the new file, so an interrupted run
    can be restarted without recomputing what it already finished.
    """
    index_path = os.path.join(dataset_dir, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No index.csv found in {dataset_dir!r}. Run generate_datasets.py first."
        )

    index_df = pd.read_csv(index_path, dtype=str)
    if p_filter is not None:
        index_df = index_df[index_df["p"] == str(p_filter)]
    if scenario_ids_file is not None:
        keep = {ln.strip() for ln in open(scenario_ids_file) if ln.strip()}
        index_df = index_df[index_df["scenario_id"].isin(keep)]
        print(f"  scenario_ids_file: {len(index_df)} of {len(keep)} listed scenarios found")

    os.makedirs(output_dir, exist_ok=True)

    # Rows already on disk are left there: --resume appends to the same file
    # rather than reading it back, so memory never holds the whole result set.
    results_path = os.path.join(output_dir, "results.csv")
    n_done       = 0
    columns      = None     # canonical CSV header, locked once known
    if resume and os.path.exists(results_path):
        columns  = list(pd.read_csv(results_path, nrows=0).columns)
        if "w0_prior_source" not in columns:
            raise SystemExit(
                f"{results_path} was written by the pre-2026-10-06 runner (no w0_prior_source "
                f"column); appending the new schema onto it would silently drop columns. "
                f"Use a fresh --output_dir."
            )
        done_ids = set(pd.read_csv(results_path, usecols=["scenario_id"],
                                   dtype=str)["scenario_id"])
        n_done   = len(done_ids)
        before   = len(index_df)
        index_df = index_df[~index_df["scenario_id"].astype(str).isin(done_ids)]
        print(f"  resume: {before - len(index_df)} scenarios already done in "
              f"{results_path}, {len(index_df)} left")

    scenarios = list(index_df.iterrows())
    total     = len(scenarios)

    if resume and total == 0:
        print(f"  resume: nothing left to run — {n_done} rows already in {results_path}")
        _print_global_summary(output_dir)
        return

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) // 2)

    print("=" * 70)
    print("Experiment Runner — Full Pipeline (parallel)")
    print("=" * 70)
    print(f"  dataset_dir : {dataset_dir}/")
    print(f"  output_dir  : {output_dir}/")
    print(f"  detector    : {detector_version}")
    print(f"  prior_mode  : {prior_mode}")
    print(f"  window_overlap : {window_overlap}")
    print(f"  hybrid      : {hybrid}")
    print(f"  bootstrap_n : {bootstrap_n}  (threshold={bootstrap_threshold})")
    for k_, v_ in sorted(run_cfg.items()):
        print(f"  {k_:<12}: {v_}")
    print(f"  methods     : CausalMorph + DirectLiNGAM + ICA-LiNGAM")
    print(f"  scenarios   : {total}")
    print(f"  workers     : {n_workers}")
    if p_filter:
        print(f"  p_filter    : {p_filter}")
    print()

    t_global   = time.time()
    group_buf  = []   # rows since the last flush; the only ones held in memory
    n_written  = n_done

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(_HERE,),
    ) as pool:
        cfg = dict(
            window_overlap=window_overlap, detector_version=detector_version,
            prior_mode=prior_mode, suppress_output=suppress_output, hybrid=hybrid,
            bootstrap_n=bootstrap_n, bootstrap_threshold=bootstrap_threshold,
            chain_prior_threshold=chain_prior_threshold, **run_cfg,
        )
        future_to_sid = {
            pool.submit(run_one, str(index_row["scenario_id"]), index_row, dataset_dir, **cfg):
            str(index_row["scenario_id"])
            for _, index_row in scenarios
        }

        for done_count, future in enumerate(as_completed(future_to_sid), 1):
            sid = future_to_sid[future]
            row = future.result()
            group_buf.append(row)

            elapsed = time.time() - t_global
            if row["status"] == "ok":
                cm_nshd  = row.get("mean_norm_shd",              float("nan"))
                dl_nshd  = row.get("directlingam_mean_norm_shd", float("nan"))
                ica_nshd = row.get("icalingam_mean_norm_shd",    float("nan"))
                print(
                    f"  [{done_count:>4}/{total}]  {elapsed:>6.0f}s  {sid}"
                    f"  CM={cm_nshd:.3f}  DL={dl_nshd:.3f}  ICA={ica_nshd:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"  [{done_count:>4}/{total}]  {elapsed:>6.0f}s  {sid}"
                    f"  ERROR: {row['error_msg'][:70]}",
                    flush=True,
                )

            if done_count % batch_size == 0:
                _print_batch_summary(group_buf, done_count // batch_size - 1)
                columns, group_buf = _append_rows(group_buf, results_path, columns)
                if not group_buf:
                    n_written = n_done + done_count
                    print(f"  → checkpoint: {n_written} rows in {results_path}", flush=True)

    if group_buf:
        _print_batch_summary(group_buf, total // batch_size)
        # A tail that is still buffered because no ok row ever appeared is
        # written on its own keys rather than dropped.
        columns, group_buf = _append_rows(group_buf, results_path, columns)
        if group_buf:
            _append_rows(group_buf, results_path, list(group_buf[0].keys()))

    print(f"\n  Final save: {n_done + total} rows → {results_path}")

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
                        help="Directory to write results.csv (default: results/)")
    parser.add_argument("--batch_size",     type=int,   default=10,
                        help="Progress-reporting group size (default: 100)")
    parser.add_argument("--workers",        type=int,   default=None,
                        help="Parallel worker processes (default: cpu_count // 2)")
    parser.add_argument("--detector",       type=str,   default="v1G", choices=["v1F", "v1G", "v1H", "v1I", "v1J", "v1K", "v1L"],
                        help="Detector version (default: v1G). "
                             "v1H = strict correlation-based detector, ~92%% precision vs v1G's ~52%%. "
                             "v1I = wavelet-coherence-based detector (Grinsted et al. 2004); "
                             "frequency-resolved joint-distribution change signal.")
    parser.add_argument("--prior_mode",     type=str,   default="chain", choices=["anchor", "chain"],
                        help="CausalMorph prior mode (default: chain)")
    parser.add_argument("--window_overlap", type=float, default=0.15,
                        help="CausalMorph window overlap fraction (default: 0.15; was 0.0).")
    parser.add_argument("--hybrid",         dest="hybrid", action="store_true", default=False,
                        help="Hybrid warm/cold model selection by residual independence (default: OFF). "
                             "Ablation (2026-05-27, 180 scenarios) found hybrid alone drops per-regime "
                             "F1 by 49%% on CausalMorph; the residual-independence selector picks 'cold' "
                             "when 'warm' was better.  Kept as opt-in for future use.")
    parser.add_argument("--no-hybrid",      dest="hybrid", action="store_false",
                        help="Disable hybrid selection (current default).")
    parser.add_argument("--bootstrap_n",    type=int,   default=20,
                        help="Bootstrap resamples per window for edge stability (default: 20; 0 = off).")
    parser.add_argument("--bootstrap_threshold", type=float, default=0.60,
                        help="Edge frequency threshold for bootstrap consensus (default: 0.60).")
    parser.add_argument("--chain_prior_threshold", type=float, default=0.05,
                        help="Edge weight threshold for binarising the chain prior between regimes "
                             "(default: 0.05; phase-2 experiment used 0.10).")
    parser.add_argument("--verbose",        action="store_true",
                        help="Show per-run pipeline output (very noisy)")
    parser.add_argument("--p_filter",       type=int,   default=None,
                        help="Only run scenarios with this node count")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip scenarios already present in output_dir/results.csv "
                             "and keep their rows (restart an interrupted run)")
    parser.add_argument("--scenario_ids_file", type=str, default=None,
                        help="Run only the scenario_ids listed in this file (one per line)")
    exp = parser.add_argument_group("experimental conditions (2026-10-06 hardening)")
    exp.add_argument("--oracle_anchor",        action="store_true",
                     help="Seed window 0 with the true regime-0 graph (pre-2026-10-06 behaviour; upper bound)")
    exp.add_argument("--oracle_segmentation",  action="store_true",
                     help="Extract on the true change points (extraction ceiling)")
    exp.add_argument("--no_segmentation",      action="store_true",
                     help="One window over the whole series (ignore non-stationarity)")
    exp.add_argument("--samples_per_node",     type=float, default=40.0,
                     help="Extraction min window = max(250, c*p); this is c (default 40)")
    exp.add_argument("--fixed_min_regime_len", type=int,   default=None,
                     help="Size the detector's windows from this instead of the true min regime length")
    exp.add_argument("--consensus_threshold",  type=float, default=0.5,
                     help="Posterior edge probability above which an edge enters the consensus (default 0.5)")
    exp.add_argument("--consensus_prior_density", type=float, default=None,
                     help="Sparsity prior rho0 for the consensus Beta(kappa*rho0, kappa*(1-rho0)); None = uniform")
    exp.add_argument("--chain_soft_lambda",    type=float, default=None,
                     help="Soft Markov chain prior q = lambda*q_prev + (1-lambda)*rho0; None = binary chain")
    exp.add_argument("--corr_gamma",           type=float, default=2.0,
                     help="v1J threshold margin: sqrt(2 ln m) + gamma")
    exp.add_argument("--adaptive_null",        action="store_true",
                     help="v1H: re-estimate the threshold from the segment since the last accepted change point")
    exp.add_argument("--det_tolerance",        type=int,   default=200,
                     help="Matching tolerance (samples) for detection precision/recall")
    exp.add_argument("--no_ablation",          action="store_true",
                     help="Skip the DirectLiNGAM / ICA-LiNGAM per-window baselines")
    exp.add_argument("--legacy_postprocessing", action="store_true",
                     help="Pre-2026-10-06 deletion rule max(min_samples/2, 10p^2), scored before extraction (old 'before')")
    args = parser.parse_args()

    run_experiments(
        dataset_dir     = args.dataset_dir,
        output_dir      = args.output_dir,
        batch_size      = args.batch_size,
        n_workers       = args.workers,
        detector_version= args.detector,
        prior_mode      = args.prior_mode,
        window_overlap  = args.window_overlap,
        hybrid          = args.hybrid,
        bootstrap_n     = args.bootstrap_n,
        bootstrap_threshold = args.bootstrap_threshold,
        chain_prior_threshold = args.chain_prior_threshold,
        suppress_output = not args.verbose,
        p_filter        = args.p_filter,
        resume          = args.resume,
        scenario_ids_file = args.scenario_ids_file,
        oracle_anchor=args.oracle_anchor, oracle_segmentation=args.oracle_segmentation,
        no_segmentation=args.no_segmentation, samples_per_node=args.samples_per_node,
        fixed_min_regime_len=args.fixed_min_regime_len,
        consensus_threshold=args.consensus_threshold,
        consensus_prior_density=args.consensus_prior_density,
        chain_soft_lambda=args.chain_soft_lambda, corr_gamma=args.corr_gamma,
        adaptive_null=args.adaptive_null, det_tolerance=args.det_tolerance,
        run_ablation=not args.no_ablation,
        legacy_postprocessing=args.legacy_postprocessing,
    )
