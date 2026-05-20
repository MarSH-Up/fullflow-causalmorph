"""
Batch experiment runner for the full causal learning pipeline.

Sweeps over a range of node counts (p_range) and runs N_EXPERIMENTS
per configuration. Each run uses a fresh random seed. Results are
collected into a single CSV.

Metrics saved per experiment
-----------------------------
Identification
  exp_id, seed, p, n_regimes, total_samples, n_true_cps

Change-point detection
  det_f1, det_precision, det_recall, det_tp, det_fp, det_fn
  n_detected_cps

Per-regime causal structure recovery (averaged across all detected windows)
  mean_shd, mean_norm_shd, mean_struct_f1, mean_struct_precision, mean_struct_recall
  n_structures

Bayesian consensus vs global ground truth (final target)
  consensus_shd, consensus_norm_shd, consensus_f1, consensus_precision, consensus_recall

Run status
  status  ("ok" | "error"), error_msg
"""

import os
import sys
import time
import traceback
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Path setup (mirrors full_pipeline.py) ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "NSD_Wavelets", "src"))
sys.path.insert(0, os.path.join(_HERE, "causalmorph"))

from full_pipeline import run_full_pipeline, compute_shd, ABLATION_METHODS


# ── Metric extraction ─────────────────────────────────────────────────────────

def _extract_metrics(p: int, n_regimes: int, seed: int, result: dict) -> dict:
    """Pull all scalar metrics out of a run_full_pipeline result dict."""
    er  = result["eval_result"]
    structs = result["structures"]
    true_adjs = result["true_adjs"]
    consensus_adj = result["consensus_adj"]
    variable_names = result["variable_names"]

    # ── detection ────────────────────────────────────────────────────────────
    det = {
        "det_f1":        round(er.f1_score, 4),
        "det_precision": round(er.precision, 4),
        "det_recall":    round(er.recall, 4),
        "det_tp":        er.true_positives,
        "det_fp":        er.false_positives,
        "det_fn":        er.false_negatives,
        "n_true_cps":    len(result["true_change_points"]),
        "n_detected_cps": len(result["detected_change_points"]),
        "total_samples": len(result["X"]),
    }

    # ── per-regime structure (averaged) ──────────────────────────────────────
    metrics_list = [s.shd_metrics for s in structs if s.shd_metrics]
    if metrics_list:
        struct = {
            "n_structures":       len(structs),
            "mean_shd":           round(np.mean([m["SHD"]            for m in metrics_list]), 3),
            "mean_norm_shd":      round(np.mean([m["normalized_shd"] for m in metrics_list]), 3),
            "mean_struct_f1":     round(np.mean([m["F1"]             for m in metrics_list]), 3),
            "mean_struct_precision": round(np.mean([m["Precision"]   for m in metrics_list]), 3),
            "mean_struct_recall": round(np.mean([m["Recall"]         for m in metrics_list]), 3),
        }
    else:
        struct = {k: np.nan for k in (
            "n_structures", "mean_shd", "mean_norm_shd",
            "mean_struct_f1", "mean_struct_precision", "mean_struct_recall",
        )}
        struct["n_structures"] = len(structs)

    # ── consensus vs global ground truth (final target regime) ───────────────
    if true_adjs:
        gt_raw = true_adjs[-1]
        gt_df = pd.DataFrame(
            (np.asarray(gt_raw.values if hasattr(gt_raw, "values") else gt_raw) != 0).astype(float),
            index=variable_names, columns=variable_names,
        )
        cm = compute_shd(consensus_adj, gt_df)
        consensus = {
            "consensus_shd":       cm["SHD"],
            "consensus_norm_shd":  cm["normalized_shd"],
            "consensus_f1":        cm["F1"],
            "consensus_precision": cm["Precision"],
            "consensus_recall":    cm["Recall"],
        }
    else:
        consensus = {k: np.nan for k in (
            "consensus_shd", "consensus_norm_shd",
            "consensus_f1", "consensus_precision", "consensus_recall",
        )}

    # ── ablation metrics (per method) ─────────────────────────────────────
    ablation = {}
    if "ablation" in result and result["ablation"]:
        for method in ABLATION_METHODS:
            structs_m = result["ablation"].get(method, [])
            metrics_m = [s.shd_metrics for s in structs_m if s.shd_metrics]
            if metrics_m:
                ablation[f"{method}_mean_shd"]       = round(np.mean([m["SHD"]            for m in metrics_m]), 3)
                ablation[f"{method}_mean_norm_shd"]   = round(np.mean([m["normalized_shd"] for m in metrics_m]), 3)
                ablation[f"{method}_mean_f1"]         = round(np.mean([m["F1"]             for m in metrics_m]), 3)
                ablation[f"{method}_mean_precision"]   = round(np.mean([m["Precision"]     for m in metrics_m]), 3)
                ablation[f"{method}_mean_recall"]      = round(np.mean([m["Recall"]        for m in metrics_m]), 3)
            else:
                for suffix in ("mean_shd", "mean_norm_shd", "mean_f1", "mean_precision", "mean_recall"):
                    ablation[f"{method}_{suffix}"] = np.nan
            # consensus per method
            ac = result.get("ablation_consensus", {}).get(method)
            if ac and true_adjs:
                gt_raw = true_adjs[-1]
                gt_df = pd.DataFrame(
                    (np.asarray(gt_raw.values if hasattr(gt_raw, "values") else gt_raw) != 0).astype(float),
                    index=variable_names, columns=variable_names,
                )
                cm_ab = compute_shd(ac["consensus_adj"], gt_df)
                ablation[f"{method}_cons_norm_shd"] = cm_ab["normalized_shd"]
                ablation[f"{method}_cons_f1"]       = cm_ab["F1"]
            else:
                ablation[f"{method}_cons_norm_shd"] = np.nan
                ablation[f"{method}_cons_f1"]       = np.nan

    return {"p": p, "n_regimes": n_regimes, "seed": seed, **det, **struct, **consensus, **ablation}


# ── Single experiment wrapper ─────────────────────────────────────────────────

def run_one(
    p: int,
    n_regimes: int,
    seed: int,
    min_samples: int,
    max_samples: int,
    base_pconn: float,
    window_overlap: float,
    suppress_output: bool,
    prior_mode: str = "chain",
    hybrid: bool = False,
    ablation: bool = False,
    detector_version: str = "v1G",
) -> dict:
    """Run one experiment; return a flat metrics dict with status/error_msg."""
    ctx = contextlib.redirect_stdout(io.StringIO()) if suppress_output else contextlib.nullcontext()
    try:
        with ctx:
            result = run_full_pipeline(
                p=p,
                n_regimes=n_regimes,
                min_samples=min_samples,
                max_samples=max_samples,
                base_pconn=base_pconn,
                window_overlap=window_overlap,
                seed=seed,
                verbose=False,
                show_plots=False,
                prior_mode=prior_mode,
                hybrid=hybrid,
                ablation=ablation,
                detector_version=detector_version,
            )
        row = _extract_metrics(p, n_regimes, seed, result)
        row["status"]    = "ok"
        row["error_msg"] = ""
    except Exception:
        row = {
            "p": p, "n_regimes": n_regimes, "seed": seed,
            "status": "error", "error_msg": traceback.format_exc(limit=3).strip(),
        }
    return row


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    p_range: range,
    n_experiments: int,
    n_regimes: int = 5,
    min_samples: int = 600,
    max_samples: int = 800,
    base_pconn: float = 0.35,
    window_overlap: float = 0.0,
    base_seed: int = 0,
    output_csv: str = "batch_results.csv",
    suppress_output: bool = True,
    prior_mode: str = "chain",
    hybrid: bool = False,
    ablation: bool = False,
    detector_version: str = "v1G",
) -> pd.DataFrame:
    """
    Run n_experiments × len(p_range) experiments and save to CSV.

    Parameters
    ----------
    p_range       : range of node counts, e.g. range(3, 9)
    n_experiments : number of independent runs per node count
    n_regimes     : number of regimes per run
    min_samples   : minimum samples per regime
    max_samples   : maximum samples per regime
    base_pconn    : base edge connection probability
    window_overlap: causal extraction window overlap fraction
    base_seed     : seeds are base_seed + exp_idx (unique across all runs)
    output_csv    : path to write the results CSV
    suppress_output: suppress per-run print output (recommended for batch)
    detector_version: "v1G" (default) or "v1F"

    Returns
    -------
    pd.DataFrame with one row per experiment
    """
    p_list = list(p_range)
    total  = len(p_list) * n_experiments
    rows   = []
    exp_id = 0

    print(f"Batch: {len(p_list)} node counts × {n_experiments} runs = {total} experiments")
    print(f"  p_range={list(p_range)},  n_regimes={n_regimes},  "
          f"samples=[{min_samples},{max_samples}],  base_pconn={base_pconn}")
    print(f"  prior_mode={prior_mode!r}  hybrid={hybrid}  ablation={ablation}  detector={detector_version}")
    print(f"  Output → {output_csv}\n")

    t0 = time.time()
    for p in p_list:
        p_errors = 0
        for exp in range(n_experiments):
            seed = base_seed + exp_id
            elapsed = time.time() - t0
            print(f"  [{exp_id+1:>4}/{total}]  p={p}  exp={exp+1}/{n_experiments}  "
                  f"seed={seed}  elapsed={elapsed:.0f}s", end="  ")

            row = run_one(
                p=p, n_regimes=n_regimes, seed=seed,
                min_samples=min_samples, max_samples=max_samples,
                base_pconn=base_pconn, window_overlap=window_overlap,
                suppress_output=suppress_output,
                prior_mode=prior_mode,
                hybrid=hybrid,
                ablation=ablation,
                detector_version=detector_version,
            )
            row["exp_id"] = exp_id

            if row["status"] == "ok":
                line = (f"det_prec={row['det_precision']:.2f}  "
                        f"cons_f1={row.get('consensus_f1', float('nan')):.2f}  "
                        f"mean_nSHD={row.get('mean_norm_shd', float('nan')):.3f}")
                if ablation:
                    for m in ABLATION_METHODS:
                        v = row.get(f"{m}_mean_norm_shd", float("nan"))
                        line += f"  {m[:6]}={v:.3f}"
                print(line)
            else:
                p_errors += 1
                print(f"ERROR: {row['error_msg'][:80]}")

            rows.append(row)
            exp_id += 1

        # Per-p summary
        p_rows = [r for r in rows if r.get("p") == p and r.get("status") == "ok"]
        if p_rows:
            df_p = pd.DataFrame(p_rows)
            print(f"\n  ── p={p} summary ({len(p_rows)} ok / {p_errors} errors) ──")
            ablation_cols = []
            if ablation:
                for m in ABLATION_METHODS:
                    ablation_cols += [f"{m}_mean_norm_shd", f"{m}_mean_f1"]
            for col in ["det_f1", "det_precision", "det_recall",
                        "mean_norm_shd", "mean_struct_f1",
                        "consensus_norm_shd", "consensus_f1"] + ablation_cols:
                if col in df_p:
                    print(f"     {col:<25}  mean={df_p[col].mean():.3f}  "
                          f"std={df_p[col].std():.3f}  "
                          f"min={df_p[col].min():.3f}  max={df_p[col].max():.3f}")
            print()

    df = pd.DataFrame(rows)

    # Reorder: identification columns first
    id_cols = ["exp_id", "seed", "p", "n_regimes", "total_samples",
               "n_true_cps", "n_detected_cps"]
    other_cols = [c for c in df.columns if c not in id_cols]
    df = df[[c for c in id_cols if c in df.columns] + other_cols]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows → {output_csv}")

    # Global summary
    ok = df[df["status"] == "ok"]
    if len(ok) > 0:
        print(f"\n{'='*60}")
        print(f"GLOBAL SUMMARY  ({len(ok)}/{len(df)} succeeded)")
        print(f"{'='*60}")
        summary_cols = [
            "det_f1", "det_precision", "det_recall",
            "mean_norm_shd", "mean_struct_f1",
            "consensus_norm_shd", "consensus_f1",
        ]
        if ablation:
            for m in ABLATION_METHODS:
                summary_cols += [f"{m}_mean_norm_shd", f"{m}_mean_f1",
                                 f"{m}_cons_norm_shd", f"{m}_cons_f1"]
        for col in summary_cols:
            if col in ok.columns:
                print(f"  {col:<28}  "
                      f"mean={ok[col].mean():.3f}  "
                      f"std={ok[col].std():.3f}")
        print(f"{'='*60}")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch causal pipeline experiments")
    parser.add_argument("--p_min",        type=int,   default=3,                help="Min nodes")
    parser.add_argument("--p_max",        type=int,   default=7,                help="Max nodes (inclusive)")
    parser.add_argument("--n_experiments",type=int,   default=10,               help="Runs per node count")
    parser.add_argument("--n_regimes",    type=int,   default=5,                help="Regimes per run")
    parser.add_argument("--min_samples",  type=int,   default=600,              help="Min samples/regime")
    parser.add_argument("--max_samples",  type=int,   default=800,              help="Max samples/regime")
    parser.add_argument("--base_pconn",   type=float, default=0.35,             help="Base edge probability")
    parser.add_argument("--window_overlap",type=float,default=0.0,              help="CausalMorph window overlap")
    parser.add_argument("--base_seed",    type=int,   default=int(time.time()) % (2**31), help="Starting seed")
    parser.add_argument("--output",       type=str,   default="batch_results.csv", help="Output CSV path")
    parser.add_argument("--verbose",      action="store_true",                  help="Show per-run output")
    parser.add_argument("--prior_mode",   type=str,   default="chain",          choices=["anchor", "chain"],
                        help="anchor: every window uses regime-0 GT prior; chain: previous window's structure")
    parser.add_argument("--hybrid",       action="store_true",
                        help="Hybrid warm/cold selection per window using residual independence score")
    parser.add_argument("--ablation",    action="store_true",
                        help="Run DirectLiNGAM-only and ICA-LiNGAM ablation on each window")
    parser.add_argument("--detector",    type=str,   default="v1G", choices=["v1F", "v1G"],
                        help="Detector version: v1F (original) or v1G (adaptive+two-pass)")
    args = parser.parse_args()

    print(f"[Batch] base_seed={args.base_seed}  detector={args.detector}")
    run_batch(
        p_range=range(args.p_min, args.p_max + 1),
        n_experiments=args.n_experiments,
        n_regimes=args.n_regimes,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        base_pconn=args.base_pconn,
        window_overlap=args.window_overlap,
        base_seed=args.base_seed,
        output_csv=args.output,
        suppress_output=not args.verbose,
        prior_mode=args.prior_mode,
        hybrid=args.hybrid,
        ablation=args.ablation,
        detector_version=args.detector,
    )
