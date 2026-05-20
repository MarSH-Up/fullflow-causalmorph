"""
Sensitivity Analysis to Errors in Tentative Causal Order

Now: Run all scenario variants from @experiments/SyntheticCausalScenarios_mixed_v2_5 such as used in @synthetic_pipeline_mac.py

Usage:
    python experiments/benchmarks/run_order_sensitivity.py <output_dir> <num_workers>

Set SAMPLE_MODE = True at the top of this file for a quick sanity check
(25 random scenarios, 1 repetition, no file moves).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "0"

import re
import math
import shutil
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from lingam import DirectLiNGAM
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import kendalltau
import psutil

warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

from core.causalmorph_algorithm import causalMorph
from utils.metrics import normalized_shd, mycomparegraphs


_SCENARIOS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "SyntheticCausalScenarios_mixed_v2_5",
)


def list_scenario_names(data_dir):
    """Return base names (without -dat.csv) for all valid scenario pairs."""
    dat_files = sorted(f for f in os.listdir(data_dir) if f.endswith("-dat.csv"))
    names = []
    for dat_file in dat_files:
        am_file = dat_file.replace("-dat.csv", "-am.csv")
        if os.path.exists(os.path.join(data_dir, am_file)):
            names.append(dat_file.replace("-dat.csv", ""))
    return names


def load_scenario(data_dir, name):
    """Load a single scenario dict by base name."""
    dat_path = os.path.join(data_dir, name + "-dat.csv")
    am_path = os.path.join(data_dir, name + "-am.csv")
    data = pd.read_csv(dat_path)
    adj_true = pd.read_csv(am_path).values
    seed_match = re.search(r"seed-(\d+)", name)
    seed = int(seed_match.group(1)) if seed_match else 42
    return {
        "adj_true": adj_true,
        "data": data,
        "name": name,
        "seed": seed,
    }

def derive_causal_order(adj_matrix):
    """Topological sort of the true adjacency matrix."""
    p = adj_matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(p))
    for i in range(p):
        for j in range(p):
            if adj_matrix[i, j] != 0:
                G.add_edge(i, j)
    return list(nx.topological_sort(G))


def perturb_causal_order(order, error_rate, seed):
    """
    Directly permute positions in the causal order.

    Selects a fraction (error_rate) of positions and randomly shuffles them,
    leaving the rest in place. At error_rate=0 the order is unchanged;
    at error_rate=1 all positions are shuffled.

    Returns (perturbed_order, metadata_dict).
    """
    rng = np.random.RandomState(seed)
    order = list(order)
    p = len(order)
    n_swap = math.ceil(error_rate * p)
    n_swap = min(n_swap, p)

    if n_swap <= 1:
        return list(order), {
            "positions_permuted": 0,
            "kendall_displacement": 0,
        }

    swap_indices = rng.choice(p, size=n_swap, replace=False)
    values_at_swap = [order[i] for i in swap_indices]
    rng.shuffle(values_at_swap)

    perturbed = list(order)
    for idx, val in zip(swap_indices, values_at_swap):
        perturbed[idx] = val

    # Count how many positions actually changed
    positions_changed = sum(1 for a, b in zip(order, perturbed) if a != b)

    return perturbed, {
        "positions_permuted": positions_changed,
        "kendall_displacement": positions_changed / p if p > 0 else 0.0,
    }

def perturb_adjacency_matrix(adj_matrix, error_rate, seed):
    """
    Rewire a fraction of edges in the adjacency matrix.

    Removes floor(error_rate * num_edges) true edges and adds the same number
    of false edges, keeping the total edge count (and density) constant.

    Returns (perturbed_adj, metadata_dict).
    """
    rng = np.random.RandomState(seed)
    adj = adj_matrix.copy().astype(np.int8)
    p = adj.shape[0]

    edges = list(zip(*np.where(adj != 0)))
    non_edges = [(i, j) for i in range(p) for j in range(p)
                 if i != j and adj[i, j] == 0]

    n_rewire = int(math.floor(error_rate * len(edges)))

    if n_rewire == 0 or len(non_edges) == 0:
        return adj, {"edges_rewired": 0}

    n_rewire = min(n_rewire, len(edges), len(non_edges))

    remove_idx = rng.choice(len(edges), size=n_rewire, replace=False)
    add_idx = rng.choice(len(non_edges), size=n_rewire, replace=False)

    for idx in remove_idx:
        i, j = edges[idx]
        adj[i, j] = 0
    for idx in add_idx:
        i, j = non_edges[idx]
        adj[i, j] = 1

    return adj, {"edges_rewired": n_rewire}


def compute_order_error(true_order, estimated_order):
    """
    Compare two causal orders and compute error metrics.

    Returns dict with:
      - positions_changed: number of positions where indices differ
      - kendall_displacement: positions_changed / p
      - kendall_tau: Kendall tau correlation (-1 to 1, higher = more similar)
    """
    true_order = list(true_order)
    estimated_order = list(estimated_order)
    p = len(true_order)
    positions_changed = sum(1 for a, b in zip(true_order, estimated_order) if a != b)
    tau, _ = kendalltau(true_order, estimated_order)
    return {
        "positions_changed": positions_changed,
        "kendall_displacement": positions_changed / p if p > 0 else 0.0,
        "kendall_tau": tau,
    }


def compute_adj_error_rate(adj_true, adj_estimated):
    """
    Compute the error rate of an estimated adjacency matrix vs the true one.

    Returns dict with:
      - edge_error_rate: FN / num_true_edges (fraction of true edges missed,
        bounded [0,1], comparable to controlled adj_error_rate)
      - f1_input: F1 score of estimated vs true
      - precision_input: precision of estimated vs true
      - recall_input: recall of estimated vs true
    """
    metrics = mycomparegraphs(adj_estimated, adj_true)
    num_true_edges = int(np.sum(adj_true != 0))
    edge_error_rate = metrics["FN"] / num_true_edges if num_true_edges > 0 else 0.0
    return {
        "edge_error_rate": edge_error_rate,
        "f1_input": metrics["F1"],
        "precision_input": metrics["Precision"],
        "recall_input": metrics["Recall"],
    }


def extract_seed(name):
    """Extract random seed from scenario name without loading data."""
    m = re.search(r"seed-(\d+)", name)
    return int(m.group(1)) if m else 42


def _build_result(experiment_type, name, p, num_edges_true, density,
                  order_error_rate, adj_error_rate, random_seed,
                  shd_orig_value, metrics_orig, shd_trans_value, metrics_trans,
                  pred_orig, pred_trans, order_meta, adj_meta,
                  extra=None):
    """Assemble a result dict with all standard columns."""
    result = {
        "experiment_type": experiment_type,
        "scenario_name": name,
        "p": p,
        "num_edges_true": num_edges_true,
        "density": density,
        "order_error_rate": order_error_rate,
        "adj_error_rate": adj_error_rate,
        "random_seed": random_seed,
        # SHD metrics
        "shd_original": shd_orig_value,
        "shd_transformed": shd_trans_value,
        "shd_absolute_original": metrics_orig["SHD"],
        "shd_absolute_transformed": metrics_trans["SHD"],
        "improvement": shd_orig_value - shd_trans_value,
        # Classification metrics
        "f1_original": metrics_orig["F1"],
        "f1_transformed": metrics_trans["F1"],
        "f1_improvement": metrics_trans["F1"] - metrics_orig["F1"],
        "tpr_original": metrics_orig["TPR"],
        "tpr_transformed": metrics_trans["TPR"],
        "tdr_original": metrics_orig["TDR"],
        "tdr_transformed": metrics_trans["TDR"],
        "acc_original": metrics_orig["ACC"],
        "acc_transformed": metrics_trans["ACC"],
        "precision_original": metrics_orig["Precision"],
        "precision_transformed": metrics_trans["Precision"],
        "recall_original": metrics_orig["Recall"],
        "recall_transformed": metrics_trans["Recall"],
        "specificity_original": metrics_orig["Specificity"],
        "specificity_transformed": metrics_trans["Specificity"],
        "mcc_original": metrics_orig["MCC"],
        "mcc_transformed": metrics_trans["MCC"],
        "mean_degree_true": metrics_orig["mean_degree_true"],
        "mean_degree_est_original": metrics_orig["mean_degree_est"],
        "mean_degree_est_transformed": metrics_trans["mean_degree_est"],
        "mae_degree_original": metrics_orig["mae_degree"],
        "mae_degree_transformed": metrics_trans["mae_degree"],
        # Edge counts
        "num_edges_pred_original": int(np.sum(pred_orig != 0)),
        "num_edges_pred_transformed": int(np.sum(pred_trans != 0)),
        # Perturbation metadata
        "positions_permuted": order_meta["positions_permuted"],
        "kendall_displacement": order_meta["kendall_displacement"],
        "edges_rewired": adj_meta["edges_rewired"],
    }
    if extra:
        result.update(extra)
    return result


def process_scenario_experiments(config):
    """
    Process ALL experiments (controlled + real pipeline) for a single scenario.

    Loads data once from disk and fits the baseline DirectLiNGAM once,
    then runs all perturbation experiments reusing those results.
    This avoids:
      - redundant baseline LiNGAM fits (was 60x per scenario, now 1x)
      - pickling full DataFrames across process boundaries
    """
    data_dir = config["data_dir"]
    scenario_name = config["scenario_name"]
    controlled_configs = config["controlled_configs"]
    real_pipeline_seeds = config["real_pipeline_seeds"]

    # ── Load data once from disk (no pickle overhead) ───────────────
    try:
        scenario = load_scenario(data_dir, scenario_name)
    except Exception as e:
        print(f"Error loading scenario {scenario_name}: {e}")
        return []

    adj_true = scenario["adj_true"].astype(np.int8)
    data = scenario["data"]
    name = scenario["name"]
    p = adj_true.shape[0]
    num_edges_true = int(np.sum(adj_true))
    density = num_edges_true / (p * (p - 1)) if p > 1 else 0.0

    # ── Fit baseline DirectLiNGAM once (shared across all experiments) ──
    try:
        model_orig = DirectLiNGAM()
        model_orig.fit(data)
        pred_orig = model_orig.adjacency_matrix_
        metrics_orig = mycomparegraphs(pred_orig, adj_true)
        shd_orig = normalized_shd(adj_true, pred_orig)
        shd_orig_value = shd_orig["normalized_shd"] if isinstance(shd_orig, dict) else shd_orig
        true_order = derive_causal_order(adj_true)
    except Exception as e:
        print(f"Error fitting baseline for {scenario_name}: {e}")
        return []

    results = []

    # ── Controlled experiments ──────────────────────────────────────
    for order_error_rate, adj_error_rate, random_seed in controlled_configs:
        try:
            causal_order, order_meta = perturb_causal_order(
                true_order, order_error_rate, seed=random_seed
            )
            adj_input, adj_meta = perturb_adjacency_matrix(
                adj_true, adj_error_rate, seed=random_seed + 1000
            )

            transformed = causalMorph(
                data,
                causal_order=causal_order,
                adjacency_matrix=pd.DataFrame(
                    adj_input, columns=data.columns, index=data.columns
                ),
                verbose=False,
            )

            if not np.all(np.isfinite(transformed.values)):
                transformed = data.copy()

            model_trans = DirectLiNGAM()
            model_trans.fit(transformed)
            pred_trans = model_trans.adjacency_matrix_
            metrics_trans = mycomparegraphs(pred_trans, adj_true)
            shd_trans = normalized_shd(adj_true, pred_trans)
            shd_trans_value = shd_trans["normalized_shd"] if isinstance(shd_trans, dict) else shd_trans

            results.append(_build_result(
                "controlled", name, p, num_edges_true, density,
                order_error_rate, adj_error_rate, random_seed,
                shd_orig_value, metrics_orig, shd_trans_value, metrics_trans,
                pred_orig, pred_trans, order_meta, adj_meta,
            ))
        except Exception as e:
            print(f"Error in controlled experiment for {name} "
                  f"(order_err={order_error_rate}, adj_err={adj_error_rate}, "
                  f"seed={random_seed}): {e}")

    # ── Real pipeline experiments ───────────────────────────────────
    estimated_order = list(model_orig.causal_order_)
    estimated_adj = (np.abs(pred_orig) > 0.05).astype(np.int8)
    order_error = compute_order_error(true_order, estimated_order)
    adj_error = compute_adj_error_rate(adj_true, estimated_adj)
    print(adj_error)
    adj_input, adj_meta = perturb_adjacency_matrix(
        adj_true, adj_error["edge_error_rate"], seed=random_seed + 1000
    )

    for random_seed in real_pipeline_seeds:
        try:
            transformed = causalMorph(
                data,
                causal_order=estimated_order,
                adjacency_matrix=pd.DataFrame(
                    adj_input, columns=data.columns, index=data.columns
                ),
                verbose=False,
            )

            if not np.all(np.isfinite(transformed.values)):
                transformed = data.copy()

            model_trans = DirectLiNGAM()
            model_trans.fit(transformed)
            pred_trans = model_trans.adjacency_matrix_
            metrics_trans = mycomparegraphs(pred_trans, adj_true)
            shd_trans = normalized_shd(adj_true, pred_trans)
            shd_trans_value = shd_trans["normalized_shd"] if isinstance(shd_trans, dict) else shd_trans

            results.append(_build_result(
                "real_pipeline", name, p, num_edges_true, density,
                order_error["kendall_displacement"],
                adj_error["edge_error_rate"],
                random_seed,
                shd_orig_value, metrics_orig, shd_trans_value, metrics_trans,
                pred_orig, pred_trans,
                {"positions_permuted": order_error["positions_changed"],
                 "kendall_displacement": order_error["kendall_displacement"]},
                {"edges_rewired": np.nan},
                extra={
                    "measured_order_error_rate": order_error["kendall_displacement"],
                    "measured_adj_error_rate": adj_error["edge_error_rate"],
                    "kendall_tau": order_error["kendall_tau"],
                    "f1_input_adj": adj_error["f1_input"],
                    "precision_input_adj": adj_error["precision_input"],
                    "recall_input_adj": adj_error["recall_input"],
                },
            ))
        except Exception as e:
            print(f"Error in real pipeline for {name} (seed={random_seed}): {e}")

    return results


ERROR_RATES = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
N_REPETITIONS = 5
BATCH_SIZE = 50
SAMPLE_MODE = True
SAMPLE_N_SCENARIOS = 20
SAMPLE_N_REPETITIONS = 1


def move_scenarios_to_processed(names, processed_dir):
    """Move dat/am files for the given scenario names into processed_dir."""
    moved = 0
    for name in names:
        for suffix in ("-dat.csv", "-am.csv"):
            src = os.path.join(_SCENARIOS_DIR, name + suffix)
            if os.path.exists(src):
                shutil.move(src, os.path.join(processed_dir, name + suffix))
                moved += 1
    return moved


def main(output_dir=".", num_workers=None):
    print("=" * 80)
    print("CausalMorph Sensitivity Analysis: Errors in Tentative Causal Order")
    print("=" * 80)

    all_names = list_scenario_names(_SCENARIOS_DIR)

    if SAMPLE_MODE:
        n_reps = SAMPLE_N_REPETITIONS
        all_names = [n for n in all_names if "_p-5_" in n or "_p-25_" in n]
        rng = np.random.RandomState(42)
        all_names = list(rng.choice(all_names, size=min(SAMPLE_N_SCENARIOS, len(all_names)), replace=False))
        print(f"\n** SAMPLE MODE: {len(all_names)} random scenarios (p=5 or p=25), "
              f"{n_reps} repetition(s) **")
    else:
        n_reps = N_REPETITIONS

    total_scenarios = len(all_names)
    n_configs = 2 * len(ERROR_RATES) - 1  # two sweeps sharing the (0,0) baseline
    total_controlled = total_scenarios * n_configs * n_reps
    total_real_pipeline = total_scenarios * n_reps
    total_experiments = total_controlled + total_real_pipeline
    exps_per_scenario = n_configs * n_reps + n_reps
    print(f"\nTotal scenarios: {total_scenarios}")
    print(f"Controlled experiments: {total_controlled}")
    print(f"Real pipeline experiments: {total_real_pipeline}")
    print(f"Total experiments: {total_experiments}")
    print(f"Experiments per scenario: {exps_per_scenario} "
          f"(1 baseline LiNGAM fit shared across all)")
    print(f"Processing in batches of {BATCH_SIZE} scenarios")

    # System info
    mem_info = psutil.virtual_memory()
    print(f"\nSystem Information:")
    print(f"  Available memory: {mem_info.available / (1024**3):.2f} GB")
    print(f"  CPU cores: {psutil.cpu_count(logical=False)}")

    if num_workers is None:
        num_workers = max(1, psutil.cpu_count(logical=False) - 2)
    print(f"  Workers: {num_workers}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "sample_" if SAMPLE_MODE else ""
    output_file = os.path.join(output_dir, f"sensitivity_results_{tag}{timestamp}.csv")
    processed_dir = os.path.join(_SCENARIOS_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Build the experiment grid (order_rate, adj_rate) pairs
    config_pairs = set()
    for rate in ERROR_RATES:
        config_pairs.add((rate, 0.0))
    for rate in ERROR_RATES:
        config_pairs.add((0.0, rate))
    config_pairs = sorted(config_pairs)

    header_written = False
    total_saved = 0
    total_moved = 0
    num_batches = math.ceil(total_scenarios / BATCH_SIZE)

    for batch_idx in range(num_batches):
        batch_names = all_names[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} "
              f"({len(batch_names)} scenarios) ---")

        # Build lightweight per-scenario configs (no data loaded in main process)
        scenario_configs = []
        for name in batch_names:
            seed = extract_seed(name)
            controlled_configs = [
                (order_rate, adj_rate, seed + rep)
                for order_rate, adj_rate in config_pairs
                for rep in range(n_reps)
            ]
            real_pipeline_seeds = [seed + rep for rep in range(n_reps)]
            scenario_configs.append({
                "data_dir": _SCENARIOS_DIR,
                "scenario_name": name,
                "controlled_configs": controlled_configs,
                "real_pipeline_seeds": real_pipeline_seeds,
            })

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_scenario_experiments, cfg): cfg
                for cfg in scenario_configs
            }

            for future in tqdm(as_completed(futures), total=len(scenario_configs),
                               desc=f"Batch {batch_idx + 1}/{num_batches}"):
                try:
                    scenario_results = future.result()
                    results.extend(scenario_results)
                except Exception as e:
                    print(f"Error in future: {e}")

        # Save batch results
        if results:
            df_batch = pd.DataFrame(results)
            df_batch.to_csv(output_file, index=False,
                            mode="a", header=not header_written)
            header_written = True
            total_saved += len(results)

        # Move processed files only in full mode
        if not SAMPLE_MODE:
            total_moved += move_scenarios_to_processed(batch_names, processed_dir)
        del results
        print(f"  Saved {total_saved} results so far")

    print(f"\nExperiments completed!")
    if SAMPLE_MODE:
        print(f"** SAMPLE MODE — files were NOT moved to processed/ **")
    print(f"Results saved to: {output_file}")
    print(f"Total successful experiments: {total_saved} / {total_experiments}")
    if not SAMPLE_MODE:
        print(f"Total files moved to processed/: {total_moved}")

    if total_saved > 0:
        df_results = pd.read_csv(output_file)
        df_controlled = df_results[df_results["experiment_type"] == "controlled"]
        df_real = df_results[df_results["experiment_type"] == "real_pipeline"]

        print("\nSummary by order_error_rate (adj_error_rate=0):")
        print("-" * 70)
        df_order = df_controlled[df_controlled["adj_error_rate"] == 0.0]
        if len(df_order) > 0:
            summary = df_order.groupby("order_error_rate").agg(
                mean_shd_improvement=("improvement", "mean"),
                mean_f1_improvement=("f1_improvement", "mean"),
                mean_shd_original=("shd_original", "mean"),
                mean_shd_transformed=("shd_transformed", "mean"),
                pct_improved=("improvement", lambda x: (x > 0).mean() * 100),
                count=("improvement", "count"),
            )
            print(summary.to_string())

        print("\nSummary by adj_error_rate (order_error_rate=0):")
        print("-" * 70)
        df_adj = df_controlled[df_controlled["order_error_rate"] == 0.0]
        if len(df_adj) > 0:
            summary = df_adj.groupby("adj_error_rate").agg(
                mean_shd_improvement=("improvement", "mean"),
                mean_f1_improvement=("f1_improvement", "mean"),
                mean_shd_original=("shd_original", "mean"),
                mean_shd_transformed=("shd_transformed", "mean"),
                pct_improved=("improvement", lambda x: (x > 0).mean() * 100),
                count=("improvement", "count"),
            )
            print(summary.to_string())

        # Real pipeline summary
        if len(df_real) > 0:
            print("\n" + "=" * 70)
            print("REAL PIPELINE RESULTS (LiNGAM estimates -> CausalMorph -> LiNGAM)")
            print("=" * 70)
            print(f"  Experiments: {len(df_real)}")
            print(f"  Mean measured order error rate: "
                  f"{df_real['measured_order_error_rate'].mean():.4f}")
            print(f"  Mean measured adj error rate:   "
                  f"{df_real['measured_adj_error_rate'].mean():.4f}")
            print(f"  Mean Kendall tau:               "
                  f"{df_real['kendall_tau'].mean():.4f}")
            print(f"  Mean F1 of input adj matrix:    "
                  f"{df_real['f1_input_adj'].mean():.4f}")
            print(f"\n  SHD Results:")
            print(f"    Mean SHD original:    {df_real['shd_original'].mean():.4f}")
            print(f"    Mean SHD transformed: {df_real['shd_transformed'].mean():.4f}")
            print(f"    Mean improvement:     {df_real['improvement'].mean():.4f}")
            pct = (df_real['improvement'] > 0).mean() * 100
            print(f"    Win rate:             {pct:.1f}%")
            print(f"\n  F1 Results:")
            print(f"    Mean F1 original:     {df_real['f1_original'].mean():.4f}")
            print(f"    Mean F1 transformed:  {df_real['f1_transformed'].mean():.4f}")
            print(f"    Mean F1 improvement:  {df_real['f1_improvement'].mean():.4f}")

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    np.random.seed(42)
    main(output_dir=output_dir, num_workers=num_workers)
