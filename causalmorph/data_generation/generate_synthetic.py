import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from components.scenarios.simple_scenario import (
    causal_graph_simple_scenario,
    causal_graph_simple_scenario_v2,
    causal_graph_simple_scenario_v3,
    causal_graph_simple_scenario_v25,
)
from components.scenarios.nonlinear_scenario import simulate_nonlinear_causal_scenario


def generate_random_dist_types(num_vars, normal_ratio=0.6):
    distribution_pool = ["normal", "uniform", "laplace", "exponential"]
    num_normals = int(num_vars * normal_ratio)
    dist_types = ["normal"] * num_normals
    other_types = [d for d in distribution_pool if d != "normal"]
    dist_types += random.choices(other_types, k=num_vars - num_normals)
    random.shuffle(dist_types)
    return dist_types


def generate_synthetic_datasets(
    output_dir="benchmarks/SyntheticCausalScenariosReduced_v2",
    repetitions=10,
    seed=42,
):
    """
    Reduced synthetic dataset generation:
    - num_vars: 5, 25, 40
    - pconn: 0.05, 0.5
    - deviation: 0.0, 0.5, 0.75
    - nsamples: 500, 5000
    - normal_ratio: 0.25, 0.75, 1.0
    Total configs: 3 × 2 × 3 × 2 × 3 × 10 = 1080 datasets
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    num_vars_values = [5, 25, 40]
    pconn_values = [0.5]
    deviation_values = [0.25, 0.5, 0.75]
    nsamples_values = [500, 5000]
    normal_ratios = [0.25, 1.0]

    total_datasets = 0
    for p in num_vars_values:
        for pconn in pconn_values:
            for dev in deviation_values:
                for n in nsamples_values:
                    for norm_ratio in normal_ratios:
                        if dev == 0.0 and norm_ratio == 1.0:
                            continue  # skip trivial case
                        total_datasets += repetitions

    with tqdm(total=total_datasets, desc="Generating synthetic datasets") as pbar:
        for num_vars in num_vars_values:
            for pconn in pconn_values:
                for deviation in deviation_values:
                    for nsamples in nsamples_values:
                        for normal_ratio in normal_ratios:
                            if deviation == 0.0 and normal_ratio == 1.0:
                                continue

                            dist_types = generate_random_dist_types(
                                num_vars, normal_ratio
                            )
                            dist_name = f"normal-{int(normal_ratio * 100)}"

                            for rep in range(1, repetitions + 1):
                                rep_seed = (
                                    seed
                                    + rep
                                    + int(normal_ratio * 100)
                                    + num_vars
                                    + int(pconn * 100)
                                )

                                adj_matrix, causal_data = (
                                    causal_graph_simple_scenario_v2(
                                        p=num_vars,
                                        pconn=pconn,
                                        dist=dist_types,
                                        deviation=deviation,
                                        seed=rep_seed,
                                        nsamples=nsamples,
                                    )
                                )

                                filename_base = (
                                    f"model_r-{rep}_p-{num_vars}_pconn-{pconn}_"
                                    f"{dist_name}_deviat-{deviation}_n-{nsamples}"
                                )

                                am_filename = os.path.join(
                                    output_dir, f"{filename_base}-am.csv"
                                )
                                dat_filename = os.path.join(
                                    output_dir, f"{filename_base}-dat.csv"
                                )

                                pd.DataFrame(adj_matrix).to_csv(
                                    am_filename, index=False
                                )
                                causal_data.to_csv(dat_filename, index=False)

                                pbar.update(1)

    print(f"✅ Generated {total_datasets} synthetic datasets in {output_dir}")


def generate_synthetic_datasets_nonlinear(
    output_dir="benchmarks/SyntheticCausalScenarios_mixed_v2_5",
    repetitions=10,
    seed=42,
):
    import os
    import random
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    # Define parameter values for synthetic dataset generation
    num_vars_values = [5, 25, 40]  # 3 values
    pconn_values = [0.25, 0.5, 0.75]  # 3 values
    deviation_values = [0, 0.25, 0.5, 0.75]  # 5 values
    nsamples_values = [50, 500, 5000]  # 3 values
    normal_ratios = [0, 0.25, 0.75, 1.0]  # 4 values
    nonlinearity_values = [0.25, 0.5, 0.75]  # 3 values
    modes = ["linear", "nonlinear"]  # 2 values

    ideal_lingam_flag = lambda mode, normal_ratio, nonlinearity: (
        mode == "linear" and normal_ratio == 1.0 and nonlinearity == 0.0
    )

    # Calculate exact number of datasets
    total_datasets = 0
    for mode in modes:
        nonlinearity_range = [0.0] if mode == "linear" else nonlinearity_values
        for p in num_vars_values:
            for pconn in pconn_values:
                for dev in deviation_values:
                    for n in nsamples_values:
                        for norm_ratio in normal_ratios:
                            for nl in nonlinearity_range:
                                total_datasets += repetitions

    print(f"Will generate {total_datasets} synthetic datasets")

    with tqdm(total=total_datasets, desc="Generating synthetic datasets") as pbar:
        for mode in modes:
            nonlinearity_range = [0.0] if mode == "linear" else nonlinearity_values
            for num_vars in num_vars_values:
                for pconn in pconn_values:
                    for deviation in deviation_values:
                        for nsamples in nsamples_values:
                            for normal_ratio in normal_ratios:
                                dist_types = generate_random_dist_types(
                                    num_vars, normal_ratio
                                )
                                dist_name = f"normal-{int(normal_ratio * 100)}"
                                for nonlinearity in nonlinearity_range:
                                    for rep in range(1, repetitions + 1):
                                        rep_seed = (
                                            seed
                                            + rep
                                            + int(normal_ratio * 100)
                                            + num_vars
                                            + int(pconn * 100)
                                        )
                                        # --- GENERATE DATASET ---
                                        adj_matrix, causal_data = (
                                            causal_graph_simple_scenario_v25(
                                                p=num_vars,
                                                pconn=pconn,
                                                dist=dist_types,
                                                deviation=deviation,
                                                seed=rep_seed,
                                                nsamples=nsamples,
                                                mode=mode,
                                                nonlinearity=nonlinearity,
                                            )
                                        )

                                        # Extra info for the filename:
                                        # Convert adj_matrix to numpy array if it's a DataFrame
                                        if isinstance(adj_matrix, pd.DataFrame):
                                            adj_matrix_np = adj_matrix.values
                                        else:
                                            adj_matrix_np = adj_matrix

                                        n_edges = int(np.sum(adj_matrix_np))
                                        density = n_edges / (num_vars * (num_vars - 1))
                                        is_ideal_lingam = ideal_lingam_flag(
                                            mode, normal_ratio, nonlinearity
                                        )
                                        is_control = is_ideal_lingam  # Ya aclaraste que _LiNGAM-ideal es tu control

                                        # Compose filename
                                        filename_base = (
                                            f"model_r-{rep}_p-{num_vars}_pconn-{pconn}_"
                                            f"{dist_name}_deviat-{deviation}_n-{nsamples}_mode-{mode}"
                                        )
                                        if mode == "nonlinear":
                                            filename_base += f"_nl-{nonlinearity}"
                                        filename_base += f"_edges-{n_edges}_dens-{density:.3f}_seed-{rep_seed}"
                                        if is_ideal_lingam:
                                            filename_base += "_LiNGAM-ideal"

                                        am_filename = os.path.join(
                                            output_dir, f"{filename_base}-am.csv"
                                        )
                                        dat_filename = os.path.join(
                                            output_dir, f"{filename_base}-dat.csv"
                                        )

                                        pd.DataFrame(adj_matrix).to_csv(
                                            am_filename, index=False
                                        )
                                        causal_data.to_csv(dat_filename, index=False)

                                        pbar.update(1)

    print(f"✅ Generated {total_datasets} synthetic datasets in {output_dir}")


if __name__ == "__main__":
    generate_synthetic_datasets_nonlinear()
