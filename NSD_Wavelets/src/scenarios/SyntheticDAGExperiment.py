import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random


class SyntheticDAGExperimentWithCPTs:

    def __init__(self, num_vars, errminmax=(0.4, 1.0)):
        """
        Initialize the experiment with given parameters.
        :param num_vars: Number of variables (nodes) in the DAG.
        :param errminmax: Range for error standard deviations.
        """
        self.num_vars = num_vars
        self.errminmax = errminmax

    @staticmethod
    def generate_dag(num_nodes, conn_prob):
        """
        Generate a synthetic DAG.
        :param num_nodes: Number of nodes.
        :param conn_prob: Connection probability.
        :return: Directed Acyclic Graph (DAG).
        """
        while True:
            G = nx.gnp_random_graph(n=num_nodes, p=conn_prob, directed=True)
            G = nx.DiGraph([(u, v) for u, v in G.edges() if u < v])  # Ensure DAG
            if len(G.nodes) == num_nodes and nx.is_directed_acyclic_graph(G):
                return G

    @staticmethod
    def generate_error_distribution(num_vars, num_samples, errminmax):
        """
        Generate disturbance variables with non-Gaussian properties (e.g., Laplacian noise).
        Also, calculate and return the noise level (average standard deviation).
        """
        # Generate standard deviations for errors
        err_std = np.random.uniform(errminmax[0], errminmax[1], size=(num_vars, 1))

        # Generate non-Gaussian noise (Laplacian distribution)
        S = np.random.laplace(
            loc=0.0, scale=1.0, size=(num_vars, num_samples)
        )  # Laplace noise

        # Normalize to match the specified error standard deviations
        S = S / (
            (np.sqrt(np.mean(S**2, axis=1, keepdims=True)) / err_std)
            @ np.ones((1, num_samples))
        )

        # Return disturbances and the average noise level
        return S.T, np.mean(err_std)

    @staticmethod
    def simulate_data(dag, disturbances, num_samples):
        """
        Simulate data from a DAG using disturbance variables.
        :param dag: Directed Acyclic Graph.
        :param disturbances: Disturbance variables.
        :param num_samples: Number of samples.
        :return: Simulated data.
        """
        nodes = list(dag.nodes)
        node_idx_map = {node: idx for idx, node in enumerate(nodes)}
        data = np.zeros((num_samples, len(nodes)))
        for node in nx.topological_sort(dag):
            idx = node_idx_map[node]
            parents = [node_idx_map[parent] for parent in dag.predecessors(node)]
            data[:, idx] = (
                np.sum(data[:, parents], axis=1) + disturbances[:, idx]
                if parents
                else disturbances[:, idx]
            )
        return data

    @staticmethod
    def generate_cpts(dag):
        """
        Generate random Conditional Probability Tables (CPTs) for the DAG.
        :param dag: Directed Acyclic Graph.
        :return: Dictionary of CPTs.
        """
        cpts = {}
        for node in dag.nodes:
            parents = list(dag.predecessors(node))
            num_parent_states = 2 ** len(parents)  # Assume binary states for simplicity
            cpt = np.random.rand(
                num_parent_states, 2
            )  # Binary states for the current node
            cpt = cpt / cpt.sum(
                axis=1, keepdims=True
            )  # Normalize to ensure probabilities sum to 1
            cpts[node] = {"parents": parents, "table": cpt}
        return cpts

    def create_multiple_experiments(self, num_experiments):
        """
        Create multiple experiments, each with a unique DAG, data, CPTs, and noise levels.
        """
        experiments = []
        for _ in range(num_experiments):
            conn_prob = random.uniform(0.2, 0.6)
            num_samples = random.randint(500, 1500)
            dag = self.generate_dag(self.num_vars, conn_prob)
            disturbances, noise_level = self.generate_error_distribution(
                self.num_vars, num_samples, self.errminmax
            )
            data = self.simulate_data(dag, disturbances, num_samples)
            cpts = self.generate_cpts(dag)
            experiments.append(
                {
                    "dag": dag,
                    "data": data,
                    "cpts": cpts,
                    "connection_probability": conn_prob,
                    "num_samples": num_samples,
                    "noise_level": noise_level,
                }
            )
        return experiments

    def print_cpts(self, cpts):
        """
        Print the CPTs in a readable format.
        :param cpts: Dictionary of CPTs.
        """
        for node, cpt_info in cpts.items():
            print(f"Node {node}:")
            print(f"  Parents: {cpt_info['parents']}")
            print(f"  CPT:")
            print(cpt_info["table"])
            print()

    def plot_dag(self, dag):
        """
        Plot the DAG.
        :param dag: Directed Acyclic Graph.
        """
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(dag)
        nx.draw(
            dag,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=2000,
            edge_color="gray",
            arrowsize=20,
        )
        plt.title("Synthetic DAG")
        plt.show()

    def plot_data(self, data, change_points=None):
        """
        Plot the simulated data.
        :param data: Simulated data.
        :param change_points: List of change points in the data (optional).
        """
        num_vars = data.shape[1]
        fig, axes = plt.subplots(num_vars, figsize=(12, 8), sharex=True)
        for i in range(num_vars):
            axes[i].plot(data[:, i], label=f"Variable {i}")
            axes[i].legend(loc="upper right")
            axes[i].grid(True)
        if change_points:
            for ax in axes:
                for cp in change_points:
                    ax.axvline(cp, color="red", linestyle="--")
        plt.suptitle("Simulated Data")
        plt.xlabel("Sample Index")
        plt.show()

    def plot_experiments(self, experiments):
        """
        Plot the DAGs and combined time series data for all experiments.
        :param experiments: List of experiments.
        """
        num_experiments = len(experiments)

        # Plot the DAGs
        plt.figure(figsize=(18, 8))
        for idx, exp in enumerate(experiments):
            plt.subplot(1, num_experiments, idx + 1)
            nx.draw(
                exp["dag"],
                pos=nx.spring_layout(exp["dag"]),
                with_labels=True,
                node_color="lightblue",
                node_size=3000,
                edge_color="gray",
                arrowsize=20,
                width=4,
                font_size=14,
            )
            plt.title(
                f"DAG {idx + 1} (pconn={exp['connection_probability']:.2f})",
                fontsize=32,
            )
        plt.tight_layout()
        plt.show()

        # Combine data sequentially and plot
        combined_data = np.vstack([exp["data"] for exp in experiments])
        fig, axes = plt.subplots(self.num_vars, figsize=(14, 10), sharex=True)
        sample_offset = 0

        for idx, exp in enumerate(experiments):
            num_samples = exp["num_samples"]
            for i in range(self.num_vars):
                axes[i].plot(
                    range(sample_offset, sample_offset + num_samples),
                    exp["data"][:, i],
                    label=f"Variable {i}",
                    linewidth=2,
                )
            sample_offset += num_samples
            for ax in axes:
                ax.axvline(
                    sample_offset,
                    color="red",
                    linestyle="--",
                    label=(
                        f"Change Point {idx + 1}"
                        if idx < len(experiments) - 1
                        else None
                    ),
                    linewidth=2,
                )

        for i in range(self.num_vars):
            axes[i].legend(loc="upper right", fontsize=10)
            axes[i].grid(True, linestyle="--", linewidth=0.5)
            axes[i].set_ylabel(f"Variable {i}", fontsize=12)
            axes[i].tick_params(axis="both", labelsize=10)

        plt.suptitle("Combined Time Series with Structure Changes", fontsize=16)
        plt.xlabel("Sample Index", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
