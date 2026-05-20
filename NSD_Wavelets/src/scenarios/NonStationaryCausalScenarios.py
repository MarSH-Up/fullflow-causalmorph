import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, List, Dict, Optional, Tuple


class NonStationaryCausalScenario:
    """
    Generate non-stationary causal scenarios following Pearl's Structural Causal Model (SCM) framework.

    This class allows creating time series data where the underlying causal structure (DAG)
    and/or parameters (weights, noise levels, etc.) change at specific time points, while
    maintaining a consistent functional form (linear or nonlinear) throughout.

    Each regime follows Pearl's SCM: X_i = f_i(PA_i, U_i)
    where PA_i are parents of X_i, U_i is exogenous noise, and f_i is the structural equation.

    In non-stationary scenarios:
    - Structural changes: The DAG structure (parent sets PA_i) changes across regimes
    - Parametric changes: Edge weights, noise distributions/levels change across regimes
    - Functional form: Remains consistent (linear or nonlinear) across all regimes
    """

    def __init__(
        self,
        p: int,
        mode: Literal["linear", "nonlinear"] = "linear",
        nonlin_func: Literal["sigmoid", "tanh", "cube", "softplus", "elu", "leaky_relu"] = "tanh",
        nonlinearity: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the non-stationary causal scenario generator.

        Parameters:
        -----------
        p : int
            Number of variables (nodes) in the causal graph
        mode : str
            'linear' or 'nonlinear' - consistent across all regimes
        nonlin_func : str
            Nonlinear activation function (only used if mode='nonlinear'):
            - 'sigmoid': σ(x) = 1/(1 + exp(-x))
            - 'tanh': tanh(x)
            - 'cube': x^3
            - 'softplus': log(1 + exp(x))
            - 'elu': x if x > 0 else α(exp(x) - 1)
            - 'leaky_relu': x if x > 0 else αx
        nonlinearity : float
            Weight of nonlinearity, 0.0 = pure linear, 1.0 = pure nonlinear
            (only used if mode='nonlinear')
        seed : int, optional
            Random seed for reproducibility
        """
        self.p = p
        self.mode = mode
        self.nonlin_func = nonlin_func
        self.nonlinearity = nonlinearity
        self.seed = seed
        self.variable_names = [f"V{i+1}" for i in range(p)]

        # Validate nonlinearity parameter
        if self.mode == "nonlinear" and not (0.0 <= self.nonlinearity <= 1.0):
            raise ValueError("nonlinearity must be between 0.0 and 1.0")

        # Set up nonlinear function (actual nonlinear functions used in ML/causal modeling)
        f_dict = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50))),  # Clip to avoid overflow
            "tanh": np.tanh,
            "cube": lambda x: x ** 3,
            "softplus": lambda x: np.log1p(np.exp(np.clip(x, -50, 50))),  # Smooth approximation of ReLU
            "elu": lambda x: np.where(x > 0, x, 0.1 * (np.exp(x) - 1)),  # α=0.1
            "leaky_relu": lambda x: np.where(x > 0, x, 0.1 * x),  # α=0.1
        }

        if self.nonlin_func not in f_dict:
            raise ValueError(f"Unknown nonlin_func: {self.nonlin_func}")

        self.nonlin_f = f_dict[self.nonlin_func]

    @staticmethod
    def _edge_change_metrics(
        base_edges: set,
        new_edges: set,
        max_edges: int,
    ) -> Dict:
        """Compute symmetric difference and Jaccard metrics between two edge sets."""
        sym_diff = base_edges ^ new_edges
        inter = base_edges & new_edges
        union = base_edges | new_edges

        # Symmetric-diff relative to max possible edges in a DAG
        pct_symdiff_max = 100.0 * (len(sym_diff) / max_edges) if max_edges > 0 else 0.0

        # Relative to base edge count (often useful)
        pct_symdiff_base = 100.0 * (len(sym_diff) / max(1, len(base_edges)))

        jaccard = (len(inter) / len(union)) if len(union) > 0 else 1.0

        return {
            "sym_diff_edges": list(sym_diff),
            "n_sym_diff": len(sym_diff),
            "pct_symdiff_over_max": pct_symdiff_max,
            "pct_symdiff_over_base": pct_symdiff_base,
            "jaccard_edges": jaccard,
            "n_intersection": len(inter),
            "n_union": len(union),
        }

    @staticmethod
    def calculate_structural_change(
        p: int,
        base_graph: nx.DiGraph,
        target_change_pct: float,
        change_type: Literal["add", "remove", "mixed"] = "mixed",
        seed: Optional[int] = None,
    ) -> Tuple[nx.DiGraph, float, Dict]:
        """
        Create a new graph by modifying edges to achieve a target percentage change.

        The percentage is calculated relative to the maximum possible edges in a DAG: p*(p-1)/2

        Parameters:
        -----------
        p : int
            Number of nodes
        base_graph : nx.DiGraph
            Starting graph structure
        target_change_pct : float
            Target percentage of edges to change (0.0 to 100.0)
        change_type : str
            "add" - only add edges
            "remove" - only remove edges
            "mixed" - both add and remove edges (50/50 split if possible)

        Returns:
        --------
        tuple : (new_graph, actual_change_pct, change_info)
            - new_graph: Modified DAG
            - actual_change_pct: Actual percentage change achieved (rounded)
            - change_info: Dict with details about changes made
        """
        # Use proper RNG
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        max_edges = p * (p - 1) // 2  # Maximum edges in a DAG
        base_edges = set(base_graph.edges())
        n_current = len(base_edges)

        # Calculate target number of edges to change
        target_n_changes = round((target_change_pct / 100.0) * max_edges)

        # Ensure at least 1 change if target_change_pct > 0
        if target_change_pct > 0 and target_n_changes == 0:
            target_n_changes = 1

        # Get all possible edges (respecting DAG constraint: i < j)
        all_possible_edges = set()
        nodes = sorted(base_graph.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                all_possible_edges.add((u, v))

        # Edges that can be added or removed
        # Sort for determinism (set iteration order is hash-randomized across Python processes)
        edges_can_add = sorted(all_possible_edges - base_edges)
        edges_can_remove = sorted(base_edges)

        # Determine how many edges to add/remove
        n_add = 0
        n_remove = 0

        if change_type == "add":
            n_add = min(target_n_changes, len(edges_can_add))
        elif change_type == "remove":
            n_remove = min(target_n_changes, len(edges_can_remove))
        elif change_type == "mixed":
            # Split changes between add and remove
            n_add = min(target_n_changes // 2, len(edges_can_add))
            n_remove = min(target_n_changes // 2, len(edges_can_remove))

            # If we couldn't achieve target with 50/50 split, compensate
            remaining = target_n_changes - (n_add + n_remove)
            if remaining > 0:
                # Try to add more edges first
                additional_add = min(remaining, len(edges_can_add) - n_add)
                n_add += additional_add
                remaining -= additional_add

                # If still remaining, try to remove more
                if remaining > 0:
                    additional_remove = min(remaining, len(edges_can_remove) - n_remove)
                    n_remove += additional_remove

        # Create new graph
        new_graph = base_graph.copy()

        # Remove edges
        edges_to_remove = []
        if n_remove > 0:
            idx = rng.choice(len(edges_can_remove), size=n_remove, replace=False)
            edges_to_remove = [edges_can_remove[i] for i in idx]
            new_graph.remove_edges_from(edges_to_remove)

        # Add edges
        edges_to_add = []
        if n_add > 0:
            idx = rng.choice(len(edges_can_add), size=n_add, replace=False)
            edges_to_add = [edges_can_add[i] for i in idx]
            new_graph.add_edges_from(edges_to_add)

        # Calculate actual change percentage
        new_edges = set(new_graph.edges())
        total_changes = n_add + n_remove
        actual_change_pct = (total_changes / max_edges) * 100.0 if max_edges > 0 else 0.0

        change_info = {
            "target_change_pct": target_change_pct,
            "actual_change_pct": actual_change_pct,
            "max_possible_edges": max_edges,
            "n_edges_added": n_add,
            "n_edges_removed": n_remove,
            "total_changes": total_changes,
            "edges_added": edges_to_add,
            "edges_removed": edges_to_remove,
            "base_n_edges": n_current,
            "new_n_edges": new_graph.number_of_edges(),
            # Include symmetric-diff metrics for accurate change measurement
            **NonStationaryCausalScenario._edge_change_metrics(base_edges, new_edges, max_edges),
        }

        return new_graph, actual_change_pct, change_info

    @staticmethod
    def create_regime_pair_with_change(
        p: int,
        base_pconn: float = 0.3,
        change_pct: float = 20.0,
        change_type: Literal["add", "remove", "mixed"] = "mixed",
        seed: Optional[int] = None,
        base_graph: Optional[nx.DiGraph] = None,
    ) -> Tuple[nx.DiGraph, nx.DiGraph, Dict]:
        """
        Create a pair of DAG structures with a controlled percentage change.

        This is a helper function for benchmarking. If base_graph is provided,
        changes are applied to it. Otherwise, a random base graph is generated.

        Parameters:
        -----------
        p : int
            Number of nodes
        base_pconn : float
            Connection probability for the base graph (only used if base_graph is None)
        change_pct : float
            Target percentage of edges to change (0.0 to 100.0)
        change_type : str
            Type of change: "add", "remove", or "mixed"
        seed : int, optional
            Random seed for reproducibility
        base_graph : nx.DiGraph, optional
            If provided, apply changes to this graph instead of generating a new one.
            This enables chaining changes for multi-regime experiments.

        Returns:
        --------
        tuple : (base_graph, changed_graph, change_info)
        """
        if seed is not None:
            np.random.seed(seed)

        if base_graph is not None:
            # Use provided graph as base
            G_base = base_graph.copy()
        else:
            # Generate random base graph
            while True:
                G_base = nx.DiGraph()
                nodes = [f"V{i+1}" for i in range(p)]
                G_base.add_nodes_from(nodes)

                # Add edges with probability pconn
                for i in range(p):
                    for j in range(i + 1, p):
                        if np.random.rand() < base_pconn:
                            G_base.add_edge(nodes[i], nodes[j])

                # Ensure at least 1 edge
                if G_base.number_of_edges() >= 1:
                    break

        # Create changed graph
        G_changed, actual_pct, change_info = (
            NonStationaryCausalScenario.calculate_structural_change(
                p=p,
                base_graph=G_base,
                target_change_pct=change_pct,
                change_type=change_type,
                seed=seed,
            )
        )

        return G_base, G_changed, change_info

    def generate_regime(
        self,
        pconn: float = 0.3,
        dist: Optional[List[str]] = None,
        deviation: float = 0.5,
        signal_strength: float = 1.0,
        nsamples: int = 500,
        max_attempts: int = 10,
        min_std: float = 1e-6,
        min_edges: int = 1,
        fixed_graph: Optional[nx.DiGraph] = None,
        regime_seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate a single regime with specific causal structure and parameters.

        Note: The functional form (linear/nonlinear) is determined at class initialization
        and remains consistent across all regimes. Only structure and parameters vary.

        Parameters:
        -----------
        pconn : float
            Connection probability for edges in the DAG
        dist : list of str, optional
            Distribution types for noise (e.g., ['normal', 'laplace', ...])
        deviation : float
            Scale of additive noise (0.0 to 3.0)
        signal_strength : float
            Overall signal strength multiplier
        nsamples : int
            Number of samples to generate
        max_attempts : int
            Maximum attempts to generate valid graph
        min_std : float
            Minimum standard deviation for variables
        min_edges : int
            Minimum number of edges required
        fixed_graph : nx.DiGraph, optional
            Use a predefined graph structure instead of generating new one
        regime_seed : int, optional
            Seed for this specific regime (overrides global seed)

        Returns:
        --------
        dict : Dictionary containing 'graph', 'data', 'adj_matrix', and 'params'
        """
        # Set seed for this regime
        if regime_seed is not None:
            np.random.seed(regime_seed)
        elif self.seed is not None:
            np.random.seed(self.seed)

        # Use fixed graph or generate new one
        if fixed_graph is not None:
            G = fixed_graph.copy()
        else:
            # Try to generate valid graph
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                G = nx.DiGraph()
                G.add_nodes_from(self.variable_names)
                num_edges = 0

                # Add edges with probability pconn (only from lower to higher indices for DAG)
                for i in range(self.p):
                    for j in range(i + 1, self.p):
                        if np.random.rand() < pconn:
                            G.add_edge(f"V{i+1}", f"V{j+1}")
                            num_edges += 1

                # Check minimum edges requirement
                if num_edges >= min_edges:
                    break

            if num_edges < min_edges:
                raise RuntimeError(f"Could not generate graph with at least {min_edges} edges")

        # Get adjacency matrix
        adj_matrix = nx.to_pandas_adjacency(G, dtype=float)

        # Initialize data to zeros (NaN would cascade through the topological order)
        data = pd.DataFrame(
            np.zeros((nsamples, self.p)),
            index=np.arange(nsamples),
            columns=self.variable_names,
            dtype=float,
        )

        # Set up noise distributions
        if dist is None:
            dist = ["normal"] * self.p
        if len(dist) != self.p:
            raise ValueError("Length of 'dist' must match p.")

        # Generate data following topological order
        causal_order = list(nx.topological_sort(G))
        weights_dict = {}  # Store weights for each node

        for i, node in enumerate(causal_order):
            parents = list(G.predecessors(node))

            if parents:
                # Generate random weights
                weights = np.random.uniform(1.0, 3.0, size=len(parents))
                weights /= np.linalg.norm(weights)
                weights *= signal_strength
                weights_dict[node] = dict(zip(parents, weights))

                # Compute parent effect
                # Use np.dot to avoid spurious FP exception warnings from Apple Accelerate BLAS
                parent_vals = np.ascontiguousarray(data[parents].values, dtype=np.float64)
                parent_effect = np.dot(parent_vals, weights)

                # Apply nonlinearity if specified (uses class-level settings)
                if self.mode == "nonlinear" and self.nonlinearity > 0:
                    transformed = self.nonlin_f(parent_effect)
                    parent_effect = (1 - self.nonlinearity) * parent_effect + self.nonlinearity * transformed
            else:
                parent_effect = np.zeros(nsamples)
                weights_dict[node] = {}

            # Generate additive noise
            eff_deviation = max(deviation, 1e-3)

            if dist[i] == "normal":
                error = np.random.normal(0, eff_deviation, nsamples)
            elif dist[i] == "uniform":
                error = np.random.uniform(-eff_deviation, eff_deviation, nsamples)
            elif dist[i] == "laplace":
                error = np.random.laplace(0, eff_deviation, nsamples)
            elif dist[i] == "exponential":
                error = np.random.exponential(eff_deviation, nsamples)
            else:
                raise ValueError(f"Unsupported distribution: {dist[i]}")

            # Final structural equation (clip to prevent overflow in deep DAGs)
            data[node] = np.clip(parent_effect + error, -50, 50)

        # Check variance
        stds = data.std()
        if not (stds > min_std).all():
            raise RuntimeError("Generated data has insufficient variance")

        return {
            "graph": G,
            "data": data,
            "adj_matrix": adj_matrix,
            "params": {
                "pconn": pconn,
                "deviation": deviation,
                "signal_strength": signal_strength,
                "dist": dist,
                "weights": weights_dict,
            }
        }

    def create_nonstationary_scenario(
        self,
        regime_configs: List[Dict],
        transition_type: Literal["abrupt", "smooth"] = "abrupt",
        transition_length: int = 0,
    ) -> Dict:
        """
        Create a non-stationary scenario with multiple regimes.

        Parameters:
        -----------
        regime_configs : list of dict
            List of configuration dictionaries for each regime.
            Each dict should contain parameters for generate_regime()
        transition_type : str
            'abrupt' for instant changes, 'smooth' for gradual transitions
        transition_length : int
            Number of samples for smooth transition (only used if transition_type='smooth')

        Returns:
        --------
        dict : Dictionary containing:
            - 'regimes': list of regime dictionaries
            - 'combined_data': concatenated time series data
            - 'change_points': list of sample indices where regimes change
            - 'regime_labels': array indicating which regime each sample belongs to
        """
        regimes = []
        change_points = []
        current_sample = 0

        # Generate each regime
        for idx, config in enumerate(regime_configs):
            regime = self.generate_regime(**config)
            regimes.append(regime)

            if idx > 0:
                change_points.append(current_sample)

            current_sample += regime["data"].shape[0]

        # Combine data from all regimes
        combined_data = pd.concat(
            [regime["data"] for regime in regimes],
            ignore_index=True
        )

        # Create regime labels
        regime_labels = np.zeros(combined_data.shape[0], dtype=int)
        current_idx = 0
        for regime_idx, regime in enumerate(regimes):
            regime_length = regime["data"].shape[0]
            regime_labels[current_idx:current_idx + regime_length] = regime_idx
            current_idx += regime_length

        return {
            "regimes": regimes,
            "combined_data": combined_data,
            "change_points": change_points,
            "regime_labels": regime_labels,
            "transition_type": transition_type,
        }

    def plot_structures(
        self,
        regimes: List[Dict],
        figsize: Tuple[int, int] = (18, 6),
        node_size: int = 2000,
        font_size: int = 12,
        title_size: int = 14,
    ):
        """
        Plot the causal structures (DAGs) for all regimes.

        Parameters:
        -----------
        regimes : list of dict
            List of regime dictionaries from create_nonstationary_scenario()
        figsize : tuple
            Figure size (width, height)
        node_size : int
            Size of nodes in the graph
        font_size : int
            Font size for node labels
        title_size : int
            Font size for subplot titles
        """
        num_regimes = len(regimes)
        fig, axes = plt.subplots(1, num_regimes, figsize=figsize)

        if num_regimes == 1:
            axes = [axes]

        for idx, (regime, ax) in enumerate(zip(regimes, axes)):
            G = regime["graph"]
            params = regime["params"]

            plt.sca(ax)
            pos = nx.spring_layout(G, seed=42)
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=node_size,
                edge_color="gray",
                arrowsize=20,
                font_size=font_size,
                ax=ax,
            )

            # Create informative title
            title = f"Regime {idx + 1}\n"
            title += f"pconn={params['pconn']:.2f}, "
            title += f"dev={params['deviation']:.2f}\n"
            title += f"signal={params['signal_strength']:.2f}"

            ax.set_title(title, fontsize=title_size)

        plt.tight_layout()
        plt.show()

    def plot_timeseries(
        self,
        combined_data: pd.DataFrame,
        change_points: List[int],
        regime_labels: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (16, 10),
        linewidth: float = 1.5,
        changepoint_linewidth: float = 2.5,
        show_regime_backgrounds: bool = True,
    ):
        """
        Plot the combined time series with change points marked.

        Parameters:
        -----------
        combined_data : pd.DataFrame
            Combined time series data
        change_points : list of int
            Indices where regime changes occur
        regime_labels : np.ndarray, optional
            Array indicating which regime each sample belongs to
        figsize : tuple
            Figure size (width, height)
        linewidth : float
            Line width for time series
        changepoint_linewidth : float
            Line width for change point markers
        show_regime_backgrounds : bool
            Whether to show colored backgrounds for different regimes
        """
        num_vars = combined_data.shape[1]
        fig, axes = plt.subplots(num_vars, 1, figsize=figsize, sharex=True)

        if num_vars == 1:
            axes = [axes]

        # Define colors for regime backgrounds
        regime_colors = plt.cm.Set3(np.linspace(0, 1, len(change_points) + 1))

        for i, (var_name, ax) in enumerate(zip(combined_data.columns, axes)):
            # Plot time series
            ax.plot(
                combined_data.index,
                combined_data[var_name],
                linewidth=linewidth,
                color='navy',
                alpha=0.7,
            )

            # Add regime backgrounds
            if show_regime_backgrounds and regime_labels is not None:
                regime_changes = [0] + change_points + [len(combined_data)]
                for regime_idx in range(len(regime_changes) - 1):
                    start = regime_changes[regime_idx]
                    end = regime_changes[regime_idx + 1]
                    ax.axvspan(
                        start, end,
                        alpha=0.15,
                        color=regime_colors[regime_idx],
                        zorder=0,
                    )

            # Mark change points
            for cp_idx, cp in enumerate(change_points):
                ax.axvline(
                    cp,
                    color='red',
                    linestyle='--',
                    linewidth=changepoint_linewidth,
                    alpha=0.8,
                    label=f'Change Point {cp_idx + 1}' if i == 0 else None,
                )

            ax.set_ylabel(var_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=10)

            # Add legend only to first subplot
            if i == 0 and change_points:
                ax.legend(loc='upper right', fontsize=10)

        plt.xlabel('Sample Index', fontsize=14, fontweight='bold')
        plt.suptitle(
            'Non-Stationary Time Series with Structural Changes',
            fontsize=16,
            fontweight='bold',
            y=0.995,
        )
        plt.tight_layout()
        plt.show()

    def plot_scenario(
        self,
        scenario: Dict,
        structures_figsize: Tuple[int, int] = (18, 6),
        timeseries_figsize: Tuple[int, int] = (16, 10),
    ):
        """
        Plot both structures and time series for a complete scenario.

        Parameters:
        -----------
        scenario : dict
            Scenario dictionary from create_nonstationary_scenario()
        structures_figsize : tuple
            Figure size for structure plots
        timeseries_figsize : tuple
            Figure size for time series plot
        """
        print("=" * 80)
        print("Non-Stationary Causal Scenario (Pearl's SCM Framework)")
        print("=" * 80)
        print(f"Functional form: {self.mode.upper()}")
        if self.mode == "nonlinear":
            print(f"Nonlinear function: {self.nonlin_func}")
            print(f"Nonlinearity weight: {self.nonlinearity:.3f}")
        print()

        print("=" * 80)
        print("Causal Structure Changes")
        print("=" * 80)
        self.plot_structures(scenario["regimes"], figsize=structures_figsize)

        print("\n" + "=" * 80)
        print("Combined Time Series with Change Points")
        print("=" * 80)
        self.plot_timeseries(
            scenario["combined_data"],
            scenario["change_points"],
            scenario["regime_labels"],
            figsize=timeseries_figsize,
        )

        # Print summary statistics
        print("\n" + "=" * 80)
        print("Scenario Summary")
        print("=" * 80)
        for idx, regime in enumerate(scenario["regimes"]):
            print(f"\nRegime {idx + 1}:")
            print(f"  Samples: {regime['data'].shape[0]}")
            print(f"  Edges: {regime['graph'].number_of_edges()}")
            print(f"  Connection prob: {regime['params']['pconn']:.3f}")
            print(f"  Noise deviation: {regime['params']['deviation']:.3f}")
            print(f"  Signal strength: {regime['params']['signal_strength']:.3f}")
            # Show which noise distributions are used
            dist_summary = ", ".join(set(regime['params']['dist']))
            print(f"  Noise distributions: {dist_summary}")

        print(f"\nTotal samples: {scenario['combined_data'].shape[0]}")
        print(f"Change points at: {scenario['change_points']}")


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EXAMPLE: Non-Stationary Causal Scenario - Pure Structural Change")
    print("=" * 80)
    print("\nThis example demonstrates Pearl's SCM framework with pure structural changes.")
    print("The functional form (linear) and all parameters remain constant.")
    print("Only the causal structure (DAG) changes between regimes.\n")

    # Initialize scenario generator with 5 variables and LINEAR mode
    scenario_gen = NonStationaryCausalScenario(
        p=5,
        mode="linear",  # Consistent functional form across all regimes
        seed=42,
    )

    # Define two regimes with SAME parameters but DIFFERENT structures
    # Parameters are kept constant to isolate the effect of structural changes
    regime_configs = [
        {
            # Regime 1: Sparse structure
            "pconn": 0.2,  # Low connection probability → sparse DAG
            "deviation": 0.5,
            "signal_strength": 1.5,
            "nsamples": 500,
            "dist": ["normal"] * 5,
            "regime_seed": 100,  # Different seed → different DAG structure
        },
        {
            # Regime 2: Dense structure (same parameters, different structure)
            "pconn": 0.5,  # High connection probability → dense DAG
            "deviation": 0.5,  # SAME as regime 1
            "signal_strength": 1.5,  # SAME as regime 1
            "nsamples": 500,  # SAME as regime 1
            "dist": ["normal"] * 5,  # SAME as regime 1
            "regime_seed": 200,  # Different seed → different DAG structure
        },
    ]

    # Generate the non-stationary scenario
    print("Generating scenario with 2 regimes (pure structural change)...")
    scenario = scenario_gen.create_nonstationary_scenario(
        regime_configs=regime_configs,
        transition_type="abrupt",
    )

    print("Done! Plotting results...\n")

    # Plot the complete scenario (structures + time series)
    scenario_gen.plot_scenario(
        scenario,
        structures_figsize=(14, 6),
        timeseries_figsize=(16, 10),
    )
