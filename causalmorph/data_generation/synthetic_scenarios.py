import numpy as np
import networkx as nx
import pandas as pd
from typing import Literal


def causal_graph_synthetic_scenarios(
    p=5,
    pconn=0.05,
    dist=None,
    deviation=0.2,  # Controls additive noise only, can range from 0 to 3.0 (0% to 300%)
    signal_strength=1.0,
    nsamples=500,
    seed=None,
    mode: Literal["linear", "nonlinear"] = "linear",
    nonlin_func: Literal["tanh", "square", "log1p"] = "tanh",
    nonlinearity=0.0,  # 0: pure linear, 0.0 < n <= 1.0: weight of non-linearity
    max_attempts=10,
    min_std=1e-6,
    min_edges=1,
):
    attempt = 0
    # Add more complex non-linear functions for richer synthetic scenarios
    f_dict = {
        "tanh": np.tanh,
        "square": np.square,
        "log1p": np.log1p,
        "cube": lambda x: x ** 3,
        "relu": lambda x: np.maximum(0, x),
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "abs": np.abs,
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "sign": np.sign,
    }
    if nonlin_func not in f_dict:
        raise ValueError(f"Unknown nonlin_func: {nonlin_func}. Available: {list(f_dict.keys())}")
    f = f_dict[nonlin_func]
    while attempt < max_attempts:
        attempt += 1
        if seed is not None:
            np.random.seed(seed + attempt)
        # Create directed acyclic graph (DAG)
        G = nx.DiGraph()
        G.add_nodes_from([f"V{i+1}" for i in range(p)])
        num_edges = 0
        # Add edges with probability pconn (only from lower to higher indices to ensure DAG)
        for i in range(p):
            for j in range(i + 1, p):
                if np.random.rand() < pconn:
                    G.add_edge(f"V{i+1}", f"V{j+1}")
                    num_edges += 1
        # Ensure minimum number of edges for interesting structure
        if num_edges < min_edges:
            continue
        adj_matrix = nx.to_pandas_adjacency(G, dtype=float)
        data = pd.DataFrame(index=np.arange(nsamples), columns=G.nodes(), dtype=float)
        if dist is None:
            dist = ["normal"] * p
        if len(dist) != p:
            raise ValueError("Length of 'dist' must match p.")
        # Generate data following topological order
        causal_order = list(nx.topological_sort(G))
        for i, node in enumerate(causal_order):
            parents = list(G.predecessors(node))
            if parents:
                # Generate random weights w_i ~ U(1.0, 3.0) for each parent
                weights = np.random.uniform(1.0, 3.0, size=len(parents))
                # Normalize: w_i = w_i / ||w||_2
                weights /= np.linalg.norm(weights)
                # Scale by signal strength: w_i = signal_strength * w_i
                weights *= signal_strength
                # Compute linear parent effect: parent_effect = X_parents @ w
                parent_effect = data[parents].values @ weights
                # === Key mechanism for linear vs nonlinear ===
                # If nonlinear mode with nonlinearity α > 0:
                # X_i = (1-α) * (X_parents @ w) + α * f(X_parents @ w) + ε_i
                # where f is the nonlinear function and ε_i is additive noise
                if mode == "nonlinear" and nonlinearity > 0:
                    transformed = f(parent_effect)
                    parent_effect = (
                        1 - nonlinearity
                    ) * parent_effect + nonlinearity * transformed
                # If linear mode, nonlinearity=0 => X_i = X_parents @ w + ε_i
            else:
                # Root node: no parents, only noise
                parent_effect = np.zeros(nsamples)
            
            # Use full deviation value without restrictions (can be up to 3.0)
            # This controls the variance/scale of additive noise: σ² or scale parameterc
            eff_deviation = max(deviation, 1e-3)  # Ensure non-zero
            
            # Additive noise ε_i ALWAYS controlled by deviation parameter
            # ε_i ~ Distribution(0, eff_deviation)
            if dist[i] == "normal":
                # ε_i ~ N(0, σ²) where σ = eff_deviation
                error = np.random.normal(0, eff_deviation, nsamples)
            elif dist[i] == "uniform":
                # ε_i ~ U(-a, a) where a = eff_deviation
                error = np.random.uniform(-eff_deviation, eff_deviation, nsamples)
            elif dist[i] == "laplace":
                # ε_i ~ Laplace(0, b) where b = eff_deviation (scale parameter)
                error = np.random.laplace(0, eff_deviation, nsamples)
            elif dist[i] == "exponential":
                # ε_i ~ Exp(λ) where λ = 1/eff_deviation (rate parameter)
                error = np.random.exponential(eff_deviation, nsamples)
            else:
                raise ValueError(f"Unsupported distribution: {dist[i]}")
            # Final structural equation: X_i = parent_effect + ε_i
            data[node] = parent_effect + error
        # Check that all variables have sufficient variance (std > min_std)
        stds = data.std()
        if (stds > min_std).all():
            return adj_matrix, data
    raise RuntimeError("Could not generate valid dataset.")
