"""
CausalMorph Stage 0: orientation of a window whose residuals the Stage II gate calls
Gaussian, i.e. a window where the linear non-Gaussian learner has no orientation
information. The skeleton comes from PC (Fisher-z), the orientation from iterative
sink removal: among the remaining nodes, the sink is the one whose residual, after a
nonparametric regression on its remaining neighbours, is most independent of them
(normalised HSIC, Gaussian kernels, median heuristic). Every remaining neighbour of
the sink becomes its parent; the sink is removed; the scores of its neighbours are
recomputed. The output is a DAG on the PC skeleton and its causal order.

Nothing here uses non-Gaussianity: it is the ANM (Hoyer et al. 2009; Peters et al.
2014) source of orientation, so it only orients where the mechanisms are nonlinear.
"""
import contextlib
import io
from typing import List, Tuple

import numpy as np
from scipy.stats import gamma as gamma_dist
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

REGRESSORS = ("mlp", "krr3", "poly3")
_N_SUB = 600   # rows used by the kernel computations (HSIC is O(n^2) in memory)


def _gram(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).reshape(len(x), -1)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    pos = d2[d2 > 0]
    s2 = np.median(pos) / 2 if len(pos) else 1.0
    return np.exp(-d2 / (2 * s2))


def hsic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """(normalised statistic in [0, 1], p-value) of independence; p from the gamma
    approximation of the biased HSIC statistic (Gretton et al. 2008)."""
    K, L = _gram(x), _gram(y); m = len(K)
    H = np.eye(m) - 1.0 / m
    Kc, Lc = H @ K @ H, H @ L @ H
    stat = (Kc * Lc).sum() / m
    nstat = (Kc * Lc).sum() / np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    v = (Kc * Lc / 6) ** 2
    var = (v.sum() - np.trace(v)) / m / (m - 1) * 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3)
    K0, L0 = K - np.diag(np.diag(K)), L - np.diag(np.diag(L))
    mux, muy = K0.sum() / m / (m - 1), L0.sum() / m / (m - 1)
    mean = (1 + mux * muy - mux - muy) / m
    if var <= 0 or mean <= 0:
        return float(nstat), 1.0
    return float(nstat), float(1 - gamma_dist.cdf(stat, mean ** 2 / var, scale=var * m / mean))


def residual(kind: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual of y on X: mlp (one tanh layer of 10 units, L-BFGS), krr3 (RBF kernel
    ridge, median heuristic, alpha 1e-3) or poly3 (polynomial ridge, degree 3, or 2
    with more than five regressors)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    if kind == "mlp":
        m = MLPRegressor(hidden_layer_sizes=(10,), activation="tanh", solver="lbfgs", alpha=1e-4, max_iter=500, random_state=0)
    elif kind == "krr3":
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); pos = d2[d2 > 0]
        m = KernelRidge(alpha=1e-3, kernel="rbf", gamma=1.0 / np.median(pos) if len(pos) else 1.0)
    elif kind == "poly3":
        m = make_pipeline(PolynomialFeatures(3 if X.shape[1] <= 5 else 2, include_bias=False), Ridge(alpha=1e-3))
    else:
        raise ValueError(f"regressor must be one of {REGRESSORS}, got {kind!r}")
    return y - m.fit(X, y).predict(X)


def pc_skeleton(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Undirected adjacency of PC with Fisher-z tests (causal-learn)."""
    from causallearn.search.ConstraintBased.PC import pc
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        cg = pc(np.asarray(X, float), alpha=alpha, indep_test="fisherz", show_progress=False)
    g = np.asarray(cg.G.graph)
    return ((g != 0) | (g.T != 0)).astype(float)


def orient_skeleton(X: np.ndarray, skel: np.ndarray, regressor: str = "mlp") -> Tuple[np.ndarray, List[int]]:
    """DAG A[from, to] on the skeleton by iterative sink removal; returns (A, causal order)."""
    p = skel.shape[0]
    R = set(range(p)); A = np.zeros((p, p)); order: List[int] = []

    def neighbours(v):
        return sorted(j for j in R if skel[v, j] and j != v)

    def score(v):
        N = neighbours(v)
        return -np.inf if not N else hsic(X[:, N], residual(regressor, X[:, N], X[:, v]))[0]

    s = {v: score(v) for v in R}
    while R:
        v = min(R, key=lambda u: s[u])
        N = neighbours(v)
        A[N, v] = 1.0
        order.insert(0, v); R.remove(v)
        for u in N:
            s[u] = score(u)
    return A, order


def stage0(X: np.ndarray, regressor: str = "mlp", seed: int = 0, n_sub: int = _N_SUB) -> Tuple[np.ndarray, List[int]]:
    """PC skeleton oriented by sink removal, on a random subsample of at most n_sub rows."""
    X = np.asarray(X, float)
    if len(X) > n_sub:
        X = X[np.random.default_rng(seed).choice(len(X), n_sub, replace=False)]
    return orient_skeleton(X, pc_skeleton(X), regressor)
