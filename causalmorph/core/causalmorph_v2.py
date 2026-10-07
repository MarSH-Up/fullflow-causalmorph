"""
CausalMorph v2.

Same three stages as the published algorithm, with Stage II replaced:

  I.   MDL-selected polynomial, Taylor linearisation (reused from v1).
  II.  Gated, sample-wise, deterministic non-Gaussianisation of the residual:
       keep the residual if Jarque-Bera rejects normality at gate_alpha,
       otherwise map it monotonically (by ranks) onto a Laplace of the same
       standard deviation.  (v1 replaced the residual by an independent,
       shuffled synthetic draw, which erased every edge leaving a non-root
       node.)
  III. Orthogonalise against [1, parents], match the residual variance,
       reconstruct.

Parents are read from the *transformed* data, so the structural equations
the downstream learner sees are mutually consistent.  Nothing random is
drawn.  Adjacency layout is the one v1 reads: parents of node i are the
non-zeros of row i.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, laplace, rankdata
from sklearn.preprocessing import StandardScaler

from .causalmorph_algorithm import taylor_linearize

_EPS_VAR = 1e-8
_MIN_ROWS = 20   # below this Stage I over-fits and the gate is meaningless


def _rank_map_to_laplace(eps: np.ndarray) -> np.ndarray:
    """Monotone, deterministic map of eps onto a Laplace with sd(eps)."""
    n = len(eps)
    u = rankdata(eps, method="average") / (n + 1.0)       # strictly in (0, 1)
    z = laplace.ppf(u)                                     # standard Laplace quantiles
    z = z - z.mean()
    sd_z = z.std()
    if sd_z < _EPS_VAR:
        return eps.copy()
    return z * (eps.std() / sd_z)


def _stage2(eps: np.ndarray, stage2: str, gate_alpha: float) -> Tuple[np.ndarray, bool, float]:
    """Returns (eps_prime, gated, jb_p)."""
    if stage2 == "off":
        return eps.copy(), False, float("nan")
    if stage2 != "rankmap":
        raise ValueError(f"stage2 must be 'rankmap' or 'off', got {stage2!r}")
    jb_p = float(jarque_bera(eps).pvalue)
    if jb_p < gate_alpha:                                  # already non-Gaussian: keep
        return eps.copy(), False, jb_p
    return _rank_map_to_laplace(eps), True, jb_p


def _stage3(eps_prime: np.ndarray, parents: np.ndarray, target_sd: float) -> Optional[np.ndarray]:
    """Project eps_prime off span{1, parents}; rescale to target_sd. None if degenerate."""
    design = np.column_stack([np.ones(len(eps_prime)), parents])
    Q, _ = np.linalg.qr(design)
    e_orth = eps_prime - Q @ (Q.T @ eps_prime)
    sd = e_orth.std()
    if sd < _EPS_VAR:
        return None
    return e_orth * (target_sd / sd)


def causalmorph_v2(
    X: pd.DataFrame,
    causal_order: List[int],
    adjacency_matrix: pd.DataFrame,
    stage2: str = "rankmap",
    gate_alpha: float = 0.01,
    return_details: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, dict]]]:
    if stage2 not in ("rankmap", "off"):
        raise ValueError(f"stage2 must be 'rankmap' or 'off', got {stage2!r}")
    var_names = list(X.columns)
    X_vals = X.to_numpy(dtype=float)
    X_new = X_vals.copy()
    adj = np.asarray(adjacency_matrix, dtype=float)
    n = X_vals.shape[0]
    details: Dict[str, dict] = {}

    for idx in causal_order:
        name = var_names[idx]
        parents = np.where(adj[idx, :] != 0)[0]
        if len(parents) == 0:
            continue                                        # root: untouched, not reported
        rec = {"gated": False, "jb_p": float("nan"), "residual_before": None,
               "residual_after": None, "skipped": None}
        details[name] = rec
        y = X_vals[:, idx]
        P = X_new[:, parents]                                # transformed parents
        if n < _MIN_ROWS:
            rec["skipped"] = "too_few_rows"; continue
        if np.any(np.var(P, axis=0) < _EPS_VAR):
            rec["skipped"] = "constant_parent"; continue
        try:
            P_scaled = StandardScaler().fit_transform(P)
            y_lin = taylor_linearize(P_scaled, y)
        except Exception as exc:                             # noqa: BLE001 — v1 behaviour
            rec["skipped"] = f"stage1:{type(exc).__name__}"; continue
        eps = y - y_lin
        sd_eps = eps.std()
        rec["residual_before"] = eps.copy()
        if sd_eps < _EPS_VAR:
            rec["skipped"] = "zero_residual"; continue
        eps_prime, gated, jb_p = _stage2(eps, stage2, gate_alpha)
        rec["gated"], rec["jb_p"] = gated, jb_p
        e_final = _stage3(eps_prime, P, sd_eps)
        if e_final is None:
            rec["skipped"] = "degenerate_after_orthogonalisation"; continue
        X_new[:, idx] = y_lin + e_final
        rec["residual_after"] = e_final

    out = pd.DataFrame(X_new, index=X.index, columns=var_names)
    return (out, details) if return_details else out
