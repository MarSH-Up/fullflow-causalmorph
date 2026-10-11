"""
CD-NOD (Huang, Zhang, Zhang, Ramsey, Sanchez-Romero, Glymour, Schölkopf;
JMLR 21(89), 2020) as an external baseline, through causal-learn's reference
implementation, evaluated with the pipeline's own metric functions.

CD-NOD runs once on the whole series with a surrogate context variable C
(the time index, as in the paper, or the detected segment label) and returns
one graph: the union of the regime graphs, with some edges left unoriented.
That single graph is scored against every true regime exactly as a pipeline
structure would be (compute_shd per regime, windowed nSHD, transition F1),
plus the orientation-free skeleton F1 so unoriented edges are not penalised
twice.

Conventions:
  causal-learn  graph[j, i] == 1 and graph[i, j] == -1  ->  i -> j
                graph[i, j] == graph[j, i] == -1         ->  i -- j (undirected)
                graph[i, j] == graph[j, i] == 1          ->  i <-> j
  returned      A[from, to] (the repository's truth convention); unoriented
                and bidirected pairs are set in both directions.

Independence tests: "kci" (kernel; what the CD-NOD paper uses; O(n^3), so it
runs on a time-uniform subsample) or "fisherz" (linear-Gaussian partial
correlation; fast on the whole series but blind to changes that are not
linear in C).
"""
import contextlib
import io
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def cdnod_to_adjacency(graph: np.ndarray, p: int) -> Tuple[np.ndarray, int]:
    """causal-learn CD-NOD graph ((p+1) x (p+1), C last) -> (A[from, to] in {0,1}, # unoriented pairs)."""
    g = np.asarray(graph)[:p, :p]
    A = np.zeros((p, p), dtype=int)
    n_unoriented = 0
    for i in range(p):
        for j in range(i + 1, p):
            a, b = g[i, j], g[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:          # i -> j
                A[i, j] = 1
            elif a == 1 and b == -1:        # j -> i
                A[j, i] = 1
            else:                           # -1/-1 undirected, 1/1 bidirected
                A[i, j] = A[j, i] = 1
                n_unoriented += 1
    return A, n_unoriented


def skeleton_prf(est, truth) -> Dict[str, float]:
    """Precision / recall / F1 on unordered adjacent pairs (orientation ignored)."""
    e = np.asarray(est.values if isinstance(est, pd.DataFrame) else est) != 0
    t = np.asarray(truth.values if isinstance(truth, pd.DataFrame) else truth) != 0
    e = e | e.T; t = t | t.T
    iu = np.triu_indices(e.shape[0], k=1)
    e, t = e[iu], t[iu]
    tp = int(np.sum(e & t)); fp = int(np.sum(e & ~t)); fn = int(np.sum(~e & t))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"Precision": P, "Recall": R, "F1": 2 * P * R / (P + R) if P + R else 0.0}


def time_uniform_subsample(T: int, n_sub: Optional[int]) -> np.ndarray:
    """n_sub row indices spread evenly over [0, T), first and last included; all rows if n_sub is None or >= T."""
    if n_sub is None or n_sub >= T:
        return np.arange(T)
    return np.unique(np.linspace(0, T - 1, n_sub).round().astype(int))


def _segment_labels(T: int, change_points: Sequence[int]) -> np.ndarray:
    lab = np.zeros(T)
    for k, cp in enumerate(sorted(change_points), start=1):
        lab[int(cp):] = k
    return lab + 1.0


def run_cdnod(X: np.ndarray, test: str = "kci", alpha: float = 0.05, n_sub: Optional[int] = 500,
              c_index: str = "time", change_points: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, dict]:
    """Run CD-NOD on the whole series. Returns (A[from, to], info)."""
    from causallearn.search.ConstraintBased.CDNOD import cdnod

    X = np.asarray(X, dtype=float)
    T, p = X.shape
    if c_index == "time":
        c = np.arange(1, T + 1, dtype=float)
    elif c_index == "segments":
        if change_points is None:
            raise ValueError("c_index='segments' needs change_points (e.g. the detector's)")
        c = _segment_labels(T, change_points)
    else:
        raise ValueError(f"c_index must be 'time' or 'segments', got {c_index!r}")
    idx = time_uniform_subsample(T, n_sub)
    t0 = time.time()
    with contextlib.redirect_stderr(io.StringIO()):       # causal-learn's skeleton search ignores show_progress
        cg = cdnod(X[idx], c[idx].reshape(-1, 1), alpha=alpha, indep_test=test, show_progress=False)
    runtime = time.time() - t0
    A, n_unoriented = cdnod_to_adjacency(cg.G.graph, p)
    g = np.asarray(cg.G.graph)
    n_context = int(np.sum((g[p, :p] != 0) | (g[:p, p] != 0)))
    return A, {"n_rows_used": int(len(idx)), "runtime_s": round(runtime, 3), "n_undirected": n_unoriented,
               "n_context_edges": n_context, "n_edges": int(A.sum() - n_unoriented)}


def evaluate_single_graph(A, true_adjs: List[pd.DataFrame], true_cps: Sequence[int], T: int,
                          names: List[str]) -> Dict[str, float]:
    """Score one graph that spans the whole series with the pipeline's metric functions."""
    from full_pipeline import RegimeStructure, compute_shd
    from evaluation.metrics import transition_f1, windowed_nshd_weighted

    p = len(names)
    A_df = pd.DataFrame(np.asarray(A, dtype=float), index=names, columns=names)
    per = [compute_shd(A_df, t) for t in true_adjs]
    sk = [skeleton_prf(A_df, t)["F1"] for t in true_adjs]
    s = RegimeStructure(regime_idx=0, window_start=0, window_end=T, n_samples=T,
                        causal_order=list(range(p)), adjacency_matrix=A_df, used_prior=False)
    last = compute_shd(A_df, pd.DataFrame((np.asarray(true_adjs[-1]) != 0).astype(float), index=names, columns=names))
    return {
        "mean_norm_shd": float(np.mean([r["normalized_shd"] for r in per])),
        "mean_struct_f1": float(np.mean([r["F1"] for r in per])),
        "mean_struct_precision": float(np.mean([r["Precision"] for r in per])),
        "mean_struct_recall": float(np.mean([r["Recall"] for r in per])),
        "mean_skeleton_f1": float(np.mean(sk)),
        "windowed_nshd_weighted": float(windowed_nshd_weighted([s], true_adjs, list(true_cps), T, p)),
        "transition_f1": float(transition_f1([s], true_adjs, list(true_cps), T, p)),
        "consensus_f1": last["F1"], "consensus_norm_shd": last["normalized_shd"],
    }
