"""
Evaluation metrics for non-stationarity detection.

Provides functions to evaluate detector performance against ground truth.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Results from evaluating detection against ground truth."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_pairs: List[Tuple[int, int]]  # (ground_truth, detection)
    mean_timing_error: Optional[float]    # Mean abs error of matched detections


def evaluate_detection(
    ground_truth: List[int],
    detections: List[int],
    tolerance: int = 50,
) -> EvaluationResult:
    """
    Evaluate detection performance against ground truth change points.

    A detection is a true positive if it's within `tolerance` samples
    of a ground truth change point.

    Parameters
    ----------
    ground_truth : List[int]
        True change point indices
    detections : List[int]
        Detected change point indices
    tolerance : int
        Maximum distance for a detection to match ground truth

    Returns
    -------
    EvaluationResult
        Evaluation metrics
    """
    ground_truth = sorted(ground_truth)
    detections = sorted(detections)

    n_gt = len(ground_truth)
    n_det = len(detections)

    if n_gt == 0 and n_det == 0:
        return EvaluationResult(
            accuracy=1.0, precision=1.0, recall=1.0, f1_score=1.0,
            true_positives=0, false_positives=0, false_negatives=0,
            matched_pairs=[], mean_timing_error=None
        )

    if n_gt == 0:
        return EvaluationResult(
            accuracy=0.0, precision=0.0, recall=1.0, f1_score=0.0,
            true_positives=0, false_positives=n_det, false_negatives=0,
            matched_pairs=[], mean_timing_error=None
        )

    if n_det == 0:
        return EvaluationResult(
            accuracy=0.0, precision=1.0, recall=0.0, f1_score=0.0,
            true_positives=0, false_positives=0, false_negatives=n_gt,
            matched_pairs=[], mean_timing_error=None
        )

    # Greedy matching: match each detection to nearest unmatched ground truth
    gt_matched = [False] * n_gt
    det_matched = [False] * n_det
    matched_pairs = []
    timing_errors = []

    # Sort by distance for greedy matching
    distances = []
    for i, gt in enumerate(ground_truth):
        for j, det in enumerate(detections):
            dist = abs(gt - det)
            if dist <= tolerance:
                distances.append((dist, i, j, gt, det))

    distances.sort(key=lambda x: x[0])

    for dist, i, j, gt, det in distances:
        if not gt_matched[i] and not det_matched[j]:
            gt_matched[i] = True
            det_matched[j] = True
            matched_pairs.append((gt, det))
            timing_errors.append(dist)

    true_positives = len(matched_pairs)
    false_positives = n_det - true_positives
    false_negatives = n_gt - true_positives

    precision = true_positives / n_det if n_det > 0 else 0.0
    recall = true_positives / n_gt if n_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy: correct detections / (total ground truth + false positives)
    accuracy = true_positives / (n_gt + false_positives) if (n_gt + false_positives) > 0 else 0.0

    mean_timing_error = np.mean(timing_errors) if timing_errors else None

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        matched_pairs=matched_pairs,
        mean_timing_error=mean_timing_error,
    )


def normalized_shd(adj_true: np.ndarray, adj_pred: np.ndarray) -> float:
    """
    Normalized Structural Hamming Distance for DAG comparison.

    SHD counts edge additions, deletions, and reversals.
    Normalized by max possible edges.

    Parameters
    ----------
    adj_true : np.ndarray
        True adjacency matrix [N, N]
    adj_pred : np.ndarray
        Predicted adjacency matrix [N, N]

    Returns
    -------
    float
        Normalized SHD (0 = identical, 1 = completely different)
    """
    n = adj_true.shape[0]
    max_edges = n * (n - 1)  # Max directed edges

    if max_edges == 0:
        return 0.0

    # Binarize
    true_bin = (adj_true != 0).astype(int)
    pred_bin = (adj_pred != 0).astype(int)

    # Count different edge types
    diff = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Edge in true but not in pred (deletion)
            if true_bin[i, j] == 1 and pred_bin[i, j] == 0:
                diff += 1
            # Edge in pred but not in true (addition)
            elif true_bin[i, j] == 0 and pred_bin[i, j] == 1:
                diff += 1

    return diff / max_edges


def normalized_shd_accuracy(adj_true: np.ndarray, adj_pred: np.ndarray) -> float:
    """
    Structural accuracy as 1 - normalized_shd.

    Returns
    -------
    float
        Accuracy (1 = identical, 0 = completely different)
    """
    return 1.0 - normalized_shd(adj_true, adj_pred)


def simplify_transitions(
    change_points: List[int],
    min_distance: int,
) -> List[int]:
    """
    Merge nearby change points.

    Parameters
    ----------
    change_points : List[int]
        Detected change points
    min_distance : int
        Minimum separation between distinct change points

    Returns
    -------
    List[int]
        Simplified change points
    """
    if not change_points:
        return []

    sorted_cps = sorted(change_points)
    simplified = [sorted_cps[0]]

    for cp in sorted_cps[1:]:
        if cp - simplified[-1] >= min_distance:
            simplified.append(cp)
        else:
            # Merge: take midpoint
            simplified[-1] = (simplified[-1] + cp) // 2

    return simplified


# ── Dynamic / non-stationary structure metrics ──────────────────────────────
#
# For non-stationary causal discovery the mean of per-regime nSHD hides three
# kinds of failure that matter:
#   1. Length bias  — small regimes count as much as long ones, but the long
#      ones carry most of the data evidence.
#   2. Coverage     — when a change point is missed, a true regime gets no
#      detected window at all; the unweighted mean silently skips it.
#   3. Transitions  — a pipeline can match each regime's structure well in
#      isolation but learn the wrong *deltas* between them; that failure mode
#      is invisible to per-regime SHD.
#
# We therefore expose four complementary metrics:
#   • windowed_nshd_weighted : length-weighted nSHD over true regimes; missed
#                              regimes counted at maximum penalty (=1.0).
#   • coverage_score         : fraction of true regimes covered ≥50% by some
#                              detected window.
#   • transition_shd         : normalised SHD between true and learned regime
#                              transitions (edge symmetric-differences).
#   • dynamic_score          : composite for headline reporting; lower=better.
#
# References for the framing: dynamic Bayesian network evaluation literature
# (Trabelsi et al. 2-TBN-SHD); non-stationary causal discovery practice
# (windowed SHD); change-point detection precision/recall conventions.


def _bin_adj(A) -> np.ndarray:
    """Return a binary square matrix with zero diagonal, accepting DataFrame or ndarray."""
    if hasattr(A, "values"):
        A = A.values
    A = np.asarray(A)
    B = (np.abs(A) > 0.05).astype(bool)
    np.fill_diagonal(B, False)
    return B


def _best_overlap_struct(structures, r_start, r_end):
    """Return (overlap, structure) for the detected window with max overlap on [r_start, r_end)."""
    best_overlap = 0
    best = None
    for s in structures:
        ov = max(0, min(s.window_end, r_end) - max(s.window_start, r_start))
        if ov > best_overlap:
            best_overlap = ov
            best = s
    return best_overlap, best


def windowed_nshd_weighted(structures, true_adjs, true_cps, T, p) -> float:
    """
    Length-weighted normalised SHD over true regimes.

    For each true regime k of length L_k, the structure with maximum
    overlap is the "aligned" prediction Ĝ_{a(k)}.  Missing alignment
    (no detected window overlaps regime k) is charged the maximum
    penalty of 1.0.

    score = (Σ_k L_k · nSHD_k) / (Σ_k L_k)
    """
    boundaries = [0] + list(true_cps) + [T]
    regime_spans = list(zip(boundaries[:-1], boundaries[1:]))
    if not regime_spans or p < 2:
        return 0.0

    max_edges = p * (p - 1)
    total_w_shd = 0.0
    total_w = 0.0

    for k, (rs, re) in enumerate(regime_spans):
        L_k = re - rs
        true_bin = _bin_adj(true_adjs[k]) if k < len(true_adjs) else None
        _, best = _best_overlap_struct(structures, rs, re)
        if best is None or true_bin is None:
            nshd_k = 1.0
        else:
            pred_bin = _bin_adj(best.adjacency_matrix)
            diff = np.sum(true_bin != pred_bin)
            nshd_k = diff / max_edges if max_edges else 0.0
        total_w_shd += L_k * nshd_k
        total_w += L_k

    return float(total_w_shd / total_w) if total_w > 0 else 1.0


def coverage_score(structures, true_cps, T, min_fraction: float = 0.5) -> float:
    """Fraction of true regimes covered ≥ `min_fraction` by some detected window."""
    boundaries = [0] + list(true_cps) + [T]
    regime_spans = list(zip(boundaries[:-1], boundaries[1:]))
    if not regime_spans:
        return 0.0
    covered = 0
    for rs, re in regime_spans:
        L = re - rs
        if L <= 0:
            continue
        best_ov, _ = _best_overlap_struct(structures, rs, re)
        if best_ov / L >= min_fraction:
            covered += 1
    return float(covered / len(regime_spans))


def transition_shd(structures, true_adjs, true_cps, T, p) -> float:
    """
    Normalised SHD between true and learned regime *transitions*.

    For each adjacent true pair (G_k, G_{k+1}) we compute the symmetric
    difference δ_k = G_k ⊕ G_{k+1} (edges that appeared or disappeared).
    Same for the aligned predictions.  The metric is the mean over all
    transitions of |δ_k_true ⊕ δ_k_pred| / max_edges.

    Captures whether the pipeline learned the right *changes*, not just
    the right average structures — the failure mode invisible to
    per-regime SHD.
    """
    boundaries = [0] + list(true_cps) + [T]
    regime_spans = list(zip(boundaries[:-1], boundaries[1:]))
    n_reg = len(regime_spans)
    if n_reg < 2 or p < 2:
        return 0.0

    # Align each true regime to the best-overlap structure
    aligned = []
    for rs, re in regime_spans:
        _, best = _best_overlap_struct(structures, rs, re)
        aligned.append(best)

    max_edges = p * (p - 1)
    total = 0.0
    n_trans = 0

    for k in range(n_reg - 1):
        if k >= len(true_adjs) or k + 1 >= len(true_adjs):
            continue
        true_curr = _bin_adj(true_adjs[k])
        true_next = _bin_adj(true_adjs[k + 1])
        true_delta = true_curr ^ true_next

        if aligned[k] is None or aligned[k + 1] is None:
            pred_delta = np.zeros_like(true_delta)
        else:
            pred_curr = _bin_adj(aligned[k].adjacency_matrix)
            pred_next = _bin_adj(aligned[k + 1].adjacency_matrix)
            pred_delta = pred_curr ^ pred_next

        diff = int(np.sum(true_delta != pred_delta))
        total += diff / max_edges
        n_trans += 1

    return float(total / n_trans) if n_trans else 0.0


def dynamic_score(
    windowed_nshd: float,
    detection_f1: float,
    coverage: float,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> float:
    """
    Composite score blending structural error, detection F1, and coverage.

    score = α · windowed_nshd + β · (1 - detection_f1) + γ · (1 - coverage)

    All inputs are in [0, 1]; output in [0, 1] with lower = better.
    Default weights put structure first (0.4), changepoint F1 and coverage
    equal (0.3 each), reflecting that without coverage the structural
    score loses its meaning.
    """
    s = alpha * windowed_nshd + beta * (1.0 - detection_f1) + gamma * (1.0 - coverage)
    return float(np.clip(s, 0.0, 1.0))


def compute_detection_summary(
    results: List[EvaluationResult],
) -> Dict[str, float]:
    """
    Compute summary statistics across multiple runs.

    Parameters
    ----------
    results : List[EvaluationResult]
        Results from multiple experiments

    Returns
    -------
    Dict[str, float]
        Mean and std of each metric
    """
    if not results:
        return {}

    metrics = {
        "accuracy": [r.accuracy for r in results],
        "precision": [r.precision for r in results],
        "recall": [r.recall for r in results],
        "f1_score": [r.f1_score for r in results],
    }

    summary = {}
    for name, values in metrics.items():
        summary[f"{name}_mean"] = np.mean(values)
        summary[f"{name}_std"] = np.std(values)

    timing_errors = [r.mean_timing_error for r in results if r.mean_timing_error is not None]
    if timing_errors:
        summary["timing_error_mean"] = np.mean(timing_errors)
        summary["timing_error_std"] = np.std(timing_errors)

    return summary
