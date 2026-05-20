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
