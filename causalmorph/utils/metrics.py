import numpy as np
import pandas as pd
import numba
from sklearn.metrics import matthews_corrcoef


def apply_shd(matrix1, matrix2):
    """
    Compute the Structural Hamming Distance (SHD) and return a normalized percentage.

    - Ensures matrices are numeric.
    - Matches shapes by truncation.
    - Converts to binary adjacency matrices.
    - Computes SHD and normalizes it to a similarity percentage.

    :param matrix1: 2D array-like (adjacency matrix of graph 1)
    :param matrix2: 2D array-like (adjacency matrix of graph 2)
    :return: Normalized similarity percentage (0 to 100)
    """

    # Ensure matrices are NumPy arrays and convert to float (handle string cases)
    matrix1 = np.array(matrix1, dtype=float)
    matrix2 = np.array(matrix2, dtype=float)

    # Ensure both matrices have the same shape by truncating to the smallest common shape
    min_rows = min(matrix1.shape[0], matrix2.shape[0])
    min_cols = min(matrix1.shape[1], matrix2.shape[1])
    matrix1 = matrix1[:min_rows, :min_cols]
    matrix2 = matrix2[:min_rows, :min_cols]

    # Convert to binary adjacency matrices (1 = edge exists, 0 = no edge)
    matrix1 = (matrix1 > 0).astype(int)
    matrix2 = (matrix2 > 0).astype(int)

    # Compute SHD (count differing elements)
    shd = np.sum(matrix1 != matrix2)

    # Normalize SHD to a percentage similarity
    total_possible_changes = matrix1.size
    similarity_percentage = 100 * (1 - (shd / total_possible_changes))

    return max(0, similarity_percentage)


def compute_shd_similarity(matrix_pred, matrix_truth):
    """
    Compute SHD and percentage of correctly identified edges compared to ground truth.

    :param matrix_pred: 2D array-like (predicted adjacency matrix)
    :param matrix_truth: 2D array-like (ground truth adjacency matrix)
    :return: Tuple (correct_edges_ratio, similarity_percentage)
    """
    from cdt.metrics import SHD

    matrix_pred = np.asarray(matrix_pred)
    matrix_truth = np.asarray(matrix_truth)

    if matrix_pred.shape != matrix_truth.shape:
        raise ValueError(f"Shape mismatch: {matrix_pred.shape} vs {matrix_truth.shape}")

    if np.isnan(matrix_pred).any() or np.isnan(matrix_truth).any():
        raise ValueError("NaNs found in input matrices.")
    if np.isinf(matrix_pred).any() or np.isinf(matrix_truth).any():
        raise ValueError("Infs found in input matrices.")

    pred_binary = (matrix_pred > 0).astype(int)
    truth_binary = (matrix_truth > 0).astype(int)

    # Compute SHD using cdt.metrics
    differences = SHD(truth_binary, pred_binary, double_for_anticausal=True)

    correct_edges = np.sum((pred_binary == 1) & (truth_binary == 1))
    total_truth_edges = np.sum(truth_binary)
    correct_edges_ratio = f"{correct_edges}/{total_truth_edges}"

    if total_truth_edges > 0:
        similarity_percentage = 100 * (correct_edges / total_truth_edges)
    else:
        similarity_percentage = 100 if np.sum(pred_binary) == 0 else 0

    return correct_edges_ratio, similarity_percentage


def normalized_shd_score(matrix_truth, matrix_pred):
    """
    Compute a normalized SHD score from 0 to 100 where 100 means identical matrices.

    Parameters:
        - matrix_truth: 2D array-like (ground truth adjacency matrix)
        - matrix_pred: 2D array-like (predicted adjacency matrix)

    Returns:
        - score: float between 0 and 100, where 100 means identical matrices
    """
    matrix_truth = np.asarray(matrix_truth)
    matrix_pred = np.asarray(matrix_pred)

    if matrix_pred.shape != matrix_truth.shape:
        raise ValueError(f"Shape mismatch: {matrix_pred.shape} vs {matrix_truth.shape}")

    if np.isnan(matrix_pred).any() or np.isnan(matrix_truth).any():
        raise ValueError("NaNs found in input matrices.")
    if np.isinf(matrix_pred).any() or np.isinf(matrix_truth).any():
        raise ValueError("Infs found in input matrices.")

    # Convert to binary matrices
    truth_binary = (matrix_truth > 0).astype(int)
    pred_binary = (matrix_pred > 0).astype(int)

    # Calculate SHD (number of differing elements)
    shd = np.sum(truth_binary != pred_binary)

    # Total possible differences is the size of the matrix
    total_possible_differences = matrix_truth.size

    # Normalize to a 0-100 scale where 100 means identical
    normalized_score = 100 * (1 - (shd / total_possible_differences))

    return normalized_score


def normalized_shd(
    ground_truth: pd.DataFrame, estimated: pd.DataFrame, threshold: float = 0.05
) -> dict:
    """
    Compute SHD and normalized SHD between binary ground truth and weighted estimated adjacency matrices.

    Parameters:
    - ground_truth: pd.DataFrame or np.ndarray (binary)
    - estimated: pd.DataFrame or np.ndarray (weighted, incl. negatives)
    - threshold: float, threshold on absolute weight to binarize

    Returns:
    - dict with:
        - shd: int, number of differing edges
        - normalized_shd: float in [0,1]
    """
    if isinstance(ground_truth, pd.DataFrame):
        ground_truth = ground_truth.values
    if isinstance(estimated, pd.DataFrame):
        estimated = estimated.values

    assert ground_truth.shape == estimated.shape, "Matrix shapes must match"
    d = ground_truth.shape[0]

    # Remove diagonals
    gt_bin = ground_truth.copy()
    est_bin = (np.abs(estimated.copy()) > threshold).astype(int)
    np.fill_diagonal(gt_bin, 0)
    np.fill_diagonal(est_bin, 0)

    shd = np.sum(gt_bin != est_bin)
    normalized = shd / (d * (d - 1))

    return {"shd": int(shd), "normalized_shd": round(normalized, 3)}


@numba.njit(parallel=True, cache=True)
def _fast_graph_metrics(amEst, amTrue):
    """Optimized core graph metrics computation with Numba"""
    n = amTrue.shape[0]
    TP = 0
    FP = 0
    FN = 0

    for i in numba.prange(n):
        for j in range(n):
            if amEst[i, j] > 0 and amTrue[i, j] > 0:
                TP += 1
            elif amEst[i, j] > 0 and amTrue[i, j] == 0:
                FP += 1
            elif amEst[i, j] == 0 and amTrue[i, j] > 0:
                FN += 1

    return TP, FP, FN


def mycomparegraphs(amEst: np.ndarray, amTrue: np.ndarray) -> dict:
    """
    Compare two adjacency matrices and compute comprehensive graph metrics.

    Parameters:
        amEst: Estimated adjacency matrix
        amTrue: True adjacency matrix

    Returns:
        Dictionary with metrics (SHD, F1, Precision, Recall, MCC, etc.)
    """
    amEst = amEst.copy().astype(np.int32)
    amTrue = amTrue.astype(np.int32)
    n = amTrue.shape[0]

    # Flatten lower triangle (no diagonals) for metrics that require 1D arrays
    tril_idx = np.tril_indices(n, k=-1)
    y_true_flat = amTrue[tril_idx]
    y_pred_flat = amEst[tril_idx]

    # Try MCC (Matthews correlation coefficient)
    try:
        mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
    except Exception:
        mcc = np.nan

    # --- Standard graph metrics ---
    if np.any(amEst == 3):
        mask = (amEst + amEst.T) < 5
        amEst[mask] = 0
        amEst = (amEst != 0).astype(np.int32)
    else:
        mask = (amEst + amEst.T) == 2
        amEst[mask] = 0

    # Use optimized Numba function for core metrics
    TP, FP, FN = _fast_graph_metrics(amEst, amTrue)
    FE = int(np.sum(amTrue))

    amEst_sym = amEst + amEst.T
    amTrue_sym = amTrue + amTrue.T
    np.fill_diagonal(amEst_sym, 1)
    np.fill_diagonal(amTrue_sym, 1)
    amEst_sym = (amEst_sym != 0).astype(np.int32)
    amTrue_sym = (amTrue_sym != 0).astype(np.int32)
    TN = int(np.sum((~amEst_sym.astype(bool)) & (~amTrue_sym.astype(bool)))) // 2

    # Core rates - vectorized operations
    TPR = np.divide(TP, (TP + FN), out=np.zeros(1), where=(TP + FN) > 0)[0]  # Recall
    FPR = np.divide(FP, (FP + TN), out=np.zeros(1), where=(FP + TN) > 0)[0]
    TDR = np.divide(TP, FE, out=np.zeros(1), where=FE > 0)[0]
    precision = np.divide(TP, (TP + FP), out=np.zeros(1), where=(TP + FP) > 0)[0]
    recall = TPR
    specificity = np.divide(TN, (TN + FP), out=np.zeros(1), where=(TN + FP) > 0)[0]

    reversed_edges = int(np.sum((amEst == 1) & (amTrue.T == 1))) - TP

    SHD = FP + FN + reversed_edges
    total_possible_edges = amTrue.shape[0] * (amTrue.shape[0] - 1)
    SHD_norm = np.divide(
        SHD, total_possible_edges, out=np.zeros(1), where=total_possible_edges > 0
    )[0]

    F1 = np.divide(
        2 * TP, (2 * TP + FP + FN), out=np.zeros(1), where=(2 * TP + FP + FN) > 0
    )[0]
    ACC = np.divide(
        (TP + TN), (TP + TN + FP + FN), out=np.zeros(1), where=(TP + TN + FP + FN) > 0
    )[0]

    # Mean degree per node
    degree_true = np.sum(amTrue, axis=1)
    degree_est = np.sum(amEst, axis=1)
    mean_degree_true = np.mean(degree_true)
    mean_degree_est = np.mean(degree_est)
    mae_degree = np.mean(np.abs(degree_true - degree_est))

    return {
        "FE": FE,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TPR": TPR,
        "FPR": FPR,
        "TDR": TDR,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": F1,
        "ACC": ACC,
        "MCC": mcc,
        "SHD": SHD,
        "reversed_edges": reversed_edges,
        "normalized_shd": SHD_norm,
        "mean_degree_true": mean_degree_true,
        "mean_degree_est": mean_degree_est,
        "mae_degree": mae_degree,
    }
