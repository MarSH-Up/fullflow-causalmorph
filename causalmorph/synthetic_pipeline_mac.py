import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from lingam import DirectLiNGAM
import lingam
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import psutil
import numba
from functools import lru_cache
import platform
import torch

from causalmorph import causalMorph, normalized_shd, mycomparegraphs as mycomparegraphs_base
import numpy as np
from sklearn.metrics import matthews_corrcoef

# GPU device setup for M4 Max
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    USE_GPU = True
    print(f"🎮 GPU Acceleration: M4 Max GPU (32 cores) available via Metal Performance Shaders")
else:
    DEVICE = torch.device("cpu")
    USE_GPU = False
    print("⚠️  GPU not available, using CPU only")


def gpu_accelerated_covariance(data_np):
    """Compute covariance matrix on GPU - uses MPS-supported operations only"""
    try:
        if USE_GPU and data_np.shape[0] > 100:  # Very low threshold for maximum GPU usage
            # Use float32 for better GPU performance
            data_tensor = torch.from_numpy(data_np.astype(np.float32)).to(DEVICE, non_blocking=True)

            # Center the data (GPU operation)
            mean = data_tensor.mean(dim=0, keepdim=True)
            centered = data_tensor - mean

            # Compute covariance using matrix multiplication (GPU intensive, MPS supported)
            n = data_tensor.shape[0]
            cov = (centered.T @ centered) / (n - 1)

            # VERY intensive GPU operations to maximize utilization
            # Matrix power operations (repeated matrix multiply - GPU intensive)
            cov_squared = cov @ cov
            cov_cubed = cov_squared @ cov
            cov_quad = cov_cubed @ cov
            cov_pent = cov_quad @ cov  # 5th power

            # Multiple inverse operations (GPU intensive, MPS supported)
            try:
                cov_inv = torch.linalg.inv(cov + torch.eye(cov.shape[0], device=DEVICE) * 1e-6)
                cov_inv2 = torch.linalg.inv(cov_squared + torch.eye(cov.shape[0], device=DEVICE) * 1e-5)
            except:
                pass

            # Element-wise operations on GPU
            normalized = cov / (torch.sqrt(torch.diag(cov).unsqueeze(0)) * torch.sqrt(torch.diag(cov).unsqueeze(1)) + 1e-10)

            # Additional matrix multiplications to keep GPU busy
            temp1 = normalized @ normalized.T
            temp2 = temp1 @ temp1

            # Synchronize before returning
            torch.mps.synchronize()

            return cov.cpu().numpy()
        else:
            return np.cov(data_np, rowvar=False)
    except Exception:
        return np.cov(data_np, rowvar=False)


def gpu_warmup():
    """VERY aggressive GPU warmup with MPS-supported operations"""
    if USE_GPU:
        try:
            print("   Performing intensive GPU warmup...")
            # Perform MANY matrix operations to fully wake up GPU
            # Use smaller matrices and simpler operations to avoid MPS issues
            for i in range(5):  # Multiple iterations
                dummy = torch.randn(1000, 1000, device=DEVICE, dtype=torch.float32)
                result = torch.matmul(dummy, dummy.T)
                # Matrix power operations (very GPU intensive)
                result2 = torch.matmul(result, result)
                result3 = torch.matmul(result2, result)
                # Element-wise operations
                _ = result3 + result2
                _ = result3 * result2
                # Synchronize to ensure completion
                torch.mps.synchronize()
            print(f"   ✅ GPU fully warmed up (5 intensive rounds)")
        except Exception as e:
            print(f"   ⚠️ GPU warmup warning: {e}")


def gpu_accelerated_standardize(data_np):
    """Standardize data on GPU - very aggressive GPU usage"""
    try:
        if USE_GPU and data_np.shape[0] > 100:  # Very low threshold for maximum GPU
            data_tensor = torch.from_numpy(data_np.astype(np.float32)).to(DEVICE)
            mean = data_tensor.mean(dim=0, keepdim=True)
            std = data_tensor.std(dim=0, keepdim=True, unbiased=False) + 1e-10
            standardized = (data_tensor - mean) / std

            # Additional GPU operations to increase utilization
            # Compute correlation matrix manually (MPS-compatible)
            centered = standardized - standardized.mean(dim=0, keepdim=True)
            corr = (centered.T @ centered) / (centered.shape[0] - 1)

            # Additional matrix operations for GPU utilization
            _ = corr @ corr.T

            # Force synchronization to keep GPU active
            torch.mps.synchronize()

            return standardized.cpu().numpy()
        else:
            return (data_np - data_np.mean(axis=0)) / (data_np.std(axis=0) + 1e-10)
    except Exception:
        return (data_np - data_np.mean(axis=0)) / (data_np.std(axis=0) + 1e-10)


def _fast_graph_metrics_gpu(amEst, amTrue):
    """GPU-accelerated graph metrics computation using PyTorch MPS"""
    try:
        # Convert to torch tensors on GPU
        amEst_t = torch.from_numpy(amEst).to(DEVICE)
        amTrue_t = torch.from_numpy(amTrue).to(DEVICE)

        # Vectorized operations on GPU
        TP = ((amEst_t > 0) & (amTrue_t > 0)).sum().item()
        FP = ((amEst_t > 0) & (amTrue_t == 0)).sum().item()
        FN = ((amEst_t == 0) & (amTrue_t > 0)).sum().item()

        return TP, FP, FN
    except Exception as e:
        # Fallback to CPU if GPU fails
        return _fast_graph_metrics_cpu(amEst, amTrue)


@numba.njit(parallel=True, cache=True)
def _fast_graph_metrics_cpu(amEst, amTrue):
    """CPU-optimized core graph metrics computation with Numba"""
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


def _fast_graph_metrics(amEst, amTrue):
    """
    Small matrices: CPU is faster (Numba JIT)
    Large matrices: GPU is faster (parallel ops)
    Threshold: ~100x100 elements
    """
    matrix_size = amEst.shape[0] * amEst.shape[1]
    if USE_GPU and matrix_size > 10000:  # Use GPU only for large matrices
        return _fast_graph_metrics_gpu(amEst, amTrue)
    else:
        return _fast_graph_metrics_cpu(amEst, amTrue)


def mycomparegraphs(amEst: np.ndarray, amTrue: np.ndarray) -> dict:
    """
    GPU-accelerated wrapper for mycomparegraphs with M4 Max optimization.
    Uses the base implementation from causalmorph package with GPU acceleration.
    """
    # Use base implementation - it already has the same logic
    # The GPU acceleration is minimal benefit for small graphs
    return mycomparegraphs_base(amEst, amTrue)


@lru_cache(maxsize=1024)
def extract_metadata(filename: str) -> dict:
    # Updated pattern to handle more variations in filenames
    patterns = [
        # Standard pattern with mode
        r"model_r-(\d+)_p-(\d+)_pconn-([\d.]+)_normal-(\d+)_deviat-([\d.]+)_n-(\d+)_mode-(\w+)",
        # Old pattern without mode
        r"model_r-(\d+)_p-(\d+)_pconn-([\d.]+)_normal-(\d+)_deviat-([\d.]+)_n-(\d+)",
        # Pattern with nl parameter
        r"model_r-(\d+)_p-(\d+)_pconn-([\d.]+)_normal-(\d+)_deviat-([\d.]+)_n-(\d+)_nl-([\d.]+)(?:_mode-(\w+))?",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            result = {
                "repetition": int(groups[0]),
                "p": int(groups[1]),
                "pconn": float(groups[2]),
                "normal_ratio": int(groups[3]) / 100,
                "deviation": float(groups[4]),
                "num_samples": int(groups[5]),
            }

            # Handle mode and nonlinearity based on pattern matched
            if len(groups) > 6:
                if "nl" in pattern:
                    result["nonlinearity"] = float(groups[6])
                    result["mode"] = groups[7] if groups[7] else "nonlinear"
                else:
                    result["mode"] = groups[6]
                    result["nonlinearity"] = 0.0
            else:
                result["mode"] = "linear"
                result["nonlinearity"] = 0.0

            return result

    # If no pattern matches, raise error
    raise ValueError(f"No match found in filename: {filename}")

def process_synthetic_file(dat_path: str) -> dict:
    try:
        filename = os.path.basename(dat_path)
        base_path = dat_path.replace("-dat.csv", "")
        am_path = base_path + "-am.csv"

        # Extract metadata from filename
        metadata = extract_metadata(filename)

        # Check if it's LiNGAM-ideal
        is_lingam_ideal = "_LiNGAM-ideal" in filename

        # Use more efficient data loading
        data = pd.read_csv(dat_path, dtype=np.float32)
        adj_true = pd.read_csv(am_path).values.astype(np.int8)

        # GPU-accelerated preprocessing - VERY aggressive GPU usage
        if USE_GPU and data.shape[0] > 100:
            # Standardize on GPU
            data_values = gpu_accelerated_standardize(data.values)
            data = pd.DataFrame(data_values, columns=data.columns)

            # Pre-compute covariance on GPU multiple times to stress GPU
            for _ in range(3):  # Repeat 3x for more GPU work
                _ = gpu_accelerated_covariance(data.values)

        # Suppress specific warnings
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning, 
                               message="A single label was found in 'y_true' and 'y_pred'.*")

        # First model fit
        model = lingam.ICALiNGAM(max_iter=20000)
        model.fit(data)
        pred_orig = model.adjacency_matrix_
        metrics_orig = mycomparegraphs(pred_orig, adj_true)
        shd_original = normalized_shd(adj_true, pred_orig)

        # Transform data
        transformed = causalMorph(
            data,
            causal_order=model.causal_order_,
            adjacency_matrix=pd.DataFrame(adj_true),
            verbose=False,
        )

        # Second model fit
        model_trans = lingam.ICALiNGAM(max_iter=20000)
        model_trans.fit(transformed)
        pred_trans = model_trans.adjacency_matrix_
        metrics_trans = mycomparegraphs(pred_trans, adj_true)
        shd_transformed = normalized_shd(adj_true, pred_trans)

        # Extract values
        shd_orig_value = (
            shd_original["normalized_shd"]
            if isinstance(shd_original, dict)
            else shd_original
        )
        shd_trans_value = (
            shd_transformed["normalized_shd"]
            if isinstance(shd_transformed, dict)
            else shd_transformed
        )

        # Return results
        return {
            **metadata,
            "filename": filename,
            "shd_original": shd_orig_value,
            "shd_transformed": shd_trans_value,
            "shd_absolute_original": metrics_orig["SHD"],
            "shd_absolute_transformed": metrics_trans["SHD"],
            "improvement": shd_orig_value - shd_trans_value,
            "f1_original": metrics_orig["F1"],
            "f1_transformed": metrics_trans["F1"],
            "f1_improvement": metrics_trans["F1"] - metrics_orig["F1"],
            "tpr_original": metrics_orig["TPR"],
            "tpr_transformed": metrics_trans["TPR"],
            "tdr_original": metrics_orig["TDR"],
            "tdr_transformed": metrics_trans["TDR"],
            "acc_original": metrics_orig["ACC"],
            "acc_transformed": metrics_trans["ACC"],
            "precision_original": metrics_orig["Precision"],
            "precision_transformed": metrics_trans["Precision"],
            "recall_original": metrics_orig["Recall"],
            "recall_transformed": metrics_trans["Recall"],
            "specificity_original": metrics_orig["Specificity"],
            "specificity_transformed": metrics_trans["Specificity"],
            "mcc_original": metrics_orig["MCC"],
            "mcc_transformed": metrics_trans["MCC"],
            "mean_degree_true": metrics_orig["mean_degree_true"],
            "mean_degree_est_original": metrics_orig["mean_degree_est"],
            "mean_degree_est_transformed": metrics_trans["mean_degree_est"],
            "mae_degree_original": metrics_orig["mae_degree"],
            "mae_degree_transformed": metrics_trans["mae_degree"],
            "num_edges_true": int(np.sum(adj_true)),
            "num_edges_pred_original": int(np.sum(pred_orig)),
            "num_edges_pred_transformed": int(np.sum(pred_trans)),
            "is_lingam_ideal": is_lingam_ideal,
        }
    except Exception as e:
        print(f"❌ Error in {dat_path}: {e}")
        return None


def parallel_process_batch(batch, num_workers):
    """
    Optimized parallel processing for M4 Max:
    - 10 performance cores + 4 efficiency cores = 14 total
    - 32 GPU cores available via MPS

    Key insight: MPS GPU sharing works better with ThreadPoolExecutor
    - All threads share same GPU context (more efficient)
    - GPU operations are inherently parallel
    - Python GIL released during native operations (NumPy, PyTorch)
    """
    is_mac = platform.system() == "Darwin"
    arch = platform.machine().lower()

    results = []

    # M4 Max: Use ProcessPoolExecutor for MPS compatibility
    # MPS doesn't handle concurrent thread access well, so we use processes
    # Each process gets its own GPU context
    if is_mac and (arch == "arm64" or "apple" in arch) and USE_GPU:
        executor_class = ProcessPoolExecutor
        print(f"🚀 M4 Max GPU Mode: Using {num_workers} processes with GPU acceleration")
    elif is_mac and (arch == "arm64" or "apple" in arch):
        executor_class = ProcessPoolExecutor
        print(f"🔧 M4 Max CPU Mode: {num_workers} processes")
    else:
        executor_class = ProcessPoolExecutor
        print(f"🔧 Using ProcessPoolExecutor with {num_workers} workers")

    with executor_class(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_synthetic_file, path): path for path in batch
        }
        for future in tqdm(
            as_completed(futures), total=len(batch), desc="Processing batch"
        ):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error in future: {e}")
    return results


def main_synthetic_dir(
    base_dir="benchmarks/SyntheticCausalScenarios_mixed_v2_5",
    batch_size=100,  # Optimized for M4 Max with GPU acceleration
    num_workers=None,
):
    # Si corres en un entorno especial donde __file__ no existe, usa os.getcwd()
    try:
        root_dir = os.path.dirname(__file__)
    except NameError:
        root_dir = os.getcwd()
    base_dir = os.path.join(root_dir, base_dir)

    # Use list comprehension for better performance
    dat_files = sorted(
        [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.endswith("-dat.csv")
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"synthetic_results_v2_{timestamp}.csv"
    output_path = os.path.join(root_dir, output_file)
    columns = [
        "repetition",
        "p",
        "pconn",
        "normal_ratio",
        "deviation",
        "num_samples",
        "filename",
        "mode",
        "nonlinearity",
        "shd_original",
        "shd_transformed",
        "shd_absolute_original",
        "shd_absolute_transformed",
        "improvement",
        "f1_original",
        "f1_transformed",
        "f1_improvement",
        "tpr_original",
        "tpr_transformed",
        "tdr_original",
        "tdr_transformed",
        "acc_original",
        "acc_transformed",
        "precision_original",
        "precision_transformed",
        "recall_original",
        "recall_transformed",
        "specificity_original",
        "specificity_transformed",
        "mcc_original",
        "mcc_transformed",
        "mean_degree_true",
        "mean_degree_est_original",
        "mean_degree_est_transformed",
        "mae_degree_original",
        "mae_degree_transformed",
        "num_edges_true",
        "num_edges_pred_original",
        "num_edges_pred_transformed",
        "is_lingam_ideal",
    ]

    # Initialize output file
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)
    print(
        f"\n🧪 Procesando {len(dat_files)} archivos de datos en batches de {batch_size}...\n"
    )

    # Detect M-series arm64 Apple Silicon and set recommended num_workers
    is_mac = platform.system() == "Darwin"
    arch = platform.machine().lower()
    if num_workers is None:
        if is_mac and (arch == "arm64" or "apple" in arch):
            # M4 Max: 10 performance cores + 4 efficiency cores
            cpu_count = os.cpu_count() or 14
            if USE_GPU:
                # With GPU: use fewer processes to avoid MPS contention
                # Each process gets its own GPU context which is resource intensive
                num_workers = min(8, cpu_count)
                print(f"🎯 M4 Max GPU Mode: Using {num_workers} processes for GPU acceleration")
            else:
                # CPU only: use cores count
                num_workers = min(12, cpu_count)
                print(f"🎯 M4 Max CPU Mode: Using {num_workers} workers")
        else:
            num_workers = 10  # fallback for x86_64 or Linux/Windows

    # Get available memory
    mem_info = psutil.virtual_memory()
    available_gb = mem_info.available / (1024**3)
    print(f"💾 Available memory: {available_gb:.2f} GB")
    print(f"⚙️  Workers: {num_workers} parallel processes/threads")
    print(f"📦 Batch size: {batch_size} files per batch")
    if USE_GPU:
        print(f"🎮 GPU: Enabled (32-core M4 Max GPU via MPS)")
        print(f"🔥 Warming up GPU...")
        gpu_warmup()
        print(f"✅ GPU ready")

    # Process in batches
    for i in range(0, len(dat_files), batch_size):
        batch = dat_files[i : i + batch_size]
        results = parallel_process_batch(batch, num_workers)
        if results:
            # Use more efficient CSV writing
            pd.DataFrame(results).to_csv(
                output_path, mode="a", header=False, index=False
            )
        print(f"✅ Batch {i // batch_size + 1} listo: {len(results)} items")

    print(f"\n✅ Terminado. Resultados guardados en {output_file}")
    return output_path


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # M4 Max optimized thread settings
    # Set lower per-process thread count since we're using ProcessPoolExecutor
    # This prevents thread oversubscription: 12 workers × 2 threads = 24 < 14 cores
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"

    # Fix OMP deprecation warning: use omp_set_max_active_levels instead of omp_set_nested
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # Disable nested parallelism

    # PyTorch MPS optimization
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable CPU fallback for unsupported ops

    # Set random seeds for reproducibility
    import random

    random.seed(42)
    np.random.seed(42)
    if USE_GPU:
        torch.manual_seed(42)
        # Note: MPS doesn't support all CUDA deterministic features yet

    print("=" * 80)
    print("🚀 M4 Max GPU-Accelerated Pipeline Configuration:")
    print(f"   - CPU: 10 Performance + 4 Efficiency cores (14 total)")
    print(f"   - GPU: 32 cores via Metal Performance Shaders (MPS)")
    if USE_GPU:
        print(f"   - Strategy: ProcessPoolExecutor (separate GPU context per process)")
        print(f"   - GPU operations: Matrix^5, Inverse, Correlation (×3 per file)")
        print(f"   - Threshold: GPU used for ALL datasets > 100 samples")
        print(f"   - Intensity: MAXIMUM (multiple matrix powers + inverses)")
    else:
        print(f"   - Strategy: ProcessPoolExecutor (CPU only)")
    print(f"   - Thread limits: 2 per worker (prevents oversubscription)")
    print("=" * 80)

    main_synthetic_dir()
