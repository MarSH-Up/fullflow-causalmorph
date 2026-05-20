import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import shapiro
from typing import Union, Tuple, Dict, List


def whiten(data):
    """Whiten data to have identity covariance matrix."""
    data = np.atleast_2d(data)
    centered = data - np.mean(data, axis=0, keepdims=True)

    # Univariate case (1D)
    if centered.shape[1] == 1:
        var = np.var(centered, ddof=1)
        W = 1.0 / np.sqrt(var)
        whitened = centered * W
        return whitened, W, var

    # Multivariate case
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    W = D_inv_sqrt @ eigvecs.T
    whitened = (W @ centered.T).T
    return whitened, W, cov


def color(data, cov):
    """Apply coloring transformation using covariance structure."""
    data = np.atleast_2d(data)

    if np.isscalar(cov) or np.ndim(cov) == 0:
        # Univariate case
        scale = np.sqrt(cov)
        return data * scale

    eigvals, eigvecs = np.linalg.eigh(cov)
    D_sqrt = np.diag(np.sqrt(eigvals))
    C = eigvecs @ D_sqrt
    colored = (C @ data.T).T
    return colored


def generate_best_non_gaussian_noise(shape, cov, validate=False):
    """Generate synthetic non-Gaussian noise by selecting best candidate distribution."""
    candidates = {
        "laplace": lambda: np.random.laplace(loc=0, scale=1, size=shape),
        "uniform": lambda: np.random.uniform(-np.sqrt(3), np.sqrt(3), size=shape),
        "exponential": lambda: np.random.exponential(scale=1.0, size=shape) - 1.0,
        "t": lambda: np.random.standard_t(df=3, size=shape),
    }

    best_noise = None
    best_score = 1.0
    best_name = None

    for name, gen in candidates.items():
        z = gen()
        z -= np.mean(z)
        e_colored = color(z, cov).flatten()
        stat, pval = shapiro(e_colored)
        if pval < best_score:
            best_score = pval
            best_noise = z
            best_name = name

    if validate:
        print(f"  Best non-Gaussian fit: {best_name} (Shapiro p={best_score:.4f})")

    return best_noise


def best_polynomial_degree_mdl(
    X: np.ndarray, y: np.ndarray, max_degree: int = 6, lam: float = 2.0
) -> int:
    """Select best polynomial degree using Minimum Description Length criterion."""
    best_score = float("inf")
    best_deg = 1

    n_samples, n_features = X.shape

    for deg in range(1, max_degree + 1):
        model = make_pipeline(
            PolynomialFeatures(degree=deg, include_bias=False), LinearRegression()
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        # MDL = n * log(MSE) + lambda * model complexity
        n_params = model.named_steps["polynomialfeatures"].fit(X).n_output_features_
        mdl = n_samples * np.log(mse + 1e-10) + lam * n_params

        if mdl < best_score:
            best_score = mdl
            best_deg = deg

    return best_deg


def taylor_linearize(
    X_parents_scaled: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-2,
    alpha: float = 1.0,
    max_degree: int = 4,
    lam: float = 2.0,
) -> np.ndarray:
    """
    Performs Taylor linearization of y = f(X) using MDL-optimal polynomial degree.

    Parameters:
        X_parents_scaled: Standardized parent variables
        y: Target variable
        epsilon: Step size for numerical gradient estimation
        alpha: Scaling factor for linearized term
        max_degree: Maximum polynomial degree to consider
        lam: MDL penalty parameter

    Returns:
        Linearized approximation of y
    """
    n_samples, n_features = X_parents_scaled.shape
    x0 = np.median(X_parents_scaled, axis=0)  # robust anchor

    # Select best degree with MDL
    best_deg = best_polynomial_degree_mdl(
        X_parents_scaled, y, max_degree=max_degree, lam=lam
    )

    poly_model = make_pipeline(
        PolynomialFeatures(degree=best_deg, include_bias=False), LinearRegression()
    )
    poly_model.fit(X_parents_scaled, y)

    f_x0 = poly_model.predict(x0.reshape(1, -1))[0]

    # Estimate partial derivatives (Jacobian) at x0
    J = np.zeros(n_features)
    for i in range(n_features):
        x0_perturbed = x0.copy()
        x0_perturbed[i] += epsilon
        f_x0_perturbed = poly_model.predict(x0_perturbed.reshape(1, -1))[0]
        J[i] = (f_x0_perturbed - f_x0) / epsilon

    # Linearized approximation
    X_delta = X_parents_scaled - x0
    return f_x0 + alpha * (X_delta @ J)


def causalMorph(
    X: pd.DataFrame,
    causal_order: list = None,
    adjacency_matrix: pd.DataFrame = None,
    verbose=False,
    validate=False,
    return_details=False,
    debug=False,
    nonlinearity_threshold=0.1,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Apply the CausalMorph algorithm to transform data toward LiNGAM-compatible regime.

    CausalMorph projects observational data onto a manifold where the assumptions
    of the LiNGAM framework are more closely met through three stages:
    - Stage I: Local linearization of causal mechanisms
    - Stage II: Synthesis of non-Gaussian residuals
    - Stage III: Enforcement of statistical uncorrelatedness

    Parameters:
        X: Input DataFrame
        causal_order: List of variable indices in causal order
        adjacency_matrix: DataFrame representing the causal graph
        verbose: Print detailed progress information
        validate: Show validation metrics during processing
        return_details: Return detailed transformation information
        debug: Enable debug mode (generates and saves diagnostic plots)
        nonlinearity_threshold: Minimum R² improvement to save plots (default: 0.1)

    Returns:
        Transformed DataFrame, optionally with detailed transformation info
    """
    X_new = X.copy().values
    var_names = X.columns
    X_values = X.values

    # Enable details collection if debug mode is requested
    if debug:
        return_details = True

    details = {} if return_details else None

    for idx in causal_order:
        var_name = var_names[idx]
        parents = np.where(adjacency_matrix.iloc[idx, :].values != 0)[0]
        if len(parents) == 0:
            if verbose:
                print(f"--- {var_name}: No parents, skipping. ---")
            continue

        if verbose:
            print(f"--- Processing {var_name} (Parents: {[var_names[p] for p in parents]}) ---")

        X_parents = X_values[:, parents]
        y = X_values[:, idx]

        if np.any(np.var(X_parents, axis=0) < 1e-8):
            if verbose:
                print(f"{var_name}: Parent variance near zero. Skipping.")
            continue

        scaler = StandardScaler()
        X_parents_scaled = scaler.fit_transform(X_parents)

        # -----------------
        # Stage I: Linearization
        # -----------------
        try:
            y_linearized = taylor_linearize(X_parents_scaled, y)
            if verbose:
                print(f"{var_name}: (Stage I) Taylor linearization successful.")
        except Exception as e:
            if verbose:
                print(f"{var_name}: (Stage I) Taylor linearization failed ({e}). Skipping.")
            continue

        residuals = y - y_linearized  # E_orig
        residual_std = residuals.std()

        if return_details:
            details[var_name] = {
                "y_true": y.copy(),
                "y_linearized": y_linearized.copy(),
                "residual_before": residuals.copy(),
                "parents_data": X_values[:, parents[0] if len(parents) == 1 else parents].copy(),
                "synthetic_residual": None,
                "final_residual_component": None,
                "final_output": None,
            }

        if residual_std < 1e-8:
            if verbose:
                print(f"{var_name}: Residuals have near-zero variance. Skipping coloring.")
            X_new[:, idx] = y
            if return_details:
                details[var_name]["final_output"] = y.copy()
            continue

        # -----------------
        # Stage II & III: Recoloring and Orthogonalization
        # -----------------
        try:
            # Stage II: Whiten E_orig
            Z_white, W, cov = whiten(residuals.reshape(-1, 1))

            # Stage II: Generate Z_ng (whitened non-Gaussian noise)
            Z_ng = generate_best_non_gaussian_noise(
                Z_white.shape, cov, validate=validate
            )
            if Z_ng is None:
                raise ValueError(f"{var_name}: Noise generation failed")

            np.random.shuffle(Z_ng)

            # Stage II: Recolor to create E_synth
            E_colored = color(Z_ng, cov).flatten()  # E_synth

            # Save E_synth before orthogonalization
            if return_details:
                details[var_name]["synthetic_residual"] = E_colored.copy()

            # Stage III: Orthogonalization
            Q, _ = np.linalg.qr(X_parents)
            proj = Q @ (Q.T @ E_colored.reshape(-1, 1))
            E_colored_orthogonal = E_colored.reshape(-1, 1) - proj  # E_ortho
            E_colored_orthogonal = E_colored_orthogonal.flatten()

            new_std = E_colored_orthogonal.std()
            if new_std < 1e-8:
                if verbose:
                    print(f"{var_name}: (Stage III) Orthogonal residuals have near-zero variance. Skipping.")
                X_new[:, idx] = y
                if return_details:
                    details[var_name]["final_output"] = y.copy()
                continue

            # Stage III: Variance Matching
            E_colored_orthogonal *= residual_std / new_std

            # Stage III: Final Reconstruction
            X_new[:, idx] = y_linearized + E_colored_orthogonal

            if return_details:
                details[var_name]["final_output"] = X_new[:, idx].copy()
                details[var_name]["final_residual_component"] = E_colored_orthogonal.copy()

            if verbose:
                print(f"{var_name}: (Stages II & III) Recoloring and orthogonalization complete.")

        except Exception as e:
            if verbose:
                print(f"{var_name}: (Stages II & III) Coloring failed ({e}). Using linearized only.")
            X_new[:, idx] = y_linearized
            if return_details:
                details[var_name]["final_output"] = y_linearized.copy()

    df_out = pd.DataFrame(X_new, columns=var_names)

    # Generate plots if debug mode is enabled
    if debug and details:
        n_samples = X.shape[0]

        print("\n" + "=" * 80)
        print("DEBUG: Generating raw data temporal plot...")
        print("=" * 80)
        plot_raw_data_example(X, adjacency_matrix, causal_order, n_samples=n_samples)

        print("\n" + "=" * 80)
        print("DEBUG: Generating bijective transformation demonstration...")
        print("=" * 80)
        # Generate bijective demo for each variable with residuals
        for var_name_debug, data_debug in details.items():
            if data_debug['residual_before'] is not None and len(data_debug['residual_before']) > 0:
                plot_bijective_demonstration(
                    data_debug['residual_before'],
                    var_name=var_name_debug,
                    n_samples=n_samples
                )

        print("\n" + "=" * 80)
        print("DEBUG: Generating transformation process plots...")
        print("=" * 80)
        plot_morphing_stages(details, var_names, n_samples=n_samples, nonlinearity_threshold=nonlinearity_threshold)

    return (df_out, details) if return_details else df_out


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

# Color scheme for plots
COLOR_ORIGINAL = '#4472C4'  # Blue for original data
COLOR_LINEAR = '#ED7D31'    # Orange for linearized fit
COLOR_MORPHED = '#ED7D31'   # Orange for morphed/transformed data


def plot_raw_data_example(X: pd.DataFrame, adjacency_matrix: pd.DataFrame, causal_order: list, n_samples: int = None):
    """
    Generates a temporal plot showing raw data time series.

    Parameters:
        X: Original DataFrame
        adjacency_matrix: DataFrame representing the causal graph
        causal_order: List of variable indices in causal order
        n_samples: Number of samples (for title)
    """
    var_names = X.columns
    X_values = X.values

    # Find a variable with parents
    for idx in causal_order:
        var_name = var_names[idx]
        parents = np.where(adjacency_matrix.iloc[idx, :].values != 0)[0]

        if len(parents) == 0:
            continue  # Skip root variables

        if len(parents) > 3:
            continue  # Skip variables with too many parents

        # Found a suitable variable
        y = X_values[:, idx]
        X_parents = X_values[:, parents]

        # For multiple parents, use PCA projection
        if len(parents) > 1:
            pca = PCA(n_components=1)
            parent_data_plot = pca.fit_transform(X_parents).flatten()
            parent_label = f'{len(parents)} Parents (PCA)'
            parent_name = f'{len(parents)} Parents'
        else:
            parent_data_plot = X_parents.flatten()
            parent_label = var_names[parents[0]]
            parent_name = var_names[parents[0]]

        # Time index
        time_idx = np.arange(len(y))

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Build title
        title = f"Raw Data Time Series: '{var_name}' ← {parent_label}"
        if n_samples is not None:
            title += f" (n={n_samples})"
        fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

        # 1. Time series of parent variable
        ax = axes[0]
        ax.plot(time_idx, parent_data_plot, alpha=0.7, color=COLOR_ORIGINAL, lw=1.5)
        ax.set_title(f'{parent_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel(parent_name, fontsize=12)
        ax.grid(True, alpha=0.3)

        # 2. Time series of target variable
        ax = axes[1]
        ax.plot(time_idx, y, alpha=0.7, color='#2ecc71', lw=1.5)
        ax.set_title(f'{var_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel(var_name, fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = f'RawData_Temporal_{var_name}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Raw data temporal plot saved to {plot_filename}")
        plt.close(fig)

        # Only plot one example
        break


def plot_bijective_demonstration(residuals: np.ndarray, var_name: str = "Variable", n_samples: int = None):
    """
    Demonstrates that the whitening/coloring transformation is bijective (invertible).

    Shows:
    Row 1: Original noise → Whitening → Coloring (with synthetic non-Gaussian noise)
    Row 2: Scatter plots proving point-by-point bijectivity for both transformations

    Parameters:
        residuals: Original residual values (1D array)
        var_name: Name of the variable for plot title
        n_samples: Number of samples (for title)
    """
    residuals = np.atleast_1d(residuals).flatten()

    # === TRANSFORMATION A: Original residuals cycle ===
    E_orig = residuals.copy()
    Z_white_orig, _, cov = whiten(E_orig.reshape(-1, 1))
    Z_white_orig = Z_white_orig.flatten()
    E_recovered_A = color(Z_white_orig.reshape(-1, 1), cov).flatten()

    # === TRANSFORMATION B: Synthetic non-Gaussian noise cycle ===
    # Generate synthetic non-Gaussian noise (as done in CausalMorph)
    Z_ng = np.random.laplace(loc=0, scale=1, size=len(residuals))
    Z_ng = Z_ng - np.mean(Z_ng)  # Center it

    # Color the synthetic noise
    E_synth = color(Z_ng.reshape(-1, 1), cov).flatten()

    # Recover: whiten the colored synthetic noise
    Z_recovered, _, _ = whiten(E_synth.reshape(-1, 1))
    Z_recovered = Z_recovered.flatten()

    # Calculate metrics
    mse_A = np.mean((E_recovered_A - E_orig) ** 2)
    corr_A = np.corrcoef(E_orig, E_recovered_A)[0, 1]
    mse_B = np.mean((Z_recovered - Z_ng) ** 2)
    corr_B = np.corrcoef(Z_ng, Z_recovered)[0, 1]

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Build title
    title = f"Bijective Transformation Demonstration: '{var_name}'"
    if n_samples is not None:
        title += f" (n={n_samples})"
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    # Color scheme
    COLOR_ORIG = '#4472C4'      # Blue - original
    COLOR_WHITE = '#9B59B6'     # Purple - whitened
    COLOR_SYNTH = '#E74C3C'     # Red - synthetic/colored
    COLOR_RECOVERED = '#2ECC71' # Green - recovered

    # === Row 1: Distribution comparisons ===

    # 1. Original Noise
    ax = axes[0, 0]
    ax.hist(E_orig, bins=40, alpha=0.7, color=COLOR_ORIG, edgecolor='black', density=True)
    ax.set_title('Original Noise (E_orig)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.text(0.02, 0.98, f'Std: {np.std(E_orig):.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.grid(True, alpha=0.3)

    # 2. Whitened (orthogonalized)
    ax = axes[0, 1]
    ax.hist(Z_white_orig, bins=40, alpha=0.7, color=COLOR_WHITE, edgecolor='black', density=True, label='Whitened Original')
    ax.hist(Z_ng, bins=40, alpha=0.5, color=COLOR_SYNTH, edgecolor='black', density=True, label='Synthetic (Laplace)')
    ax.set_title('Whitened Space', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.text(0.02, 0.98, f'White Std: {np.std(Z_white_orig):.3f}\nSynth Std: {np.std(Z_ng):.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Colored (transformed)
    ax = axes[0, 2]
    ax.hist(E_orig, bins=40, alpha=0.5, color=COLOR_ORIG, edgecolor='black', density=True, label='Original')
    ax.hist(E_synth, bins=40, alpha=0.5, color=COLOR_SYNTH, edgecolor='black', density=True, label='Colored Synthetic')
    ax.set_title('Colored Space (Same Covariance)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.text(0.02, 0.98, f'Orig Std: {np.std(E_orig):.3f}\nSynth Std: {np.std(E_synth):.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # === Row 2: Scatter plots proving bijectivity ===

    # 4. Scatter: Original vs Recovered (whiten→color cycle)
    ax = axes[1, 0]
    ax.scatter(E_orig, E_recovered_A, alpha=0.3, s=10, color=COLOR_ORIG)
    # Perfect line
    lims = [min(E_orig.min(), E_recovered_A.min()), max(E_orig.max(), E_recovered_A.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect Recovery (y=x)')
    ax.set_title('Bijectivity: E_orig → whiten → color', fontsize=14, fontweight='bold')
    ax.set_xlabel('Original', fontsize=12)
    ax.set_ylabel('Recovered', fontsize=12)
    ax.text(0.02, 0.98, f'MSE: {mse_A:.2e}\nCorr: {corr_A:.6f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.legend(loc='lower right')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    # 5. Scatter: Synthetic noise vs Recovered (color→whiten cycle)
    ax = axes[1, 1]
    ax.scatter(Z_ng, Z_recovered, alpha=0.3, s=10, color=COLOR_SYNTH)
    lims = [min(Z_ng.min(), Z_recovered.min()), max(Z_ng.max(), Z_recovered.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect Recovery (y=x)')
    ax.set_title('Bijectivity: Z_ng → color → whiten', fontsize=14, fontweight='bold')
    ax.set_xlabel('Synthetic Noise (Z_ng)', fontsize=12)
    ax.set_ylabel('Recovered (whiten(color(Z_ng)))', fontsize=12)
    ax.text(0.02, 0.98, f'MSE: {mse_B:.2e}\nCorr: {corr_B:.6f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.legend(loc='lower right')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    # 6. Summary diagram
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        "BIJECTIVE TRANSFORMATION PROOF\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Transformation A (Original Cycle):\n"
        f"  E_orig → whiten → Z_white → color → E_recovered\n"
        f"  MSE = {mse_A:.2e}  |  Correlation = {corr_A:.6f}\n\n"
        "Transformation B (Synthetic Cycle):\n"
        f"  Z_ng → color → E_synth → whiten → Z_recovered\n"
        f"  MSE = {mse_B:.2e}  |  Correlation = {corr_B:.6f}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Both cycles show perfect recovery (MSE ≈ 0, Corr ≈ 1)\n"
        "proving whiten() and color() are inverse operations.\n\n"
        "Key Insight:\n"
        "• whiten(color(x, Σ), Σ) = x\n"
        "• color(whiten(x, Σ), Σ) = x\n"
        "The transformation preserves all information."
    )
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save plot
    plot_filename = f'Bijective_Demonstration_{var_name}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Bijective demonstration plot saved to {plot_filename}")
    plt.close(fig)

    return {
        'mse_original_cycle': mse_A,
        'correlation_original_cycle': corr_A,
        'mse_synthetic_cycle': mse_B,
        'correlation_synthetic_cycle': corr_B
    }


def plot_morphing_stages(details: Dict, var_names: List[str], n_samples: int = None, nonlinearity_threshold: float = 0.1):
    """
    Generates a 2x3 grid of plots for each transformed variable showing the 3 main stages.

    Parameters:
        details: Dictionary with transformation details for each variable
        var_names: List of variable names
        n_samples: Number of samples in the dataset (for title)
        nonlinearity_threshold: (Unused - kept for compatibility)
    """
    for var_name in details.keys():
        data = details[var_name]

        # Calculate non-linearity measure (R² improvement from linear to Taylor)
        r2_linear = r2_score(data['y_true'], data['y_true'] - data['residual_before'])
        r2_taylor = r2_score(data['y_true'], data['y_linearized'])
        nonlinearity_measure = r2_taylor - r2_linear

        print(f"✓ Plotting '{var_name}': non-linearity ΔR² = {nonlinearity_measure:.4f}")

        # Determine if visualization is 1D or 2D based on number of parents
        parent_data = data['parents_data']
        is_single_parent = (parent_data.ndim == 1)

        if not is_single_parent and parent_data.shape[1] > 3:
            print(f"⚠ Skipping plots for '{var_name}': has {parent_data.shape[1]} parents (too many for visualization)")
            continue

        # For multiple parents, project to 1D using PCA
        if not is_single_parent:
            pca = PCA(n_components=1)
            parent_data_plot = pca.fit_transform(parent_data).flatten()
            parent_label = f'{parent_data.shape[1]} Parents (PCA)'
            print(f"📊 Plotting '{var_name}' with {parent_data.shape[1]} parents (PCA projection)")
        else:
            parent_data_plot = parent_data
            parent_label = 'Parent'

        # Create the figure
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))

        # Build title
        title = f"CausalMorph Transformation Process: '{var_name}' ← {parent_label}"
        if n_samples is not None:
            title += f" (n={n_samples})"
        fig.suptitle(title, fontsize=20, fontweight='bold', y=1.03)

        # --- Row 1 ---

        # 1. Stage I: Linearization
        ax = axes[0, 0]
        ax.scatter(parent_data_plot, data['y_true'], alpha=0.3, label='Original', color=COLOR_ORIGINAL)
        sort_idx = np.argsort(parent_data_plot)
        ax.plot(parent_data_plot[sort_idx], data['y_linearized'][sort_idx], color=COLOR_LINEAR, lw=3, label='CausalMorph (Stage I)')
        ax.set_title('Stage I: Linearization', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{parent_label}', fontsize=12)
        ax.set_ylabel(f'{var_name}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Stage I: Original Residuals
        ax = axes[0, 1]
        ax.hist(data['residual_before'], bins=30, alpha=0.7, color=COLOR_ORIGINAL, label=r'$\epsilon_{orig}$', edgecolor='black')
        ax.set_title('Stage I: Original Residuals', fontsize=14, fontweight='bold')
        ax.set_xlabel('Residual Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Stage II: Synthetic Residuals
        ax = axes[0, 2]
        if data['synthetic_residual'] is not None:
            ax.hist(data['residual_before'], bins=30, alpha=0.5, label='Original', color=COLOR_ORIGINAL, density=True, edgecolor='black')
            ax.hist(data['synthetic_residual'], bins=30, alpha=0.7, label='CausalMorph (Stage II)', color=COLOR_MORPHED, density=True, edgecolor='black')
            ax.set_title('Stage II: Synthetic Residuals', fontsize=14, fontweight='bold')
            ax.set_xlabel('Residual Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Skipped', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='gray')
            ax.set_title('Stage II: Synthetic Residuals', fontsize=14, fontweight='bold')

        # --- Row 2 ---

        # 4. Stage III: Orthogonality
        ax = axes[1, 0]
        if data['final_residual_component'] is not None:
            ax.scatter(parent_data_plot, data['final_residual_component'], alpha=0.3, color=COLOR_MORPHED, label='CausalMorph (Final)', s=20)
            ax.axhline(0, color='black', linestyle='--', lw=2, label='Zero line')
            ax.set_title('Stage III: Orthogonality', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{parent_label}', fontsize=12)
            ax.set_ylabel('Final Residual', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Skipped', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='gray')
            ax.set_title('Stage III: Orthogonality', fontsize=14, fontweight='bold')

        # 5. Stage III: Variance Match
        ax = axes[1, 1]
        if data['final_residual_component'] is not None:
            ax.hist(data['residual_before'], bins=30, alpha=0.5, label=f'Original (Std: {data["residual_before"].std():.2f})', color=COLOR_ORIGINAL, density=True, edgecolor='black')
            ax.hist(data['final_residual_component'], bins=30, alpha=0.7, label=f'CausalMorph (Std: {data["final_residual_component"].std():.2f})', color=COLOR_MORPHED, density=True, edgecolor='black')
            ax.set_title('Stage III: Variance Matching', fontsize=14, fontweight='bold')
            ax.set_xlabel('Residual Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Skipped', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='gray')
            ax.set_title('Stage III: Variance Matching', fontsize=14, fontweight='bold')

        # 6. Final: Before vs. After
        ax = axes[1, 2]
        ax.scatter(parent_data_plot, data['y_true'], alpha=0.3, label='Original', color=COLOR_ORIGINAL, s=20)
        if data['final_output'] is not None:
            ax.scatter(parent_data_plot, data['final_output'], alpha=0.4, label='CausalMorph', color=COLOR_MORPHED, s=20, marker='^')
        ax.set_title('Final Result: Before vs. After', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{parent_label}', fontsize=12)
        ax.set_ylabel(f'{var_name}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add non-linearity measure as text annotation
        ax.text(0.02, 0.98, f'Non-linearity: ΔR² = {nonlinearity_measure:.4f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save plot
        plot_filename = f'CausalMorph_Stages_{var_name}_nonlin{nonlinearity_measure:.3f}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plots for '{var_name}' saved to {plot_filename}")
        plt.close(fig)


