import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, shapiro, probplot

from core.causalmorph_algorithm import whiten, color


def generate_aligned_causalmorph_plots(residuals, var_name="V4"):
    """
    Generate publication-quality plots with aligned color scheme.

    Colors aligned to Results_VariableSize.png:
    - Deep Blue: Original/Baseline
    - Coral/Orange: CausalMorph/Synthetic
    - Green: Bijectivity/Recovery
    """
    # 1. Vibrant colors for publication
    COLOR_BASELINE = '#2563EB'  # Vibrant Blue (Original/Baseline)
    COLOR_MORPH = '#F97316'     # Vibrant Orange (CausalMorph/Synthetic)
    COLOR_BIJECTIVE = '#10B981' # Vibrant Emerald Green (Bijectivity/Recovery)

    # 2. Data processing (Algorithm Stage II)
    E_orig = residuals.flatten()
    E_orig = E_orig - np.mean(E_orig)

    # Stages of CausalMorph
    Z_white, _, cov = whiten(E_orig.reshape(-1, 1))
    Z_white = Z_white.flatten()

    # Synthetic non-Gaussian noise generation
    Z_ng = np.random.laplace(loc=0, scale=1, size=len(E_orig))
    Z_ng = (Z_ng - np.mean(Z_ng)) / np.std(Z_ng)
    E_synth = color(Z_ng.reshape(-1, 1), cov).flatten()

    # Recovery for bijectivity check
    Z_recovered, _, _ = whiten(E_synth.reshape(-1, 1))
    Z_recovered = Z_recovered.flatten()

    # 3. Plotting
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Panel A: Distribution Morphing
    ax = axes[0, 0]
    ax.hist(E_orig, bins=50, alpha=0.6, label=r'Original $\epsilon_{orig}$', color=COLOR_BASELINE, density=True, edgecolor='white', linewidth=0.5)
    ax.hist(E_synth, bins=50, alpha=0.6, label=r'Synthetic $\epsilon_{synth}$', color=COLOR_MORPH, density=True, edgecolor='white', linewidth=0.5)
    ax.set_title("A. Distribution Morphing", fontweight='bold')
    ax.set_xlabel("Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel B: Q-Q Plot
    ax = axes[0, 1]
    probplot(E_synth, plot=ax)
    ax.get_lines()[0].set_markerfacecolor(COLOR_MORPH)
    ax.get_lines()[0].set_markeredgecolor('white')
    ax.get_lines()[0].set_markeredgewidth(0.5)
    ax.get_lines()[0].set_markersize(6)
    ax.get_lines()[0].set_alpha(0.7)
    ax.get_lines()[1].set_color('#DC2626')  # Vibrant red reference line
    ax.get_lines()[1].set_linewidth(2)
    ax.set_title("B. Q-Q Plot: Synthetic Residuals", fontweight='bold')
    ax.grid(True, alpha=0.2)

    # Panel C: Numerical Bijectivity (Latent Space)
    ax = axes[1, 0]
    ax.scatter(Z_ng, Z_recovered, s=35, color=COLOR_BIJECTIVE, alpha=0.6, edgecolors='white', linewidths=0.3, label='Recovered Latents')
    ax.plot([Z_ng.min(), Z_ng.max()], [Z_ng.min(), Z_ng.max()], 'r--', lw=2.5, label='Perfect Recovery')
    ax.set_title("C. Numerical Bijectivity", fontweight='bold')
    ax.set_xlabel(r"Latent $z_{ng}$", fontsize=11)
    ax.set_ylabel(r"Recovered $z_{rec}$", fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    # Panel D: Cycle Integrity (Original Space)
    ax = axes[1, 1]
    E_rec_A = color(Z_white.reshape(-1, 1), cov).flatten()
    ax.scatter(E_orig, E_rec_A, s=35, color=COLOR_BASELINE, alpha=0.6, edgecolors='white', linewidths=0.3, label='Recovered')
    ax.plot([E_orig.min(), E_orig.max()], [E_orig.min(), E_orig.max()], 'r--', lw=2.5, label='Perfect Recovery')
    ax.set_title(r"D. Cycle Integrity: $\epsilon_{orig} \to \epsilon_{rec}$", fontweight='bold')
    ax.set_xlabel("Original", fontsize=11)
    ax.set_ylabel("Recovered", fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plot_filename = f"Transformation_Proof_{var_name}_Aligned.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"✓ Transformation proof plot saved to {plot_filename}")
    plt.close(fig)

    # Console Stats Printing
    print("\n" + "=" * 40)
    print(f"{'Metric':<12} | {'Original':>10} | {'Synthetic':>10}")
    print("-" * 40)
    print(f"{'Mean':<12} | {np.mean(E_orig):>10.4f} | {np.mean(E_synth):>10.4f}")
    print(f"{'Std Dev':<12} | {np.std(E_orig):>10.4f} | {np.std(E_synth):>10.4f}")
    print(f"{'Skewness':<12} | {skew(E_orig):>10.4f} | {skew(E_synth):>10.4f}")
    print(f"{'Kurtosis':<12} | {kurtosis(E_orig):>10.4f} | {kurtosis(E_synth):>10.4f}")
    n_shapiro = min(4999, len(E_orig))
    print(f"{'Shapiro-P':<12} | {shapiro(E_orig[:n_shapiro])[1]:>10.2e} | {shapiro(E_synth[:n_shapiro])[1]:>10.2e}")
    print("=" * 40)


# Alias for backward compatibility
plot_statistical_preservation = generate_aligned_causalmorph_plots
generate_causalmorph_plots = generate_aligned_causalmorph_plots
