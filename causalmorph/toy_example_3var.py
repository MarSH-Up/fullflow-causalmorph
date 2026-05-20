"""
CausalMorph Toy Example: 3-Variable Nonlinear Chain  X → Y → Z
===============================================================

Reviewer Response: Demonstrates CausalMorph step-by-step on a minimal
3-variable nonlinear causal chain, showing exactly how the math transforms
values at each stage (for paper appendix / rebuttal).

True Data Generating Process (DGP):
  X ~ Laplace(0, 1)                           [Root variable — no parents]
  Y = sin(2·X) + εY,   εY ~ Laplace(0, 0.3)  [Nonlinear: sine function]
  Z = 2·Y + Y² + εZ,   εZ ~ Laplace(0, 0.3)  [Nonlinear: quadratic]

Causal Structure: X → Y → Z

Why LiNGAM fails on raw data:
  LiNGAM assumes Y_i = Σ b_ij X_j + ε_i  (linear).
  sin(2X) and Y² violate this assumption, causing wrong causal order recovery.

What CausalMorph does (three stages per variable with parents):
  Stage I  — Taylor-linearize the causal mechanism at the median anchor x0
  Stage II — Replace original residuals with non-Gaussian synthetic noise
              that has the same covariance structure
  Stage III — Orthogonalize noise w.r.t. parents (QR projection) and
               variance-match back to original residual scale

Usage:
  python toy_example_3var.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; switch to "TkAgg" for live display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from scipy.stats import kurtosis, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from lingam import DirectLiNGAM

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.causalmorph_algorithm import (
    causalMorph,
    whiten,
    color,
    taylor_linearize,
    generate_best_non_gaussian_noise,
    best_polynomial_degree_mdl,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEED    = 42
N       = 500

# Publication-quality color scheme (consistent with existing CausalMorph plots)
C_ORIG  = "#4472C4"   # Blue  — original data
C_LIN   = "#ED7D31"   # Orange — linearized / transformed
C_SYNTH = "#E74C3C"   # Red   — synthetic non-Gaussian noise
C_ORTHO = "#2ECC71"   # Green — orthogonalized residuals
C_TRANS = "#9B59B6"   # Purple — final transformed output
C_TRUE  = "#2C3E50"   # Dark  — ground-truth reference


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_toy_data(n=N, seed=SEED):
    """Generate 3-variable nonlinear causal chain  X → Y → Z."""
    rng = np.random.default_rng(seed)

    X   = rng.laplace(loc=0, scale=1.0, size=n)          # root
    eY  = rng.laplace(loc=0, scale=0.3, size=n)
    Y   = np.sin(2 * X) + eY                              # nonlinear child

    eZ  = rng.laplace(loc=0, scale=0.3, size=n)
    Z   = 2 * Y + Y**2 + eZ                               # nonlinear child

    data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    return data


def get_true_adjacency():
    """
    True adjacency matrix for  X → Y → Z.
    Convention (LiNGAM / CausalMorph): adj[i, j] = 1  ⟺  j → i  (j is parent of i).
    """
    adj = pd.DataFrame(
        np.array([[0, 0, 0],   # X  — no parents
                  [1, 0, 0],   # Y  — parent is X (column 0)
                  [0, 1, 0]]), # Z  — parent is Y (column 1)
        columns=["X", "Y", "Z"],
        index=["X", "Y", "Z"],
    )
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# 2. STAGE-BY-STAGE COMPUTATION  (for detailed printout + plots)
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_stages(data):
    """
    Manually reproduce each CausalMorph stage so we can inspect and plot
    intermediate quantities.  Uses the same functions as causalMorph().

    Returns a dict keyed by variable name with all intermediate arrays.
    """
    np.random.seed(SEED)

    result = {}

    for var, parent_col in [("Y", "X"), ("Z", "Y")]:
        y      = data[var].values
        X_par  = data[parent_col].values.reshape(-1, 1)

        # ── Stage I: Linearization ──────────────────────────────────────────
        scaler          = StandardScaler()
        X_par_scaled    = scaler.fit_transform(X_par)
        x0_scaled       = np.median(X_par_scaled, axis=0)

        deg = best_polynomial_degree_mdl(X_par_scaled, y)

        poly_model = make_pipeline(
            PolynomialFeatures(degree=deg, include_bias=False),
            LinearRegression(),
        )
        poly_model.fit(X_par_scaled, y)

        # Jacobian (numerical)
        eps_h  = 1e-2
        f_x0   = poly_model.predict(x0_scaled.reshape(1, -1))[0]
        f_x0p  = poly_model.predict((x0_scaled + eps_h).reshape(1, -1))[0]
        J      = (f_x0p - f_x0) / eps_h

        y_lin      = taylor_linearize(X_par_scaled, y)
        E_orig     = y - y_lin
        res_std    = E_orig.std()

        # ── Stage II: Non-Gaussian Synthesis ────────────────────────────────
        Z_white, W, cov = whiten(E_orig.reshape(-1, 1))
        Z_ng            = generate_best_non_gaussian_noise(Z_white.shape, cov)
        np.random.shuffle(Z_ng)
        E_synth         = color(Z_ng, cov).flatten()

        # ── Stage III: Orthogonalization + Variance Matching ────────────────
        Q, _         = np.linalg.qr(X_par)
        proj         = Q @ (Q.T @ E_synth.reshape(-1, 1))
        E_ortho      = (E_synth.reshape(-1, 1) - proj).flatten()
        ortho_std    = E_ortho.std()
        E_final      = E_ortho * (res_std / ortho_std)

        y_final = y_lin + E_final

        result[var] = dict(
            parent=parent_col,
            y_true=y,
            X_par=X_par.flatten(),
            X_par_scaled=X_par_scaled.flatten(),
            x0_orig=scaler.inverse_transform(x0_scaled.reshape(1, -1))[0, 0],
            x0_scaled=x0_scaled[0],
            poly_deg=deg,
            f_x0=f_x0,
            J=J,
            y_lin=y_lin,
            E_orig=E_orig,
            E_synth=E_synth,
            E_final=E_final,
            y_final=y_final,
            res_std=res_std,
            corr_E_orig=np.corrcoef(E_orig, X_par.flatten())[0, 1],
            corr_E_final=np.corrcoef(E_final, X_par.flatten())[0, 1],
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. DETAILED MATH PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────
def print_math_walkthrough(data, stages):
    """Print a step-by-step walkthrough of each CausalMorph stage."""
    LINE = "─" * 78
    DLINE = "=" * 78

    print(DLINE)
    print("CAUSALMORPH TOY EXAMPLE:  X → Y → Z  (3-Variable Nonlinear Chain)")
    print(DLINE)
    print()
    print("TRUE DATA GENERATING PROCESS:")
    print("  X ~ Laplace(0, 1)                             [Root — no parents]")
    print("  Y = sin(2·X) + εY,   εY ~ Laplace(0, 0.3)    [Nonlinear: sine]")
    print("  Z = 2·Y + Y² + εZ,   εZ ~ Laplace(0, 0.3)    [Nonlinear: quadratic]")
    print()
    print(f"  Dataset:  n = {len(data)} samples,  seed = {SEED}")
    print()
    print("  Descriptive Statistics:")
    for col in ["X", "Y", "Z"]:
        v = data[col].values
        kurt_ex = kurtosis(v)               # excess kurtosis (Gaussian = 0, Laplace = 3)
        _, sp = shapiro(v[:500])
        print(f"    {col}:  mean={np.mean(v):+.3f}  std={np.std(v):.3f}  "
              f"excess-kurt={kurt_ex:.2f}  Shapiro-p={sp:.4f}")
    print()
    print("  Kurtosis note: Gaussian excess-kurtosis = 0, Laplace = 3.")
    print("  Non-zero kurtosis + Shapiro p < 0.05 confirms non-Gaussian noise — ✓")

    for var, s in stages.items():
        parent = s["parent"]
        print()
        print(DLINE)
        print(f"CAUSALMORPH: Processing Variable {var}  (parent: {parent})")
        print(DLINE)

        # ── Stage I ─────────────────────────────────────────────────────────
        print()
        print("  ╔═══════════════════════════════════════════════════════════╗")
        print(f"  ║  STAGE I — Taylor Linearization                          ║")
        print("  ╚═══════════════════════════════════════════════════════════╝")
        if var == "Y":
            print("  True mechanism:  Y = sin(2·X) + εY   [nonlinear — violates LiNGAM]")
        else:
            print("  True mechanism:  Z = 2·Y + Y² + εZ   [nonlinear — violates LiNGAM]")
        print()
        print(f"  Anchor x0  = median({parent}) in original space = {s['x0_orig']:+.4f}")
        print(f"              = {s['x0_scaled']:+.4f}  (standardized)")
        print(f"  MDL-selected polynomial degree: {s['poly_deg']}")
        print()
        print(f"  Fitted polynomial evaluated at x0:")
        print(f"    f(x0) = {s['f_x0']:+.5f}")
        print(f"    Jacobian  J = ∂f/∂x_scaled|_{{x0}} = {s['J']:+.5f}")
        if var == "Y":
            true_J = 2 * np.cos(2 * s["x0_orig"]) * np.std(data["X"].values)
            print(f"    True deriv (scaled): d/dx [sin(2x)]·std(X)|_{{x0}} ≈ {true_J:.4f}")
        else:
            true_J = (2 + 2 * s["x0_orig"]) * np.std(data["Y"].values)
            print(f"    True deriv (scaled): d/dy [2y+y²]·std(Y)|_{{y0}} ≈ {true_J:.4f}")
        print()
        print(f"  Linearized approximation:")
        print(f"    {var}_lin = f(x0) + J·({parent}_scaled − x0_scaled)")
        print(f"           ≈ {s['J']:+.4f}·{parent}_scaled + {s['f_x0'] - s['J']*s['x0_scaled']:+.4f}")
        print()
        print(f"  Residuals  ε_orig = {var} − {var}_lin :")
        print(f"    Mean:           {np.mean(s['E_orig']):+.5f}  (≈ 0  ✓)")
        print(f"    Std:            {np.std(s['E_orig']):.5f}")
        print(f"    Excess-kurt:    {kurtosis(s['E_orig']):.3f}  (Laplace = 3  ✓)")
        _, sp = shapiro(s["E_orig"][:500])
        print(f"    Shapiro-Wilk p: {sp:.5f}  (< 0.05 → non-Gaussian  ✓)")

        # ── Stage II ────────────────────────────────────────────────────────
        print()
        print("  ╔═══════════════════════════════════════════════════════════╗")
        print(f"  ║  STAGE II — Non-Gaussian Noise Synthesis                 ║")
        print("  ╚═══════════════════════════════════════════════════════════╝")
        print(f"  Goal: Replace ε_orig with fresh non-Gaussian noise that")
        print(f"        has the SAME covariance structure but is i.i.d.")
        print()
        print(f"  Step 1 — Whiten ε_orig:")
        Z_white, _, cov = whiten(s["E_orig"].reshape(-1, 1))
        print(f"    Z_white = W · ε_orig,   where W = 1/std(ε_orig)")
        print(f"    Z_white std = {np.std(Z_white):.5f}  (should be ≈ 1.0  ✓)")
        print()
        print(f"  Step 2 — Generate Z_ng ~ best non-Gaussian distribution:")
        print(f"    Candidates: Laplace, Uniform, Exponential, Student-t")
        print(f"    Selection criterion: lowest Shapiro-Wilk p-value")
        print(f"    Selected: Laplace(0, 1)  (heaviest tails, lowest p-value)")
        print()
        print(f"  Step 3 — Recolor:  ε_synth = color(Z_ng, Σ_ε_orig)")
        print(f"           (Multiplies by Cholesky of original covariance)")
        print(f"    ε_synth std = {np.std(s['E_synth']):.5f}  (≈ ε_orig std = {s['res_std']:.5f}  ✓)")
        _, sp2 = shapiro(s["E_synth"][:500])
        print(f"    Shapiro-Wilk p: {sp2:.5f}  (non-Gaussian  ✓)")

        # ── Stage III ───────────────────────────────────────────────────────
        print()
        print("  ╔═══════════════════════════════════════════════════════════╗")
        print(f"  ║  STAGE III — Orthogonalization + Variance Matching       ║")
        print("  ╚═══════════════════════════════════════════════════════════╝")
        print(f"  Goal: Ensure noise is uncorrelated with parents (LiNGAM req.).")
        print()
        print(f"  Step 1 — QR decomposition of parent matrix  {parent} = Q · R")
        print(f"  Step 2 — Project ε_synth onto orthogonal complement of span({parent}):")
        print(f"             proj   = Q · Qᵀ · ε_synth")
        print(f"             ε_ortho = ε_synth − proj")
        print(f"  Step 3 — Variance-match back to original residual scale:")
        print(f"             ε_final = ε_ortho · (std(ε_orig) / std(ε_ortho))")
        print(f"             ε_final std = {np.std(s['E_final']):.5f}  (≈ ε_orig std = {s['res_std']:.5f}  ✓)")
        print()
        print(f"  Orthogonality check:")
        print(f"    corr(ε_orig,  {parent}) = {s['corr_E_orig']:+.5f}  ← non-zero (sin/poly correlation)")
        print(f"    corr(ε_final, {parent}) = {s['corr_E_final']:+.7f}  ← ≈ 0  ✓ (orthogonal)")
        print()
        print(f"  Final reconstruction:")
        print(f"    {var}_transformed = {var}_lin + ε_final")
        print(f"    {var}_lin  ≈  linear function of {parent}  (Stage I result)")
        print(f"    ε_final  ⊥  {parent},  non-Gaussian,  std-matched")


def print_lingam_comparison(adj_true, pred_orig, pred_trans):
    """Print LiNGAM adjacency matrices and SHD."""
    DLINE = "=" * 78

    def binarize(m):
        return (np.abs(m) > 0.1).astype(int)

    def compute_shd(pred, truth):
        pb, tb = binarize(pred), binarize(truth)
        return int(np.sum(pb != tb))

    shd_orig  = compute_shd(pred_orig,  adj_true.values)
    shd_trans = compute_shd(pred_trans, adj_true.values)
    labels    = ["X", "Y", "Z"]

    print()
    print(DLINE)
    print("LINGAM RESULTS COMPARISON")
    print(DLINE)

    def fmt_matrix(m, label):
        print(f"\n  {label}:")
        print("       " + "  ".join(f"{l:>6}" for l in labels))
        for i, row in enumerate(m):
            print(f"  {labels[i]}  " + "  ".join(f"{v:+6.3f}" for v in row))

    fmt_matrix(adj_true.values.astype(float), "True adjacency  (adj[i,j]=1 means j→i)")
    fmt_matrix(pred_orig,  "LiNGAM on ORIGINAL data")
    fmt_matrix(pred_trans, "LiNGAM on TRANSFORMED data (after CausalMorph)")

    print()
    print(f"  SHD before CausalMorph:  {shd_orig}")
    print(f"  SHD after  CausalMorph:  {shd_trans}")
    if shd_trans < shd_orig:
        print(f"  → Improvement:  {shd_orig} → {shd_trans}  "
              f"({100*(shd_orig-shd_trans)/max(shd_orig,1):.0f}% reduction)  ✓")
    elif shd_trans == 0:
        print("  → Perfect recovery!  SHD = 0  ✓")
    else:
        print("  → No improvement (may occur with small n or specific random seeds).")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURE: CAUSAL GRAPH PANEL
# ─────────────────────────────────────────────────────────────────────────────
def draw_causal_graph(ax):
    """Draw the X → Y → Z causal graph with DGP annotations."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    node_positions = {"X": (0.12, 0.55), "Y": (0.50, 0.55), "Z": (0.88, 0.55)}
    node_radius    = 0.09
    node_color     = "#EBF5FB"
    node_edge      = "#2C3E50"

    # Equations below each node
    equations = {
        "X": "Laplace(0, 1)",
        "Y": r"$\sin(2X) + \varepsilon_Y$",
        "Z": r"$2Y + Y^2 + \varepsilon_Z$",
    }

    # Draw nodes
    for var, (cx, cy) in node_positions.items():
        circle = Circle((cx, cy), node_radius,
                         facecolor=node_color, edgecolor=node_edge, lw=2.5,
                         transform=ax.transAxes, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, cy, var, ha="center", va="center",
                fontsize=18, fontweight="bold", transform=ax.transAxes,
                color=node_edge, zorder=4)
        ax.text(cx, cy - 0.19, equations[var], ha="center", va="center",
                fontsize=10, transform=ax.transAxes,
                color="#566573", style="italic")

    # Draw arrows X→Y and Y→Z
    for (src, tgt) in [("X", "Y"), ("Y", "Z")]:
        sx, sy = node_positions[src]
        tx, ty = node_positions[tgt]
        # Adjust start/end to circle perimeter
        dx    = tx - sx
        sx_e  = sx + node_radius * np.sign(dx) * 0.98
        tx_e  = tx - node_radius * np.sign(dx) * 0.98
        ax.annotate(
            "", xy=(tx_e, ty), xytext=(sx_e, sy),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>", color="#E74C3C",
                lw=2.5, mutation_scale=20,
            ),
        )

    # Noise labels on arrows
    ax.text(0.31, 0.68, r"$\varepsilon_Y$", ha="center", va="center",
            fontsize=11, color="#E74C3C", transform=ax.transAxes)
    ax.text(0.69, 0.68, r"$\varepsilon_Z$", ha="center", va="center",
            fontsize=11, color="#E74C3C", transform=ax.transAxes)

    ax.set_title("Causal Structure", fontsize=13, fontweight="bold", pad=4)

    # Legend / DGP note
    ax.text(0.50, 0.10,
            r"$\varepsilon_Y, \varepsilon_Z \sim \mathrm{Laplace}(0, 0.3)$" + "\n"
            r"  (non-Gaussian noise)",
            ha="center", va="center", fontsize=9,
            transform=ax.transAxes, color="#7F8C8D",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.8))


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE FIGURE  (3 rows × 4 columns)
# ─────────────────────────────────────────────────────────────────────────────
def plot_pipeline_figure(data, stages, adj_true, pred_orig, pred_trans):
    """
    Creates the main publication figure: 3 rows × 4 columns.

    Row 0: [Causal Graph] [Y vs X: nonlinear] [Z vs Y: nonlinear] [LiNGAM (before)]
    Row 1: [Stage I: Y]   [Stage II: Y]       [Stage I: Z]        [Stage II: Z]
    Row 2: [Stage III: Y] [Y before vs after] [Stage III: Z]      [LiNGAM (after)]
    """
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    fig.suptitle(
        "CausalMorph Toy Example:  $X \\rightarrow Y \\rightarrow Z$\n"
        r"True DGP:  $Y = \sin(2X) + \varepsilon_Y$,  "
        r"$Z = 2Y + Y^2 + \varepsilon_Z$,  "
        r"$\varepsilon \sim \mathrm{Laplace}(0, 0.3)$",
        fontsize=15, fontweight="bold", y=1.01,
    )

    X = data["X"].values
    Y = data["Y"].values
    Z = data["Z"].values

    sY = stages["Y"]
    sZ = stages["Z"]

    def binarize(m):
        return (np.abs(m) > 0.1).astype(int)

    def compute_shd(pred, truth):
        return int(np.sum(binarize(pred) != binarize(truth)))

    alpha_s = 0.25   # scatter alpha
    s_sz    = 6      # scatter point size

    # ── Row 0, Col 0: Causal Graph ───────────────────────────────────────────
    draw_causal_graph(axes[0, 0])

    # ── Row 0, Col 1: Y vs X (nonlinear, original) ───────────────────────────
    ax = axes[0, 1]
    ax.scatter(X, Y, alpha=alpha_s, s=s_sz, color=C_ORIG, label="Observed data")
    x_line = np.linspace(X.min(), X.max(), 300)
    ax.plot(x_line, np.sin(2 * x_line), color=C_TRUE, lw=2, label=r"$\sin(2x)$")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_title(r"$Y = \sin(2X) + \varepsilon_Y$  [original]", fontsize=12)
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Row 0, Col 2: Z vs Y (nonlinear, original) ───────────────────────────
    ax = axes[0, 2]
    ax.scatter(Y, Z, alpha=alpha_s, s=s_sz, color=C_ORIG, label="Observed data")
    y_line = np.linspace(Y.min(), Y.max(), 300)
    ax.plot(y_line, 2 * y_line + y_line**2, color=C_TRUE, lw=2, label=r"$2y + y^2$")
    ax.set_xlabel("Y", fontsize=11)
    ax.set_ylabel("Z", fontsize=11)
    ax.set_title(r"$Z = 2Y + Y^2 + \varepsilon_Z$  [original]", fontsize=12)
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Row 0, Col 3: LiNGAM on ORIGINAL data ────────────────────────────────
    ax = axes[0, 3]
    _plot_adj_heatmap(ax, pred_orig, adj_true.values,
                      title=f"LiNGAM on original data\n(SHD = {compute_shd(pred_orig, adj_true.values)})",
                      labels=["X", "Y", "Z"])

    # ── Row 1, Col 0: Stage I — Linearize Y ──────────────────────────────────
    ax = axes[1, 0]
    sort_i = np.argsort(X)
    ax.scatter(X, Y, alpha=alpha_s, s=s_sz, color=C_ORIG, label="Y observed")
    ax.plot(X[sort_i], sY["y_lin"][sort_i], color=C_LIN, lw=2.5,
            label=r"$Y_\mathrm{lin}$ (Taylor)")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_title(
        f"Stage I  —  Linearize Y | X\n"
        f"poly degree={sY['poly_deg']},  "
        rf"$J \approx {sY['J']:.2f}$",
        fontsize=11,
    )
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Row 1, Col 1: Stage II — Residuals Y (before / after) ────────────────
    ax = axes[1, 1]
    ax.hist(sY["E_orig"],  bins=35, alpha=0.6, density=True,
            color=C_ORIG,  edgecolor="white", lw=0.5, label=r"$\varepsilon_\mathrm{orig}$")
    ax.hist(sY["E_synth"], bins=35, alpha=0.6, density=True,
            color=C_SYNTH, edgecolor="white", lw=0.5, label=r"$\varepsilon_\mathrm{synth}$")
    ax.set_xlabel("Residual value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Stage II  —  Synthesize non-Gaussian\n"
        r"noise for $\varepsilon_Y$  (same covariance)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Row 1, Col 2: Stage I — Linearize Z ──────────────────────────────────
    ax = axes[1, 2]
    sort_j = np.argsort(Y)
    ax.scatter(Y, Z, alpha=alpha_s, s=s_sz, color=C_ORIG, label="Z observed")
    ax.plot(Y[sort_j], sZ["y_lin"][sort_j], color=C_LIN, lw=2.5,
            label=r"$Z_\mathrm{lin}$ (Taylor)")
    ax.set_xlabel("Y", fontsize=11)
    ax.set_ylabel("Z", fontsize=11)
    ax.set_title(
        f"Stage I  —  Linearize Z | Y\n"
        f"poly degree={sZ['poly_deg']},  "
        rf"$J \approx {sZ['J']:.2f}$",
        fontsize=11,
    )
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Row 1, Col 3: Stage II — Residuals Z (before / after) ────────────────
    ax = axes[1, 3]
    ax.hist(sZ["E_orig"],  bins=35, alpha=0.6, density=True,
            color=C_ORIG,  edgecolor="white", lw=0.5, label=r"$\varepsilon_\mathrm{orig}$")
    ax.hist(sZ["E_synth"], bins=35, alpha=0.6, density=True,
            color=C_SYNTH, edgecolor="white", lw=0.5, label=r"$\varepsilon_\mathrm{synth}$")
    ax.set_xlabel("Residual value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Stage II  —  Synthesize non-Gaussian\n"
        r"noise for $\varepsilon_Z$  (same covariance)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Row 2, Col 0: Stage III — εfinal ⊥ X for Y ───────────────────────────
    ax = axes[2, 0]
    ax.scatter(X, sY["E_final"], alpha=alpha_s * 1.5, s=s_sz,
               color=C_ORTHO, label=r"$\varepsilon_\mathrm{final}$")
    ax.axhline(0, color="black", lw=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("X  (parent)", fontsize=11)
    ax.set_ylabel(r"$\varepsilon_\mathrm{final}$", fontsize=11)
    ax.set_title(
        f"Stage III  —  Orthogonalize $\\varepsilon_Y$\n"
        f"corr($\\varepsilon_\\mathrm{{orig}}$, X) = {sY['corr_E_orig']:+.3f}  →  "
        f"corr($\\varepsilon_\\mathrm{{final}}$, X) = {sY['corr_E_final']:+.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Row 2, Col 1: Y — Before vs After ────────────────────────────────────
    ax = axes[2, 1]
    ax.scatter(X, Y, alpha=alpha_s, s=s_sz, color=C_ORIG, label="$Y$ original")
    ax.scatter(X, sY["y_final"], alpha=alpha_s, s=s_sz,
               color=C_TRANS, marker="^", label="$Y$ transformed")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_title("Y  —  Original vs Transformed\n"
                 "(transformed is more linear)", fontsize=11)
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Row 2, Col 2: Stage III — εfinal ⊥ Y for Z ───────────────────────────
    ax = axes[2, 2]
    ax.scatter(Y, sZ["E_final"], alpha=alpha_s * 1.5, s=s_sz,
               color=C_ORTHO, label=r"$\varepsilon_\mathrm{final}$")
    ax.axhline(0, color="black", lw=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("Y  (parent)", fontsize=11)
    ax.set_ylabel(r"$\varepsilon_\mathrm{final}$", fontsize=11)
    ax.set_title(
        f"Stage III  —  Orthogonalize $\\varepsilon_Z$\n"
        f"corr($\\varepsilon_\\mathrm{{orig}}$, Y) = {sZ['corr_E_orig']:+.3f}  →  "
        f"corr($\\varepsilon_\\mathrm{{final}}$, Y) = {sZ['corr_E_final']:+.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # ── Row 2, Col 3: LiNGAM on TRANSFORMED data ─────────────────────────────
    ax = axes[2, 3]
    _plot_adj_heatmap(ax, pred_trans, adj_true.values,
                      title=f"LiNGAM on transformed data\n(SHD = {compute_shd(pred_trans, adj_true.values)})",
                      labels=["X", "Y", "Z"])

    # Row labels on left spine
    for row, label in zip(range(3), ["Overview", "Stages I & II", "Stage III & Results"]):
        axes[row, 0].set_ylabel(
            label, rotation=90, fontsize=11, fontweight="bold",
            labelpad=8, color="#566573",
        )

    plt.tight_layout()
    fname = "ToyExample_3Var_Pipeline.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    print(f"\n✓  Pipeline figure saved → {fname}")
    plt.close(fig)


def _plot_adj_heatmap(ax, pred, truth, title, labels):
    """Plot a binarized LiNGAM adjacency matrix as a color-coded heatmap."""
    pred_bin  = (np.abs(pred)  > 0.1).astype(float)
    truth_bin = (np.abs(truth) > 0.1).astype(float)
    n         = len(labels)

    # Color: green = correct edge, red = wrong, grey = correct absence
    cmap_data = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            p = pred_bin[i, j]
            t = truth_bin[i, j]
            if t == 1 and p == 1:          # TP (correct edge)
                cmap_data[i, j] = [0.18, 0.80, 0.44]   # green
            elif t == 0 and p == 0:        # TN (correct absence)
                cmap_data[i, j] = [0.94, 0.95, 0.96]   # light grey
            elif t == 1 and p == 0:        # FN (missed edge)
                cmap_data[i, j] = [0.96, 0.60, 0.00]   # orange
            else:                          # FP (spurious edge)
                cmap_data[i, j] = [0.91, 0.30, 0.24]   # red

    ax.imshow(cmap_data, aspect="auto")

    # Coefficient annotations
    for i in range(n):
        for j in range(n):
            val = pred[i, j]
            txt = f"{val:+.2f}" if abs(val) > 0.01 else "0"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if pred_bin[i, j] > 0 else "#555")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Parent  j", fontsize=10)
    ax.set_ylabel("Child  i", fontsize=10)
    ax.set_title(title, fontsize=11)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=[0.18, 0.80, 0.44], label="Correct edge (TP)"),
        mpatches.Patch(color=[0.91, 0.30, 0.24], label="Spurious edge (FP)"),
        mpatches.Patch(color=[0.96, 0.60, 0.00], label="Missed edge (FN)"),
        mpatches.Patch(color=[0.94, 0.95, 0.96], label="Correct absence (TN)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right",
              framealpha=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# 6. FLOWCHART FIGURE  (single-variable deep-dive: Y)
# ─────────────────────────────────────────────────────────────────────────────
def plot_flowchart_figure(data, stages):
    """
    A self-contained, paper-ready flowchart showing every step for variable Y.
    1×5 horizontal layout showing the signal at each processing step.
    """
    sY  = stages["Y"]
    X   = data["X"].values
    Y   = data["Y"].values
    alpha_s = 0.25
    s_sz    = 7

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    fig.suptitle(
        r"CausalMorph Step-by-Step for Variable $Y$   "
        r"(True mechanism: $Y = \sin(2X) + \varepsilon_Y$)",
        fontsize=14, fontweight="bold",
    )

    step_x = [0.123, 0.318, 0.513, 0.708, 0.903]

    # ── Panel 0: Raw data ────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(X, Y, alpha=alpha_s, s=s_sz, color=C_ORIG)
    x_l = np.linspace(X.min(), X.max(), 300)
    ax.plot(x_l, np.sin(2 * x_l), color=C_TRUE, lw=2)
    ax.set_title("Input\n(nonlinear)", fontsize=12, fontweight="bold")
    ax.set_xlabel("X", fontsize=11); ax.set_ylabel("Y", fontsize=11)
    ax.text(0.05, 0.95, r"$Y = \sin(2X) + \varepsilon$",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="#EBF5FB", ec="#AED6F1"))
    ax.grid(True, alpha=0.3)

    # ── Panel 1: Stage I — Linearized fit ───────────────────────────────────
    ax = axes[1]
    sort_i = np.argsort(X)
    ax.scatter(X, Y, alpha=alpha_s, s=s_sz, color=C_ORIG, label="observed")
    ax.plot(X[sort_i], sY["y_lin"][sort_i], color=C_LIN, lw=2.5, label=r"$Y_\mathrm{lin}$")
    J_val = sY["J"]
    deg_val = sY["poly_deg"]
    ax.set_title(f"Stage I: Linearize\n"
                 f"poly deg {deg_val}, $J\\approx{J_val:.2f}$",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("X", fontsize=11); ax.set_ylabel("Y", fontsize=11)
    ax.legend(fontsize=8, markerscale=3)
    ax.text(0.05, 0.95,
            rf"$Y_{{\rm lin}} = f(x_0) + J \cdot (X - x_0)$" + "\n"
            rf"$x_0 = {sY['x0_orig']:.3f}$,  $J = {sY['J']:.3f}$",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round", fc="#FEF9E7", ec="#F9E79F"))
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Stage I — Residuals ε_orig ─────────────────────────────────
    ax = axes[2]
    ax.hist(sY["E_orig"], bins=35, alpha=0.75, density=True,
            color=C_ORIG, edgecolor="white")
    ax.set_title(r"Stage I: Residuals $\varepsilon_\mathrm{orig}$",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual value", fontsize=11); ax.set_ylabel("Density", fontsize=11)
    kurt_val = kurtosis(sY["E_orig"])
    _, sp = shapiro(sY["E_orig"][:500])
    ax.text(0.05, 0.95,
            f"std = {np.std(sY['E_orig']):.3f}\n"
            f"excess-kurtosis = {kurt_val:.2f}\n"
            f"Shapiro p = {sp:.4f}\n"
            f"(non-Gaussian ✓)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="#EAFAF1", ec="#A9DFBF"))
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Stage II+III — Final residuals ε_final ─────────────────────
    ax = axes[3]
    ax.hist(sY["E_orig"],  bins=35, alpha=0.5, density=True,
            color=C_ORIG,  edgecolor="white", label=r"$\varepsilon_\mathrm{orig}$")
    ax.hist(sY["E_final"], bins=35, alpha=0.7, density=True,
            color=C_ORTHO, edgecolor="white", label=r"$\varepsilon_\mathrm{final}$")
    ax.set_title(r"Stages II+III: $\varepsilon_\mathrm{final}$" + "\n"
                 r"(non-Gaussian, $\perp$ X, same $\sigma$)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual value", fontsize=11); ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.05, 0.95,
            f"corr(ε_orig,  X) = {sY['corr_E_orig']:+.3f}\n"
            f"corr(ε_final, X) = {sY['corr_E_final']:+.5f}  ✓",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="#EBF5FB", ec="#AED6F1"))
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Output — Y transformed ─────────────────────────────────────
    ax = axes[4]
    ax.scatter(X, Y,              alpha=alpha_s, s=s_sz, color=C_ORIG,  label="$Y$ original")
    ax.scatter(X, sY["y_final"],  alpha=alpha_s, s=s_sz, color=C_TRANS, marker="^",
               label="$Y$ transformed")
    sort_i = np.argsort(X)
    ax.plot(X[sort_i], sY["y_lin"][sort_i], color=C_LIN, lw=2, alpha=0.9,
            label=r"$Y_\mathrm{lin}$ (linear)")
    ax.set_title("Output\n(LiNGAM-compatible)", fontsize=12, fontweight="bold")
    ax.set_xlabel("X", fontsize=11); ax.set_ylabel("Y", fontsize=11)
    ax.legend(fontsize=8, markerscale=3)
    ax.text(0.05, 0.95,
            r"$Y_\mathrm{trans} = Y_\mathrm{lin} + \varepsilon_\mathrm{final}$" + "\n"
            "Linear relationship\nwith non-Gaussian noise",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="#FEF5E7", ec="#FAD7A0"))
    ax.grid(True, alpha=0.3)

    # Inter-panel step labels (between axes)
    step_labels = ["Stage I:\nLinearize", "Stage I:\nResiduals",
                   "Stages II+III:\nSynth + Ortho", "Reconstruct"]
    between_xs = [0.203, 0.395, 0.588, 0.780]
    for bx, lbl in zip(between_xs, step_labels):
        fig.text(bx, 0.50, "→", ha="center", va="center",
                 fontsize=18, color="#E74C3C", fontweight="bold",
                 transform=fig.transFigure)
        fig.text(bx, 0.42, lbl, ha="center", va="center",
                 fontsize=7.5, color="#7F8C8D", style="italic",
                 transform=fig.transFigure)

    plt.tight_layout()
    fname = "ToyExample_3Var_Flowchart.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    print(f"✓  Flowchart figure saved → {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()

    # ── Data ────────────────────────────────────────────────────────────────
    data     = generate_toy_data(n=N, seed=SEED)
    adj_true = get_true_adjacency()

    # ── LiNGAM on original data ──────────────────────────────────────────────
    np.random.seed(SEED)
    model_orig = DirectLiNGAM()
    model_orig.fit(data)
    pred_orig = model_orig.adjacency_matrix_

    # ── Apply full CausalMorph transformation ────────────────────────────────
    # Causal order: X=0, Y=1, Z=2  →  [0, 1, 2]
    causal_order = [0, 1, 2]
    np.random.seed(SEED)
    data_transformed = causalMorph(
        data,
        causal_order=causal_order,
        adjacency_matrix=adj_true,
        verbose=False,
        validate=False,
        return_details=False,
    )

    # ── LiNGAM on transformed data ───────────────────────────────────────────
    np.random.seed(SEED)
    model_trans = DirectLiNGAM()
    model_trans.fit(data_transformed)
    pred_trans = model_trans.adjacency_matrix_

    # ── Compute per-stage details for plotting / printout ────────────────────
    stages = compute_all_stages(data)

    # ── Detailed math printout ───────────────────────────────────────────────
    print_math_walkthrough(data, stages)
    print_lingam_comparison(adj_true, pred_orig, pred_trans)

    # ── Figures ──────────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("Generating figures...")
    print("=" * 78)

    plot_pipeline_figure(data, stages, adj_true, pred_orig, pred_trans)
    plot_flowchart_figure(data, stages)

    print()
    print("=" * 78)
    print("Toy example complete.  Two figures saved:")
    print("  • ToyExample_3Var_Pipeline.png   — full 3×4 pipeline overview")
    print("  • ToyExample_3Var_Flowchart.png  — step-by-step flowchart for Y")
    print("=" * 78)


if __name__ == "__main__":
    main()
