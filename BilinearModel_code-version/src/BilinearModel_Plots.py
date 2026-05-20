import matplotlib.pyplot as plt
import numpy as np


def plot_Stimulus(
    U_stimulus, timestamps, fig, ax, actual_changes, detected_changes, fontsize=12
):
    nRegions = U_stimulus.shape[0] if U_stimulus.ndim == 2 else 1
    for i in range(nRegions):
        ax.plot(timestamps, U_stimulus[i] if nRegions > 1 else U_stimulus)
    y_positions = np.linspace(
        max(U_stimulus.flatten()) * 0.9,
        max(U_stimulus.flatten()) * 0.1,
        len(actual_changes) + len(detected_changes),
    )
    for i, change in enumerate(actual_changes):
        y_pos = y_positions[i]
        ax.axvline(x=change, color="k", linestyle="--", label="Actual Change Point")
        ax.annotate(
            f"Actual Change\nat {change:.1f}s",
            xy=(change, y_pos),
            xytext=(change + 5, y_pos),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=fontsize,
        )
    for i, change in enumerate(detected_changes):
        y_pos = y_positions[len(actual_changes) + i]
        ax.axvline(
            x=change,
            color="r",
            linestyle="--",
            label=f"Detected Change at {change:.1f}s",
        )
        ax.annotate(
            f"Detected Change\nat {change:.1f}s",
            xy=(change, y_pos),
            xytext=(change + 5, y_pos),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=fontsize,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stimulus Value")
    ax.set_title("Stimulus vs Time")
    ax.legend(
        [f"Region {i+1}" for i in range(nRegions)]
        + ["Actual Change Point", "Detected Change Point"]
    )
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)


def plot_neurodynamics(
    Z, timestamps, fig, ax, actual_changes, detected_changes, fontsize=12
):
    nRegions = Z.shape[1]
    for i in range(nRegions):
        ax.plot(timestamps, Z[:, i], label=f"Region {i+1}")
    y_positions = np.linspace(
        max(Z.flatten()) * 0.9,
        max(Z.flatten()) * 0.1,
        len(actual_changes) + len(detected_changes),
    )
    for i, change in enumerate(actual_changes):
        y_pos = y_positions[i]
        ax.axvline(x=change, color="k", linestyle="--", label="Actual Change Point")
        ax.annotate(
            f"Actual Change\nat {change:.1f}s",
            xy=(change, y_pos),
            xytext=(change + 5, y_pos),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=fontsize,
        )
    for i, change in enumerate(detected_changes):
        y_pos = y_positions[len(actual_changes) + i]
        ax.axvline(
            x=change,
            color="r",
            linestyle="--",
            label=f"Detected Change at {change:.1f}s",
        )
        ax.annotate(
            f"Detected Change\nat {change:.1f}s",
            xy=(change, y_pos),
            xytext=(change + 5, y_pos),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=fontsize,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurodynamic Value")
    ax.set_title("Neurodynamics vs Time")
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)


def plot_DHDQ(
    dq, dh, timestamps, fig, ax, actual_changes, detected_changes, fontsize=12
):
    nRegions = dq.shape[0]
    for r in range(nRegions):
        ax.plot(timestamps, dq[r, :], label=f"dq Region {r+1}")
        ax.plot(timestamps, dh[r, :], label=f"dh Region {r+1}")
    y_positions = np.linspace(
        max(np.hstack((dq, dh)).flatten()) * 0.9,
        max(np.hstack((dq, dh)).flatten()) * 0.1,
        len(actual_changes) + len(detected_changes),
    )
    for i, change in enumerate(actual_changes):
        y_pos = y_positions[i]
        ax.axvline(x=change, color="k", linestyle="--", label="Actual Change Point")
        ax.annotate(
            f"Actual Change\nat {change:.1f}s",
            xy=(change, y_pos),
            xytext=(change + 5, y_pos),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=fontsize,
        )
    for i, change in enumerate(detected_changes):
        y_pos = y_positions[len(actual_changes) + i]
        ax.axvline(
            x=change,
            color="r",
            linestyle="--",
            label=f"Detected Change at {change:.1f}s",
        )
        ax.annotate(
            f"Detected Change\nat {change:.1f}s",
            xy=(change, y_pos),
            xytext=(change + 5, y_pos),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=fontsize,
        )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Relative Hemoglobin Concentration")
    ax.set_title("dq and dh over time")
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)


def plot_Y(Y, timestamps, fig, ax, actual_changes, detected_changes, fontsize=12):
    nRegions = Y.shape[0] // 2
    for r in range(nRegions):
        ax.plot(timestamps, Y[2 * r, :], label=f"dxy-Hb Region {r+1}")
        ax.plot(timestamps, Y[2 * r + 1, :], label=f"oxy-Hb Region {r+1}")
    y_positions = np.linspace(
        max(Y.flatten()) * 0.9,
        max(Y.flatten()) * 0.1,
        len(actual_changes) + len(detected_changes),
    )
    for i, change in enumerate(actual_changes):
        y_pos = y_positions[i]
        ax.axvline(x=change, color="k", linestyle="--", label="Actual Change Point")
        ax.annotate(
            f"Actual Change\nat {change:.1f}s",
            xy=(change, y_pos),
            xytext=(change + 5, y_pos),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=fontsize,
        )
    for i, change in enumerate(detected_changes):
        y_pos = y_positions[len(actual_changes) + i]
        ax.axvline(
            x=change,
            color="r",
            linestyle="--",
            label=f"Detected Change at {change:.1f}s",
        )
        ax.annotate(
            f"Detected Change\nat {change:.1f}s",
            xy=(change, y_pos),
            xytext=(change + 5, y_pos),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=fontsize,
        )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Optical Density Changes")
    ax.set_title("Optical Density Changes over time for dxy-Hb and oxy-Hb")
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
