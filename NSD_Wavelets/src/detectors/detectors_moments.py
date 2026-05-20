import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller, kpss
import lingam
import numpy as np
import pandas as pd

all_lingam_models = []
all_cpts = []
transition_counts = []

def apply_LiNGAM(dq_data, window_label):
    # Assume dq_data is already transposed to have shape (n_samples, n_regions)
    n_samples, n_regions = dq_data.shape

    # Generate a list of node names if not provided
    nodes_names = [f"Region {i + 1}" for i in range(n_regions)]

    # Apply the LiNGAM algorithm
    model = lingam.DirectLiNGAM()
    if np.any(np.isnan(dq_data)) or np.any(np.isinf(dq_data)):
        print("Error: dq_data contains NaN or inf values.")
    else:
        model.fit(dq_data)

    # print(is_dag(model.adjacency_matrix_), model.adjacency_matrix_)
    # plot_lingam(model.adjacency_matrix_, nodes_names, None, "Lingam Full")

    return model

def discretize_data(data, bins=4):
    """Discretize the continuous data into bins for each variable."""
    return np.array(
        [
            np.digitize(data[:, i], np.histogram(data[:, i], bins=bins)[1][:-1])
            for i in range(data.shape[1])
        ]
    ).T

def generate_node_labels(num_nodes):
    """
    Generate an array of node labels in the format ["Node 0", "Node 1", ..., "Node n"].

    Parameters:
        num_nodes (int): Number of nodes.

    Returns:
        list: List of node labels as strings.
    """
    return [f"Node {i}" for i in range(num_nodes)]



def learn_cpts(data, adjacency_matrix, labels=None):
    """
    Calculate CPTs using Maximum Likelihood Estimation for all nodes,
    including marginal probabilities for nodes without parents.

    Parameters:
        data (np.ndarray): The input data array with shape (Time, Nodes).
        adjacency_matrix (np.ndarray): The adjacency matrix defining the structure.
        labels (list of str, optional): List of labels for each node. If provided,
                                        will use these labels in the CPTs.

    Returns:
        dict: A dictionary with CPTs for each node.
    """

    # If no labels are provided, use default labels as 'Node_0', 'Node_1', etc.
    if labels is None:
        labels = [f"Node_{i}" for i in range(adjacency_matrix.shape[0])]

    # Prepare to store CPTs
    cpts = {}

    for child_node in range(adjacency_matrix.shape[0]):
        # Find parents of the current node
        parents = np.where(adjacency_matrix[:, child_node] == 1)[0]

        # If there are no parents, calculate marginal probabilities
        if len(parents) == 0:
            # Count occurrences of each value of the child node
            child_counts = pd.Series(data[:, child_node]).value_counts(normalize=True)
            cpts[labels[child_node]] = (
                child_counts.to_frame(name="Prob")
                .reset_index()
                .rename(columns={"index": labels[child_node]})
            )
        else:
            # Calculate conditional probabilities given the parents
            parent_data = data[:, parents]
            child_data = data[:, child_node]
            # Use the provided labels for columns
            df = pd.DataFrame(
                np.column_stack((parent_data, child_data)),
                columns=[f"{labels[p]}" for p in parents] + [labels[child_node]],
            )

            # Count occurrences and normalize to get conditional probabilities
            cpt = (
                df.groupby([labels[p] for p in parents])[labels[child_node]]
                .value_counts(normalize=True)
                .unstack()
                .fillna(0)
            )
            cpts[labels[child_node]] = cpt.reset_index()

    return cpts


def adf_test(series):
    result = adfuller(series)
    return result[0], result[1]


def kpss_test(series):
    result = kpss(series, regression="c")
    return result[0], result[1]


def analyze_stationarity(window_signal):
    adf_stat, adf_p = adf_test(window_signal)
    kpss_stat, kpss_p = kpss_test(window_signal)

    return {"adf_p": adf_p, "kpss_p": kpss_p}


def moment_variance(signal):
    return np.var(signal)


def moment_skew(signal):
    return skew(signal)


def moment_skew(signal):
    return kurtosis(signal)


def evaluate_stationarity(moments, std_multiplier=1.5):
    """
    Evaluate the stationarity of a signal based on statistical moments.

    Parameters:
        moments (list of tuples): List containing tuples of statistical moments (mean, variance, skewness, kurtosis) for each window.
        std_multiplier (float): Multiplier for standard deviation to determine the threshold.

    Returns:
        dict: Dictionary indicating whether the signal is stationary based on each moment and overall.
    """
    moments_array = np.array(moments)
    mean_values = moments_array[:, 0]
    variance_values = moments_array[:, 1]
    skewness_values = moments_array[:, 2]
    kurtosis_values = moments_array[:, 3]

    # Calculate mean and standard deviation of the moments
    mean_mean = np.mean(mean_values)
    std_mean = np.std(mean_values)

    mean_variance = np.mean(variance_values)
    std_variance = np.std(variance_values)

    mean_skewness = np.mean(skewness_values)
    std_skewness = np.std(skewness_values)

    mean_kurtosis = np.mean(kurtosis_values)
    std_kurtosis = np.std(kurtosis_values)

    # Determine thresholds based on standard deviation
    mean_threshold = std_multiplier * std_mean
    variance_threshold = std_multiplier * std_variance
    skewness_threshold = std_multiplier * std_skewness
    kurtosis_threshold = std_multiplier * std_kurtosis

    # Initialize variables to track consistency
    is_mean_stationary = True
    is_variance_stationary = True
    is_skewness_stationary = True
    is_kurtosis_stationary = True

    # Check consistency across windows
    for moment in moments:
        if abs(moment[0] - mean_mean) > mean_threshold:
            is_mean_stationary = False
        if abs(moment[1] - mean_variance) > variance_threshold:
            is_variance_stationary = False
        if abs(moment[2] - mean_skewness) > skewness_threshold:
            is_skewness_stationary = False
        if abs(moment[3] - mean_kurtosis) > kurtosis_threshold:
            is_kurtosis_stationary = False

    # Determine overall stationarity
    is_stationary = (
        is_mean_stationary
        and is_variance_stationary
        and is_skewness_stationary
        and is_kurtosis_stationary
    )

    return {
        "Mean Stationary": is_mean_stationary,
        "Variance Stationary": is_variance_stationary,
        "Skewness Stationary": is_skewness_stationary,
        "Kurtosis Stationary": is_kurtosis_stationary,
        "Overall Stationary": is_stationary,
    }


def moments_observation(moments, num_windows, experiment_name="images/temp"):
    experiment_dir = experiment_name
    os.makedirs(experiment_dir, exist_ok=True)
    means = [moment[0] for moment in moments]
    variances = [moment[1] for moment in moments]
    skewnesses = [moment[2] for moment in moments]
    kurtoses = [moment[3] for moment in moments]

    # Window data
    # Plotting the statistical moments over time
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    axs[0].plot(range(1, num_windows + 1), means, marker="o")
    axs[0].set_title("Mean Over Time")
    axs[0].set_xlabel("Window Index")
    axs[0].set_ylabel("Mean")
    axs[0].grid(True)

    axs[1].plot(range(1, num_windows + 1), variances, marker="o", color="orange")
    axs[1].set_title("Variance Over Time")
    axs[1].set_xlabel("Window Index")
    axs[1].set_ylabel("Variance")
    axs[1].grid(True)

    axs[2].plot(range(1, num_windows + 1), skewnesses, marker="o", color="green")
    axs[2].set_title("Skewness Over Time")
    axs[2].set_xlabel("Window Index")
    axs[2].set_ylabel("Skewness")
    axs[2].grid(True)

    axs[3].plot(range(1, num_windows + 1), kurtoses, marker="o", color="red")
    axs[3].set_title("Kurtosis Over Time")
    axs[3].set_xlabel("Window Index")
    axs[3].set_ylabel("Kurtosis")
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "statistical_moments_observations.png"))
    plt.show()


def adjacency_matrix_binary(adjacency_matrix):

    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[i])):
            if adjacency_matrix[i][j] > 0:
                adjacency_matrix[i][j] = 1
            elif adjacency_matrix[i][j] < 0:
                adjacency_matrix[i][j] = -1

    return adjacency_matrix



def analyze_signal_windows(
    signal_piecewise,
    window_size,
    overlap,
    analyze_stationarity,
    moment_variance,
    threshold=0.5,  # Fraction of channels required to detect non-stationarity
):
    """
    Analyze signal windows for stationarity across multiple channels.

    Parameters:
        signal_piecewise (np.ndarray): Signal array of shape (CHANNELS, TIME).
        window_size (int): Size of each window (applied to the time dimension).
        overlap (int): Overlap between consecutive windows (applied to the time dimension).
        analyze_stationarity (function): Function to perform stationarity tests.
        moment_variance (function): Function to calculate variance for moments.
        threshold (float): Fraction of channels required to detect non-stationarity.

    Returns:
        tuple: Results including moments, stationarity status, and transitions.
    """
    num_channels, num_timepoints = signal_piecewise.shape
    num_windows = (num_timepoints - window_size) // (window_size - overlap) + 1

    moments = []
    stationarity_results = []
    stationarity_status = []
    independent_stationarity_status = []
    transition_sample_means = []  # To store mean sample indices of transitions

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signals = signal_piecewise[:, start:end]  # All channels in the window

        # Store aggregated results for all channels
        channel_stationarity = []

        # Calculate moments and stationarity tests for each channel
        for ch in range(num_channels):
            window_signal = window_signals[ch, :]
            mean_value = np.mean(window_signal)
            variance_value = moment_variance(window_signal)
            skewness_value = skew(window_signal)
            kurtosis_value = kurtosis(window_signal)
            moments.append((mean_value, variance_value, skewness_value, kurtosis_value))

            stationarity_result = analyze_stationarity(window_signal)
            stationarity_results.append(stationarity_result)

            adf_p_value = stationarity_result["adf_p"]
            kpss_p_value = stationarity_result["kpss_p"]
            adf_status = "Stationary" if adf_p_value < 0.05 else "Non-Stationary"
            kpss_status = "Stationary" if kpss_p_value > 0.05 else "Non-Stationary"

            # Determine final status for the channel
            if adf_status == "Stationary" and kpss_status == "Stationary":
                final_status = "Stationary"
            elif adf_status == "Non-Stationary" and kpss_status == "Non-Stationary":
                final_status = "Non-Stationary"
            else:
                final_status = "Non-Stationary"  # Conservative assumption

            channel_stationarity.append(final_status)

        # Aggregate channel-wise stationarity
        non_stationary_channels = channel_stationarity.count("Non-Stationary")
        if non_stationary_channels / num_channels >= threshold:
            overall_status = "Non-Stationary"
        else:
            overall_status = "Stationary"

        independent_stationarity_status.append(overall_status)

    for i in range(1, num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        current_result = independent_stationarity_status[i]
        prev_result = independent_stationarity_status[i - 1]
        current_window_signal_full = signal_piecewise[:, start:end]

        transition_detected = current_result != prev_result
        if transition_detected:
            mean_sample = (start + end) // 2  # Calculate the mean sample number
            transition_sample_means.append(mean_sample)  # Add to the list

            print("Triggering LiNGAM...")
            lingamModel = apply_LiNGAM(current_window_signal_full.T, f"Window {i}")
            print(lingamModel._adjacency_matrix.shape)
            binary_adjacency = adjacency_matrix_binary(lingamModel._adjacency_matrix)
            discretized_data = discretize_data(current_window_signal_full.T, 4)
            cpts = learn_cpts(
                discretized_data,
                binary_adjacency,
                generate_node_labels(len(lingamModel.adjacency_matrix_)),
            )

            # Save the learned model, CPTs, and count
            all_lingam_models.append(lingamModel)
            all_cpts.append(cpts)
            transition_counts.append(i)

        stationarity_status.append(current_result)

    return (
        moments,
        independent_stationarity_status,
        stationarity_status,
        all_lingam_models,
        all_cpts,
        transition_counts,
        transition_sample_means,  # Return the detected transitions
    )
