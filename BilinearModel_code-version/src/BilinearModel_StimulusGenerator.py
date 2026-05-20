import numpy as np


def bilinear_model_stimulus_train_generator_constant(
    freq, action_time, rest_time, cycles, nRegions
):
    """
    Generate a stimulus train for a bilinear model across multiple brain regions.

    Parameters:
    - freq: Sampling frequency.
    - action_time: Duration of the active period for the stimulus in seconds.
    - rest_time: Duration of the rest period between stimuli in seconds.
    - cycles: Number of stimulus cycles.
    - nRegions: Number of brain regions.

    Returns:
    - U: Stimulus matrix. Shape: (nRegions, total number of samples).
    - timestamps: Array of time points for each sample.
    """

    # Calculate number of samples for each period (activation and rest)
    activation_samples = int(action_time * freq)
    rest_samples = int(rest_time * freq)
    cycle_samples = activation_samples + rest_samples

    # Initialize U with zeros; represents absence of stimulus
    U = np.zeros((nRegions, cycle_samples * cycles))

    # Fill U with pulses (value 1) representing stimulus activation for each cycle
    for i in range(cycles):
        U[:, i * cycle_samples : i * cycle_samples + activation_samples] = 1
        U[:, i * cycle_samples : i * cycle_samples + activation_samples] = 1

    # Create timestamps for the entire period
    Time_period = (action_time + rest_time) * cycles
    timestamps = np.arange(0, Time_period, 1 / freq)

    return U, timestamps


def bilinear_model_stimulus_train_generator(
    freq, action_times, rest_times, cycles_list, nRegions
):
    """
    Generate a stimulus train for a bilinear model across multiple brain regions.

    Parameters:
    - freq: Sampling frequency.
    - action_times: List of durations of the active period for the stimulus in seconds for each region.
    - rest_times: List of durations of the rest period between stimuli in seconds for each region.
    - cycles_list: List of number of stimulus cycles for each region.
    - nRegions: Number of brain regions.

    Returns:
    - U: Stimulus matrix. Shape: (nRegions, total number of samples).
    - timestamps: Array of time points for each sample.
    """

    # Initialize a list to store the stimulus for each region
    U_list = []

    # Maximum number of samples across all regions
    max_samples = 0

    # Generate stimulus for each region
    for i in range(nRegions):
        # Calculate number of samples for each period (activation and rest)
        activation_samples = int(action_times[i] * freq)
        rest_samples = int(rest_times[i] * freq)
        cycle_samples = activation_samples + rest_samples

        # Initialize U_region with zeros; represents absence of stimulus
        U_region = np.zeros(cycle_samples * cycles_list[i])

        # Fill U_region with pulses (value 1) representing stimulus activation for each cycle
        for j in range(cycles_list[i]):
            U_region[j * cycle_samples : j * cycle_samples + activation_samples] = 1

        # Append U_region to U_list
        U_list.append(U_region)

        # Update max_samples
        max_samples = max(max_samples, len(U_region))

    # Pad other regions with zeros to match the maximum number of samples
    for i in range(nRegions):
        if len(U_list[i]) < max_samples:
            U_list[i] = np.pad(U_list[i], (0, max_samples - len(U_list[i])))

    # Convert U_list to a numpy array
    U = np.vstack(U_list)

    # Create timestamps for the entire period
    Time_period = max_samples / freq
    timestamps = np.arange(0, Time_period, 1 / freq)

    return U, timestamps
