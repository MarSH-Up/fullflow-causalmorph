import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

# Define paths to easily import custom modules.
# Assuming the script is located two directories deep from the root of the project.
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(
    os.path.join(
        current_directory,
        "..",
    )
)
print("Root", root_directory)
# Adding the root directory to the system path
sys.path.append(root_directory)

from BilinearModel_Miscellaneous import *


def semisynthecticDataExtraction(
    nRegions, freq, HemodynamicSampleLength, semiSyntheticNoiseFreq=4
):
    # Construct the absolute paths to the files
    # src/NoiseData
    deoxy_file_path = os.path.join(root_directory, "src", "NoiseData", "deoxyhb_2.txt")
    oxy_file_path = os.path.join(root_directory, "src", "NoiseData", "oxyhb_2.txt")

    def read_random_column(file_path, nRegions):
        with open(file_path, "r") as file:
            data = np.loadtxt(file)
        random_columns = []
        for _ in range(nRegions):
            col_index = random.randint(0, data.shape[1] - 1)
            random_columns.append(data[:, col_index])
        return random_columns

    deoxy_data = read_random_column(deoxy_file_path, nRegions)
    oxy_data = read_random_column(oxy_file_path, nRegions)

    adjusted_length = int(
        HemodynamicSampleLength * freq / semiSyntheticNoiseFreq
    )  # Assuming original frequency is 4Hz
    resampled_data = []
    for i in range(nRegions):
        resampled_deoxy = scipy.signal.resample(deoxy_data[i], adjusted_length)
        resampled_oxy = scipy.signal.resample(oxy_data[i], adjusted_length)
        resampled_data.append((resampled_deoxy, resampled_oxy))

    return resampled_data


def add_noise_to_hemodynamics(
    deltaQ, deltaH, semisynthetic_noises, percent_error, timestamps
):
    nRegions = deltaQ.shape[0]  # Assuming deltaQ and deltaH have shapes (nRegions, N)

    # Generate unique white noise for each region
    whiteNoise_all_regions = np.array(
        [generate_white_noise_laplace(timestamps) for _ in range(nRegions)]
    )

    # Resample the white noise to match the second dimension of deltaQ and deltaH
    whiteNoise_resampled = np.array(
        [scipy.signal.resample(wn, deltaH.shape[1]) for wn in whiteNoise_all_regions]
    )
    whiteNoise_gain = calculate_gain(percent_error, max_amplitude(deltaH))

    # Add the white noise to deltaQ and deltaH
    deltaQ += whiteNoise_resampled * (whiteNoise_gain / 12)
    deltaH += whiteNoise_resampled * (whiteNoise_gain / 12)

    for deoxy_noise, oxy_noise in semisynthetic_noises:
        # Reshape or resample the noise to match the shape of deltaQ and deltaH
        deoxy_noise_resampled = scipy.signal.resample(deoxy_noise, deltaQ.shape[1])
        oxy_noise_resampled = scipy.signal.resample(oxy_noise, deltaH.shape[1])

        # Calculate gain
        gain_deoxy = calculate_gain(percent_error, max_amplitude(deltaQ))
        gain_oxy = calculate_gain(percent_error, max_amplitude(deltaH))

        # Apply noise
        deltaQ += deoxy_noise_resampled * gain_deoxy
        deltaH += oxy_noise_resampled * gain_oxy

    return deltaQ, deltaH


def add_noise_to_hemodynamics_v1(
    deltaQ, deltaH, semisynthetic_noises, percent_error, timestamps
):
    nRegions = deltaQ.shape[0]  # Assuming deltaQ and deltaH have shapes (nRegions, N)

    # Generate and resample white noise for each region
    whiteNoise_all_regions = np.array(
        [generate_white_noise(timestamps) for _ in range(nRegions)]
    )
    whiteNoise_resampled = np.array(
        [scipy.signal.resample(wn, deltaQ.shape[1]) for wn in whiteNoise_all_regions]
    )

    # Initialize list to store all noises
    all_noises = []
    # Add white noise to the list
    for wn in whiteNoise_resampled:
        all_noises.append((wn, wn))  # Adding twice because we have deltaQ and deltaH

    # Add semisynthetic noises to the list
    for deoxy_noise, oxy_noise in semisynthetic_noises:
        # Reshape or resample the noise to match the shape of deltaQ and deltaH
        deoxy_noise_resampled = scipy.signal.resample(deoxy_noise, deltaQ.shape[1])
        oxy_noise_resampled = scipy.signal.resample(oxy_noise, deltaH.shape[1])
        all_noises.append((deoxy_noise_resampled, oxy_noise_resampled))

    # Calculate total amplitude of all noises
    total_noise_amp = sum(
        max_amplitude(noise[0]) + max_amplitude(noise[1]) for noise in all_noises
    )

    # Calculate desired total amplitude based on percent_error
    desired_amp = max(max_amplitude(deltaQ), max_amplitude(deltaH)) * (
        percent_error / 100
    )

    # Calculate scale factor
    scale_factor = desired_amp / total_noise_amp if total_noise_amp != 0 else 0

    # Apply scaled noises to deltaQ and deltaH
    for noiseQ, noiseH in all_noises:
        deltaQ += noiseQ * scale_factor
        deltaH += noiseH * scale_factor

    return deltaQ, deltaH


def plotSemiSyntheticData(resampled_data):
    plt.figure(figsize=(12, 6))
    for i, (deoxy, oxy) in enumerate(resampled_data):
        plt.subplot(len(resampled_data), 1, i + 1)
        plt.plot(deoxy, label=f"Region {i+1} Deoxy", linewidth=2)
        plt.plot(oxy, label=f"Region {i+1} Oxy", linewidth=2)
        plt.legend()

    plt.suptitle("Resampled Data Plot")
    plt.xlabel("Data Points")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
