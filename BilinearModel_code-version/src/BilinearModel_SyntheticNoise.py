import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Define paths to easily import custom modules.
# Assuming the script is located two directories deep from the root of the project.
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
print("Root", root_directory)
# Adding the root directory to the system path
sys.path.append(root_directory)

from BilinearModel_Miscellaneous import *


def generate_physiological_noise(timestamps, parameter):
    """Generates more realistic physiological noise for a given physiological parameter."""
    parameters = {
        "heart": {"frequency": 1.08, "std_freq": 0.16},
        "breathing": {"frequency": 0.22, "std_freq": 0.07},
        "vasomotion": {"frequency": 0.082, "std_freq": 0.016},
    }
    if parameter not in parameters:
        raise ValueError("Parameter must be 'heart', 'breathing', or 'vasomotion'")
    param_info = parameters[parameter]
    base_freq = np.abs(
        np.random.normal(param_info["frequency"], param_info["std_freq"])
    )
    primary_noise = np.sin(2 * np.pi * base_freq * timestamps)
    harmonic_noise = generate_harmonic_noise(base_freq, timestamps)
    pink_noise_component = pink_noise(len(timestamps))
    pink_noise_component *= np.std(primary_noise) / np.std(
        pink_noise_component
    )  # normalize amplitude
    return primary_noise + harmonic_noise + pink_noise_component


def generate_harmonic_noise(base_freq, timestamps, num_harmonics=5):
    """Generates harmonic components of physiological noise."""
    harmonic_noise = np.zeros_like(timestamps)
    for i in range(1, num_harmonics + 1):
        harmonic_freq = (i + 1) * base_freq
        harmonic_noise += 0.5 ** (i + 1) * np.sin(
            2 * np.pi * harmonic_freq * timestamps
        )
    return harmonic_noise


def generate_noise(timestamps, noise_type):
    """Factory function for generating different types of noise."""
    if noise_type == "white":
        return generate_white_noise_laplace(timestamps)
    else:
        return generate_physiological_noise(timestamps, noise_type)


def synthetic_physiological_noise_model_v1(
    timestamps,
    noise_types,
    HemodynamicSample,
    percent_error=0,
):
    """Generates physiological noises and applies gains to them."""
    noises = [generate_noise(timestamps, noise_type) for noise_type in noise_types]
    physiological_signals_amp = [max_amplitude(noise) for noise in noises]
    gains = physiologicalNoise_gains(
        physiological_signals_amp, percent_error, max_amplitude(HemodynamicSample)
    )
    return [noise * gain for noise, gain in zip(noises, gains)]


def synthetic_physiological_noise_model(
    timestamps, noise_types, HemodynamicSample, percent_error=0
):
    """Generates physiological noises and scales them to a combined amplitude that is a percentage of the HemodynamicSample's amplitude."""
    # Generate individual noises
    noises = [generate_noise(timestamps, noise_type) for noise_type in noise_types]

    # Calculate the desired total amplitude of the noises
    hemodynamic_max_amp = max_amplitude(HemodynamicSample)
    desired_total_noise_amp = hemodynamic_max_amp * (percent_error / 100)

    # Calculate current total amplitude of the noises
    current_total_noise_amp = sum(max_amplitude(noise) for noise in noises)

    # If the current total amplitude is zero, avoid division by zero
    if current_total_noise_amp == 0:
        raise ValueError("Current total amplitude of noises is zero, cannot scale.")

    # Calculate the scaling factor
    scale_factor = desired_total_noise_amp / current_total_noise_amp

    # Apply the scaling factor to each noise signal
    scaled_noises = [noise * scale_factor for noise in noises]

    return scaled_noises


def combine_noises(noises_with_gains, nRegions):
    # Sum up all the noises
    combined_noise = np.sum(np.array(noises_with_gains), axis=0)

    # Adjust the combined_noise to match the number of regions
    if combined_noise.ndim == 1:
        # If combined_noise is 1D, repeat it for each region
        combined_noise = np.tile(combined_noise, (nRegions, 1))
    else:
        # If combined_noise is already 2D, but does not match nRegions, handle accordingly
        # Example: take only the first nRegions rows, or repeat/interpolate to match nRegions
        combined_noise = (
            combined_noise[:nRegions, :]
            if combined_noise.shape[0] >= nRegions
            else np.vstack([combined_noise] * nRegions)
        )

    return combined_noise


def synthetic_noise_plots(timestamps, noises, labels):
    """Plots different types of synthetic noises."""
    plt.figure(figsize=(12, 8))
    if len(noises) != len(labels):
        raise ValueError("The length of noises and labels must be the same")
    for i, (noise, label) in enumerate(zip(noises, labels), start=1):
        plt.subplot(len(noises), 1, i)
        plt.plot(timestamps, noise, label=f"{label} Noise", linewidth=0.75)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"{label} Noise")
        plt.legend()
    plt.tight_layout()
    plt.show()
