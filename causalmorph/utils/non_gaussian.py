import numpy as np
import scipy.stats as stats


def check_non_gaussian(data):
    """
    Check if data is non-gaussian using multiple statistical tests.

    Args:
        data: Input time series data

    Returns:
        bool: True if data is likely Gaussian, False if non-Gaussian
    """
    # Normalize data for better test results
    normalized_data = (data - np.mean(data)) / np.std(data)

    # Shapiro-Wilk test (best for smaller samples, n < 2000)
    _, shapiro_p = stats.shapiro(normalized_data)

    # D'Agostino-Pearson test (tests skewness and kurtosis)
    _, dagostino_p = stats.normaltest(normalized_data)

    # Kolmogorov-Smirnov test (compares with normal distribution)
    _, ks_p = stats.kstest(normalized_data, "norm")

    # Anderson-Darling test (more sensitive at tails than KS)
    anderson_result = stats.anderson(normalized_data, "norm")
    # Get critical value at 5% significance
    anderson_critical_val = anderson_result.critical_values[2]  # 5% significance level
    anderson_stat = anderson_result.statistic
    anderson_significant = anderson_stat > anderson_critical_val

    # Calculate skewness and kurtosis
    skewness = abs(stats.skew(normalized_data))
    kurtosis = abs(stats.kurtosis(normalized_data))

    # Combine results (weighted approach)
    # Lower p-values indicate non-Gaussian data
    p_threshold = 0.05
    test_results = [
        shapiro_p < p_threshold,  # Weight: 1.0
        dagostino_p < p_threshold,  # Weight: 1.0
        ks_p < p_threshold,  # Weight: 0.8
        anderson_significant,  # Weight: 1.0
        skewness > 0.5,  # Weight: 0.7
        abs(kurtosis) > 0.5,  # Weight: 0.7
    ]

    weights = [1.0, 1.0, 0.8, 1.0, 0.7, 0.7]
    weighted_sum = sum(r * w for r, w in zip(test_results, weights))

    # If weighted sum is greater than half the total possible weight,
    # consider the data non-Gaussian
    # print(f"Weighted sum: {weighted_sum} of {sum(weights)}")
    is_gaussian = weighted_sum < sum(weights) / 2

    return is_gaussian
