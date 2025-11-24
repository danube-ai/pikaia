import numpy as np


def max_scaler(arr: list[int | float] | np.ndarray) -> np.ndarray:
    """
    Scales the input array by dividing each value by the maximum value.

    This method normalizes the data so the largest value becomes 1. Other values
    are scaled proportionally. Does not guarantee the minimum is 0 unless min is 0.

    Args:
        arr (list | np.ndarray):
            Input 1D array or list of numeric values.

    Returns:
        np.ndarray: Scaled array with values in [0, 1] (if all values are non-negative).

    Raises:
        ValueError: If the input array is empty or max value is zero.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Input array must not be empty")

    max_val = np.max(arr)
    if max_val == 0:
        raise ValueError("Maximum value must not be zero")

    return arr / max_val


def min_max_scaler(arr: list[int | float] | np.ndarray) -> np.ndarray:
    """
    Scales the input array to the range [0, 1] using min-max scaling.

    This function performs min-max normalization, which transforms the data
    by subtracting the minimum value and dividing by the range (max - min).
    This ensures all values are in the [0, 1] interval, which is required for
    compatibility with genetic algorithms that expect normalized inputs.

    Args:
        arr (list | np.ndarray):
            Input 1D array or list of numeric values.

    Returns:
        np.ndarray: Scaled array with values in [0, 1].

    Raises:
        ValueError: If the input array is empty.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Input array must not be empty")

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val == min_val:
        return np.zeros_like(arr)

    return (arr - min_val) / (max_val - min_val)


def robust_scaler(arr: list[int | float] | np.ndarray) -> np.ndarray:
    """
    Scales the input array using robust scaling (median and IQR) and then normalizes to [0, 1].

    Robust scaling uses the median and interquartile range (IQR) to scale the data,
    making it less sensitive to outliers compared to standard scaling methods.
    After scaling, the result is normalized to the [0, 1] range to ensure
    compatibility with genetic algorithms.

    Args:
        arr (list | np.ndarray):
            Input 1D array or list of numeric values.

    Returns:
        np.ndarray: Scaled array with values in [0, 1].

    Raises:
        ValueError: If the input array is empty.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Input array must not be empty")

    median = np.median(arr)
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25

    if iqr == 0:
        return np.zeros_like(arr)

    scaled = (arr - median) / iqr
    min_scaled = np.min(scaled)
    max_scaled = np.max(scaled)

    if max_scaled == min_scaled:
        return np.zeros_like(arr)

    return (scaled - min_scaled) / (max_scaled - min_scaled)


def power_transform(
    arr: list[int | float] | np.ndarray, power: float = 0.5
) -> np.ndarray:
    """
    Applies a power transformation to the input array and scales to [0, 1].

    Power transformation raises each value to the specified power, which can
    help stabilize variance and make the data more Gaussian-like. This is
    particularly useful for data with skewed distributions. After transformation,
    the data is scaled to the [0, 1] range.

    Args:
        arr (list | np.ndarray):
            Input 1D array or list of positive numeric values.
        power (float):
            Power to raise the values to. Default is 0.5 (square root).

    Returns:
        np.ndarray: Transformed and scaled array with values in [0, 1].

    Raises:
        ValueError: If the input array contains non-positive values or is empty.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Input array must not be empty")

    if np.any(arr <= 0):
        raise ValueError("Power transform requires all values to be positive")

    transformed = arr**power
    min_t = np.min(transformed)
    max_t = np.max(transformed)

    if max_t == min_t:
        return np.zeros_like(arr)

    return (transformed - min_t) / (max_t - min_t)


def z_score_scaler(arr: list[int | float] | np.ndarray) -> np.ndarray:
    """
    Applies z-score normalization and scales to [0, 1].

    Z-score normalization (standardization) centers the data around the mean
    with a standard deviation of 1. This is useful for algorithms that assume
    normally distributed data. After standardization, the result is scaled
    to the [0, 1] range for genetic algorithm compatibility.

    Args:
        arr (list | np.ndarray):
            Input 1D array or list of numeric values.

    Returns:
        np.ndarray: Normalized and scaled array with values in [0, 1].

    Raises:
        ValueError: If the input array is empty.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Input array must not be empty")

    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return np.zeros_like(arr)

    z = (arr - mean) / std
    min_z = np.min(z)
    max_z = np.max(z)

    if max_z == min_z:
        return np.zeros_like(arr)

    return (z - min_z) / (max_z - min_z)
