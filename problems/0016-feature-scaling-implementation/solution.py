import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
    if data.size == 0:
        raise ValueError("Input data must not be empty")

    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array (n_samples, n_features)")

    data = np.asarray(data, dtype=np.float64)

    # ---------- Standardization ----------
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    standardized_data = np.divide(
        data - mean,
        std,
        out=np.zeros_like(data),
        where=std != 0
    )

    # ---------- Min-Max Normalization ----------
    min_val = data.min(axis=0)
    range_val = data.max(axis=0) - min_val

    normalized_data = np.divide(
        data - min_val,
        range_val,
        out=np.zeros_like(data),
        where=range_val != 0
    )

    return standardized_data, normalized_data
