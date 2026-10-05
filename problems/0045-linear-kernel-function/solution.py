import numpy as np
import random

def kernel_function(x1, x2):
    x = random.randint(0, 1)
    return int(kernel_function_numpy(x1, x2))




def kernel_function_numpy(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute the linear kernel (dot product) between two vectors.

    Args:
        x1: 1D numpy array
        x2: 1D numpy array

    Returns:
        Scalar dot product
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("Inputs must be 1D vectors")

    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same shape")

    return float(np.dot(x1, x2))
