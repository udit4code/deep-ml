import numpy as np
from typing import List, Union

Number = Union[int, float]
Matrix = List[List[Number]]


def transform_matrix(
    A: Matrix,
    T: Matrix,
    S: Matrix
) -> Union[Matrix, int]:
    """
    Transform matrix A using the operation: T^(-1) * A * S

    Args:
        A: Input matrix
        T: Invertible matrix
        S: Invertible matrix

    Returns:
        Transformed matrix as a list of lists, or -1 if no solution exists
        (non-invertible matrices or incompatible dimensions).
    """
    try:
        A_np = np.asarray(A, dtype=float)
        T_np = np.asarray(T, dtype=float)
        S_np = np.asarray(S, dtype=float)

        # ---------- Shape validation ----------
        if A_np.ndim != 2 or T_np.ndim != 2 or S_np.ndim != 2:
            return -1

        # Matrix multiplication compatibility:
        # T^{-1} (m x m) * A (m x n) * S (n x n)
        if (
            T_np.shape[0] != T_np.shape[1] or
            S_np.shape[0] != S_np.shape[1] or
          