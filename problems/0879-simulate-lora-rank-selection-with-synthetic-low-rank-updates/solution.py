import numpy as np

def lora_rank_sweep(delta_W_true: list[list[float]], ranks: list[int]) -> list[float]:
    """
    Compute Frobenius reconstruction error of the best rank-r approximation
    of delta_W_true for each r in ranks.

    Args:
        delta_W_true: target weight-update matrix, shape (m, n)
        ranks: list of candidate ranks to evaluate

    Returns:
        List of reconstruction errors, one per rank, in the same order as `ranks`.
    """
    # Convert input to a NumPy array
    W = np.array(delta_W_true, dtype=float)
    
    # Compute the SVD of the target matrix
    # full_matrices=False returns U (m x k), S (k,), Vt (k x n) where k = min(m, n)
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    
    errors = []
    for r in ranks:
        if r >= len(S):
            # If requested rank is greater than or equal to the matrix rank, error is 0
            err = 0.0
        else:
            # Frobenius error of rank-r truncation is sqrt(sum of squares of remaining singular values)
            err = float(np.sqrt(np.sum(S[r:]**2)))
        errors.append(err)
        
    return errors
