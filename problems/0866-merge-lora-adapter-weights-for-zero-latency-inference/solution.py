import numpy as np

def merge_lora(W0: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int, merge: bool = True) -> np.ndarray:
    """
    Merge or unmerge LoRA adapter weights with a base weight matrix.

    Args:
        W0: Base weight matrix of shape (d, k). If merge=False, this is the already-merged matrix.
        A: LoRA matrix of shape (r, k).
        B: LoRA matrix of shape (d, r).
        alpha: Scaling factor.
        r: LoRA rank.
        merge: If True, return W0 + (alpha/r) * B @ A. If False, return W0 - (alpha/r) * B @ A.

    Returns:
        The resulting weight matrix as a numpy array of shape (d, k).
    """
    # Convert inputs to standard NumPy arrays
    W0 = np.asarray(W0, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    
    # Calculate the low-rank update scaled by alpha / rank
    scaling = alpha / r
    delta = scaling * (B @ A)
    
    # If merge is True, add the low-rank delta to the base weights
    # If merge is False, subtract the low-rank delta to unmerge them
    if merge:
        return W0 + delta
    else:
        return W0 - delta