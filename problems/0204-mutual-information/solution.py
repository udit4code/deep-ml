import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
	"""
	Compute the mutual information between two random variables.
	
	Args:
		joint_prob: 2D joint probability distribution P(X,Y)
	
	Returns:
		Mutual information I(X;Y)
	"""
	# Convert to NumPy array (float64 for numerical stability)
    Pxy = np.asarray(joint_prob, dtype=np.float64)

    if Pxy.ndim != 2:
        raise ValueError("joint_prob must be a 2D array")

    # Validate probabilities
    if np.any(Pxy < 0):
        raise ValueError("joint_prob must contain non-negative values")

    # total = Pxy.sum()
    # if not np.isclose(total, 1.0, atol=1e-2):
    #     raise ValueError(f"joint_prob must sum to 1 (got {total})")

    # Marginal distributions
    Px = Pxy.sum(axis=1, keepdims=True)  # shape: (|X|, 1)
    Py = Pxy.sum(axis=0, keepdims=True)  # shape: (1, |Y|)
    
    # Why keepdims=True?
    # Enables broadcast-safe division
    # Avoids accidental shape bugs
    # Faster than reshaping 