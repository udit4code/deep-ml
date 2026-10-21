import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    """
	# In terms of dimension indexing, batch_size = 0, sequence_length = 1, feature_dimension = 2
    batch_size, sequence_length, feature_dimension = X.shape
    
    # Step 1: Compute Mean over the feature_dimension (axis = 2)
    x_mean = np.mean(X, axis=2, keepdims=True)

    # Step 2: Compute Variance over the feature dimension (axis = 2)
    x_var = np.var(X, axis=2, keepdims=True)
    
    # Step 3: Normalize X using mean, variance, and epsilon
    x_norm = (X - x_mean) / np.sqrt(x_var + epsilon)
    
    # Step 4: Scale (gamma) and shift (beta)
    out = gamma * x_norm + beta
    
    return out


