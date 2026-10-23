import numpy as np

def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Batch Normalization on BCHW input.
    """
    assert X.ndim == 4, f"Number of dimensions in X is {X.ndim}"
    batch_size, channels, height, width = X.shape 
    
    if training:
        # Step 1: Compute mean and biased variance across batch, height, and width axes (0, 2, 3)
        batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)  # Default ddof=0 (biased variance)
        
        # Step 2: Normalize the input tensor
        X_hat = (X - batch_mean) / np.sqrt(batch_var + epsilon)
        
        # Step 3: Scale (gamma) and shift (beta)
        out = gamma * X_hat + beta
        
        # Step 4: Initialize running statistics if they are None
        if running_mean is None:
            running_mean = np.zeros((1, channels, 1, 1), dtype=X.dtype)
        if running_var is None:
            running_var = np.ones((1, channels, 1, 1), dtype=X.dtype)
            
        # Step 5: Update running statistics using exponential moving average
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var
        
    else:
        if running_mean is None or running_var is None:
            raise ValueError("running_mean and running_var must be provided when training=False")
            
        # Normalize using pre-computed running statistics during evaluation/inference
        X_hat = (X - running_mean) / np.sqrt(running_var + epsilon)
        out = gamma * X_hat + beta
        
    return out, running_mean, running_var