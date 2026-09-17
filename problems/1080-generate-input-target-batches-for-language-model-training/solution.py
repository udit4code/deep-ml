import numpy as np

def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    """
    Samples random training batches for next-token prediction.
    
    Returns a numpy array of shape (2, batch_size, block_size) where:
    - index 0 contains the input sequences (x)
    - index 1 contains the target sequences (y), shifted by one position.
    """
    # Step 1 : Initialize the random number generator
    rng = np.random.default_rng(seed)
    
    # Step 2 : Calculate the max index such that the target window (i + 1 + block_size) fits in data
    # len(data) - block_size ensures the last valid starting index allows for x and y
    high_val = len(data) - block_size
    
    if high_val <= 0:
        raise ValueError("data length must be strictly greater than block_size.")
    
    # Step 3 : Sample random starting offsets
    start_indices = rng.integers(0, high_val, size=batch_size)
    
    # Step 4 : Construct input (x) and target (y) matrices
    x = np.stack([data[i : i + block_size] for i in start_indices])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in start_indices])
    
    # Step 5 : Combine into a single array of shape (2, batch_size, block_size)
    return np.array([x, y], dtype=data.dtype)
        