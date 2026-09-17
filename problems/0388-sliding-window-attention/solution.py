import numpy as np

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    # Now, we will use optimized implementation, where we do not materialize the Q @ K.T matrix. Our goal is not to compute dot products which we will anyways discard 
    seq_len, d_k = Q.shape 
    _, d_v = V.shape 
    # Step 1 : Form an output matrix for attention scores of seq_len x seq_len 
    output = np.empty((seq_len, d_v), dtype=Q.dtype)
    scale = np.sqrt(d_k)

    # Step 2 : We process row by row for output 
    for row_idx in range(seq_len):
        # Step 2.1 : Extract K_window and V_window for current row_idx
        col_start = max(0, row_idx - window_size)
        col_end = min(seq_len, row_idx + window_size + 1)

        K_window = K[col_start : col_end]
        V_window = V[col_start : col_end]

        # Step 2.2 : Compute K_window @ Q[row_idx]
        scores = K_window @ Q[row_idx] / scale 

        # Step 2.3 : stable softmax for the scores 
        scores = scores - np.max(scores)

        weights = np.exp(scores)

        partition_function = np.sum(weights)

        weights = weights / partition_function 

        # Step 4 : Final output by weighted sum across values for the current window 
        output[row_idx] = weights @ V_window 

    return output


def sliding_window_attention_naive(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    # Naive approach : We will stick to dense implementation, with sliding_mask. Expected Time complexity : O(n^2 x d_v) 
    seq_len, d_k = Q.shape
    # Step 1 : prepare masking matrix 
    indices = np.arange(seq_len)
    row_indices = indices[:, None] # (n, 1)
    col_indices = indices[None, :] # (1, n)
    # Via broadcasting, we get difference matrix. 
    # difference[i][j] = absolute value of (j - i) 
    difference = np.abs(row_indices - col_indices)
    sliding_mask = difference <= window_size

    # Step 2 : Compute dense attention scores 
    scores = Q @ K.T / np.sqrt(d_k)

    # Step 3 : Apply the sliding window mask 
    scores = np.where(sliding_mask, scores, -np.inf)

    # Step 4 : Apply Softmax in a stable way 
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    partition_function = np.sum(exp_scores, axis=1, keepdims=True)
    weights = exp_scores / partition_function

    # Step 5 : Multiply weights with value V 
    output = weights @ V 
    return output
