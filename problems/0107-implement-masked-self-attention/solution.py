import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Compute masked self-attention.
    """

    seq_len, d_k = Q.shape

    # Step 1 : Compute scaled attention scores
    scores = (Q @ K.T) / np.sqrt(d_k)

    # Step 2 : Apply mask
    if mask is not None:
        scores = scores + mask 

    # Step 3 : Numerically stable softmax
    scores = scores - np.max(scores, axis=1, keepdims=True)

    exp_scores = np.exp(scores)
    partition_function = np.sum(exp_scores, axis=1, keepdims=True)

    attention_weights = exp_scores / partition_function

    # Step 4 : Weighted sum of values
    output = attention_weights @ V

    return output