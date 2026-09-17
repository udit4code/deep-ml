import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	seq_len, d_k = Q.shape 
	_, d_v = V.shape 
	# Step 1 : Get scaled scores
	scores = ((Q @ K.T )/ np.sqrt(d_k))
	if mask is not None:
		scores = np.where(mask, scores, -np.inf)
	# Step 2 : Apply softmax with stability on scores 
	scores = scores - np.max(scores, axis=1, keepdims=True)
	exp_scores = np.exp(scores)
	partition_function = np.sum(exp_scores, axis=1, keepdims=True)
	attention_weights = exp_scores / partition_function
	# Step 3 : Do a product of attention_weights with V 
	output = attention_weights @ V 

	return (output, attention_weights)

