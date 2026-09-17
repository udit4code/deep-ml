import numpy as np

def sparse_window_attention(Q, K, V, w, scale_factor=None):
    seq_len, d_k = Q.shape 
    _, d_v = V.shape
    # Step 1 : Construct local window indices
    # First, create offsets = [-w, -(w - 1), .. -1, 0, 1, ..., w - 1, w]
    # Say, window_size w = 1. So, offsets = [-1, 0, 1] whose shape is (3,)
    offsets = np.arange(-w, w + 1) 
    # Say, seq_len = 3. Then, positions = [[0], [1], [2]], whose shape is (3,1) = (seq_len, 1)
    positions = np.arange(seq_len) 
    positions = positions[:, None] 
    # Via broadcasting, (3, 1) + (3,) = (3, 1) + (1,3) = (3, 3)
    # So, [[0], [1], [2]] + [-1, 0, 1] = [[0, 0, 0], [1, 1, 1], [2, 2, 2]] + [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] = [[-1, 0, 1], [0, 1, 2], [1, 2, 3]]
    indices = positions + offsets 
    # How many of the positions in indices are within the range [0, seq_len - 1] ? 
    valid = (indices >= 0) & (indices < seq_len)
    # np.clip forces every value in an array to stay within a specified range: values below the minimum become the minimum, values above the maximum become the maximum, and values already inside the range remain unchanged. 
    safe_indices = np.clip(indices, 0, seq_len - 1)

    # Step 2 : Get local K and V, via using safe_indices
    K_local = K[safe_indices]
    V_local = V[safe_indices]

    # Step 3 : Compute attention scores from Q and K_local 
    # Q[:, None, :] : (seq_len, 1, d_k)
    # K_local       : (seq_len, window_size, d_k)
    # Broadcasting:(seq_len, 1, d_k) * (seq_len, window_size, d_k) = (seq_len, window_size, d_k)
    # Then sum over d_k.
    # We can think of it as doing this for every query position simultaneously. 
    # This is fully vectorized and time complexity is O(L x w x d), where L = seq_len 
    scores = np.sum(Q[:, None, :] * K_local, axis=-1) 
    scores = scores / np.sqrt(d_k)

    # Step 4: Mask invalid boundary positions
    scores = np.where(valid, scores, -np.inf)

    # Step 5 : apply stable softmax onto attention scores 
    scores_max = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    partition_function = np.sum(exp_scores, axis=-1, keepdims=True)

    # Shape of attention_scores is (seq_len, window_size)
    attention_scores = exp_scores / partition_function 

    # Step 6 : Multiply attention_scores with V_local 
    # Shape of attention_scores is (seq_len, w) 
    # Shape of V_local is (seq_len, w, d_v).
    # We do [:, :, None] on attention_scores, to change its shape to (seq_len, w, 1), which can be broadcasted with V_local. 
    # So, shape of attention_scores[:, :, None] * V_local is (seq_len, w, d_v). 
    # We sum over the w dimension and dimension of output is (seq_len, d_v), as we have collapsed dimension w. 
    # What we have done is : For every token, multiply each value vector in its local window by its attention weight, then add those weighted vectors together.
    output = np.sum(attention_scores[:, :, None] * V_local, axis=1)
    return output
