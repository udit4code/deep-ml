import numpy as np

# In Multi-Query attention, we have many query heads, and all query heads share only one key head and one value head. We never copy/repeat K/V for every query head 
def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    Please note that d_k, d_v are defined for 1 head. 
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """
    seq_len, d_model = X.shape 
    # W_queries is a list of W_query for each head.  
    num_heads = len(W_queries) # Also, referred to as H. 

    # Step 1 : Get all Q_heads 
    # How ? We project X to W_query for each query head, by iterating through W_queries.  
    # Each W_query is of shape (d_model, d_k) and X is of shape (seq_len, d_model).
    # So, Q_head = X @ W_query , (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
    # We have to stack all these Q_heads together and add a new axis for head-dimension.
    # So, shape of Q_heads = (H, seq_len, d_k) 
    Q_heads = np.stack(
        [X @ W_query for W_query in W_queries], 
        axis=0
    )

    # Step 2 : Project X onto W_k and W_v subspaces 
    # Shape of K is (seq_len, d_k) and shape of V is (seq_len, d_v)
    K = X @ W_key 
    V = X @ W_value

    # Step 3 : Extract d_h, which will be needed later in normalisation of scores for each head 
    # One confusion that needs to be avoided here is : W_key's shape is (d_model, d_k), but here, d_k is actually d_h , defined per head. 
    d_k = K.shape[-1]

    # Step 4 : Compute attention scores for all heads, using vectorized operations 
    # Q_heads has shape (H, seq_len, d_k) and K.T has shape (d_k, seq_len).  
    # Via broadcasting, scores will have shape (H, seq_len, seq_len)
    scores = Q_heads @ K.T 
    scores = scores / np.sqrt(d_k)

    # Step 5 : Compute stable softmax over keys 
    # Shape of scores is still (H, seq_len, seq_len)
    scores = scores - np.max(scores,axis=-1,keepdims=True)
    exp_scores = np.exp(scores)
    attention = exp_scores / np.sum(exp_scores,axis=-1, keepdims=True)

    # Step 6 : Compute outputs for all heads 
    # So, shape of attention is (H, seq_len, seq_len)
    # and shape of V is (seq_len, d_v).
    # Via broadcasting, (H, seq_len, seq_len) @ (seq_len, d_v) = (H, seq_len, d_v)
    head_outputs = attention @ V 

    # Step 7 : Concatenate all head outputs 
    # (H, seq_len, d_v) -> By permuting dimension indices from [0, 1, 2] to [1, 0, 2], we get (seq_len, H, d_v) 
    # After that, we reshape from (seq_len, H, d_v)  to (seq_len, H x d_v) 
    d_v = V.shape[-1]
    concatenated = head_outputs.transpose(1, 0, 2).reshape(seq_len, num_heads * d_v)

    # Step 8 : Final output projection of concatenated head outputs
    # (seq_len, H x d_v) @ (H x d_v, d_model) = (seq_len, d_model) Via broadcasting
    output = concatenated @ W_out 
    return output



