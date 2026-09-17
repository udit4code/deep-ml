import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    # Step 1 : Project the new token embedding into query, key and value vectors using the provided weight matrices 
    # x_new (d_model, ) @ W_Q (d_model, d_k) = q_new (1, d_k)
    # So, q_new, k_new, v_new are row vectors : 1 x d_k and 1 x d_v shape 
    q_new = (x_new @ W_Q).reshape(1, -1) 
    k_new = (x_new @ W_K).reshape(1, -1)
    v_new = (x_new @ W_V).reshape(1, -1) 
    # Step 2 : Check if cache exists. Based on it 2 flows. 
    K_cache = None 
    V_cache = None
    if cache:
        # Step 2.1 : If cache is available, then, first unload cached keys and values and then, update them further with latest k_new and v_new 
        K_cache, V_cache = cache 
        # We want to stack k_new and v_new vertically on the cache, as they are 1 x d_k and 1 x d_v row vectors
        K_cache = np.vstack((K_cache, k_new))
        V_cache = np.vstack((V_cache, v_new))
    else:
        # Step 2.2 : If cache is empty, then, update it with latest key and value.  
        K_cache = k_new.reshape(1, -1) # So that we can reshape to (1, d_k)
        V_cache = v_new.reshape(1, -1) # So that we can reshape to (1, d_v)
    # We expect shape of K_cache to be (t, d_k) and V_cache to be (t, d_v), assuming we have seen t tokens so far. 
    # Step 3 : Compute scaled dot-product attention between new query and all cached keys 
    # Step 3.1 : (1, d_k) @ (t, d_k).T = (1, d_k) @ (d_k, t) = (1, t)
    scores = q_new @ K_cache.T 
    # Step 3.2 : Normalize the scores via d_k
    d_k = q_new.shape[1]
    scores /= np.sqrt(d_k)
    # Step 3.3 : Apply softmax, which doesn't change shape. 
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    # Shape of weights is still (1, t)
    # Step 4 : Compute the attention output vector  
    # (1, t) @ (t, d_v) = (1, d_v) 
    attention_output_vector = weights @ V_cache

    attention_output_vector = attention_output_vector.squeeze(0)

    updated_cache = (K_cache, V_cache)
    return attention_output_vector, updated_cache 
