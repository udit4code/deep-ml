def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    # Step 1 : Unpack metadata from model_config
    num_layers = model_config.get("num_layers")
    num_attention_heads = model_config.get("num_attention_heads")

    # In case num of key-value heads is not specified,
    # then, it is fair to assume the case of standard Multi-head-attention,
    # where, each query head has its own key-value head
    num_kv_heads = model_config.get("num_kv_heads", num_attention_heads)
    hidden_size = model_config.get("hidden_size")
    dtype_bytes = model_config.get("dtype_bytes")

    # Step 2 : Compute total_elements per layer (Factor of 2 for both Keys and Values)
    head_dimension = hidden_size // num_attention_heads
    total_elements_per_layer = 2 * batch_size * seq_len * num_kv_heads * head_dimension

    # Step 3 : Compute total elements across all layers
    kv_cache_elements = total_elements_per_layer * num_layers

    # Step 4 : Compute bytes needed to store kv_cache_elements
    kv_cache_size_bytes = kv_cache_elements * dtype_bytes
    kv_cache_size_mb = kv_cache_size_bytes / (1024 ** 2)
    per_layer_size_mb = kv_cache_size_mb / num_layers

    # Step 5 : Compute per_token_size_kb (memory footprint for 1 token across all layers)
    per_token_size_bytes = 2 * num_layers * num_kv_heads * head_dimension * dtype_bytes
    per_token_size_kb = per_token_size_bytes / 1024

    return {
        "kv_cache_elements": kv_cache_elements,
        "kv_cache_size_bytes": kv_cache_size_bytes,
        "kv_cache_size_mb": round(kv_cache_size_mb, 4),
        "per_layer_size_mb": round(per_layer_size_mb, 4),
        "per_token_size_kb": round(per_token_size_kb, 4),
    }