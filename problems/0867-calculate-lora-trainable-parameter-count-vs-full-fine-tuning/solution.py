def lora_param_count(num_layers: int, d_model: int, num_target_matrices: int, rank: int) -> dict:
    """
    Compute LoRA vs full parameter counts and compression ratio.

    Args:
        num_layers: Number of layers in the model.
        d_model: Hidden dimension size.
        num_target_matrices: Number of adapted weight matrices per layer.
        rank: Rank r of the low-rank adapters.

    Returns:
        Dict with fields 'lora_params', 'full_params', and 'compression_ratio'.
    """
    total_matrices = num_layers * num_target_matrices
    
    # Full parameters for d_model x d_model matrices across all target layers
    full_params = total_matrices * (d_model * d_model)
    
    # LoRA parameters: A (rank x d_model) + B (d_model x rank) per matrix
    lora_params = total_matrices * (2 * rank * d_model)
    
    # Compression ratio of full parameters relative to LoRA parameters
    compression_ratio = full_params / lora_params if lora_params > 0 else 0.0
    
    return {
        'lora_params': int(lora_params),
        'full_params': int(full_params),
        'compression_ratio': float(compression_ratio)
    }