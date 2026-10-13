def lora_memory_footprint(num_params_total: int, lora_target_shapes: list, rank: int, bytes_per_element: int = 2) -> dict:
    """
    Compute GPU memory footprint (in bytes) for full fine-tuning vs low-rank
    adapter fine-tuning.

    Args:
        num_params_total: Total number of parameters in the pre-trained model.
        lora_target_shapes: List of (d_in, d_out) tuples for adapted linear layers.
        rank: Rank r of every low-rank adapter.
        bytes_per_element: Bytes per stored element (e.g. 2 for FP16, 4 for FP32).

    Returns:
        Dict with integer fields 'full_ft_bytes', 'lora_bytes', 'lora_trainable_params'.
    """
    # 1. Calculate total trainable parameters for LoRA
    lora_trainable_params = 0
    for d_in, d_out in lora_target_shapes:
        lora_trainable_params += (rank * d_in) + (d_out * rank)
        
    # 2. Memory cost per trainable parameter (weights, gradients, and optimizer states)
    trainable_param_bytes = 4 * bytes_per_element
    
    # 3. Full Fine-Tuning Memory: All parameters are trainable
    full_ft_bytes = num_params_total * trainable_param_bytes
    
    # 4. LoRA Fine-Tuning Memory:
    # - Base model parameters are frozen (stored at base bytes_per_element)
    # - LoRA trainable parameters carry the full training memory overhead
    lora_bytes = (num_params_total * bytes_per_element) + (lora_trainable_params * trainable_param_bytes)
    
    return {
        'full_ft_bytes': int(full_ft_bytes),
        'lora_bytes': int(lora_bytes),
        'lora_trainable_params': int(lora_trainable_params)
    }