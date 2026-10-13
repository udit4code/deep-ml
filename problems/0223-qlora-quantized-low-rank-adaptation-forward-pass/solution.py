import numpy as np

def qlora_forward(
    x: list[list[float]],
    quantized_W: list[list[int]],
    scale: float,
    zero_point: float,
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:
    """
    QLoRA forward pass with 4-bit quantized frozen weights.
    
    Args:
        x: Input matrix (batch_size x in_features)
        quantized_W: 4-bit quantized weights (in_features x out_features)
                     Values are integers that need to be dequantized
        scale: Quantization scale factor
        zero_point: Quantization zero point for dequantization
        A: LoRA matrix A (rank x out_features) - full precision
        B: LoRA matrix B (in_features x rank) - full precision
        alpha: LoRA scaling factor
        
    Returns:
        Output matrix (batch_size x out_features)
    """
    # Convert inputs to NumPy arrays
    x_arr = np.asarray(x, dtype=float)
    q_W_arr = np.asarray(quantized_W, dtype=float)
    A_arr = np.asarray(A, dtype=float)
    B_arr = np.asarray(B, dtype=float)
    
    # 1. Dequantize base weights using the matching formula: (Q * scale) + zero_point
    W_deq = (q_W_arr * scale) + zero_point
    
    # 2. Base model forward pass
    base_out = x_arr @ W_deq
    
    # 3. Efficient low-rank path: (x @ B) @ A
    lora_out = (x_arr @ B_arr) @ A_arr
    
    # 4. Standard LoRA scaling (alpha / rank)
    rank = A_arr.shape[0]
    scaling = (alpha / rank) if rank > 0 else 0.0
    
    # Combine base output and scaled adapter output
    final_out = base_out + scaling * lora_out
    
    return final_out.tolist()