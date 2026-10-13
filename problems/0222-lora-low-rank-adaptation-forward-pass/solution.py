import torch

def lora_forward(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Compute the LoRA forward pass using PyTorch.
    
    Args:
        x: Input tensor (batch_size x in_features)
        W: Frozen pretrained weights (in_features x out_features)
        A: LoRA matrix A (rank x out_features)
        B: LoRA matrix B (in_features x rank)
        alpha: LoRA scaling factor
        
    Returns:
        Output tensor (batch_size x out_features)
    """
    # Extract rank from matrix dimensions (A is rank x out_features)
    rank = A.shape[0]
    
    # Pretrained path
    base_out = x @ W
    
    # Efficient low-rank path: compute (x @ B) @ A instead of x @ (B @ A)
    # We should avoid materializing the full-weight B @ A. 
    lora_out = (x @ B) @ A

    # Standard LoRA scaling formula: (alpha / rank)
    scaling = alpha / rank
    
    return base_out + scaling * lora_out