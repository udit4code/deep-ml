import torch
from typing import Optional

def to_categorical(x: torch.Tensor, n_col: Optional[int] = None) -> torch.Tensor:
    """
    Perform one-hot encoding on a 1D integer tensor `x`. If `n_col` is not provided, infer it from the max value in `x`.
    """
    # Hint: You can use torch.nn.functional.one_hot
    if x.dim() != 1:
        raise ValueError("Input tensor must be 1D")
    
    if not torch.is_floating_point(x) and not torch.is_complex(x):
        x = x.long()
    else:
        raise ValueError("Input tensor must contain integer values")
    
    if n_col is None:
        n_col = int(torch.max(x).item()) + 1
    
    n_row = x.size(0)
    
    # Create zero matrix
    one_hot = torch.zeros(n_row, n_col, device=x.device, dtype=torch.float32)
    
    # Set correct indices to 1
    one_hot[torch.arange(n_row), x] = 1
    
    return one_hot
