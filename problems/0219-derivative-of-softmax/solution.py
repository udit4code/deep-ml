import torch

def softmax_derivative_v3(x: torch.Tensor) -> torch.Tensor:
    # Assuming x is (B, N) and you want (B, N, N):
    s = torch.softmax(x, dim=-1)          # (B, N)
    diag = torch.diag_embed(s)             # (B, N, N)
    outer = s.unsqueeze(-1) * s.unsqueeze(-2)
    return diag - outer

def softmax_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the softmax function using PyTorch.
    
    Args:
        x: Input tensor
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here - you can use torch.autograd.functional.jacobian
    # or compute it directly from softmax output
    return  softmax_derivative_v3(x)