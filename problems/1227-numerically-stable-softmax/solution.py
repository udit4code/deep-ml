import torch

def softmax(x, dim):
    """Numerically stable softmax along dim.

    Args:
        x (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as x; slices along dim sum to 1
    """
    # Step 1 : Get max along dim 
    x_max = torch.max(x, dim=dim, keepdim=True).values 
    # Step 2 : Exponentiate (x - x_max)
    exp_x = torch.exp(x - x_max)
    # Step 3 : Get denominator 
    denominator = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / denominator
