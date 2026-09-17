import torch

# LayerNorm normalises the features of a single example to have zero mean and unit variance. Unlike BatchNorm which mixes statistics across the batch dimension, LayerNorm treats each example as independent.  

# LayerNorm is used in transformer architecture
def layer_norm(x, gamma, beta, eps=1e-5):
    # Step 1 : Get mean, biased variance and standard deviation over the last dimension of x 
    # For broadcasting to work, we must use keepdim=True. 
    mean = torch.mean(x, dim=-1, keepdim=True)
    std = torch.std(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, correction=0, keepdim=True)
    # Step 2 : Standardize x 
    x_h = (x - mean)/torch.sqrt(variance + eps)
    # Step 3 : Apply hadamard product between x_h and gamma and then add beta 
    y = gamma * x_h + beta 
    return y
