import torch

def linear_backward(grad_output, x, W):
    # Forward pass : y = x @ W.T + b
    # Step 1 : dL_by_dx = dL_by_dy @ W
    dL_by_dx = grad_output @ W
    
    # Step 2 : dL_by_dW = dL_by_dy.T @ x
    dL_by_dW = grad_output.T @ x
    
    # Step 3 : dL_by_db = sum over the batch dimension (dim 0) 
    # Why ? Because the bias vector b is added directly to every single row (sample) along the batch dimension, its gradient accumulates across the entire batch. 
    dL_by_db = grad_output.sum(dim=0)
    return (dL_by_dx, dL_by_dW, dL_by_db)
