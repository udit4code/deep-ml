import torch

def grad_of_quadratic(x_value: float) -> float:
    # Step 1 : Create a Leaf Tensor with gradient enabled
    x = torch.tensor(x_value, dtype=torch.float32, requires_grad=True)
    # Step 2 : Compute f = x^2 + 3x + 2
    y = x**2 + 3*x + 2 
    # Step 3 : Compute gradient via back-propagation 
    y.backward()
    
    return x.grad.item()


