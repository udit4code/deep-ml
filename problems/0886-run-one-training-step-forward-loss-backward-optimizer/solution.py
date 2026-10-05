import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # Step 1 : build a SGD optimizer for the model parameters 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Step 2 : Reset existing gradients to zero before calculation 
    optimizer.zero_grad() 
    # Step 3 : Run the Forward pass 
    predictions = model(x) 
    # Step 4 : Compute loss between ground truth y and predictions 
    loss = F.mse_loss(predictions, y) 
    # Step 5 : Run backward pass 
    loss.backward()
    # Step 6 : Update model weights on computed gradients 
    optimizer.step()
    return loss.item()

