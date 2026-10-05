import torch
import torch.nn as nn

# What is nn.module ? The nn.module is the base class for all neural network modules in PyTorch.  
# Hence, every module : model, layer, loss function or custom component must inherit from this class.  
# It tracks parameters automatically, handles hardware shifts (from CPU to MPS or GPU) and manages execution modes -> model.train() and model.eval()


class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        # Step 1 : call the parent constructor
        super().__init__()
        # Step 2 : We register the linear layer transformation via a nn.Linear module 
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y 
