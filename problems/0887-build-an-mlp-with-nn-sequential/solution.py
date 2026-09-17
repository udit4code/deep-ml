import torch
import torch.nn as nn

# nn.Sequential is a container class in PyTorch that wraps a sequence of modules or layers into a single unit. Data passed into an nn.Sequential block flows automatically through each internal layer in the exact structural order they were defined. 

# Why do we need it ? 
# 1. It eliminates forward pass boiler-plate. It writes the forward() pass for you behind the scenes, passing the output of one layer directly as the input to the next.  
# 2. It improves Code Readability: It cleanly groups related layers (like a convolutional block or an MLP) together, reducing clutter inside our main nn.Module.  
# 3. It simplifies Tracking: Like any standard nn.Module, treating the sequence as a single unit allows us to move, freeze, or save all internal parameters with a single command.

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    # Return a sequential container executing layers in structural order
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), 
        nn.ReLU(), 
        nn.Linear(hidden_dim, out_dim)
    )
