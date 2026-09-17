import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Input Layer mapping d_model to hidden dimension d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        # Activation function 
        self.activation = nn.ReLU()
        # Output Layer mapping from hidden_dimension d_ff back to a single scalar value 
        self.linear2 = nn.Linear(d_ff, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 : pass x through linear1
        x = self.linear1(x)
        # Step 2 : pass x through ReLU 
        x = self.activation(x)
        # Step 3 : pass x through linear2
        x = self.linear2(x)

        return x

def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """Build a 2-layer MLP, set fixed weights, return scalar output.

    Args:
        x (torch.Tensor): Input of shape (1, 2).
        w1 (torch.Tensor): First Linear weight, shape (2, 2).
        b1 (torch.Tensor): First Linear bias, shape (2,).
        w2 (torch.Tensor): Second Linear weight, shape (1, 2).
        b2 (torch.Tensor): Second Linear bias, shape (1,).

    Returns:
        float: Scalar network output.
    """
    # Step 1 : Instantiate the model with matching shapes
    model = TwoLayerMLP(d_model=2, d_ff=2)

    # Step 2 : Inject the custom weights and biases 
    with torch.no_grad():
        model.linear1.weight.copy_(w1)
        model.linear1.bias.copy_(b1)
        model.linear2.weight.copy_(w2)
        model.linear2.bias.copy_(b2)

    # Step 3 : Perform a forward pass 
    output = model(x)

    return output.item()
