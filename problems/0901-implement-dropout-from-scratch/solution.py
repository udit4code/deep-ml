import torch

def dropout_in_training_mode(x: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return torch.zeros_like(x)
    if p <= 0.0:
        return x

    # Step 1: Create a binary mask where elements are kept with probability (1 - p)
    mask = (torch.rand_like(x) > p).float()
    
    # Step 2: Scale surviving elements by 1 / (1 - p) (Inverted Dropout)
    return x * mask / (1.0 - p)

def dropout_in_eval_mode(x: torch.Tensor, p:float) -> torch.Tensor:
    return x  

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if training:
        return dropout_in_training_mode(x, p) 
    else:
        return dropout_in_eval_mode(x, p)
