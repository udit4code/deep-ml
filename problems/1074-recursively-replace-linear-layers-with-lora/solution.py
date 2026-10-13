import numpy as np

class Module:
    def __init__(self):
        self._modules = {}

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.zeros((out_features, in_features))
        self.bias = np.zeros(out_features)
        self.weight_requires_grad = True
        self.bias_requires_grad = True

class LoRALayer:
    def __init__(self, in_features, out_features, rank, alpha):
        self.rank = rank
        self.alpha = alpha
        self.A = np.zeros((rank, in_features))
        self.B = np.zeros((out_features, rank))
        self.A_requires_grad = True
        self.B_requires_grad = True

class LinearWithLoRA:
    def __init__(self, linear, rank, alpha):
        self.linear = linear
        self.linear.weight_requires_grad = False
        self.linear.bias_requires_grad = False
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)


def replace_linear_with_lora(module, rank, alpha, seen=None):
    """
    Recursively replaces all instances of Linear with LinearWithLoRA within a module.
    """
    if seen is None:
        seen = set()
    
    # Avoid infinite loops on circular references
    if id(module) in seen:
        return
        
    seen.add(id(module))
    for name, child in list(module._modules.items()):
        if isinstance(child, Linear):
            # Replace Linear with LinearWithLoRA in-place
            module._modules[name] = LinearWithLoRA(child, rank, alpha)
        elif isinstance(child, Module) and not isinstance(child, LinearWithLoRA):
            # Recurse into other modules, skipping the newly created wrapper
            replace_linear_with_lora(child, rank, alpha, seen)
            
    return module


