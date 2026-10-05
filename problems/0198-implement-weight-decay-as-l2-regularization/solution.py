import numpy as np


def apply_weight_decay(
    parameters: list[list[float]],
    gradients: list[list[float]],
    lr: float,
    weight_decay: float,
    apply_to_all: list[bool],
) -> list[list[float]]:
    """Apply weight decay (L2 regularization) to parameters using NumPy for

    vectorization.

    Args:
        parameters: List of parameter arrays
        gradients: List of gradient arrays
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight
          decay

    Returns:
        Updated parameters as a list of lists of floats
    """
    updated_parameters = []

    for idx, (param_list, grad_list) in enumerate(zip(parameters, gradients)):
        # Convert sub-lists to NumPy arrays for vectorized operations
        w = np.array(param_list, dtype=np.float64)
        grad = np.array(grad_list, dtype=np.float64)

        # Apply weight decay transformation conditionally
        if apply_to_all[idx]:
            grad += weight_decay * w

        # Perform vectorized gradient descent update step
        updated_w = w - lr * grad

        # Convert back to regular Python list to match original type signatures
        updated_parameters.append(updated_w.tolist())

    return updated_parameters
