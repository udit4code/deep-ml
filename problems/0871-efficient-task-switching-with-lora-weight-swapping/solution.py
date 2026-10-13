import numpy as np

def lora_task_switch(W0: np.ndarray, adapters: list, task_sequence: list, x: np.ndarray, scaling: float = 1.0) -> np.ndarray:
    """
    Apply a sequence of LoRA-adapted linear transformations to x, switching
    adapters according to task_sequence while keeping W0 frozen.

    Args:
        W0: Base frozen weight matrix of shape (d_out, d_in).
        adapters: List of (A, B) tuples; A has shape (r, d_in), B has shape (d_out, r).
        task_sequence: List of indices into `adapters` indicating which task to use.
        x: Input batch of shape (batch, d_in).
        scaling: Scalar multiplier applied to the low-rank update.

    Returns:
        numpy array of shape (len(task_sequence), batch, d_out) with stacked outputs.
    """
    # Compute base model output once (batch, d_out)
    base_out = x @ W0.T
    
    outputs = []
    for task_idx in task_sequence:
        A, B = adapters[task_idx]
        
        # Efficient low-rank path: (x @ A.T) @ B.T
        lora_out = (x @ A.T) @ B.T
        
        # Combine base and scaled low-rank adapter outputs
        task_out = base_out + scaling * lora_out
        outputs.append(task_out)
        
    # Stack along a new leading dimension -> (len(task_sequence), batch, d_out)
    return np.stack(outputs, axis=0)