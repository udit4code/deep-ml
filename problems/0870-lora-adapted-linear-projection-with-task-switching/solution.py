import numpy as np

class LoRALinear:
    def __init__(self, W0: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float):
        """
        W0: frozen base weight, shape (d_out, d_in)
        A:  trainable low-rank factor, shape (r, d_in)
        B:  trainable low-rank factor, shape (d_out, r)
        alpha: scaling factor for the low-rank update
        """
        self.W0 = W0
        self.A = A
        self.B = B
        self.alpha = alpha
        self.r = A.shape[0]  # Rank r

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass efficiently without materializing (d_out, d_in) intermediates.
        """
        # Base model output: (batch_size, d_out)
        base_out = x @ self.W0.T
        
        # Low-rank adapter output computed as (x @ A.T) @ B.T
        lora_out = (x @ self.A.T) @ self.B.T
        
        # Standard LoRA scaling: alpha / rank
        scaling = self.alpha / self.r
        
        return base_out + scaling * lora_out

    def num_trainable_params(self) -> int:
        """
        Returns the total number of trainable parameters in matrices A and B.
        """
        return self.A.size + self.B.size

    def merged_weight(self) -> np.ndarray:
        """
        Returns the effective merged weight matrix of shape (d_out, d_in).
        W_merged = W0 + (alpha / r) * (B @ A)
        """
        scaling = self.alpha / self.r
        return self.W0 + scaling * (self.B @ self.A)

    def switch_task(self, A_new: np.ndarray, B_new: np.ndarray) -> None:
        """
        Switches the LoRA adapters for a new task by replacing matrices A and B.
        """
        self.A = A_new
        self.B = B_new
        self.r = A_new.shape[0]