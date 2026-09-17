import numpy as np

def gpt_feedforward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """GPT-style position-wise feedforward block.

    Args:
        x:  array of shape (batch_size, num_tokens, emb_dim)
        W1: array of shape (emb_dim, 4*emb_dim)
        b1: array of shape (4*emb_dim,)
        W2: array of shape (4*emb_dim, emb_dim)
        b2: array of shape (emb_dim,)

    Returns:
        Array of shape (batch_size, num_tokens, emb_dim).
    """
    # Step 1 : First linear projection (Upsample to 4 * emb_dim)
    # x is (B, T, D), W1 is (D, 4D) -> hidden is (B, T, 4D)
    hidden = x @ W1 + b1
    
    # Step 2 : Apply GELU activation function (approximate version)
    # GELU(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    gelu_act = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * np.power(hidden, 3))))
    
    # 3. Second linear projection (Downsample back to emb_dim)
    # gelu_act is (B, T, 4D), W2 is (4D, D) -> output is (B, T, D)
    output = gelu_act @ W2 + b2
    
    return output
