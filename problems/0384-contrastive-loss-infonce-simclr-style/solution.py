import numpy as np

def contrastive_loss(embeddings: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    
    Args:
        embeddings: Array of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).
    
    Returns:
        The mean contrastive loss as a float.
    """
    # Step 1: L2 Normalize embeddings to the unit hypersphere
    # Intuition: SimCLR typically computes cosine similarity via dot products 
    # of L2-normalized vectors to bound similarities between -1 and 1.
    # Shape Evolution:
    #   embeddings : (2N, d)
    #   norms      : (2N, 1)
    #   embeddings : (2N, d) [normalized rows]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-8, a_max=None)
    
    total_samples, d = embeddings.shape
    
    # Step 2: Compute pairwise similarity matrix (logits)
    # Shape Evolution:
    #   embeddings   : (2N, d)
    #   embeddings.T : (d, 2N)
    #   logits       : (2N, 2N) [pairwise similarity matrix]
    #
    # Intuition:
    # - Entry (i, j) represents the similarity between view i and view j.
    # - Division by 'temperature' controls the sharpness of the probability distribution.
    logits = (embeddings @ embeddings.T) / temperature

    # Step 3: Mask out self-similarity (diagonal elements)
    # Intuition: A sample should not be contrasted against itself in the denominator 
    # of the softmax computation, so we set the diagonal to negative infinity (-inf).
    np.fill_diagonal(logits, -float('inf'))

    # Step 4: Define positive pair indices using bitwise XOR
    # Intuition: Since consecutive pairs (2i, 2i+1) are positive:
    # - Index 0 pairs with 1 (0 ^ 1 = 1)
    # - Index 1 pairs with 0 (1 ^ 1 = 0)
    # - Index 2 pairs with 3 (2 ^ 1 = 3), etc.
    # Shape Evolution:
    #   pos_indices : (2N,)
    pos_indices = np.arange(total_samples) ^ 1

    # Step 5: Compute Numerically Stable Cross-Entropy Loss (LogSumExp trick)
    # Shape Evolution:
    #   logits      : (2N, 2N)
    #   max_logits  : (2N, 1) [Max value per row for stability]
    #   exp_logits  : (2N, 2N) [Exponentiated shifted logits]
    #   log_sum_exp : (2N, 1) [Log of sum of exponentials along each row]
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    log_sum_exp = max_logits + np.log(np.sum(exp_logits, axis=1, keepdims=True))

    # Step 6: Extract positive pair logits and compute row-wise loss
    # Shape Evolution:
    #   pos_logits   : (2N,) [Extracted similarity of each view to its positive counterpart]
    #   loss_per_row : (2N,) [Scalar loss for each of the 2N augmented views]
    pos_logits = logits[np.arange(total_samples), pos_indices]
    loss_per_row = -pos_logits + log_sum_exp.squeeze()

    # Step 7: Average loss over all 2N views
    total_loss = np.mean(loss_per_row)

    return float(total_loss)