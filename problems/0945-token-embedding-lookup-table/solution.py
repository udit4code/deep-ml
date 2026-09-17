import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0) -> list:
    """
    Build a random embedding table of shape (vocab_size, embed_dim) using
    np.random.default_rng(seed).standard_normal(...), then return the rows
    corresponding to token_ids as a nested list.
    """
    # Step 1 : Initialize the generator with the provided seed
    rng = np.random.default_rng(seed)
    
    # Step 2 : Generate the full embedding matrix
    embedding_table = rng.standard_normal((vocab_size, embed_dim))
    
    # Step 3 : Look up the rows for the given token IDs and convert to a nested list
    return embedding_table[token_ids].tolist()