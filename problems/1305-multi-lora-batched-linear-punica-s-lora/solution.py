import numpy as np

def multi_lora_linear(x, W, A, B, lora_ids, scales):
    """
    Computes a batched linear layer where different rows in the batch 
    can use completely different LoRA adapters (or no adapter at all).
    """
    
    # Convert inputs into standard NumPy arrays
    # This ensures our inputs are ready for matrix math.
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # If the user passed a single 1D row instead of a batch (2D), 
    # we wrap it into a 2D matrix with 1 row: shape (1, in_features)
    if x.ndim == 1:
        x = x[np.newaxis, :]
        
    batch_size = x.shape[0]
    
    # Calculate the Base Model Output
    # Regular neural network pass: multiplying input 'x' by the frozen weights 'W'.
    # We check the shape of W to ensure the rows and columns align properly.
    if W.shape[0] == x.shape[1]:
        base_out = x @ W
    else:
        base_out = x @ W.T
        
    out_features = base_out.shape[1]
    
    # Figure out how many total adapters we have available in our collection (A)
    num_adapters = len(A) if hasattr(A, '__len__') else A.shape[0]
    
    # Handle Scaling Factors Flexibly
    # Scaling controls how strong the LoRA adjustment is. 
    # Scales can be a single number, a list per adapter, or a list per row.
    if isinstance(scales, (int, float)):
        # If it's a single number, apply that same strength to every row in the batch
        scales_per_row = np.full(batch_size, float(scales), dtype=float)
    else:
        scales = np.asarray(scales, dtype=float)
        if scales.ndim == 0:
            scales_per_row = np.full(batch_size, float(scales))
        elif len(scales) == num_adapters:
            # If the user gave one scale per adapter, map them to the correct rows using lora_ids
            scales_per_row = np.zeros(batch_size, dtype=float)
            for i, uid in enumerate(lora_ids):
                if uid >= 0 and uid < len(scales):
                    scales_per_row[i] = scales[uid]
        elif len(scales) == batch_size:
            # If the user gave a scale for every individual row, use them directly
            scales_per_row = scales
        else:
            # Fallback default if shapes don't match cleanly
            scales_per_row = np.full(batch_size, float(scales[0]) if len(scales) > 0 else 1.0, dtype=float)
            
    # Prepare Output Container for LoRA Updates
    lora_ids = np.asarray(lora_ids, dtype=int)
    
    # Create an array of zeros to hold the low-rank additions
    lora_out = np.zeros((batch_size, out_features), dtype=float)
    
    # Find all unique adapter IDs requested in this batch (e.g., [0, 1, -1])
    unique_ids = np.unique(lora_ids)
    
    # Process Each Adapter Group Separately
    # Instead of looping row-by-row (which is slow), we group rows that share 
    # the exact same adapter and compute them together efficiently in a batch.
    for uid in unique_ids:
        # Negative IDs (like -1) mean "no adapter for this row", so we skip them!
        if uid < 0:
            continue
            
        # Create a True/False mask to find which rows in our batch want this specific adapter ID
        mask = (lora_ids == uid)
        if not np.any(mask):
            continue
            
        # Grab the specific A and B matrices for this adapter ID
        A_k = np.asarray(A[uid], dtype=float)
        B_k = np.asarray(B[uid], dtype=float)
        
        # Pull out only the input rows that belong to this adapter
        x_sub = x[mask] # Shape: (number_of_matching_rows, in_features)
        
        # The Efficient LoRA Math Shortcut
        # Instead of multiplying B @ A first (which creates a massive slow matrix),
        # we calculate: (x_sub @ A_k.T) @ B_k.T. This keeps our math lightning fast!
        if A_k.ndim == 2 and B_k.ndim == 2:
            if A_k.shape[1] == x_sub.shape[1]:
                sub_out = (x_sub @ A_k.T) @ B_k.T
            elif A_k.shape[0] == x_sub.shape[1]:
                sub_out = (x_sub @ A_k) @ B_k.T
            else:
                sub_out = (x_sub @ A_k.T) @ B_k.T
        else:
            sub_out = (x_sub @ A_k.T) @ B_k.T
            
        # Apply Scaling and Store Results
        # Multiply the low-rank output by its corresponding scaling factor
        sub_scales = scales_per_row[mask, np.newaxis]
        lora_out[mask] = sub_scales * sub_out
        
    # Combine Everything and Return
    # Final Output = Base Model Output + Scaled Low-Rank Adapter Output
    return base_out + lora_out