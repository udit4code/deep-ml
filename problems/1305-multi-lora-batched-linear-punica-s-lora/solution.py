import numpy as np

def multi_lora_linear(x, W, A, B, lora_ids, scales):
    """
    Batched linear layer with per-row LoRA adapters and flexible scaling support.
    """
    
    # Step 1: Convert inputs into NumPy arrays
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Ensure x is 2D: (batch_size, in_features)
    if x.ndim == 1:
        x = x[np.newaxis, :]
        
    batch_size = x.shape[0]
    
    # Step 2: Compute the base model output (x @ W)
    if W.shape[0] == x.shape[1]:
        base_out = x @ W
    else:
        base_out = x @ W.T
        
    out_features = base_out.shape[1]
    num_adapters = len(A) if hasattr(A, '__len__') else A.shape[0]
    
    # Step 3: Handle scales flexibly (supports scalar, per-adapter, or per-row lists)
    if isinstance(scales, (int, float)):
        # If a single scalar is given, broadcast it to every row
        scales_per_row = np.full(batch_size, float(scales), dtype=float)
    else:
        scales = np.asarray(scales, dtype=float)
        if scales.ndim == 0:
            scales_per_row = np.full(batch_size, float(scales))
        elif len(scales) == num_adapters:
            # If scales match the number of adapters, map them per-adapter using lora_ids
            scales_per_row = np.zeros(batch_size, dtype=float)
            for i, uid in enumerate(lora_ids):
                if uid >= 0 and uid < len(scales):
                    scales_per_row[i] = scales[uid]
        elif len(scales) == batch_size:
            # If scales match the batch size, use them per-row directly
            scales_per_row = scales
        else:
            scales_per_row = np.full(batch_size, float(scales[0]) if len(scales) > 0 else 1.0, dtype=float)
            
    # Step 4: Prepare output array for low-rank updates
    lora_ids = np.asarray(lora_ids, dtype=int)
    lora_out = np.zeros((batch_size, out_features), dtype=float)
    
    # Step 5: Group and compute updates by unique adapter ID
    unique_ids = np.unique(lora_ids)
    for uid in unique_ids:
        # Negative IDs (like -1) mean "no adapter", skip them
        if uid < 0:
            continue
            
        mask = (lora_ids == uid)
        if not np.any(mask):
            continue
            
        A_k = np.asarray(A[uid], dtype=float)
        B_k = np.asarray(B[uid], dtype=float)
        
        x_sub = x[mask] # Matching input rows
        
        # Step 6: Efficient low-rank forward pass: (x_sub @ A_k.T) @ B_k.T
        if A_k.ndim == 2 and B_k.ndim == 2:
            if A_k.shape[1] == x_sub.shape[1]:
                sub_out = (x_sub @ A_k.T) @ B_k.T
            elif A_k.shape[0] == x_sub.shape[1]:
                sub_out = (x_sub @ A_k) @ B_k.T
            else:
                sub_out = (x_sub @ A_k.T) @ B_k.T
        else:
            sub_out = (x_sub @ A_k.T) @ B_k.T
            
        # Step 7: Apply the corresponding scaling factor
        sub_scales = scales_per_row[mask, np.newaxis]
        lora_out[mask] = sub_scales * sub_out
        
    # Step 8: Combine base output and low-rank adapter output
    return base_out + lora_out