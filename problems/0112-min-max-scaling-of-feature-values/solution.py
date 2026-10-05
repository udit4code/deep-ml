def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    if not x:
        raise ValueError("Input sequence must not be empty")

    min_val = min(x)
    max_val = max(x)
    range_val = max_val - min_val

    # Edge case: all values are identical
    if range_val == 0:
        # Convention: map all values to 0.0
        # (alternatives could be 0.5 or 1.0 depending on downstream needs)
        return [0.0 for _ in x]

    return [(value - min_val) / range_val for value in x]