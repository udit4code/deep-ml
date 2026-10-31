def exp_weighted_average(Q1: float, rewards: list, alpha: float) -> float:
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards computed directly from the formula.
    """
    k = len(rewards)
    if k == 0:
        return float(Q1)
    
    # Step 1 : Compute the decaying contribution of the initial estimate Q1
    q1_term = ((1 - alpha) ** k) * Q1
    
    # Step 2 : Compute the weighted sum of all rewards directly from the formula
    rewards_term = 0.0
    for j, r in enumerate(rewards):
        # More recent rewards (higher index j) have a smaller exponent, 
        # meaning less decay and higher weight.
        exponent = k - 1 - j
        rewards_term += alpha * ((1 - alpha) ** exponent) * r
        
    return q1_term + rewards_term