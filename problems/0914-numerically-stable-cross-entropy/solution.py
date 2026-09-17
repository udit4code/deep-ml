import torch

def cross_entropy(logits, targets):
    N, C = logits.shape 
    # Step 1 : Find the maximum value along the class or target-label dimension
    # In this case, for each row of logits, find the row_max
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]

    # Step 2 : Apply log-sum-exp stabilization trick 
    exp = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp, dim=-1, keepdim=True)
    log_sum_exp = max_logits + torch.log(sum_exp)

    # Step 3 : Compute log probabilities : logits - log_sum_exp
    log_probs = logits - log_sum_exp

    # Step 4 : Gather log probabilities of the target classes 
    gathered_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Return the average negative log likelihood
    return -gathered_log_probs.mean()