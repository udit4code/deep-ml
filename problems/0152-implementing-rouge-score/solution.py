# Implement your function below.

from collections import Counter

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 precision, recall and F1.
    """

    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()

    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)

    overlap = 0

    for token, ref_count in reference_counts.items():
        overlap += min(
            ref_count,
            candidate_counts.get(token, 0)
        )

    precision = (
        overlap / len(candidate_tokens)
        if candidate_tokens
        else 0.0
    )

    recall = (
        overlap / len(reference_tokens)
        if reference_tokens
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# def rouge_1_score(reference: str, candidate: str) -> dict:
#     """
#     Compute ROUGE-1 score between reference and candidate texts.
    
#     Returns a dictionary with precision, recall, and f1.
#     """
#     # Your code here
#     pass