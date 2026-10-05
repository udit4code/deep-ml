import numpy as np
from collections import Counter


def get_ngrams(tokens: list[str], n: int) -> Counter:
    """
    Generate n-gram counts from a token list.
    """
    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]
    return Counter(ngrams)


def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences.

    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)

    Returns:
        BLEU score between 0 and 1
    """

    # Edge case: empty candidate
    if len(candidate) == 0:
        return 0.0

    candidate_len = len(candidate)

    # ------------------------------------------------------------
    # Choose reference length:
    # closest length to candidate
    # if tie -> choose shorter
    # ------------------------------------------------------------
    ref_lens = [len(ref) for ref in references]

    best_ref_len = min(
        ref_lens,
        key=lambda r: (abs(r - candidate_len), r)
    )

    precisions = []

    # ------------------------------------------------------------
    # Modified n-gram precision
    # ------------------------------------------------------------
    for n in range(1, max_n + 1):

        # Not enough tokens for this n-gram order
        total_candidate_ngrams = len(candidate) - n + 1

        if total_candidate_ngrams <= 0:
            return 0.0

        candidate_ngrams = get_ngrams(candidate, n)

        # Maximum reference counts for clipping
        max_ref_counts = Counter()

        for ref in references:
            ref_ngrams = get_ngrams(ref, n)

            for ng, count in ref_ngrams.items():
                max_ref_counts[ng] = max(
                    max_ref_counts[ng],
                    count
                )

        # Clipped counts
        clipped_count = 0

        for ng, count in candidate_ngrams.items():
            clipped_count += min(count, max_ref_counts.get(ng, 0))

        precision = clipped_count / total_candidate_ngrams

        # BLEU requirement:
        # return 0 if ANY precision is zero
        if precision == 0:
            return 0.0

        precisions.append(precision)

    # ------------------------------------------------------------
    # Geometric mean of precisions
    # ------------------------------------------------------------
    log_precisions = [np.log(p) for p in precisions]
    geo_mean = np.exp(np.mean(log_precisions))

    # ------------------------------------------------------------
    # Brevity penalty
    # ------------------------------------------------------------
    if candidate_len > best_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - best_ref_len / candidate_len)

    bleu = brevity_penalty * geo_mean

    return float(bleu)