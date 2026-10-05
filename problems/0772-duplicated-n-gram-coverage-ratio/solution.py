from collections import Counter

def dup_ngram_ratio(text: str, n: int) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

    freq = Counter(ngrams)

    repeated_count = sum(
        1
        for ng in ngrams
        if freq[ng] > 1
    )

    return repeated_count / len(ngrams)