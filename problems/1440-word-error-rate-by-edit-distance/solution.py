import re


def normalize_text(s: str) -> str:
    """
    Lowercase, remove punctuation (. , ! ? ; : " '), 
    collapse runs of whitespace, and strip leading/trailing spaces.
    """
    s = s.lower()
    # Remove specified punctuation characters: . , ! ? ; : " '
    s = s.translate(str.maketrans("", "", '.,!?:;"\''))
    # Collapse runs of whitespace and strip
    return re.sub(r"\s+", " ", s).strip()


def edit_ops(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int, int]:
    """
    Computes Levenshtein alignment returning (substitutions, deletions, insertions).
    
    When backtracing ties occur, preference order is:
    Substitution > Deletion > Insertion.
    """
    R, H = len(ref_words), len(hyp_words)

    # dp[i][j] = min edit distance between ref_words[:i] and hyp_words[:j]
    dp = [[0] * (H + 1) for _ in range(R + 1)]

    for i in range(R + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j

    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,  # Substitution / Match
                dp[i - 1][j] + 1,         # Deletion
                dp[i][j - 1] + 1          # Insertion
            )

    # Backtrace from dp[R][H] to dp[0][0]
    i, j = R, H
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        # 1. Prefer Substitution / Match if valid
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 1:
                    substitutions += 1
                i -= 1
                j -= 1
                continue

        # 2. Prefer Deletion if valid
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue

        # 3. Prefer Insertion if valid
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue

    return (substitutions, deletions, insertions)


def wer(ref: str, hyp: str) -> float:
    """
    Compute Word Error Rate (WER) between a single reference and hypothesis string.
    """
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    S, D, I = edit_ops(ref_words, hyp_words)
    N = len(ref_words)

    return round((S + D + I) / max(1, N), 4)


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """
    Compute total Word Error Rate across a corpus of reference and hypothesis pairs.
    """
    total_errors = 0
    total_ref_words = 0

    for ref, hyp in zip(refs, hyps):
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)

        ref_words = ref_norm.split() if ref_norm else []
        hyp_words = hyp_norm.split() if hyp_norm else []

        S, D, I = edit_ops(ref_words, hyp_words)
        total_errors += (S + D + I)
        total_ref_words += len(ref_words)

    return round(total_errors / max(1, total_ref_words), 4)
