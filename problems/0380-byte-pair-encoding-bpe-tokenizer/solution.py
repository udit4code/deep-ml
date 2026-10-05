from collections import defaultdict

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.

    Args:
        corpus: Dictionary mapping space-separated token sequences to frequencies.
                Example:
                {
                    "l o w </w>": 5,
                    "l o w e r </w>": 2,
                    "n e w </w>": 6
                }

        num_merges: Number of merge operations.

    Returns:
        List of merged token pairs.
    """

    # Convert each string into a tuple of symbols
    vocab = {tuple(word.split()): freq for word, freq in corpus.items()}

    merges = []
    for _ in range(num_merges):
        # Count adjacent symbol pairs
        pair_counts = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
        if not pair_counts:
            break
        # Most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)
        merged_token = "".join(best_pair)
        # Replace occurrences of best_pair
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        vocab = new_vocab

    return merges