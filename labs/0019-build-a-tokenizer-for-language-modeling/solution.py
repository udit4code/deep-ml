from collections import Counter


def train_tokenizer(corpus, vocab_size):
    """
    Train a Byte Pair Encoding (BPE) tokenizer.

    Args:
        corpus: list[str]
        vocab_size: maximum vocabulary size

    Returns:
        encode, decode
    """

    # ------------------------------------------------------------------
    # Initial vocabulary (every unique character)
    # ------------------------------------------------------------------

    alphabet = sorted(set("".join(corpus)))

    token_to_id = {}
    id_to_token = {}

    for i, ch in enumerate(alphabet):
        token_to_id[ch] = i
        id_to_token[i] = ch

    # Every document is initially represented as characters
    sequences = [list(doc) for doc in corpus]

    merges = []

    # ------------------------------------------------------------------
    # Learn merges
    # ------------------------------------------------------------------

    while len(token_to_id) < vocab_size:

        pair_counts = Counter()

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += 1

        if not pair_counts:
            break

        best_pair, freq = pair_counts.most_common(1)[0]

        # No useful merge
        if freq < 2:
            break

        new_token = best_pair[0] + best_pair[1]

        if new_token in token_to_id:
            break

        new_id = len(token_to_id)

        token_to_id[new_token] = new_id
        id_to_token[new_id] = new_token

        merges.append(best_pair)

        # Replace occurrences of the merged pair
        new_sequences = []

        for seq in sequences:

            merged = []
            i = 0

            while i < len(seq):

                if (
                    i < len(seq) - 1
                    and seq[i] == best_pair[0]
                    and seq[i + 1] == best_pair[1]
                ):
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(seq[i])
                    i += 1

            new_sequences.append(merged)

        sequences = new_sequences

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(text):

        tokens = list(text)

        # Apply merges in training order
        for left, right in merges:

            merged_token = left + right

            i = 0
            new_tokens = []

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return [token_to_id[t] for t in tokens]

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(ids):
        return "".join(id_to_token[i] for i in ids)

    return encode, decode