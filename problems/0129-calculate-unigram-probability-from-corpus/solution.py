def unigram_probability(corpus: str, word: str) -> float:
    unigrams = corpus.split(" ")
    frequency_map = {}
    for unigram in unigrams:
        if unigram not in frequency_map:
            frequency_map[unigram] = 1
        else:
            frequency_map[unigram] += 1
    return frequency_map[word] / len(unigrams)