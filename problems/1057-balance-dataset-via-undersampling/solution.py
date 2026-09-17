def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    # Step 1 : Get the count of the minority class 
    freq_map = {}
    min_freq = float("inf")
    for example in data:
        sample, label = example
        if label not in freq_map:
            freq_map[label] = 0
        freq_map[label] += 1
    for label_key, label_freq in freq_map.items():
        min_freq = min(min_freq, label_freq)
    # Step 2 : Trim the data list  
    undersampled_data = [ ]
    freq_map = {label : 0 for label in freq_map.keys()}
    for example in data:
        sample, label = example
        if freq_map[label] + 1 <= min_freq:
            freq_map[label] += 1
            undersampled_data.append(example)
        else:
            freq_map[label] += 1
        
    return undersampled_data
