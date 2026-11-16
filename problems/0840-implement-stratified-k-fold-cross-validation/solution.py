import numpy as np

def stratified_kfold_indices(y, n_splits) -> list:
    """
    Generate train/test indices for stratified K-fold cross-validation.

    Args:
        y: 1D array-like of integer class labels
        n_splits: number of folds

    Returns:
        A list of [train_indices, test_indices] pairs (as Python lists), one per fold.
    """
    # Step 1: Convert y to a numpy array and get basic properties
    y = np.asarray(y)
    n_samples = len(y)
    classes = np.unique(y)
    
    # Step 2: Group sample indices by their respective class labels
    class_indices = {}
    for cls in classes:
        class_indices[cls] = np.where(y == cls)[0]
        
    # Step 3: Split each class's indices into n_splits roughly equal parts
    fold_class_splits = {}
    for cls, indices in class_indices.items():
        fold_class_splits[cls] = np.array_split(indices, n_splits)
        
    # Step 4: Construct train/test index pairs for each fold
    folds = []
    for i in range(n_splits):
        test_indices = []
        
        # Step 4.1 : Collect the i-th partition from every class to form the test set for fold i
        for cls in classes:
            test_indices.extend(fold_class_splits[cls][i])
            
        # Step 4.2 : Convert test indices to a standard Python list of integers
        test_indices = sorted([int(idx) for idx in test_indices])
        
        # Step 4.3 : Define train indices as all sample indices not included in the test set
        mask = np.ones(n_samples, dtype=bool)
        mask[test_indices] = False
        
        # Step 4.4 Convert train indices to a standard Python list of integers using .tolist()
        train_indices = sorted(np.where(mask)[0].tolist())
        
        # Step 4.5 : Append the [train_indices, test_indices] Python list pair for this fold
        folds.append([train_indices, test_indices])
        
    return folds