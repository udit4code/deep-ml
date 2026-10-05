import numpy as np
from typing import Optional, Iterator, Tuple, Union

class BatchIterator:

    __slots__ = (
        "X",
        "y",
        "batch_size",
        "n_samples",
        "_has_labels",
        "_index",
    )

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 64,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = X.shape[0]
        self._has_labels = y is not None
        self._index = 0  # iteration pointer

    def __iter__(self) -> Iterator[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.n_samples:
            raise StopIter