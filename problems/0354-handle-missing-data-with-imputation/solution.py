import numpy as np
from abc import ABC, abstractmethod
from collections import Counter


class ImputationStrategy(ABC):
    @abstractmethod
    def apply_imputation(self, data: np.ndarray) -> np.ndarray:
        pass


class MeanStrategy(ImputationStrategy):
    def apply_imputation(self, data: np.ndarray) -> np.ndarray:
        """
        Replace NaN values column-wise using mean.
        """
        result = data.copy()

        for col in range(result.shape[1]):
            column_data = result[:, col]

            # Compute mean ignoring NaN
            mean_value = np.nanmean(column_data)

            # Replace NaN with mean
            nan_mask = np.isnan(column_data)
            column_data[nan_mask] = mean_value

        return result


class MedianStrategy(ImputationStrategy):
    def apply_imputation(self, data: np.ndarray) -> np.ndarray:
        """
        Replace NaN values column-wise using median.
        """
        result = data.copy()

        for col in range(result.shape[1]):
            column_data = result[:, col]

            # Compute median ignoring NaN
            median_value = np.nanmedian(column_data)

            # Replace NaN with median
            nan_mask = np.isnan(column_data)
            column_data[nan_mask] = median_value

        return result


class ModeStrategy(ImputationStrategy):
    def apply_imputation(self, data: np.ndarray) -> np.ndarray:
        """
        Replace NaN values column-wise using mode.
        """
        result = data.copy()

        for col in range(result.shape[1]):
            column_data = result[:, col]

            # Remove NaNs before mode calculation
            non_nan_values = column_data[~np.isnan(column_data)]

            if len(non_nan_values) == 0:
                continue

            # Find mode
            counts = Counter(non_nan_values)
            mode_value = counts.most_common(1)[0][0]

            # Replace NaNs with mode
            nan_mask = np.isnan(column_data)
            column_data[nan_mask] = mode_value

        return result


def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.

    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'

    Returns:
        2D numpy array with missing values imputed
    """

    strategies = {
        "mean": MeanStrategy(),
        "median": MedianStrategy(),
        "mode": ModeStrategy()
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. "
            f"Choose from {list(strategies.keys())}"
        )

    return strategies[strategy].apply_imputation(data)


# Example Usage
# if __name__ == "__main__":
#     data = np.array([
#         [1, 2, np.nan],
#         [4, np.nan, 6],
#         [7, 8, 9]
#     ], dtype=float)

#     print("Original Data:")
#     print(data)

#     print("\nMean Imputation:")
#     print(impute_missing_data(data, strategy="mean"))

#     print("\nMedian Imputation:")
#     print(impute_missing_data(data, strategy="median"))

#     print("\nMode Imputation:")
#     print(impute_missing_data(data, strategy="mode"))