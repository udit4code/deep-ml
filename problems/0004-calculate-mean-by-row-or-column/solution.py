import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    m = np.array(matrix)
    row_count, col_count = m.shape
    if mode == 'row':
        return m.mean(axis=1)
    elif mode == 'column':
        return m.mean(axis=0)
    else:
        raise Exception(f"Invalid mode {mode}")
	return [ ]

