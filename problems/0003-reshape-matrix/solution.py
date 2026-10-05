import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	m = len(a)
    n = len(a[0]) if m > 0 else 0
    if m * n != new_shape[0] * new_shape[1]:
        return []  # Return [] if reshaping is not possible
    reshaped_matrix = np.array(a).reshape(new_shape).tolist()
    return reshaped_matrix