import numpy as np

def make_diagonal(x):
	n = len(x)
    diagonal_matrix = [[0] * n for _ in range(n)]
    for idx in range(n):
        diagonal_matrix[idx][idx] = x[idx]
    return diagonal_matrix