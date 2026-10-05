import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.array(X, dtype=np.float64)
	y = np.array(y, dtype=np.float64)

	A = X.T @ X 
	b = X.T @ y 

	return np.linalg.solve(A, b)