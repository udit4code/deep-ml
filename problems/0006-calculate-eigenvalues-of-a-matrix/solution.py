def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	import numpy as np 
	coeffs = np.poly(matrix)
	# print("coeffs : ", coeffs)
	eigenvalues = np.roots(coeffs) 
	return eigenvalues