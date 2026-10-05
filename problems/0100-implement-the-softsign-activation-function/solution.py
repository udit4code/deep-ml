import numpy as np 

def softsign(x: float) -> float:
	"""
	Implements the Softsign activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Softsign of the input	"""
	# Your code here
	soft_sign_val = x /(1 + np.abs(x))
	return np.round(soft_sign_val,4)