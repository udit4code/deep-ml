import numpy as np 
import math 

def sigmoid(x: float) -> float:
	val = None 
	if x >= 0:
		# Prevents overflow from large negative numbers
        val = 1.0 / (1.0 + np.exp(-x))
    else:
        # Prevents overflow from large positive exponent values
        val = math.exp(x) / (1.0 + np.exp(x))
	return float(val)

def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	# Your code here
	return x * sigmoid(x)