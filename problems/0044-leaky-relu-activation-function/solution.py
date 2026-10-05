import numpy as np 

def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	z = np.asarray(z)
	if z > 0:
		return z 
	else:
		return alpha * z
	
