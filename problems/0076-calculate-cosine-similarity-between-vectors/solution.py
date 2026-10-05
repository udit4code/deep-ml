
import numpy as np

def cosine_similarity(v1, v2):
	# Implement your code here
	x = np.array(v1)
    y = np.array(v2)
    similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return round(similarity, 3)