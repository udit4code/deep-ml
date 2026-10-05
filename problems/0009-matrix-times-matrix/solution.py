import numpy as np 

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices.")

    if a.shape[1] != b.shape[0]:
        return -1

    return (a @ b).tolist()