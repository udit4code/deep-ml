import numpy as np

import numpy as np

def covariance_from_joint_pmf_optimized(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Optimized computation of Covariance of X and Y from their joint PMF 
    using matrix multiplication instead of outer product allocation.
    """
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    
    # Step 1: Compute marginal probabilities
    p_x = joint_pmf.sum(axis=1)
    p_y = joint_pmf.sum(axis=0)
    
    # Step 2: Compute expected values E[X] and E[Y]
    e_x = np.dot(x_arr, p_x)
    e_y = np.dot(y_arr, p_y)
    
    # Step 3: Compute E[XY] efficiently via matrix-vector multiplication
    # Equivalent to: sum_{i,j} x_i * y_j * P(X=x_i, Y=y_j)
    # This avoids allocating a full M x N outer product matrix in memory.
    e_xy = np.dot(x_arr, np.dot(joint_pmf, y_arr))
    
    # Step 4: Covariance formula
    return float(e_xy - (e_x * e_y))
    
def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    
    Returns:
        Covariance of X and Y as a float
    """
    x_arr = np.array(x_values, dtype=float)
    y_arr = np.array(y_values, dtype=float)
    
    # Step 1: Compute marginal probabilities P(X=x_i) and P(Y=y_j)
    p_x = np.sum(joint_pmf, axis=1)  # Sum across columns for each row
    p_y = np.sum(joint_pmf, axis=0)  # Sum across rows for each column
    
    # Step 2: Compute expected values E[X] and E[Y]
    e_x = np.sum(x_arr * p_x)
    e_y = np.sum(y_arr * p_y)
    
    # Step 3: Compute expected value E[XY] using the outer product of x and y values
    xy_matrix = np.outer(x_arr, y_arr)
    e_xy = np.sum(xy_matrix * joint_pmf)
    
    # Step 4: Compute Covariance using the formula Cov(X, Y) = E[XY] - E[X]E[Y]
    covariance = e_xy - (e_x * e_y)
    
    return float(covariance)