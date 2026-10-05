import math

def rotation_layer(X, angle):
    return rotate_points(X, angle)



def rotate_points(X: list[list[float]], angle: float) -> list[list[float]]:
    """
    Rotates a list of 2D points by a given angle in radians.
    
    Rotation Matrix R:
        [ cos(θ)  -sin(θ) ]
        [ sin(θ)   cos(θ) ]
        
    Properties:
    - Orthonormal columns: ||col1||^2 = ||col2||^2 = 1, col1 · col2 = 0
    - Determinant: cos²(θ) - (-sin²(θ)) = cos²(θ) + sin²(θ) = +1
    """
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    
    # R = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
    rotated_points = []
    for x, y in X:
        x_prime = x * cos_theta - y * sin_theta
        y_prime = x * sin_theta + y * cos_theta
        rotated_points.append([x_prime, y_prime])
        
    return rotated_points