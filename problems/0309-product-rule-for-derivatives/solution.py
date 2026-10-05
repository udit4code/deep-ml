import numpy as np

def compute_polynomial_derivative(coeffs: list) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)

    if len(coeffs) <= 1:
        return np.array([0.0])

    # d/dx(a_i x^i) = i * a_i x^(i-1)
    return coeffs[1:] * np.arange(1, len(coeffs))


def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.

    Args:
        f_coeffs: Coefficients in ascending order.
        g_coeffs: Coefficients in ascending order.

    Returns:
        Coefficients of (f*g)' in ascending order,
        rounded to 4 decimal places.
    """
    f = np.asarray(f_coeffs, dtype=float)
    g = np.asarray(g_coeffs, dtype=float)

    f_prime = compute_polynomial_derivative(f)
    g_prime = compute_polynomial_derivative(g)

    # Product rule: (fg)' = f'g + fg'
    result = np.convolve(f_prime, g) + np.convolve(f, g_prime)

    return np.round(result, 4).tolist()
