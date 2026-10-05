def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    m = len(a)
    n = len(a[0]) if m > 0 else 0
    transpose = [[0] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            transpose[j][i] = a[i][j]
    return transpose