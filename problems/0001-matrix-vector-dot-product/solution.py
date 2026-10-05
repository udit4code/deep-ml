def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if not a:
        return []   
    n = len(a[0])
    # Validate dimensions : a(m x n) and b(n x 1)
    if len(b) != n:
        return -1
    result = []
    for row in a:
        dot_product = 0
        for j in range(n):
            dot_product += row[j] * b[j]
        result.append(dot_product)
    return result