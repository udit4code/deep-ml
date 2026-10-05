def calculate_covariance_matrix_v1(vectors: list[list[float]]) -> list[list[float]]:
	# We will not use numpy for this implementation.
    n = len(vectors)
    result = [ ]
    if n > 0:
        m = len(vectors[0])
        # It means that we have n features and each feature has m observations
        feature_means = [ ]
        for feature in vectors:
            result.append([0.0] * n)
            feature_total = sum(feature)
            feature_mean = feature_total / m 
            feature_means.append(feature_mean)
        for idx in range(0, n, 1):
            for jdx in range(idx, n, 1):
                cov = 0.0
                for kdx in range(m):
                    cov += (vectors[idx][kdx] - feature_means[idx]) * (vectors[jdx][kdx] - feature_means[jdx])
                cov /= (m - 1)
                result[idx][jdx] = cov
                result[jdx][idx] = cov # As covariance matrix is symmetric
    return result

def calculate_covariance_matrix(vectors: list[list[float]]) -