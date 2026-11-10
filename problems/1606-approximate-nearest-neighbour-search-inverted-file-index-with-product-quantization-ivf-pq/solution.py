import numpy as np


def _cdist_sq(A, B):
  """Computes pairwise squared Euclidean distances between A (n, d) and B (m, d)

  from first principles using the expansion:
      ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
  This vectorizes distance calculations entirely in NumPy to avoid slow Python
  loops.
  """
  # ||a||^2 shape: (n, 1)
  A_norm = (A ** 2).sum(axis=1, keepdims=True)
  # ||b||^2 shape: (1, m)
  B_norm = (B ** 2).sum(axis=1, keepdims=True)
  # 2 * a . b shape: (n, m)
  dot_prod = np.dot(A, B.T)
  # Broadcasted result: (n, m)
  return A_norm + B_norm.T - 2 * dot_prod


def lloyd(X, k, iters):
  """K-means clustering using Lloyd's algorithm from first principles.

  Principles:
    1. Expectation (Assignment): Assign each data point to its closest
    centroid using Voronoi partitioning (minimum distance).
    2. Maximization (Update): Recompute each centroid as the mean of all
    points assigned to it.
  """
  n, d = X.shape
  # Initialize centroids using the first k rows of the dataset as a deterministic seed
  centroids = X[:k].copy()

  for _ in range(iters):
    # Step 1: Compute distances from all points to all current centroids
    dists = _cdist_sq(X, centroids)

    # Assign each point to the index of the nearest centroid (Voronoi regions)
    labels = np.argmin(dists, axis=1)

    # Step 2: Update centroids as the mean of assigned points
    new_centroids = np.empty_like(centroids)
    for j in range(k):
      mask = labels == j
      if np.any(mask):
        new_centroids[j] = X[mask].mean(axis=0)
      else:
        # Prevent empty cluster collapse by keeping the previous centroid position
        new_centroids[j] = centroids[j]
    centroids = new_centroids

  return centroids


def build_ivf(X, centroids):
  """Builds an Inverted File Index (IVF) from first principles.

  Principles:
    Partition the vector space into Voronoi cells defined by the centroids.
    An inverted list maps each centroid ID to an ascending list of database
    vector indices belonging to that cell's Voronoi region.
  """
  # Find the closest centroid for every vector in the dataset
  dists = _cdist_sq(X, centroids)
  labels = np.argmin(dists, axis=1)
  k_c = centroids.shape[0]

  ivf = []
  for j in range(k_c):
    # Gather indices of all vectors falling into the j-th Voronoi cell
    indices = np.where(labels == j)[0]
    # Ensure indices are sorted in ascending order for cache-friendly access
    indices.sort()
    ivf.append(indices.tolist())

  return ivf


def train_pq(X, m, k, iters):
  """Trains Product Quantization (PQ) codebooks from first principles.

  Principles:
    Break the high-dimensional space into m independent orthogonal subspaces
    of dimension d/m (Cartesian product decomposition). Train independent
    k-means codebooks on each subspace to capture local distributions efficiently.
  """
  n, d = X.shape
  d_sub = d // m  # Dimension of each subspace
  codebooks = np.empty((m, k, d_sub))

  for j in range(m):
    # Slice out the j-th subspace columns for all vectors
    X_sub = X[:, j * d_sub : (j + 1) * d_sub]
    # Run Lloyd's k-means to find k centroids (codewords) in this subspace
    codebooks[j] = lloyd(X_sub, k, iters)

  return codebooks


def encode_pq(X, codebooks):
  """Encodes raw vectors into Product Quantization compressed codes.

  Principles:
    Replace each vector component (subspace) with the index of its nearest
    codeword from the trained subspace codebook, compressing a float vector
    of size d into m small integers.
  """
  m, k, d_sub = codebooks.shape
  n = X.shape[0]
  codes = np.empty((n, m), dtype=np.int32)

  for j in range(m):
    # Extract the j-th subspace of all vectors
    X_sub = X[:, j * d_sub : (j + 1) * d_sub]
    # Compute distances to all codewords in this subspace codebook
    dists = _cdist_sq(X_sub, codebooks[j])
    # Store the index of the closest codeword
    codes[:, j] = np.argmin(dists, axis=1)

  return codes


def adc_distances(q, codes, codebooks):
  """Computes Asymmetric Distance Computation (ADC) from first principles.

  Principles:
    Keep the query vector uncompressed (floating point), but use the quantized
    codes for database vectors. Precompute a distance lookup table of shape
    (m, k) between query sub-vectors and all codewords. The distance to any
    encoded vector is then computed via direct table lookup and summation
    across subspaces without decompression.
  """
  m, k, d_sub = codebooks.shape
  n, _ = codes.shape

  # Step 1: Build distance lookup table of shape (m, k)
  # table[j, c] = squared distance between query sub-vector j and codeword c
  table = np.empty((m, k))
  for j in range(m):
    q_sub = q[j * d_sub : (j + 1) * d_sub]
    diff = codebooks[j] - q_sub
    table[j] = np.sum(diff ** 2, axis=1)

  # Step 2: Fast vectorized lookup and summation across subspaces
  # table[np.arange(m)[:, None], codes.T] gathers the distance values for each code,
  # and sum(axis=0) accumulates distances across all m subspaces.
  distances = np.sum(table[np.arange(m)[:, None], codes.T], axis=0)
  return distances


def ivf_pq_search(q, codes, codebooks, centroids, ivf, nprobe, k):
  """Performs Approximate Nearest Neighbor search using IVF-PQ.

  Principles:
    1. Coarse Pruning: Locate the nprobe closest Voronoi centroids to the query.
    2. Candidate Gathering: Collect only the database vector indices stored in
       those nprobe cells (skipping the rest of the dataset).
    3. Fine Scoring: Compute Asymmetric Distances (ADC) for candidate vectors.
    4. Top-k Selection: Return the top-k smallest distance matches.
  """
  # Step 1: Find the nprobe nearest IVF centroids to the query
  dists_centroids = np.sum((centroids - q) ** 2, axis=1)
  nearest_cells = np.argsort(dists_centroids)[:nprobe]

  # Step 2: Gather candidate database vector IDs from selected inverted lists
  candidate_ids = []
  for cell_id in nearest_cells:
    candidate_ids.extend(ivf[cell_id])

  if not candidate_ids:
    return np.array([], dtype=np.int32)

  # Remove duplicate candidates (in case vectors fall near Voronoi boundaries)
  candidate_ids = np.unique(candidate_ids)

  # Step 3: Compute ADC distances strictly for the candidate subset
  sub_codes = codes[candidate_ids]
  cand_dists = adc_distances(q, sub_codes, codebooks)

  # Step 4: Efficiently retrieve top-k global indices using partial sorting
  if len(candidate_ids) <= k:
    sorted_idx = np.argsort(cand_dists)
  else:
    # Partition array so the smallest k elements are placed first (O(N) average)
    part_idx = np.argpartition(cand_dists, k)[:k]
    sorted_idx = part_idx[np.argsort(cand_dists[part_idx])]

  return candidate_ids[sorted_idx]


def recall_at_k(found, exact):
  """Calculates Recall@K from first principles.

  Principles:
    Measures retrieval accuracy by finding the intersection size between
    the approximate search results (`found`) and the exact ground truth
    nearest neighbors (`exact`), divided by $k$.
  """
  set_found = set(found)
  set_exact = set(exact)
  if not set_exact:
    return 0.0

  # Cardinality of the intersection of retrieved and true ground-truth sets
  intersection = len(set_found & set_exact)
  recall = intersection / len(set_exact)

  return round(float(recall), 4)
