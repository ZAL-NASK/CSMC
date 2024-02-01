import numpy as np
from numpy import ndarray


def subspace_coherence(U: np.ndarray) -> float:
    """Calculate coherence of the linear subspace."""
    r = np.linalg.matrix_rank(U)
    n = U.shape[0]
    subspace_coherence = -1
    for i in range(n):
        eye = np.zeros((n))
        eye[i] = 1
        # np.linalg.norm(U.reshape(3, 1) @ U.reshape(1, 3)
        PU = U @ np.linalg.inv(U.T @ U) @ U.T
        coherence = np.linalg.norm(np.dot(PU, eye)) ** 2 * (float(n) / r)
        if coherence > subspace_coherence:
            subspace_coherence = coherence
    return subspace_coherence


def matrix_coherence(X: ndarray) -> float:
    """Calculate matrix coherence."""
    U, s, VT = np.linalg.svd(X, full_matrices=True, compute_uv=True)
    r = np.linalg.matrix_rank(X)
    U = U[:, :r]
    V = VT[:r, :].T
    c1 = subspace_coherence(U)
    c2 = subspace_coherence(V)
    return max(c1, c2)
