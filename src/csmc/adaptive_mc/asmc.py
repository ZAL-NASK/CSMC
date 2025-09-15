"""Matrix completion with adaptive column sampling:
Krishnamurthy, A., & Singh, A. (2014). On the Power of Adaptivity in Matrix Completion and Approximation.
ArXiv, abs/1407.3619."""
from typing import Tuple

import numpy as np


def orth_proj_perp(U):
    n = U.shape[0]
    P = U @ np.linalg.pinv(U.T @ U) @ U.T
    return np.eye(n) - P


def adaptive_mc(M: np.ndarray, p: float) -> Tuple[np.ndarray, float]:
    n_rows, n_cols = M.shape
    m = int(p * n_rows)
    omega = np.random.choice(n_rows, size=m, replace=True)

    M_filled = np.empty_like(M)
    M_filled[:, 0] = M[:, 0]
    U = M[:, 0].reshape(-1, 1)
    counter = 0
    for col_idx in range(1, n_cols):
        U_omega = U[omega, :]
        UToUoinv = np.linalg.pinv(U_omega.T @ U_omega)
        PUo = U_omega @ UToUoinv @ U_omega.T
        projected_column = PUo @ M[omega, col_idx]
        if np.linalg.norm(M[omega, col_idx] - projected_column) > 1e-5:
            M_filled[:, col_idx] = M[:, col_idx]
            M_col_proj = orth_proj_perp(U) @ M[:, col_idx]
            U = np.column_stack((U, M_col_proj))
            omega = np.random.choice(n_rows, size=m, replace=True)
            counter += n_rows
        else:
            M_filled[:, col_idx] = U @ UToUoinv @ U_omega.T @ M[omega, col_idx]
            counter += m
    p_observation = counter / M.size
    return M_filled, p_observation
