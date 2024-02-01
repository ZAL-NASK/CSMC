"""Matrix completion with nuclear norm minimization using SDP solvers."""
import cvxpy as cp
import numpy as np


class NuclearNormMin:
    """Class for completing matrix using nuclear norm minimization as SDP."""

    def __init__(self, M_incomplete: np.ndarray, solver: str = "SCS") -> None:
        self.solver = solver

    def fit_transform(self, M_incomplete: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """Matrix completion logic."""
        return self.nn_complete(M_incomplete, missing_mask)

    def nn_complete(self, M_incomplete: np.ndarray, missing_mask: np.ndarray | None) -> np.ndarray:
        """Fill M_incomplete with nuclear norm minimization."""
        if missing_mask is None:
            missing_mask = np.isnan(M_incomplete)
        M_incomplete[missing_mask] = 0
        M_filled = cp.Variable(M_incomplete.shape)
        prob = cp.Problem(cp.Minimize(cp.norm(M_filled, p="nuc")),
                          [cp.multiply(~missing_mask, M_filled) == cp.multiply(~missing_mask, M_incomplete)])
        prob.solve(solver=cp.SCS, verbose=False)
        return M_filled.value
