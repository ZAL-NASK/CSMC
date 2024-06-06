"""Matrix completion with Frank-Wolfe (Conditional Gradient Method)."""
from abc import abstractmethod
from typing import Any, Generic

import fbpca
import numpy as np
import torch
from numpy import floating, ndarray
from torch import Tensor

from csmc.settings import T, LOGGER, UNSUPPORTED_MSG
from scipy.sparse.linalg import svds

class CGMBase(Generic[T]):
    """Class for completing matrix using  Frank-Wolfe (Conditional Gradient Method)."""
    def __init__(self, X :T, lambda_: float = 0.01, threshold: float = 0.0000001, max_iter: int = 1000,
                 max_rank: int | None = None, numlib: str = "numpy", **kwargs) -> None:
        self.numlib = numlib
        self.lambda_ = lambda_
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, M: T, missing_mask: T,
            Z_init: T | None = None) -> T:
        """Complete matrix."""
        return self.solve(M, missing_mask, Z_init=Z_init)

    def fit_transform(self, M: T, missing_mask: T, Z_init: T | None =None) -> T:
        """Complete matrix."""
        return self.fit(M, missing_mask, Z_init=Z_init)


    def _get_weights(self, iter) -> tuple:
        """Return weights"""
        q = 2/(iter+2)
        p = 1 - q
        return p, q

    def solve(self, X: T, missing_mask: T,
              Z_init: T | None = None) -> T:
        """Perform Frank-Wolfe method."""
        Z_old = self._init(X.shape) if Z_init is None else Z_init
        for iter_ in range(self.max_iter):
            grad = Z_old - X
            grad[missing_mask] = 0
            uvT = self._svd_rank_one(grad)
            S = -self.lambda_ * uvT
            p, q = self._get_weights(iter_)
            Z_new = p * Z_old + q * S
            if self._converged(Z_new, Z_old):
                LOGGER.debug(f"Converged after {iter_} iterations. ")
                break
            Z_old = Z_new
        return Z_old

    def _converged(self, Z_new: T, Z_old: T) -> np.bool_:
        """Check convergence conditions."""
        return self._approx_err(Z_old, Z_new) < self.threshold

class CGM_N(CGMBase):
    """A numpy array interface."""

    def _init(self, shape):
        """Initialize as in Hazan's Algorithm (6 in Thesis Jaggi)"""
        #return np.zeros(shape)
        m, n = shape
        u = np.random.rand(m)
        v = np.random.rand(n)
        u_hat = u / np.linalg.norm(u)
        v_hat = v / np.linalg.norm(v)
        return np.outer(u_hat, v_hat)

    def _svd_rank_one(self, M: ndarray) -> tuple:
        # Performs rank one svd using ARPACK solver wrapped by scipy.linalg.svds
        u, s, vT = svds(M,  k=1,  solver='arpack')
        return np.outer(u, vT.T)

    def _approx_err(self, X: ndarray, Y: ndarray) -> floating[Any]:
        """Approximate error."""
        return np.linalg.norm(X - Y) / np.linalg.norm(X)

class CGM_T(CGMBase):
    """Torch tensor interface."""

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _init(self, shape):
        """Initialize as in Hazan's Algorithm (6 in Thesis Jaggi)"""
        #return np.zeros(shape)
        m, n = shape
        u = torch.randn(m)
        v = torch.rand(n)
        u_hat = u / torch.norm(u)
        v_hat = v / torch.linalg.norm(v)
        return torch.outer(u_hat, v_hat)

    def _svd_rank_one(self, M: ndarray) -> Tensor:
        MtM = M.T @ M
        eigvals, v = torch.lobpcg(MtM)
        u = M  @ v
        u = torch.nn.functional.normalize(u, dim=0)
        return torch.outer(u[:, 0], v[:, 0])

    def _approx_err(self, X: ndarray, Y: ndarray) -> floating[Any]:
        """Approximate error."""
        return np.linalg.norm(X - Y) / np.linalg.norm(X)

class CGM:
    """Class for CSMC."""

    def __new__(cls, X: T, *args, **kwargs):
        """Create CSMC object based on the type of X."""
        if isinstance(X, np.ndarray):
            return CGM_N(X, *args, **kwargs)
        elif isinstance(X, torch.Tensor):
            return CGM_T(X, *args, **kwargs)
        else:
            raise TypeError(UNSUPPORTED_MSG)
