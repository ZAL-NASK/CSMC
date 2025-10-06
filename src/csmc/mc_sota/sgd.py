"""
SGD for matrix completion
Tong, T., Ma, C., & Chi, Y. (2021).
Accelerating ill-conditioned low-rank matrix estimation via scaled gradient descent.
Journal of Machine Learning Research, 22(150), 1-63.
"""

from abc import abstractmethod
from typing import Generic

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from csmc.settings import T, UNSUPPORTED_MSG


class SGDBase(Generic[T]):
    """Base class for SGD."""

    def __init__(self, X: T, stepsize: float, rank: int, max_iter: int = 1000,
                 tol: float = 1e-10
                 ) -> None:
        self.stepsize = stepsize
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol

    def spectral_initialization(self, X, r, p):
        """
        Spectral initialization: rank-r SVD of (Y/p)
        """
        U, S, Vt = self._svd(X / p)
        U0 = U[:, :r]
        S0 = self._diag(np.sqrt(S[:r]))  # sqrt once
        V0 = Vt[:r, :].T
        L0 = U0 @ S0
        R0 = V0 @ S0
        return L0, R0

    def fit_transform(self, X: T) -> T:
        """
        Scaled Gradient Descent for matrix completion.
        """

        missing_mask = self._missing_mask(X)
        p = np.sum(~missing_mask) / X.size
        self._prepare(X, missing_mask)
        L, R = self.spectral_initialization(X, self.rank, p)
        for iter_ in range(self.max_iter):
            X_filled = L @ R.T
            Z = np.multiply(~missing_mask, X_filled - X)
            L = L - (self.stepsize / p) * Z @ R @ np.linalg.inv(R.T @ R)
            R = R - (self.stepsize / p) * Z.T @ L @ np.linalg.inv(L.T @ L)
            X_new = L @ R.T
            dist = np.linalg.norm(X_filled - X_new, 'fro') / max(np.linalg.norm(X, 'fro'), 1)
            if dist < self.tol:
                break
        return X_new

    @abstractmethod
    def _missing_mask(self, X: T) -> T:
        pass

    def _prepare(self, X: T, missing_mask: T) -> None:
        X[missing_mask] = 0


class SGD_N(SGDBase[np.ndarray]):
    """Class for SGD for numpy arrays."""

    def _missing_mask(self, X: np.ndarray) -> np.ndarray:
        return np.isnan(X)

    def _svd(self, M: ndarray) -> tuple:
        """Perform SVD."""
        return np.linalg.svd(M, full_matrices=False)

    def _diag(self, s: ndarray) -> ndarray:
        """Diagonalize singular vector."""
        return np.diag(s)


class SGD_T:
    """Class for CUR for torch tensors (not implemented yet)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _svd(self, M: Tensor) -> tuple:
        """Perform SVD."""
        return torch.linalg.svd(
            M,
            full_matrices=False)

    def _missing_mask(self, X: Tensor) -> Tensor:
        """Mask out missing values."""
        return torch.isnan(X)


class SGD:
    """Class for SGD."""

    def __new__(cls, X: T, *args, **kwargs):
        """Create SGD object based on the type of X."""
        if isinstance(X, np.ndarray):
            return SGD_N(X, *args, **kwargs)
        elif isinstance(X, torch.Tensor):
            return SGD_T(X, *args, **kwargs)
        else:
            raise TypeError(UNSUPPORTED_MSG)
