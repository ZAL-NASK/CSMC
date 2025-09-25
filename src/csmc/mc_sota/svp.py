"""
SVP for matrix completion

Jain, P. &amp; Netrapalli, P.. (2015).
Fast Exact Matrix Completion with Finite Samples.
Proceedings of The 28th Conference on Learning Theory, in Proceedings of Machine Learning Research
 Available from https://proceedings.mlr.press/v40/Jain15.html.


"""

from abc import abstractmethod
from typing import Generic

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from csmc.settings import T, UNSUPPORTED_MSG


class SVPBase(Generic[T]):
    """Base class for Singular Value Projection (SVP)."""

    def __init__(self, X: T, rank: int, step_size: int = -1, max_iter: int = 1000, tol: float = 1e-6) -> None:
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Singular Value Projection (SVP) for matrix completion
        with partitioned observation sets (Algorithm 1).
        """

        missing_mask = self._missing_mask(X)
        ok_mask = ~missing_mask
        self._prepare(X, missing_mask)
        n1, n2 = X.shape
        X_filled = np.zeros((n1, n2))
        if self.step_size == -1:
            self.stepsize = ((n1 * n2) / (np.sqrt(self.max_iter) * np.sum(ok_mask)))

        for t in range(self.max_iter):
            residual = ok_mask * (X_filled - X)
            Y = X_filled - self.stepsize * residual
            U, s, Vt = self._svd(Y)

            s[self.rank:] = 0
            X_new = U @ np.diag(s) @ Vt
            dist = np.linalg.norm(X - X_new, 'fro') / max(np.linalg.norm(X, 'fro'), 1)
            if dist < self.tol:
                break
            X_filled = X_new

        return X_filled

    @abstractmethod
    def _missing_mask(self, X: T) -> T:
        pass

    @abstractmethod
    def _svd(self, M: T) -> tuple:
        pass

    @abstractmethod
    def _diag(self, s: T) -> T:
        pass

    def _prepare(self, X: T, missing_mask: T) -> None:
        """Replace missing entries with 0 so arithmetic is valid."""
        X[missing_mask] = 0


class SVP_N(SVPBase[np.ndarray]):
    """SVP implementation for numpy arrays."""

    def _missing_mask(self, X: np.ndarray) -> np.ndarray:
        return np.isnan(X)

    def _svd(self, M: ndarray) -> tuple:
        return np.linalg.svd(M, full_matrices=False)

    def _diag(self, s: ndarray) -> ndarray:
        return np.diag(s)


class SVP_T(SVPBase[Tensor]):
    """SVP implementation for torch tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _missing_mask(self, X: Tensor) -> Tensor:
        return torch.isnan(X)

    def _svd(self, M: Tensor) -> tuple:
        return torch.linalg.svd(M, full_matrices=False)

    def _diag(self, s: Tensor) -> Tensor:
        return torch.diag(s)


class SVP:
    """Factory for SVP."""

    def __new__(cls, X: T, *args, **kwargs):
        if isinstance(X, np.ndarray):
            return SVP_N(X, *args, **kwargs)
        elif isinstance(X, torch.Tensor):
            return SVP_T(X, *args, **kwargs)
        else:
            raise TypeError(UNSUPPORTED_MSG)
