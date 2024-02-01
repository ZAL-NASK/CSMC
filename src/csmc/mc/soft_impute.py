"""Matrix completion with Proximal Gradient Descent."""
from abc import abstractmethod
from typing import Any, Generic

import fbpca
import numpy as np
import torch
from numpy import floating, ndarray
from torch import Tensor

from csmc.settings import T, LOGGER, UNSUPPORTED_MSG


class SoftImputeBase(Generic[T]):
    """A class for completing matrix using Soft Impute algorithm, based on Proximal Gradient Descent method."""

    def __init__(self, X :T, lambda_: float = 10, threshold: float = 0.001, max_iter: int = 10000,
                 max_rank: int | None = None, numlib: str = "numpy", **kwargs) -> None:
        self.numlib = numlib
        self.lambda_ = lambda_
        self.threshold = threshold
        self.max_iter = max_iter
        self.max_rank = max_rank
        self.svd_randomized = True

    def shrinkage_operator(self, M: T, max_rank: int | None = None) -> tuple:
        """Apply Soft Thresholding to the matrix."""
        (U, s, V) = self._svd(M, max_rank)
        s_shrinked = self._shrink(s)
        rank = (s_shrinked > 0).sum()
        s_shrinked = s_shrinked[:rank]
        U_shrinked = U[:, :rank]
        V_shrinked = V[:rank, :]
        S_shrinked = self._diag(s_shrinked)
        M_shrinked = U_shrinked @ (S_shrinked @ V_shrinked)
        return M_shrinked, rank


    def solve(self, X: T, missing_mask: T,
              Z_init: T | None = None) -> T:
        """Perform Proximal Gradient Descent method."""
        Z_old = self._init(X.shape) if Z_init is None else Z_init
        ok_mask = ~missing_mask
        rank = self.max_rank
        for iter_ in range(self.max_iter):
            Z_old[ok_mask] = X[ok_mask]
            if rank:
                Z_new, rank = self.shrinkage_operator(Z_old, rank)
            else:
                Z_new, rank = self.shrinkage_operator(Z_old)
            if self._converged(Z_new, Z_old):
                LOGGER.debug(f"Converged after {iter_} iterations and rankd {rank} ")
                break
            Z_old = Z_new
        Z_new[ok_mask] = X[ok_mask]
        return Z_new

    def _converged(self, Z_new: T, Z_old: T) -> np.bool_:
        """Check convergence conditions."""
        return self._approx_err(Z_old, Z_new) < self.threshold

    def fit(self, M: T, missing_mask: T,
            Z_init: T | None = None) -> T:
        """Complete matrix."""
        return self.solve(M, missing_mask, Z_init=Z_init)

    def fit_transform(self, M: T, missing_mask: T, Z_init: T | None =None) -> T:
        """Complete matrix."""
        return self.fit(M, missing_mask, Z_init=Z_init)

    @abstractmethod
    def _svd(self, X: T, max_rank: int | None = None) -> tuple:
        pass

    @abstractmethod
    def _shrink(self, s: T) -> T:
        pass

    @abstractmethod
    def _diag(self, s: T) -> T:
        pass

    @abstractmethod
    def _init(self, shape: tuple) -> T:
        pass

    @abstractmethod
    def _ok_mask(self, X: T) -> T:
        pass

    @abstractmethod
    def _approx_err(self, X: T, Y: T) -> floating[Any]:
        pass


class SoftImpute_N(SoftImputeBase):
    """A numpy array interface."""

    def _svd(self, M: ndarray, max_rank: int | None = None) -> tuple:
        """Perform SVD."""
        if max_rank and self.svd_randomized:
            LOGGER.debug("Performing randomized SVD.")
            (U, s, V) = fbpca.pca(
                M,
                max_rank)

        else:
            LOGGER.debug("Performing full SVD.")
            (U, s, V) = np.linalg.svd(
                M,
                full_matrices=False,
                compute_uv=True)
            # (U, s, V) = fbpca.pca(
            #     M)
        return (U, s, V)

    def _shrink(self, s: ndarray) -> ndarray:
        """Shrink singular vector."""
        return np.maximum(s - self.lambda_, np.zeros_like(s))

    def _diag(self, s: ndarray) -> ndarray:
        """Diagonalize singular vector."""
        return np.diag(s)

    def _init(self, shape: tuple) -> ndarray:
        """Set initial point."""
        return np.zeros(shape)

    def _ok_mask(self, X: ndarray) -> ndarray:
        """Mask out missing values."""
        return ~np.isnan(X)

    def _approx_err(self, X: ndarray, Y: ndarray) -> floating[Any]:
        """Approximate error."""
        return np.linalg.norm(X - Y) / np.linalg.norm(X)


class SoftImpute_T(SoftImputeBase):
    """Torch tensor interface."""

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _svd(self, M: Tensor, max_rank: int | None = None) -> tuple:
        """Perform SVD."""
        if max_rank and self.svd_randomized:
            (U, s, V) = torch.svd_lowrank(
                M,
                max_rank)
            V = V.T

        else:
            (U, s, V) = torch.linalg.svd(
                M,
                full_matrices=False)
        return (U, s, V)

    def _shrink(self, s: Tensor) -> Tensor:
        """Shrink singular vector."""
        return torch.maximum(s - self.lambda_, torch.zeros_like(s))

    def _diag(self, s: Tensor) -> Tensor:
        """Diagonalize singular vector."""
        return torch.diag(s)

    def _init(self, shape: tuple) -> Tensor:
        """Set initial point."""
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _ok_mask(self, X: Tensor) -> Tensor:
        """Mask out missing values."""
        return ~torch.isnan(X)

    def _approx_err(self, X: Tensor, Y: Tensor) -> floating[Any]:
        """Approximate error."""
        return torch.linalg.norm(X - Y) / torch.linalg.norm(X)


class SoftImpute:
    """Class for CSMC."""

    def __new__(cls, X: T, *args, **kwargs):
        """Create CSMC object based on the type of X."""
        if isinstance(X, np.ndarray):
            return SoftImpute_N(X, *args, **kwargs)
        elif isinstance(X, torch.Tensor):
            return SoftImpute_T(X, *args, **kwargs)
        else:
            raise TypeError(UNSUPPORTED_MSG)
