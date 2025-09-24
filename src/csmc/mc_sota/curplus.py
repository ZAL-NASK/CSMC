"""CUR+ based matrix completion. Similar to CSMC it randomly samples column and row submatrices, fill them with nuclear norm minimization
In the second stage it solves the regression problem from:
Miao Xu, Rong Jin, and Zhi-Hua Zhou. 2015. CUR algorithm for partially observed matrices.
In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37 (ICML'15). JMLR.org, 1412–1421.
"""

from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Generic

import numpy as np
import torch
from torch import Tensor

from csmc.css import uniform
from csmc.mc.nn_completion import NuclearNormMin
from csmc.settings import T, LOGGER, UNSUPPORTED_MSG


class FillMethod(Enum):
    """Options for the initial imputation."""

    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"


class CURBase(Generic[T]):
    """Base class for CUR+ sampling model."""

    def __init__(self, X: T, col_number: int, rank: int, col_select: Callable = uniform,
                 solver: Callable = NuclearNormMin,
                 fill_method: FillMethod = FillMethod.ZERO,
                 lambda_: float = 0) -> None:
        self.col_number = col_number
        self.rank = rank
        self.col_select = col_select
        self.fill_method = fill_method
        self.solver = solver
        self.lambda_ = lambda_

    def fit_transform(self, X: T) -> T:
        """Complete matrix with CUR."""
        X_tmp = self._copy(X)
        missing_mask = self._missing_mask(X_tmp)
        self._prepare(X_tmp, missing_mask)
        # TODO vary row_number
        cols_indices = self.col_select(X_tmp, self.col_number)
        row_indices = self.col_select(X_tmp.T, self.col_number)
        self.cols_indices = cols_indices
        self.rows_indices = row_indices
        C_incomplete = X_tmp[:, cols_indices]
        R_incomplete = X_tmp[row_indices, :]
        cols_missing = missing_mask[:, cols_indices]
        rows_missing = missing_mask[row_indices, :]
        C_filled = self.fill_columns(C_incomplete, cols_missing)
        LOGGER.debug("Column submatrix filled.")
        R_filled = self.fill_columns(R_incomplete, rows_missing)
        LOGGER.debug("Row submatrix filled.")
        return self.transform(X, C_filled, R_filled, cols_indices, row_indices, ~missing_mask)

    def fill_columns(self, C_incomplete: T, missing_mask: T) -> T:
        """Complete column submatrix."""
        solver = self.solver(C_incomplete, lambda_=self.lambda_,
                             max_rank=self.max_rank) if self.lambda_ else self.solver(C_incomplete)
        return solver.fit_transform(C_incomplete, missing_mask)

    def _fill_columns_with_fn(self, X: T, missing_mask: T, col_fn: Callable) -> None:
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values

    def transform(self, X_org: T, C_filled: T, R_filled: T, cols_indices: T, rows_indices: T,
                  ok_mask: T) -> T:
        """Solve the least squares problem."""
        X_filled = self._copy(X_org)
        for i, ci in enumerate(cols_indices):
            X_filled[:, ci] = C_filled[:, i]
        for i, ri in enumerate(rows_indices):
            X_filled[ri, :] = R_filled[i, :]
        X_filled[ok_mask] = X_org[ok_mask]
        return self._transform(X_org, C_filled, R_filled, ok_mask)

    @abstractmethod
    def _copy(self, X: T) -> T:
        pass

    @abstractmethod
    def _missing_mask(self, X: T) -> T:
        pass

    @abstractmethod
    def _prepare(self, X: T, missing_mask: T) -> None:
        pass


class CUR_N(CURBase[np.ndarray]):
    """Class for CUR for numpy arrays."""

    def _copy(self, X: np.ndarray) -> np.ndarray:
        return np.copy(X)

    def _missing_mask(self, X: np.ndarray) -> np.ndarray:
        return np.isnan(X)

    def _prepare(self, X: np.ndarray, missing_mask: np.ndarray) -> None:
        if self.fill_method == FillMethod.ZERO:
            X[missing_mask] = 0
        elif self.fill_method == FillMethod.MEAN:
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif self.fill_method == FillMethod.MEDIAN:
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif self.fill_method == FillMethod.MIN:
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)

    def _transform(self, X_org: np.ndarray, C_filled: np.ndarray, R_filled: np.ndarray,
                   ok_mask: np.ndarray) -> np.ndarray:
        """Solve the regression problem."""
        U, _, _ = np.linalg.svd(C_filled, full_matrices=False)
        U_hat = U[:, :self.rank]
        V, _, _ = np.linalg.svd(R_filled.T, full_matrices=False)
        V_hat = V[:, :self.rank]
        omega_card = np.sum(ok_mask)
        x_omega = np.empty(omega_card)
        F = np.empty([omega_card, self.rank * self.rank])
        omega = np.argwhere(ok_mask)
        for idx, (i, j) in enumerate(omega):
            Q = np.outer(U_hat[i, :], V_hat[j, :])
            F[idx, :] = Q.ravel()
            x_omega[idx] = X_org[i, j]
        Z_vec, residuals, rank_F, s = np.linalg.lstsq(F, x_omega, rcond=None)
        Z_opt = np.reshape(Z_vec, (self.rank, self.rank))
        out = U_hat @ Z_opt @ V_hat.T
        out[ok_mask] = X_org[ok_mask]
        return out


class CUR_T:
    """Class for CUR+ for torch tensors (not implemented yet)."""

    def __init__(self, X: Tensor, **kwargs) -> None:
        raise NotImplementedError(
            "CUR_T is not implemented yet. Torch.Tensor support will be added in a future version."
        )


class CUR:
    """Class for CUR+."""

    def __new__(cls, X: T, *args, **kwargs):
        """Create CUR object based on the type of X."""
        if isinstance(X, np.ndarray):
            return CUR_N(X, *args, **kwargs)
        elif isinstance(X, torch.Tensor):
            return CUR_T(X, *args, **kwargs)
        else:
            raise TypeError(UNSUPPORTED_MSG)
