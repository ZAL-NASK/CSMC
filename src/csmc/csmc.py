"""CSMC logic."""

from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Generic

import numpy as np
import torch
from torch import Tensor

from csmc.settings import T, LOGGER, UNSUPPORTED_MSG
from csmc.css import uniform
from csmc.mc.nn_completion import NuclearNormMin
from csmc.transform import dls



class FillMethod(Enum):
    """Options for the initial imputation."""

    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"


class CSMCBase(Generic[T]):
    """Base class for CSMC model."""

    def __init__(self, X: T, col_number: int, col_select: Callable = uniform,
                 transform: Callable = dls, solver: Callable = NuclearNormMin,
                 threshold: float = 0, fill_method: FillMethod = FillMethod.ZERO,
                 lambda_: float = 0, max_rank: int | None = None) -> None:
        self.col_number = col_number
        self.col_select = col_select
        self._transform = transform
        self.threshold = threshold
        self.fill_method = fill_method
        self.solver = solver
        self.lambda_ = lambda_
        self.max_rank = max_rank
        self.C_incomplete = None
        self.phase_fill_result = False

    def fit_transform(self, X: T) -> T:
        """Complete matrix with CSMC."""
        X_tmp = self._copy(X)
        missing_mask = self._missing_mask(X_tmp)
        self._prepare(X_tmp, missing_mask)
        if self.C_incomplete is None:
            cols_indices = self.col_select(X_tmp, self.col_number)
            self.cols_indices = cols_indices
            C_incomplete = X_tmp[:, cols_indices]
            cols_missing = missing_mask[:, cols_indices]
        else:
            cols_indices = self.cols_indices
            C_incomplete = self.C_incomplete
            cols_missing = self.cols_missing
        C_filled = self.fill_columns(C_incomplete, cols_missing)
        LOGGER.debug("Column submatrix filled.")
        return self.transform(X, C_filled, cols_indices, ~missing_mask)

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

    def transform(self, X_org: T, C_filled: T, cols_indices: T,
                  ok_mask: T) -> T:
        """Solve the least squares problem."""
        X_filled = self._copy(X_org)
        for i, ci in enumerate(cols_indices):
            X_filled[:, ci] = C_filled[:, i]
        X_filled[ok_mask] = X_org[ok_mask]
        out = self._transform(X_filled, ok_mask, C_filled)
        out[ok_mask] = X_org[ok_mask]
        if self.phase_fill_result:
            return out, X_filled
        return out

    @abstractmethod
    def _copy(self, X: T) -> T:
        pass

    @abstractmethod
    def _missing_mask(self, X: T) -> T:
        pass

    @abstractmethod
    def _prepare(self, X: T, missing_mask: T) -> None:
        pass


class CSMC_N(CSMCBase[np.ndarray]):
    """Class for CSMC for numpy arrays."""

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


class CSMC_T(CSMCBase[Tensor]):
    """Class for CSMC for torch tensors."""

    def __init__(self, X: Tensor, **kwargs) -> None:
        super().__init__(X, **kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _copy(self, X: Tensor) -> Tensor:
        return torch.clone(X)

    def _missing_mask(self, X: Tensor) -> Tensor:
        return torch.isnan(X)

    def _prepare(self, X: Tensor, missing_mask: Tensor) -> None:
        X[missing_mask] = 0


class CSMC:
    """Class for CSMC."""

    def __new__(cls, X: T, *args, **kwargs):
        """Create CSMC object based on the type of X."""
        if isinstance(X, np.ndarray):
            return CSMC_N(X, *args, **kwargs)
        elif isinstance(X, torch.Tensor):
            return CSMC_T(X, *args, **kwargs)
        else:
            raise TypeError(UNSUPPORTED_MSG)
