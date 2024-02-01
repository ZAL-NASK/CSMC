"""Evaluation methods."""
import math

import numpy as np
import torch
from torch import Tensor

from csmc.settings import T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def approx_err(X: T, Y: T, numlib: str ="numpy") -> float:
    """Relative approximation error."""
    lib = np.linalg if numlib == "numpy" else torch
    return lib.linalg.norm(X - Y) / lib.linalg.norm(Y)


def approx_err_unknown(X: T, Y: T, missing_mask: T, numlib: str ="numpy") -> float:
    """Relative approximation over unknown entries."""
    lib = np.linalg if numlib == "numpy" else torch
    return lib.norm(X[missing_mask] - Y[missing_mask]) / lib.norm(Y[missing_mask])


def nmae_unknown(X: T, Y: T, ok_mask: T, numlib: str="numpy") -> float:
    """NMAE over unknown entries."""
    lib = np if numlib == "numpy" else torch
    return lib.sum(lib.abs(X[ok_mask] - Y[ok_mask])) / (lib.sum(ok_mask) * lib.max(Y[ok_mask]) - lib.min(Y[ok_mask]))


def approx_err_unknown_torch(X: Tensor, Y: Tensor, ok_mask: Tensor) -> float:
    """Relative approximation error for tensors."""
    return torch.norm(X[ok_mask] - Y[ok_mask]) / torch.norm(X[ok_mask])


def snr(X: T, Y: T, numlib: str="numpy") -> float:
    """Signal to noise ratio."""
    lib = np.linalg if numlib == "numpy" else torch
    return 20 * math.log10(lib.linalg.norm(Y) / lib.linalg.norm(X - Y))


def _round(X: T, numlib: str ="numpy") -> T:
    lib = np if numlib == "numpy" else torch
    X_r = lib.round(X * 2) / 2
    min_value = 0.5
    max_value = 5
    X_r[X_r < min_value] = min_value
    X_r[X_r > max_value] = min_value
    return X_r


def hits(X: T, Y: T, ok_mask: T, numlib: str="numpy") -> float:
    """Hits number."""
    X_r = _round(X, numlib)
    Y_r = _round(Y, numlib)
    lib = np if numlib == "numpy" else torch
    return lib.sum(X_r[ok_mask] == Y_r[ok_mask]) / lib.sum(ok_mask)
