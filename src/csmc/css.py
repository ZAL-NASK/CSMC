"""Column Subset Selection methods."""

import numpy as np
import torch
from numpy import ndarray

from csmc.settings import T

"""
Column subset selection methods.
"""


def uniform(X: T, no_col: int) -> T:
    """Select columns according to uniform distribution."""
    if isinstance(X, ndarray):
        cols = np.random.default_rng().choice(X.shape[1], no_col, replace=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cols = torch.randperm(X.shape[1], dtype=torch.int32, device=device)[:no_col]
    return cols
