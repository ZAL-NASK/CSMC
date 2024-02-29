"""The second stage of CSMC methods."""

import math

import cvxpy as cp
import numpy as np
import torch
from scipy.linalg import lstsq
from torch import Tensor
from torch.autograd.variable import Variable

from csmc.settings import T


# @numba.njit(parallel=True)
# def _dls(X: np.ndarray, ok_mask: np.ndarray, C: np.ndarray, Y: np.ndarray, n: int) -> None:
#     """Solve the least squares for each column of array X."""
#
#     for i in numba.prange(n):
#         mask_i = ok_mask[:, i]
#
#         #
#         # # Using numpy's pinv and matrix multiplication for solving the least squares problem
#         Y[i, :] = np.linalg.pinv(C[mask_i]) @ X[mask_i, i]


def _dls(X: np.ndarray, ok_mask: np.ndarray, C: np.ndarray, Y: np.ndarray, n: int) -> None:
    """Solve the least squares for each column of array X."""

    for i in range(n):
        mask_i = ok_mask[:, i]
        Y[:, i] = lstsq(C[mask_i], X[mask_i, i], lapack_driver="gelsy")[0]


def _dls_torch(X: T, ok_mask: T, C: T, Y: T, n: int, cuda_support: bool) -> None:
    for i in range(n):
        mask_i = ok_mask[:, i]
        if cuda_support:
            try:
                Y[:, i] = torch.linalg.lstsq(C[mask_i], X[mask_i, i], driver="gels")[0]
            except RuntimeError:
                print("Falling back")
                Y[:, i] = torch.linalg.pinv(C[mask_i]) @ X[mask_i, i]
        else:
            try:
                Y[:, i] = torch.linalg.lstsq(C[mask_i], X[mask_i, i], driver="gelsy")[0]
            except RuntimeError:
                print("Falling back")
                Y[:, i] = torch.linalg.lstsq(C[mask_i], X[mask_i, i], driver="gelsd")[0]


def dls(X: T, ok_mask: T, C: T) -> T:
    """Solve direct least squares."""
    m, n = X.shape
    _, k = C.shape
    if isinstance(X, np.ndarray):
        Y = np.empty((k, n), dtype=X.dtype)
        _dls(X, ok_mask, C, Y, n)
    else:
        cuda_support = torch.cuda.is_available()
        device = torch.device("cuda") if cuda_support else torch.device("cpu")
        Y = torch.zeros((k, n), dtype=torch.float32, device=device)
        _dls_torch(X, ok_mask, C, Y, n, cuda_support)
    return C @ Y


def ls_convex(X: T, ok_mask: T, C: T) -> T:
    """Solve the least squares with cvxpy optimizer."""
    m, n = X.shape
    _, k = C.shape
    Y = cp.Variable((k, n))
    X[~ok_mask] = 0
    obj = cp.sum_squares(cp.multiply(ok_mask, X) - cp.multiply(ok_mask, C @ Y))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.SCS, use_indirect=False)
    return C @ Y.value


def sgrad_torch(X: Tensor, ok_mask: Tensor, C: Tensor, max_iters: int = 1000, lr: float | None = None) -> Tensor:
    """Solve the least squares with stochastic gradient descent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = X.shape
    _, k = C.shape
    if lr is None:
        lr = 1 / math.sqrt(n)

    Y = Variable(torch.empty((k, n), dtype=torch.float32, device=device), requires_grad=True)
    Y.data.normal_(std=0.01)
    opt = torch.optim.Adam([Y], lr=lr)
    X[~ok_mask] = 0

    def loss():
        return torch.mean(torch.mul(ok_mask, (C.mm(Y) - X)) ** 2)

    for _ in range(max_iters):
        opt.zero_grad()
        loss_ = loss()
        loss_.backward()
        opt.step()
    return C @ Y
