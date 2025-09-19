"""Matrix completion with MC2:
  Armin Eftekhari, Michael B Wakin, Rachel A Ward, MC2: a two-phase algorithm for leveraged matrix completion,
  Information and Inference: A Journal of the IMA, Volume 7, Issue 3, September 2018, Pages 581–604,
  https://doi.org/10.1093/imaiai/iax020"""
from typing import Tuple

import numpy as np

from csmc import NuclearNormMin


def mc2(M_incomplete: np.ndarray, M: np.ndarray, r: int, kappa: float = 1) -> Tuple:
    missing_mask = np.isnan(M_incomplete)
    ok_mask = ~missing_mask
    rng = np.random.default_rng(1)
    Y = np.copy(M_incomplete)
    Y[missing_mask] = 0
    Y_norm_sq = np.linalg.norm(Y, "fro") ** 2
    n, m = Y.shape
    mu_hat = (n * kappa ** 2) * np.sum(Y ** 2, axis=1) / Y_norm_sq
    nu_hat = (m * kappa ** 2) * np.sum(Y ** 2, axis=0) / Y_norm_sq
    factor = (r * (np.log(n + m) ** 2)) / min(n, m)
    P = factor * (mu_hat[:, None] + nu_hat[None, :])
    fraction_observed = np.sum(ok_mask) / M_incomplete.size
    beta = (fraction_observed * n * m) / np.sum(P)
    P = beta * P
    extra_samples = rng.uniform(size=(n, m)) < P
    ok_mask = ok_mask | extra_samples
    M_incomplete[ok_mask] = M[ok_mask]
    p_observed = np.sum(ok_mask) / M_incomplete.size
    solver = NuclearNormMin(M_incomplete)
    M_filled = solver.fit_transform(M_incomplete, missing_mask=np.isnan(M_incomplete))
    return M_filled, p_observed
