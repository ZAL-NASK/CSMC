import csv
import time
from pathlib import Path

import kagglehub
from fancyimpute import MatrixFactorization

from csmc import CSMC
from csmc.mc_sota.sgd import SGD
from csmc.mc_sota.svp import SVP
from tests.data_generation import set_seed

seed = 2025
set_seed(seed + 1)
import os

import pandas as pd
import numpy as np

local_path = Path("JESTER")
path = kagglehub.dataset_download("crawford/jester-online-joke-recommender")

df = pd.read_csv(os.path.join(path, "jesterfinal151cols.csv"))
df = pd.read_csv("jester-data-2.csv")
R = df.drop(columns=df.columns[0]).to_numpy()
R = np.where(R == 99, np.nan, R)
J
R = R.T

test_fraction = 0.8
n_trials = 5
n_selected_cols = 500
rank = 6
max_iter = 1000

import numpy as np


def drop_all_nan(R):
    mask_rows = ~np.isnan(R).all(axis=1)
    mask_cols = ~np.isnan(R).all(axis=0)
    R_clean = R[mask_rows][:, mask_cols]
    return R_clean


def split_mc_indices(R, observed, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)

    train_idx = set(map(tuple, observed))
    test_idx = set()

    row_counts = np.count_nonzero(~np.isnan(R), axis=1)
    col_counts = np.count_nonzero(~np.isnan(R), axis=0)

    for (u, i) in observed:
        if rng.random() < test_fraction:

            if row_counts[u] > 1 and col_counts[i] > 1:
                train_idx.remove((u, i))
                test_idx.add((u, i))
                row_counts[u] -= 1
                col_counts[i] -= 1

    R_train = R.copy().astype(float)
    for (u, i) in test_idx:
        R_train[u, i] = np.nan

    return list(train_idx), list(test_idx), R_train


R = drop_all_nan(R)

rows_all_nan = np.isnan(R).all(axis=1)
cols_all_nan = np.isnan(R).all(axis=0)


def mae_on_test(R_pred, R_true, test_idx):
    diffs = []
    for u, i in test_idx:
        diffs.append(abs(R_true[u, i] - R_pred[u, i]))
    return np.mean(diffs)


def nmae_on_test(R_pred, R_true, test_idx, r_min=-10, r_max=10):
    mae = mae_on_test(R_pred, R_true, test_idx)
    return mae / (r_max - r_min)


observed = np.argwhere(~np.isnan(R))
print(f"Matrix shape: {R.shape}, observed entries: {len(observed)}, density {len(observed) / R.size}")

csv_path = "jester_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "trial", "method", "NMAE", "MAE", "time"
    ])

for trial in range(n_trials):
    train_idx, test_idx, R_train = split_mc_indices(R, observed, test_fraction=test_fraction)

    R_train = R.copy()
    for u, i in test_idx:
        R_train[u, i] = np.nan

    base_log_data = [trial]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        print("Starting CSMC")
        R_train_1 = R_train.copy()
        solver = CSMC(R_train_1, col_number=n_selected_cols)
        start = time.perf_counter()
        R_filled = solver.fit_transform(R_train_1)
        # R_filled[R_filled > 10] = 10
        # R_filled[R_filled < -10] = -10
        print(np.linalg.matrix_rank(R_filled))
        elapsed = time.perf_counter() - start
        mae = mae_on_test(R_filled, R, test_idx)
        nmae = nmae_on_test(R_filled, R, test_idx)
        print(f" MAE={mae:.4f}, NMAE={nmae:.4f}, time={elapsed:.4f}")

        writer.writerow(base_log_data + ["CSMC", nmae, mae, elapsed])

        # print("Starting NN")
        # R_train_1 = R_train.copy()
        # solver = NuclearNormMin(R_train_1)
        # start = time.perf_counter()
        # R_filled = solver.fit_transform(R_train_1, missing_mask=np.isnan(R_train_1))
        # mae = mae_on_test(R_filled, R, test_idx)
        # nmae = nmae_on_test(R_filled, R, test_idx)
        # print(f" MAE={mae:.4f}, NMAE={nmae:.4f}")

        for r in [10, 20, 50]:
            R_train_1 = R_train.copy()
            print(f"Starting SGD-{r}")
            solver = SGD(R_train_1, stepsize=0.1, rank=r, max_iter=max_iter)
            start = time.perf_counter()
            R_filled = solver.fit_transform(R_train_1)
            elapsed = time.perf_counter() - start
            # R_filled[R_filled > 10] = 10
            # R_filled[R_filled < -10] = -10
            mae = mae_on_test(R_filled, R, test_idx)
            nmae = nmae_on_test(R_filled, R, test_idx)

            print(f" MAE={mae:.4f}, NMAE={nmae:.4f}, time {elapsed:.4f}")
            writer.writerow(base_log_data + [f"SGD-{r}", nmae, mae, elapsed])

        for r in [10, 20, 50]:
            R_train_1 = R_train.copy()
            print(f"Starting SVP-{r}")
            solver = SVP(R_train_1, rank=r)
            start = time.perf_counter()
            R_filled = solver.fit_transform(R_train_1)
            elapsed = time.perf_counter() - start
            mae = mae_on_test(R_filled, R, test_idx)
            nmae = nmae_on_test(R_filled, R, test_idx)
            print(f" MAE={mae:.4f}, NMAE={nmae:.4f}, time={elapsed:.4f}")
            writer.writerow(base_log_data + [f"SVP-{r}", nmae, mae, elapsed])

        for sv in [0.001, 0.01, 0.1]:
            print("Starting MF")
            R_train_1 = R_train.copy()
            solver = MatrixFactorization(rank=rank, max_iters=max_iter, shrinkage_value=sv)
            start = time.perf_counter()
            R_filled = solver.fit_transform(R_train_1)
            elapsed = time.perf_counter() - start
            mae = mae_on_test(R_filled, R, test_idx)
            nmae = nmae_on_test(R_filled, R, test_idx)
            print(f"MAE={mae:.4f}, NMAE={nmae:.4f}")
            writer.writerow(base_log_data + [f"MF-{sv}", nmae, mae, elapsed])
