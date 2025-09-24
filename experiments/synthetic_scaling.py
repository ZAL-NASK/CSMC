import csv
import time

import numpy as np
from fancyimpute import MatrixFactorization

from csmc import CSMC, NuclearNormMin
from csmc.errors.errors import approx_err
from csmc.mc_sota.sgd import SGD
from tests.data_generation import create_rank_k_dataset, set_seed

n_rows = 400
n_cols = 1000
seed = 2025
set_seed(seed)

ranks = [5]
n_trials = 20
fraction_missing = 0.8
c_rate = 0.3

csv_path = "synthetic_benchmark_scaling.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "trial", "n_rows", "n_cols", "rank", "fraction_missing",
        "method", "error", "time", "extra"
    ])
for n_cols in [800, 1000, 2000, 5000, 10000]:
    for rank in ranks:
        for trial in range(n_trials):
            base_log_data = [trial, n_rows, n_cols, rank, fraction_missing]

            M, M_incomplete_tmp, omega, ok_mask = create_rank_k_dataset(
                n_rows=n_rows, n_cols=n_cols, k=rank,
                gaussian=True, fraction_missing=fraction_missing,
            )
            n_selected_cols = int(c_rate * n_cols)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                M_incomplete = np.copy(M_incomplete_tmp)
                print("Starting CSMC")
                solver = CSMC(M_incomplete, col_number=n_selected_cols)
                start = time.perf_counter()
                M_filled = solver.fit_transform(M_incomplete)
                elapsed_time = time.perf_counter() - start
                error = approx_err(M_filled, M)
                print(f"CSMC {error} {elapsed_time}")
                writer.writerow(base_log_data + ["CSNN-0.4", error, elapsed_time, ""])

                print("Starting NN")
                M_incomplete = np.copy(M_incomplete_tmp)
                solver = NuclearNormMin(M_incomplete)
                start = time.perf_counter()
                M_filled = solver.fit_transform(M_incomplete, missing_mask=np.isnan(M_incomplete))
                elapsed_time = time.perf_counter() - start
                error = approx_err(M_filled, M)
                print(f"NN {error}")
                writer.writerow(base_log_data + ["NN", error, elapsed_time, 1 - fraction_missing])

                for r in [rank - 1, rank, rank + 1]:
                    print("Starting SGD")
                    M_incomplete = np.copy(M_incomplete_tmp)
                    solver = SGD(M_incomplete, stepsize=0.1, rank=r)
                    start = time.perf_counter()
                    M_filled = solver.fit_transform(M_incomplete)
                    elapsed_time = time.perf_counter() - start
                    error = approx_err(M_filled, M)
                    print(f"SGD {error} {elapsed_time}")
                    writer.writerow(base_log_data + ["SGD", error, elapsed_time, 1 - fraction_missing])

                print("Starting MF")
                M_incomplete = np.copy(M_incomplete_tmp)
                solver = MatrixFactorization(rank=rank, max_iters=100, shrinkage_value=0.001)
                start = time.perf_counter()
                M_filled = solver.fit_transform(M_incomplete)
                elapsed_time = time.perf_counter() - start
                error = approx_err(M_filled, M)
                print(f"SGD {error} {elapsed_time}")
                writer.writerow(base_log_data + ["MF", error, elapsed_time, 1 - fraction_missing])
