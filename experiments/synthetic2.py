import csv
import time

import numpy as np

from csmc import CSMC, NuclearNormMin
from csmc.adaptive_mc.asmc import adaptive_mc
from csmc.adaptive_mc.mc2 import mc2
from csmc.errors.errors import approx_err
from csmc.mc_sota.curplus import CUR
from tests.data_generation import create_rank_k_dataset, set_seed

n_rows = 500
n_cols = 1000
seed = 2025
set_seed(seed)

ranks = [10]
n_trials = 20
fraction_missing = 0.8
c_rate = 0.3

csv_path = "synthetic_benchmark2.csv"

# Write header once
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "trial", "n_rows", "n_cols", "rank", "fraction_missing",
        "method", "error", "time", "extra"
    ])

for rank in ranks:
    for trial in range(n_trials):
        base_log_data = [trial, n_rows, n_cols, rank, fraction_missing]

        M, M_incomplete_tmp, omega, ok_mask = create_rank_k_dataset(
            n_rows=n_rows, n_cols=n_cols, k=rank,
            gaussian=True, fraction_missing=fraction_missing, noise=0.3
        )
        n_selected_cols = int(c_rate * n_cols)

        # Always open in append mode so logs are written immediately
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            M_incomplete = np.copy(M_incomplete_tmp)
            print("Starting CSMC")
            solver = CSMC(M_incomplete, col_number=n_selected_cols)
            start = time.perf_counter()
            M_filled = solver.fit_transform(M_incomplete)
            elapsed_time = time.perf_counter() - start
            error = approx_err(M_filled, M)
            writer.writerow(base_log_data + ["CSNN-0.4", error, elapsed_time, ""])

            print("Starting NN")
            M_incomplete = np.copy(M_incomplete_tmp)
            solver = NuclearNormMin(M_incomplete)
            start = time.perf_counter()
            M_filled = solver.fit_transform(M_incomplete, missing_mask=np.isnan(M_incomplete))
            elapsed_time = time.perf_counter() - start
            error = approx_err(M_filled, M)
            writer.writerow(base_log_data + ["NN", error, elapsed_time, 1 - fraction_missing])

            for cur_rank in [rank, 200]:
                M_incomplete = np.copy(M_incomplete_tmp)
                print(f"Starting CUR with rank {cur_rank}")
                solver = CUR(M_incomplete, col_number=n_selected_cols, rank=cur_rank)
                start = time.perf_counter()
                M_filled = solver.fit_transform(M_incomplete)
                elapsed_time = time.perf_counter() - start
                error = approx_err(M_filled, M)
                print(f"CUR {error}")
                writer.writerow(base_log_data + [f"CUR-{cur_rank}", error, elapsed_time, 1 - fraction_missing])

            for mc2_rank in [rank + 1, 100]:
                M_incomplete = np.copy(M_incomplete_tmp)
                random_mask = (np.random.rand(*ok_mask.shape) > 0.5)
                ok_mask = ok_mask * random_mask
                M_incomplete[~ok_mask] = np.nan
                print(f"Starting MC2 with rank {mc2_rank}")
                start = time.perf_counter()
                M_filled = mc2(M_incomplete, np.array(M), mc2_rank)
                elapsed_time = time.perf_counter() - start
                error = approx_err(M_filled, M)
                writer.writerow(base_log_data + [f"MC2-{mc2_rank}", error, elapsed_time, 1 - fraction_missing])

            print("Starting AMC")
            M_incomplete = np.copy(M_incomplete_tmp)
            start = time.perf_counter()
            M_filled, p_observation = adaptive_mc(M, 0.18)
            elapsed_time = time.perf_counter() - start
            error = approx_err(M_filled, M)
            writer.writerow(base_log_data + [f"AMC-{p_observation}", error, elapsed_time, p_observation])
