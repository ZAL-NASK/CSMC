from tests.data_generation import create_rank_k_dataset
from csmc import CSMC, NuclearNormMin
from csmc.errors.errors import approx_err
import time
import csv
import numpy as np
from csmc.mc_sota.curplus import CUR
from csmc.adaptive_mc.mc2 import mc2
from csmc.adaptive_mc.asmc import adaptive_mc

n_rows = 500
n_cols = 1000

ranks = [5, 10]
n_trials = 5
fraction_missing = 0.8
c_rate = 0.4

csv_file = open("synthetic_benchmark2.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

for rank in ranks:
    for trial in (n_trials):
        base_log_data = [trial, n_rows, n_cols, rank, fraction_missing, error, elapsed_time]
        M, M_incomplete, omega, ok_mask = create_rank_k_dataset(n_rows=n_rows, n_cols=n_cols, k=rank,
                                                                gaussian=True,
                                                                fraction_missing=fraction_missing,
                                                                noise=0.3)
        n_selected_cols = int(c_rate * n_cols)
        print("Starting CSMC")
        solver = CSMC(M_incomplete, col_number=n_selected_cols)
        start = time.perf_counter()
        M_filled = solver.fit_transform(M_incomplete)
        elapsed_time = time.perf_counter() - start
        error = approx_err(M_filled, M)
        log_data = base_log_data + ["CSNN-0.4", error, elapsed_time]
        csv_writer.writerow(log_data)

        print("Starting NN")
        solver = NuclearNormMin(M_incomplete)
        start = time.perf_counter()
        M_filled = solver.fit_transform(M_incomplete, missing_mask=np.isnan(M_incomplete))
        elapsed_time = time.perf_counter() - start
        error = approx_err(M_filled, M)
        log_data = base_log_data + ["NN", error, elapsed_time]
        csv_writer.writerow(log_data)

        for cur_rank in [rank, 100]:
            print(f"Starting CUR with rank {cur_rank}")
            solver = CUR(M_incomplete, col_number=n_selected_cols, rank=cur_rank)
            start = time.perf_counter()
            M_filled = solver.fit_transform(M_incomplete)
            elapsed_time = time.perf_counter() - start
            error = approx_err(M_filled, M)
            log_data = base_log_data + [f"CUR-{cur_rank}", error, elapsed_time]
            csv_writer.writerow(log_data)


        for mc2_rank in [rank+1, 100]:
            print(f"Starting MC2 with rank {cur_rank}")
            start = time.perf_counter()
            M_filled = mc2(M_incomplete, M, mc2_rank)
            elapsed_time = time.perf_counter() - start
            error = approx_err(M_filled, M)
            log_data = base_log_data + [f"MC2-{mc2_rank}", error, elapsed_time]
            csv_writer.writerow(log_data)


        print("Starting AMC")
        start = time.perf_counter()
        M_filled, p_observation = adaptive_mc(M, 0.18)
        elapsed_time = time.perf_counter() - start
        error = approx_err(M_filled, M)
        log_data = base_log_data + [f"AMC-{p_observation}", error, elapsed_time]
        csv_writer.writerow(log_data)