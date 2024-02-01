import time

import numpy as np

from csmc import CSMC
from csmc.css import uniform
from csmc.errors.errors import approx_err, approx_err_unknown, nmae_unknown, snr
from csmc.mc.nn_completion import NuclearNormMin
from csmc.transform import dls
from tests.data_generation import create_rank_k_dataset
from tests.linalg import matrix_coherence

n_rows = 300
n_cols = 1000

ranks = [10]

n_trials = 20

c_rates = [i * 0.1 for i in range(1, 10)]

# missing_rates = [i * 0.1 for i in range(1, 10)]
missing_rates = [0.8]
lib = "numpy"

log_name = "synthetic_benchmark"


def evaluate(M_filled, M, missing_mask):
    approx_err_unknown_ = approx_err_unknown(M_filled, M, missing_mask, numlib=lib)
    nmae_unknown_ = nmae_unknown(M_filled, M, missing_mask, numlib=lib)
    approx_err_ = approx_err(M_filled, M, numlib=lib)
    success = int(approx_err_ <= 10 ** -2)
    snr_ = snr(M_filled, M, numlib=lib)
    return [approx_err_, approx_err_unknown_, nmae_unknown_, success, snr_]


for trial in range(5):
    print(f'Starting trial {trial}')
    for fraction_missing in missing_rates:
        for rank in ranks:
            print(f'Rank {rank}')
            M, M_incomplete, omega, ok_mask = create_rank_k_dataset(n_rows=n_rows, n_cols=n_cols, k=rank,
                                                                    gaussian=True,
                                                                    fraction_missing=fraction_missing,
                                                                    noise=0.3)
            M_incomplete_tmp = np.copy(M_incomplete)
            M_incomplete_tmp[~ok_mask] = 0

            missing_mask = ~ok_mask
            base_log_data = [trial, n_rows, n_cols, rank, fraction_missing]
            for c_rate in c_rates:
                n_selected_cols = int(c_rate * n_cols)
                solver = CSMC(M_incomplete, col_number=n_selected_cols, solver=NuclearNormMin, col_select=uniform, transform=dls)
                start = time.perf_counter()
                M_filled = solver.fit_transform(M_incomplete)
                elapsed_time = time.perf_counter() - start
                print(elapsed_time)
                try:
                    metrics = evaluate(M_filled, M, missing_mask)
                except:
                    metrics = evaluate(M_filled.numpy(), M, missing_mask)
                print(metrics)
                # log_data = base_log_data + [c_rate, elapsed_time, mc, f'CSNN_{c_rate}'] + metrics
                # log(log_data, file_name=log_name)

            solver = NuclearNormMin(M_incomplete)
            start = time.perf_counter()
            M_filled = solver.fit_transform(M_incomplete, missing_mask)
            elapsed_time = time.perf_counter() - start
            # approx_err_unknown_ = rmse_omega(M_filled, M, missing_mask, numlib=lib)
            # nmae_omega_ = nmae_omega(M_filled, M, missing_mask, numlib=lib)
            # rmse_ = rmse(M_filled, M, numlib=lib)
            # print(rmse_, elapsed_time)
            # snr_ = snr(M_filled, M, numlib=lib)
            # success = int(rmse_ <= 10 ** -2)
            # log_data = base_log_data + [1, elapsed_time, rmse_, approx_err_unknown_,
            #                             nmae_omega_, snr_, success, mc, 'NN']
            #
            # log(log_data, file_name=log_name)

print('Finished benchmark')
