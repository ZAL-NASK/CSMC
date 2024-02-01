# CSMC 

CSMC is a Python library for performing column subset selection in matrix completion tasks. It provides an implementation of the CSSMC method, which aims to complete missing entries in a matrix using a subset of columns.

Columns Selected Matrix Completion (CSMC) is a two-stage approach for low-rank matrix recovery. In the first stage, CSMC samples columns of the input matrix  and recovers a smaller column submatrix.
In the second stage, it solves a least squares problem to reconstruct the whole matrix.

<img src="resources/CSMC.png" alt="Alt text" width="400px" />


## Installation

You can install CSMC using pip:

```bash
pip install -i https://test.pypi.org/simple/ csmc
```

## Usage

```python
from tests.data_generation import create_rank_k_dataset
from csmc import CSMC

n_rows = 300
n_cols = 1000
rank = 10
M, M_incomplete, omega, ok_mask = create_rank_k_dataset(n_rows=n_rows, n_cols=n_cols, k=rank,
                                                        gaussian=True,
                                                        fraction_missing=0.8)
solver = CSMC(M_incomplete, col_number=400)
M_filled = solver.fit_transform(M_incomplete)
```

## Citation

Krajewska, A., Niewiadomska-Szynkiewicz E. (2023). Matrix Completion with Column Subset Selection.