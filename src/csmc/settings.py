import logging
import sys
from typing import TypeVar

from numpy import ndarray
from torch import Tensor
import torch
LOGGER = logging.getLogger("__csmc__")
LOGGER.propagate = False
LOGGER.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
_formatter = logging.Formatter(
    fmt="(CSMC) %(asctime)s: %(message)s", datefmt="%b %d %I:%M:%S %p"
)
_stream_handler.setFormatter(_formatter)
LOGGER.addHandler(_stream_handler)

T = TypeVar("T", ndarray, Tensor)

UNSUPPORTED_MSG ="Unsupported data type for X."

torch.set_num_threads(8)