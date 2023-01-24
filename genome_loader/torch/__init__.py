try:
    import torch
except ImportError:
    raise ImportError("The `torch` module requires PyTorch.")

import warnings
from abc import abstractmethod
from typing import Protocol, Type

import pandas as pd

from genome_loader.gloader.experimental import Queries
from genome_loader.utils import PathType

from .data import ConsensusGLDataset, GLDropN, NDimSampler, PredictToTensorStore

# INVALID_POLARS_TO_TORCH_TYPES: list[Type[pl.DataType]] = [
#     pl.Utf8,
#     pl.Object,
#     pl.List,
#     pl.Date,
#     pl.Datetime,
#     pl.Time,
#     pl.Categorical,
#     pl.Null,
# ]


def parse_queries(queries_path: PathType) -> Queries:
    _queries = pd.read_csv(queries_path, dtypes={"contig": pl.Utf8})
    required_cols = ["contig", "start", "strand", "sample", "ploid_idx"]
    missing_cols = [col for col in required_cols if col not in _queries.columns]
    if len(missing_cols) > 0:
        raise ValueError("Missing required columns:", missing_cols)
    non_query_cols = [
        (col, dtype)
        for col, dtype in zip(_queries.columns, _queries.dtypes)
        if col not in required_cols
    ]
    invalid_cols_dtypes = [
        (col, dtype.string_repr())
        for col, dtype in non_query_cols
        if dtype in INVALID_POLARS_TO_TORCH_TYPES
    ]
    if len(invalid_cols_dtypes) > 0:
        msg = [f"{col}, dtype: {dtype}" for col, dtype in invalid_cols_dtypes]
        warnings.warn(f"Found columns that can't be converted to PyTorch: {msg}")
    return _queries.select(pl.exclude([x[0] for x in invalid_cols_dtypes]))


class TorchCollator(Protocol):
    def __call__(self, batch_indices: list[int]) -> dict[str, torch.Tensor]:
        ...
