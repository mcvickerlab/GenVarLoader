from asyncio import Future
from functools import partial
from pathlib import Path
from typing import List, Optional, cast

import numpy as np
import pandas as pd
import tensorstore as ts

from genvarloader.loaders.types import Queries, QueriesSchema, _TStore
from genvarloader.types import PathType


def ts_readonly_zarr(path: PathType, **kwargs) -> Future[_TStore]:
    return ts.open(  # type: ignore
        {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}},
        read=True,
        write=False,
        open=True,
        create=False,
        delete_existing=False,
        **kwargs
    )


def read_queries(queries_path: PathType) -> Queries:
    """Read queries from a file. Can be a CSV, TSV, or Apache feather file. Raises an error otherwise.

    Parameters
    ----------
    queries_path : str, Path
    """
    # get correct read function based on the file extension
    queries_path = Path(queries_path)
    if queries_path.suffix == ".csv":
        read_fn = partial(pd.read_csv, dtype={"contig": str, "sample": str})
    elif queries_path.suffix in {".tsv", ".txt"}:
        read_fn = partial(pd.read_csv, sep="\t", dtype={"contig": str, "sample": str})
    elif queries_path.suffix in {".fth", ".feather"}:
        read_fn = pd.read_feather
    else:
        raise ValueError("Unknown file type for queries file.")

    # read and validate
    queries = cast(pd.DataFrame, read_fn(queries_path))
    queries = QueriesSchema.validate(queries)
    queries = cast(Queries, queries)
    return queries


def read_narrowpeak(narrowpeak_path: PathType) -> pd.DataFrame:
    narrowpeaks = pd.read_csv(
        narrowpeak_path,
        sep="\t",
        header=None,
        skiprows=lambda x: x in ["track", "browser"],
        names=[
            "contig",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signalval",
            "pval",
            "qval",
            "peak",
        ],
        dtype={"contig": str},
    )
    return narrowpeaks


def read_narrowpeak_as_queries(
    narrowpeak_path: PathType,
    length: int,
    samples: Optional[List[str]] = None,
    ploid_idx: Optional[List[int]] = None,
) -> Queries:
    """Convert a narrow peak file to a queries file, optionally adding samples and ploid indices.

    Parameters
    ----------
    narrowpeak_path : str or Path
    length : int
        Length of desired queries.
    qvalue_cutoff: float, optional
        Ignore any peaks with a q-value greater than this cutoff.
    samples : list[str], optional
    ploid_idx : list[int], optional

    Returns
    -------
    queries : Queries
    """
    queries = read_narrowpeak(narrowpeak_path)
    # peak loc = start + peak offset
    # query start = peak loc - ceil(length / 2)
    queries["start"] = queries["start"] + queries["peak"] - np.ceil(length / 2)
    queries = queries[["contig", "start", "strand"]]
    if samples is not None:
        sample_df = pd.DataFrame({"sample": samples})
        queries = queries.merge(sample_df, how="cross")
    if ploid_idx is not None:
        ploid_idx_df = pd.DataFrame({"ploid_idx": ploid_idx})
        queries = queries.merge(ploid_idx_df, how="cross")
    queries = QueriesSchema.validate(queries)
    queries = cast(Queries, queries)
    return queries
