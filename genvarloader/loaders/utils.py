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
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
            "peak",
        ],
        dtype={"chrom": str},
    )
    return narrowpeaks


def read_narrowpeak_as_queries(
    narrowpeak_path: PathType,
    length: int,
    samples: Optional[List[str]] = None,
) -> Queries:
    """Read a .narrowPeak file as queries centered around peaks, optionally adding samples.

    Parameters
    ----------
    narrowpeak_path : str or Path
    length : int
        Length of desired queries.
    samples : list[str], optional

    Returns
    -------
    queries : Queries
    """
    queries = read_narrowpeak(narrowpeak_path)
    # peak loc = start + peak offset
    # query start = peak loc - ceil(length / 2)
    queries["chromStart"] = (
        queries["chromStart"] + queries["peak"] - np.ceil(length / 2)
    ).astype(int)
    queries = queries[["chrom", "chromStart", "strand"]].rename(
        columns={"chrom": "contig", "chromStart": "start"}
    )
    if samples is not None:
        sample_df = pd.DataFrame({"sample": samples})
        queries = queries.merge(sample_df, how="cross")
    queries = QueriesSchema.validate(queries)
    queries = cast(Queries, queries)
    return queries


def read_broadpeak(broadpeak_path: PathType):
    broadpeaks = pd.read_csv(
        broadpeak_path,
        sep="\t",
        header=None,
        skiprows=lambda x: x in ["track", "browser"],
        names=[
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
        ],
        dtype={"chrom": str},
    )
    return broadpeaks


def read_broadpeak_as_queries(
    broadpeak_path: PathType,
    length: int,
    samples: Optional[List[str]] = None,
) -> Queries:
    """Read a .broadPeak file as queries centered around them, optionally adding samples.

    Parameters
    ----------
    narrowpeak_path : str or Path
    length : int
        Length of desired queries.
    samples : list[str], optional

    Returns
    -------
    queries : Queries
    """
    queries = read_broadpeak(broadpeak_path)
    # midpoint = (start + end) / 2
    # query start = midpoint - length / 2
    queries["chromStart"] = (
        ((queries["chromStart"] + queries["chromEnd"]) / 2 - length / 2)
        .round()
        .astype(int)
    )
    queries = queries[["chrom", "chromStart", "strand"]].rename(
        columns={"chrom": "contig", "chromStart": "start"}
    )
    if samples is not None:
        sample_df = pd.DataFrame({"sample": samples})
        queries = queries.merge(sample_df, how="cross")
    queries = QueriesSchema.validate(queries)
    queries = cast(Queries, queries)
    return queries
