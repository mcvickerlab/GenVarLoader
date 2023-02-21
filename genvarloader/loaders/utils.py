from asyncio import Future
from functools import partial
from pathlib import Path
from typing import cast

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
        read_fn = partial(
            pd.read_csv, dtype={"contig": str, "strand": str, "sample": str}
        )
    elif queries_path.suffix in {".tsv", ".txt"}:
        read_fn = partial(
            pd.read_csv, sep="\t", dtype={"contig": str, "strand": str, "sample": str}
        )
    elif queries_path.suffix in {".fth", ".feather"}:
        read_fn = pd.read_feather
    else:
        raise ValueError("Unknown file type for queries file.")

    # read and validate
    queries = cast(pd.DataFrame, read_fn(queries_path))
    queries = QueriesSchema.validate(queries)
    queries = cast(Queries, queries)
    return queries
