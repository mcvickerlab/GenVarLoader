from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, cast

import dask.array as da
import numba as nb
import numpy as np
import polars as pl
import xarray as xr

from .types import Reader


@nb.njit(nogil=True, parallel=True, cache=True)
def assign_vals(out, sample_idx, starts, ends, vals):
    # all intervals in an RLE table are guaranteed to be non-overlapping
    for i in nb.prange(len(starts)):
        out[sample_idx[i], starts[i] : ends[i]] = vals[i]


class RLE_Table(Reader):
    def __init__(
        self,
        name: str,
        table: Union[str, Path, pl.DataFrame],
        samples: Optional[Sequence[str]] = None,
        validate=True,
        zero_indexed=True,
        lazy=True,
        **table_reader_kwargs,
    ):
        """Read values from a run-length encoded (RLE) table. This corresponds to the
        UCSC BED5+ format, where sample names are found in the `name` column and values
        for each interval are in the `score` column.

        Parameters
        ----------
        name : str
            Name of the reader.
        table : Union[str, Path, pl.DataFrame]
            DataFrame, or path to one, in the UCSC BED5+ format.
        samples : Sequence[str], optional
            Names of samples to potentially include in calls to `read()`. By default all
            samples are used.
        validate : bool, optional
            Whether to validate the RLE table, by default True.
        zero_indexed : bool, optional
            Whether the coordinates are 0-indexed, by default True
        lazy : bool, optional
            Whether to hold the table in-memory or query it lazily, leaving it on-disk
            to reduce memory usage at the cost of query speed. Ignored if `table` is a
            DataFrame.
        **table_reader_kwargs
            Passed to the table reader function (either pl.scan_csv or pl.scan_ipc).
        """

        if isinstance(table, (str, Path)):
            table = Path(table)
            suffixes = set(table.suffixes)
            reader_kwargs: Dict[str, Any] = {}
            reader_kwargs.update(table_reader_kwargs)
            if ".csv" in suffixes:
                reader_kwargs["separator"] = ","
                reader = pl.scan_csv
            elif {".txt", ".tsv"} & suffixes:
                reader_kwargs["separator"] = "\t"
                reader = pl.scan_csv
            elif {".fth", ".feather", ".ipc", ".arrow"} & suffixes:
                reader = pl.scan_ipc
            else:
                raise ValueError(f"Table has unrecognized file extension: {table.name}")
            _table = reader(table, **reader_kwargs)
        else:
            _table = table.lazy()

        if not lazy:
            _table = _table.collect()

        if not zero_indexed:
            _table = _table.with_columns(pl.col("chromStart") - 1)

        if validate:
            _table = validate_rle_table(_table).lazy()

        if samples is not None:
            _samples = np.asarray(samples).astype(str)
            _table = _table.filter(pl.col("name").is_in(_samples))
        else:
            _samples = (
                _table.lazy()
                .select(pl.col("name").unique(maintain_order=True))
                .collect()["name"]
                .to_numpy()
            )

        self.name = name
        self.table = _table
        self.virtual_data = xr.DataArray(
            da.empty(len(_samples), dtype=np.float64),
            name=name,
            dims="sample",
            coords={"sample": _samples},
        )

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        samples = cast(Optional[Sequence[str]], kwargs.get("sample", None))
        if samples is None:
            samples = self.virtual_data["sample"].to_numpy()
            n_samples = self.virtual_data.sizes["sample"]
        else:
            n_samples = len(samples)

        length = end - start

        q = self.table.lazy().filter(
            (pl.col("chrom") == contig)
            & (pl.col("chromStart") < end)
            & (pl.col("chromEnd") > start)
        )

        # get samples in order requested
        if samples is not None:
            q = q.filter(pl.col("name").is_in(samples))
            with pl.StringCache():
                pl.Series(samples, dtype=pl.Categorical)
                q = q.sort(
                    pl.col("name").cast(pl.Categorical), "chromStart", "chromEnd"
                ).collect()
        else:
            with pl.StringCache():
                pl.Series(samples, dtype=pl.Categorical)
                q = q.sort(
                    pl.col("name").cast(pl.Categorical), "chromStart", "chromEnd"
                ).collect()

        cols = q.select(
            pl.col("name").rle_id(),
            pl.col("chromStart") - start,
            pl.col("chromEnd") - start,
            "score",
        ).get_columns()

        out = np.zeros(shape=(n_samples, length), dtype=np.float64)
        sample_idx, starts, ends, vals = [c.to_numpy() for c in cols]
        assign_vals(out, sample_idx, starts, ends, vals)

        return xr.DataArray(out, dims=["sample", "length"], coords={"sample": samples})


def validate_rle_table(df: Union[pl.DataFrame, pl.LazyFrame]):
    _df = df.lazy()

    schema = {
        "chrom": pl.Utf8,
        "chromStart": pl.Int64,
        "chromEnd": pl.Int64,
        "name": pl.Utf8,
        "score": pl.Float64,
    }

    missing_cols = set(schema.keys()) - set(_df.columns)
    if len(missing_cols) > 0:
        raise ValueError(
            f"""RLE table is missing (or has misnamed) expected columns:
            {missing_cols}"""
        )

    mismatched_dtypes = {
        c: (dt1, _df.schema[c]) for c, dt1 in schema.items() if dt1 != _df.schema[c]
    }
    if len(mismatched_dtypes) > 0:
        raise ValueError(
            f"""RLE table has incorrect dtypes.
            Mismatched dtypes (expected, seen): {mismatched_dtypes}"""
        )

    _df = _df.collect()
    if _df.height == 0:
        raise ValueError("No entries in table.")

    null_count = _df.null_count()
    if null_count.to_numpy().sum() > 0:
        raise ValueError(
            f"Table has null values. Number of nulls per column:\n{null_count}"
        )

    duplicated = _df.is_duplicated()
    if duplicated.any():
        raise ValueError(f"Table has {duplicated.sum()} duplicate entries.")

    overlaps = (
        _df.lazy()
        .sort("chromStart", "chromEnd")
        .groupby("name", "chrom")
        .agg(
            pl.col("chromStart", "chromEnd"),
            (
                pl.col("chromStart").shift_and_fill(np.inf, periods=-1)
                < pl.col("chromEnd")
            ).alias("overlapping"),
        )
        .explode("chromStart", "chromEnd", "overlapping")
        .filter(pl.col("overlapping") | pl.col("overlapping").shift(1))
        .sort("name", "chrom", "chromStart", "chromEnd")
        .collect()
    )
    if overlaps.height > 0:
        raise ValueError(f"Found overlapping intervals in the table:\n{overlaps}")

    return _df
