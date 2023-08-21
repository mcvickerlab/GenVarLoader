from pathlib import Path
from typing import Optional, Sequence, Union

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
    ):
        """Read values from a run-length encoded (RLE) table. This corresponds to the
        UCSC BED5+ format, where sample names are found in the `name` column and values
        for each interval are in the `score` column. Note this currently keeps the table
        in-memory but is very fast.

        Parameters
        ----------
        name : str
            Name of the reader.
        df : Union[str, Path, pl.DataFrame]
            DataFrame, or path to one, in the UCSC BED5+ format.
        samples : Sequence[str], optional
            Names of samples to potentially include in calls to `read()`. By default all
            samples are used.
        validate : bool, optional
            Whether to validate the RLE table, by default True.
        zero_indexed : bool, optional
            Whether the coordinates are 0-indexed, by default True
        """

        if isinstance(table, (str, Path)):
            _table = pl.scan_csv(table)
        else:
            _table = table.lazy()

        if not zero_indexed:
            _table = _table.with_columns(pl.col("chromStart") - 1)

        if validate:
            _table = validate_rle_table(_table).lazy()

        if samples is not None:
            _table = _table.filter(pl.col("sample").is_in(samples))
            with pl.StringCache():
                pl.Series(samples, dtype=pl.Categorical)
                _table = _table.sort(
                    pl.col("sample").cast(pl.Categorical), "chromStart", "end"
                )

        _table = _table.collect()
        _samples = _table["sample"].unique(maintain_order=True)

        self.name = name
        self.table = _table
        self.virtual_data = xr.DataArray(
            da.empty(len(_samples), dtype=np.float64),
            name=name,
            dims="sample",
            coords={"sample": _samples.to_numpy()},
        )

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        samples = kwargs.get("samples", None)
        if samples is None:
            n_samples = self.virtual_data.sizes["sample"]
        else:
            n_samples = len(samples)

        length = end - start

        q = self.table.lazy().filter(
            (pl.col("contig") == contig)
            & ((pl.col("chromStart") < end) | (pl.col("end") > start))
        )

        # get samples in order requested
        if samples is not None:
            q = q.filter(pl.col("sample").is_in(samples))
            with pl.StringCache():
                pl.Series(samples, dtype=pl.Categorical)
                q = q.sort(pl.col("sample").cast(pl.Categorical), "chromStart", "end")

        cols = (
            q.select(
                pl.col("sample").rle_id(),
                pl.col("chromStart") - start,
                pl.col("end") - start,
                "value",
            )
            .collect()
            .get_columns()
        )

        out = np.zeros(shape=(n_samples, length), dtype=np.float64)
        sample_idx, starts, ends, vals = [c.to_numpy() for c in cols]
        assign_vals(out, sample_idx, starts, ends, vals)

        return xr.DataArray(out, dims=["sample", "length"], coords={"sample": samples})


def validate_rle_table(df: pl.LazyFrame):
    schema = {
        "chrom": pl.Utf8,
        "chromStart": pl.Int64,
        "chromEnd": pl.Int64,
        "name": pl.Utf8,
        "score": pl.Float64,
    }

    missing_cols = set(schema.keys()) - set(df.columns)
    if len(missing_cols) > 0:
        raise ValueError(
            f"""RLE table is missing (or has misnamed) expected columns:
            {missing_cols}"""
        )

    dtypes = {c: dt for c, dt in df.schema.items() if c in schema.keys()}
    if sum(schema[c] != df.schema[c] for c in schema.keys()) > 0:
        raise ValueError(
            f"""RLE table has incorrect dtypes.\n
            Expected: {schema}\n
            Received: {dtypes}"""
        )

    _df = df.collect()
    if _df.height == 0:
        raise ValueError("No entries in table.")

    null_count = _df.null_count()
    if null_count.to_numpy().sum() > 0:
        raise ValueError(
            f"Table has null values. Number of nulls per column:\n{null_count}"
        )

    overlaps = (
        _df.lazy()
        .sort("chromStart", "chromEnd")
        .groupby("sample", "chrom")
        .agg(
            pl.col("chromStart", "chromEnd"),
            (
                pl.col("chromStart").shift_and_fill(np.inf, periods=-1)
                < pl.col("chromEnd")
            ).alias("overlapping"),
        )
        .explode("chromStart", "chromEnd", "overlapping")
        .filter(pl.col("overlapping") | pl.col("overlapping").shift(1))
        .sort("sample", "chrom", "chromStart", "chromEnd")
        .collect()
    )
    if overlaps.height > 0:
        raise ValueError(f"Found overlapping intervals in the table:\n{overlaps}")

    return _df
