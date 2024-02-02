from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numba as nb
import numpy as np
import polars as pl
import xarray as xr
from numpy.typing import NDArray

from .types import Reader


class Intervals(Reader):
    chunked = False

    def __init__(
        self,
        name: str,
        table: Union[str, Path, pl.DataFrame],
        subset: Optional[Mapping[str, Sequence[Any]]] = None,
        value_columns: Optional[Sequence[str]] = None,
        value_dim: Optional[str] = None,
        validate=True,
        zero_indexed=True,
        lazy=True,
        **table_reader_kwargs,
    ):
        """Read values from a table of intervals. This corresponds to the
        UCSC BED3+ format, the first three columns must be `chrom`, `chromStart`, and
        `chromEnd`, and other columns can be anything. By default, the 4th column is
        used as the value column, but this can be changed with `value_columns`. Any
        columns that aren't a value column are treated as grouping columns, and can be
        used to subset the table with `subset`. The order of the columns will determine
        the order of the dimensions in the output array:
        - If there are no grouping columns, the output array will have shape
        `(n_values, length)` and the order will match the order of `value_columns`.
        - If there are grouping columns, the output array will have shape
        `(group_1, ..., group_n, n_values, length)` and the order of each group will
        match the order of `subset`.

        Parameters
        ----------
        name : str
            Name of the reader.
        table : Union[str, Path, pl.DataFrame]
            DataFrame, or path to one, in UCSC BED3+ format.
        subset : Mapping[str, Sequence[str]], optional
            Mapping of column names to groups of interest, by default None (all groups
            used). For example, if there was a column "name" with sample names, to
            only include samples "A" and "B" `subset={"name": ["A", "B"]}`.
        value_columns : Optional[Sequence[str]], optional
            Which columns to use as values, by default None (4th column used).
        validate : bool, optional
            Whether to validate the RLE table, by default True.
        zero_indexed : bool, optional
            Whether the coordinates are 0-indexed, by default True
        lazy : bool, optional
            Whether to hold the table in-memory or query it lazily, leaving it on-disk
            to reduce memory usage at the cost of query speed. Ignored if `table` is a
            DataFrame.
        **table_reader_kwargs
            Passed to the table reader function (either pl.scan_csv or pl.scan_ipc). This
            can be used to rename columns.
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

        if value_columns is None:
            value_columns = [_table.columns[3]]

        if subset is None:
            grouping_cols = get_grouping_cols(_table, value_columns)
            _subset = {
                c: _table.lazy()
                .select(c)
                .unique(maintain_order=True)
                .collect()[c]
                .to_numpy()
                for c in grouping_cols
            }
        else:
            grouping_cols = [c for c in subset.keys()]
            for col, groups in subset.items():
                _table = _table.filter(pl.col(col).is_in(groups))
            _subset = {c: np.asarray(group) for c, group in subset.items()}

        if value_dim is None:
            value_dim = "values"

        _subset[value_dim] = np.asarray(value_columns)

        if validate:
            _table = validate(_table, grouping_cols, value_columns).lazy()

        self.name = name
        self.value_dim = value_dim
        self.value_columns = value_columns
        self.table = _table
        self.dtype = (
            _table.lazy().select(value_columns[0]).head(1).collect().to_numpy().dtype
        )
        self.coords = _subset
        self.sizes = {c: len(g) for c, g in _subset.items()}
        contigs = self.table.lazy().select(pl.col("chrom").unique()).collect()["chrom"]
        self.contig_starts_with_chr = self.infer_contig_prefix(contigs)

    def rev_strand_fn(self, arr: NDArray):
        return arr[..., ::-1]

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        groups = {}
        sizes = {}
        for c in self.coords:
            if c == self.value_dim:
                continue
            groups[c] = kwargs.get(c, self.coords[c])
            sizes[c] = len(groups[c])
        subset = bool(set(kwargs.keys()) & set(self.coords.keys()))

        contig = self.normalize_contig_name(contig)
        length = end - start

        q = self.table.lazy().filter(
            (pl.col("chrom") == contig)
            & (pl.col("chromStart") < end)
            & (pl.col("chromEnd") > start)
        )

        # get groups in order requested
        if subset:
            q = q.filter(*(pl.col(c).is_in(g) for c, g in groups.items()))
            with pl.StringCache():
                for c, g in groups.items():
                    pl.Series(g, dtype=pl.Categorical)
                q = q.sort(
                    pl.col(*groups.keys()).cast(pl.Categorical),
                    "chromStart",
                    "chromEnd",
                )

        q = q.collect()

        cols = q.select(
            # TODO: generalize getting group idx to multiple columns
            pl.col("name").rle_id().alias("sample_idx"),
            pl.col("chromStart") - start,
            pl.col("chromEnd") - start,
            *self.value_columns,
        ).get_columns()

        out = np.zeros(
            shape=(*self.sizes.values(), len(self.value_columns), length),
            dtype=self.dtype,
        ).squeeze()
        sample_idx, starts, ends, vals = [c.to_numpy() for c in cols]
        assign_vals(out, sample_idx, starts, ends, vals)

        dims = list(self.sizes.keys()) + ["values", "length"]
        return xr.DataArray(out, dims=dims)


def get_grouping_cols(
    df: Union[pl.DataFrame, pl.LazyFrame], value_columns: Sequence[str]
):
    bed3_cols = ["chrom", "chromStart", "chromEnd"]
    grouping_cols = [c for c in df.columns if c not in list(value_columns) + bed3_cols]
    return grouping_cols


def validate(
    df: Union[pl.DataFrame, pl.LazyFrame],
    grouping_cols: Sequence[str],
    value_columns: Sequence[str],
):
    _df = df.lazy()

    df_bed3_schema = _df.select("chrom", "chromStart", "chromEnd").schema

    bed3_schema = {
        "chrom": pl.Utf8,
        "chromStart": pl.Int64,
        "chromEnd": pl.Int64,
    }

    missing_cols = set(bed3_schema.keys()) - set(df_bed3_schema.keys())
    if len(missing_cols) > 0:
        raise ValueError(
            f"""RLE table is missing (or has misnamed) BED3 columns:
            {missing_cols}"""
        )

    mismatched_dtypes = {
        c: (dt1, df_bed3_schema[c])
        for c, dt1 in bed3_schema.items()
        if dt1 != df_bed3_schema[c]
    }
    if mismatched_dtypes:
        raise ValueError(
            f"""RLE table has incorrect dtypes for BED3 columns.
            Mismatched dtypes (expected, seen): {mismatched_dtypes}"""
        )

    val_schema = _df.select(*value_columns).schema
    first_dtype = next(iter(val_schema.values()))
    mismatched_dtypes = {
        c: (first_dtype, dt1) for c, dt1 in val_schema.items() if dt1 != first_dtype
    }
    if mismatched_dtypes:
        raise ValueError(
            f"""RLE table has incorrect dtypes for value columns.
            Mismatched dtypes (expected, seen): {mismatched_dtypes}"""
        )

    non_numeric_cols = [c for c, dt in val_schema.items() if not dt.is_numeric()]
    if non_numeric_cols:
        raise ValueError(
            f"""RLE table has non-numeric value columns.
            Value columns: {non_numeric_cols}"""
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
        .groupby(*grouping_cols, "chrom")
        .agg(
            pl.col("chromStart", "chromEnd"),
            (
                pl.col("chromStart").shift(-1, fill_value=np.inf) < pl.col("chromEnd")
            ).alias("overlapping"),
        )
        .explode("chromStart", "chromEnd", "overlapping")
        .filter(pl.col("overlapping") | pl.col("overlapping").shift(1))
        .collect()
    )
    if overlaps.height > 0:
        raise ValueError(f"Found overlapping intervals in the table:\n{overlaps}")

    return _df


@nb.njit(nogil=True, parallel=True, cache=True)
def assign_vals(out, sample_idx, starts, ends, vals):
    # all intervals in an RLE table are guaranteed to be non-overlapping
    for i in nb.prange(len(starts)):
        out[sample_idx[i], starts[i] : ends[i]] = vals[i]
