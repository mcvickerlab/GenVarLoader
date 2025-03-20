from itertools import accumulate, chain, repeat
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    TypeGuard,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandera as pa
import pandera.typing as pat
import polars as pl
from numpy.typing import NDArray

from ._types import DTYPE, Idx

__all__ = [
    "read_bedlike",
    "with_length",
]

T = TypeVar("T")


def is_dtype(arr: NDArray, dtype: type[DTYPE]) -> TypeGuard[NDArray[DTYPE]]:
    return np.issubdtype(arr.dtype, dtype)


def _process_bed(bed: Union[str, Path, pl.DataFrame], fixed_length: int):
    if isinstance(bed, (str, Path)):
        bed = read_bedlike(bed)

    if "strand" in bed and bed["strand"].dtype == pl.Utf8:
        bed = bed.with_columns(
            pl.col("strand").replace({"-": -1, "+": 1}, return_dtype=pl.Int8)
        )
    else:
        bed = bed.with_columns(strand=pl.lit(1, dtype=pl.Int8))

    if "region_idx" not in bed:
        bed = bed.with_row_count("region_idx")

    bed = bed.sort("chrom", "chromStart")

    return with_length(bed, fixed_length)


def with_length(bed: pl.DataFrame, length: int) -> pl.DataFrame:
    """Expands or shrinks each region in a BED-like DataFrame to be fixed-length windows
    centered around the midpoint of each region or the "peak" column if it is present.

    .. important::

        The "peak" column is described in the `narrowPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format12>`_
        and `broadPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format13>`_ specifications. It is a 0-based
        offset from chromStart, so be sure not to encode your "peak" column as an absolute position!

    Parameters
    ----------
    bed
        BED-like DataFrame with at least the columns "chrom", "chromStart", and "chromEnd".
    length
        Length of the fixed-length windows.
    """
    if "peak" in bed:
        center = pl.col("chromStart") + pl.col("peak")
    else:
        center = (pl.col("chromStart") + pl.col("chromEnd")) // 2
    left = length // 2
    right = length - left
    bed = bed.with_columns(
        chromStart=(center - left).cast(pl.Int64),
        chromEnd=(center + right).cast(pl.Int64),
    )
    return bed


def _random_chain(
    *iterables: Iterable[T], seed: Optional[int] = None
) -> Generator[T, None, None]:
    """Chain iterables, randomly sampling from each until they are all exhausted."""
    rng = np.random.default_rng(seed)
    iterators = {i: iter(it) for i, it in enumerate(iterables)}
    while iterators:
        i = rng.choice(list(iterators.keys()))
        try:
            yield next(iterators[i])
        except StopIteration:
            del iterators[i]


def read_bedlike(path: Union[str, Path]) -> pl.DataFrame:
    """Reads a bed-like (i.e. `BED3+ <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`_) file as a
    polars DataFrame. The file type is inferred from the file extension. "Bed-like" refers to files
    with extension :code:`.bed`, :code:`.narrowPeak`, :code:`.broadPeak`, or otherwise a tabular (CSV, TSV, or feather/arrow)
    file with at least the BED3 columns. For tabular data, extra columns that are not part of the BED
    specification are allowed and kept in the resulting DataFrame.

    Parameters
    ----------
    path
        Path to the file.
    """
    path = Path(path)
    if ".bed" in path.suffixes:
        return _read_bed(path)
    elif ".narrowPeak" in path.suffixes:
        return _read_narrowpeak(path)
    elif ".broadPeak" in path.suffixes:
        return _read_broadpeak(path)
    else:
        try:
            return _read_bed_table(path)
        except ValueError:
            raise ValueError(
                f"""Unrecognized file extension: {"".join(path.suffixes)}. Expected one 
                of .bed, .narrowPeak, .broadPeak, or a table file (e.g. .csv, .tsv)"""
            )


class _BEDSchema(pa.DataFrameModel):
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: Optional[pat.Series[str]] = pa.Field(nullable=True)
    score: Optional[pat.Series[float]] = pa.Field(nullable=True)
    strand: Optional[pat.Series[str]] = pa.Field(isin=["+", "-", "."], nullable=True)
    thickStart: Optional[pat.Series[int]] = pa.Field(nullable=True)
    thickEnd: Optional[pat.Series[int]] = pa.Field(nullable=True)
    itemRgb: Optional[pat.Series[str]] = pa.Field(nullable=True)
    blockCount: Optional[pat.Series[pa.UInt]] = pa.Field(nullable=True)
    blockSizes: Optional[pat.Series[str]] = pa.Field(nullable=True)
    blockStarts: Optional[pat.Series[str]] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_bed(bed_path: Union[str, Path]):
    with open(bed_path) as f:
        skip_rows = 0
        while (line := f.readline()).startswith(("track", "browser")):
            skip_rows += 1
    n_cols = line.count("\t") + 1
    bed_cols = [
        "chrom",
        "chromStart",
        "chromEnd",
        "name",
        "score",
        "strand",
        "thickStart",
        "thickEnd",
        "itemRgb",
        "blockCount",
        "blockSizes",
        "blockStarts",
    ]
    bed = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=bed_cols[:n_cols],
        schema_overrides={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    bed = _BEDSchema.to_schema()(bed)
    return pl.from_pandas(bed)


def _read_bed_table(table: Union[str, Path], **table_reader_kwargs):
    table = Path(table)
    suffixes = set(table.suffixes)
    reader_kwargs: Dict[str, Any] = {}
    reader_kwargs.update(table_reader_kwargs)
    if ".csv" in suffixes:
        reader_kwargs["separator"] = ","
        reader_kwargs["dtypes"] = {"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8}
        reader = pl.scan_csv
    elif {".txt", ".tsv"} & suffixes:
        reader_kwargs["separator"] = "\t"
        reader_kwargs["dtypes"] = {"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8}
        reader = pl.scan_csv
    elif {".fth", ".feather", ".ipc", ".arrow"} & suffixes:
        reader = pl.scan_ipc  # type : ignore[assignment]
    else:
        raise ValueError(f"Table has unrecognized file extension: {table.name}")
    bed = reader(table, **reader_kwargs).collect().to_pandas()
    bed = _BEDSchema.to_schema()(bed)
    return pl.from_pandas(bed)


class _NarrowPeakSchema(pa.DataFrameModel):
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: pat.Series[str] = pa.Field(nullable=True)
    score: pat.Series[float] = pa.Field(nullable=True)
    strand: pat.Series[str] = pa.Field(isin=["+", "-", "."], nullable=True)
    signalValue: pat.Series[float] = pa.Field(nullable=True)
    pValue: pat.Series[float] = pa.Field(nullable=True)
    qValue: pat.Series[float] = pa.Field(nullable=True)
    peak: pat.Series[int] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_narrowpeak(narrowpeak_path: Union[str, Path]):
    with open(narrowpeak_path) as f:
        skip_rows = 0
        while f.readline().startswith(("track", "browser")):
            skip_rows += 1
    narrowpeaks = pl.read_csv(
        narrowpeak_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=[
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
        schema_overrides={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    narrowpeaks = _NarrowPeakSchema.to_schema()(narrowpeaks)
    return pl.from_pandas(narrowpeaks)


class _BroadPeakSchema(pa.DataFrameModel):
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: pat.Series[str] = pa.Field(nullable=True)
    score: pat.Series[float] = pa.Field(nullable=True)
    strand: pat.Series[str] = pa.Field(isin=["+", "-", "."], nullable=True)
    signalValue: pat.Series[float] = pa.Field(nullable=True)
    pValue: pat.Series[float] = pa.Field(nullable=True)
    qValue: pat.Series[float] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_broadpeak(broadpeak_path: Union[str, Path]):
    with open(broadpeak_path) as f:
        skip_rows = 0
        while f.readline().startswith(("track", "browser")):
            skip_rows += 1
    broadpeaks = pl.read_csv(
        broadpeak_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=[
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
        schema_overrides={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    broadpeaks = _BroadPeakSchema.to_schema()(broadpeaks)
    return pl.from_pandas(broadpeaks)


def _cartesian_product(arrays: Sequence[NDArray]) -> NDArray:
    """Get the cartesian product of multiple arrays such that each entry corresponds to
    a unique combination of the input arrays' values.
    """
    # https://stackoverflow.com/a/49445693
    la = len(arrays)
    shape = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(shape, dtype=dtype)
    arrs = (*accumulate(chain((arr,), repeat(0, la - 1)), np.ndarray.__getitem__),)
    idx = slice(None), *repeat(None, la - 1)
    for i in range(la - 1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[: la - i]]
        arrs[i - 1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)


def _get_rel_starts(
    starts: NDArray[np.int64], ends: NDArray[np.int64]
) -> NDArray[np.int64]:
    rel_starts: NDArray[np.int64] = np.concatenate(
        [[0], (ends - starts).cumsum()[:-1]], dtype=np.int64
    )
    return rel_starts


DTYPE = TypeVar("DTYPE", bound=np.generic)


def _normalize_contig_name(contig: str, contigs: Iterable[str]) -> Optional[str]:
    """Normalize the contig name to adhere to the convention of the underlying file.
    i.e. remove or add "chr" to the contig name.

    Parameters
    ----------
    contig : str

    Returns
    -------
    str
        Normalized contig name.
    """
    for c in contigs:
        # exact match, remove chr, add chr
        if contig == c or contig[3:] == c or f"chr{contig}" == c:
            return c
    return None


ITYPE = TypeVar("ITYPE", bound=np.integer)


def _offsets_to_lengths(offsets: NDArray[ITYPE]) -> NDArray[ITYPE]:
    """Converts offsets to the number of elements in each group.

    Notes
    -----
    This function will silently fail with wraparound values if the offsets are
    not sorted in ascending order."""
    return np.diff(offsets)


def _lengths_to_offsets(
    lengths: NDArray[np.integer], dtype: type[ITYPE] = np.int64
) -> NDArray[ITYPE]:
    """Converts the number of elements in each group to a 1D array of offsets."""
    offsets = np.empty(lengths.size + 1, dtype=dtype)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return offsets


def idx_like_to_array(idx: Idx, max_len: int) -> NDArray[np.intp]:
    """Convert an index-like object to an array of non-negative indices. Shapes of multi-dimensional
    indices are preserved."""
    if isinstance(idx, slice):
        _idx = np.arange(max_len, dtype=np.intp)[idx]
    elif isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, np.bool_):
        _idx = idx.nonzero()[0]
    elif isinstance(idx, Sequence):
        _idx = np.asarray(idx, np.intp)
    else:
        _idx = idx

    if isinstance(_idx, (int, np.integer)):
        _idx = np.array([_idx], np.intp)

    # unable to type narrow from NDArray[bool] since it's a generic type
    _idx = cast(NDArray[np.intp], _idx)

    # handle negative indices
    _idx[_idx < 0] += max_len

    return _idx
