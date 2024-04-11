from itertools import accumulate, chain, repeat
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import pandera as pa
import pandera.typing as pat
import polars as pl
from numpy.typing import NDArray

T = TypeVar("T")


def process_bed(bed: Union[str, Path, pl.DataFrame], fixed_length: int):
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


def with_length(bed: pl.DataFrame, length: int):
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


def random_chain(
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
    """Reads a bed-like (BED3+) file as a pandas DataFrame. The file type is inferred
    from the file extension.

    Parameters
    ----------
    path : Union[str, Path]

    Returns
    -------
    polars.DataFrame
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
                f"""Unrecognized file extension: {''.join(path.suffixes)}. Expected one 
                of .bed, .narrowPeak, .broadPeak, or a table file (e.g. .csv, .tsv)"""
            )


class BEDSchema(pa.DataFrameModel):
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
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    bed = BEDSchema.to_schema()(bed)
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
    bed = BEDSchema.to_schema()(bed)
    return pl.from_pandas(bed)


class NarrowPeakSchema(pa.DataFrameModel):
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
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    narrowpeaks = NarrowPeakSchema.to_schema()(narrowpeaks)
    return pl.from_pandas(narrowpeaks)


class BroadPeakSchema(pa.DataFrameModel):
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
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    broadpeaks = BroadPeakSchema.to_schema()(broadpeaks)
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


def get_rel_starts(
    starts: NDArray[np.int64], ends: NDArray[np.int64]
) -> NDArray[np.int64]:
    rel_starts: NDArray[np.int64] = np.concatenate(
        [[0], (ends - starts).cumsum()[:-1]], dtype=np.int64
    )
    return rel_starts


DTYPE = TypeVar("DTYPE", bound=np.generic)


def splice_subarrays(
    arr: NDArray[DTYPE], starts: NDArray[np.int64], ends: NDArray[np.int64]
) -> NDArray[DTYPE]:
    """Splice subarrays from a larger array.

    Parameters
    ----------
    arr : NDArray
        Array to splice from.
    starts : NDArray[np.int64]
        Start coordinates, 0-based.
    ends : NDArray[np.int64]
        End coordinates, 0-based exclusive.

    Returns
    -------
    out : NDArray
        Spliced array.
    """
    start = starts.min()
    rel_starts = get_rel_starts(starts, ends)
    total_length = (ends - starts).sum()
    shape = arr.shape[:-1] + (total_length,)
    out = np.empty(shape, dtype=arr.dtype)
    for rel_start, s, e in zip(rel_starts, starts - start, ends - start):
        length = e - s
        out[..., rel_start : rel_start + length] = arr[..., s:e]
    return out


def splice_and_rc_subarrays(
    arr: NDArray[DTYPE],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    strands: NDArray[np.int8],
    rc_fn: Callable[[NDArray[DTYPE]], NDArray[DTYPE]],
) -> NDArray[DTYPE]:
    """Splice subarrays from a larger array and reverse-complement them.

    Parameters
    ----------
    arr : NDArray
        Array to splice from.
    starts : NDArray[np.int64]
        Start coordinates, 0-based.
    ends : NDArray[np.int64]
        End coordinates, 0-based exclusive.
    strands : NDArray[np.int8]
        Strand of each query region. 1 for forward, -1 for reverse. If None, defaults
        to forward strand.
    rc_fn : Callable[[NDArray], NDArray]
        Function to reverse-complement the subarrays.

    Returns
    -------
    out : NDArray
        Spliced and reverse-complemented array.
    """
    start = starts.min()
    rel_starts = get_rel_starts(starts, ends)
    total_length = (ends - starts).sum()
    shape = arr.shape[:-1] + (total_length,)
    out = np.empty(shape, dtype=arr.dtype)
    for rel_start, s, e, strand in zip(
        rel_starts, starts - start, ends - start, strands
    ):
        length = e - s
        if strand == 1:
            subarr = arr[..., s:e]
        else:
            subarr = rc_fn(arr[..., s:e])
        out[..., rel_start : rel_start + length] = subarr
    return out


def splice_and_rev_subarrays(
    arr: NDArray[DTYPE],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    strands: NDArray[np.int8],
) -> NDArray[DTYPE]:
    """
    Splices and reverses subarrays of a given array based on the start and end indices
    and strand orientation.

    Parameters
    ----------
    arr : NDArray
        Array to splice from.
    starts : NDArray[np.int64]
        Start coordinates, 0-based.
    ends : NDArray[np.int64]
        End coordinates, 0-based exclusive.
    strands : NDArray[np.int8]
        Strand of each query region. 1 for forward, -1 for reverse. If None, defaults
        to forward strand.

    Returns
    -------
    out : NDArray
        Spliced and reversed array.
    """
    start = starts.min()
    rel_starts = get_rel_starts(starts, ends)
    total_length = (ends - starts).sum()
    shape = arr.shape[:-1] + (total_length,)
    out = np.empty(shape, dtype=arr.dtype)
    for rel_start, s, e, strand in zip(
        rel_starts, starts - start, ends - start, strands
    ):
        length = e - s
        out[..., rel_start : rel_start + length] = arr[..., s:e:strand]
    return out


def normalize_contig_name(contig: str, contigs: Iterable[str]) -> Optional[str]:
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


def offsets_to_lengths(offsets: NDArray[ITYPE]) -> NDArray[ITYPE]:
    """Converts offsets to the number of elements in each group.

    Notes
    -----
    This function will silently fail with wraparound values if the offsets are
    not sorted in ascending order."""
    return np.diff(offsets)


def lengths_to_offsets(
    lengths: NDArray[np.integer], dtype: type[ITYPE] = np.int32
) -> NDArray[ITYPE]:
    """Converts the number of elements in each group to offsets."""
    offsets = np.empty(len(lengths) + 1, dtype=dtype)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return offsets
