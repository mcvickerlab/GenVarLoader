from itertools import accumulate, chain, repeat
from pathlib import Path
from typing import (
    Any,
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
import polars as pl
from numpy.typing import NDArray
from seqpro.bed import read_bedlike, with_length

from ._types import DTYPE, Idx

__all__ = [
    "read_bedlike",
    "with_length",
]

T = TypeVar("T")


def is_dtype(arr: Any, dtype: type[DTYPE]) -> TypeGuard[NDArray[DTYPE]]:
    return isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, dtype)


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
    elif is_dtype(idx, np.bool_):
        _idx = idx.nonzero()[0]
    elif isinstance(idx, Sequence):
        _idx = np.array(idx, np.intp)
    else:
        _idx = idx

    if isinstance(_idx, (int, np.integer)):
        _idx = np.array([_idx], np.intp)

    # unable to type narrow from NDArray[bool] since it's a generic type
    _idx = cast(NDArray[np.intp], _idx)

    # handle negative indices
    _idx[_idx < 0] += max_len

    return _idx
