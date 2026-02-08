from collections.abc import Iterable, Sequence
from typing import Any, TypeGuard, TypeVar, cast, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ._types import DTYPE, Idx


def is_dtype(arr: Any, dtype: type[DTYPE]) -> TypeGuard[NDArray[DTYPE]]:
    return isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, dtype)


def get_rel_starts(
    starts: NDArray[np.int64], ends: NDArray[np.int64]
) -> NDArray[np.int64]:
    rel_starts: NDArray[np.int64] = np.concatenate(
        [[0], (ends - starts).cumsum()[:-1]], dtype=np.int64
    )
    return rel_starts


@overload
def normalize_contig_name(contig: str, contigs: Iterable[str]) -> str | None: ...
@overload
def normalize_contig_name(
    contig: list[str], contigs: Iterable[str]
) -> list[str | None]: ...
def normalize_contig_name(
    contig: str | list[str], contigs: Iterable[str]
) -> str | None | list[str | None]:
    """Normalize the contig name to match the naming scheme of `contigs`.

    Parameters
    ----------
    contig : str
        Contig name to normalize.
    contigs : Iterable[str]
        Collection of contig names to normalize against.
    """
    _contigs = (
        {f"{c[3:]}": c for c in contigs if c.startswith("chr")}
        | {f"chr{c}": c for c in contigs if not c.startswith("chr")}
        | {c: c for c in contigs}
    )
    if isinstance(contig, str):
        return _contigs.get(contig, None)
    else:
        return [_contigs.get(c, None) for c in contig]


ITYPE = TypeVar("ITYPE", bound=np.integer)


def lengths_to_offsets(
    lengths: NDArray[np.integer], dtype: type[ITYPE] = np.int64
) -> NDArray[ITYPE]:
    """Converts the number of elements in each group to a 1D array of offsets."""
    offsets = np.empty(lengths.size + 1, dtype=dtype)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return offsets


def idx_like_to_array(idx: Idx, max_len: int) -> NDArray[np.integer]:
    """Convert an index-like object to an array of non-negative indices. Shapes of multi-dimensional
    indices are preserved."""
    if isinstance(idx, (Sequence, pl.Series)):
        idx = cast(NDArray, np.array(idx))
        assert is_dtype(idx, np.integer) or is_dtype(idx, np.bool_)

    if isinstance(idx, slice):
        _idx = np.arange(max_len, dtype=np.intp)[idx]
    elif is_dtype(idx, np.bool_):
        _idx = idx.nonzero()[0]
    else:
        _idx = idx

    if (
        isinstance(_idx, (int, np.integer))
        or isinstance(_idx, np.ndarray)
        and _idx.ndim == 0
    ):
        _idx = np.atleast_1d(_idx)

    # unable to type narrow from NDArray[bool] since it's a generic type
    _idx = cast(NDArray[np.integer], _idx)

    # handle negative indices
    _idx[_idx < 0] += max_len

    return _idx
