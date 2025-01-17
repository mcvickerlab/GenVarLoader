from typing import Sequence, Tuple

import numba as nb
import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from .._types import Idx
from .._utils import DTYPE

__all__ = []


@nb.njit(nogil=True, cache=True)
def padded_slice(arr: NDArray, start: int, stop: int, pad_val: int):
    pad_left = -min(0, start)
    pad_right = max(0, stop - len(arr))

    if pad_left == 0 and pad_right == 0:
        out = arr[start:stop]
        return out

    out = np.empty(stop - start, arr.dtype)

    if pad_left > 0 and pad_right > 0:
        out_stop = len(out) - pad_right
        out[:pad_left] = pad_val
        out[pad_left:out_stop] = arr[:]
        out[out_stop:] = pad_val
    elif pad_left > 0:
        out[:pad_left] = pad_val
        out[pad_left:] = arr[:stop]
    elif pad_right > 0:
        out_stop = len(out) - pad_right
        out[:out_stop] = arr[start:]
        out[out_stop:] = pad_val

    return out


def idx_like_to_array(idx: Idx, max_len: int) -> NDArray[np.integer]:
    """Convert an index-like object to an array of non-negative indices. Shapes of multi-dimensional
    indices are preserved."""
    if isinstance(idx, slice):
        start = 0 if idx.start is None else idx.start
        stop = max_len if idx.stop is None else idx.stop
        _idx = np.arange(start, stop, idx.step, np.intp)
    elif isinstance(idx, Sequence):
        _idx = np.asarray(idx, np.intp)
    else:
        _idx = idx

    # handle negative indices
    if isinstance(_idx, (int, np.integer)):
        _idx = np.array([_idx], np.intp)

    _idx[_idx < 0] += max_len

    return _idx


def oidx_to_raveled_idx(row_idx: ArrayLike, col_idx: ArrayLike, shape: Tuple[int, int]):
    row_idx = np.asarray(row_idx)
    col_idx = np.asarray(col_idx)
    full_array_linear_indices = np.ravel_multi_index(
        (row_idx[:, None], col_idx), shape
    ).ravel()
    return full_array_linear_indices


def regions_to_bed(regions: NDArray[np.int32], contigs: Sequence[str]) -> pl.DataFrame:
    """Convert regions to a BED3 DataFrame.

    Parameters
    ----------
    regions : NDArray
        Shape = (n_regions, 3) Regions.
    contigs : Sequence[str]
        Contigs.

    Returns
    -------
    pl.DataFrame
        Bed DataFrame.
    """
    cols = ["chrom", "chromStart", "chromEnd", "strand"]
    bed = pl.DataFrame(regions, schema=cols)
    cmap = dict(enumerate(contigs))
    bed = bed.with_columns(
        pl.col("chrom").replace_strict(cmap, return_dtype=pl.Utf8),
        pl.col("chromStart", "chromEnd").cast(pl.Int64),
        pl.col("strand").replace_strict({1: "+", -1: "-"}, return_dtype=pl.Utf8),
    ).select(cols)
    return bed


@nb.njit(nogil=True, cache=True)
def splits_sum_le_value(arr: NDArray[np.number], max_value: float) -> NDArray[np.intp]:
    """Get index offsets for groups that sum to no more than a value.
    Note that values greater than the maximum will be kept in their own group.

    Parameters
    ----------
    arr : NDArray[np.number]
        Array to split.
    max_value : float
        Maximum value.

    Returns
    -------
    NDArray[np.intp]
        Split indices.

    Examples
    --------
    >>> splits_sum_le_value(np.array([5, 5, 11, 9, 2, 7]), 10)
    # (5 5) (11) (9) (2 7)
    array([0, 2, 3, 4, 6])
    """
    indices = [0]
    current_sum = 0
    for idx, value in enumerate(arr):
        current_sum += value
        if current_sum > max_value:
            indices.append(idx)
            current_sum = value
    indices.append(len(arr))
    return np.array(indices, np.intp)


def reduceat_offsets(
    ufunc: np.ufunc, arr: NDArray[DTYPE], offsets: NDArray[np.integer], axis: int = 0
) -> NDArray[DTYPE]:
    """Reduce an array at offsets.

    Parameters
    ----------
    ufunc : np.ufunc
        Ufunc.
    arr : NDArray[np.number]
        Array to reduce.
    offsets : NDArray[np.int32]
        Offsets.
    axis : int, optional
        Axis, by default 0.

    Returns
    -------
    out_array
        Reduced array.
    """
    n_reductions = len(offsets) - 1

    if axis < 0:
        axis = arr.ndim + axis

    no_var_idx = np.searchsorted(offsets, offsets[-1])

    # ensure arr and out are aligned and of same dtype to (hopefully) avoid a copy
    # https://numpy.org/doc/stable/dev/internals.code-explanations.html#reduceat
    out_arr = np.empty(
        (*arr.shape[:axis], n_reductions, *arr.shape[axis + 1 :]), arr.dtype
    )
    indices = [slice(None)] * arr.ndim
    indices[axis] = slice(None, no_var_idx)
    indices = tuple(indices)
    ufunc.reduceat(arr, offsets[:no_var_idx], axis=axis, out=out_arr[indices])

    identity_indices = [slice(None)] * arr.ndim
    identity_indices[axis] = slice(no_var_idx, None)
    identity_indices = tuple(identity_indices)
    out_arr[identity_indices] = ufunc.identity
    return out_arr.swapaxes(axis, -1)
