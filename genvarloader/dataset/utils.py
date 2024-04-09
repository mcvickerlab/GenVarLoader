from typing import Sequence, Tuple

import numba as nb
import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray


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


def subset_to_full_raveled_mapping(
    full_shape: Tuple[int, int], ax1_indices: ArrayLike, ax2_indices: ArrayLike
):
    # Generate a grid of indices for the subset array
    row_indices, col_indices = np.meshgrid(ax1_indices, ax2_indices, indexing="ij")

    # Flatten the grid to get all combinations of row and column indices in the subset
    row_indices_flat = row_indices.ravel()
    col_indices_flat = col_indices.ravel()

    # Convert these subset indices to linear indices in the context of the full array
    # This leverages the fact that the linear index in a 2D array is given by: index = row * num_columns + column
    full_array_linear_indices = row_indices_flat * full_shape[1] + col_indices_flat

    return full_array_linear_indices


def regions_to_bed(regions: NDArray, contigs: Sequence[str]) -> pl.DataFrame:
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
    bed = pl.DataFrame(
        regions, schema=["chrom", "chromStart", "chromEnd"]
    ).with_columns(pl.all().cast(pl.Int64))
    cmap = dict(enumerate(contigs))
    bed = bed.with_columns(pl.col("chrom").replace(cmap, return_dtype=pl.Utf8))
    return bed


@nb.njit(nogil=True, cache=True)
def splits_sum_le_value(arr: NDArray[np.number], max_value: float) -> NDArray[np.intp]:
    """Split an array into contiguous sections where the sum is less than or equal to a value.

    Parameters
    ----------
    arr : NDArray[np.number]
        Array to split.
    max_value : float
        Maximum value.

    Returns
    -------
    NDArray[np.int32]
        Split indices.
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
