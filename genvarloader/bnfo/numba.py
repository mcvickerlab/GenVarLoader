import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(nogil=True, parallel=True, cache=True)
def multi_slice(
    arr: NDArray,
    starts: NDArray[np.integer],
    length: int,
) -> NDArray:  # type: ignore
    out = np.empty(shape=(len(starts), *arr.shape[:-1], length), dtype=arr.dtype)
    for i in nb.prange(len(starts)):
        out[i] = arr[..., starts[i] : starts[i] + length]
    return out


@nb.njit(nogil=True, cache=True)
def partition_regions(
    starts: NDArray[np.int64], ends: NDArray[np.int64], max_length: int
):
    partitions = np.zeros_like(starts)
    partition = 0
    curr_length = ends[0] - starts[0]
    for i in range(1, len(partitions)):
        curr_length += ends[i] - ends[i - 1]
        if curr_length > max_length:
            partition += 1
            curr_length = ends[i] - starts[i]
        partitions[i] = partition
    return partitions
