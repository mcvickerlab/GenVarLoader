from typing import Optional, Union

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.guvectorize("(n),(),()->(n)", target="parallel", cache=True)
def gufunc_multi_slice(
    arr: NDArray,
    start: Union[int, NDArray[np.integer]],
    length: int,
    res: Optional[NDArray] = None,
) -> NDArray:  # type: ignore
    res[:length] = arr[start : start + length]  # type: ignore


@nb.njit(nogil=True, cache=True)
def partition_regions(
    start: NDArray[np.int64], end: NDArray[np.int64], max_length: int
):
    partitions = np.zeros_like(start)
    partition = 0
    curr_length = end[0] - start[0]
    for i in range(1, len(partitions)):
        curr_length += end[i] - end[i - 1]
        if curr_length > max_length:
            partition += 1
            curr_length = end[i] - start[i]
        partitions[i] = partition
    return partitions
