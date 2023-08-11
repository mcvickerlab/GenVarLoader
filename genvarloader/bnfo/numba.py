from typing import Optional, Union

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.guvectorize("(n),(),(),(l)->(l)", target="parallel", cache=True)
def gufunc_multi_slice(
    arr: NDArray,
    start: Union[int, NDArray[np.integer]],
    placeholder: NDArray,
    res: Optional[NDArray] = None,
) -> NDArray:  # type: ignore
    length = len(placeholder)
    res[:] = arr[start : start + length]  # type: ignore


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
