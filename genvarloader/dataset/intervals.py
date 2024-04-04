from typing import Optional, Sequence, Union

import numba as nb
import numpy as np
from attrs import define
from numpy.typing import NDArray

Idx = Union[int, np.integer, Sequence[int], NDArray[np.integer], slice]
ListIdx = Union[Sequence[int], NDArray[np.integer]]


@define
class Intervals:
    intervals: NDArray[np.uint32]  # (n_intervals, 2)
    values: NDArray[np.float32]  # (n_intervals)
    offsets: NDArray[np.uint32]  # (n_queries + 1)

    def __len__(self) -> int:
        return len(self.offsets) - 1


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_tracks(
    interval_idxs: NDArray[np.intp],
    region_idxs: NDArray[np.intp],
    regions: NDArray[np.int32],
    intervals: NDArray[np.uint32],
    values: NDArray[np.float32],
    offsets: NDArray[np.uint32],
    query_length: int,
    out: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """Convert intervals to tracks at base-pair resolution.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    intervals : NDArray[np.uint32]
        Shape = (n_intervals, 2) Sorted intervals, each is (start, end).
    values : NDArray[np.float32]
        Shape = (n_intervals) Values.
    offsets : NDArray[np.uint32]
        Shape = (n_queries + 1) Offsets into intervals and values.
    query_length : int
        Length of each query.

    Returns
    -------
    out : NDArray[np.float32]
        Shape = (n_queries, query_length) Tracks.
    """
    n_queries = len(interval_idxs)
    if out is None:
        out = np.zeros((n_queries, query_length), np.float32)
    for query in nb.prange(n_queries):
        interval_idx = interval_idxs[query]
        o_s, o_e = offsets[interval_idx], offsets[interval_idx + 1]
        n_intervals = o_e - o_s
        if n_intervals == 0:
            out[query] = 0
            continue

        q_s = regions[region_idxs[query], 1]
        for interval in nb.prange(o_s, o_e):
            start, end = intervals[interval] - q_s
            if start > query_length:
                break
            out[query, start:end] = values[interval]
    return out
