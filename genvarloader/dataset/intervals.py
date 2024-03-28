from typing import Sequence, Union

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

    def __getitem__(self, ds_idx: ListIdx) -> "Intervals":
        intervals = []
        values = []
        offsets = np.empty(len(ds_idx) + 1, dtype=np.uint32)
        offsets[0] = 0
        for output_idx, i in enumerate(ds_idx, 1):
            s, e = self.offsets[i], self.offsets[i + 1]
            offsets[output_idx] = e - s
            if e > s:
                intervals.append(self.intervals[s:e])
                values.append(self.values[s:e])

        if len(intervals) == 0:
            intervals = np.empty((0, 2), dtype=self.intervals.dtype)
            values = np.empty(0, dtype=self.values.dtype)
        else:
            intervals = np.concatenate(intervals)
            values = np.concatenate(values)

        offsets = offsets.cumsum(dtype=np.uint32)

        return Intervals(intervals, values, offsets)


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_values(
    regions: NDArray[np.int32],
    intervals: NDArray[np.uint32],
    values: NDArray[np.float32],
    offsets: NDArray[np.uint32],
    query_length: int,
):
    """Convert intervals to values at base-pair resolution.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    intervals : NDArray[np.uint32]
        Shape = (n_intervals, 2) Intervals.
    values : NDArray[np.float32]
        Shape = (n_intervals,) Values.
    offsets : NDArray[np.uint32]
        Shape = (n_queries + 1,) Offsets into intervals and values.
    query_length : int
        Length of each query.
    """
    n_regions = len(regions)
    out = np.zeros((n_regions, query_length), np.float32)
    for region in nb.prange(n_regions):
        q_s = regions[region, 1]
        o_s, o_e = offsets[region], offsets[region + 1]
        n_intervals = o_e - o_s
        if n_intervals == 0:
            out[region] = 0
            continue

        for interval in nb.prange(o_s, o_e):
            i_s, i_e = intervals[interval] - q_s
            out[region, i_s:i_e] = values[interval]
    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_hap_values(
    regions: NDArray[np.int32],
    shifts: NDArray[np.uint32],
    intervals: NDArray[np.uint32],
    values: NDArray[np.float32],
    offsets: NDArray[np.uint32],
    query_length: int,
):
    """Convert intervals to values at base-pair resolution.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    shifts : NDArray[np.uint32]
        Shape = (n_queries, ploidy) Shifts for each query.
    intervals : NDArray[np.uint32]
        Shape = (n_intervals, 2) Intervals.
    values : NDArray[np.float32]
        Shape = (n_intervals,) Values.
    offsets : NDArray[np.uint32]
        Shape = (n_queries + 1,) Offsets into intervals and values.
    query_length : int
        Length of each query.
    """
    n_queries = len(regions)
    ploidy = shifts.shape[1]
    out = np.zeros((n_queries, ploidy, query_length), np.float32)
    for query in nb.prange(n_queries):
        q_s = regions[query, 1]
        o_s, o_e = offsets[query], offsets[query + 1]
        n_intervals = o_e - o_s

        if n_intervals == 0:
            out[query] = 0
            continue

        for hap in nb.prange(ploidy):
            shift = shifts[query, hap]
            for interval in nb.prange(o_s, o_e):
                i_s, i_e = intervals[interval] - q_s + shift
                out[query, hap, i_s:i_e] = values[interval]
    return out
