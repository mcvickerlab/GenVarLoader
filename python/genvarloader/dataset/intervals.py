import numba as nb
import numpy as np
import taichi as ti
from numpy.typing import NDArray

from ..types import INTERVAL_DTYPE


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_tracks(
    offset_idxs: NDArray[np.intp],
    regions: NDArray[np.int32],
    intervals: NDArray[np.void],
    offsets: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Convert intervals to tracks at base-pair resolution.

    Parameters
    ----------
    interval_idxs : NDArray[np.intp]
        Shape = (n_queries) Indexes into offsets.
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    intervals : NDArray[np.void]
        Shape = (n_intervals) Sorted intervals, each is (start, end, value).
    offsets : NDArray[np.uint32]
        Shape = (n_interval_sets + 1) Offsets into intervals and values.
        For a GVL Dataset, n_interval_sets = n_samples * n_regions with that layout.

    Returns
    -------
    out : NDArray[np.float32]
        Shape = (n_queries) Ragged array of tracks.
    """
    n_queries = regions.shape[0]
    length_per_query = regions[:, 2] - regions[:, 1]
    out_idxs = np.empty(n_queries, np.int32)
    out_idxs[0] = 0
    out_idxs[1:] = length_per_query[:-1].cumsum()
    out = np.zeros(length_per_query.sum(), np.float32)
    for query in nb.prange(n_queries):
        idx = offset_idxs[query]
        o_s, o_e = offsets[idx], offsets[idx + 1]
        n_intervals = o_e - o_s
        q_s = regions[query, 1]
        query_length = length_per_query[query]
        out_idx = out_idxs[query]
        _out = out[out_idx : out_idx + query_length]
        if n_intervals == 0:
            _out[:] = 0
            continue

        for interval in nb.prange(o_s, o_e):
            itv = intervals[interval]
            start, end = itv.start - q_s, itv.end - q_s
            if start >= query_length:
                break
            _out[start:end] = itv.value
    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def tracks_to_intervals(
    regions: NDArray[np.int32],
    tracks: NDArray[np.float32],
):
    """Convert tracks to intervals. Note that this will include 0-value intervals.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    tracks : NDArray[np.float32]
        Shape = (n_queries*query_length) Ragged array of tracks.
    offsets : NDArray[np.uint32]
        Shape = (n_queries + 1) Offsets into intervals and values.
    query_length : int
        Length of each query.

    Returns
    -------
    out : NDArray[np.void]
        Shape = (n_intervals) Intervals.

    Notes
    -----
    Implementation closely follows [CUDA RLE](https://erkaman.github.io/posts/cuda_rle.html).
    """
    n_queries = len(regions)
    length_per_query = regions[:, 2] - regions[:, 1]

    track_offsets = np.empty(n_queries + 1, np.int32)
    track_offsets[0] = 0
    track_offsets[1:] = length_per_query.cumsum()

    n_intervals = np.empty(n_queries, np.int32)
    scanned_masks = np.empty_like(tracks, np.int32)
    for query in nb.prange(n_queries):
        track = tracks[track_offsets[query] : track_offsets[query + 1]]
        scanned_backward_mask = scanned_masks[
            track_offsets[query] : track_offsets[query + 1]
        ]
        _scanned_mask(track, scanned_backward_mask)
        n_intervals[query] = scanned_backward_mask[-1]

    interval_offsets = np.empty(n_queries + 1, np.int32)
    interval_offsets[0] = 0
    interval_offsets[1:] = n_intervals.cumsum()

    all_intervals = np.empty(interval_offsets[-1], INTERVAL_DTYPE)
    for query in nb.prange(n_queries):
        start = regions[query, 1]
        scanned_backward_mask = scanned_masks[
            track_offsets[query] : track_offsets[query + 1]
        ]
        compacted_backward_mask = _compact_mask(scanned_backward_mask)
        track = tracks[track_offsets[query] : track_offsets[query + 1]]
        values = track[compacted_backward_mask[:-1]]
        s = interval_offsets[query]
        #! parallel=True does not implement assignment to non-scalar views of structured arrays
        #! so, we must explicitly iterate over the slice
        for i in nb.prange(len(values)):
            itv = all_intervals[i + s]
            itv["start"] = compacted_backward_mask[i] + start
            itv["end"] = compacted_backward_mask[i + 1] + start
            itv["value"] = values[i]

    return all_intervals, interval_offsets


@nb.njit(parallel=True, nogil=True, cache=True)
def _scanned_mask(track: NDArray[np.float32], out: NDArray[np.int32]):
    backward_mask = np.empty(len(track), np.bool_)
    backward_mask[0] = True
    backward_mask[1:] = track[:-1] != track[1:]
    out[:] = backward_mask.cumsum()


@nb.njit(parallel=True, nogil=True, cache=True)
def _compact_mask(
    scanned_backward_mask: NDArray[np.int32],
):
    n_elems = len(scanned_backward_mask)
    n_runs = scanned_backward_mask[-1]
    compacted_backward_mask = np.empty(n_runs + 1, np.int32)
    compacted_backward_mask[-1] = n_elems
    for i in nb.prange(n_elems):
        if i == 0:
            compacted_backward_mask[i] = 0
        # 0 < i < n_elems - 1
        elif scanned_backward_mask[i] != scanned_backward_mask[i - 1]:
            compacted_backward_mask[scanned_backward_mask[i] - 1] = i
    return compacted_backward_mask


@ti.dataclass
class _Interval:
    start: ti.i32
    end: ti.i32
    value: ti.f32


@ti.kernel
def ti_intervals_to_tracks(
    interval_idxs: ti.types.ndarray(ti.i32, shape=1),  # type: ignore
    interval_idx_to_offset: ti.types.ndarray(ti.i32, ndim=1),  # type: ignore
    offset_idxs: ti.types.ndarray(ti.i32, ndim=1),  # type: ignore
    regions: ti.types.ndarray(ti.i32, ndim=2),  # type: ignore
    intervals: ti.template(),  # type: ignore
    out: ti.types.ndarray(ti.f32, ndim=1),  # type: ignore
    out_offsets: ti.types.ndarray(ti.i32, ndim=1),  # type: ignore
):
    """Decompress intervals to tracks at base-pair resolution.

    Parameters
    ----------
    interval_to_offset : NDArray[int32]
        Mapping from
    """
    for i in interval_idxs:
        interval = interval_idxs[i]
        query = interval_idx_to_offset[i]
        offset_idxs[query]

        q_s = regions[query, 1]
        query_length = regions[query, 2] - regions[query, 1]
        out_idx = out_offsets[query]
        _out = out[out_idx : out_idx + query_length]

        itv = intervals[interval]
        start, end = itv.start - q_s, itv.end - q_s
        if start >= query_length:
            continue
        for i in range(start, end):
            _out[i] = itv.value
