import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..types import INTERVAL_DTYPE


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_tracks(
    interval_idxs: NDArray[np.intp],
    regions: NDArray[np.int32],
    intervals: NDArray[np.void],
    offsets: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Convert intervals to tracks at base-pair resolution.

    Parameters
    ----------
    interval_idxs : NDArray[np.intp]
        Shape = (n_queries) Indexes into intervals and values.
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    intervals : NDArray[np.int32]
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
        Shape = (n_queries*query_length) Ragged array of tracks.
    """
    n_queries = regions.shape[0]
    length_per_query = regions[:, 2] - regions[:, 1]
    out_idxs = np.empty(n_queries, np.int32)
    out_idxs[0] = 0
    out_idxs[1:] = length_per_query[:-1].cumsum()
    out = np.zeros(length_per_query.sum(), np.float32)
    for query in nb.prange(n_queries):
        interval_idx = interval_idxs[query]
        o_s, o_e = offsets[interval_idx], offsets[interval_idx + 1]
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
            if start > query_length:
                break
            _out[start:end] = itv.value
    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def tracks_to_intervals(
    region_idx: NDArray[np.intp],
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

    interval_offsets = np.empty(n_queries + 1, np.int32)
    interval_offsets[0] = 0
    n_intervals = 0
    scanned_masks = np.empty_like(tracks, np.int32)
    for query in nb.prange(n_queries):
        track = tracks[track_offsets[query] : track_offsets[query + 1]]
        scanned_backward_mask = scanned_masks[
            track_offsets[query] : track_offsets[query + 1]
        ]
        _scanned_mask(track, scanned_backward_mask)
        n_intervals += scanned_backward_mask[-1]
        interval_offsets[query + 1] = n_intervals

    all_intervals = np.empty(n_intervals, INTERVAL_DTYPE)
    for query in nb.prange(n_queries):
        start = regions[region_idx[query], 1]
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
    compacted_backward_mask = np.empty(scanned_backward_mask[-1] + 1, np.int32)
    compacted_backward_mask[0] = 0
    compacted_backward_mask[-1] = len(scanned_backward_mask)
    for i in nb.prange(len(scanned_backward_mask) - 1):
        if scanned_backward_mask[i] != scanned_backward_mask[i - 1]:
            compacted_backward_mask[scanned_backward_mask[i] - 1] = i
    return compacted_backward_mask
