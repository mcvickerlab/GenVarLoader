import numba as nb
import numpy as np
from numpy.typing import NDArray

from .._types import INTERVAL_DTYPE

__all__ = []


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_tracks(
    offset_idxs: NDArray[np.integer],
    regions: NDArray[np.int32],
    intervals: NDArray[np.void],
    offsets: NDArray[np.int64],
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
    data : NDArray[np.float32]
        Shape = (n_queries) Values for ragged array of tracks.
    offsets : NDArray[np.int32]
        Shape = (n_queries + 1) Offsets for ragged array of tracks.
    """
    n_queries = regions.shape[0]
    length_per_query = regions[:, 2] - regions[:, 1]
    out_idx = np.empty(n_queries + 1, np.int64)
    out_idx[0] = 0
    out_idx[1:] = length_per_query.cumsum()
    out = np.zeros(out_idx[-1], np.float32)
    out_idx = out_idx[:-1]
    for query in nb.prange(n_queries):
        idx = offset_idxs[query]
        o_s, o_e = offsets[idx], offsets[idx + 1]
        n_intervals = o_e - o_s
        q_s = regions[query, 1]
        query_length = length_per_query[query]
        _out_idx = out_idx[query]
        _out = out[_out_idx : _out_idx + query_length]
        if n_intervals == 0:
            continue

        #! a data race will occur if there are any overlapping intervals
        for interval in nb.prange(o_s, o_e):
            itv = intervals[interval]
            start, end = itv.start - q_s, itv.end - q_s
            if start < query_length:
                _out[start:end] = itv.value
    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def tracks_to_intervals(
    regions: NDArray[np.int32],
    tracks: NDArray[np.float32],
    track_offsets: NDArray[np.int64],
):
    """Convert tracks to intervals. Note that this will include 0-value intervals.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    tracks : NDArray[np.float32]
        Shape = (n_queries*query_length) Ragged array of tracks.
    offsets : NDArray[np.int64]
        Shape = (n_queries + 1) Offsets into ragged track data.

    Returns
    -------
    out : NDArray[np.void]
        Shape = (n_intervals) Intervals.

    Notes
    -----
    Implementation closely follows [CUDA RLE](https://erkaman.github.io/posts/cuda_rle.html).
    """
    n_queries = len(regions)

    n_intervals = np.empty(n_queries, np.int32)
    scanned_masks = np.empty_like(tracks, np.int64)
    for query in nb.prange(n_queries):
        o_s = track_offsets[query]
        o_e = track_offsets[query + 1]
        if o_s == o_e:
            n_intervals[query] = 0
            continue
        track = tracks[o_s:o_e]
        scanned_backward_mask = scanned_masks[o_s:o_e]
        _scanned_mask(track, scanned_backward_mask)
        n_intervals[query] = scanned_backward_mask[-1]

    interval_offsets = np.empty(n_queries + 1, np.int64)
    interval_offsets[0] = 0
    interval_offsets[1:] = n_intervals.cumsum()

    all_intervals = np.empty(interval_offsets[-1], INTERVAL_DTYPE)
    for query in nb.prange(n_queries):
        o_s = track_offsets[query]
        o_e = track_offsets[query + 1]
        if o_s == o_e:
            continue
        scanned_backward_mask = scanned_masks[o_s:o_e]
        compacted_backward_mask = _compact_mask(scanned_backward_mask)
        track = tracks[o_s:o_e]
        values = track[compacted_backward_mask[:-1]]
        s = interval_offsets[query]
        start = regions[query, 1]
        #! parallel=True does not implement assignment to non-scalar views of structured arrays
        #! so, we must explicitly iterate over the slice
        for i in nb.prange(len(values)):
            itv = all_intervals[i + s]
            itv["start"] = compacted_backward_mask[i] + start
            itv["end"] = compacted_backward_mask[i + 1] + start
            itv["value"] = values[i]

    return all_intervals, interval_offsets


@nb.njit(parallel=True, nogil=True, cache=True)
def _scanned_mask(track: NDArray[np.float32], out: NDArray[np.int64]):
    backward_mask = np.empty(len(track), np.bool_)
    backward_mask[0] = True
    backward_mask[1:] = track[:-1] != track[1:]
    out[:] = backward_mask.cumsum()


@nb.njit(parallel=True, nogil=True, cache=True)
def _compact_mask(
    scanned_backward_mask: NDArray[np.int64],
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
