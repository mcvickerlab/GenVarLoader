import numba as nb
import numpy as np
from numpy.typing import NDArray

from .._ragged import INTERVAL_DTYPE

__all__ = []


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_tracks(
    offset_idxs: NDArray[np.integer],
    starts: NDArray[np.int32],
    out_offsets: NDArray[np.int64],
    intervals: NDArray[np.void],
    itv_offsets: NDArray[np.int64],
    out: NDArray[np.float32],
):
    """Convert intervals to tracks at base-pair resolution.
    Assumptions:
    - intervals are sorted by start
    - intervals do not overlap
    - no intervals start before query start

    Parameters
    ----------
    offset_idxs : NDArray[np.intp]
        Shape = (batch) Indexes into offsets.
    starts : NDArray[np.int32]
        Shape = (batch) Starts for each query.
    out_offsets : NDArray[np.int64]
        Shape = (batch + 1) Offsets into output tracks.
    intervals : NDArray[np.void]
        Ragged shape = (regions*samples*intervals) Sorted intervals with struct dtype: (start: i32, end: i32, value: f32).
    itv_offsets : NDArray[np.uint32]
        Shape = (regions*samples + 1) Offsets into intervals and values.
        For a GVL Dataset, n_interval_sets = n_samples * n_regions with that layout.

    Returns
    -------
    data : NDArray[np.float32]
        Ragged shape = (batch*length) Values for ragged array of tracks.
    offsets : NDArray[np.int32]
        Shape = (batch + 1) Offsets for ragged array of tracks.
    """
    n_queries = len(starts)
    out[:] = 0.0
    for query in nb.prange(n_queries):
        idx = offset_idxs[query]
        itv_s, itv_e = itv_offsets[idx], itv_offsets[idx + 1]
        n_intervals = itv_e - itv_s
        if n_intervals == 0:
            continue

        out_s, out_e = out_offsets[query], out_offsets[query + 1]
        length = out_e - out_s
        _out = out[out_s:out_e]

        query_start = starts[query]

        # if parallelized, a data race will occur if there are any overlapping intervals
        for interval in range(itv_s, itv_e):
            itv = intervals[interval]
            #! assumes itv.start >= query_start
            start, end = itv.start - query_start, itv.end - query_start
            if start < length:
                _out[start:end] = itv.value
            else:
                #! assumes intervals are sorted by start
                # cannot break if parallelized
                break


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
