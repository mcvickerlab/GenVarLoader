import numpy as np
from numpy.typing import NDArray

from ..genvarloader import intervals_to_tracks as _intervals_to_tracks_rust
from ..genvarloader import tracks_to_intervals as _tracks_to_intervals_rust
from .._threads import should_parallelize

__all__ = []


def intervals_to_tracks(
    offset_idxs: NDArray[np.integer],
    starts: NDArray[np.int32],
    itv_starts: NDArray[np.int32],
    itv_ends: NDArray[np.int32],
    itv_values: NDArray[np.float32],
    itv_offsets: NDArray[np.int64],
    out: NDArray[np.float32],
    out_offsets: NDArray[np.int64],
) -> None:
    """Paint base-pair-resolution tracks from intervals, writing ``out`` in place.

    Dispatches to the Rust backend. Read-only inputs are coerced to canonical dtypes so
    the backend receives byte-identical bytes; ``out`` is passed through untouched so
    in-place writes land in the caller's buffer.
    """
    offset_idxs = np.ascontiguousarray(offset_idxs, dtype=np.int64)
    starts = np.ascontiguousarray(starts, dtype=np.int32)
    itv_starts = np.ascontiguousarray(itv_starts, dtype=np.int32)
    itv_ends = np.ascontiguousarray(itv_ends, dtype=np.int32)
    itv_values = np.ascontiguousarray(itv_values, dtype=np.float32)
    itv_offsets = np.ascontiguousarray(itv_offsets, dtype=np.int64)
    out_offsets = np.ascontiguousarray(out_offsets, dtype=np.int64)
    _intervals_to_tracks_rust(
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )


def tracks_to_intervals(
    regions: NDArray[np.int32],
    tracks: NDArray[np.float32],
    track_offsets: NDArray[np.int64],
) -> tuple[
    NDArray[np.int32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int64]
]:
    """RLE-encode a ragged f32 track buffer into (starts, ends, values, offsets) intervals.

    Includes 0-value intervals (no filtering on value == 0.0). Dispatches to the Rust backend. Read-only inputs
    are coerced to canonical dtypes so both backends receive byte-identical bytes.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query (contig_idx, start, end).
    tracks : NDArray[np.float32]
        Shape = (total_track_len,) Ragged flat array of track values.
    track_offsets : NDArray[np.int64]
        Shape = (n_queries + 1,) Offsets into ragged track data.

    Returns
    -------
    all_starts : NDArray[np.int32]
    all_ends : NDArray[np.int32]
    all_values : NDArray[np.float32]
    interval_offsets : NDArray[np.int64]
    """
    regions = np.ascontiguousarray(regions, dtype=np.int32)
    tracks = np.ascontiguousarray(tracks, dtype=np.float32)
    track_offsets = np.ascontiguousarray(track_offsets, dtype=np.int64)
    total_bytes = int(track_offsets[-1]) * 4  # f32 = 4 bytes per element
    return _tracks_to_intervals_rust(
        regions, tracks, track_offsets, should_parallelize(total_bytes)
    )
