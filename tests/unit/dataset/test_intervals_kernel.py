import numpy as np

from genvarloader._dataset._intervals import intervals_to_tracks


def test_intervals_to_tracks_empty():
    """No intervals -> output is zeros."""
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array([0], dtype=np.int32)
    itv_starts = np.array([], dtype=np.int32)
    itv_ends = np.array([], dtype=np.int32)
    itv_values = np.array([], dtype=np.float32)
    itv_offsets = np.array([0, 0], dtype=np.int64)
    out = np.empty(5, dtype=np.float32)
    out_offsets = np.array([0, 5], dtype=np.int64)

    intervals_to_tracks(
        offset_idxs, starts, itv_starts, itv_ends, itv_values,
        itv_offsets, out, out_offsets,
    )
    np.testing.assert_equal(out, np.zeros(5, dtype=np.float32))


def test_intervals_to_tracks_single_interval():
    """One interval covers part of the output."""
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array([0], dtype=np.int32)
    itv_starts = np.array([1], dtype=np.int32)
    itv_ends = np.array([4], dtype=np.int32)
    itv_values = np.array([2.5], dtype=np.float32)
    itv_offsets = np.array([0, 1], dtype=np.int64)
    out = np.empty(5, dtype=np.float32)
    out_offsets = np.array([0, 5], dtype=np.int64)

    intervals_to_tracks(
        offset_idxs, starts, itv_starts, itv_ends, itv_values,
        itv_offsets, out, out_offsets,
    )
    np.testing.assert_equal(out, np.array([0.0, 2.5, 2.5, 2.5, 0.0], dtype=np.float32))


def test_intervals_to_tracks_multiple_non_overlapping():
    """Two non-overlapping intervals, gap between them filled with zeros."""
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array([0], dtype=np.int32)
    itv_starts = np.array([0, 3], dtype=np.int32)
    itv_ends = np.array([2, 5], dtype=np.int32)
    itv_values = np.array([1.0, 3.0], dtype=np.float32)
    itv_offsets = np.array([0, 2], dtype=np.int64)
    out = np.empty(5, dtype=np.float32)
    out_offsets = np.array([0, 5], dtype=np.int64)

    intervals_to_tracks(
        offset_idxs, starts, itv_starts, itv_ends, itv_values,
        itv_offsets, out, out_offsets,
    )
    np.testing.assert_equal(out, np.array([1.0, 1.0, 0.0, 3.0, 3.0], dtype=np.float32))


def test_intervals_to_tracks_offset_query_start():
    """Query starts at non-zero — intervals are in absolute coords; output is relative."""
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array([10], dtype=np.int32)
    itv_starts = np.array([11], dtype=np.int32)
    itv_ends = np.array([13], dtype=np.int32)
    itv_values = np.array([7.0], dtype=np.float32)
    itv_offsets = np.array([0, 1], dtype=np.int64)
    out = np.empty(4, dtype=np.float32)
    out_offsets = np.array([0, 4], dtype=np.int64)

    intervals_to_tracks(
        offset_idxs, starts, itv_starts, itv_ends, itv_values,
        itv_offsets, out, out_offsets,
    )
    np.testing.assert_equal(out, np.array([0.0, 7.0, 7.0, 0.0], dtype=np.float32))
