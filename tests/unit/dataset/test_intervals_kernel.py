import numpy as np
import pytest
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
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
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
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
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
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
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
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )
    np.testing.assert_equal(out, np.array([0.0, 7.0, 7.0, 0.0], dtype=np.float32))


def _run(backend, monkeypatch, starts, itv_starts, itv_ends, itv_values, out_len):
    """Single query, one interval-slice; force `backend` and return the out buffer."""
    monkeypatch.setenv("GVL_BACKEND", backend)
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array(starts, dtype=np.int32)
    itv_starts = np.array(itv_starts, dtype=np.int32)
    itv_ends = np.array(itv_ends, dtype=np.int32)
    itv_values = np.array(itv_values, dtype=np.float32)
    itv_offsets = np.array([0, len(itv_starts)], dtype=np.int64)
    out = np.empty(out_len, dtype=np.float32)
    out_offsets = np.array([0, out_len], dtype=np.int64)
    intervals_to_tracks(
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )
    return out


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_interval_starts_before_query_full_cover(backend, monkeypatch):
    # issue #242: interval [96,114) value 5, query_start=100, length=10 -> all 5s
    out = _run(backend, monkeypatch, [100], [96], [114], [5.0], 10)
    np.testing.assert_array_equal(out, np.full(10, 5.0, dtype=np.float32))


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_interval_starts_before_query_partial(backend, monkeypatch):
    # interval [8,13) value 5, query_start=10, length=5 -> [5,5,5,0,0]
    out = _run(backend, monkeypatch, [10], [8], [13], [5.0], 5)
    np.testing.assert_array_equal(
        out, np.array([5.0, 5.0, 5.0, 0.0, 0.0], dtype=np.float32)
    )


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_interval_left_overlap_end_in_window(backend, monkeypatch):
    # interval [4,8) value 5, query_start=10, length=5 -> all zeros (no overlap)
    out = _run(backend, monkeypatch, [10], [4], [8], [5.0], 5)
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_interval_fully_left_of_query(backend, monkeypatch):
    # interval [2,6) ends at/below query_start=10 -> all zeros
    out = _run(backend, monkeypatch, [10], [2], [6], [5.0], 5)
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))
