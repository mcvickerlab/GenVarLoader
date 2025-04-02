import numpy as np
from attrs import define
from einops import repeat
from genvarloader._dataset._intervals import intervals_to_tracks, tracks_to_intervals
from genvarloader._ragged import INTERVAL_DTYPE, RaggedIntervals
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases


@define
class Data:
    interval_idx: NDArray[np.intp]
    regions: NDArray[np.int32]
    intervals: RaggedIntervals
    track: NDArray[np.float32]
    t_offsets: NDArray[np.int64]


def case_simple():
    """Return intervals and values. After decompressing, should yield:
    [1, 1, 1, 0, 2, 0, 0, 0, 0, 3]
    """
    coordinates = np.array(
        [
            [0, 3],
            [3, 4],
            [4, 5],
            [5, 9],
            [9, 10],
        ],
        dtype=np.int32,
    )
    values = np.array([1, 0, 2, 0, 3], dtype=np.float32)
    n_intervals = len(values)
    intervals = np.empty(n_intervals, dtype=INTERVAL_DTYPE)
    intervals["start"] = coordinates[:, 0]
    intervals["end"] = coordinates[:, 1]
    intervals["value"] = values
    intervals = RaggedIntervals.from_lengths(
        intervals, np.array([n_intervals], np.int32)
    )

    interval_idx = np.array([0], dtype=np.intp)

    regions = np.array([[0, 0, 10]], dtype=np.int32)  # coordinates 0:0-10

    track = np.array([1, 1, 1, 0, 2, 0, 0, 0, 0, 3], dtype=np.float32)
    t_offsets = np.array([0, len(track)], np.int64)
    return Data(interval_idx, regions, intervals, track, t_offsets)


def case_two_regions():
    """Return intervals and values. After decompressing, should yield:
    [1, 1, 1, 0, 2, 0, 0, 0, 0, 3]
    """
    coordinates = np.array(
        [
            [0, 3],
            [3, 4],
            [4, 5],
            [5, 9],
            [9, 10],
        ],
        dtype=np.int32,
    )
    values = np.array([1, 0, 2, 0, 3], dtype=np.float32)
    n_intervals = len(values)

    intervals = np.empty(n_intervals, dtype=INTERVAL_DTYPE)
    intervals["start"] = coordinates[:, 0]
    intervals["end"] = coordinates[:, 1]
    intervals["value"] = values
    intervals = repeat(intervals, "n -> (r n)", r=2)

    intervals = RaggedIntervals.from_lengths(
        intervals, np.array([n_intervals, n_intervals], np.int32)
    )

    interval_idx = np.array([0, 1], dtype=np.intp)

    regions = np.array([[0, 0, 10]], dtype=np.int32)  # coordinates 0:0-10
    regions = repeat(regions, "n d -> (r n) d", r=2)

    track = np.array([1, 1, 1, 0, 2, 0, 0, 0, 0, 3], dtype=np.float32)
    tracks = repeat(track, "n -> (r n)", r=2)
    t_offsets = np.array([0, len(track), tracks.size], np.int64)
    return Data(interval_idx, regions, intervals, tracks, t_offsets)


@parametrize_with_cases("data", cases=".")
def test_intervals_to_tracks(data: Data):
    intervals = data.intervals
    out = np.empty_like(data.track)
    out_offsets = data.t_offsets.copy()
    intervals_to_tracks(
        offset_idxs=data.interval_idx,
        starts=data.regions[:, 1],
        intervals=intervals.data,
        itv_offsets=intervals.offsets,
        out=out,
        out_offsets=out_offsets,
    )
    np.testing.assert_array_equal(out, data.track)


@parametrize_with_cases("data", cases=".")
def test_tracks_to_intervals(data: Data):
    intervals, offsets = tracks_to_intervals(data.regions, data.track, data.t_offsets)
    intervals = RaggedIntervals.from_offsets(intervals, 1, offsets)
    np.testing.assert_array_equal(intervals.data, data.intervals.data)
    np.testing.assert_array_equal(intervals.offsets, data.intervals.offsets)
