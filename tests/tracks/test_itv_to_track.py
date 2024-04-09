import numpy as np
from attrs import define
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases

from genvarloader.dataset.intervals import intervals_to_tracks, tracks_to_intervals
from genvarloader.types import INTERVAL_DTYPE, RaggedIntervals


@define
class Case:
    interval_idx: NDArray[np.intp]
    region_idx: NDArray[np.intp]
    regions: NDArray[np.int32]
    intervals: RaggedIntervals
    track: NDArray[np.float32]


def case_simple():
    """Return intervals and values. After decompressing, should yield:
    [1, 1, 1, 0, 2, 0, 0, 3, 3, 3]
    """
    coordinates = np.array(
        [
            [0, 3],
            [3, 4],
            [4, 5],
            [5, 7],
            [7, 10],
        ],
        dtype=np.int32,
    )
    values = np.array([1.0, 0, 2.0, 0, 3.0], dtype=np.float32)
    intervals = np.empty(5, dtype=INTERVAL_DTYPE)
    intervals["start"] = coordinates[:, 0]
    intervals["end"] = coordinates[:, 1]
    intervals["value"] = values
    intervals = RaggedIntervals.from_lengths(
        intervals, np.array([len(values)], np.int32)
    )
    interval_idx = np.array([0], dtype=np.intp)
    region_idx = np.array([0], dtype=np.intp)
    regions = np.array([[0, 0, 10]], dtype=np.int32)  # coordinates 0:0-10
    track = np.array([1, 1, 1, 0, 2, 0, 0, 3, 3, 3], dtype=np.float32)
    return Case(interval_idx, region_idx, regions, intervals, track)


@parametrize_with_cases("case", cases=".")
def test_intervals_to_tracks(case: Case):
    """Test intervals_to_values."""
    intervals = case.intervals
    out = intervals_to_tracks(
        case.interval_idx, case.regions, intervals.data, intervals.offsets
    )
    np.testing.assert_array_equal(out, case.track)


@parametrize_with_cases("case", cases=".")
def test_tracks_to_intervals(case: Case):
    """Test tracks_to_intervals."""
    intervals, offsets = tracks_to_intervals(case.region_idx, case.regions, case.track)
    intervals = RaggedIntervals.from_offsets(intervals, 1, offsets)
    np.testing.assert_array_equal(intervals.data, case.intervals.data)
    np.testing.assert_array_equal(intervals.offsets, case.intervals.offsets)
