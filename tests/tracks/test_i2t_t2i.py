import numpy as np
from attrs import define
from einops import repeat
from genvarloader._dataset._intervals import intervals_to_tracks, tracks_to_intervals
from genvarloader._ragged import RaggedIntervals
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases
from seqpro.rag import Ragged


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
    lengths = np.array([len(values)], np.uint32)
    starts = Ragged.from_lengths(coordinates[:, 0], lengths)
    ends = Ragged.from_lengths(coordinates[:, 1], lengths)
    values = Ragged.from_lengths(values, lengths)
    intervals = RaggedIntervals(starts, ends, values)

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
    lengths = np.array([len(values)], np.uint32)
    coordinates = repeat(coordinates, "n d -> (r n) d", r=2)
    values = repeat(values, "n -> (r n)", r=2)
    lengths = repeat(lengths, "n -> (r n)", r=2)

    starts = Ragged.from_lengths(coordinates[:, 0], lengths)
    ends = Ragged.from_lengths(coordinates[:, 1], lengths)
    values = Ragged.from_lengths(values, lengths)
    intervals = RaggedIntervals(starts, ends, values)

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
        itv_starts=intervals.starts.data,
        itv_ends=intervals.ends.data,
        itv_values=intervals.values.data,
        itv_offsets=intervals.values.offsets,
        out=out,
        out_offsets=out_offsets,
    )
    np.testing.assert_array_equal(out, data.track)


@parametrize_with_cases("data", cases=".")
def test_tracks_to_intervals(data: Data):
    batch_size = len(data.regions)
    shape = (batch_size, None)
    starts, ends, values, offsets = tracks_to_intervals(
        data.regions, data.track, data.t_offsets
    )
    starts = Ragged.from_offsets(starts, shape, offsets)
    ends = Ragged.from_offsets(ends, shape, offsets)
    values = Ragged.from_offsets(values, shape, offsets)
    intervals = RaggedIntervals(starts, ends, values)
    
    np.testing.assert_array_equal(intervals.starts.data, data.intervals.starts.data)
    np.testing.assert_array_equal(intervals.ends.data, data.intervals.ends.data)
    np.testing.assert_array_equal(intervals.values.data, data.intervals.values.data)
    np.testing.assert_array_equal(
        intervals.values.offsets, data.intervals.values.offsets
    )
