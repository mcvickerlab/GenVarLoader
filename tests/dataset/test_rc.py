import numpy as np
from genvarloader import Ragged
from genvarloader._ragged import _reverse_helper
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases


def track_one():
    data = np.arange(5, dtype=np.float32)
    offsets = np.array([0, len(data)], dtype=np.int64)
    desired = data[::-1].copy()
    track = Ragged.from_offsets(data, 1, offsets)
    return track, desired


def track_two():
    data = np.array([4, 5, 1, 2, 3], dtype=np.float32)
    offsets = np.array([0, 2, len(data)], dtype=np.int64)
    desired = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    track = Ragged.from_offsets(data, 2, offsets)
    return track, desired


@parametrize_with_cases("track, desired", cases=".", prefix="track_")
def test_reverse(track: Ragged, desired: NDArray):
    _reverse_helper(track.data, track.offsets, mask=np.full(track.lengths.size, True))
    np.testing.assert_equal(track.data, desired)
