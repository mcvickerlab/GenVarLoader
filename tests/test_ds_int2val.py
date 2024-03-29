from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pytest_cases import fixture

from genvarloader.dataset.intervals import intervals_to_values


@fixture
def intervals():
    """Return intervals and values. After decompressing, should yield:
    [1, 1, 1, 0, 2, 0, 0, 3, 3, 3]
    """
    intervals = np.array([[0, 3], [4, 5], [7, 10]], dtype=np.uint32)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return intervals, values


def test_intervals_to_values(intervals: Tuple[NDArray[np.uint32], NDArray[np.float32]]):
    """Test intervals_to_values."""
    interval_idxs = np.array([0], dtype=np.intp)
    region_idxs = np.array([0], dtype=np.intp)
    regions = np.array([[0, 0, 10]], dtype=np.int32)
    offsets = np.array([0, 3], dtype=np.uint32)
    query_length = 10
    out = intervals_to_values(
        interval_idxs, region_idxs, regions, *intervals, offsets, query_length
    )
    desired = np.array([[1, 1, 1, 0, 2, 0, 0, 3, 3, 3]], dtype=np.float32)
    np.testing.assert_array_equal(out, desired)
