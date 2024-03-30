import numpy as np

from genvarloader.utils import splice_subarrays


def test_splice_subarrays():
    arr = np.arange(10)
    starts = np.array([0, 5, 9])
    ends = np.array([3, 8, 10])

    actual = splice_subarrays(arr, starts, ends)
    desired = np.array([0, 1, 2, 5, 6, 7, 9])
    np.testing.assert_equal(actual, desired)
