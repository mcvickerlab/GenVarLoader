import numpy as np

from genvarloader.pgen import weighted_activity_selection


def test_max_dels():
    starts = np.array([1, 1, 1, 6, 3, 9])
    ends = np.array([3, 5, 6, 9, 9, 10])
    sorter = np.argsort(ends)
    starts = starts[sorter]
    ends = ends[sorter]

    actual_max_dels = weighted_activity_selection(starts, ends)
    desired_max_dels = np.array([2, 4, 5, 8, 8, 9])[sorter]

    np.testing.assert_equal(actual_max_dels, desired_max_dels)
