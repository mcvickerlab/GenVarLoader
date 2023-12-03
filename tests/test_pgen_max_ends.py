import numpy as np

from genvarloader.pgen import weighted_activity_selection


def test_biallelic_snps():
    raise NotImplementedError
    starts = np.array([0])
    positions = np.array([0, 1, 2, 3, 4, 5])
    start_idxs = np.array([0])
    target_lengths = np.array([10])
    size_diffs = np.array([0, 0, 0, 0, 0, 0])

    desired_ends = np.array([10])
    desired_idxs = np.array([6])

    actual_ends, actual_idxs = weighted_activity_selection(
        starts, start_idxs, positions, size_diffs, target_lengths
    )

    np.testing.assert_equal(actual_ends, desired_ends)
    np.testing.assert_equal(actual_idxs, desired_idxs)


def test_snps_and_insertions():
    raise NotImplementedError
    starts = np.array([0])
    positions = np.array([0, 0, 0, 1, 1, 1])
    start_idxs = np.array([0])
    target_lengths = np.array([10])
    size_diffs = np.array([0, 5, 7, 0, 0, 0])

    desired_ends = np.array([10])
    desired_idxs = np.array([6])

    actual_ends, actual_idxs = weighted_activity_selection(
        starts, start_idxs, positions, size_diffs, target_lengths
    )

    np.testing.assert_equal(actual_ends, desired_ends)
    np.testing.assert_equal(actual_idxs, desired_idxs)
