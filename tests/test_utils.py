import numpy as np
from genvarloader._dataset._utils import splits_sum_le_value


def test_splits_sum_le_value():
    max_size = 10
    sizes = np.array([3, 5, 2, 4, 7, 5, 2], np.int32)
    splits = splits_sum_le_value(sizes, max_size)
    np.testing.assert_equal(splits, np.array([0, 3, 4, 5, 7], np.intp))
    np.testing.assert_array_less(np.add.reduceat(sizes, splits[:-1]), max_size + 1)
