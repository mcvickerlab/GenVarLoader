import numpy as np
from pytest_cases import parametrize_with_cases
from utils import nonoverlapping_intervals


def case_gaps_no_max_width():
    return 10000, 1, None, 100000


def case_gaps_with_max_width():
    return 10000, 1, 3, 100000 * 3


def case_no_gaps_at_min_width():
    return 10000, 1, None, 10000


def case_no_gaps_at_max_width():
    return 10000, 1, 3, 10000 * 3


@parametrize_with_cases("n, min_width, max_width, high", cases=".")
def test_nonoverlapping_intervals(n, min_width, max_width, high):
    low = 0
    for _ in range(1000):
        itvs = nonoverlapping_intervals(n, low, high, max_width=max_width)
        widths = np.diff(itvs, axis=1).squeeze()
        gaps = itvs[1:, 0] - itvs[:-1, 1]

        assert itvs.min() >= low
        np.testing.assert_array_less(-1, gaps)
        np.testing.assert_array_less(min_width - 1, widths)

        if max_width is not None:
            assert itvs.max() <= high
            np.testing.assert_array_less(widths, max_width + 1)
