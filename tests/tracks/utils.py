from typing import Optional

import numpy as np


# vectorized version of https://stackoverflow.com/a/60171360
def nonnegative_integers_with_sum(n, total, rng: np.random.Generator):
    if n < 1 or total < 0:
        raise ValueError
    if total == 0:
        return np.zeros(n, np.int32)
    ls = np.empty(n + 1, np.int32)
    ls[0] = 0
    ls[1:-1] = rng.integers(0, total + 1, n - 1)
    ls[-1] = total
    ls.sort()
    rv = np.diff(ls).astype(np.int32)
    return rv


# vectorized version of https://stackoverflow.com/a/60171360
def nonoverlapping_intervals(
    n: int,
    low: int,
    high: int,
    min_width: int = 1,
    max_width: Optional[int] = None,
    seed: Optional[int] = None,
):
    if min_width < 1:
        raise ValueError("min_width must be greater than or equal to 1")
    if max_width is not None and max_width < min_width:
        raise ValueError("max_width must be greater than or equal to min_width")

    # total gap length = max_length - n_intervals * width
    max_gap_length = (high - low) - n * min_width
    if max_gap_length < 0:
        raise ValueError("Not enough space for n intervals of min_width")

    rng = np.random.default_rng(seed)

    gap_lengths = nonnegative_integers_with_sum(n, max_gap_length, rng)
    lengths = gap_lengths + min_width

    min_starts = np.full_like(lengths, low)
    min_starts[1:] += lengths[:-1].cumsum()

    starts = rng.integers(min_starts, min_starts + gap_lengths + 1, dtype=np.int32)

    min_ends = np.maximum(starts + min_width, min_starts + gap_lengths)
    max_ends = min_starts + lengths + 1
    ends = rng.integers(min_ends, max_ends, dtype=np.int32)

    if max_width is not None:
        ends = np.minimum(ends, starts + max_width)
        gaps = np.empty(n, np.int32)
        gaps[:-1] = starts[1:] - ends[:-1]
        gaps[-1] = high - ends[-1]
        shift = rng.integers(0, gaps + 1, dtype=np.int32)
        starts += shift
        ends += shift

    rvs = np.stack([starts, ends], 1)
    return rvs


n = 3
low = 0
max_width = 5
high = n * 5

itvs = nonoverlapping_intervals(n, low, high, max_width=max_width, seed=0)
itvs

_max = -1
for _ in range(10000):
    itvs = nonoverlapping_intervals(n, low, high, max_width=max_width)
    _max = max(_max, itvs.max())
    widths = np.diff(itvs, axis=1).squeeze()
    gaps = itvs[1:, 0] - itvs[:-1, 1]
    np.testing.assert_array_less(0, widths)
    np.testing.assert_array_less(-1, gaps)
_max
