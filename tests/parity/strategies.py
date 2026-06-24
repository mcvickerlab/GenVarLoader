"""Hypothesis input strategies per migrated kernel (byte-identical generators)."""

from __future__ import annotations

import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import strategies as st


def splits_inputs():
    arrays = hnp.arrays(
        dtype=np.int64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=64),
        # non-negative values mirror the production input (memory-per-region counts)
        elements=st.integers(min_value=0, max_value=10_000),
    )
    max_values = st.floats(
        min_value=0.0, max_value=50_000.0, allow_nan=False, allow_infinity=False
    )
    return st.tuples(arrays, max_values)
