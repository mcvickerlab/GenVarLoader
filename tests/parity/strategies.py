"""Hypothesis input strategies per migrated kernel (contract-valid generators)."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st


@st.composite
def intervals_to_tracks_inputs(draw):
    """Contract-valid inputs for ``intervals_to_tracks``.

    One interval-slice per query (offset_idxs = arange). Within a slice the
    intervals are sorted, non-overlapping, and start at >= the query's start.
    Covers empty slices, in-bounds paints, end-clamp (interval end past the
    out length), and break (interval start past the out length).
    """
    n_queries = draw(st.integers(min_value=1, max_value=6))

    starts_list: list[int] = []
    out_lengths: list[int] = []
    counts: list[int] = []
    itv_starts_all: list[int] = []
    itv_ends_all: list[int] = []
    itv_values_all: list[float] = []

    for _ in range(n_queries):
        qstart = draw(st.integers(min_value=0, max_value=500))
        length = draw(st.integers(min_value=0, max_value=48))
        starts_list.append(qstart)
        out_lengths.append(length)

        m = draw(st.integers(min_value=0, max_value=6))
        cur = qstart + draw(st.integers(min_value=0, max_value=10))  # first start >= qstart
        for _ in range(m):
            width = draw(st.integers(min_value=1, max_value=20))
            itv_starts_all.append(cur)
            itv_ends_all.append(cur + width)
            itv_values_all.append(
                draw(st.floats(width=32, allow_nan=False, allow_infinity=False))
            )
            cur = cur + width + draw(st.integers(min_value=0, max_value=10))  # gap -> non-overlap
        counts.append(m)

    offset_idxs = np.arange(n_queries, dtype=np.int64)
    starts = np.array(starts_list, dtype=np.int32)
    itv_starts = np.array(itv_starts_all, dtype=np.int32)
    itv_ends = np.array(itv_ends_all, dtype=np.int32)
    itv_values = np.array(itv_values_all, dtype=np.float32)
    itv_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out_offsets = np.concatenate([[0], np.cumsum(out_lengths)]).astype(np.int64)

    return offset_idxs, starts, itv_starts, itv_ends, itv_values, itv_offsets, out_offsets
