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
        cur = qstart + draw(
            st.integers(min_value=0, max_value=10)
        )  # first start >= qstart
        for _ in range(m):
            width = draw(st.integers(min_value=1, max_value=20))
            itv_starts_all.append(cur)
            itv_ends_all.append(cur + width)
            itv_values_all.append(
                draw(st.floats(width=32, allow_nan=False, allow_infinity=False))
            )
            cur = (
                cur + width + draw(st.integers(min_value=0, max_value=10))
            )  # gap -> non-overlap
        counts.append(m)

    offset_idxs = np.arange(n_queries, dtype=np.int64)
    starts = np.array(starts_list, dtype=np.int32)
    itv_starts = np.array(itv_starts_all, dtype=np.int32)
    itv_ends = np.array(itv_ends_all, dtype=np.int32)
    itv_values = np.array(itv_values_all, dtype=np.float32)
    itv_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out_offsets = np.concatenate([[0], np.cumsum(out_lengths)]).astype(np.int64)

    return (
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out_offsets,
    )


@st.composite
def _sparse_geno(draw, max_queries=4, max_ploidy=2, max_vars_per_group=5,
                 max_total_unique=12):
    """Shared sparse-genotype layout: returns
    (geno_offset_idx (q,p) int64, geno_v_idxs int32, geno_offsets (n+1,) int64,
     v_starts int32, ilens int32, q_starts int32, q_ends int32).
    geno_offset_idx is arange so each (q,p) row maps to its own offset slice."""
    n_unique = draw(st.integers(min_value=1, max_value=max_total_unique))
    v_starts = np.sort(
        draw(st.lists(st.integers(0, 1000), min_size=n_unique, max_size=n_unique)
             .map(np.array))
    ).astype(np.int32)
    ilens = np.array(
        draw(st.lists(st.integers(-5, 5), min_size=n_unique, max_size=n_unique)),
        dtype=np.int32,
    )
    n_q = draw(st.integers(1, max_queries))
    p = draw(st.integers(1, max_ploidy))
    n_groups = n_q * p
    counts = [draw(st.integers(0, max_vars_per_group)) for _ in range(n_groups)]
    v_idx_list = []
    for c in counts:
        # sorted variant indices within a group (reconstruction assumes sorted pos)
        idxs = sorted(draw(st.lists(st.integers(0, n_unique - 1),
                                    min_size=c, max_size=c)))
        v_idx_list.extend(idxs)
    geno_v_idxs = np.array(v_idx_list, dtype=np.int32)
    geno_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    geno_offset_idx = np.arange(n_groups, dtype=np.int64).reshape(n_q, p)
    q_starts = np.array(
        draw(st.lists(st.integers(0, 800), min_size=n_q, max_size=n_q)), np.int32
    )
    q_ends = (q_starts + draw(st.integers(1, 200))).astype(np.int32)
    return (geno_offset_idx, geno_v_idxs, geno_offsets, v_starts, ilens,
            q_starts, q_ends)


@st.composite
def get_diffs_sparse_inputs(draw):
    (goi, gvi, goff, vstarts, ilens, qstarts, qends) = draw(_sparse_geno())
    mode = draw(st.sampled_from(["plain", "keep", "query"]))
    twod = draw(st.booleans())
    offsets = goff if not twod else np.stack([goff[:-1], goff[1:]]).astype(np.int64)
    n_groups = goi.size
    total = int(goff[-1])
    if mode == "plain":
        return (goi, gvi, offsets, ilens, None, None, None, None, None)
    if mode == "keep":
        keep = np.array(
            draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_
        )
        return (goi, gvi, offsets, ilens, keep, goff.copy(), None, None, None)
    # query mode (optionally also keep)
    keep = None
    keep_off = None
    if draw(st.booleans()):
        keep = np.array(
            draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_
        )
        keep_off = goff.copy()
    return (goi, gvi, offsets, ilens, keep, keep_off, qstarts, qends, vstarts)


@st.composite
def choose_exonic_variants_inputs(draw):
    (goi, gvi, goff, vstarts, ilens, qstarts, qends) = draw(_sparse_geno())
    twod = draw(st.booleans())
    offsets = goff if not twod else np.stack([goff[:-1], goff[1:]]).astype(np.int64)
    return (qstarts, qends, goi, gvi, offsets, vstarts, ilens)


@st.composite
def gather_rows_inputs(draw, dtype=np.int32):
    n_groups = draw(st.integers(1, 6))
    counts = [draw(st.integers(0, 5)) for _ in range(n_groups)]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        elements = st.floats(width=32, allow_nan=False, allow_infinity=False)
    else:
        elements = st.integers(0, 1000)
    data = np.array(
        draw(st.lists(elements, min_size=total, max_size=total)), dt
    )
    n_rows = draw(st.integers(1, 8))
    goi = np.array(
        draw(st.lists(st.integers(0, n_groups - 1), min_size=n_rows, max_size=n_rows)),
        np.int64,
    )
    twod = draw(st.booleans())
    off = offsets if not twod else np.stack([offsets[:-1], offsets[1:]]).astype(np.int64)
    return (goi, off, data)


@st.composite
def gather_alleles_inputs(draw):
    n_unique = draw(st.integers(1, 8))
    lens = [draw(st.integers(0, 5)) for _ in range(n_unique)]
    allele_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    total = int(allele_offsets[-1])
    allele_bytes = np.array(
        draw(st.lists(st.integers(0, 255), min_size=total, max_size=total)), np.uint8
    )
    m = draw(st.integers(0, 10))
    v_idxs = np.array(
        draw(st.lists(st.integers(0, n_unique - 1), min_size=m, max_size=m)), np.int32
    )
    return (v_idxs, allele_bytes, allele_offsets)


@st.composite
def compact_keep_inputs(draw, dtype):
    """Generate (values[dtype], row_offsets int64, keep bool) for compact_keep tests."""
    n_rows = draw(st.integers(1, 6))
    counts = [draw(st.integers(0, 5)) for _ in range(n_rows)]
    row_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(row_offsets[-1])
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        elements = st.floats(width=32, allow_nan=False, allow_infinity=False)
    else:
        elements = st.integers(0, 1000)
    values = np.array(
        draw(st.lists(elements, min_size=total, max_size=total)), dt
    )
    keep = np.array(
        draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_
    )
    return (values, row_offsets, keep)
