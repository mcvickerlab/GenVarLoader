"""Unit tests for build_splice_plan: permutation + offset math."""

import numpy as np
import pytest

from genvarloader._dataset._splice import SplicePlan, build_splice_plan


def test_plan_no_inner_axes():
    """E=1 case (RefDataset): plan is essentially an identity grouping."""
    # 2 splice rows × 1 sample, row 0 has 2 elements, row 1 has 1 element.
    # B = 3 queries total.
    lengths = np.array([3, 4, 5], dtype=np.int32)  # shape (3,)
    splice_row_offsets = np.array([0, 2, 3], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    # No inner fixed, so perm is identity.
    np.testing.assert_array_equal(plan.perm, [0, 1, 2])
    np.testing.assert_array_equal(plan.permuted_lengths, [3, 4, 5])
    np.testing.assert_array_equal(plan.permuted_out_offsets, [0, 3, 7, 12])
    # group_offsets at (row, sample) granularity: 2 entries + 1.
    np.testing.assert_array_equal(plan.group_offsets, [0, 7, 12])
    assert plan.out_shape == (2, 1, None)


def test_plan_ploidy_2():
    """B=3 queries × P=2 ploidy. Each splice row's ploidies must be contiguous.

    Splice layout:
      row 0, sample 0 = elements [0, 1]  (2 elements)
      row 1, sample 0 = elements [2]     (1 element)

    Inner-fixed lengths (B, P) where B=3, P=2:
      query 0 (row 0 elem 0): ploidy lens [10, 11]
      query 1 (row 0 elem 1): ploidy lens [20, 21]
      query 2 (row 1 elem 0): ploidy lens [30, 31]

    Desired permuted order (row, sample, ploidy, element) C-order:
      (r=0, s=0, p=0, e=0), (r=0, s=0, p=0, e=1),
      (r=0, s=0, p=1, e=0), (r=0, s=0, p=1, e=1),
      (r=1, s=0, p=0, e=0),
      (r=1, s=0, p=1, e=0)

    k_idx in current layout is (query, ploidy) C-order:
      k = [(q0,p0), (q0,p1), (q1,p0), (q1,p1), (q2,p0), (q2,p1)]
        = [0, 1, 2, 3, 4, 5]

    So perm pulls k-indices in this order:
      k(q=0, p=0)=0, k(q=1, p=0)=2,   # row 0 sample 0 ploidy 0 elements
      k(q=0, p=1)=1, k(q=1, p=1)=3,   # row 0 sample 0 ploidy 1 elements
      k(q=2, p=0)=4,                  # row 1 sample 0 ploidy 0
      k(q=2, p=1)=5,                  # row 1 sample 0 ploidy 1
    """
    lengths = np.array(
        [[10, 11], [20, 21], [30, 31]], dtype=np.int32
    )  # (B=3, P=2)
    splice_row_offsets = np.array([0, 2, 3], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    np.testing.assert_array_equal(plan.perm, [0, 2, 1, 3, 4, 5])
    np.testing.assert_array_equal(
        plan.permuted_lengths, [10, 20, 11, 21, 30, 31]
    )
    np.testing.assert_array_equal(
        plan.permuted_out_offsets, [0, 10, 30, 41, 62, 92, 123]
    )
    # group_offsets at (row, sample, ploidy) granularity: 2*1*2 = 4 cells + 1.
    # cell sums: row0,s0,p0 = 10+20=30; row0,s0,p1 = 11+21=32; row1,s0,p0 = 30; row1,s0,p1 = 31.
    np.testing.assert_array_equal(plan.group_offsets, [0, 30, 62, 92, 123])
    assert plan.out_shape == (2, 1, 2, None)


def test_plan_multi_sample_ploidy_2():
    """n_samples=2, ploidy=2. Verify (row, sample, ploidy) C-order."""
    # 1 splice row × 2 samples. Row has 2 elements.
    # B = 4 queries (row, sample, element) C-order:
    #   q0 = (r=0, s=0, e=0)
    #   q1 = (r=0, s=0, e=1)
    #   q2 = (r=0, s=1, e=0)
    #   q3 = (r=0, s=1, e=1)
    lengths = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32
    )  # (B=4, P=2)
    splice_row_offsets = np.array([0, 2, 4], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=2,
        n_rows=1,
    )
    # k_idx = query * P + ploidy
    # Desired order: (r, s, p, e). For r=0:
    #   s=0, p=0: e=0 → k(q=0,p=0)=0; e=1 → k(q=1,p=0)=2
    #   s=0, p=1: e=0 → k(q=0,p=1)=1; e=1 → k(q=1,p=1)=3
    #   s=1, p=0: e=0 → k(q=2,p=0)=4; e=1 → k(q=3,p=0)=6
    #   s=1, p=1: e=0 → k(q=2,p=1)=5; e=1 → k(q=3,p=1)=7
    np.testing.assert_array_equal(plan.perm, [0, 2, 1, 3, 4, 6, 5, 7])
    np.testing.assert_array_equal(
        plan.permuted_lengths, [1, 3, 2, 4, 5, 7, 6, 8]
    )
    # group_offsets at (1, 2, 2) granularity = 4 cells + 1.
    # cell sums: 1+3=4, 2+4=6, 5+7=12, 6+8=14
    np.testing.assert_array_equal(plan.group_offsets, [0, 4, 10, 22, 36])
    assert plan.out_shape == (1, 2, 2, None)


def test_plan_total_bytes_consistent():
    """sum(lengths) == permuted_out_offsets[-1] == group_offsets[-1]."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 20, size=(6, 3), dtype=np.int32)
    splice_row_offsets = np.array([0, 2, 4, 6], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=3,
    )
    total = int(lengths.sum())
    assert int(plan.permuted_out_offsets[-1]) == total
    assert int(plan.group_offsets[-1]) == total


def test_plan_single_element_rows():
    """Every splice row has exactly one element — no concatenation needed."""
    lengths = np.array([[5, 6], [7, 8]], dtype=np.int32)
    splice_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    # With singleton splice rows the permutation still groups by (r, s, p).
    np.testing.assert_array_equal(plan.perm, [0, 1, 2, 3])
    np.testing.assert_array_equal(plan.permuted_lengths, [5, 6, 7, 8])


def test_plan_inner_fixed_size_3():
    """E=3: e.g. a track axis of 3 stacked tracks. Verify general inner-fixed handling."""
    # 1 splice row × 1 sample × 2 elements × 3 tracks.
    # B = 2 queries × 3 tracks = 6 inner k-indices.
    lengths = np.array(
        [[1, 2, 3], [4, 5, 6]], dtype=np.int32
    )  # (B=2, T=3)
    splice_row_offsets = np.array([0, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=1,
    )
    # k = query*T + t. Desired order (r=0, s=0, t, e):
    #   t=0: e=0 → k=0; e=1 → k=3
    #   t=1: e=0 → k=1; e=1 → k=4
    #   t=2: e=0 → k=2; e=1 → k=5
    np.testing.assert_array_equal(plan.perm, [0, 3, 1, 4, 2, 5])
    np.testing.assert_array_equal(plan.permuted_lengths, [1, 4, 2, 5, 3, 6])
    np.testing.assert_array_equal(plan.group_offsets, [0, 5, 12, 21])
    assert plan.out_shape == (1, 1, 3, None)


def test_plan_dtype_invariants():
    """perm is intp, lengths is int32, offsets is int64-compatible."""
    lengths = np.array([3, 4], dtype=np.int32)
    splice_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    assert plan.perm.dtype == np.intp
    assert plan.permuted_lengths.dtype == np.int32
    # offset arrays use seqpro's OFFSET_TYPE (int64).
    assert plan.permuted_out_offsets.dtype == np.int64
    assert plan.group_offsets.dtype == np.int64
