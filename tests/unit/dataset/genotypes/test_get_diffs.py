import numpy as np
from genvarloader._dataset._genotypes import get_diffs_sparse


def test_get_diffs_fast_path_no_variants():
    """Empty variant set -> diff is 0."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([], dtype=np.int32)
    geno_offsets = np.array([0, 0], dtype=np.int64)
    ilens = np.array([], dtype=np.int32)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
    )

    np.testing.assert_equal(diffs, np.array([[0]], dtype=np.int32))


def test_get_diffs_fast_path_sum():
    """Without spanning info, diff is sum of ilens for selected variants."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0, 1, 2], dtype=np.int32)
    geno_offsets = np.array([0, 3], dtype=np.int64)
    ilens = np.array([1, -2, 3], dtype=np.int32)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
    )

    np.testing.assert_equal(diffs, np.array([[2]], dtype=np.int32))


def test_get_diffs_fast_path_with_keep():
    """`keep` mask selects subset of variants."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0, 1, 2], dtype=np.int32)
    geno_offsets = np.array([0, 3], dtype=np.int64)
    ilens = np.array([1, -2, 3], dtype=np.int32)
    keep = np.array([True, False, True], dtype=np.bool_)
    keep_offsets = np.array([0, 3], dtype=np.int64)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
        keep=keep,
        keep_offsets=keep_offsets,
    )

    np.testing.assert_equal(diffs, np.array([[4]], dtype=np.int32))


def test_get_diffs_spanning_del_clipped_at_start():
    """Deletion starts before region — only the in-region portion counts.

    DEL at pos 0, ilen=-3. Atomized: v_end = 0 - min(0, -3) + 1 = 4.
    Region [2, 10). Clipping: v_ilen += max(0, 2 - 0 - 1) = 1 -> -2.
                              v_ilen += max(0, 4 - 10) = 0.
    Final per-haplotype diff: -2.
    """
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0], dtype=np.int32)
    geno_offsets = np.array([0, 1], dtype=np.int64)
    v_starts = np.array([0], dtype=np.int32)
    ilens = np.array([-3], dtype=np.int32)
    q_starts = np.array([2], dtype=np.int32)
    q_ends = np.array([10], dtype=np.int32)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
        q_starts=q_starts,
        q_ends=q_ends,
        v_starts=v_starts,
    )

    np.testing.assert_equal(diffs, np.array([[-2]], dtype=np.int32))


def test_get_diffs_variant_outside_region_skipped():
    """Variants past q_end do not contribute (kernel breaks out of loop)."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0, 1], dtype=np.int32)
    geno_offsets = np.array([0, 2], dtype=np.int64)
    v_starts = np.array([0, 20], dtype=np.int32)
    ilens = np.array([2, 5], dtype=np.int32)
    q_starts = np.array([0], dtype=np.int32)
    q_ends = np.array([10], dtype=np.int32)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
        q_starts=q_starts,
        q_ends=q_ends,
        v_starts=v_starts,
    )

    np.testing.assert_equal(diffs, np.array([[2]], dtype=np.int32))
