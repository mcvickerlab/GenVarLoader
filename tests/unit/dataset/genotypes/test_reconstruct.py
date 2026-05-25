# %%
import numpy as np
from genoray._svar import dense2sparse
from genoray._types import V_IDX_TYPE
from seqpro.rag import Ragged
from genvarloader._dataset._genotypes import reconstruct_haplotype_from_sparse
from pytest_cases import parametrize_with_cases


# %%
def case_snps():
    v_starts = np.array([1, 3], np.int32)
    ilens = np.zeros(2, dtype=np.int32)

    # (s p v)
    genos = np.array([[[0, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"A" + b"T", dtype=np.uint8)
    alt_offsets = np.array([0, 1, 2], dtype=np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 1

    desired = np.frombuffer(b"CGT", dtype="S1")
    annot_v_idxs = np.array([-1, -1, 1], dtype=np.int32)
    annot_pos = np.array([1, 2, 3], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        alt_alleles,
        alt_offsets,
        ref,
        sparse_genos,
        ref_start,
        desired,
        annot_v_idxs,
        annot_pos,
    )


def case_indels():
    v_starts = np.array([1, 3], np.int32)
    ilens = np.array([-1, 1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"G" + b"AT", dtype=np.uint8)
    alt_offsets = np.array([0, 1, 3], np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 0

    desired = np.frombuffer(b"AGAT", dtype="S1")
    annot_v_idxs = np.array([-1, 0, 1, 1], dtype=np.int32)
    annot_pos = np.array([0, 1, 3, 3], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        alt_alleles,
        alt_offsets,
        ref,
        sparse_genos,
        ref_start,
        desired,
        annot_v_idxs,
        annot_pos,
    )


def case_spanning_del_pad():
    v_starts = np.array([0], np.int32)
    ilens = np.array([-1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"G", dtype=np.uint8)
    alt_offsets = np.array([0, 1], np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 1

    desired = np.frombuffer(b"GGN", dtype="S1")
    annot_v_idxs = np.array([-1, -1, -1], dtype=np.int32)
    annot_pos = np.array([2, 3, np.iinfo(np.int32).max], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        alt_alleles,
        alt_offsets,
        ref,
        sparse_genos,
        ref_start,
        desired,
        annot_v_idxs,
        annot_pos,
    )


def case_shift_ins():
    v_starts = np.array([1, 3], np.int32)
    ilens = np.array([1, 1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=np.int32)

    shift = 1
    alt_alleles = np.frombuffer(b"TC" + b"GA", dtype=np.uint8)
    alt_offsets = np.array([0, 2, 4], np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 0

    desired = np.frombuffer(b"TCGG", dtype="S1")
    annot_v_idxs = np.array([0, 0, -1, 1], dtype=np.int32)
    annot_pos = np.array([1, 1, 2, 3], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        alt_alleles,
        alt_offsets,
        ref,
        sparse_genos,
        ref_start,
        desired,
        annot_v_idxs,
        annot_pos,
    )


def case_ref_only():
    """No variants applied — output is pure reference slice."""
    v_starts = np.array([], np.int32)
    ilens = np.array([], np.int32)
    genos = np.zeros((1, 1, 0), dtype=np.int8)  # (s p v)
    var_idxs = np.array([], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"", dtype=np.uint8)
    alt_offsets = np.array([0], dtype=np.uintp)
    ref = np.frombuffer(b"ACGT", dtype=np.uint8)
    ref_start = 0

    desired = np.frombuffer(b"ACGT", dtype="S1")
    annot_v_idxs = np.array([-1, -1, -1, -1], dtype=np.int32)
    annot_pos = np.array([0, 1, 2, 3], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts, ilens, shift, alt_alleles, alt_offsets, ref,
        sparse_genos, ref_start, desired, annot_v_idxs, annot_pos,
    )


def case_spanning_del_end():
    """Deletion runs past region end — output truncates at out length."""
    v_starts = np.array([2], np.int32)
    ilens = np.array([-2], dtype=np.int32)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"G", dtype=np.uint8)
    alt_offsets = np.array([0, 1], np.uintp)
    ref = np.frombuffer(b"ACGTA", dtype=np.uint8)
    ref_start = 0

    desired = np.frombuffer(b"ACGNN", dtype="S1")
    annot_v_idxs = np.array([-1, -1, 0, -1, -1], dtype=np.int32)
    annot_pos = np.array([0, 1, 2, np.iinfo(np.int32).max, np.iinfo(np.int32).max], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts, ilens, shift, alt_alleles, alt_offsets, ref,
        sparse_genos, ref_start, desired, annot_v_idxs, annot_pos,
    )


def case_overlapping_variants():
    """Two variants at same position — first ALT wins, second skipped."""
    v_starts = np.array([1, 1], np.int32)
    ilens = np.zeros(2, dtype=np.int32)
    # Both variants are ALT so both enter sparse data; kernel applies the first
    # (variant 0, ALT=T) and skips the second (v_pos < ref_idx after first applied).
    genos = np.array([[[1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"T" + b"G", dtype=np.uint8)
    alt_offsets = np.array([0, 1, 2], dtype=np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 0

    desired = np.frombuffer(b"ATGG", dtype="S1")
    annot_v_idxs = np.array([-1, 0, -1, -1], dtype=np.int32)
    annot_pos = np.array([0, 1, 2, 3], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts, ilens, shift, alt_alleles, alt_offsets, ref,
        sparse_genos, ref_start, desired, annot_v_idxs, annot_pos,
    )


@parametrize_with_cases(
    "v_starts, ilens, shift, alt_alleles, alt_offsets, ref, sparse_genos, ref_start, desired, annot_v_idxs, annot_pos",
    cases=".",
)
def test_sparse(
    v_starts,
    ilens,
    shift,
    alt_alleles,
    alt_offsets,
    ref,
    sparse_genos: Ragged[V_IDX_TYPE],
    ref_start,
    desired,
    annot_v_idxs,
    annot_pos,
):
    actual = np.empty(len(ref) - ref_start, np.uint8)
    actual_annot_v_idxs = np.empty(len(ref) - ref_start, np.int32)
    actual_annot_pos = np.empty(len(ref) - ref_start, np.int32)
    reconstruct_haplotype_from_sparse(
        v_idxs=sparse_genos.data,
        v_starts=v_starts,
        ilens=ilens,
        shift=shift,
        alt_alleles=alt_alleles,
        alt_offsets=alt_offsets,
        ref=ref,
        ref_start=ref_start,
        out=actual,
        pad_char=ord(b"N"),
        annot_v_idxs=actual_annot_v_idxs,
        annot_ref_pos=actual_annot_pos,
    )

    np.testing.assert_equal(actual.view("S1"), desired)
    np.testing.assert_equal(actual_annot_v_idxs, annot_v_idxs)
    np.testing.assert_equal(actual_annot_pos, annot_pos)


# %%
# test_sparse(*case_snps())
# test_sparse(*case_indels())
# test_sparse(*case_spanning_del_pad())
# test_sparse(*case_shift_ins())
