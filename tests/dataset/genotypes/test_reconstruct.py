# %%
import numpy as np
from genvarloader._dataset._genotypes import (
    SparseGenotypes,
    reconstruct_haplotype_from_sparse,
)
from pytest_cases import parametrize_with_cases


# %%
def case_snps():
    positions = np.array([1, 3], np.int32)
    sizes = np.zeros(2, dtype=np.int32)

    # (s p v)
    genos = np.array([[[0, 1]]], dtype=np.int8)
    first_v_idxs = np.array([0], dtype=np.int32)
    offsets = np.array([0, 2], np.int64)

    shift = 0
    alt_alleles = np.frombuffer(b"A" + b"T", dtype=np.uint8)
    alt_offsets = np.array([0, 1, 2], dtype=np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 1

    desired = np.frombuffer(b"CGT", dtype="S1")
    annot_v_idxs = np.array([-1, -1, 1], dtype=np.int32)
    annot_pos = np.array([1, 2, 3], dtype=np.int32)

    sparse_genos = SparseGenotypes.from_dense(
        genos=genos, first_v_idxs=first_v_idxs, offsets=offsets
    )

    return (
        positions,
        sizes,
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
    positions = np.array([1, 3], np.int32)
    sizes = np.array([-1, 1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1, 1]]], dtype=np.int8)
    first_v_idxs = np.array([0], dtype=np.int32)
    offsets = np.array([0, 2], np.int64)

    shift = 0
    alt_alleles = np.frombuffer(b"G" + b"AT", dtype=np.uint8)
    alt_offsets = np.array([0, 1, 3], np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 0

    desired = np.frombuffer(b"AGAT", dtype="S1")
    annot_v_idxs = np.array([-1, 0, 1, 1], dtype=np.int32)
    annot_pos = np.array([0, 1, 3, 3], dtype=np.int32)

    sparse_genos = SparseGenotypes.from_dense(
        genos=genos, first_v_idxs=first_v_idxs, offsets=offsets
    )

    return (
        positions,
        sizes,
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
    positions = np.array([0], np.int32)
    sizes = np.array([-1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1]]], dtype=np.int8)
    first_v_idxs = np.array([0], dtype=np.int32)
    offsets = np.array([0, 1], np.int64)

    shift = 0
    alt_alleles = np.frombuffer(b"G", dtype=np.uint8)
    alt_offsets = np.array([0, 1], np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 1

    desired = np.frombuffer(b"GGN", dtype="S1")
    annot_v_idxs = np.array([-1, -1, -1], dtype=np.int32)
    annot_pos = np.array([2, 3, -1], dtype=np.int32)

    sparse_genos = SparseGenotypes.from_dense(
        genos=genos, first_v_idxs=first_v_idxs, offsets=offsets
    )

    return (
        positions,
        sizes,
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
    positions = np.array([1, 3], np.int32)
    sizes = np.array([1, 1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1, 1]]], dtype=np.int8)
    first_v_idxs = np.array([0], dtype=np.int32)
    offsets = np.array([0, 2], np.int64)

    shift = 1
    alt_alleles = np.frombuffer(b"TC" + b"GA", dtype=np.uint8)
    alt_offsets = np.array([0, 2, 4], np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 0

    desired = np.frombuffer(b"TCGG", dtype="S1")
    annot_v_idxs = np.array([0, 0, -1, 1], dtype=np.int32)
    annot_pos = np.array([1, 1, 2, 3], dtype=np.int32)

    sparse_genos = SparseGenotypes.from_dense(
        genos=genos, first_v_idxs=first_v_idxs, offsets=offsets
    )

    return (
        positions,
        sizes,
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


@parametrize_with_cases(
    "positions, sizes, shift, alt_alleles, alt_offsets, ref, sparse_genos, ref_start, desired, annot_v_idxs, annot_pos",
    cases=".",
)
def test_sparse(
    positions,
    sizes,
    shift,
    alt_alleles,
    alt_offsets,
    ref,
    sparse_genos: SparseGenotypes,
    ref_start,
    desired,
    annot_v_idxs,
    annot_pos,
):
    offset_idx = 0
    actual = np.empty(len(ref) - ref_start, np.uint8)
    actual_annot_v_idxs = np.empty(len(ref) - ref_start, np.int32)
    actual_annot_pos = np.empty(len(ref) - ref_start, np.int32)
    reconstruct_haplotype_from_sparse(
        offset_idx=offset_idx,
        geno_v_idxs=sparse_genos.variant_idxs,
        geno_offsets=sparse_genos.offsets,
        positions=positions,
        sizes=sizes,
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
