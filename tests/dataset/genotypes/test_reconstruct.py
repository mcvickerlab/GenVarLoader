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
    alt_alleles = np.frombuffer(b"AT", dtype=np.uint8)
    alt_offsets = np.array([0, 1, 2], dtype=np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    desired = np.frombuffer(b"CGT", dtype="S1")
    ref_start = 1

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
        desired,
        ref_start,
    )


def case_indels():
    positions = np.array([1, 3], np.int32)
    sizes = np.array([-1, 1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1, 1]]], dtype=np.int8)
    first_v_idxs = np.array([0], dtype=np.int32)
    offsets = np.array([0, 2], np.int64)

    shift = 0
    alt_alleles = np.frombuffer(b"GAT", dtype=np.uint8)
    alt_offsets = np.array([0, 1, 3], np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    desired = np.frombuffer(b"AGAT", dtype="S1")
    ref_start = 0

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
        desired,
        ref_start,
    )


def case_spanning_del():
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
    desired = np.frombuffer(b"GGN", dtype="S1")
    ref_start = 1

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
        desired,
        ref_start,
    )


def case_shift_ins():
    positions = np.array([1, 3], np.int32)
    sizes = np.array([1, 1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1, 1]]], dtype=np.int8)
    first_v_idxs = np.array([0], dtype=np.int32)
    offsets = np.array([0, 2], np.int64)

    shift = 1
    alt_alleles = np.frombuffer(b"TCGA", dtype=np.uint8)
    alt_offsets = np.array([0, 2, 4], np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    desired = np.frombuffer(b"TCGG", dtype="S1")
    ref_start = 0

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
        desired,
        ref_start,
    )


@parametrize_with_cases(
    "positions, sizes, shift, alt_alleles, alt_offsets, ref, sparse_genos, desired, ref_start",
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
    desired,
    ref_start,
):
    offset_idx = 0
    actual = np.empty(len(ref) - ref_start, np.uint8)
    reconstruct_haplotype_from_sparse(
        offset_idx=offset_idx,
        variant_idxs=sparse_genos.variant_idxs,
        offsets=sparse_genos.offsets,
        positions=positions,
        sizes=sizes,
        shift=shift,
        alt_alleles=alt_alleles,
        alt_offsets=alt_offsets,
        ref=ref,
        ref_start=ref_start,
        out=actual,
        pad_char=ord(b"N"),
    )

    np.testing.assert_equal(actual.view("S1"), desired)


# %%
test_sparse(*case_snps())
test_sparse(*case_indels())
test_sparse(*case_spanning_del())
test_sparse(*case_shift_ins())
