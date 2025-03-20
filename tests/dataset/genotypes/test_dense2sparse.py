# %%

import numpy as np
from einops import repeat
from genvarloader._dataset._genotypes import (
    DenseGenotypes,
    SparseGenotypes,
    SparseSomaticGenotypes,
)
from pytest_cases import parametrize_with_cases


# %%
def case_snps():
    n_samples = 2
    ploidy = 2
    n_regions = 1
    genos = np.array([0, 1, -9], np.int8)
    n_alt = (genos == 1).sum()
    n_variants = len(genos)
    genos = repeat(genos, "v -> s p v", s=n_samples, p=ploidy)
    first_v_idxs = np.arange(n_regions, dtype=np.int32)
    offsets = np.array([0, n_variants], np.int64)

    sparse_v_idxs = (genos == 1).nonzero()[-1].astype(np.int32)
    sparse_offsets = np.empty(n_regions * ploidy * n_samples + 1, np.int64)
    sparse_offsets[0] = 0
    sparse_offsets[1:] = n_alt * np.arange(1, n_regions * ploidy * n_samples + 1)
    dense = DenseGenotypes(
        genos=genos, first_v_idxs=first_v_idxs, offsets=offsets, n_samples=n_samples
    )
    sparse = SparseGenotypes(
        variant_idxs=sparse_v_idxs,
        offsets=sparse_offsets,
        n_regions=n_regions,
        n_samples=n_samples,
        ploidy=ploidy,
    )

    sparse_v_idxs = (genos == 1).any(1).nonzero()[-1].astype(np.int32)
    sparse_offsets = np.empty(n_regions * n_samples + 1, np.int64)
    sparse_offsets[0] = 0
    sparse_offsets[1:] = n_alt * np.arange(1, n_regions * n_samples + 1)
    dosages = np.array([0, 0.5, -9.0], np.float32)
    dosages = repeat(dosages, "v -> s v", s=n_samples)
    sparse_somatic = SparseSomaticGenotypes(
        variant_idxs=sparse_v_idxs,
        dosages=dosages[:, [1]].T.ravel(),
        offsets=sparse_offsets,
        n_regions=n_regions,
        n_samples=n_samples,
    )

    ilens = np.zeros(n_variants, np.int32)
    positions = np.arange(n_variants, dtype=np.int32)
    starts = np.arange(n_regions, dtype=np.int32)
    lengths = np.full(n_regions, 3, np.int32)
    max_ends = np.array([3], np.int32)

    return (
        dense,
        sparse,
        sparse_somatic,
        ilens,
        positions,
        starts,
        lengths,
        max_ends,
        dosages,
    )


def case_indels():
    n_samples = 2
    ploidy = 2
    n_regions = 1
    genos = np.array([0, 1, -9, 1], np.int8)
    n_alt = (genos > 0).sum()
    n_variants = len(genos)
    genos = repeat(genos, "v -> s p v", s=n_samples, p=ploidy)
    first_v_idxs = np.arange(n_regions, dtype=np.int32)
    offsets = np.array([0, n_variants], np.int64)

    sparse_v_idxs = (genos > 0).nonzero()[-1].astype(np.int32)
    sparse_offsets = np.empty(n_regions * ploidy * n_samples + 1, np.int64)
    sparse_offsets[0] = 0
    sparse_offsets[1:] = n_alt * np.arange(1, n_regions * ploidy * n_samples + 1)
    dense = DenseGenotypes(
        genos=genos, first_v_idxs=first_v_idxs, offsets=offsets, n_samples=n_samples
    )
    sparse = SparseGenotypes(
        variant_idxs=sparse_v_idxs,
        offsets=sparse_offsets,
        n_regions=n_regions,
        n_samples=n_samples,
        ploidy=ploidy,
    )

    sparse_v_idxs = (genos == 1).any(1).nonzero()[-1].astype(np.int32)
    sparse_offsets = np.empty(n_regions * n_samples + 1, np.int64)
    sparse_offsets[0] = 0
    sparse_offsets[1:] = n_alt * np.arange(1, n_regions * n_samples + 1)
    dosages = np.array([0, 0.5, -9.0, 0.4], np.float32)
    dosages = repeat(dosages, "v -> s v", s=n_samples)
    sparse_somatic = SparseSomaticGenotypes(
        variant_idxs=sparse_v_idxs,
        dosages=dosages[:, [1, 3]].T.ravel(),
        offsets=sparse_offsets,
        n_regions=n_regions,
        n_samples=n_samples,
    )

    ilens = np.array([0, -2, 0, 2])
    positions = np.arange(n_variants, dtype=np.int32)
    starts = np.arange(n_regions, dtype=np.int32)
    lengths = np.full(n_regions, 4, np.int32)
    max_ends = np.array([4], np.int32)

    return (
        dense,
        sparse,
        sparse_somatic,
        ilens,
        positions,
        starts,
        lengths,
        max_ends,
        dosages,
    )


@parametrize_with_cases(
    "dense, sparse, sparse_somatic, ilens, positions, starts, length, max_ends, dosages",
    cases=".",
)
def test_from_dense(
    dense: DenseGenotypes,
    sparse: SparseGenotypes,
    sparse_somatic: SparseSomaticGenotypes,
    ilens,
    positions,
    starts,
    length,
    max_ends,
    dosages,
):
    desired = sparse
    actual = SparseGenotypes.from_dense(dense.genos, dense.first_v_idxs, dense.offsets)

    np.testing.assert_equal(actual.offsets, desired.offsets)
    np.testing.assert_equal(actual.variant_idxs, desired.variant_idxs)


@parametrize_with_cases(
    "dense, sparse, sparse_somatic, ilens, positions, starts, lengths, max_ends, dosages",
    cases=".",
)
def test_from_dense_with_length(
    dense: DenseGenotypes,
    sparse: SparseGenotypes,
    sparse_somatic: SparseSomaticGenotypes,
    ilens,
    positions,
    starts,
    lengths,
    max_ends,
    dosages,
):
    desired = sparse
    desired_max_ends = max_ends

    actual, max_ends = SparseGenotypes.from_dense_with_length(
        dense.genos,
        dense.first_v_idxs,
        dense.offsets,
        ilens,
        positions,
        starts,
        lengths,
    )

    np.testing.assert_equal(actual.offsets, desired.offsets)
    np.testing.assert_equal(actual.variant_idxs, desired.variant_idxs)
    np.testing.assert_equal(max_ends, desired_max_ends)


@parametrize_with_cases(
    "dense, sparse, sparse_somatic, ilens, positions, starts, lengths, max_ends, dosages",
    cases=".",
)
def test_somatic_from_dense(
    dense: DenseGenotypes,
    sparse: SparseGenotypes,
    sparse_somatic: SparseSomaticGenotypes,
    ilens,
    positions,
    starts,
    lengths,
    max_ends,
    dosages,
):
    desired = sparse_somatic
    actual = SparseSomaticGenotypes.from_dense(
        dense.genos, dense.first_v_idxs, dense.offsets, dosages
    )

    np.testing.assert_equal(actual.offsets, desired.offsets)
    np.testing.assert_equal(actual.variant_idxs, desired.variant_idxs)
    np.testing.assert_equal(actual.dosages, desired.dosages)


@parametrize_with_cases(
    "dense, sparse, sparse_somatic, ilens, positions, starts, lengths, max_ends, dosages",
    cases=".",
)
def test_somatic_from_dense_with_length(
    dense: DenseGenotypes,
    sparse: SparseGenotypes,
    sparse_somatic: SparseSomaticGenotypes,
    ilens,
    positions,
    starts,
    lengths,
    max_ends,
    dosages,
):
    desired = sparse_somatic
    desired_max_ends = max_ends

    actual, max_ends = SparseSomaticGenotypes.from_dense_with_length(
        dense.genos,
        dense.first_v_idxs,
        dense.offsets,
        ilens,
        positions,
        starts,
        lengths,
        dosages,
    )

    np.testing.assert_equal(actual.offsets, desired.offsets)
    np.testing.assert_equal(actual.variant_idxs, desired.variant_idxs)
    np.testing.assert_equal(max_ends, desired_max_ends)


# %%
# test_from_dense_with_length(*case_snps())
# test_from_dense_with_length(*case_indels())
