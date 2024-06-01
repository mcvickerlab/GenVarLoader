# %%

import numpy as np
from einops import repeat
from genvarloader.dataset.genotypes import DenseGenotypes, SparseGenotypes
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
    offsets = np.array([0, n_variants], np.int32)

    sparse_v_idxs = (genos == 1).nonzero()[-1].astype(np.int32)
    sparse_offsets = np.empty(n_regions * ploidy * n_samples + 1, np.int32)
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

    ilens = np.zeros(n_variants, np.int32)
    positions = np.arange(n_variants, dtype=np.int32)
    starts = np.arange(n_regions, dtype=np.int32)
    length = 3
    max_ends = np.array([length], np.int32)

    return dense, sparse, ilens, positions, starts, length, max_ends


def case_indels():
    n_samples = 2
    ploidy = 2
    n_regions = 1
    genos = np.array([0, 1, -9, 1], np.int8)
    n_alt = (genos > 0).sum()
    n_variants = len(genos)
    genos = repeat(genos, "v -> s p v", s=n_samples, p=ploidy)
    first_v_idxs = np.arange(n_regions, dtype=np.int32)
    offsets = np.array([0, n_variants], np.int32)

    sparse_v_idxs = (genos > 0).nonzero()[-1].astype(np.int32)
    sparse_offsets = np.empty(n_regions * ploidy * n_samples + 1, np.int32)
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

    ilens = np.array([0, -2, 0, 2])
    positions = np.arange(n_variants, dtype=np.int32)
    starts = np.arange(n_regions, dtype=np.int32)
    length = 4
    max_ends = np.array([length], np.int32)

    return dense, sparse, ilens, positions, starts, length, max_ends


@parametrize_with_cases(
    "dense, sparse, ilens, positions, starts, length, max_ends", cases="."
)
def test_from_dense(
    dense: DenseGenotypes,
    sparse: SparseGenotypes,
    ilens,
    positions,
    starts,
    length,
    max_ends,
):
    desired = sparse
    actual = SparseGenotypes.from_dense(dense.genos, dense.first_v_idxs, dense.offsets)

    np.testing.assert_equal(actual.offsets, desired.offsets)
    np.testing.assert_equal(actual.variant_idxs, desired.variant_idxs)


@parametrize_with_cases(
    "dense, sparse, ilens, positions, starts, length, max_ends", cases="."
)
def test_from_dense_with_length(
    dense: DenseGenotypes,
    sparse: SparseGenotypes,
    ilens,
    positions,
    starts,
    length,
    max_ends,
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
        length,
    )

    np.testing.assert_equal(actual.offsets, desired.offsets)
    np.testing.assert_equal(actual.variant_idxs, desired.variant_idxs)
    np.testing.assert_equal(max_ends, desired_max_ends)


# %%
test_from_dense_with_length(*case_snps())
test_from_dense_with_length(*case_indels())
