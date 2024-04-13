from typing import Tuple

import numpy as np
from einops import repeat
from genvarloader.dataset.genotypes import DenseGenotypes, SparseGenotypes
from pytest_cases import fixture


@fixture
def genos():
    n_samples = 2
    genos = np.array(
        [
            [-9, -9],
            [0, 1],
            [1, -9],
            [-9, -9],
            [1, 1],
            [1, -9],
        ],
        np.int8,
    )
    genos = repeat(genos, "v p -> s p v", s=n_samples)
    first_v_idxs = np.array([0, 1], np.uint32)
    offsets = np.array([0, 3, 6], np.uint32)
    # 2 regions
    # 2 samples
    # 2 haplotypes
    # each has exactly 1 non-alt genotype

    sparse_v_idxs = np.array([2, 1, 2, 1, 2, 3, 2, 2, 3, 2], np.int32)
    sparse_offsets = np.array([0, 1, 2, 3, 4, 6, 7, 9, 10], np.int32)
    dense = DenseGenotypes(
        genos=genos, first_v_idxs=first_v_idxs, offsets=offsets, n_samples=n_samples
    )
    sparse = SparseGenotypes(
        variant_idxs=sparse_v_idxs,
        offsets=sparse_offsets,
        n_regions=1,
        n_samples=n_samples,
        ploidy=2,
    )
    return dense, sparse


def test_from_dense(genos: Tuple[DenseGenotypes, SparseGenotypes]):
    dense, desired = genos

    actual = SparseGenotypes.from_dense(dense.genos, dense.first_v_idxs, dense.offsets)

    np.testing.assert_equal(actual.offsets, desired.offsets)
    np.testing.assert_equal(actual.variant_idxs, desired.variant_idxs)


def test_from_dense_with_length(genos: Tuple[DenseGenotypes, SparseGenotypes]):
    dense, desired = genos
    ilens = np.array([0, 0, -3, 0], np.int32)
    positions = np.array([0, 1, 2, 3], np.int32)
    starts = np.array([0, 1], np.int32)
    length = 3
    desired_max_ends = np.array([6, 7], np.int32)

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
