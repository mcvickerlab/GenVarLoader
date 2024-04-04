import numpy as np
from einops import repeat
from pytest_cases import fixture

from genvarloader.dataset.genotypes import DenseGenotypes, SparseGenotypes


@fixture
def dense_genos() -> DenseGenotypes:
    n_samples = 2
    genos = np.array(
        [
            [-9, -9],
            [0, 1],
            [1, -9],
        ],
        np.int8,
    )
    genos = repeat(genos, "v p -> (s v) p", s=n_samples)
    first_v_idxs = np.array([0], np.uint32)
    offsets = np.array([0, 3], np.uint32)
    return DenseGenotypes(
        genos=genos, first_v_idxs=first_v_idxs, offsets=offsets, n_samples=n_samples
    )


def test_dense2sparse(dense_genos: DenseGenotypes):
    n_samples = dense_genos.n_samples
    desired_genos = np.array(
        [
            [0, 1],
            [1, -9],
        ],
        np.int8,
    )
    desired_genos = repeat(desired_genos, "v p -> (s v) p", s=n_samples)
    desired = SparseGenotypes(
        genos=desired_genos,
        variant_idxs=np.array([1, 2, 1, 2], np.uintp),
        offsets=np.array([0, 2, 4], np.uintp),
        n_samples=n_samples,
        n_regions=1,
    )

    actual = dense_genos.to_sparse()

    np.testing.assert_equal(actual.genos, desired.genos)
    np.testing.assert_equal(actual.variant_idxs, desired.variant_idxs)
    np.testing.assert_equal(actual.offsets, desired.offsets)
    assert actual.n_samples == desired.n_samples
    assert actual.n_regions == desired.n_regions
