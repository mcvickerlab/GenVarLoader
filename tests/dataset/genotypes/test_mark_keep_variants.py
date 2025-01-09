import random

import numba as nb
import numpy as np
from genvarloader._dataset._genotypes import mark_keep_variants
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases


def case_snps():
    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 2, 3], np.int32)
    sizes = np.array([0, 0, 0], np.int32)

    desired = np.ones(3, np.bool_)

    return variant_idxs, dosages, positions, sizes, desired


def case_indels_0():
    # 000
    #   1
    #    0/1

    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 3, 4], np.int32)
    sizes = np.array([-3, 1, 0], np.int32)

    desired = ...

    return variant_idxs, dosages, positions, sizes, desired


def case_indels_1():
    # 000
    #   1
    #    0/1

    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 3, 4], np.int32)
    sizes = np.array([-3, 1, 0], np.int32)

    desired = np.array([])

    return variant_idxs, dosages, positions, sizes, desired


def case_del_span_start():
    # 00|0
    #  -|
    #  1|1
    #     0/1

    positions = np.array([0, 1, 1, 4], np.int32)
    sizes = np.array([-2, 0, -1, 0], np.int32)
    ref_start = 2

    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)

    # 1 random value for group assignments
    # 1 random value for end assignment
    _seed(0)
    rand = _random(2)
    if rand[0] < 0.5:
        # group = 0
        # probs = []
        pass
    desired = ...

    return variant_idxs, dosages, positions, sizes, ref_start, desired


def case_shift_ins():
    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 3], np.int32)
    sizes = np.array([-3, 0], np.int32)

    desired = ...

    return variant_idxs, dosages, positions, sizes, desired


@parametrize_with_cases(
    "variant_idxs, dosages, positions, sizes, ref_start, desired", cases="."
)
def test_mark_keep_variants(
    variant_idxs: NDArray[np.int32],
    dosages: NDArray[np.float32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    ref_start: int,
    desired: NDArray[np.bool_],
):
    n_variants = len(variant_idxs)

    groups = np.empty(n_variants, np.uint32)
    ends = np.empty(n_variants, np.uint32)

    _seed(0)
    actual = mark_keep_variants(
        variant_idxs, dosages, positions, sizes, ref_start, groups, ends
    )

    np.testing.assert_array_equal(actual, desired)


@nb.njit
def _seed(seed: int):
    random.seed(seed)


@nb.njit
def _random(n_values: int):
    return [random.random() for _ in range(n_values)]
