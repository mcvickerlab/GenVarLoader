import random

import numba as nb
import numpy as np
from genvarloader._dataset._genotypes import _mark_keep_variants
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases


def case_snps():
    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 2, 3], np.int32)
    sizes = np.array([0, 0, 0], np.int32)
    ref_start = 0
    target_len = 4

    desired = np.ones(3, np.bool_)

    return variant_idxs, dosages, positions, sizes, ref_start, target_len, desired


def case_extra_snps():
    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 2, 3], np.int32)
    sizes = np.array([0, 0, 0], np.int32)
    ref_start = 0
    target_len = 2

    desired = np.array([True, False, False], np.bool_)

    return variant_idxs, dosages, positions, sizes, ref_start, target_len, desired


def case_indels_0():
    # group 0
    # 0 1 2 3 4  5
    # r d - - - r/s

    # group 1
    # 0 1 2 3  4  5
    # r r r ii r r/s

    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 3, 5], np.int32)
    sizes = np.array([-3, 1, 0], np.int32)
    ref_start = 0
    target_len = 8

    # v2 group assignment = 1 rand
    # group selection = 1 rand
    _seed(0)
    rands = _random(2)
    if rands[0] < 0.5:
        groups = np.array([0, 1, 1], np.uint32)
        ref_ends = np.array([5, 6, 0], np.uint32)
    else:
        groups = np.array([0, 1, 0], np.uint32)
        ref_ends = np.array([6, 4, 0], np.uint32)
    n_groups = 2

    cum_prop = np.empty(n_groups, np.float32)
    for g in range(n_groups):
        v_starts = positions[variant_idxs[groups == g]]
        v_ends = v_starts - np.minimum(0, sizes[variant_idxs[groups == g]]) + 1
        ref_lengths = np.minimum(v_ends, ref_ends[g]) - np.maximum(v_starts, ref_start)
        cum_prop[g] = (dosages[groups == g] / ref_lengths).sum()
    cum_prop = cum_prop.cumsum()
    cum_prop /= cum_prop[-1]
    keep_group = (rands[1] <= cum_prop).sum() - 1
    desired = groups == keep_group

    return variant_idxs, dosages, positions, sizes, ref_start, target_len, desired


def case_del_span_start():
    # group 0
    # 0 1 2 3 4 ...
    # . . - r s ...

    variant_idxs = np.array([0, 1, 2], np.int32)
    dosages = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([0, 1, 4], np.int32)
    sizes = np.array([-2, 0, 0], np.int32)
    ref_start = 2
    target_len = 10

    desired = np.array([True, False, True], np.bool_)

    return variant_idxs, dosages, positions, sizes, ref_start, target_len, desired


@parametrize_with_cases(
    "variant_idxs, dosages, positions, sizes, ref_start, target_len, desired", cases="."
)
def test_mark_keep_variants(
    variant_idxs: NDArray[np.int32],
    dosages: NDArray[np.float32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    ref_start: int,
    target_len: int,
    desired: NDArray[np.bool_],
):
    n_variants = len(variant_idxs)

    groups = np.empty(n_variants, np.uint32)
    ends = np.empty(n_variants, np.uint32)
    write_lens = np.empty(n_variants, np.uint32)

    _seed(0)
    actual = _mark_keep_variants(
        variant_idxs,
        dosages,
        positions,
        sizes,
        ref_start,
        groups,
        ends,
        write_lens,
        target_len,
    )

    np.testing.assert_array_equal(actual, desired)


@nb.njit
def _seed(seed: int):
    random.seed(seed)


@nb.njit
def _random(n_values: int):
    return [random.random() for _ in range(n_values)]
