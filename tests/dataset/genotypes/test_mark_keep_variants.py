import random

import numba as nb
import numpy as np
from genvarloader._dataset._genotypes import _choose_unphased_variants
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases


def case_snps():
    variant_idxs = np.array([0, 1, 2], np.int32)
    ccfs = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 2, 3], np.int32)
    sizes = np.array([0, 0, 0], np.int32)
    query_start = 0
    query_end = 4

    desired = np.ones(3, np.bool_)

    return variant_idxs, ccfs, positions, sizes, query_start, query_end, desired


def case_extra_snps():
    variant_idxs = np.array([0, 1, 2], np.int32)
    ccfs = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 2, 3], np.int32)
    sizes = np.array([0, 0, 0], np.int32)
    query_start = 0
    query_end = 2

    desired = np.array([True, False, False], np.bool_)

    return variant_idxs, ccfs, positions, sizes, query_start, query_end, desired


def case_indels_0():
    # group 0
    # 0 1 2 3 4  5
    # r d - - - r/s

    # group 1
    # 0 1 2 3  4  5
    # r r r ii r r/s

    variant_idxs = np.array([0, 1, 2], np.int32)
    ccfs = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([1, 3, 5], np.int32)
    sizes = np.array([-3, 1, 0], np.int32)
    query_start = 0
    query_end = 8

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
        ref_lengths = np.minimum(v_ends, ref_ends[g]) - np.maximum(
            v_starts, query_start
        )
        cum_prop[g] = (ccfs[groups == g] / ref_lengths).sum()
    cum_prop = cum_prop.cumsum()
    cum_prop /= cum_prop[-1]
    keep_group = (rands[1] <= cum_prop).sum() - 1
    desired = groups == keep_group

    return variant_idxs, ccfs, positions, sizes, query_start, query_end, desired


def case_del_span_start():
    # group 0
    # 0 1 2 3 4 ...
    # . . - r s ...

    variant_idxs = np.array([0, 1, 2], np.int32)
    ccfs = np.array([0.3, 0.2, 0.1], np.float32)
    positions = np.array([0, 1, 4], np.int32)
    sizes = np.array([-2, 0, 0], np.int32)
    query_start = 2
    query_end = 12

    desired = np.array([True, False, True], np.bool_)

    return variant_idxs, ccfs, positions, sizes, query_start, query_end, desired


@parametrize_with_cases(
    "variant_idxs, ccfs, positions, sizes, query_start, query_end, desired",
    cases=".",
)
def test_mark_keep_variants(
    variant_idxs: NDArray[np.int32],
    ccfs: NDArray[np.float32],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    query_start: int,
    query_end: int,
    desired: NDArray[np.bool_],
):
    n_variants = len(variant_idxs)

    groups = np.empty(n_variants, np.uint32)
    ends = np.empty(n_variants, np.uint32)
    write_lens = np.empty(n_variants, np.uint32)

    _seed(0)
    actual = _choose_unphased_variants(
        variant_idxs=variant_idxs,
        ccfs=ccfs,
        positions=positions,
        sizes=sizes,
        query_start=query_start,
        groups=groups,
        ref_ends=ends,
        write_lens=write_lens,
        query_end=query_end,
        deterministic=False,
    )

    np.testing.assert_array_equal(actual, desired)


@nb.njit
def _seed(seed: int):
    random.seed(seed)


@nb.njit
def _random(n_values: int):
    return [random.random() for _ in range(n_values)]
