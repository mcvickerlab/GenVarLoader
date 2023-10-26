import numpy as np
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases

from genvarloader.bnfo.pgen import group_duplicates


def groups_no_dups():
    positions = np.array([1, 2, 3], dtype=np.int32)
    groups = np.array([0, 1, 2, 3], dtype=np.int32)
    return positions, groups


def groups_one_group():
    positions = np.array([1, 1, 1], dtype=np.int32)
    groups = np.array([0, 2], dtype=np.int32)
    return positions, groups


def groups_unsorted():
    positions = np.array([3, 3, 2, 2, 1], dtype=np.int32)
    groups = np.array([0, 2, 4, 5], dtype=np.int32)
    return positions, groups


def groups_interspersed_dups():
    positions = np.array([1, 3, 3, 4, 5, 5], dtype=np.int32)
    groups = np.array([0, 1, 3, 4, 6], dtype=np.int32)
    return positions, groups


def groups_consecutive_dups():
    positions = np.array([1, 1, 2, 2], dtype=np.int32)
    groups = np.array([0, 2, 4], dtype=np.int32)
    return positions, groups


@parametrize_with_cases(["positions", "groups"], cases=".", prefix="groups_")
def test_group_duplicates(positions: NDArray[np.int32], groups: NDArray[np.int32]):
    test_groups = group_duplicates(positions)
    assert groups == test_groups


def test_g2a_biallelic():
    raise NotImplementedError


def test_g2a_half_call_split_multiallelic():
    raise NotImplementedError


def test_g2a_no_half_call_split_multiallelic():
    raise NotImplementedError
