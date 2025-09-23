import numpy as np
import pytest
from genvarloader._dataset._indexing import DatasetIndexer
from genvarloader._utils import idx_like_to_array
from pytest_cases import fixture, parametrize_with_cases


@fixture
def dsi():
    n_samples = 2
    r_idx = np.array([1, 2, 0])
    s_idx = np.arange(n_samples)
    samples = ["Aang", "Katara"]
    return DatasetIndexer.from_region_and_sample_idxs(r_idx, s_idx, samples)


def case_1_region_all_samples():
    r_idx = 0
    s_idx = None
    desired_r_idx = np.array([1])
    desired_s_idx = np.array([0, 1])
    return r_idx, s_idx, desired_r_idx, desired_s_idx


def case_1_region_1_sample():
    r_idx = 0
    s_idx = 1
    desired_r_idx = np.array([1])
    desired_s_idx = np.array([1])
    return r_idx, s_idx, desired_r_idx, desired_s_idx


def case_all_regions_1_sample():
    r_idx = None
    s_idx = 1
    desired_r_idx = np.array([1, 2, 0])
    desired_s_idx = np.array([1])
    return r_idx, s_idx, desired_r_idx, desired_s_idx


def case_2_regions_1_sample():
    r_idx = [0, 2]
    s_idx = 1
    desired_r_idx = np.array([1, 0])
    desired_s_idx = np.array([1])
    return r_idx, s_idx, desired_r_idx, desired_s_idx


def case_2_regions_all_samples():
    r_idx = [0, 2]
    s_idx = None
    desired_r_idx = np.array([1, 0])
    desired_s_idx = np.array([0, 1])
    return r_idx, s_idx, desired_r_idx, desired_s_idx


@parametrize_with_cases("r_idx, s_idx, desired_r_idx, desired_s_idx", cases=".")
def test_subset(dsi: DatasetIndexer, r_idx, s_idx, desired_r_idx, desired_s_idx):
    subset = dsi.subset_to(r_idx, s_idx)
    if r_idx is not None:
        r_idx = idx_like_to_array(r_idx, dsi.n_regions)
    else:
        r_idx = np.arange(dsi.n_regions, dtype=np.intp)
    if s_idx is not None:
        s_idx = idx_like_to_array(s_idx, dsi.n_samples)
    else:
        s_idx = np.arange(dsi.n_samples, dtype=np.intp)
    assert subset.n_regions == len(r_idx)
    assert subset.n_samples == len(s_idx)
    np.testing.assert_equal(subset._r_idx, desired_r_idx)
    np.testing.assert_equal(subset._s_idx, desired_s_idx)


def test_subset_to_full(dsi: DatasetIndexer):
    subset = dsi.subset_to([0, 2], [0])
    full = subset.to_full_dataset()
    assert full.n_regions == dsi.n_regions
    assert full.n_samples == dsi.n_samples
    np.testing.assert_equal(full._r_idx, dsi._r_idx)
    np.testing.assert_equal(full._s_idx, dsi._s_idx)


# full dataset indices
# [
#   [2, 3]
#   [4, 5]
#   [0, 1]
# ]


def getitem_squeeze():
    regions = 0
    samples = "Aang"
    s_idx = 0

    des_idx = np.array([2])
    des_squeeze = True
    des_reshape = None
    return regions, samples, s_idx, des_idx, des_squeeze, des_reshape


def getitem_1d():
    regions = 0
    samples = ["Aang", "Katara"]
    s_idx = [0, 1]

    des_idx = np.array([2, 3])
    des_squeeze = False
    des_reshape = (1, 2)
    return regions, samples, s_idx, des_idx, des_squeeze, des_reshape


def getitem_slice_scalar():
    regions = slice(2)
    samples = "Aang"
    s_idx = 0

    des_idx = np.array([2, 4])
    des_squeeze = False
    des_reshape = None
    return regions, samples, s_idx, des_idx, des_squeeze, des_reshape


def getitem_slice_slice():
    regions = slice(2)
    samples = slice(1)
    s_idx = [0]

    des_idx = np.array([2, 4])
    des_squeeze = False
    des_reshape = (2, 1)
    return regions, samples, s_idx, des_idx, des_squeeze, des_reshape


def getitem_reshape():
    regions = 0
    samples = [["Aang", "Katara"], ["Katara", "Aang"]]
    s_idx = [[0, 1], [1, 0]]

    des_idx = np.array([[2, 3], [3, 2]]).ravel()
    des_squeeze = False
    des_reshape = (2, 2)
    return regions, samples, s_idx, des_idx, des_squeeze, des_reshape


@pytest.mark.xfail(strict=True, raises=KeyError, reason="Zuko is not in the dataset")
def getitem_missing():
    regions = 0
    samples = ["Aang", "Zuko"]
    s_idx = [0, -1]

    des_idx = None
    des_squeeze = None
    des_reshape = None
    return regions, samples, s_idx, des_idx, des_squeeze, des_reshape


@parametrize_with_cases(
    "regions, samples, s_idx, desired_idx, desired_squeeze, desired_reshape",
    cases=".",
    prefix="getitem_",
)
def test_parse_idx(
    dsi: DatasetIndexer,
    regions,
    samples,
    s_idx,
    desired_idx,
    desired_squeeze,
    desired_reshape,
):
    idx, squeeze, reshape = dsi.parse_idx((regions, samples))

    np.testing.assert_equal(idx, desired_idx)
    assert squeeze == desired_squeeze
    assert reshape == desired_reshape

    idx, squeeze, reshape = dsi.parse_idx((regions, s_idx))

    np.testing.assert_equal(idx, desired_idx)
    assert squeeze == desired_squeeze
    assert reshape == desired_reshape

def test_parse_idx_subset(dsi: DatasetIndexer, regions, samples, s_idx, desired_idx, desired_squeeze, desired_reshape):
    subset = dsi.subset_to(regions, samples)
    idx, squeeze, reshape = subset.parse_idx((regions, s_idx))

    np.testing.assert_equal(idx, desired_idx)
    assert squeeze == desired_squeeze
    assert reshape == desired_reshape