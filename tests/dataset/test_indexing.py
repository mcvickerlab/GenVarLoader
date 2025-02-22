import numpy as np
from genvarloader._dataset._indexing import DatasetIndexer
from genvarloader._utils import idx_like_to_array
from pytest_cases import fixture, parametrize_with_cases


@fixture
def dsi():
    n_samples = 2
    r_idx = np.array([1, 2, 0])
    s_idx = np.arange(n_samples)
    return DatasetIndexer.from_region_and_sample_idxs(r_idx, s_idx)


def case_1_region_all_samples():
    r_idx = 0
    s_idx = None
    expected_idx_map = np.array([2, 3])
    return r_idx, s_idx, expected_idx_map


def case_1_region_1_sample():
    r_idx = 0
    s_idx = 1
    expected_idx_map = np.array([3])
    return r_idx, s_idx, expected_idx_map


def case_all_regions_1_sample():
    r_idx = None
    s_idx = 1
    expected_idx_map = np.array([3, 5, 1])
    return r_idx, s_idx, expected_idx_map


def case_2_regions_1_sample():
    r_idx = [0, 2]
    s_idx = 1
    expected_idx_map = np.array([3, 1])
    return r_idx, s_idx, expected_idx_map


def case_2_regions_all_samples():
    r_idx = [0, 2]
    s_idx = None
    expected_idx_map = np.array([2, 3, 0, 1])
    return r_idx, s_idx, expected_idx_map


@parametrize_with_cases("r_idx, s_idx, expected_idx_map", cases=".")
def test_subset(dsi: DatasetIndexer, r_idx, s_idx, expected_idx_map):
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
    np.testing.assert_equal(subset.i2d_map, expected_idx_map)


def test_subset_to_full(dsi: DatasetIndexer):
    subset = dsi.subset_to([0, 2], [0])
    full = subset.to_full_dataset()
    assert full.n_regions == dsi.n_regions
    assert full.n_samples == dsi.n_samples
    np.testing.assert_equal(full.i2d_map, dsi.i2d_map)
