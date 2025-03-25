import numpy as np
from genvarloader._dataset._utils import splits_sum_le_value
from genvarloader._utils import _normalize_contig_name
from pytest_cases import parametrize_with_cases


def test_splits_sum_le_value():
    max_size = 10
    sizes = np.array([3, 5, 2, 4, 7, 5, 2], np.int32)
    splits = splits_sum_le_value(sizes, max_size)
    np.testing.assert_equal(splits, np.array([0, 3, 4, 5, 7], np.intp))
    np.testing.assert_array_less(np.add.reduceat(sizes, splits[:-1]), max_size + 1)


def contig_match():
    unnormed = "chr1"
    source = ["chr1", "chr2"]
    desired = "chr1"
    return unnormed, source, desired


def contig_add_match():
    unnormed = "1"
    source = ["chr1", "chr2"]
    desired = "chr1"
    return unnormed, source, desired


def contig_strip_match():
    unnormed = "chr1"
    source = ["1", "2"]
    desired = "1"
    return unnormed, source, desired


def contig_no_match():
    unnormed = "chr3"
    source = ["chr1", "chr2"]
    desired = None
    return unnormed, source, desired


def contig_list():
    unnormed = ["chr1", "1", "chr3"]
    source = ["chr1", "chr2"]
    desired = ["chr1", "chr1", None]
    return unnormed, source, desired


@parametrize_with_cases("unnormed, source, desired", cases=".", prefix="contig_")
def test_normalize_contig_name(unnormed, source, desired):
    assert _normalize_contig_name(unnormed, source) == desired
