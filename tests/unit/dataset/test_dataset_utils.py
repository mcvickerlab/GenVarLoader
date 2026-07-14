"""Unit tests for public helpers in genvarloader._dataset._utils."""

from __future__ import annotations

import numpy as np
from genoray._contigs import ContigNormalizer
from genvarloader._dataset._utils import (
    bed_to_regions,
    oidx_to_raveled_idx,
    padded_slice,
    reduceat_offsets,
    regions_to_bed,
)


def test_oidx_to_raveled_idx_cartesian_product():
    # 3x4 array; pick rows [0, 2] and cols [1, 3] -> linear indices for the
    # full cartesian product. For row-major shape (3, 4):
    # (0, 1) -> 1, (0, 3) -> 3, (2, 1) -> 9, (2, 3) -> 11
    rows = [0, 2]
    cols = [1, 3]
    out = oidx_to_raveled_idx(rows, cols, (3, 4))
    np.testing.assert_array_equal(out, np.array([1, 3, 9, 11]))


def test_oidx_to_raveled_idx_matches_manual_ravel():
    rng = np.random.default_rng(0)
    shape = (5, 7)
    rows = rng.integers(0, shape[0], size=4)
    cols = rng.integers(0, shape[1], size=3)
    out = oidx_to_raveled_idx(rows, cols, shape)
    # Expected: for each row r in rows, for each col c in cols -> r*ncols + c
    expected = np.array([r * shape[1] + c for r in rows for c in cols])
    np.testing.assert_array_equal(out, expected)


def test_reduceat_offsets_basic_sum():
    # arr length 6, offsets define 3 groups: [0:2], [2:5], [5:6]
    arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    offsets = np.array([0, 2, 5, 6], dtype=np.int64)
    out = reduceat_offsets(np.add, arr, offsets, axis=0)
    # Groups sum to: 1+2=3, 3+4+5=12, 6=6
    np.testing.assert_array_equal(out, np.array([3, 12, 6]))


def test_reduceat_offsets_with_empty_trailing_groups():
    # When offsets has duplicated final value, trailing reductions should be
    # filled with the ufunc identity (0 for np.add).
    arr = np.array([1, 2, 3, 4], dtype=np.int64)
    # 4 reductions; first 2 use data, last 2 are "no-var" -> identity
    offsets = np.array([0, 2, 4, 4, 4], dtype=np.int64)
    out = reduceat_offsets(np.add, arr, offsets, axis=0)
    # First: 1+2=3, second: 3+4=7, then identity (0), identity (0)
    np.testing.assert_array_equal(out, np.array([3, 7, 0, 0]))


def test_reduceat_offsets_2d_negative_axis():
    # Reduce along axis=-1 (columns) of a 2D array.
    arr = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.int64)
    offsets = np.array([0, 2, 4], dtype=np.int64)
    out = reduceat_offsets(np.add, arr, offsets, axis=-1)
    # For each row: sum cols[0:2], sum cols[2:4]
    # The function does swapaxes(axis, -1) at the end. Since we already reduce
    # along the last axis, swapaxes(-1, -1) is a no-op, so shape stays (2, 2).
    expected = np.array([[1 + 2, 3 + 4], [10 + 20, 30 + 40]])
    np.testing.assert_array_equal(out, expected)


# Light coverage of the other public helpers so the file stands on its own.


def test_padded_slice_left_and_right_pad():
    arr = np.array([1, 2, 3], dtype=np.int32)
    out = np.zeros(7, dtype=np.int32)
    res = padded_slice(arr, -2, 5, pad_val=-1, out=out)
    # start=-2 -> pad_left=2, stop=5 -> pad_right=2, middle = arr
    np.testing.assert_array_equal(res, np.array([-1, -1, 1, 2, 3, -1, -1]))


def test_regions_to_bed_and_back_roundtrip():
    regions = np.array(
        [[0, 100, 200, 1], [1, 50, 150, -1]],
        dtype=np.int32,
    )
    contigs = ["chr1", "chr2"]
    bed = regions_to_bed(regions, contigs)
    assert bed.columns == ["chrom", "chromStart", "chromEnd", "strand"]
    assert bed["chrom"].to_list() == ["chr1", "chr2"]
    assert bed["strand"].to_list() == ["+", "-"]

    cn = ContigNormalizer(contigs)
    back = bed_to_regions(bed, cn)
    np.testing.assert_array_equal(back, regions)
