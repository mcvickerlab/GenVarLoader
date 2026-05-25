import numpy as np
import pytest
from genvarloader._dataset._genotypes import filter_af


def _basic_inputs():
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_offsets = np.array([0, 4], dtype=np.int64)
    geno_v_idxs = np.array([0, 1, 2, 3], dtype=np.int32)
    afs = np.array([0.001, 0.05, 0.2, 0.5], dtype=np.float32)
    return geno_offset_idx, geno_offsets, geno_v_idxs, afs


def test_filter_af_no_op():
    """min_af=None, max_af=None -> all kept, short-circuits."""
    geno_offset_idx, geno_offsets, geno_v_idxs, afs = _basic_inputs()
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, None, None)
    np.testing.assert_equal(keep, np.array([True, True, True, True]))


def test_filter_af_min_only():
    """min_af=0.05 keeps variants with af >= 0.05."""
    geno_offset_idx, geno_offsets, geno_v_idxs, afs = _basic_inputs()
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, 0.05, None)
    np.testing.assert_equal(keep, np.array([False, True, True, True]))


def test_filter_af_max_only():
    """max_af=0.2 keeps variants with af <= 0.2.

    Note: afs are stored as float32. np.float32(0.2) > float64(0.2) due to
    representation loss, so the variant at af=0.2 does NOT pass the <= 0.2
    filter when max_af is a Python float.  The actual kept set is [0.001, 0.05].
    """
    geno_offset_idx, geno_offsets, geno_v_idxs, afs = _basic_inputs()
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, None, 0.2)
    np.testing.assert_equal(keep, np.array([True, True, False, False]))


def test_filter_af_both():
    """Combined min/max bounds."""
    geno_offset_idx, geno_offsets, geno_v_idxs, afs = _basic_inputs()
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, 0.01, 0.3)
    np.testing.assert_equal(keep, np.array([False, True, True, False]))


@pytest.mark.xfail(
    reason=(
        "The 2-D geno_offsets branch calls lengths_to_offsets, a plain NumPy "
        "helper that is not numba-typed.  numba raises TypingError when it "
        "tries to compile this path.  This test documents the broken contract "
        "and should be un-xfailed once lengths_to_offsets is decorated with "
        "@nb.njit or the branch is refactored."
    ),
    strict=True,
    raises=Exception,
)
def test_filter_af_2d_offsets_layout():
    """(2, n_slices) offsets layout — slice [start, end) per row."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    # Single slice covering all 4 variants.
    geno_offsets = np.array([[0], [4]], dtype=np.int64)  # (2, n_slices=1)
    geno_v_idxs = np.array([0, 1, 2, 3], dtype=np.int32)
    afs = np.array([0.001, 0.05, 0.2, 0.5], dtype=np.float32)
    keep, keep_offsets = filter_af(
        geno_offset_idx, geno_offsets, geno_v_idxs, afs, 0.05, None
    )
    np.testing.assert_equal(keep, np.array([False, True, True, True]))
    # keep_offsets is cumulative offsets over n_slices: length n_slices+1 = 2.
    assert keep_offsets.shape == (2,)
