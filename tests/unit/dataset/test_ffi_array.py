"""_ffi_array boundary guard (Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from genvarloader._dataset._utils import _ffi_array


def test_passes_contiguous_correct_dtype():
    arr = np.arange(10, dtype=np.int32)
    out = _ffi_array(arr, np.int32, "geno_v_idxs")
    assert out is arr  # zero-copy: same object


def test_raises_on_non_contiguous():
    base = np.zeros((10, 3), dtype=np.int32)
    strided = base[:, 1]  # non-contiguous column view
    assert not strided.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="geno_v_idxs"):
        _ffi_array(strided, np.int32, "geno_v_idxs")


def test_raises_on_wrong_dtype():
    arr = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="itv_starts"):
        _ffi_array(arr, np.int32, "itv_starts")
