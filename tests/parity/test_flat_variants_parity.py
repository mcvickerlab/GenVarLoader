"""flat_variants kernels: rust vs frozen golden (oracle frozen Phase 5 W5)."""

from __future__ import annotations

import numpy as np
import pytest

from genvarloader._dataset._flat_variants import (
    _compact_keep,
    _fill_empty_fixed,
    _fill_empty_scalar,
    _fill_empty_seq,
    _gather_rows,
)
from tests.parity import _golden

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Golden replay tests (one per golden name)
# ---------------------------------------------------------------------------


def test_gather_rows_i32_golden():
    cases = _golden.load_golden("gather_rows_i32")
    assert cases, "empty golden"
    _golden.replay_tuple("gather_rows_i32", cases)


def test_gather_rows_f32_golden():
    cases = _golden.load_golden("gather_rows_f32")
    assert cases, "empty golden"
    _golden.replay_tuple("gather_rows_f32", cases)


def test_gather_alleles_golden():
    cases = _golden.load_golden("gather_alleles")
    assert cases, "empty golden"
    _golden.replay_tuple("gather_alleles", cases)


def test_compact_keep_i32_golden():
    cases = _golden.load_golden("compact_keep_i32")
    assert cases, "empty golden"
    _golden.replay_tuple("compact_keep_i32", cases)


def test_compact_keep_f32_golden():
    cases = _golden.load_golden("compact_keep_f32")
    assert cases, "empty golden"
    _golden.replay_tuple("compact_keep_f32", cases)


def test_fill_empty_scalar_i32_golden():
    cases = _golden.load_golden("fill_empty_scalar_i32")
    assert cases, "empty golden"
    _golden.replay_tuple("fill_empty_scalar_i32", cases)


def test_fill_empty_scalar_f32_golden():
    cases = _golden.load_golden("fill_empty_scalar_f32")
    assert cases, "empty golden"
    _golden.replay_tuple("fill_empty_scalar_f32", cases)


def test_fill_empty_fixed_i32_golden():
    cases = _golden.load_golden("fill_empty_fixed_i32")
    assert cases, "empty golden"
    _golden.replay_tuple("fill_empty_fixed_i32", cases)


def test_fill_empty_fixed_f32_golden():
    cases = _golden.load_golden("fill_empty_fixed_f32")
    assert cases, "empty golden"
    _golden.replay_tuple("fill_empty_fixed_f32", cases)


def test_fill_empty_seq_u8_golden():
    cases = _golden.load_golden("fill_empty_seq_u8")
    assert cases, "empty golden"
    _golden.replay_tuple("fill_empty_seq_u8", cases)


def test_fill_empty_seq_i32_golden():
    cases = _golden.load_golden("fill_empty_seq_i32")
    assert cases, "empty golden"
    _golden.replay_tuple("fill_empty_seq_i32", cases)


# ---------------------------------------------------------------------------
# Dtype regression tests (no hypothesis, no dispatch)
# ---------------------------------------------------------------------------


def test_gather_rows_dtype_regression():
    """_gather_rows must preserve dtype and values — no silent down-cast."""
    # float32 case: the original corruption (0.25 -> 0 as int32)
    goi = np.array([0], np.intp)
    offsets = np.array([0, 2], np.int64)
    data_f32 = np.array([0.25, 0.75], np.float32)
    out_f32, off_f32 = _gather_rows(goi, offsets, data_f32)
    assert out_f32.dtype == np.float32, f"Expected float32, got {out_f32.dtype}"
    np.testing.assert_array_equal(out_f32, np.array([0.25, 0.75], np.float32))
    assert off_f32.tolist() == [0, 2]

    # int64 case: arbitrary "other" dtype must not be coerced to int32
    data_i64 = np.array([100_000_000, 200_000_000], np.int64)
    out_i64, off_i64 = _gather_rows(goi, offsets, data_i64)
    assert out_i64.dtype == np.int64, f"Expected int64, got {out_i64.dtype}"
    np.testing.assert_array_equal(out_i64, data_i64)
    assert off_i64.tolist() == [0, 2]


def test_compact_keep_dtype_regression():
    """_compact_keep must preserve dtype without down-casting.

    The i32/f32 Rust cores handle those two dtypes. All other dtypes (e.g.
    int16, int64 for custom FORMAT fields, issue #231) must round-trip via the
    numba fallback with the exact same dtype and values.
    """
    row_offsets = np.array([0, 2, 3], np.int64)
    keep = np.array([True, False, True], np.bool_)

    # int16: should NOT be widened to int32
    vals_i16 = np.array([10, 20, 30], np.int16)
    out_i16, off_i16 = _compact_keep(vals_i16, row_offsets, keep)
    assert out_i16.dtype == np.int16, f"Expected int16, got {out_i16.dtype}"
    np.testing.assert_array_equal(out_i16, np.array([10, 30], np.int16))
    assert off_i16.tolist() == [0, 1, 2]

    # int64: should NOT be narrowed to int32
    vals_i64 = np.array([100_000_000_000, 200_000_000_000, 300_000_000_000], np.int64)
    out_i64, off_i64 = _compact_keep(vals_i64, row_offsets, keep)
    assert out_i64.dtype == np.int64, f"Expected int64, got {out_i64.dtype}"
    np.testing.assert_array_equal(
        out_i64, np.array([100_000_000_000, 300_000_000_000], np.int64)
    )
    assert off_i64.tolist() == [0, 1, 2]


def test_fill_empty_scalar_dtype_regression():
    """_fill_empty_scalar must preserve dtype — no down-cast for non-i32/f32.

    int16 is a representative custom FORMAT field dtype (issue #231).
    The empty row's fill slot must carry the int16 fill value exactly.
    """
    # offsets: 3 rows with middle row empty → [0, 2, 2, 3]
    data = np.array([10, 20, 30], np.int16)
    offsets = np.array([0, 2, 2, 3], np.int64)
    fill = np.int16(99)
    out, new_off = _fill_empty_scalar(data, offsets, fill)
    assert out.dtype == np.int16, f"Expected int16, got {out.dtype}"
    np.testing.assert_array_equal(out, np.array([10, 20, 99, 30], np.int16))
    assert new_off.tolist() == [0, 2, 3, 4]


def test_fill_empty_fixed_dtype_regression():
    """_fill_empty_fixed must preserve dtype — no down-cast for non-i32/f32.

    int16 is representative of custom FORMAT flank tokens (issue #231).
    The empty row's `inner` fill slots must carry the int16 fill value exactly.
    """
    # 2 rows: offsets [0,1,1], inner=2 — second row empty.
    data = np.array([7, 8], np.int16)  # 1 var * 2 inner
    offsets = np.array([0, 1, 1], np.int64)
    fill = np.int16(42)
    out, new_off = _fill_empty_fixed(data, offsets, 2, fill)
    assert out.dtype == np.int16, f"Expected int16, got {out.dtype}"
    np.testing.assert_array_equal(out, np.array([7, 8, 42, 42], np.int16))
    assert new_off.tolist() == [0, 1, 2]


def test_fill_empty_seq_dtype_regression():
    """_fill_empty_seq must preserve dtype for int32 token windows.

    A single uint8-only Rust core would silently corrupt int32 token values
    (e.g. token 999 → 0xE7 = 231 when truncated to uint8).
    This test verifies that int32 token windows round-trip exactly through
    the dispatch wrapper, including the dummy token in the empty slot.
    """
    # 2 rows: var_offsets [0,0,2] — row 0 is empty.
    # Row 1: 2 variants with tokens [100, 200] and [300].
    # seq_offsets: [0,2,3].
    # dummy int32 token = 999 (> 255 — would be corrupted if truncated to uint8).
    data = np.array([100, 200, 300], np.int32)
    var_offsets = np.array([0, 0, 2], np.int64)
    seq_offsets = np.array([0, 2, 3], np.int64)
    dummy = np.array([999], np.int32)

    nd, nvar, nseq = _fill_empty_seq(data, var_offsets, seq_offsets, dummy)

    assert nd.dtype == np.int32, f"Expected int32, got {nd.dtype}"
    # new_var: row 0 empty→1 dummy, row 1 has 2 vars → [0, 1, 3]
    assert nvar.tolist() == [0, 1, 3], f"new_var wrong: {nvar.tolist()}"
    # new_seq: dummy len=1, var0 len=2, var1 len=1 → [0, 1, 3, 4]
    assert nseq.tolist() == [0, 1, 3, 4], f"new_seq wrong: {nseq.tolist()}"
    # new_data: [999] (dummy), [100,200] (var0 tokens), [300] (var1 tokens)
    np.testing.assert_array_equal(nd, np.array([999, 100, 200, 300], np.int32))
