import numpy as np
import pytest
from hypothesis import given, settings

from genvarloader._dataset import _flat_variants  # noqa: F401  (triggers register())
from genvarloader._dataset._flat_variants import _gather_rows
from genvarloader._dataset._genotypes import _as_starts_stops
from tests.parity._harness import assert_kernel_parity_tuple
from tests.parity.strategies import gather_alleles_inputs, gather_rows_inputs

pytestmark = pytest.mark.parity


@settings(deadline=None)
@given(gather_rows_inputs(dtype=np.int32))
def test_gather_rows_parity(inputs):
    goi, offsets, data = inputs
    assert_kernel_parity_tuple(
        "gather_rows_i32",
        np.ascontiguousarray(goi, np.int64),
        _as_starts_stops(offsets),
        np.ascontiguousarray(data, np.int32),
    )


@settings(deadline=None)
@given(gather_rows_inputs(dtype=np.float32))
def test_gather_rows_f32_parity(inputs):
    goi, offsets, data = inputs
    assert_kernel_parity_tuple(
        "gather_rows_f32",
        np.ascontiguousarray(goi, np.int64),
        _as_starts_stops(offsets),
        np.ascontiguousarray(data, np.float32),
    )


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


@settings(deadline=None)
@given(gather_alleles_inputs())
def test_gather_alleles_parity(inputs):
    v_idxs, allele_bytes, allele_offsets = inputs
    assert_kernel_parity_tuple(
        "gather_alleles",
        np.ascontiguousarray(v_idxs, np.int32),
        np.ascontiguousarray(allele_bytes, np.uint8),
        np.ascontiguousarray(allele_offsets, np.int64),
    )
