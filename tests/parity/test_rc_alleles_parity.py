"""rc_alleles: rust vs frozen golden (oracle frozen Phase 5 W5).

The hypothesis-driven numba-comparison test has been replaced with frozen-golden
replay.  The dispatch-call-count smoke test is preserved using make_kernel_spy
(which keeps _dispatch usage inside _golden.py, not here).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_flat_alleles_reverse_masked_uses_rc_alleles():
    """_FlatAlleles.reverse_masked must call the dispatched rc_alleles kernel."""
    from genvarloader._dataset._flat_variants import _FlatAlleles

    spy, calls, restore = _golden.make_kernel_spy("rc_alleles")
    try:
        # one row (b=1, ploidy=1), two alleles "AC","G".
        byte_data = np.frombuffer(b"ACG", np.uint8).copy()
        seq_offsets = np.array([0, 2, 3], np.int64)
        var_offsets = np.array([0, 2], np.int64)
        fa = _FlatAlleles(byte_data, seq_offsets, var_offsets, (1, 1, None))
        fa.reverse_masked(np.array([True], np.bool_))
        assert calls["n"] == 1
        # "AC"->"GT", "G"->"C"
        assert fa.byte_data.tobytes() == b"GTC"
    finally:
        restore()


def test_rc_alleles_golden():
    """Rust rc_alleles must equal the frozen golden (cross-checked vs numba at freeze time)."""
    cases = _golden.load_golden("rc_alleles")
    assert cases, "empty golden"
    rust_fn = _golden.RUST_KERNELS["rc_alleles"]
    for ci, (inputs, golden) in enumerate(cases):
        init_data, seq_offsets, var_offsets, mask = inputs
        buf = np.ascontiguousarray(init_data, np.uint8)
        rust_fn(buf, seq_offsets, var_offsets, mask)
        np.testing.assert_array_equal(
            buf, golden, err_msg=f"rc_alleles case {ci} mismatch"
        )
