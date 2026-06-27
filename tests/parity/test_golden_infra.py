# tests/parity/test_golden_infra.py
"""Self-tests for the golden snapshot/replay infrastructure."""
from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from tests.parity import _golden


def test_collect_examples_deterministic():
    s = st.integers(0, 1_000_000)
    a = _golden.collect_examples(s, 20)
    b = _golden.collect_examples(s, 20)
    assert a == b
    assert len(a) == 20


def test_save_load_roundtrip_mixed(tmp_path, monkeypatch):
    monkeypatch.setattr(_golden, "GOLDEN_DIR", tmp_path)
    cases = [
        ((np.arange(3, dtype=np.int32), None, 5), np.arange(3, dtype=np.int32) * 2),
        ((np.zeros(0, np.uint8),), np.zeros(0, np.uint8)),
    ]
    _golden.save_golden("demo", cases)
    back = _golden.load_golden("demo")
    assert len(back) == 2
    np.testing.assert_array_equal(back[0][0][0], cases[0][0][0])
    assert back[0][0][1] is None
    assert back[0][0][2] == 5


def test_rust_kernels_table_callable():
    # Every registered name resolves to a real callable imported directly.
    assert _golden.RUST_KERNELS, "RUST_KERNELS is empty"
    for name, fn in _golden.RUST_KERNELS.items():
        assert callable(fn), f"{name} -> {fn!r} not callable"
