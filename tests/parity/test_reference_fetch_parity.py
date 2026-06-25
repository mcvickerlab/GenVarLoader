"""Parity backstop for Reference.fetch (rerouted through dispatched get_reference).

fetch builds regions=(contig_idx, start, end) and out_offsets, then calls the
same get_reference core used by the main reference read path. This test flips
GVL_BACKEND and asserts byte-identical fetched sequence across backends, with a
spy proving the rust get_reference kernel is actually invoked.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader._dispatch as _dispatch

pytestmark = pytest.mark.parity


def test_reference_fetch_parity(reference, monkeypatch):
    ref = reference
    contigs = ref.contigs[:1]
    starts = np.array([0], dtype=np.int64)
    ends = np.array([50], dtype=np.int64)

    numba_fn, rust_fn = _dispatch.backends("get_reference")
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return rust_fn(*a, **k)

    orig = dict(_dispatch._REGISTRY["get_reference"])
    _dispatch.register("get_reference", numba=numba_fn, rust=_spy, default="numba")
    try:
        monkeypatch.setenv("GVL_BACKEND", "rust")
        out_rust = ref.fetch(contigs, starts, ends)
        rust_calls = calls["n"]
        monkeypatch.setenv("GVL_BACKEND", "numba")
        out_numba = ref.fetch(contigs, starts, ends)
        assert calls["n"] == rust_calls, "rust spy fired during numba read"
    finally:
        _dispatch._REGISTRY["get_reference"] = orig

    assert rust_calls > 0, "rust get_reference never invoked via fetch — vacuous"
    np.testing.assert_array_equal(
        np.asarray(out_numba.data), np.asarray(out_rust.data)
    )
    np.testing.assert_array_equal(
        np.asarray(out_numba.offsets, np.int64),
        np.asarray(out_rust.offsets, np.int64),
    )
