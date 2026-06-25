"""Parity tests for shift_and_realign_tracks_sparse (batch kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings

from genvarloader._dataset import _tracks  # noqa: F401 — triggers register()
from tests.parity.strategies import shift_and_realign_tracks_inputs

pytestmark = pytest.mark.parity


def _assert_parity(total_out: int, inputs: tuple) -> None:
    """Check that the out buffer is byte-identical between numba and Rust.

    The numba parallel=True batch driver has a known SystemError for certain
    inputs (negative slice index inside prange, same root cause as the
    haplotype reconstruct kernel). We skip those inputs via ``assume(False)``
    so Hypothesis discards them rather than reporting a test failure.
    """
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("shift_and_realign_tracks_sparse")

    def run_numba():
        out = np.zeros(total_out, np.float32)
        args_list = [out] + list(inputs)
        try:
            numba_fn(*args_list)
        except SystemError:
            return None
        return out

    def run_rust():
        out = np.zeros(total_out, np.float32)
        args_list = [out] + list(inputs)
        rust_fn(*args_list)
        return out

    out_n = run_numba()
    if out_n is None:
        assume(False)
        return  # unreachable, keeps type-checkers happy

    out_r = run_rust()

    np.testing.assert_array_equal(out_n, out_r, err_msg="out mismatch (tracks)")


@settings(deadline=None, max_examples=500)
@given(shift_and_realign_tracks_inputs())
def test_shift_and_realign_tracks_all_strategies(args):
    total_out, inputs = args
    _assert_parity(total_out, inputs)
