"""Parity tests for shift_and_realign_tracks_sparse (batch kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings

from genvarloader._dataset import _tracks  # noqa: F401 — triggers register()
from tests.parity.strategies import shift_and_realign_tracks_inputs

pytestmark = pytest.mark.parity


def _assert_parity(total_out: int, inputs: tuple) -> None:
    """Check that the out buffer is byte-identical between numba and Rust.

    Both kernels now fully write every output position (including the
    trailing-fill overshoot sub-domain where a deletion drives track_idx past
    the track end), so no exclusion guards are needed.
    """
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("shift_and_realign_tracks_sparse")

    out_n = np.zeros(total_out, np.float32)
    numba_fn(*([out_n] + list(inputs)))

    out_r = np.zeros(total_out, np.float32)
    rust_fn(*([out_r] + list(inputs)))

    np.testing.assert_array_equal(out_n, out_r, err_msg="out mismatch (tracks)")


@settings(deadline=None, max_examples=500)
@given(shift_and_realign_tracks_inputs())
def test_shift_and_realign_tracks_all_strategies(args):
    total_out, inputs = args
    _assert_parity(total_out, inputs)
