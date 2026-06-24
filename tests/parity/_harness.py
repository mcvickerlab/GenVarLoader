"""Run both registered backends and assert byte-identical output."""

from __future__ import annotations

import numpy as np

from genvarloader import _dispatch


def assert_kernel_parity(name: str, *inputs) -> None:
    numba_fn, rust_fn = _dispatch.backends(name)
    got_numba = numba_fn(*inputs)
    got_rust = rust_fn(*inputs)
    assert got_numba.dtype == got_rust.dtype, (
        f"{name}: dtype {got_numba.dtype} != {got_rust.dtype}"
    )
    assert got_numba.shape == got_rust.shape, (
        f"{name}: shape {got_numba.shape} != {got_rust.shape}"
    )
    np.testing.assert_array_equal(got_numba, got_rust)
