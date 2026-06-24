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


def assert_inplace_kernel_parity(name, inputs, out_factory, out_index) -> None:
    """Parity for kernels that WRITE an output buffer in place (return None).

    ``inputs`` is the read-only argument tuple WITHOUT the out buffer. A fresh
    out buffer is built per backend via ``out_factory()`` and inserted at
    positional ``out_index``. Asserts the two written buffers are byte-identical.
    """
    numba_fn, rust_fn = _dispatch.backends(name)

    out_numba = out_factory()
    args = list(inputs)
    args.insert(out_index, out_numba)
    numba_fn(*args)

    out_rust = out_factory()
    args = list(inputs)
    args.insert(out_index, out_rust)
    rust_fn(*args)

    assert out_numba.dtype == out_rust.dtype, (
        f"{name}: dtype {out_numba.dtype} != {out_rust.dtype}"
    )
    assert out_numba.shape == out_rust.shape, (
        f"{name}: shape {out_numba.shape} != {out_rust.shape}"
    )
    np.testing.assert_array_equal(out_numba, out_rust)


def assert_kernel_parity_tuple(name: str, *inputs) -> None:
    """Parity for kernels that RETURN one array or a tuple of arrays.

    Normalizes a non-tuple return into a 1-tuple, then asserts each element is
    byte-identical (dtype, shape, values) between the numba and rust backends.
    """
    numba_fn, rust_fn = _dispatch.backends(name)
    got_numba = numba_fn(*inputs)
    got_rust = rust_fn(*inputs)
    if not isinstance(got_numba, tuple):
        got_numba = (got_numba,)
    if not isinstance(got_rust, tuple):
        got_rust = (got_rust,)
    assert len(got_numba) == len(got_rust), (
        f"{name}: tuple len {len(got_numba)} != {len(got_rust)}"
    )
    for i, (a, b) in enumerate(zip(got_numba, got_rust)):
        a = np.asarray(a)
        b = np.asarray(b)
        assert a.dtype == b.dtype, f"{name}[{i}]: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{name}[{i}]: shape {a.shape} != {b.shape}"
        np.testing.assert_array_equal(a, b)
