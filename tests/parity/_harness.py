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


def assert_kernel_parity_dict(name: str, *inputs) -> None:
    """Parity for kernels that RETURN a dict of ``{name: (data, seq_offsets)}``.

    Asserts both backends produce identical key sets, and for each key the
    ``(data, seq_offsets)`` pair is byte-identical (dtype, shape, values).
    """
    numba_fn, rust_fn = _dispatch.backends(name)
    got_numba = numba_fn(*inputs)
    got_rust = rust_fn(*inputs)
    assert set(got_numba.keys()) == set(got_rust.keys()), (
        f"{name}: dict keys {set(got_numba.keys())} != {set(got_rust.keys())}"
    )
    for k in sorted(got_numba.keys()):
        nb_data, nb_off = got_numba[k]
        rs_data, rs_off = got_rust[k]
        nb_data = np.asarray(nb_data)
        rs_data = np.asarray(rs_data)
        nb_off = np.asarray(nb_off, np.int64)
        rs_off = np.asarray(rs_off, np.int64)
        assert nb_data.dtype == rs_data.dtype, (
            f"{name}['{k}'].data: dtype {nb_data.dtype} != {rs_data.dtype}"
        )
        assert nb_data.shape == rs_data.shape, (
            f"{name}['{k}'].data: shape {nb_data.shape} != {rs_data.shape}"
        )
        np.testing.assert_array_equal(
            nb_data, rs_data, err_msg=f"{name}['{k}'].data mismatch"
        )
        np.testing.assert_array_equal(
            nb_off, rs_off, err_msg=f"{name}['{k}'].offsets mismatch"
        )
