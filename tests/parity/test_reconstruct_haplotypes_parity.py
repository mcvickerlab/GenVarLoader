"""Parity tests for reconstruct_haplotypes_from_sparse (batch kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings

from genvarloader._dataset import _genotypes  # noqa: F401 — triggers register()
from tests.parity.strategies import reconstruct_haplotypes_inputs

pytestmark = pytest.mark.parity


def _make_out_factory(total_out: int):
    def factory():
        return np.empty(total_out, np.uint8)

    return factory


def _assert_non_annotated_parity(total_out: int, inputs: tuple) -> None:
    """Check that the out buffer is byte-identical between numba and Rust.

    The numba parallel batch driver has a known SystemError for certain inputs
    (negative slice index inside prange, same root cause as the annotated path).
    We skip those inputs via ``assume(False)`` so Hypothesis discards them
    rather than reporting a test failure.
    """
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("reconstruct_haplotypes_from_sparse")

    def run_numba():
        out = np.empty(total_out, np.uint8)
        args_list = [out] + list(inputs)
        numba_fn(*args_list)
        return out

    def run_rust():
        out = np.empty(total_out, np.uint8)
        args_list = [out] + list(inputs)
        rust_fn(*args_list)
        return out

    # numba's parallel=True batch kernel has a pre-existing SystemError on
    # some inputs (negative slice index inside prange).  Skip those inputs so
    # Hypothesis discards them.
    try:
        out_n = run_numba()
    except SystemError:
        assume(False)
        return  # unreachable, but keeps type-checkers happy

    out_r = run_rust()

    np.testing.assert_array_equal(out_n, out_r, err_msg="out mismatch (non-annotated)")


@settings(deadline=None)
@given(reconstruct_haplotypes_inputs(annotate=False))
def test_reconstruct_haplotypes_non_annotated(args):
    total_out, inputs = args
    _assert_non_annotated_parity(total_out, inputs)


def _assert_annotated_parity(total_out: int, inputs: tuple) -> None:
    """Check all three inplace buffers (out, annot_v_idxs, annot_ref_pos) match.

    The numba parallel batch driver has a known SystemError for certain inputs
    when annotation arrays are provided (numba parallel=True + negative slice
    index in annotated path).  We skip those inputs via ``assume(False)`` so
    Hypothesis discards them rather than reporting a test failure.
    """
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("reconstruct_haplotypes_from_sparse")

    def run_numba():
        out = np.empty(total_out, np.uint8)
        annot_v = np.empty(total_out, np.int32)
        annot_pos = np.empty(total_out, np.int32)
        args_list = [out] + list(inputs[:-2]) + [annot_v, annot_pos]
        numba_fn(*args_list)
        return out, annot_v, annot_pos

    def run_rust():
        out = np.empty(total_out, np.uint8)
        annot_v = np.empty(total_out, np.int32)
        annot_pos = np.empty(total_out, np.int32)
        args_list = [out] + list(inputs[:-2]) + [annot_v, annot_pos]
        rust_fn(*args_list)
        return out, annot_v, annot_pos

    # numba's parallel=True batch kernel has a pre-existing SystemError on
    # some annotated inputs (negative slice index inside prange).  Skip those
    # inputs so Hypothesis discards them.
    try:
        out_n, av_n, ap_n = run_numba()
    except SystemError:
        assume(False)
        return  # unreachable, but keeps type-checkers happy

    out_r, av_r, ap_r = run_rust()

    np.testing.assert_array_equal(out_n, out_r, err_msg="out mismatch (annotated)")
    np.testing.assert_array_equal(av_n, av_r, err_msg="annot_v_idxs mismatch")
    np.testing.assert_array_equal(ap_n, ap_r, err_msg="annot_ref_pos mismatch")


@settings(deadline=None)
@given(reconstruct_haplotypes_inputs(annotate=True))
def test_reconstruct_haplotypes_annotated(args):
    total_out, inputs = args
    _assert_annotated_parity(total_out, inputs)
