"""Parity tests for reconstruct_haplotypes_from_sparse (batch kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings

from genvarloader._dataset import _genotypes  # noqa: F401 — triggers register()
from tests.parity.strategies import reconstruct_haplotypes_inputs

pytestmark = pytest.mark.parity


def _assert_non_annotated_parity(total_out: int, inputs: tuple) -> None:
    """Check that the out buffer is byte-identical between numba and Rust.

    Both kernels now fully write every output position (including the
    trailing-fill overshoot sub-domain where a deletion drives ref_idx past
    the contig end), so no exclusion guards are needed.
    """
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("reconstruct_haplotypes_from_sparse")

    out_n = np.empty(total_out, dtype=np.uint8)
    numba_fn(*([out_n] + list(inputs)))

    out_r = np.empty(total_out, dtype=np.uint8)
    rust_fn(*([out_r] + list(inputs)))

    np.testing.assert_array_equal(out_n, out_r, err_msg="out mismatch (non-annotated)")


@settings(deadline=None)
@given(reconstruct_haplotypes_inputs(annotate=False))
def test_reconstruct_haplotypes_non_annotated(args):
    total_out, inputs = args
    _assert_non_annotated_parity(total_out, inputs)


def _assert_annotated_parity(total_out: int, inputs: tuple) -> None:
    """Check all three inplace buffers (out, annot_v_idxs, annot_ref_pos) match.

    Both kernels now fully write every output position (including the
    trailing-fill overshoot sub-domain), so no exclusion guards are needed.
    """
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("reconstruct_haplotypes_from_sparse")

    out_n = np.empty(total_out, dtype=np.uint8)
    av_n = np.empty(total_out, dtype=np.int32)
    ap_n = np.empty(total_out, dtype=np.int32)

    numba_fn(*([out_n] + list(inputs[:-2]) + [av_n, ap_n]))

    out_r = np.empty(total_out, dtype=np.uint8)
    av_r = np.empty(total_out, dtype=np.int32)
    ap_r = np.empty(total_out, dtype=np.int32)
    rust_fn(*([out_r] + list(inputs[:-2]) + [av_r, ap_r]))

    np.testing.assert_array_equal(out_n, out_r, err_msg="out mismatch (annotated)")
    np.testing.assert_array_equal(av_n, av_r, err_msg="annot_v_idxs mismatch")
    np.testing.assert_array_equal(ap_n, ap_r, err_msg="annot_ref_pos mismatch")


@settings(deadline=None)
@given(reconstruct_haplotypes_inputs(annotate=True))
def test_reconstruct_haplotypes_annotated(args):
    total_out, inputs = args
    _assert_annotated_parity(total_out, inputs)
