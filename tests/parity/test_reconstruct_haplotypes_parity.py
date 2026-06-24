"""Parity tests for reconstruct_haplotypes_from_sparse (batch kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings

from genvarloader._dataset import _genotypes  # noqa: F401 — triggers register()
from tests.parity._harness import assert_inplace_kernel_parity
from tests.parity.strategies import reconstruct_haplotypes_inputs

pytestmark = pytest.mark.parity


def _make_out_factory(total_out: int):
    def factory():
        return np.empty(total_out, np.uint8)

    return factory


@settings(deadline=None)
@given(reconstruct_haplotypes_inputs(annotate=False))
def test_reconstruct_haplotypes_non_annotated(args):
    total_out, inputs = args
    assert_inplace_kernel_parity(
        "reconstruct_haplotypes_from_sparse",
        inputs,
        _make_out_factory(total_out),
        out_index=0,
    )


def _assert_annotated_parity(total_out: int, inputs: tuple) -> None:
    """Check all three inplace buffers (out, annot_v_idxs, annot_ref_pos) match."""
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("reconstruct_haplotypes_from_sparse")

    def run(fn):
        out = np.empty(total_out, np.uint8)
        annot_v = np.empty(total_out, np.int32)
        annot_pos = np.empty(total_out, np.int32)
        # inputs: (out_offsets, regions, shifts, geno_offset_idx, geno_offsets,
        #          geno_v_idxs, v_starts, ilens, alt_alleles, alt_offsets,
        #          ref_, ref_offsets, pad_char, keep, keep_offsets, None, None)
        # Replace last two Nones with actual annotation buffers.
        args_list = [out] + list(inputs[:-2]) + [annot_v, annot_pos]
        fn(*args_list)
        return out, annot_v, annot_pos

    out_n, av_n, ap_n = run(numba_fn)
    out_r, av_r, ap_r = run(rust_fn)

    np.testing.assert_array_equal(out_n, out_r, err_msg="out mismatch (annotated)")
    np.testing.assert_array_equal(av_n, av_r, err_msg="annot_v_idxs mismatch")
    np.testing.assert_array_equal(ap_n, ap_r, err_msg="annot_ref_pos mismatch")


@settings(deadline=None)
@given(reconstruct_haplotypes_inputs(annotate=True))
def test_reconstruct_haplotypes_annotated(args):
    total_out, inputs = args
    _assert_annotated_parity(total_out, inputs)
