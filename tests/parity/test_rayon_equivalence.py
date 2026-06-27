"""Serial vs parallel rust output must be byte-identical (and == golden).

Tests that reconstruct_haplotypes_from_sparse produces identical output regardless of
whether parallel=False (serial rayon-free path) or parallel=True (rayon par_iter path).
Both must also match the frozen golden captured from the Rust implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity

# RUST_KERNELS stores the thin C1 shim that wraps the bare FFI function with a
# `parallel=False` default (so existing golden replays stay serial); it forwards
# *args and `parallel` straight through to the FFI. The FFI accepts `parallel` as
# a keyword argument (PyO3 registers all pyfunction args as keyword-capable), so
# passing parallel=True/False here exercises both branches.
_fn = _golden.RUST_KERNELS["reconstruct_haplotypes_from_sparse"]


def test_reconstruct_haplotypes_serial_eq_parallel():
    """For every frozen golden case: serial == parallel == golden (byte-identical)."""
    cases = _golden.load_golden("reconstruct_haplotypes_from_sparse")
    assert cases, "empty golden — run generate_goldens.py first"

    for ci, (inputs, golden) in enumerate(cases):
        golden_arr = np.asarray(golden)
        outs: dict[bool, np.ndarray] = {}
        for parallel in (False, True):
            out = np.zeros(golden_arr.shape, golden_arr.dtype)
            # inputs tuple: (out_offsets, regions, shifts, geno_offset_idx,
            #                geno_offsets_2d, geno_v_idxs, v_starts, ilens,
            #                alt_alleles, alt_offsets, reference, ref_offsets,
            #                pad_char, keep, keep_offsets, None, None)
            # The FFI takes `out` as the first positional arg; inputs do NOT include out.
            args = list(inputs)
            args.insert(0, out)
            _fn(*args, parallel=parallel)
            outs[parallel] = out

        np.testing.assert_array_equal(
            outs[False],
            outs[True],
            err_msg=f"case {ci}: serial != parallel",
        )
        np.testing.assert_array_equal(
            outs[True],
            golden_arr,
            err_msg=f"case {ci}: parallel != golden",
        )
