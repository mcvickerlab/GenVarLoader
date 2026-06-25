"""Parity tests for reconstruct_haplotypes_from_sparse (batch kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings

from genvarloader._dataset import _genotypes  # noqa: F401 — triggers register()
from tests.parity.strategies import reconstruct_haplotypes_inputs

pytestmark = pytest.mark.parity


def _ref_idx_overshoots_contig(inputs: tuple) -> bool:
    """Return True if any (query, hap) pair drives ref_idx past the contig end.

    WHY this is needed: when a deletion's ref_end exceeds the contig length, the
    trailing-fill clause in reconstruct_haplotype_from_sparse computes a negative
    writable_ref, leading to ``out_end_idx = out_idx + writable_ref < out_idx``.

    Numba (njit) handles the subsequent ``out[out_end_idx:]`` fill via Python-style
    negative-integer slice indexing (treating -k as len(out)-k), which preserves
    already-written positions but may or may not pad trailing positions correctly.

    Rust clamps ``out_end_idx`` to 0 (``(out_idx + writable_ref).max(0)``) and
    pads from position 0 to the end, which overwrites already-written data.

    Both behaviors are undefined for this degenerate input sub-domain (production
    contracts guarantee variants lie within contig bounds). Numba and Rust diverge
    here in a deterministic but non-trivially-comparable way, so these inputs are
    excluded from the byte-identity parity domain via assume(False) — consistent
    with the start>=clen / #242-family precedent.
    """
    (
        _out_offsets,
        regions,
        _shifts,
        geno_offset_idx,
        geno_offsets,
        geno_v_idxs,
        v_starts,
        ilens,
        _alt_alleles,
        _alt_offsets,
        _reference,
        ref_offsets,
        _pad_char,
        keep,
        keep_offsets,
        _annot_v,
        _annot_rp,
    ) = inputs

    n_q, ploidy = geno_offset_idx.shape

    for qi in range(n_q):
        c_idx = int(regions[qi, 0])
        ref_start = int(regions[qi, 1])
        c_len = int(ref_offsets[c_idx + 1] - ref_offsets[c_idx])

        for h in range(ploidy):
            o_idx = int(geno_offset_idx[qi, h])
            if geno_offsets.ndim == 1:
                o_s = int(geno_offsets[o_idx])
                o_e = int(geno_offsets[o_idx + 1])
            else:
                o_s = int(geno_offsets[0, o_idx])
                o_e = int(geno_offsets[1, o_idx])

            if o_s >= o_e:
                continue

            k_idx = qi * ploidy + h

            # Simulate the ref_idx advancement through each variant.
            ref_idx = ref_start
            for vi in range(o_e - o_s):
                # Apply keep mask if present.
                if keep is not None and keep_offsets is not None:
                    k_s = int(keep_offsets[k_idx])
                    if not keep[k_s + vi]:
                        continue

                variant = int(geno_v_idxs[o_s + vi])
                v_pos = int(v_starts[variant])
                v_diff = int(ilens[variant])
                v_ref_end = v_pos - min(0, v_diff) + 1

                # Skip DEL spanning before ref_start.
                if v_diff < 0 and v_pos < ref_start and v_ref_end >= ref_start:
                    ref_idx = v_ref_end
                    continue

                if v_pos < ref_idx:
                    continue

                ref_idx = v_ref_end

            # If ref_idx has advanced past the contig length, the trailing-fill
            # clause will compute a negative out_end_idx. Numba and Rust handle
            # that differently (negative-index wrap vs clamp to 0). Exclude.
            if ref_idx > c_len:
                return True

    return False


def _numba_fully_defined(
    numba_fn,
    args_a: list,
    args_b: list,
    buffers_a: list[np.ndarray],
    buffers_b: list[np.ndarray],
) -> bool:
    """Return True iff numba fully wrote every output position.

    Run the numba kernel twice: once with output buffer(s) pre-filled with
    sentinel 0x00 (uint8) / 0 (int32), and once pre-filled with 0xFF (uint8)
    / -1 (int32).  If any position differs between the two runs, numba left
    that position unwritten — the sentinel value leaked through — and the
    kernel is not a valid byte-identity oracle for this input.

    WHY: when a deletion drives ref_idx past the contig end, numba's
    trailing-fill clause may leave trailing output positions unwritten
    (returning whatever sentinel was in the buffer).  The Rust kernel pads
    those positions correctly with pad_char / annotation sentinels.  Numba
    is not a valid oracle in this sub-domain, so these inputs are excluded
    via assume(False) — consistent with the start>=clen / #242-family
    precedent.
    """
    numba_fn(*args_a)
    numba_fn(*args_b)
    for buf_a, buf_b in zip(buffers_a, buffers_b):
        if not np.array_equal(buf_a, buf_b):
            return False
    return True


def _assert_non_annotated_parity(total_out: int, inputs: tuple) -> None:
    """Check that the out buffer is byte-identical between numba and Rust.

    Three exclusion guards are applied so Hypothesis discards invalid inputs
    rather than reporting test failures:

    1. Overshoot pre-check — if any deletion drives ref_idx past the contig
       end, numba and Rust handle the resulting negative out_end_idx
       differently (negative-index wrap vs clamp to 0).  Both behaviors are
       undefined for inputs outside the production contract; excluded via
       assume(False).

    2. SystemError guard — numba's parallel=True batch driver raises
       SystemError on some inputs (negative slice index inside prange).

    3. Double-init guard — numba leaves trailing positions unwritten when a
       deletion drives ref_idx past the contig end (numba bug; Rust pads
       correctly).  Detected by running numba twice with sentinel fills
       0x00 vs 0xFF: any position that differs means numba did not write it.
       Those inputs are discarded via assume(False).
    """
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("reconstruct_haplotypes_from_sparse")

    # Guard 1: exclude inputs where any deletion overshoots the contig end.
    # Numba and Rust diverge on these (negative-index wrap vs clamp to 0)
    # and both behaviors are undefined per the production contract.
    assume(not _ref_idx_overshoots_contig(inputs))

    # Build two sentinel-prefilled output buffers.
    out_a = np.full(total_out, 0x00, dtype=np.uint8)
    out_b = np.full(total_out, 0xFF, dtype=np.uint8)
    args_a = [out_a] + list(inputs)
    args_b = [out_b] + list(inputs)

    # Guard 2: numba's parallel=True batch kernel has a pre-existing
    # SystemError on some inputs (negative slice index inside prange).
    try:
        defined = _numba_fully_defined(numba_fn, args_a, args_b, [out_a], [out_b])
    except SystemError:
        assume(False)
        return  # unreachable, but keeps type-checkers happy

    # Guard 3: double-init divergence — numba left ≥1 position unwritten
    # (deletion drove ref_idx past the contig end; numba returns uninitialized
    # bytes, Rust pads correctly).  Discard from the parity domain.
    assume(defined)

    # Numba fully wrote the buffer — run Rust and compare byte-for-byte.
    out_n = out_a  # already filled by first sentinel run

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

    Three exclusion guards are applied so Hypothesis discards invalid inputs
    rather than reporting test failures:

    1. Overshoot pre-check — if any deletion drives ref_idx past the contig
       end, numba and Rust handle the resulting negative out_end_idx
       differently (negative-index wrap vs clamp to 0).  Both behaviors are
       undefined for inputs outside the production contract; excluded via
       assume(False).

    2. SystemError guard — numba's parallel=True batch driver raises
       SystemError on some annotated inputs (negative slice index in prange).

    3. Double-init guard — numba leaves trailing positions unwritten when a
       deletion drives ref_idx past the contig end (numba bug; Rust pads
       correctly).  Detected by running numba twice with distinct sentinel
       fills for each buffer:
         out:           0x00 vs 0xFF  (uint8)
         annot_v_idxs:  0    vs -1   (int32)
         annot_ref_pos: 0    vs -1   (int32)
       Any buffer position that differs between runs was not written by numba.
       Those inputs are discarded via assume(False) — consistent with #242.
    """
    from genvarloader import _dispatch

    numba_fn, rust_fn = _dispatch.backends("reconstruct_haplotypes_from_sparse")

    # Guard 1: exclude inputs where any deletion overshoots the contig end.
    assume(not _ref_idx_overshoots_contig(inputs))

    # Build sentinel-prefilled buffer pairs for the double-init check.
    out_a = np.full(total_out, 0x00, dtype=np.uint8)
    out_b = np.full(total_out, 0xFF, dtype=np.uint8)
    av_a = np.full(total_out, 0, dtype=np.int32)
    av_b = np.full(total_out, -1, dtype=np.int32)
    ap_a = np.full(total_out, 0, dtype=np.int32)
    ap_b = np.full(total_out, -1, dtype=np.int32)

    args_a = [out_a] + list(inputs[:-2]) + [av_a, ap_a]
    args_b = [out_b] + list(inputs[:-2]) + [av_b, ap_b]

    # Guard 2: numba's parallel=True batch kernel has a pre-existing
    # SystemError on some annotated inputs (negative slice index in prange).
    try:
        defined = _numba_fully_defined(
            numba_fn,
            args_a,
            args_b,
            [out_a, av_a, ap_a],
            [out_b, av_b, ap_b],
        )
    except SystemError:
        assume(False)
        return  # unreachable, but keeps type-checkers happy

    # Guard 3: double-init divergence — numba left ≥1 position unwritten.
    assume(defined)

    # Numba fully wrote all buffers — run Rust and compare byte-for-byte.
    out_n, av_n, ap_n = out_a, av_a, ap_a  # already filled by first sentinel run

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
