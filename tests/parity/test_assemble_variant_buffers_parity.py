"""Parity: the new assemble_variant_buffers mega-call (rust) must be
byte-identical to the composed numba oracle for variants + variant-windows,
across the ref/alt mode matrix, the flank ride-along, and empty selections."""

import numpy as np
import pytest

import genvarloader._dataset._flat_variants  # noqa: F401  (triggers register())
from tests.parity._harness import assert_kernel_parity_dict

pytestmark = pytest.mark.parity


def _reference():
    # single contig of 40 bytes, ASCII A/C/G/T cycling.
    bases = np.frombuffer(b"ACGT", np.uint8)
    ref = np.tile(bases, 10).astype(np.uint8)
    ref_offsets = np.array([0, ref.size], np.int64)
    return ref, ref_offsets


def _lut(dtype):
    # A->0 C->1 G->2 T->3, everything else (incl. N) -> 4 (unknown).
    lut = np.full(256, 4, dtype)
    for i, b in enumerate(b"ACGT"):
        lut[b] = i
    return lut


def _globals():
    # 3 global variants: alt "A","CG","T"; ref "C","G","AA".
    alt_data = np.frombuffer(b"ACGT", np.uint8)
    alt_off = np.array([0, 1, 3, 4], np.int64)
    ref_data = np.frombuffer(b"CGAA", np.uint8)
    ref_off = np.array([0, 1, 2, 4], np.int64)
    v_starts = np.array([5, 12, 20], np.int32)
    ilens = np.array([0, -1, 1], np.int32)  # SNP, 1bp del, 1bp ins
    return alt_data, alt_off, ref_data, ref_off, v_starts, ilens


@pytest.mark.parametrize("tok_dtype", [np.uint8, np.int32])
@pytest.mark.parametrize("ref_mode,alt_mode", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_windows_mode_matrix(tok_dtype, ref_mode, alt_mode):
    ref, ref_offsets = _reference()
    alt_data, alt_off, ref_data, ref_off, v_starts, ilens = _globals()
    lut = _lut(tok_dtype)
    # one row selecting all 3 variants
    v_idxs = np.array([0, 1, 2], np.int32)
    row_offsets = np.array([0, 3], np.int64)
    v_contigs = np.zeros(3, np.int32)
    assert_kernel_parity_dict(
        "assemble_variant_buffers",
        1,  # windows
        v_idxs,
        row_offsets,
        alt_data,
        alt_off,
        ref_data,
        ref_off,
        False,
        False,
        ref_mode,
        alt_mode,
        2,
        lut,
        v_contigs,
        v_starts,
        ilens,
        ref,
        ref_offsets,
        ord("N"),
    )


@pytest.mark.parametrize("tok_dtype", [np.uint8, np.int32])
@pytest.mark.parametrize(
    "want_ref,want_flank", [(False, False), (True, False), (False, True), (True, True)]
)
def test_variants_mode_matrix(tok_dtype, want_ref, want_flank):
    ref, ref_offsets = _reference()
    alt_data, alt_off, ref_data, ref_off, v_starts, ilens = _globals()
    lut = _lut(tok_dtype) if want_flank else None
    v_idxs = np.array([2, 0, 1], np.int32)
    row_offsets = np.array([0, 1, 3], np.int64)  # 2 rows
    v_contigs = np.zeros(3, np.int32)
    assert_kernel_parity_dict(
        "assemble_variant_buffers",
        0,  # variants
        v_idxs,
        row_offsets,
        alt_data,
        alt_off,
        ref_data,
        ref_off,
        want_ref,
        want_flank,
        0,
        0,
        2,
        lut,
        v_contigs,
        v_starts,
        ilens,
        ref,
        ref_offsets,
        ord("N"),
    )


@pytest.mark.parametrize("mode,ref_mode,alt_mode", [(0, 0, 0), (1, 1, 1)])
def test_empty_selection(mode, ref_mode, alt_mode):
    """A row that selects zero variants must round-trip identically."""
    ref, ref_offsets = _reference()
    alt_data, alt_off, ref_data, ref_off, v_starts, ilens = _globals()
    lut = _lut(np.uint8)
    v_idxs = np.array([], np.int32)
    row_offsets = np.array([0, 0], np.int64)  # 1 empty row
    v_contigs = np.array([], np.int32)
    assert_kernel_parity_dict(
        "assemble_variant_buffers",
        mode,
        v_idxs,
        row_offsets,
        alt_data,
        alt_off,
        ref_data,
        ref_off,
        False,
        (mode == 0),
        ref_mode,
        alt_mode,
        2,
        lut,
        v_contigs,
        v_starts,
        ilens,
        ref,
        ref_offsets,
        ord("N"),
    )
