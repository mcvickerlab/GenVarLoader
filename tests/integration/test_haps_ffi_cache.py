"""Haps caches FFI-ready sub-linear arrays once (Task 5)."""

from __future__ import annotations

import numpy as np

import genvarloader as gvl
from genvarloader._dataset._haps import Haps


def _haps(track_dataset_path, reference) -> Haps:
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_seqs(
        "haplotypes"
    )
    seqs = ds._seqs
    assert isinstance(seqs, Haps)
    return seqs


def test_ffi_static_cached(track_dataset_path, reference):
    haps = _haps(track_dataset_path, reference)
    first = haps.ffi_static
    second = haps.ffi_static
    assert first is second  # cached, computed once


def test_ffi_static_contiguous_and_typed(track_dataset_path, reference):
    s = _haps(track_dataset_path, reference).ffi_static
    assert s.v_starts.dtype == np.int32 and s.v_starts.flags["C_CONTIGUOUS"]
    assert s.ilens.dtype == np.int32 and s.ilens.flags["C_CONTIGUOUS"]
    assert s.alt_alleles.dtype == np.uint8 and s.alt_alleles.flags["C_CONTIGUOUS"]
    assert s.alt_offsets.dtype == np.int64 and s.alt_offsets.flags["C_CONTIGUOUS"]
    assert s.ref is not None and s.ref.dtype == np.uint8 and s.ref.flags["C_CONTIGUOUS"]
    assert s.ref_offsets is not None and s.ref_offsets.dtype == np.int64


def test_ffi_static_v_starts_matches_source(track_dataset_path, reference):
    haps = _haps(track_dataset_path, reference)
    np.testing.assert_array_equal(
        haps.ffi_static.v_starts, np.asarray(haps.variants.start, np.int32)
    )
