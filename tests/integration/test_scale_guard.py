"""Scale-guard: no per-batch copy materializes a memmap on the read path (Task 4).

Mirrors the py-spy diagnostic that found the defect: monkeypatch
np.ascontiguousarray over one ds[r, s] and assert zero copies whose source
.base is an np.memmap.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl


@pytest.fixture
def _no_memmap_copies(monkeypatch):
    real = np.ascontiguousarray
    offenders: list[str] = []

    def spy(a, dtype=None, *args, **kwargs):
        arr = np.asarray(a)
        base = getattr(arr, "base", None)
        if isinstance(base, np.memmap) or isinstance(arr, np.memmap):
            # A copy would be forced iff non-contiguous or dtype-mismatched.
            would_copy = (not arr.flags["C_CONTIGUOUS"]) or (
                dtype is not None and arr.dtype != np.dtype(dtype)
            )
            if would_copy:
                offenders.append(f"{getattr(arr, 'shape', None)} {arr.dtype}->{dtype}")
        return real(a, dtype, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", spy)
    return offenders


def test_tracks_only_no_memmap_copy(track_dataset_path, reference, _no_memmap_copies):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_tracks("cov")
    _ = ds[0, 0]
    assert _no_memmap_copies == [], f"sample-scale memmap copies: {_no_memmap_copies}"


def test_haps_no_memmap_copy(track_dataset_path, reference, _no_memmap_copies):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_seqs(
        "haplotypes"
    )
    _ = ds[0, 0]
    assert _no_memmap_copies == [], f"sample-scale memmap copies: {_no_memmap_copies}"


def test_annotated_no_memmap_copy(track_dataset_path, reference, _no_memmap_copies):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_seqs(
        "annotated"
    )
    _ = ds[0, 0]
    assert _no_memmap_copies == [], f"sample-scale memmap copies: {_no_memmap_copies}"


def test_haps_and_tracks_no_memmap_copy(
    track_dataset_path, reference, _no_memmap_copies
):
    ds = (
        gvl.Dataset.open(track_dataset_path, reference=reference)
        .with_seqs("haplotypes")
        .with_tracks("cov")
    )
    _ = ds[0, 0]
    assert _no_memmap_copies == [], f"sample-scale memmap copies: {_no_memmap_copies}"


def test_annotated_and_tracks_no_memmap_copy(
    track_dataset_path, reference, _no_memmap_copies
):
    ds = (
        gvl.Dataset.open(track_dataset_path, reference=reference)
        .with_seqs("annotated")
        .with_tracks("cov")
    )
    _ = ds[0, 0]
    assert _no_memmap_copies == [], f"sample-scale memmap copies: {_no_memmap_copies}"
