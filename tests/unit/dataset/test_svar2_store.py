"""Svar2Store pyclass: opens one query-only genoray_core ContigReader per contig
at construction (the SVAR2 analog of SVAR1's cached FFI-static), held for the
store's lifetime. Built from a real .svar2 store via genoray's conversion pipeline.
"""

from __future__ import annotations

from pathlib import Path

from genvarloader.genvarloader import Svar2Store  # compiled extension


def test_store_opens_contigs(svar2_store_2s: Path):
    store = Svar2Store(str(svar2_store_2s), ["chr1"], n_samples=2, ploidy=2)
    assert store.contigs() == ["chr1"]
