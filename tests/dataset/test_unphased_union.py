"""Tests for the ploidy-1 unphased union view (issue #222).

Uses the session-scoped ``snap_dataset`` fixture (tests/dataset/conftest.py):
a phased VCF dataset, ploidy 2, opened with a reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from genvarloader._dataset._flat_variants import VarWindowOpt


def test_with_settings_stores_unphased_union(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_settings(unphased_union=True)
    assert ds._seqs.unphased_union is True
    # Original dataset is unchanged (immutability).
    assert snap_dataset._seqs.unphased_union is False


def test_with_settings_unphased_union_requires_genotypes(
    reference, source_bed, tmp_path
):
    import pyBigWig

    import genvarloader as gvl

    # Reference-only dataset (no variants) -> _seqs is Ref, not Haps.
    bw_path = tmp_path / "dummy.bw"
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    with pyBigWig.open(str(bw_path), "w") as bw:
        bw.addHeader(contig_sizes, maxZooms=0)
        bw.addEntries(["chr1"], [499_990], ends=[500_030], values=[1.0])

    out = tmp_path / "ref_only.gvl"
    gvl.write(
        path=out, bed=source_bed, tracks=gvl.BigWigs("sig", {"dummy": str(bw_path)})
    )
    ds = gvl.Dataset.open(out, reference=reference)
    with pytest.raises(ValueError, match="genotypes"):
        ds.with_settings(unphased_union=True)
