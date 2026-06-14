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


def test_ploidy_reports_one_under_union(snap_dataset):
    baseline = snap_dataset.with_seqs("variants")
    assert baseline.ploidy == 2  # stored diploid
    u = baseline.with_settings(unphased_union=True)
    assert u.ploidy == 1


def test_n_variants_collapses_to_union_count(snap_dataset):
    baseline = snap_dataset.with_seqs("variants")
    # (R, S, ploidy)
    n2 = baseline.n_variants()
    assert n2.shape[-1] == 2

    u = baseline.with_settings(unphased_union=True)
    nu = u.n_variants()
    # Folded to a single haploid slot.
    assert nu.shape[-1] == 1
    # Naive union count == sum of per-haplotype counts (no dedup).
    np.testing.assert_array_equal(nu[..., 0], n2.sum(-1))
    # The folded result must keep the int32 return contract (sum() upcasts to int64).
    assert nu.dtype == np.int32


def test_n_variants_collapse_preserves_leading_shape(snap_dataset):
    u = snap_dataset.with_seqs("variants").with_settings(unphased_union=True)
    n2 = snap_dataset.with_seqs("variants").n_variants()
    nu = u.n_variants()
    # Region/sample axes unchanged, only ploidy axis folded 2 -> 1.
    assert nu.shape == (*n2.shape[:-1], 1)
