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


@pytest.mark.parametrize("kind", ["haplotypes", "annotated"])
def test_union_then_phased_seqs_raises(snap_dataset, kind):
    # Flag set first, then request a phased sequence type via with_seqs.
    u = snap_dataset.with_seqs("variants").with_settings(unphased_union=True)
    with pytest.raises(ValueError, match="unphased_union"):
        u.with_seqs(kind)


@pytest.mark.parametrize("kind", ["haplotypes", "annotated"])
def test_phased_seqs_then_union_raises(snap_dataset, kind):
    # Phased sequence type first, then the flag via with_settings.
    ds = snap_dataset.with_seqs(kind)
    with pytest.raises(ValueError, match="unphased_union"):
        ds.with_settings(unphased_union=True)


def test_union_allows_variant_windows(snap_dataset):
    # variant-windows is the supported output and must NOT raise.
    ds = (
        snap_dataset.with_seqs("variants")
        .with_tracks(False)
        .with_settings(unphased_union=True)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    assert ds._seqs.unphased_union is True


def _windows_ds(snap_dataset, union: bool):
    ds = snap_dataset.with_tracks(False).with_seqs("variants")
    if union:
        ds = ds.with_settings(unphased_union=True)
    return ds.with_output_format("flat").with_seqs(
        "variant-windows",
        VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
    )


def test_variant_windows_union_collapses_ploidy_axis(snap_dataset):
    ploidy = snap_dataset._seqs.genotypes.shape[-2]
    assert ploidy == 2

    u = _windows_ds(snap_dataset, union=True)
    out = u[[[0, 1]], [[0, 1]]]  # out_reshape == (1, 2)
    # Ploidy axis folded 2 -> 1; scalar field shape (1, 2, 1, None).
    assert out.shape == (1, 2, 1, None)
    # Window buffers carry the extra ragged window axis: (1, 2, 1, None, None).
    assert out.ref_window.shape == (1, 2, 1, None, None)
    assert out.alt_window.shape == (1, 2, 1, None, None)
    # to_ragged must still work (offsets/data consistent after the fold).
    out.ref_window.to_ragged()
    out.alt_window.to_ragged()


def test_variant_windows_union_count_matches_sum_over_haplotypes(snap_dataset):
    baseline = _windows_ds(snap_dataset, union=False)
    union = _windows_ds(snap_dataset, union=True)

    r_idx = np.arange(min(4, snap_dataset.shape[0]))

    # One array index + a slice over all samples => "combo" (outer-product)
    # indexing, giving the (R, S) leading shape the assertions below expect.
    # (Two equal-shaped arrays would pair element-wise, not outer-product.)
    b = baseline[r_idx, :]  # shape (R, S, ploidy, None)
    u = union[r_idx, :]  # shape (R, S, 1, None)

    # Per (region, sample): total variants across both haplotypes == union count.
    rb = b.fields["start"].to_ragged()  # Ragged (R, S, ploidy, ~v)
    ru = u.fields["start"].to_ragged()  # Ragged (R, S, 1, ~v)
    import awkward as ak

    base_counts = ak.to_numpy(ak.sum(ak.num(rb, axis=-1), axis=-1))  # (R, S)
    union_counts = ak.to_numpy(ak.sum(ak.num(ru, axis=-1), axis=-1))  # (R, S)
    np.testing.assert_array_equal(union_counts, base_counts)

    # Content check: the union's single row must be exactly hap-0's starts then
    # hap-1's starts concatenated (the fold reorders nothing and drops nothing).
    R, S = base_counts.shape
    for i in range(R):
        for j in range(S):
            # baseline: concat across the ploidy axis, in haplotype order
            expected = []
            for p in range(rb.shape[2]):
                expected.extend(ak.to_list(rb[i, j, p]))
            got = ak.to_list(ru[i, j, 0])
            assert got == expected, f"union row ({i},{j}) != concat of haplotype starts"


def test_ragged_variants_union_folds(snap_dataset):
    # The ragged "variants" output also honors the flag (decodes flat, then converts).
    u = (
        snap_dataset.with_tracks(False)
        .with_seqs("variants")
        .with_settings(unphased_union=True)
    )
    out = u[0, 0]  # RaggedVariants
    # alt layout shape is (ploidy, ~v, ~l) for a single (region, sample); ploidy == 1.
    import awkward as ak

    assert len(ak.to_list(out["alt"])) == 1


def test_unphased_union_toggle_off_restores_diploid(snap_dataset):
    """Clearing unphased_union=False restores ploidy 2 and allows phased output."""
    # Step 1: enable union -> ploidy 1.
    u = snap_dataset.with_seqs("variants").with_settings(unphased_union=True)
    assert u.ploidy == 1
    assert u.n_variants().shape[-1] == 1

    # Step 2: clear the flag -> back to diploid.
    cleared = u.with_settings(unphased_union=False)
    assert cleared.ploidy == 2
    assert cleared.n_variants().shape[-1] == 2

    # Step 3: requesting a phased output must NOT raise an unphased_union ValueError.
    # snap_dataset has tracks; disable them so with_seqs("haplotypes") is unambiguous.
    cleared_no_tracks = cleared.with_tracks(False)
    # This must not raise (the unphased_union flag is cleared).
    cleared_no_tracks.with_seqs("haplotypes")


def test_unphased_union_composes_with_af_filter(filtered_svar, source_bed, ref_fasta, tmp_path):
    """Union fold commutes with AF filtering: fold(AF-filtered) == AF-filtered-union."""
    import genvarloader as gvl

    out = tmp_path / "af_union.gvl"
    gvl.write(path=out, bed=source_bed, variants=filtered_svar, overwrite=True)

    base = (
        gvl.Dataset.open(out, reference=ref_fasta)
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ilen", "start", "AF"], min_af=0.3)
    )
    union = base.with_settings(unphased_union=True)

    # Diploid AF-filtered n_variants shape: (R, S, 2)
    n_base = base.n_variants()
    assert n_base.shape[-1] == 2

    # Union AF-filtered n_variants shape: (R, S, 1)
    n_union = union.n_variants()
    assert n_union.shape[-1] == 1

    # The union count must equal the sum over the ploidy axis of the AF-filtered baseline.
    np.testing.assert_array_equal(n_union[..., 0], n_base.sum(-1))
