"""Unit tests for Dataset.output_format field and with_output_format method."""

import pytest


def test_with_output_format_sets_field(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    assert ds.output_format == "ragged"
    flat = ds.with_output_format("flat")
    assert flat.output_format == "flat"
    # original is unchanged (frozen dataclass / replace semantics)
    assert ds.output_format == "ragged"


def test_with_output_format_rejects_bad_value(snap_dataset):
    with pytest.raises(ValueError):
        snap_dataset.with_output_format("nope")


def test_with_settings_dummy_variant_sets_field(snap_dataset):
    import genvarloader as gvl

    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    dv = gvl.DummyVariant(start=-1, alt=b"N", ref=b"N")
    ds2 = ds.with_settings(dummy_variant=dv)
    assert ds2._seqs.dummy_variant == dv
    # original unchanged
    assert ds._seqs.dummy_variant is None
    # disable with False
    ds3 = ds2.with_settings(dummy_variant=False)
    assert ds3._seqs.dummy_variant is None


def test_dummy_variant_rejected_on_non_variant_kind(snap_dataset):
    import genvarloader as gvl

    ds = snap_dataset.with_seqs("haplotypes").with_settings(
        dummy_variant=gvl.DummyVariant()
    )
    # guard fires at access time when the output kind is not variants
    with pytest.raises(ValueError):
        _ = ds[0]


def test_with_settings_dummy_variant_false_noop_on_non_variant(snap_dataset):
    # disabling (False) on a ref-only dataset (Ref _seqs, not Haps) must be a
    # harmless no-op — it previously raised ValueError spuriously.
    # snap_dataset has a reference, so with_seqs("reference") gives a Ref _seqs.
    ds = snap_dataset.with_seqs("reference").with_tracks(False)
    # should NOT raise
    ds2 = ds.with_settings(dummy_variant=False)
    assert ds2 is not None
    # Also confirm idempotent clear on a variants dataset that never had dummy_variant set:
    # calling False on a Haps dataset with no prior dummy_variant leaves it None.
    ds_haps = snap_dataset.with_seqs("variants").with_tracks(False)
    assert ds_haps._seqs.dummy_variant is None
    ds_haps2 = ds_haps.with_settings(dummy_variant=False)
    assert ds_haps2._seqs.dummy_variant is None


def test_dummy_variant_export():
    import genvarloader as gvl

    from genvarloader._dataset._flat_variants import DummyVariant

    assert gvl.DummyVariant is DummyVariant
