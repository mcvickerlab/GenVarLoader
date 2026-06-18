import awkward as ak
import pytest

import genvarloader as gvl
from genvarloader._dataset._reconstruct import HapsTracks, SeqsTracks


def test_default_haps_tracks_realigns():
    ds = gvl.get_dummy_dataset()  # default: haplotypes + tracks
    assert type(ds._recon) is HapsTracks
    assert ds.realign_tracks is True


def test_realign_false_haps_tracks_uses_seqstracks_and_is_reference_coord():
    ds = gvl.get_dummy_dataset()
    asis = (
        ds.with_seqs("haplotypes")
        .with_tracks(["read-depth"])
        .with_settings(realign_tracks=False)
    )
    assert type(asis._recon) is SeqsTracks

    # As-is track must equal the solo (reference-coordinate) track values.
    solo = ds.with_seqs(None).with_tracks(["read-depth"])
    _, t = asis[[0], [0]]
    t_solo = solo[[0], [0]]
    assert ak.to_list(t) == ak.to_list(t_solo)


def test_intervals_plus_haplotypes_requires_realign_false():
    ds = gvl.get_dummy_dataset()  # default haplotypes + tracks, realign True
    with pytest.raises(ValueError, match="realign"):
        ds.with_tracks(["read-depth"], kind="intervals")


def test_intervals_plus_haplotypes_ok_when_realign_false():
    ds = gvl.get_dummy_dataset()
    out = ds.with_settings(realign_tracks=False).with_tracks(
        ["read-depth"], kind="intervals"
    )[[0], [0]]
    seqs, itvs = out
    assert isinstance(itvs, gvl.RaggedIntervals)


def test_insertion_fill_rejected_when_realign_false():
    ds = gvl.get_dummy_dataset().with_settings(realign_tracks=False)
    with pytest.raises(ValueError, match="realign"):
        ds.with_insertion_fill(gvl.Constant(value=0.0))


def test_realign_tracks_survives_with_len():
    ds = gvl.get_dummy_dataset().with_settings(realign_tracks=False)
    ds2 = ds.with_len("variable")
    assert ds2.realign_tracks is False
    assert type(ds2._recon) is SeqsTracks


def _vw_opt():
    return gvl.VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4)


def test_variant_windows_plus_float_tracks():
    ds = gvl.get_dummy_dataset()
    vw = (
        ds.with_settings(realign_tracks=False)
        .with_output_format("flat")
        .with_seqs("variant-windows", _vw_opt())
        .with_tracks(["read-depth"])  # default kind="tracks"
    )
    out = vw[[0, 1], [0, 1]]
    assert isinstance(out, tuple) and len(out) == 2
    windows, tracks = out
    assert type(windows).__name__ == "_FlatVariantWindows"
    assert type(tracks).__name__ == "_Flat"  # FlatRagged float track


def test_variant_windows_plus_intervals():
    ds = gvl.get_dummy_dataset()
    vw = (
        ds.with_settings(realign_tracks=False)
        .with_output_format("flat")
        .with_seqs("variant-windows", _vw_opt())
        .with_tracks(["read-depth"], kind="intervals")
    )
    windows, itvs = vw[[0, 1], [0, 1]]
    assert type(windows).__name__ == "_FlatVariantWindows"
    assert type(itvs).__name__ == "FlatIntervals"


def test_variant_windows_plus_tracks_requires_realign_false():
    ds = gvl.get_dummy_dataset()  # tracks active by default, realign True
    with pytest.raises(ValueError, match="realign"):
        ds.with_output_format("flat").with_seqs("variant-windows", _vw_opt())
