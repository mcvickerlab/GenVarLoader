"""Instance-axis slicing of flat window / flank types matches per-item output."""

import numpy as np
import genvarloader as gvl


def _win_eq(a, b):
    """Compare two _FlatVariantWindows via to_ragged() awkward lists."""
    da, db = a.to_ragged(), b.to_ragged()
    assert set(da) == set(db), f"keys differ: {set(da)} vs {set(db)}"
    for k in da:
        assert da[k].to_ak().to_list() == db[k].to_ak().to_list(), f"{k} mismatch"


def test_flat_variant_windows_slice_matches_per_item():
    ds = (
        gvl.get_dummy_dataset()
        .with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(
                flank_length=2,
                token_alphabet=b"ACGT",
                unknown_token=4,
                ref="window",
                alt="allele",
            ),
        )
    )
    r = np.array([0, 0, 1], np.intp)
    s = np.array([0, 1, 0], np.intp)
    batch = ds[r, s]  # one _FlatVariantWindows over 3 instances
    sliced = batch[1:3]  # instances 1,2
    expected = ds[r[1:3], s[1:3]]
    _win_eq(sliced, expected)


def test_flat_variants_flank_tokens_slice_carries_tokens():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("variants")
        .with_tracks(False)
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=0)
        .with_output_format("flat")
    )
    r = np.array([0, 0, 1], np.intp)
    s = np.array([0, 1, 0], np.intp)
    batch = ds[r, s]
    assert batch.flank_tokens is not None
    sliced = batch[1:3]  # must NOT raise, must keep flank_tokens
    assert sliced.flank_tokens is not None
    exp = ds[r[1:3], s[1:3]]
    np.testing.assert_array_equal(
        sliced.flank_tokens.to_ragged().to_ak().to_list(),
        exp.flank_tokens.to_ragged().to_ak().to_list(),
    )
