import numpy as np
from genvarloader._dataset._flat_flanks import build_token_lut


def test_build_token_lut_dna():
    lut, dtype = build_token_lut(b"ACGT", unknown_token=4)
    assert lut.shape == (256,)
    assert dtype == np.uint8
    # alphabet bytes map to their index
    assert lut[ord("A")] == 0
    assert lut[ord("C")] == 1
    assert lut[ord("G")] == 2
    assert lut[ord("T")] == 3
    # everything else -> unknown_token
    assert lut[ord("N")] == 4
    assert lut[0] == 4
    # tokenizing via fancy index works
    seq = np.frombuffer(b"ACGTN", dtype=np.uint8)
    assert lut[seq].tolist() == [0, 1, 2, 3, 4]


def test_build_token_lut_dtype_promotes_to_int32():
    # max token id 300 doesn't fit in uint8 -> int32
    lut, dtype = build_token_lut(bytes(range(200)), unknown_token=300)
    assert dtype == np.int32
    assert lut.dtype == np.int32


def test_with_settings_stores_flank_config(snap_dataset):
    ds = (
        snap_dataset.with_seqs("variants")
        .with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)
    )
    haps = ds._seqs
    assert haps.flank_length == 5
    assert haps.token_lut is not None
    assert haps.token_lut[ord("A")] == 0
    assert haps.token_dtype == np.uint8


def test_with_settings_flank_length_zero_disables(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_settings(
        flank_length=0, token_alphabet=b"ACGT", unknown_token=4
    )
    assert ds._seqs.flank_length == 0
