import numpy as np
from genvarloader._dataset._flat_flanks import build_token_lut, compute_flank_tokens
from genvarloader._dataset._flat_variants import _FlatWindow, _FlatVariantWindows


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


def test_with_settings_token_alphabet_requires_unknown_token(snap_dataset):
    import pytest

    with pytest.raises(ValueError, match="set together"):
        snap_dataset.with_seqs("variants").with_settings(token_alphabet=b"ACGT")


def test_with_settings_flank_requires_genotypes(reference, source_bed, tmp_path):
    import pytest
    import pyBigWig

    # Open a reference-only dataset (no variants) so _seqs is Ref, not Haps.
    # gvl.write requires at least tracks or variants, so provide a minimal BigWig.
    import genvarloader as gvl

    bw_path = tmp_path / "dummy.bw"
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    with pyBigWig.open(str(bw_path), "w") as bw:
        bw.addHeader(contig_sizes, maxZooms=0)
        bw.addEntries(["chr1"], [499_990], ends=[500_030], values=[1.0])

    out = tmp_path / "ref_only.gvl"
    gvl.write(path=out, bed=source_bed, tracks=gvl.BigWigs("sig", {"dummy": str(bw_path)}))
    ds = gvl.Dataset.open(out, reference=reference)
    # _seqs is Ref here; flank settings require Haps (genotypes present).
    with pytest.raises(ValueError, match="genotypes"):
        ds.with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)


def test_with_settings_flank_length_without_lut_errors(snap_dataset):
    import pytest

    with pytest.raises(ValueError, match="token LUT"):
        snap_dataset.with_seqs("variants").with_settings(flank_length=5)


def _oracle_flank_tokens(reference, v_contigs, starts, ilens, flank_len, lut):
    """Independent reference: fetch [start-L,start) and [end,end+L) per variant,
    tokenize, lay out [flank5|flank3]."""
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    f5 = f5.reshape(len(starts), flank_len)
    f3 = f3.reshape(len(starts), flank_len)
    flank_bytes = np.concatenate([f5, f3], axis=1)  # (n_var, 2L)
    return lut[flank_bytes]


def test_compute_flank_tokens_unit(snap_dataset):
    haps = snap_dataset.with_seqs("variants").with_settings(
        flank_length=3, token_alphabet=b"ACGT", unknown_token=4
    )._seqs
    ref = haps.reference
    lut = haps.token_lut
    # one (b=1, ploidy=1) group with two variants
    v_contigs = np.array([0, 0], dtype=np.int32)
    starts = np.array([10, 20], dtype=np.int32)
    ilens = np.array([0, -2], dtype=np.int32)  # SNP, 2bp deletion
    row_offsets = np.array([0, 2], dtype=np.int64)
    tokens, off = compute_flank_tokens(
        ref, v_contigs, starts, ilens, flank_len=3, lut=lut, row_offsets=row_offsets
    )
    expected = _oracle_flank_tokens(ref, v_contigs, starts, ilens, 3, lut)
    np.testing.assert_array_equal(tokens.reshape(-1, 6), expected)
    np.testing.assert_array_equal(off, row_offsets)


def test_flat_window_to_ragged_roundtrip():
    import awkward as ak

    # two groups (b=2, p=1), variant counts [2, 1]; window lens [3, 4 | 2]
    token_data = np.arange(3 + 4 + 2, dtype=np.uint8)
    seq_offsets = np.array([0, 3, 7, 9], dtype=np.int64)  # per-variant
    var_offsets = np.array([0, 2, 3], dtype=np.int64)  # per group
    shape = (2, 1, None, None)
    w = _FlatWindow(token_data, seq_offsets, var_offsets, shape)
    rag = w.to_ragged()  # ak.Array (2, 1, ~v, ~win) — two ragged axes
    assert rag.ndim == 4
    # element-identical content after wrapping
    np.testing.assert_array_equal(
        np.asarray(ak.flatten(rag, axis=None)).view(np.uint8), token_data
    )
    # structure preserved: variant counts per (b, p) and window length per variant
    assert ak.to_list(ak.num(rag, axis=2)) == [[2], [1]]
    assert ak.to_list(ak.num(rag, axis=3)) == [[[3, 4]], [[2]]]
