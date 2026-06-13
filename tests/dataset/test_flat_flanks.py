import numpy as np
from genvarloader._dataset._flat_flanks import build_token_lut, compute_flank_tokens, compute_windows
from genvarloader._dataset._flat_variants import _FlatWindow, _FlatVariantWindows
from genvarloader._dataset._flat_flanks import (
    compute_ref_window,
    compute_alt_window,
    tokenize_alleles,
)
from genvarloader._dataset._flat_variants import VarWindowOpt


def _flatten_lut_flanks(ref, contigs, starts, ilens, flank_len, lut):
    """Independent oracle: per-variant [flank5|flank3] tokens, shape (n_var, 2L)."""
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = ref.fetch(contigs, starts - flank_len, starts).data.view(np.uint8).reshape(-1, flank_len)
    f3 = ref.fetch(contigs, ends, ends + flank_len).data.view(np.uint8).reshape(-1, flank_len)
    return lut[np.concatenate([f5, f3], axis=1)]  # (n_var, 2L)


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


def _oracle_windows(reference, v_contigs, starts, ilens, alt_data, alt_seq_off,
                    flank_len, lut):
    ends = starts - np.minimum(ilens, 0) + 1
    # ref_window: single contiguous read [start-L, end+L)
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    ref_tok = lut[rw.data.view(np.uint8)]
    # alt_window: flank5 + alt + flank3
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    f5 = f5.reshape(len(starts), flank_len)
    f3 = f3.reshape(len(starts), flank_len)
    alt_rows = []
    for i in range(len(starts)):
        a = alt_data[alt_seq_off[i]:alt_seq_off[i + 1]]
        alt_rows.append(np.concatenate([f5[i], a, f3[i]]))
    alt_tok = lut[np.concatenate(alt_rows)] if alt_rows else np.empty(0, lut.dtype)
    return ref_tok, np.asarray(rw.offsets), alt_tok


def test_compute_windows_unit(snap_dataset):
    haps = snap_dataset.with_seqs("variants").with_settings(
        flank_length=3, token_alphabet=b"ACGT", unknown_token=4
    )._seqs
    ref, lut = haps.reference, haps.token_lut
    v_contigs = np.array([0, 0], dtype=np.int32)
    starts = np.array([10, 20], dtype=np.int32)
    ilens = np.array([0, -2], dtype=np.int32)
    # alt alleles: "AC" and "T"
    alt_data = np.frombuffer(b"ACT", dtype=np.uint8).copy()
    alt_seq_off = np.array([0, 2, 3], dtype=np.int64)
    row_offsets = np.array([0, 2], dtype=np.int64)
    ref_w, alt_w = compute_windows(
        ref, v_contigs, starts, ilens, alt_data, alt_seq_off, 3, lut, row_offsets
    )
    e_ref_tok, e_ref_off, e_alt_tok = _oracle_windows(
        ref, v_contigs, starts, ilens, alt_data, alt_seq_off, 3, lut
    )
    np.testing.assert_array_equal(ref_w.data, e_ref_tok)
    np.testing.assert_array_equal(ref_w.seq_offsets, e_ref_off)
    np.testing.assert_array_equal(alt_w.data, e_alt_tok)
    # alt_window offsets: per-variant window length = 2*flank_len + alt_len, cumsum from 0
    alt_lens = np.diff(alt_seq_off)
    e_alt_off = np.concatenate(
        [[0], np.cumsum(2 * 3 + alt_lens)]
    ).astype(np.int64)
    np.testing.assert_array_equal(alt_w.seq_offsets, e_alt_off)


def test_flank_tokens_end_to_end_matches_oracle(snap_dataset):
    import awkward as ak

    L = 5
    flat_ds = (
        snap_dataset.with_seqs("variants").with_tracks(False)
        .with_output_format("flat")
        .with_settings(flank_length=L, token_alphabet=b"ACGT", unknown_token=4)
    )
    rag_ds = snap_dataset.with_seqs("variants").with_tracks(False)
    idx = ([0, 1, 2], [0, 1, 2])

    flat = flat_ds[idx]
    rag = rag_ds[idx]
    assert flat.flank_tokens is not None

    ref = snap_dataset._seqs.reference
    lut = flat_ds._seqs.token_lut
    ploidy = snap_dataset._seqs.genotypes.shape[-2]

    # per-variant contigs: region contig repeated by ploidy then by variant counts,
    # matching get_variants_flat's (b, p, ~v) C-order.
    ds_idx, _, _ = snap_dataset._idxer.parse_idx(idx)
    r_idx, _ = np.unravel_index(np.asarray(ds_idx), snap_dataset._idxer.full_shape)
    region_contigs = snap_dataset._full_regions[r_idx, 0]
    counts = np.asarray(ak.flatten(ak.num(rag.start, axis=-1), axis=None))
    starts = np.asarray(ak.flatten(rag.start, axis=None))
    ilens = np.asarray(ak.flatten(rag.ilen, axis=None))
    v_contigs = np.repeat(np.repeat(region_contigs, ploidy), counts)

    expected = _flatten_lut_flanks(ref, v_contigs, starts, ilens, L, lut)  # (n_var, 2L)
    got = np.asarray(flat.flank_tokens.to_ragged().data)  # (n_var, 2L)
    np.testing.assert_array_equal(got, expected)


def test_variant_windows_kind_end_to_end(snap_dataset):
    ds = (
        snap_dataset.with_tracks(False)
        .with_output_format("flat")
        .with_settings(flank_length=4, token_alphabet=b"ACGT", unknown_token=4)
        .with_seqs("variant-windows")
    )
    out = ds[[0, 1], [0, 1]]
    assert out.ref_window is not None and out.alt_window is not None
    assert "alt" not in out.fields and "ref" not in out.fields


def test_variant_windows_requires_flank_settings(snap_dataset):
    import pytest

    with pytest.raises(ValueError, match="flank"):
        snap_dataset.with_seqs("variant-windows")  # no flank_length set


def test_variant_windows_requires_flat_output(snap_dataset):
    import pytest

    ds = (
        snap_dataset.with_tracks(False)
        .with_settings(flank_length=4, token_alphabet=b"ACGT", unknown_token=4)
        .with_seqs("variant-windows")
    )  # output_format defaults to "ragged"
    with pytest.raises(ValueError, match="flat"):
        _ = ds[[0, 1], [0, 1]]


def test_variant_windows_reshape_preserves_ploidy(snap_dataset):
    # A 2-D index (out_reshape != None) drives the _reshape_outer path. Regression
    # for a bug where windows dropped the ploidy dim during reshape.
    ds = (
        snap_dataset.with_tracks(False)
        .with_output_format("flat")
        .with_settings(flank_length=4, token_alphabet=b"ACGT", unknown_token=4)
        .with_seqs("variant-windows")
    )
    ploidy = snap_dataset._seqs.genotypes.shape[-2]
    out = ds[[[0, 1]], [[0, 1]]]  # out_reshape == (1, 2)
    # scalar start field shape: (1, 2, ploidy, None)
    assert out.shape == (1, 2, ploidy, None)
    # windows carry an extra ragged window axis: (1, 2, ploidy, None, None)
    assert out.ref_window.shape == (1, 2, ploidy, None, None)
    assert out.alt_window.shape == (1, 2, ploidy, None, None)
    # to_ragged must still work (offsets/data consistent after reshape)
    out.ref_window.to_ragged()
    out.alt_window.to_ragged()


def test_varwindowopt_defaults():
    opt = VarWindowOpt(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)
    assert opt.ref == "window" and opt.alt == "window"
    opt2 = VarWindowOpt(flank_length=5, token_alphabet=b"ACGT", unknown_token=4,
                        ref="allele", alt="window")
    assert opt2.ref == "allele"


def test_compute_ref_alt_window_split_matches_compute_windows(snap_dataset):
    # compute_windows must equal (compute_ref_window, compute_alt_window)
    from genvarloader._dataset._flat_flanks import compute_windows

    haps = snap_dataset.with_seqs("variants").with_settings(
        flank_length=3, token_alphabet=b"ACGT", unknown_token=4
    )._seqs
    ref, lut = haps.reference, haps.token_lut
    v_contigs = np.array([0, 0], dtype=np.int32)
    starts = np.array([10, 20], dtype=np.int32)
    ilens = np.array([0, -2], dtype=np.int32)
    alt_data = np.frombuffer(b"ACT", dtype=np.uint8).copy()
    alt_seq_off = np.array([0, 2, 3], dtype=np.int64)
    row_offsets = np.array([0, 2], dtype=np.int64)

    ref_w, alt_w = compute_windows(
        ref, v_contigs, starts, ilens, alt_data, alt_seq_off, 3, lut, row_offsets
    )
    ref_w2 = compute_ref_window(ref, v_contigs, starts, ilens, 3, lut, row_offsets)
    alt_w2 = compute_alt_window(
        ref, v_contigs, starts, ilens, alt_data, alt_seq_off, 3, lut, row_offsets
    )
    np.testing.assert_array_equal(ref_w.data, ref_w2.data)
    np.testing.assert_array_equal(ref_w.seq_offsets, ref_w2.seq_offsets)
    np.testing.assert_array_equal(alt_w.data, alt_w2.data)
    np.testing.assert_array_equal(alt_w.seq_offsets, alt_w2.seq_offsets)


def test_tokenize_alleles_bare(snap_dataset):
    haps = snap_dataset.with_seqs("variants").with_settings(
        flank_length=3, token_alphabet=b"ACGT", unknown_token=4
    )._seqs
    lut = haps.token_lut
    alt_data = np.frombuffer(b"ACT", dtype=np.uint8).copy()
    alt_seq_off = np.array([0, 2, 3], dtype=np.int64)
    row_offsets = np.array([0, 2], dtype=np.int64)
    w = tokenize_alleles(alt_data, alt_seq_off, lut, row_offsets)
    # bare allele tokens == LUT applied directly to allele bytes, no flanks
    np.testing.assert_array_equal(w.data, lut[alt_data])
    np.testing.assert_array_equal(w.seq_offsets, alt_seq_off)
    np.testing.assert_array_equal(w.var_offsets, row_offsets)


def test_flat_variant_windows_optional_fields():
    # _FlatVariantWindows now holds optional ref_window/alt_window/ref/alt
    from genvarloader._dataset._flat_variants import _FlatWindow, _FlatVariantWindows
    from genvarloader._flat import _Flat

    data = np.arange(4, dtype=np.uint8)
    seq_off = np.array([0, 2, 4], dtype=np.int64)
    var_off = np.array([0, 2], dtype=np.int64)
    w = _FlatWindow(data, seq_off, var_off, (1, 1, None, None))
    start = _Flat.from_offsets(np.array([10, 20]), (1, 1, None), var_off)
    # ref as window, alt as bare allele
    fvw = _FlatVariantWindows({"start": start}, ref_window=w, alt=w)
    assert fvw.ref_window is not None and fvw.alt is not None
    assert fvw.alt_window is None and fvw.ref is None
    rag = fvw.to_ragged()
    assert "ref_window" in rag and "alt" in rag
    assert "alt_window" not in rag and "ref" not in rag
    # reshape/squeeze only act on present fields
    fvw.reshape((1, 1))
    fvw.squeeze(0)
