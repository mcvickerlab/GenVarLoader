import numpy as np
import pytest
from genvarloader._dataset._flat_flanks import (
    build_token_lut,
    compute_flank_tokens,
    compute_windows,
)
from genvarloader._dataset._flat_variants import _FlatWindow, _FlatVariantWindows
from genvarloader._dataset._flat_flanks import (
    compute_ref_window,
    compute_alt_window,
    tokenize_alleles,
)
from genvarloader._dataset._flat_variants import DummyVariant, VarWindowOpt


def _flatten_lut_flanks(ref, contigs, starts, ilens, flank_len, lut):
    """Independent oracle: per-variant [flank5|flank3] tokens, shape (n_var, 2L)."""
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = (
        ref.fetch(contigs, starts - flank_len, starts)
        .data.view(np.uint8)
        .reshape(-1, flank_len)
    )
    f3 = (
        ref.fetch(contigs, ends, ends + flank_len)
        .data.view(np.uint8)
        .reshape(-1, flank_len)
    )
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
    ds = snap_dataset.with_seqs("variants").with_settings(
        flank_length=5, token_alphabet=b"ACGT", unknown_token=4
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
    gvl.write(
        path=out, bed=source_bed, tracks=gvl.BigWigs("sig", {"dummy": str(bw_path)})
    )
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
    haps = (
        snap_dataset.with_seqs("variants")
        .with_settings(flank_length=3, token_alphabet=b"ACGT", unknown_token=4)
        ._seqs
    )
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


def _oracle_windows(
    reference, v_contigs, starts, ilens, alt_data, alt_seq_off, flank_len, lut
):
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
        a = alt_data[alt_seq_off[i] : alt_seq_off[i + 1]]
        alt_rows.append(np.concatenate([f5[i], a, f3[i]]))
    alt_tok = lut[np.concatenate(alt_rows)] if alt_rows else np.empty(0, lut.dtype)
    return ref_tok, np.asarray(rw.offsets), alt_tok


def test_compute_windows_unit(snap_dataset):
    haps = (
        snap_dataset.with_seqs("variants")
        .with_settings(flank_length=3, token_alphabet=b"ACGT", unknown_token=4)
        ._seqs
    )
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
    e_alt_off = np.concatenate([[0], np.cumsum(2 * 3 + alt_lens)]).astype(np.int64)
    np.testing.assert_array_equal(alt_w.seq_offsets, e_alt_off)


def test_flank_tokens_end_to_end_matches_oracle(snap_dataset):
    import awkward as ak

    L = 5
    flat_ds = (
        snap_dataset.with_seqs("variants")
        .with_tracks(False)
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
    got = np.asarray(flat.flank_tokens.to_ragged().data).reshape(
        -1, 2 * L
    )  # (n_var, 2L)
    np.testing.assert_array_equal(got, expected)


def test_variant_windows_kind_end_to_end(snap_dataset):
    from genvarloader._dataset._flat_variants import VarWindowOpt

    ds = (
        snap_dataset.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    out = ds[[0, 1], [0, 1]]
    # default: both alleles are windows
    assert out.ref_window is not None and out.alt_window is not None
    assert out.ref is None and out.alt is None
    assert "alt" not in out.fields and "ref" not in out.fields


def test_variant_windows_requires_opt(snap_dataset):
    import pytest

    with pytest.raises(ValueError, match="VarWindowOpt"):
        snap_dataset.with_seqs("variant-windows")  # no VarWindowOpt


def test_variant_windows_requires_flat_output(snap_dataset):
    import pytest
    from genvarloader._dataset._flat_variants import VarWindowOpt

    ds = snap_dataset.with_tracks(False).with_seqs(
        "variant-windows",
        VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
    )  # output_format defaults to "ragged"
    with pytest.raises(ValueError, match="flat"):
        _ = ds[[0, 1], [0, 1]]


def test_variant_windows_reshape_preserves_ploidy(snap_dataset):
    # A 2-D index (out_reshape != None) drives the _reshape_outer path. Regression
    # for a bug where windows dropped the ploidy dim during reshape.
    from genvarloader._dataset._flat_variants import VarWindowOpt

    ds = (
        snap_dataset.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
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


@pytest.mark.parametrize(
    "ref_mode,alt_mode",
    [
        ("window", "window"),
        ("window", "allele"),
        ("allele", "window"),
        ("allele", "allele"),
    ],
)
def test_variant_windows_matrix_fields(snap_dataset, ref_mode, alt_mode):
    from genvarloader._dataset._flat_variants import VarWindowOpt

    ds = (
        snap_dataset.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(
                flank_length=4,
                token_alphabet=b"ACGT",
                unknown_token=4,
                ref=ref_mode,
                alt=alt_mode,
            ),
        )
    )
    out = ds[[0, 1], [0, 1]]
    # ref slot
    if ref_mode == "window":
        assert out.ref_window is not None and out.ref is None
    else:
        assert out.ref is not None and out.ref_window is None
    # alt slot
    if alt_mode == "window":
        assert out.alt_window is not None and out.alt is None
    else:
        assert out.alt is not None and out.alt_window is None
    # all present buffers convert to ragged without error
    out.to_ragged()


def test_variant_windows_ref_window_alt_allele_oracle(snap_dataset):
    # The user's case: ref=window, alt=bare allele. Verify both against oracles.
    import awkward as ak
    from genvarloader._dataset._flat_variants import VarWindowOpt
    from genvarloader._dataset._flat_flanks import build_token_lut

    L = 4
    base = snap_dataset.with_settings(rc_neg=False)
    ds = (
        base.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(
                flank_length=L,
                token_alphabet=b"ACGT",
                unknown_token=4,
                ref="window",
                alt="allele",
            ),
        )
    )
    rag = base.with_seqs("variants").with_tracks(False)
    idx = ([0, 1, 2], [0, 1, 2])
    out = ds[idx]
    rv = rag[idx]

    ref = snap_dataset._seqs.reference
    lut, _ = build_token_lut(b"ACGT", 4)
    ploidy = snap_dataset._seqs.genotypes.shape[-2]
    ds_idx, _, _ = snap_dataset._idxer.parse_idx(idx)
    r_idx, _ = np.unravel_index(np.asarray(ds_idx), snap_dataset._idxer.full_shape)
    region_contigs = snap_dataset._full_regions[r_idx, 0]
    counts = np.asarray(ak.flatten(ak.num(rv.start, axis=-1), axis=None))
    starts = np.asarray(ak.flatten(rv.start, axis=None))
    ilens = np.asarray(ak.flatten(rv.ilen, axis=None))
    v_contigs = np.repeat(np.repeat(region_contigs, ploidy), counts)
    ends = starts - np.minimum(ilens, 0) + 1

    # ref window oracle: [start-L, end+L) read tokenized
    rw = ref.fetch(v_contigs, starts - L, ends + L)
    exp_ref = lut[rw.data.view(np.uint8)]
    got_ref_flat = np.asarray(ak.flatten(out.ref_window.to_ragged(), axis=None))
    np.testing.assert_array_equal(got_ref_flat, exp_ref)

    # alt bare allele oracle: tokenized alt bytes (no flanks)
    alt_bytes = np.asarray(ak.flatten(rv.alt, axis=None)).view(np.uint8)
    exp_alt = lut[alt_bytes]
    got_alt_flat = np.asarray(ak.flatten(out.alt.to_ragged(), axis=None))
    np.testing.assert_array_equal(got_alt_flat, exp_alt)


def test_varwindowopt_defaults():
    opt = VarWindowOpt(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)
    assert opt.ref == "window" and opt.alt == "window"
    opt2 = VarWindowOpt(
        flank_length=5,
        token_alphabet=b"ACGT",
        unknown_token=4,
        ref="allele",
        alt="window",
    )
    assert opt2.ref == "allele"


def test_compute_ref_alt_window_split_matches_compute_windows(snap_dataset):
    # compute_windows must equal (compute_ref_window, compute_alt_window)
    from genvarloader._dataset._flat_flanks import compute_windows

    haps = (
        snap_dataset.with_seqs("variants")
        .with_settings(flank_length=3, token_alphabet=b"ACGT", unknown_token=4)
        ._seqs
    )
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
    haps = (
        snap_dataset.with_seqs("variants")
        .with_settings(flank_length=3, token_alphabet=b"ACGT", unknown_token=4)
        ._seqs
    )
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
    from genvarloader._dataset._flat_variants import _FlatWindow
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


def test_public_exports():
    import genvarloader as gvl

    assert hasattr(gvl, "FlatVariantWindows")
    assert hasattr(gvl, "VarWindowOpt")
    assert hasattr(gvl, "FlatVariants")  # from sub-project A
    assert "FlatVariantWindows" in gvl.__all__
    assert "VarWindowOpt" in gvl.__all__
    # VarWindowOpt is constructible and is the documented config object
    opt = gvl.VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4)
    assert opt.ref == "window" and opt.alt == "window"


# ---------------------------------------------------------------------------
# Acceptance tests for flat flank ride-along tokens (mode C)
# ---------------------------------------------------------------------------


def _oracle_flank_from_ragged(dataset, idx, flank_len, lut):
    """Independent oracle: (n_var, 2L) flank tokens for ``dataset[idx]`` in flat
    flank-ride-along mode, computed from the ragged variants output + region contigs."""
    import awkward as ak

    rag = dataset.with_seqs("variants").with_tracks(False)[idx]
    ploidy = dataset._seqs.genotypes.shape[-2]
    ds_idx, _, _ = dataset._idxer.parse_idx(idx)
    r_idx, _ = np.unravel_index(np.asarray(ds_idx), dataset._idxer.full_shape)
    region_contigs = dataset._full_regions[r_idx, 0]
    counts = np.asarray(ak.flatten(ak.num(rag.start, axis=-1), axis=None))
    starts = np.asarray(ak.flatten(rag.start, axis=None))
    ilens = np.asarray(ak.flatten(rag.ilen, axis=None))
    v_contigs = np.repeat(np.repeat(region_contigs, ploidy), counts)
    ref = dataset._seqs.reference
    return _flatten_lut_flanks(ref, v_contigs, starts, ilens, flank_len, lut)


@pytest.mark.parametrize(
    "idx",
    [
        (0, 2),  # scalar (squeezed) — region 0 / sample 2 has 1 variant
        ([1, 2, 3], [0, 1, 2]),  # paired list — each (region, sample) has variants
        ([1, 1], [0, 1]),  # same region, two samples — both have variants
    ],
)
def test_flank_tokens_index_matrix(snap_dataset, idx):
    L = 5
    flat_ds = (
        snap_dataset.with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
        .with_settings(flank_length=L, token_alphabet=b"ACGT", unknown_token=4)
    )
    flat = flat_ds[idx]
    lut = flat_ds._seqs.token_lut
    expected = _oracle_flank_from_ragged(snap_dataset, idx, L, lut)
    got = np.asarray(flat.flank_tokens.to_ragged().data).reshape(
        -1, 2 * L
    )  # (n_var, 2L)
    np.testing.assert_array_equal(got, expected)


def test_oob_flank_padding(snap_dataset):
    """OOB reference positions tokenize to unknown_token (4).

    Region 0 / sample 2 has a variant at start=110 on chr1.  With L=256 the
    flank5 window spans positions [110-256, 110) = [-146, 110).  Positions
    [-146, 0) are outside the contig — the reference reader pads them as 'N'
    which the LUT maps to unknown_token=4.  We assert that:
      - start - L < 0 (confirmed OOB)
      - at least (L - start) tokens in the flank5 column equal 4
    """
    L = 256
    # Region 0, sample 2: variant at start=110 on chr1 (confirmed during exploration)
    flat_ds = (
        snap_dataset.with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
        .with_settings(flank_length=L, token_alphabet=b"ACGT", unknown_token=4)
    )
    import awkward as ak

    rag_ds = snap_dataset.with_seqs("variants").with_tracks(False)
    rag = rag_ds[[0], [2]]
    starts = np.asarray(ak.flatten(rag.start, axis=None))
    assert len(starts) > 0, "expected at least one variant in region 0 / sample 2"
    min_start = int(starts.min())
    # Confirm the variant actually crosses position 0
    assert min_start < L, (
        f"variant start {min_start} is not < L={L}; adjust the region/sample or L"
    )
    n_oob = L - min_start  # positions strictly outside [0, contig_end)

    flat = flat_ds[[0], [2]]
    toks = np.asarray(flat.flank_tokens.to_ragged().data).reshape(
        -1, 2 * L
    )  # (n_var, 2L)
    assert toks.size > 0, "expected non-empty token array"
    # The first n_oob tokens of flank5 for each variant must be unknown_token
    flank5 = toks[:, :L]
    assert (flank5[:, :n_oob] == 4).all(), (
        f"expected first {n_oob} flank5 tokens to be unknown_token=4 for start={min_start} L={L}"
    )


def test_flank_tokens_empty_region_row(snap_dataset):
    """A (region, sample) with zero variants must produce a zero-length row."""
    import awkward as ak

    L = 5
    flat_ds = (
        snap_dataset.with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
        .with_settings(flank_length=L, token_alphabet=b"ACGT", unknown_token=4)
    )
    rag_ds = snap_dataset.with_seqs("variants").with_tracks(False)
    target = None
    for r in range(snap_dataset._full_regions.shape[0]):
        rag = rag_ds[[r], [0]]
        if int(ak.sum(ak.num(rag.start, axis=-1))) == 0:
            target = r
            break
    if target is None:
        pytest.skip("no empty (region, sample) variant set in this fixture")
    flat = flat_ds[[target], [0]]
    rg = flat.flank_tokens.to_ragged()
    # zero variants -> zero rows of flank token pairs
    assert np.asarray(rg.data).size == 0


def test_no_awkward_on_flank_hot_path(snap_dataset, monkeypatch):
    """The flat decode path must not invoke awkward __getitem__ during indexing."""
    import awkward as ak

    calls = {"n": 0}
    orig = ak.highlevel.Array.__getitem__

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(ak.highlevel.Array, "__getitem__", spy)
    flat_ds = (
        snap_dataset.with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
        .with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)
    )
    calls["n"] = 0
    _ = flat_ds[[0, 1, 2], [0, 1, 2]]  # indexing only; do NOT call to_ragged()
    assert calls["n"] == 0, (
        f"awkward __getitem__ was called {calls['n']} time(s) on the flat flank hot path"
    )


def test_variant_windows_rejects_active_tracks(snap_dataset):
    # snap_dataset has an active track; variant-windows + tracks must raise a
    # clear error (not a cryptic AssertionError from the haps+tracks path).
    with pytest.raises(ValueError, match="variant-windows.*tracks"):
        snap_dataset.with_output_format("flat").with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )


def test_with_settings_stores_unknown_token(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_settings(
        flank_length=5, token_alphabet=b"ACGT", unknown_token=4
    )
    assert ds._seqs.unknown_token == 4


def _find_empty_region(snap_dataset):
    import awkward as ak

    rag_ds = snap_dataset.with_seqs("variants").with_tracks(False)
    for r in range(snap_dataset._full_regions.shape[0]):
        rag = rag_ds[[r], [0]]
        if int(ak.sum(ak.num(rag.start, axis=-1))) == 0:
            return r
    return None


def test_dummy_flank_tokens_fills_empty_region_all_unk(snap_dataset):

    target = _find_empty_region(snap_dataset)
    if target is None:
        pytest.skip("no empty (region, sample) variant set in this fixture")
    L = 5
    ploidy = snap_dataset._seqs.genotypes.shape[-2]
    flat_ds = (
        snap_dataset.with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
        .with_settings(flank_length=L, token_alphabet=b"ACGT", unknown_token=4)
        .with_settings(dummy_variant=DummyVariant(start=-1, alt=b"N", ref=b"N"))
    )
    flat = flat_ds[[target], [0]]
    rg = flat.flank_tokens.to_ragged()
    toks = np.asarray(rg.data)
    # one dummy variant per (region, sample, ploid) group => ploidy dummies, each 2L tokens
    assert toks.tolist() == [4] * (ploidy * 2 * L)


def test_dummy_variant_windows_fill_empty_region_all_unk(snap_dataset):
    import awkward as ak

    target = _find_empty_region(snap_dataset)
    if target is None:
        pytest.skip("no empty (region, sample) variant set in this fixture")
    L = 5
    ploidy = snap_dataset._seqs.genotypes.shape[-2]
    opt = VarWindowOpt(flank_length=L, token_alphabet=b"ACGT", unknown_token=4)
    flat_ds = (
        snap_dataset.with_output_format("flat")
        .with_tracks(False)
        .with_seqs("variant-windows", opt)
        .with_settings(dummy_variant=DummyVariant(start=-1, alt=b"N", ref=b"N"))
    )
    flat = flat_ds[[target], [0]]
    # one dummy per (region, sample, ploid) group => ploidy dummies, each 2L+1 unknown tokens
    for w in (flat.ref_window, flat.alt_window):
        vals = np.asarray(ak.flatten(w.to_ragged(), axis=None))
        assert vals.tolist() == [4] * (ploidy * (2 * L + 1))


def test_variant_windows_single_fetch_per_decode(snap_dataset, monkeypatch):
    """ref=window, alt=window decode must call Reference.fetch exactly once."""
    import genvarloader._dataset._reference as refmod
    from genvarloader._dataset._flat_variants import VarWindowOpt

    calls = {"n": 0}
    orig = refmod.Reference.fetch

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(refmod.Reference, "fetch", spy)

    ds = (
        snap_dataset.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    calls["n"] = 0
    out = ds[[0, 1, 2], [0, 1, 2]]
    assert out.ref_window is not None and out.alt_window is not None
    assert calls["n"] == 1, (
        f"expected 1 reference.fetch for both-window decode, got {calls['n']}"
    )


def test_no_awkward_on_dummy_window_hot_path(snap_dataset, monkeypatch):
    import awkward as ak

    target = _find_empty_region(snap_dataset)
    if target is None:
        pytest.skip("no empty (region, sample) variant set in this fixture")
    opt = VarWindowOpt(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)
    flat_ds = (
        snap_dataset.with_output_format("flat")
        .with_tracks(False)
        .with_seqs("variant-windows", opt)
        .with_settings(dummy_variant=DummyVariant(start=-1, alt=b"N", ref=b"N"))
    )
    calls = {"n": 0}
    orig = ak.highlevel.Array.__getitem__

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(ak.highlevel.Array, "__getitem__", spy)
    _ = flat_ds[[target], [0]]  # indexing only; do NOT call to_ragged()
    assert calls["n"] == 0, f"awkward __getitem__ called {calls['n']}x on dummy path"
