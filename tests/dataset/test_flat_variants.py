from __future__ import annotations

import numpy as np
import awkward as ak
import pytest

from genvarloader import RaggedVariants
from genvarloader._dataset._flat_variants import (
    DummyVariant,
    _fill_empty_fixed,
    _fill_empty_seq,
    _FlatAlleles,
    _FlatVariants,
    _FlatVariantWindows,
    _FlatWindow,
)
from genvarloader._flat import _Flat
from genvarloader._dataset._haps import _build_allele_layout
from genvarloader._ragged import reverse_complement  # the awkward reference
from seqpro.rag import Ragged
from seqpro.rag._array import Ragged as _ArrayRagged


def _make_rv(alt_rows, ref_rows, starts, group_off, ploidy):
    """alt_rows/ref_rows: list[bytes] per variant; group_off: variant boundaries per (b*p) row."""

    def alleles(rows):
        data = np.frombuffer(b"".join(rows), np.uint8)
        off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
        return _build_allele_layout(data, off, np.asarray(group_off, np.int64), ploidy)

    alt = alleles(alt_rows)
    ref = alleles(ref_rows)
    start = Ragged.from_offsets(
        np.asarray(starts, np.int32),
        (len(group_off) - 1, None),
        np.asarray(group_off, np.int64),
    )
    return RaggedVariants(alt=alt, start=start, ref=ref)


def _ref_rc(rv, to_rc):
    """Old awkward idiom, computed independently."""
    alt = ak.to_packed(ak.where(to_rc, reverse_complement(rv["alt"]), rv["alt"]))
    ref = ak.to_packed(ak.where(to_rc, reverse_complement(rv["ref"]), rv["ref"]))
    return alt, ref


@pytest.mark.parametrize(
    "mask",
    [
        np.array([True, True]),  # all
        np.array([False, False]),  # none (early return)
        np.array([True, False]),  # mixed
    ],
)
def test_rc_matches_awkward(mask):
    # b=2, p=1, group_off over 2 rows: row0 has 2 variants, row1 has 1
    group_off = [0, 2, 3]
    rv = _make_rv(
        [b"ACG", b"T", b"GG"], [b"A", b"CC", b"T"], [1, 5, 9], group_off, ploidy=1
    )
    exp_alt, exp_ref = _ref_rc(rv, mask)
    rv.rc_(mask)
    assert ak.to_list(rv["alt"]) == ak.to_list(exp_alt)
    assert ak.to_list(rv["ref"]) == ak.to_list(exp_ref)


def test_rc_none_means_all():
    group_off = [0, 2, 3]
    rv = _make_rv(
        [b"ACG", b"T", b"GG"], [b"A", b"CC", b"T"], [1, 5, 9], group_off, ploidy=1
    )
    exp_alt, exp_ref = _ref_rc(rv, np.ones(2, bool))
    rv.rc_(None)
    assert ak.to_list(rv["alt"]) == ak.to_list(exp_alt)
    assert ak.to_list(rv["ref"]) == ak.to_list(exp_ref)


def test_rc_ploidy2_broadcast():
    """ploidy=2: np.repeat(to_rc, ploidy) must broadcast the per-batch mask across both
    haplotypes.  b=2, p=2 → group_off has 5 entries (b*p + 1).
    Row layout: batch0/hap0=[v0,v1], batch0/hap1=[v2], batch1/hap0=[v3], batch1/hap1=[v4].
    to_rc=[True, False] → batch0 (haps 0 and 1) gets RC'd; batch1 (haps 0 and 1) stays.

    Note: _make_rv passes shape (b*p, None) to Ragged.from_offsets which is incompatible
    with ploidy=2 allele arrays of shape (b, p, ~v, ~l).  We build the RaggedVariants
    directly here with shape (b, p, None) for start so that ak.zip inside __init__ works.
    """
    # b=2, p=2 → 4 rows; group_off cumsum of [2, 1, 1, 1]
    group_off = np.array([0, 2, 3, 4, 5], np.int64)
    alt_rows = [b"ACG", b"T", b"GG", b"CA", b"AT"]
    ref_rows = [b"A", b"CC", b"T", b"G", b"C"]
    starts = [1, 5, 9, 2, 7]
    ploidy = 2

    def alleles(rows):
        data = np.frombuffer(b"".join(rows), np.uint8)
        off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
        return _build_allele_layout(data, off, group_off, ploidy)

    alt = alleles(alt_rows)
    ref = alleles(ref_rows)
    start = Ragged.from_offsets(np.asarray(starts, np.int32), (2, 2, None), group_off)
    rv = RaggedVariants(alt=alt, start=start, ref=ref)

    to_rc = np.array([True, False])
    # capture reference BEFORE in-place mutation
    exp_alt, exp_ref = _ref_rc(rv, to_rc)
    rv.rc_(to_rc)
    assert ak.to_list(rv["alt"]) == ak.to_list(exp_alt)
    assert ak.to_list(rv["ref"]) == ak.to_list(exp_ref)


def test_to_packed_matches_awkward_contiguous():
    group_off = [0, 2, 3]
    rv = _make_rv(
        [b"ACG", b"T", b"GG"], [b"A", b"CC", b"T"], [1, 5, 9], group_off, ploidy=1
    )
    exp = ak.to_packed(ak.Array(rv))  # old behavior
    got = rv.to_packed()
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
    np.testing.assert_array_equal(
        np.asarray(got["start"].data), np.asarray(exp["start"].data)
    )


def test_to_packed_ploidy2():
    """ploidy=2: the RegularArray(..., ploidy) rebuild and rebased_group/ploidy interaction
    must produce byte-identical output to ak.to_packed for both alt and ref, and
    array-identical data for start.  Uses DIFFERENT allele bytes in the two batches so
    a wrong ploidy stride would produce a detectable mismatch.

    b=2, p=2 → group_off has b*p+1=5 entries.
    Row layout: batch0/hap0=[v0,v1], batch0/hap1=[v2], batch1/hap0=[v3], batch1/hap1=[v4].
    """
    group_off = np.array([0, 2, 3, 4, 5], np.int64)
    # batch0 uses A/C/G bytes; batch1 uses T bytes → different per batch
    alt_rows = [b"ACG", b"T", b"GG", b"CA", b"AT"]
    ref_rows = [b"A", b"CC", b"T", b"G", b"TT"]
    starts = [1, 5, 9, 2, 7]
    ploidy = 2

    def alleles(rows):
        data = np.frombuffer(b"".join(rows), np.uint8)
        off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
        return _build_allele_layout(data, off, group_off, ploidy)

    alt = alleles(alt_rows)
    ref = alleles(ref_rows)
    start = Ragged.from_offsets(np.asarray(starts, np.int32), (2, 2, None), group_off)
    rv = RaggedVariants(alt=alt, start=start, ref=ref)

    exp = ak.to_packed(ak.Array(rv))
    got = rv.to_packed()

    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
    np.testing.assert_array_equal(
        np.asarray(got["start"].data), np.asarray(exp["start"].data)
    )

    # Also verify sliced ploidy=2: drop the first batch row and repack
    sliced = rv[1:]
    exp_sliced = ak.to_packed(ak.Array(sliced))
    got_sliced = sliced.to_packed()

    assert ak.to_list(got_sliced["alt"]) == ak.to_list(exp_sliced["alt"])
    assert ak.to_list(got_sliced["ref"]) == ak.to_list(exp_sliced["ref"])
    np.testing.assert_array_equal(
        np.asarray(got_sliced["start"].data), np.asarray(exp_sliced["start"].data)
    )


def test_to_packed_matches_awkward_sliced():
    # a sliced RaggedVariants has non-zero-based / scattered offsets -> to_packed must contiguate
    group_off = [0, 2, 3, 5]
    rv = _make_rv(
        [b"ACG", b"T", b"GG", b"AA", b"C"],
        [b"A", b"CC", b"T", b"G", b"TT"],
        [1, 5, 9, 12, 20],
        group_off,
        ploidy=1,
    )
    sliced = rv[
        1:
    ]  # drop the first (b,p) row; rv[1:] preserves RaggedVariants subclass
    exp = ak.to_packed(ak.Array(sliced))
    got = sliced.to_packed()
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
    np.testing.assert_array_equal(
        np.asarray(got["start"].data), np.asarray(exp["start"].data)
    )


def test_to_packed_numeric_field_reorders_through_indexed_view():
    # After fancy-indexing a RaggedVariants, its numeric `start` field is a Ragged
    # backed by an IndexedArray layout. seqpro 0.15.1's IndexedArray unbox fix lets
    # `.to_packed()` materialize + reorder it correctly with no gvl change. The full
    # RaggedVariants.to_packed() over alt/ref on non-canonical views lands in a later
    # task; this isolates the numeric path.
    group_off = [0, 2, 3, 5]
    rv = _make_rv(
        [b"A", b"C", b"G", b"T", b"N"],
        [b"a", b"c", b"g", b"t", b"n"],
        [10, 20, 30, 40, 50],
        group_off,
        ploidy=1,
    )
    fancy = RaggedVariants.from_ak(rv[np.array([2, 0])])
    start = fancy["start"]
    assert isinstance(start, (Ragged, _ArrayRagged))
    from awkward.contents import IndexedArray

    assert isinstance(start.layout, IndexedArray)
    got = start.to_packed()
    exp = ak.to_packed(ak.Array(fancy)["start"])
    assert ak.to_list(got) == ak.to_list(exp)


def test_pack_alleles_kernel_identity_and_reorder():
    from genvarloader._dataset._rag_variants import _pack_alleles

    # 3 variant rows, leaf "ACGTGG", alleles [ACG, T, GG]; rows: [v0,v1],[v2]
    leaf = np.frombuffer(b"ACGTGG", np.uint8)
    allele_starts = np.array([0, 3, 4], np.int64)
    allele_stops = np.array([3, 4, 6], np.int64)
    var_starts = np.array(
        [0, 2], np.int64
    )  # row0 -> alleles[0:2], row1 -> alleles[2:3]
    var_stops = np.array([2, 3], np.int64)

    # identity order
    packed, allele_off, group_off = _pack_alleles(
        np.array([0, 1], np.int64),
        var_starts,
        var_stops,
        allele_starts,
        allele_stops,
        leaf,
    )
    assert bytes(packed) == b"ACGTGG"
    assert allele_off.tolist() == [0, 3, 4, 6]
    assert group_off.tolist() == [0, 2, 3]

    # reversed row order
    packed, allele_off, group_off = _pack_alleles(
        np.array([1, 0], np.int64),
        var_starts,
        var_stops,
        allele_starts,
        allele_stops,
        leaf,
    )
    assert bytes(packed) == b"GGACGT"
    assert allele_off.tolist() == [0, 2, 5, 6]
    assert group_off.tolist() == [0, 1, 3]


def test_is_canonical_alleles():
    from genvarloader._dataset._rag_variants import _is_canonical_alleles

    rv = _make_rv([b"A", b"C"], [b"a", b"c"], [1, 2], [0, 1, 2], ploidy=1)
    assert _is_canonical_alleles(rv["alt"].layout) is True
    fancy = RaggedVariants.from_ak(rv[np.array([1, 0])])
    assert _is_canonical_alleles(fancy["alt"].layout) is False


def test_decompose_alleles_reversed():
    from genvarloader._dataset._rag_variants import _decompose_alleles, _pack_alleles

    rv = _make_rv(
        [b"A", b"C", b"G", b"T", b"N"],
        [b"a", b"c", b"g", b"t", b"n"],
        [1, 2, 3, 4, 5],
        [0, 2, 3, 5],
        ploidy=1,
    )
    fancy = RaggedVariants.from_ak(rv[np.array([2, 0])])
    row_src, var_starts, var_stops, allele_starts, allele_stops, leaf, ploidy = (
        _decompose_alleles(fancy["alt"])
    )
    assert ploidy == 1
    packed, allele_off, group_off = _pack_alleles(
        row_src, var_starts, var_stops, allele_starts, allele_stops, leaf
    )
    from genvarloader._dataset._haps import _build_allele_layout

    rebuilt = _build_allele_layout(packed, allele_off, group_off, ploidy)
    assert ak.to_list(rebuilt) == ak.to_list(fancy["alt"])


@pytest.mark.parametrize("transform", ["reverse", "fancy"])
def test_to_packed_alt_ref_on_lazy_views(transform):
    # 4 rows, 6 variants total: group_off=[0,2,3,5,6]
    group_off = [0, 2, 3, 5, 6]
    rv = _make_rv(
        [b"ACG", b"T", b"GG", b"AA", b"C", b"TTT"],
        [b"A", b"CC", b"T", b"G", b"TT", b"C"],
        [1, 5, 9, 12, 20, 25],
        group_off,
        ploidy=1,
    )
    view = rv[::-1] if transform == "reverse" else rv[np.array([2, 0, 3, 1])]
    view = RaggedVariants.from_ak(view)
    got = view.to_packed()
    exp = ak.to_packed(ak.Array(view))
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
    np.testing.assert_array_equal(
        np.asarray(got["start"].data), np.asarray(exp["start"].data)
    )


@pytest.mark.parametrize("transform", ["reverse", "fancy"])
def test_rc_on_lazy_views_matches_reference(transform):
    group_off = [0, 2, 3, 5, 6]
    rv = _make_rv(
        [b"ACG", b"T", b"GG", b"AA", b"C", b"TTT"],
        [b"A", b"CC", b"T", b"G", b"TT", b"C"],
        [1, 5, 9, 12, 20, 25],
        group_off,
        ploidy=1,
    )
    view = rv[::-1] if transform == "reverse" else rv[np.array([2, 0, 3, 1])]
    view = RaggedVariants.from_ak(view)

    n = view.shape[0]
    mask = np.ones(n, np.bool_)
    exp_alt, exp_ref = _ref_rc(
        view, mask
    )  # independent awkward reference (defined at top of file)

    out = view.rc_(mask)
    assert ak.to_list(out["alt"]) == ak.to_list(exp_alt)
    assert ak.to_list(out["ref"]) == ak.to_list(exp_ref)


def test_to_packed_ploidy2_reordered():
    # b=2, p=2 -> 4 (b*p) rows; variant counts per row = [2, 1, 1, 1] -> 5 variants.
    # alt/ref carry exactly ONE allele per variant, so n_alleles == n_variants == 5.
    group_off = np.array([0, 2, 3, 4, 5], np.int64)
    # alt alleles: ["AC","G","T","GG","A"] -> b"ACGTGGA" (lengths 2,1,1,2,1)
    alt = _build_allele_layout(
        np.frombuffer(b"ACGTGGA", np.uint8),
        np.array([0, 2, 3, 4, 6, 7], np.int64),
        group_off,
        ploidy=2,
    )
    # ref alleles: ["a","c","g","t","n"] -> b"acgtn"
    ref = _build_allele_layout(
        np.frombuffer(b"acgtn", np.uint8),
        np.array([0, 1, 2, 3, 4, 5], np.int64),
        group_off,
        ploidy=2,
    )
    start = Ragged.from_offsets(
        np.array([1, 2, 3, 4, 5], np.int32), (2, 2, None), group_off
    )
    rv = RaggedVariants(alt=alt, start=start, ref=ref)
    fancy = RaggedVariants.from_ak(rv[np.array([1, 0])])  # swap the two batches
    got = fancy.to_packed()
    exp = ak.to_packed(ak.Array(fancy))
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])


def test_rc_on_lazy_view_mixed_mask():
    # A mixed (not all-True) mask on a reordered (non-canonical) view exercises
    # mask/row alignment through the to_packed() materialize-then-recurse path.
    group_off = [0, 2, 3, 5, 6]
    rv = _make_rv(
        [b"ACG", b"T", b"GG", b"AA", b"C", b"TTT"],
        [b"A", b"CC", b"T", b"G", b"TT", b"C"],
        [1, 5, 9, 12, 20, 25],
        group_off,
        ploidy=1,
    )
    view = RaggedVariants.from_ak(rv[np.array([2, 0, 3, 1])])
    mask = np.array([True, False, True, False])
    exp_alt, exp_ref = _ref_rc(view, mask)
    out = view.rc_(mask)
    assert ak.to_list(out["alt"]) == ak.to_list(exp_alt)
    assert ak.to_list(out["ref"]) == ak.to_list(exp_ref)


def test_to_packed_explicit_listarray_variant_level():
    # Hand-build a variant-level ListArray (starts/stops), as the user bug report hit.
    from awkward.contents import ListArray, ListOffsetArray, RegularArray, NumpyArray
    from awkward.index import Index

    def listarray_alleles(joined_bytes, allele_off, starts, stops):
        leaf = NumpyArray(
            np.frombuffer(joined_bytes, np.uint8), parameters={"__array__": "byte"}
        )
        allele = ListOffsetArray(
            Index(np.asarray(allele_off, np.int64)),
            leaf,
            parameters={"__array__": "bytestring"},
        )
        var = ListArray(
            Index(np.asarray(starts, np.int64)),
            Index(np.asarray(stops, np.int64)),
            allele,
        )
        return ak.Array(RegularArray(var, 1))

    alt = listarray_alleles(b"ACGTGG", [0, 3, 4, 6], [0, 2], [2, 3])
    ref = listarray_alleles(b"ACCT", [0, 1, 3, 4], [0, 2], [2, 3])
    start = Ragged.from_offsets(
        np.array([1, 5, 9], np.int32), (2, None), np.array([0, 2, 3], np.int64)
    )
    rv = RaggedVariants(alt=alt, start=start, ref=ref)

    got = rv.to_packed()
    exp = ak.to_packed(ak.Array(rv))
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])


def test_fill_empty_fixed_inserts_unk_block_for_empty_rows():
    inner = 2  # pretend 2L = 2
    # 2 rows: row0 empty, row1 has one variant (2 tokens).
    data = np.array([7, 8], np.int32)  # row1 variant: tokens [7, 8]
    offsets = np.array([0, 0, 1], np.int64)  # row0 [0,0) empty; row1 [0,1)
    nd, noff = _fill_empty_fixed(data, offsets, inner, 4)
    assert nd.dtype == np.int32
    # row0 gets one dummy variant -> variant counts [1, 1]
    assert noff.tolist() == [0, 1, 2]
    # row0's dummy block is [4, 4]; row1 unchanged [7, 8]
    assert nd.tolist() == [4, 4, 7, 8]


def test_fill_empty_seq_preserves_int32_dtype_and_fills_unk():
    # Two (b*p) rows: row 0 empty, row 1 has one 2-token variant.
    data = np.array([7, 8], np.int32)  # row 1's single variant tokens
    var_offsets = np.array([0, 0, 1], np.int64)  # row0: [0,0) empty; row1: [0,1)
    seq_offsets = np.array([0, 2], np.int64)  # variant 0 spans data[0:2]
    dummy = np.array([4, 4, 4], np.int32)  # all-unk window, len 3
    nd, nvar, nseq = _fill_empty_seq(data, var_offsets, seq_offsets, dummy)
    assert nd.dtype == np.int32
    # row0 got one dummy variant of length 3; row1 unchanged (one 2-token variant)
    assert nvar.tolist() == [0, 1, 2]
    assert nseq.tolist() == [0, 3, 5]
    assert nd.tolist() == [4, 4, 4, 7, 8]


def _win(data, seq_off, var_off):
    return _FlatWindow(
        np.asarray(data, np.int32),
        np.asarray(seq_off, np.int64),
        np.asarray(var_off, np.int64),
        (1, 1, None, None),
    )


def test_flatvariantwindows_fill_empty_groups_all_unk():
    # 2 (b*p) rows: row0 empty, row1 has one variant.
    # scalar start: row0 empty, row1 has start=100
    start = _Flat.from_offsets(
        np.array([100], np.int32), (1, 1, None), np.array([0, 0, 1], np.int64)
    )
    # alt_window for row1's single variant: a length-3 window [5,6,7]
    aw = _win([5, 6, 7], [0, 3], [0, 0, 1])
    win = _FlatVariantWindows({"start": start}, alt_window=aw)

    dummy = DummyVariant(start=-1, alt=b"N")  # 1-byte alt
    L = 5
    out = win.fill_empty_groups(dummy, unk=4, flank_length=L)

    # scalar: row0 filled with start=-1
    s = out.fields["start"]
    assert s.offsets.tolist() == [0, 1, 2]
    assert s.data.tolist() == [-1, 100]
    # alt_window: row0 dummy window len 2L + len("N") = 11, all unk=4
    w = out.alt_window
    assert w.var_offsets.tolist() == [0, 1, 2]
    assert w.seq_offsets.tolist() == [0, 11, 14]  # dummy(11) then row1's window(3)
    assert w.data[:11].tolist() == [4] * 11
    assert w.data[11:].tolist() == [5, 6, 7]
    assert w.data.dtype == np.int32


def test_flatvariants_fill_empty_groups_fills_flank_tokens():
    # 2 (b*p) rows: row0 empty, row1 one variant.
    start = _Flat.from_offsets(
        np.array([100], np.int32), (1, 1, None), np.array([0, 0, 1], np.int64)
    )
    alt = _FlatAlleles(
        np.frombuffer(b"A", np.uint8).copy(),
        np.array([0, 1], np.int64),  # seq offsets
        np.array([0, 0, 1], np.int64),  # var offsets (row0 empty)
        (1, 1, None),
    )
    fv = _FlatVariants({"start": start, "alt": alt})
    L = 3
    # flank_tokens: row1's single variant has 2L=6 tokens; shape carries 2L inner.
    fv.flank_tokens = _Flat(
        np.arange(6, dtype=np.int32),
        np.array([0, 0, 1], np.int64),
        (1, 1, None, 2 * L),
    )

    dummy = DummyVariant(start=-1, alt=b"N")
    out = fv.fill_empty_groups(dummy, unk=4)

    # flank_tokens row0 gets a 2L=6 run of unk; row1 unchanged.
    ft = out.flank_tokens
    assert ft.offsets.tolist() == [0, 1, 2]
    assert ft.data[:6].tolist() == [4] * 6
    assert ft.data[6:].tolist() == list(range(6))
    assert ft.data.dtype == np.int32
    # scalar still filled
    assert out.fields["start"].data.tolist() == [-1, 100]
