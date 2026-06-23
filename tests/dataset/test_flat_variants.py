from __future__ import annotations

import numpy as np
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
from seqpro.rag import Ragged

_COMP_TABLE = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")


def _rc_bytes(seq: bytes) -> bytes:
    """Pure-Python reverse-complement of a bytestring."""
    return seq.translate(_COMP_TABLE)[::-1]


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


def _rv_alt_list(rv: RaggedVariants):
    """Return alt alleles as nested Python list (bytes leaves) from a _core.Ragged."""
    return rv.alt.to_ak().to_list()


def _rv_ref_list(rv: RaggedVariants):
    """Return ref alleles as nested Python list (bytes leaves) from a _core.Ragged."""
    return rv.ref.to_ak().to_list()


def _ref_rc(rv: RaggedVariants, to_rc):
    """Pure-Python reference oracle: reverse-complement rows selected by ``to_rc``.

    Works on _core.Ragged (opaque-string); uses .to_ak().to_list() to materialise
    as nested Python list then applies per-row byte RC independently of production code.
    """
    rows_alt = _rv_alt_list(rv)
    rows_ref = _rv_ref_list(rv)
    # to_rc may be shorter than rows_alt when ploidy > 1; broadcast per-batch mask
    n = len(rows_alt)
    m = len(to_rc)
    if m < n:
        repeat = n // m
        to_rc_expanded = np.repeat(to_rc, repeat)
    else:
        to_rc_expanded = np.asarray(to_rc, dtype=bool)
    exp_alt = [
        [_rc_bytes(a) for a in row] if flip else row
        for row, flip in zip(rows_alt, to_rc_expanded)
    ]
    exp_ref = [
        [_rc_bytes(r) for r in row] if flip else row
        for row, flip in zip(rows_ref, to_rc_expanded)
    ]
    return exp_alt, exp_ref


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
    out = rv.rc_(mask)
    assert _rv_alt_list(out) == exp_alt
    assert _rv_ref_list(out) == exp_ref


def test_rc_none_means_all():
    group_off = [0, 2, 3]
    rv = _make_rv(
        [b"ACG", b"T", b"GG"], [b"A", b"CC", b"T"], [1, 5, 9], group_off, ploidy=1
    )
    exp_alt, exp_ref = _ref_rc(rv, np.ones(2, bool))
    out = rv.rc_(None)
    assert _rv_alt_list(out) == exp_alt
    assert _rv_ref_list(out) == exp_ref


def test_rc_ploidy2_broadcast():
    """ploidy=2: np.repeat(to_rc, ploidy) must broadcast the per-batch mask across both
    haplotypes.  b=2, p=2 → group_off has 5 entries (b*p + 1).
    Row layout: batch0/hap0=[v0,v1], batch0/hap1=[v2], batch1/hap0=[v3], batch1/hap1=[v4].
    to_rc=[True, False] → batch0 (haps 0 and 1) gets RC'd; batch1 (haps 0 and 1) stays.

    Note: _make_rv passes shape (b*p, None) to Ragged.from_offsets which is incompatible
    with ploidy=2 allele arrays of shape (b, p, ~v, ~l).  We build the RaggedVariants
    directly here with shape (b, p, None) for start so that Ragged.from_fields inside __init__ works.
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
    # capture reference oracle before calling rc_
    exp_alt, exp_ref = _ref_rc(rv, to_rc)
    out = rv.rc_(to_rc)
    assert _rv_alt_list(out) == exp_alt
    assert _rv_ref_list(out) == exp_ref


def test_to_packed_contiguous():
    """to_packed() on a canonical (contiguous, zero-based) RaggedVariants preserves
    content AND produces zero-based, contiguous offsets.

    Input: b=2, p=1.  Row 0 has variants [ACG, T]; row 1 has [GG].
    Expected alt content (hand-derived):  [[b'ACG', b'T'], [b'GG']]
    Expected ref content (hand-derived):  [[b'A', b'CC'], [b'T']]
    Expected starts (hand-derived):       [1, 5, 9]
    The packing invariant: offsets start at 0 and the final offset equals the
    total variant count (2+1=3), confirming the buffer is compact and zero-based.
    """
    group_off = [0, 2, 3]
    rv = _make_rv(
        [b"ACG", b"T", b"GG"], [b"A", b"CC", b"T"], [1, 5, 9], group_off, ploidy=1
    )
    got = rv.to_packed()

    # Content: hand-written expected values, NOT derived from calling to_packed().
    assert _rv_alt_list(got) == [[b"ACG", b"T"], [b"GG"]]
    assert _rv_ref_list(got) == [[b"A", b"CC"], [b"T"]]
    np.testing.assert_array_equal(
        np.asarray(got.start.data), np.array([1, 5, 9], np.int32)
    )

    # Packing invariant: offsets must be zero-based and compact (last == n_variants).
    packed_offsets = np.asarray(got.start.offsets)
    assert packed_offsets[0] == 0, "packed offsets must start at 0"
    assert packed_offsets[-1] == 3, "packed last offset must equal total variant count"


def test_to_packed_ploidy2():
    """ploidy=2: to_packed must produce byte-identical alt/ref content.
    b=2, p=2 → group_off has b*p+1=5 entries."""
    group_off = np.array([0, 2, 3, 4, 5], np.int64)
    # batch0 uses A/C/G bytes; batch1 uses T bytes
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

    got = rv.to_packed()
    assert _rv_alt_list(got) == _rv_alt_list(rv)
    assert _rv_ref_list(got) == _rv_ref_list(rv)
    np.testing.assert_array_equal(
        np.asarray(got.start.data), np.asarray(starts, np.int32)
    )

    # Also verify sliced ploidy=2: drop the first batch row and repack
    sliced = rv[1:]
    got_sliced = sliced.to_packed()
    assert _rv_alt_list(got_sliced) == _rv_alt_list(sliced)
    assert _rv_ref_list(got_sliced) == _rv_ref_list(sliced)


def test_to_packed_sliced():
    """to_packed() on a sliced (non-zero-based) RaggedVariants contiguates correctly."""
    group_off = [0, 2, 3, 5]
    rv = _make_rv(
        [b"ACG", b"T", b"GG", b"AA", b"C"],
        [b"A", b"CC", b"T", b"G", b"TT"],
        [1, 5, 9, 12, 20],
        group_off,
        ploidy=1,
    )
    sliced = rv[1:]
    got = sliced.to_packed()
    assert _rv_alt_list(got) == _rv_alt_list(sliced)
    assert _rv_ref_list(got) == _rv_ref_list(sliced)
    np.testing.assert_array_equal(
        np.asarray(got.start.data), np.asarray([9, 12, 20], np.int32)
    )


@pytest.mark.parametrize("transform", ["reverse", "fancy"])
def test_to_packed_alt_ref_on_lazy_views_p(transform):
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
    # view is already a RaggedVariants (from __getitem__) — no from_ak needed
    got = view.to_packed()
    # packed should preserve same logical content as view
    assert _rv_alt_list(got) == _rv_alt_list(view)
    assert _rv_ref_list(got) == _rv_ref_list(view)
    np.testing.assert_array_equal(
        np.asarray(got.start.data), np.asarray(view.start.to_packed().data)
    )


def test_to_packed_ploidy2_reordered():
    # b=2, p=2 -> 4 (b*p) rows; variant counts per row = [2, 1, 1, 1] -> 5 variants.
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
    # swap the two batches (slicing returns RaggedVariants directly)
    fancy = rv[np.array([1, 0])]
    got = fancy.to_packed()
    # packed content must match the (reordered) logical view
    assert _rv_alt_list(got) == _rv_alt_list(fancy)
    assert _rv_ref_list(got) == _rv_ref_list(fancy)


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
