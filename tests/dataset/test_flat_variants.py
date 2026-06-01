from __future__ import annotations

import numpy as np
import awkward as ak
import pytest

from genvarloader import RaggedVariants
from genvarloader._ragged import reverse_complement  # the awkward reference
from seqpro.rag import Ragged
from genvarloader._dataset._haps import _build_allele_layout


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
