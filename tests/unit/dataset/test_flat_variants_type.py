from __future__ import annotations

import awkward as ak
import numpy as np

from genvarloader import RaggedVariants
from genvarloader._dataset._flat_variants import _FlatAlleles, _FlatVariants


def _alleles(rows, group_off, ploidy):
    """rows: list[bytes] per variant; group_off: per-(b*p)-row variant boundaries.
    shape is (b, p, None) where b = (len(group_off)-1) // ploidy.
    """
    data = np.frombuffer(b"".join(rows), np.uint8).copy()
    seq_off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    b = (len(group_off) - 1) // ploidy
    return _FlatAlleles(
        byte_data=data,
        seq_offsets=seq_off,
        var_offsets=np.asarray(group_off, np.int64),
        shape=(b, ploidy, None),
    )


def test_flat_variants_to_ragged_matches_handbuilt():
    # b=2, p=1: group_off has 3 entries (2*1+1)
    # row0 has 2 variants, row1 has 1 variant
    group_off = [0, 2, 3]
    ploidy = 1
    alt = _alleles([b"ACG", b"T", b"GG"], group_off, ploidy)
    ref = _alleles([b"A", b"CC", b"T"], group_off, ploidy)
    from genvarloader._flat import _Flat

    start = _Flat.from_offsets(
        np.array([1, 5, 9], np.int32), (2, None), np.asarray(group_off, np.int64)
    )
    fv = _FlatVariants(fields={"alt": alt, "start": start, "ref": ref})
    rv = fv.to_ragged()

    assert isinstance(rv, RaggedVariants)
    # _build_allele_layout with ploidy=1 produces shape (b, p, ~v, ~l) = (2, 1, ~v, ~l)
    assert ak.to_list(rv["alt"]) == [[[b"ACG", b"T"]], [[b"GG"]]]
    assert ak.to_list(rv["ref"]) == [[[b"A", b"CC"]], [[b"T"]]]
    assert ak.to_list(rv["start"]) == [[1, 5], [9]]


def test_flat_variants_squeeze_leading_axis():
    """_FlatVariants.squeeze(0) drops a leading size-1 fixed axis from all fields,
    including _FlatAlleles fields, and to_ragged() still produces correct output."""
    # b=2, p=1 but wrapped in a leading size-1 axis -> shape (1, 2, 1, None)
    # group_off: 1*2*1=2 rows, row0 has 2 variants, row1 has 1 variant
    group_off = [0, 2, 3]
    ploidy = 1

    # Build _FlatAlleles with shape (1, 2, 1, None) — leading size-1 axis
    data_bytes = [b"ACG", b"T", b"GG"]
    byte_data = np.frombuffer(b"".join(data_bytes), np.uint8).copy()
    seq_off = np.concatenate([[0], np.cumsum([len(r) for r in data_bytes])]).astype(np.int64)
    alt = _FlatAlleles(
        byte_data=byte_data,
        seq_offsets=seq_off,
        var_offsets=np.asarray(group_off, np.int64),
        shape=(1, 2, ploidy, None),
    )

    from genvarloader._flat import _Flat

    # _Flat with shape (1, 2, None) — leading size-1 axis
    start = _Flat.from_offsets(
        np.array([1, 5, 9], np.int32), (1, 2, None), np.asarray(group_off, np.int64)
    )

    ref_bytes = [b"A", b"CC", b"T"]
    ref_byte_data = np.frombuffer(b"".join(ref_bytes), np.uint8).copy()
    ref_seq_off = np.concatenate([[0], np.cumsum([len(r) for r in ref_bytes])]).astype(np.int64)
    ref = _FlatAlleles(
        byte_data=ref_byte_data,
        seq_offsets=ref_seq_off,
        var_offsets=np.asarray(group_off, np.int64),
        shape=(1, 2, ploidy, None),
    )

    fv = _FlatVariants(fields={"alt": alt, "start": start, "ref": ref})

    # Before squeeze: shape has leading 1
    assert fv.shape == (1, 2, None)
    assert alt.shape == (1, 2, 1, None)

    # squeeze(0) drops the leading size-1 axis
    fv2 = fv.squeeze(0)
    assert fv2.shape == (2, None), f"expected (2, None), got {fv2.shape}"
    assert fv2.fields["alt"].shape == (2, 1, None), (
        f"expected (2, 1, None), got {fv2.fields['alt'].shape}"
    )

    # Verify squeeze(0) on _FlatAlleles matches the house pattern from _Flat.squeeze(0)
    flat_squeezed = start.squeeze(0)
    assert flat_squeezed.shape == (2, None)

    # to_ragged() still works correctly after squeeze
    rv = fv2.to_ragged()
    assert isinstance(rv, RaggedVariants)
    assert ak.to_list(rv["alt"]) == [[[b"ACG", b"T"]], [[b"GG"]]]
    assert ak.to_list(rv["ref"]) == [[[b"A", b"CC"]], [[b"T"]]]
    assert ak.to_list(rv["start"]) == [[1, 5], [9]]
