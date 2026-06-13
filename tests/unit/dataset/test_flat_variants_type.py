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
    seq_off = np.concatenate([[0], np.cumsum([len(r) for r in data_bytes])]).astype(
        np.int64
    )
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
    ref_seq_off = np.concatenate([[0], np.cumsum([len(r) for r in ref_bytes])]).astype(
        np.int64
    )
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


def test_compact_keep_compacts_v_idxs_and_offsets():
    """_compact_keep drops masked-out variants per row and rebuilds offsets.
    This is the AF-filter compaction kernel; exercised here in isolation since
    snap_dataset (a phased VCF) cannot carry an AF cache for integration."""
    from genvarloader._dataset._flat_variants import _compact_keep

    # 2 rows: row0 has variants [10,11,12], row1 has [20,21]
    v_idxs = np.array([10, 11, 12, 20, 21], np.int32)
    row_offsets = np.array([0, 3, 5], np.int64)
    # keep middle of row0 dropped, all of row1
    keep = np.array([True, False, True, True, True], np.bool_)

    new_v, new_off = _compact_keep(v_idxs, row_offsets, keep)
    np.testing.assert_array_equal(new_v, [10, 12, 20, 21])
    np.testing.assert_array_equal(new_off, [0, 2, 4])


def test_compact_keep_used_for_dosage_values():
    """When compacting dosage, _compact_keep is called with the dosage VALUES in
    place of v_idxs and the UNFILTERED row offsets — verify it gathers correctly."""
    from genvarloader._dataset._flat_variants import _compact_keep

    dos = np.array([0.5, 0.1, 0.9, 0.2, 0.8], np.float32)
    row_offsets = np.array([0, 3, 5], np.int64)
    keep = np.array([True, False, True, True, True], np.bool_)

    new_dos, new_off = _compact_keep(dos, row_offsets, keep)
    np.testing.assert_array_equal(new_dos, np.array([0.5, 0.9, 0.2, 0.8], np.float32))
    np.testing.assert_array_equal(new_off, [0, 2, 4])
    assert new_dos.dtype == np.float32


def test_flat_alleles_to_ragged_multidim_outer():
    """_FlatAlleles.to_ragged() rebuilds the full regular-axis stack for shapes
    with more than one leading fixed dim (b, s, p, ~v, ~l)."""
    # shape (2, 1, 2, None): b=2, s=1, p=2 -> 4 b*p rows.
    group_off = [0, 1, 1, 2, 2]  # rows: [v], [], [v], []
    alt = _alleles([b"AC", b"GGG"], group_off, ploidy=2)
    alt = _FlatAlleles(alt.byte_data, alt.seq_offsets, alt.var_offsets, (2, 1, 2, None))
    rv = alt.to_ragged()
    assert ak.to_list(rv) == [[[[b"AC"], []]], [[[b"GGG"], []]]]


def test_public_flat_exports():
    import genvarloader as gvl

    assert gvl.FlatRagged is not None
    assert gvl.FlatAnnotatedHaps is not None
    assert gvl.FlatVariants is not None
    assert gvl.FlatAlleles is not None
    # aliases point at the existing internals
    from genvarloader._flat import _Flat, _FlatAnnotatedHaps
    from genvarloader._dataset._flat_variants import _FlatVariants, _FlatAlleles

    assert gvl.FlatRagged is _Flat
    assert gvl.FlatAnnotatedHaps is _FlatAnnotatedHaps
    assert gvl.FlatVariants is _FlatVariants
    assert gvl.FlatAlleles is _FlatAlleles


def test_fill_empty_scalar_kernel():
    from genvarloader._dataset._flat_variants import _fill_empty_scalar

    data = np.array([10, 11, 20], np.int32)
    offsets = np.array([0, 0, 2, 2, 3], np.int64)  # rows: empty, [10,11], empty, [20]
    new_data, new_off = _fill_empty_scalar(data, offsets, np.int32(-1))
    assert new_off.tolist() == [0, 1, 3, 4, 5]
    assert new_data.tolist() == [-1, 10, 11, -1, 20]


def test_fill_empty_seq_kernel():
    from genvarloader._dataset._flat_variants import _fill_empty_seq

    # 3 rows: empty, ["AC","G"], empty
    data = np.frombuffer(b"ACG", np.uint8).copy()
    var_off = np.array([0, 0, 2, 2], np.int64)      # per-row variant boundaries
    seq_off = np.array([0, 2, 3], np.int64)         # per-variant byte boundaries
    dummy = np.frombuffer(b"N", np.uint8).copy()
    nd, nvar, nseq = _fill_empty_seq(data, var_off, seq_off, dummy)
    assert nvar.tolist() == [0, 1, 3, 4]            # each empty row gains 1 variant
    assert nseq.tolist() == [0, 1, 3, 4, 5]         # dummy(1) AC(2) G(1) dummy(1)
    assert bytes(nd) == b"NACGN"


def test_fill_empty_groups_roundtrip():
    import awkward as ak

    from genvarloader._dataset._flat_variants import DummyVariant, _FlatAlleles, _FlatVariants
    from genvarloader._flat import _Flat

    # b*p = 3 rows: row0 empty, row1 has [b"AC", b"G"], row2 empty
    group_off = np.array([0, 0, 2, 2], np.int64)
    alt = _FlatAlleles(
        byte_data=np.frombuffer(b"ACG", np.uint8).copy(),
        seq_offsets=np.array([0, 2, 3], np.int64),
        var_offsets=group_off.copy(),
        shape=(3, None),
    )
    start = _Flat.from_offsets(np.array([5, 9], np.int32), (3, None), group_off.copy())
    ilen = _Flat.from_offsets(np.array([1, -1], np.int32), (3, None), group_off.copy())
    fv = _FlatVariants(fields={"alt": alt, "start": start, "ilen": ilen})
    filled = fv.fill_empty_groups(DummyVariant(start=-1, alt=b"N"))
    rv = filled.to_ragged()
    # empty rows now hold exactly the dummy; non-empty row unchanged
    assert ak.to_list(rv["alt"]) == [[b"N"], [b"AC", b"G"], [b"N"]]
    assert ak.to_list(rv["start"]) == [[-1], [5, 9], [-1]]


def test_fill_empty_groups_noop_when_no_empties():
    import awkward as ak

    from genvarloader._dataset._flat_variants import DummyVariant, _FlatAlleles, _FlatVariants
    from genvarloader._flat import _Flat

    group_off = np.array([0, 1, 2], np.int64)  # every row has 1 variant
    alt = _FlatAlleles(np.frombuffer(b"AG", np.uint8).copy(),
                       np.array([0, 1, 2], np.int64), group_off.copy(), (2, None))
    start = _Flat.from_offsets(np.array([3, 7], np.int32), (2, None), group_off.copy())
    ilen = _Flat.from_offsets(np.array([0, 0], np.int32), (2, None), group_off.copy())
    fv = _FlatVariants(fields={"alt": alt, "start": start, "ilen": ilen})
    filled = fv.fill_empty_groups(DummyVariant())
    assert ak.to_list(filled.to_ragged()["alt"]) == [[b"A"], [b"G"]]
    assert ak.to_list(filled.to_ragged()["start"]) == [[3], [7]]
