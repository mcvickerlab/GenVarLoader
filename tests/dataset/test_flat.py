import numpy as np
from seqpro.rag import Ragged

from genvarloader._flat import _Flat


def _rag(data, shape, offsets):
    return Ragged.from_offsets(data, shape, np.asarray(offsets, np.int64))


def test_to_ragged_roundtrip():
    data = np.arange(6, dtype=np.int32)
    f = _Flat.from_offsets(data, (2, None), np.array([0, 3, 6], np.int64))
    r = f.to_ragged()
    np.testing.assert_array_equal(r.data, data)
    np.testing.assert_array_equal(r.offsets, [0, 3, 6])
    assert r.shape == (2, None)


def test_to_fixed_matches_ragged_to_numpy():
    data = np.arange(8, dtype=np.float32)
    off = np.array([0, 4, 8], np.int64)
    f = _Flat.from_offsets(data, (2, None), off)
    expected = _rag(data, (2, None), off).to_numpy()
    np.testing.assert_array_equal(f.to_fixed(4), expected)


def test_to_fixed_multi_outer():
    data = np.arange(12, dtype=np.float32)
    off = np.arange(0, 13, 2, dtype=np.int64)  # 6 rows of length 2
    f = _Flat.from_offsets(data, (3, 2, None), off)
    out = f.to_fixed(2)
    assert out.shape == (3, 2, 2)
    np.testing.assert_array_equal(out.reshape(-1), data)


def test_to_padded_matches_seqpro():
    data = np.array([1, 2, 3, 4, 5], np.int32)
    off = np.array([0, 2, 5], np.int64)  # rows len 2 and 3
    f = _Flat.from_offsets(data, (2, None), off)
    from genvarloader._ragged import to_padded

    expected = to_padded(_rag(data, (2, None), off), -1)
    np.testing.assert_array_equal(f.to_padded(-1), expected)


def test_view_changes_dtype_not_offsets():
    data = np.zeros(4, np.uint8)
    f = _Flat.from_offsets(data, (2, None), np.array([0, 2, 4], np.int64))
    fv = f.view("S1")
    assert fv.data.dtype == np.dtype("S1")
    np.testing.assert_array_equal(fv.offsets, f.offsets)


def test_squeeze_outer_one():
    data = np.arange(4, dtype=np.int32)
    f = _Flat.from_offsets(data, (1, 2, None), np.array([0, 2, 4], np.int64))
    s = f.squeeze(0)
    assert s.shape == (2, None)
    np.testing.assert_array_equal(s.offsets, f.offsets)


def test_reverse_masked_int_matches_awkward():
    import awkward as ak

    data = np.arange(10, dtype=np.int32)
    off = np.array([0, 3, 6, 10], np.int64)  # 3 rows
    mask = np.array([True, False, True])
    f = _Flat.from_offsets(data.copy(), (3, None), off)
    out = f.reverse_masked(mask)
    # awkward reference: reverse masked rows only
    rag = _rag(data.copy(), (3, None), off)
    expected = ak.to_packed(ak.where(mask, rag[..., ::-1], rag))
    np.testing.assert_array_equal(out.data, ak.flatten(expected, None).to_numpy())
    np.testing.assert_array_equal(out.offsets, off)


def test_reverse_masked_dna_matches_existing():
    from genvarloader._ragged import reverse_complement_masked, _COMP  # noqa

    seq = np.frombuffer(b"ACGTAACCGGTT", dtype="S1")
    off = np.array([0, 4, 12], np.int64)  # 2 rows
    mask = np.array([True, False])
    f = _Flat.from_offsets(seq.view(np.uint8).copy(), (2, None), off)
    out = f.reverse_masked(mask, comp=_COMP).view("S1")
    expected = reverse_complement_masked(_rag(seq.copy(), (2, None), off), mask)
    np.testing.assert_array_equal(out.data, np.asarray(expected.data))


def test_flat_annotated_to_ragged():
    from genvarloader._flat import _Flat, _FlatAnnotatedHaps

    off = np.array([0, 2, 4], np.int64)
    h = _Flat.from_offsets(
        np.frombuffer(b"ACGT", "S1").view(np.uint8).copy(), (2, None), off
    )
    v = _Flat.from_offsets(np.array([0, 1, 2, 3], np.int32), (2, None), off)
    p = _Flat.from_offsets(np.array([10, 11, 12, 13], np.int32), (2, None), off)
    rah = _FlatAnnotatedHaps(h, v, p).to_ragged()
    assert rah.haps.data.dtype == np.dtype("S1")
    np.testing.assert_array_equal(np.asarray(rah.var_idxs.data), [0, 1, 2, 3])
