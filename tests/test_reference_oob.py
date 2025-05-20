import numpy as np
from genvarloader._dataset._reference import Reference, get_reference


def make_ref():
    seq = np.frombuffer(b"ACGT", np.uint8)
    offsets = np.array([0, len(seq)], np.uint64)
    return Reference(seq, ["chr1"], offsets, ord("N"))


def test_fetch_out_of_bounds_right():
    ref = make_ref()
    seq = ref.fetch("chr1", 6, 10)
    np.testing.assert_equal(seq, np.frombuffer(b"NNNN", "S1"))


def test_fetch_entirely_left():
    ref = make_ref()
    seq = ref.fetch("chr1", -5, -1)
    np.testing.assert_equal(seq, np.frombuffer(b"NNNN", "S1"))


def test_get_reference_out_of_bounds():
    ref = make_ref()
    regions = np.array([[0, 6, 10, 1]], np.int32)
    out_offsets = np.array([0, 4], np.int64)
    out = get_reference(regions, out_offsets, ref.reference, ref.offsets, ref.pad_char)
    np.testing.assert_equal(out.view("S1"), np.frombuffer(b"NNNN", "S1"))

