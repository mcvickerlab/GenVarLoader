import numpy as np
from genvarloader._dataset._reconstruct import Haps, Reference, _Variants, RaggedVariants
from genvarloader._variants._records import RaggedAlleles
from genvarloader._dataset._genotypes import SparseGenotypes
from genvarloader._utils import _lengths_to_offsets


def _make_reference(seq: bytes) -> Reference:
    arr = np.frombuffer(seq, dtype=np.uint8)
    return Reference(arr, ["chr1"], np.array([0, len(arr)], np.uint64), ord("N"))


def test_get_variants_out_of_region():
    ref = _make_reference(b"ACGT")
    alts = RaggedAlleles.from_offsets(
        data=np.frombuffer(b"GT", dtype="|S1"), shape=2, offsets=np.array([0, 1, 2], np.int64)
    )
    vars = _Variants(
        v_starts=np.array([1, 3], np.int32),
        ilens=np.array([0, 0], np.int32),
        alts=alts,
    )
    sg = SparseGenotypes.from_dense(
        genos=np.array([[[1, 1]]], np.int8),
        first_v_idxs=np.array([0], np.int32),
        offsets=np.array([0, 2], np.int64),
        ccfs=np.zeros((1, 2), np.float32),
    )
    haps = Haps(ref, vars, sg, None, RaggedVariants)

    regions = np.array([[0, 0, 3]], np.int32)
    out_lengths = np.array([[3]], np.int32)
    out_offsets = _lengths_to_offsets(out_lengths)
    shifts = np.array([[0]], np.int32)

    ragv = haps._get_variants(np.array([0]), regions, out_offsets, shifts, None, None)
    assert ragv.v_starts.lengths[0, 0] == 1
    np.testing.assert_equal(ragv.v_starts.data[:1], np.array([1], np.int32))


def test_get_variants_shift_excludes():
    ref = _make_reference(b"ACGG")
    alts = RaggedAlleles.from_offsets(
        data=np.frombuffer(b"TCG", dtype="|S1"), shape=2, offsets=np.array([0, 2, 3], np.int64)
    )
    vars = _Variants(
        v_starts=np.array([1, 3], np.int32),
        ilens=np.array([1, 0], np.int32),
        alts=alts,
    )
    sg = SparseGenotypes.from_dense(
        genos=np.array([[[1, 1]]], np.int8),
        first_v_idxs=np.array([0], np.int32),
        offsets=np.array([0, 2], np.int64),
        ccfs=np.zeros((1, 2), np.float32),
    )
    haps = Haps(ref, vars, sg, None, RaggedVariants)

    regions = np.array([[0, 0, 4]], np.int32)
    out_lengths = np.array([[3]], np.int32)
    out_offsets = _lengths_to_offsets(out_lengths)
    shifts = np.array([[3]], np.int32)

    ragv = haps._get_variants(np.array([0]), regions, out_offsets, shifts, None, None)
    assert ragv.v_starts.lengths[0, 0] == 1
    np.testing.assert_equal(ragv.v_starts.data[:1], np.array([3], np.int32))
