"""Unit tests for the ``_rc`` packing invariant on ``Ragged``.

Pins the behavior of ``ak.to_packed(ak.where(...))`` wrapping inside
``Dataset._rc``: the resulting ``Ragged`` must have its content buffer
length equal to the sum of its logical lengths (no doubled-buffer leak
from ``ak.where``).

Originally lived in ``tests/integration/dataset/test_rc_packing.py``;
extracted to the unit tier because it constructs synthetic ``Ragged``
inputs and exercises only the in-memory packing path — no Dataset, no
disk I/O.
"""

import awkward as ak
import numpy as np
from genvarloader import RaggedVariants
from genvarloader._ragged import Ragged, reverse_complement
from pytest_cases import parametrize_with_cases
from seqpro.rag import Ragged as SpRagged


def _buffer_matches_lengths(rag: Ragged) -> bool:
    """Packed invariant: raw content equals the sum of the logical lengths."""
    return len(rag.data) == int(rag.lengths.sum())


def case_all_false():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([False, False, False])


def case_all_true():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([True, True, True])


def case_mixed():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([False, True, False])


@parametrize_with_cases("rag, to_rc", cases=".", prefix="case_")
def test_rc_returns_packed_buffer(rag: Ragged, to_rc: np.ndarray):
    # mimic Dataset._rc exactly
    packed = Ragged(
        ak.to_packed(ak.where(to_rc, reverse_complement(rag.to_ak()), rag.to_ak()))
    )
    assert _buffer_matches_lengths(packed), (
        f"buffer doubled (len={len(packed.data)}, expected={int(packed.lengths.sum())})"
    )
    # and the content is correct for each row
    original = rag.to_ak().to_list()
    rc = reverse_complement(rag.to_ak()).to_list()
    got = packed.to_ak().to_list()
    for i, flip in enumerate(to_rc):
        expected = rc[i] if flip else original[i]
        assert got[i] == expected


def test_rc_all_complements_and_reverses():
    var_off = np.array([0, 1, 2], np.int64)  # 2 groups, 1 variant each
    char_off = np.array([0, 2, 5], np.int64)
    alt = SpRagged.from_offsets(
        np.frombuffer(b"ACGTA", "S1").copy(),
        (2, 1, None, None),
        [var_off, char_off],
    )
    start = SpRagged.from_offsets(np.array([0, 0], np.int32), (2, 1, None), var_off)
    rv = RaggedVariants(
        alt=alt,
        start=start,
        ilen=SpRagged.from_offsets(np.zeros(2, np.int32), (2, 1, None), var_off),
    )
    out = rv.rc_(np.array([True, True]))
    assert out.alt.to_ak().to_list() == [[b"GT"], [b"TAC"]]  # AC->GT, GTA->TAC


def test_rc_ploidy2_partial_mask():
    """Empirically validate the p>1 mask-expansion path in rc_.

    Shape: (b=2, p=2, ~v).  Four groups in layout order:
      group 0 (b0, ploidy0): variants [b"AC", b"G"]
      group 1 (b0, ploidy1): variants [b"T"]
      group 2 (b1, ploidy0): variants [b"GTA"]
      group 3 (b1, ploidy1): variants [b"CC", b"A"]

    to_rc = [True, False]:
      batch 0 → ALL its alleles (3 total) are RC'd:
        b"AC" → b"GT", b"G" → b"C", b"T" → b"A"
      batch 1 → ALL its alleles (3 total) are LEFT UNCHANGED.

    Mask expansion:
      batch_starts = [0, 2]  (step by p=2 into var_off)
      alleles_per_batch[0] = var_off[2] - var_off[0] = 3 - 0 = 3
      alleles_per_batch[1] = var_off[4] - var_off[2] = 6 - 3 = 3
      allele_mask = np.repeat([True, False], [3, 3]) = [T,T,T,F,F,F]
    """
    # var_off: 4 groups (b*p = 4), 5 entries
    #   group 0 → 2 variants, group 1 → 1, group 2 → 1, group 3 → 2
    var_off = np.array([0, 2, 3, 4, 6], np.int64)

    # char_off: 6 alleles total, 7 entries
    #   AC(2) | G(1) | T(1) | GTA(3) | CC(2) | A(1)
    char_off = np.array([0, 2, 3, 4, 7, 9, 10], np.int64)

    # raw chars: "AC" + "G" + "T" + "GTA" + "CC" + "A"
    chars = np.frombuffer(b"ACGTGTACCA", "S1").copy()

    alt = SpRagged.from_offsets(chars, (2, 2, None, None), [var_off, char_off])

    # start/ilen: one entry per allele (6 alleles), same var_off
    start = SpRagged.from_offsets(np.zeros(6, np.int32), (2, 2, None), var_off)
    ilen = SpRagged.from_offsets(np.zeros(6, np.int32), (2, 2, None), var_off)

    rv = RaggedVariants(alt=alt, start=start, ilen=ilen)
    out = rv.rc_(np.array([True, False]))

    # to_ak() flattens (b, p) into a single outer list of b*p groups:
    #   group 0 (b0, ploidy0): AC→GT, G→C
    #   group 1 (b0, ploidy1): T→A
    #   group 2 (b1, ploidy0): GTA unchanged
    #   group 3 (b1, ploidy1): CC, A unchanged
    assert out.alt.to_ak().to_list() == [
        [b"GT", b"C"],  # b0/ploidy0: RC'd
        [b"A"],  # b0/ploidy1: RC'd
        [b"GTA"],  # b1/ploidy0: unchanged
        [b"CC", b"A"],  # b1/ploidy1: unchanged
    ]


def test_to_packed_after_slice_roundtrips():
    var_off = np.array([0, 2, 3, 4], np.int64)  # 3 groups (b=3,p=1)
    char_off = np.array([0, 2, 3, 6, 7], np.int64)
    alt = SpRagged.from_offsets(
        np.frombuffer(b"ACGTTTX", "S1").copy(),
        (3, 1, None, None),
        [var_off, char_off],
    )
    start = SpRagged.from_offsets(
        np.array([1, 2, 3, 4], np.int32), (3, 1, None), var_off
    )
    rv = RaggedVariants(
        alt=alt,
        start=start,
        ilen=SpRagged.from_offsets(np.zeros(4, np.int32), (3, 1, None), var_off),
    )
    sub = rv[np.array([2, 0])].to_packed()
    assert sub.alt.to_ak().to_list() == [[b"X"], [b"AC", b"G"]]
    assert sub.start.to_ak().to_list() == [[4], [1, 2]]
