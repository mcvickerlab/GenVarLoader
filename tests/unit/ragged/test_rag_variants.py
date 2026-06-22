import numpy as np
from genoray._svar import POS_TYPE
from genvarloader import RaggedVariants
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases
from seqpro.rag import Ragged, lengths_to_offsets


# ---------------------------------------------------------------------------
# Helpers: build Ragged inputs for the rc_ case functions
# ---------------------------------------------------------------------------


def _char_ragged(
    p: int, chars: NDArray, l_offsets: NDArray, v_offsets: NDArray
) -> Ragged:
    """Build an S1 char Ragged with shape (b, p, ~v, ~l) from raw buffers.

    b is inferred from v_offsets: len(v_offsets)-1 == b*p.
    """
    b = (len(v_offsets) - 1) // p
    return Ragged.from_offsets(chars, (b, p, None, None), [v_offsets, l_offsets])


def _num_ragged(p: int, data: NDArray, v_offsets: NDArray) -> Ragged:
    """Build a numeric Ragged with shape (b, p, ~v)."""
    b = (len(v_offsets) - 1) // p
    return Ragged.from_offsets(data, (b, p, None), v_offsets)


# ---------------------------------------------------------------------------
# Case functions for test_rc (rc_ is deferred to Task G3 — marked xfail)
# ---------------------------------------------------------------------------


def rc_no_rc():
    # (b=2, p=1, ~v, ~l): variants 0, 2 with alleles [A], [CT]
    l_lens = np.array([1, 2], np.int32)
    l_offsets = lengths_to_offsets(l_lens)
    v_lens = np.array([0, 2], np.int32)
    v_offsets = lengths_to_offsets(v_lens)
    chars = np.frombuffer(b"ACT", dtype="S1").copy()
    alt = _char_ragged(1, chars, l_offsets, v_offsets)
    ilen = _num_ragged(1, np.array([0, 1], np.int32), v_offsets)
    start = _num_ragged(1, np.array([0, 1], POS_TYPE), v_offsets)
    dosage = _num_ragged(1, np.array([0.1, 0.2], np.float32), v_offsets)

    ragv = RaggedVariants(alt=alt, start=start, ilen=ilen, dosage=dosage)
    to_rc = np.zeros(2, np.bool_)
    desired = RaggedVariants(alt=alt, start=start, ilen=ilen, dosage=dosage)

    return ragv, to_rc, desired


def rc_second_batch():
    l_lens = np.array([1, 2], np.int32)
    l_offsets = lengths_to_offsets(l_lens)
    v_lens = np.array([0, 2], np.int32)
    v_offsets = lengths_to_offsets(v_lens)
    chars = np.frombuffer(b"ACT", dtype="S1").copy()
    rc_chars = np.frombuffer(b"TAG", dtype="S1").copy()
    alt = _char_ragged(1, chars, l_offsets, v_offsets)
    rc_alt = _char_ragged(1, rc_chars, l_offsets, v_offsets)
    ilen = _num_ragged(1, np.array([0, 1], np.int32), v_offsets)
    start = _num_ragged(1, np.array([0, 1], POS_TYPE), v_offsets)
    dosage = _num_ragged(1, np.array([0.1, 0.2], np.float32), v_offsets)

    ragv = RaggedVariants(alt=alt, start=start, ilen=ilen, dosage=dosage)
    to_rc = np.array([False, True])
    desired = RaggedVariants(alt=rc_alt, start=start, ilen=ilen, dosage=dosage)

    return ragv, to_rc, desired


def rc_all():
    l_lens = np.array([1, 2], np.int32)
    l_offsets = lengths_to_offsets(l_lens)
    v_lens = np.array([1, 1], np.int32)
    v_offsets = lengths_to_offsets(v_lens)
    chars = np.frombuffer(b"ACT", dtype="S1").copy()
    rc_chars = np.frombuffer(b"TAG", dtype="S1").copy()
    alt = _char_ragged(1, chars, l_offsets, v_offsets)
    rc_alt = _char_ragged(1, rc_chars, l_offsets, v_offsets)
    ilen = _num_ragged(1, np.array([0, 1], np.int32), v_offsets)
    start = _num_ragged(1, np.array([0, 1], POS_TYPE), v_offsets)
    dosage = _num_ragged(1, np.array([0.1, 0.2], np.float32), v_offsets)

    ragv = RaggedVariants(alt=alt, start=start, ilen=ilen, dosage=dosage)
    to_rc = None
    desired = RaggedVariants(alt=rc_alt, start=start, ilen=ilen, dosage=dosage)

    return ragv, to_rc, desired


@parametrize_with_cases("ragv, to_rc, desired", cases=".", prefix="rc_")
def test_rc(ragv: RaggedVariants, to_rc: NDArray[np.bool_], desired: RaggedVariants):
    rc_ragv = ragv.rc_(to_rc)

    assert rc_ragv.alt.to_ak().to_list() == desired.alt.to_ak().to_list()
    assert rc_ragv.ilen.to_ak().to_list() == desired.ilen.to_ak().to_list()
    assert rc_ragv.start.to_ak().to_list() == desired.start.to_ak().to_list()
    if "dosage" in ragv.fields:
        assert rc_ragv.dosage.to_ak().to_list() == desired.dosage.to_ak().to_list()


# ---------------------------------------------------------------------------
# New tests for Task G1: construction + indexing
# ---------------------------------------------------------------------------


def _char_alt(var_off, char_off, chars):
    return Ragged.from_offsets(chars, (2, 1, None, None), [var_off, char_off])


def test_construct_from_char_and_numeric_fields():
    var_off = np.array([0, 2, 3], np.int64)  # b=2,p=1 -> 2 groups
    char_off = np.array([0, 2, 3, 6], np.int64)
    alt = _char_alt(var_off, char_off, np.frombuffer(b"ACGTTT", "S1").copy())
    start = Ragged.from_offsets(np.array([10, 20, 30], np.int32), (2, 1, None), var_off)
    ref = _char_alt(var_off, char_off, np.frombuffer(b"AAGTTT", "S1").copy())
    rv = RaggedVariants(alt=alt, start=start, ref=ref)
    assert rv.shape[0] == 2
    assert len(rv) == 2
    assert rv.alt.to_ak().to_list() == [[b"AC", b"G"], [b"TTT"]]
    assert rv.start.to_ak().to_list() == [[[10, 20]], [[30]]]
    # ilen derived from alt/ref char lengths
    assert rv.ilen.to_ak().to_list() == [[[0, 0]], [[0]]]


def test_getitem_returns_raggedvariants():
    var_off = np.array([0, 2, 3], np.int64)
    char_off = np.array([0, 2, 3, 6], np.int64)
    alt = _char_alt(var_off, char_off, np.frombuffer(b"ACGTTT", "S1").copy())
    start = Ragged.from_offsets(np.array([10, 20, 30], np.int32), (2, 1, None), var_off)
    rv = RaggedVariants(
        alt=alt,
        start=start,
        ilen=Ragged.from_offsets(np.zeros(3, np.int32), (2, 1, None), var_off),
    )
    sub = rv[0]
    assert isinstance(sub, RaggedVariants)
    assert sub.alt.to_ak().to_list() == [[b"AC", b"G"]]


def test_construct_asymmetric_alt_ref_indel():
    """Regression: _share_offsets must use inner (char-level) offsets for opaque-string
    fields, not the outer (variant-level) offsets. Asymmetric alt/ref lengths expose the
    bug — the ref field would be reconstructed with wrong str_offsets, corrupting
    .to_ak() output."""
    # b=1, p=1, 2 variants:
    #   variant 0: alt=b"A"  (len 1), ref=b"ACG" (len 3) → ilen = 1 - 3 = -2
    #   variant 1: alt=b"GT" (len 2), ref=b"T"   (len 1) → ilen = 2 - 1 =  1
    var_off = np.array([0, 2], np.int64)  # 1 group with 2 variants

    # alt char Ragged: chars "A", "GT" → l_offsets [0,1,3], v_offsets [0,2]
    alt_chars = np.frombuffer(b"AGT", dtype="S1").copy()
    alt_l_off = np.array([0, 1, 3], np.int64)  # per-variant char boundaries
    alt_v_off = np.array([0, 2], np.int64)  # 1 group, 2 variants
    alt = Ragged.from_offsets(alt_chars, (1, 1, None, None), [alt_v_off, alt_l_off])

    # ref char Ragged: chars "ACG", "T" → l_offsets [0,3,4], v_offsets [0,2]
    ref_chars = np.frombuffer(b"ACGT", dtype="S1").copy()
    ref_l_off = np.array(
        [0, 3, 4], np.int64
    )  # per-variant char boundaries (DIFFERENT from alt)
    ref_v_off = np.array([0, 2], np.int64)
    ref = Ragged.from_offsets(ref_chars, (1, 1, None, None), [ref_v_off, ref_l_off])

    start = Ragged.from_offsets(np.array([10, 20], np.int32), (1, 1, None), var_off)
    rv = RaggedVariants(alt=alt, start=start, ref=ref)

    assert rv.alt.to_ak().to_list() == [[b"A", b"GT"]]
    assert rv.ref.to_ak().to_list() == [[b"ACG", b"T"]]
    assert rv.ilen.to_ak().to_list() == [[[-2, 1]]]


def test_pad_fills_empty_groups_only():
    var_off = np.array([0, 2, 2, 3], np.int64)  # group1 empty
    char_off = np.array([0, 2, 3, 6], np.int64)
    alt = Ragged.from_offsets(
        np.frombuffer(b"ACGTTT", "S1").copy(), (3, 1, None, None), [var_off, char_off]
    )
    start = Ragged.from_offsets(np.array([1, 2, 3], np.int32), (3, 1, None), var_off)
    rv = RaggedVariants(
        alt=alt,
        start=start,
        ilen=Ragged.from_offsets(np.zeros(3, np.int32), (3, 1, None), var_off),
    )
    out = rv.pad()
    assert out.alt.to_ak().to_list() == [[b"AC", b"G"], [b"N"], [b"TTT"]]
    assert out.start.to_ak().to_list() == [[[1, 2]], [[-1]], [[3]]]


def test_pad_missing_pad_value_raises():
    """pad() must raise ValueError when a field has no default pad value."""
    var_off = np.array([0, 2, 2], np.int64)  # group 1 empty
    char_off = np.array([0, 1, 2], np.int64)
    alt = Ragged.from_offsets(
        np.frombuffer(b"AC", "S1").copy(), (2, 1, None, None), [var_off, char_off]
    )
    start = Ragged.from_offsets(np.array([1, 2], np.int32), (2, 1, None), var_off)
    qual = Ragged.from_offsets(
        np.array([30.0, 40.0], np.float32), (2, 1, None), var_off
    )
    rv = RaggedVariants(
        alt=alt,
        start=start,
        ilen=Ragged.from_offsets(np.zeros(2, np.int32), (2, 1, None), var_off),
        qual=qual,
    )
    import pytest

    with pytest.raises(ValueError, match="qual"):
        rv.pad()  # no pad value supplied for "qual"
