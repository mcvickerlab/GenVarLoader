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
from genvarloader._ragged import Ragged, reverse_complement
from pytest_cases import parametrize_with_cases


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
