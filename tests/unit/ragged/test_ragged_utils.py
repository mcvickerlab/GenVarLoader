"""Unit tests for public utilities in ``genvarloader._ragged``.

Covers the non-trivial helpers exposed by the module: the ``to_padded``
free function (bytes path used by ``_dataset/_reference.py``), the
``RaggedIntervals`` dataclass methods (``prepend_pad_itv``, indexing,
reshape) and ``reverse_complement``.

Note: the numeric branch of ``to_padded`` (anything that is not
``np.bytes_``) currently raises inside ``Ragged.__init__`` because
``ak.pad_none(..., clip=True)`` collapses the ragged dimension into a
``RegularArray``. The bytes branch (``ak_str.rpad``) preserves the
ragged structure and works as expected. The numeric path is exercised
only by ``RaggedIntervals.to_padded`` / ``RaggedAnnotatedHaps.to_padded``
— which are part of the public API but have no in-tree callers. Those
methods are skipped here and called out in the task report.
"""

import numpy as np
from seqpro.rag import Ragged

from genvarloader._ragged import (
    RaggedIntervals,
    reverse_complement,
    to_padded,
)


def _make_intervals_3d() -> RaggedIntervals:
    """Build a (batch=2, tracks=1, ~itvs) RaggedIntervals.

    Group [0,0] has 2 intervals; group [1,0] has 1 interval.
    """
    offsets = np.array([0, 2, 3], dtype=np.int64)
    starts = Ragged.from_offsets(
        np.array([0, 10, 5], dtype=np.int32), (2, 1, None), offsets
    )
    ends = Ragged.from_offsets(
        np.array([5, 20, 15], dtype=np.int32), (2, 1, None), offsets
    )
    values = Ragged.from_offsets(
        np.array([1.0, 2.0, 3.0], dtype=np.float32), (2, 1, None), offsets
    )
    return RaggedIntervals(starts, ends, values)


def test_to_padded_bytes_pads_with_byte_value():
    data = np.frombuffer(b"ACGTA", dtype="S1")
    rag = Ragged.from_lengths(data, np.array([2, 3], dtype=np.int64))

    out = to_padded(rag, b"N")

    assert out.shape == (2, 3)
    assert out.dtype == np.dtype("S1")
    np.testing.assert_array_equal(out[0], np.frombuffer(b"ACN", dtype="S1"))
    np.testing.assert_array_equal(out[1], np.frombuffer(b"GTA", dtype="S1"))


def test_to_padded_bytes_no_padding_needed():
    """When all groups already have equal length, output is a clean reshape."""
    data = np.frombuffer(b"ACGT", dtype="S1")
    rag = Ragged.from_lengths(data, np.array([2, 2], dtype=np.int64))

    out = to_padded(rag, b"N")

    np.testing.assert_array_equal(out, np.array([[b"A", b"C"], [b"G", b"T"]], dtype="S1"))


def test_ragged_intervals_shape_and_reshape():
    itvs = _make_intervals_3d()
    assert itvs.shape == (2, 1, None)

    reshaped = itvs.reshape((1, 2, None))
    assert reshaped.shape == (1, 2, None)
    # underlying data is unchanged
    np.testing.assert_array_equal(reshaped.starts.data, itvs.starts.data)
    np.testing.assert_array_equal(reshaped.values.data, itvs.values.data)


def test_ragged_intervals_getitem_preserves_structure():
    itvs = _make_intervals_3d()

    # Pick the first batch row; the wrapped Ragged operates per outer axis
    sub = itvs[0:1]
    # All three components are indexed consistently
    np.testing.assert_array_equal(sub.starts.to_ak().to_list(), [[[0, 10]]])
    np.testing.assert_array_equal(sub.ends.to_ak().to_list(), [[[5, 20]]])
    assert sub.values.to_ak().to_list() == [[[1.0, 2.0]]]


def test_ragged_intervals_prepend_pad_itv_adds_one_per_group():
    itvs = _make_intervals_3d()
    out = itvs.prepend_pad_itv(start=-7, end=-8, value=9.0)

    # Every (batch, track) group gets one extra interval at the front.
    np.testing.assert_array_equal(out.values.lengths, np.array([[3], [2]]))

    assert out.starts.to_ak().to_list() == [[[-7, 0, 10]], [[-7, 5]]]
    assert out.ends.to_ak().to_list() == [[[-8, 5, 20]], [[-8, 15]]]
    assert out.values.to_ak().to_list() == [[[9.0, 1.0, 2.0]], [[9.0, 3.0]]]


def test_reverse_complement_direct():
    """``reverse_complement`` runs against an awkward ragged byte array."""
    data = np.frombuffer(b"ACGTNN", dtype="S1")
    rag = Ragged.from_lengths(data, np.array([4, 2], dtype=np.int64))

    rc = reverse_complement(rag.to_ak())

    # ak_str returns bytes per row; ACGT -> ACGT, NN passes through unchanged
    # because the C-level translation table maps only A<->T, C<->G and
    # leaves other bytes as identity.
    assert rc.to_list() == [b"ACGT", b"NN"]

    # Asymmetric case: "AAAC" -> reverse complement -> "GTTT"
    data2 = np.frombuffer(b"AAAC", dtype="S1")
    rag2 = Ragged.from_lengths(data2, np.array([4], dtype=np.int64))
    rc2 = reverse_complement(rag2.to_ak())
    assert rc2.to_list() == [b"GTTT"]
