"""Unit tests for public utilities in ``genvarloader._ragged``.

Covers the non-trivial helpers exposed by the module: the ``to_padded``
free function (bytes path used by ``_dataset/_reference.py``), and the
``RaggedIntervals`` dataclass methods (``prepend_pad_itv``, indexing,
reshape).
"""

import numpy as np
from genvarloader._ragged import (
    RaggedIntervals,
    to_padded,
)
from seqpro.rag import Ragged


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

    np.testing.assert_array_equal(
        out, np.array([[b"A", b"C"], [b"G", b"T"]], dtype="S1")
    )


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


def test_to_padded_numeric_branch():
    """``to_padded`` should work for numeric dtypes, not just bytes."""
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    rag = Ragged.from_lengths(data, np.array([2, 3], dtype=np.int64))

    out = to_padded(rag, -1)

    assert out.shape == (2, 3)
    assert out.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(out[0], np.array([1, 2, -1], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([3, 4, 5], dtype=np.int32))


def test_ragged_intervals_to_padded():
    itvs = _make_intervals_3d()
    starts, ends, values = itvs.to_padded(start=-1, end=-1, value=0.0)

    # Shape: (batch=2, tracks=1, max_itvs=2)
    assert starts.shape == (2, 1, 2)
    assert ends.shape == (2, 1, 2)
    assert values.shape == (2, 1, 2)
    np.testing.assert_array_equal(
        starts, np.array([[[0, 10]], [[5, -1]]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        ends, np.array([[[5, 20]], [[15, -1]]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        values, np.array([[[1.0, 2.0]], [[3.0, 0.0]]], dtype=np.float32)
    )
