"""kind=2 flank_tokens round-trip over the shm layout.

The fixture below is pinned to the layout produced by the real
``get_variants_flat`` builder (verified against
``get_dummy_dataset().with_seqs('variants')...`` with ``flank_length=2``):
``flank_tokens.offsets`` is variant-level (length ``b*p + 1``), identical in
kind to the ``start`` field's offsets -- NOT byte-scaled by ``2*flank_len``.
Only ``flank_tokens.data`` (length ``n_variants * 2*flank_len``) and
``flank_tokens.shape`` (trailing fixed ``2*flank_len`` dim) carry the flank
length.
"""

import numpy as np

from genvarloader._dataset._flat_variants import _FlatVariants
from genvarloader._flat import _Flat
from genvarloader._shm_layout import HEADER_RESERVED, read_chunk, write_chunk


def test_kind2_flank_tokens_roundtrip():
    # 2 instances, ploidy 1, 3 variants total (row0: 2 variants, row1: 1
    # variant), 2L=4 tokens per variant.
    start = _Flat(
        np.array([1, 2, 3], np.int32), np.array([0, 2, 3], np.int64), (2, 1, None)
    )
    ft = _Flat(
        np.arange(3 * 4, dtype=np.uint8),  # 3 variants * 2L(=4) tokens
        np.array([0, 2, 3], np.int64),  # variant-level offsets, same as start's
        (2, 1, None, 4),
    )
    fv = _FlatVariants({"start": start})
    fv.flank_tokens = ft
    buf = memoryview(bytearray(HEADER_RESERVED + (1 << 16)))
    write_chunk(buf, [fv], n_instances=2)
    n, views = read_chunk(buf, copy=True, flat=True)
    out = views[0]
    assert out.flank_tokens is not None
    assert out.flank_tokens.shape == (2, 1, None, 4)
    np.testing.assert_array_equal(
        np.asarray(out.flank_tokens.data), np.asarray(ft.data)
    )
    np.testing.assert_array_equal(
        np.asarray(out.flank_tokens.offsets), np.asarray(ft.offsets)
    )


def test_kind2_flank_tokens_absent_stays_none():
    """A _FlatVariants with no flank_tokens round-trips flank_tokens=None."""
    start = _Flat(
        np.array([1, 2, 3], np.int32), np.array([0, 2, 3], np.int64), (2, 1, None)
    )
    fv = _FlatVariants({"start": start})
    buf = memoryview(bytearray(HEADER_RESERVED + (1 << 16)))
    write_chunk(buf, [fv], n_instances=2)
    n, views = read_chunk(buf, copy=True, flat=True)
    assert views[0].flank_tokens is None
