"""kind=4 round-trip for _FlatVariantWindows over the shm layout."""

import numpy as np
from genvarloader._dataset._flat_variants import _FlatVariantWindows, _FlatWindow
from genvarloader._flat import _Flat
from genvarloader._shm_layout import HEADER_RESERVED, read_chunk, write_chunk


def _make_fvw():
    # 2 instances, ploidy 1. start scalar field + ref_window + alt (bare) slots.
    start = _Flat(
        np.array([10, 20, 30], np.int32), np.array([0, 2, 3], np.int64), (2, 1, None)
    )
    # ref_window: b*p=2 rows; var_offsets len 3; per-variant token runs via seq_offsets.
    rw = _FlatWindow(
        data=np.array([1, 2, 3, 4, 1, 2], np.uint8),
        seq_offsets=np.array([0, 3, 4, 6], np.int64),  # 3 variants
        var_offsets=np.array([0, 2, 3], np.int64),  # 2 rows -> 2,1 variants
        shape=(2, 1, None, None),
    )
    al = _FlatWindow(
        data=np.array([0, 1, 2, 3], np.uint8),
        seq_offsets=np.array([0, 1, 3, 4], np.int64),
        var_offsets=np.array([0, 2, 3], np.int64),
        shape=(2, 1, None, None),
    )
    return _FlatVariantWindows({"start": start}, ref_window=rw, alt=al)


def test_kind4_roundtrip():
    fvw = _make_fvw()
    buf = memoryview(bytearray(HEADER_RESERVED + (1 << 16)))
    write_chunk(buf, [fvw], n_instances=2)
    n, views = read_chunk(buf, copy=True, flat=True)
    assert n == 2 and len(views) == 1
    out = views[0]
    assert isinstance(out, _FlatVariantWindows)
    a, b = fvw.to_ragged(), out.to_ragged()
    assert set(a) == set(b)
    for k in a:
        assert a[k].to_ak().to_list() == b[k].to_ak().to_list(), f"{k} mismatch"
