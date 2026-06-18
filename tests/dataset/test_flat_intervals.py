import awkward as ak
import numpy as np

from genvarloader._flat import _Flat
from genvarloader._ragged import FlatIntervals, RaggedIntervals


def _flat(data, offsets, dtype):
    return _Flat.from_offsets(
        np.asarray(data, dtype), (2, None), np.asarray(offsets, np.int64)
    )


def test_flat_intervals_to_ragged_roundtrip():
    # Two groups: group 0 has 1 interval, group 1 has 2 intervals.
    offsets = [0, 1, 3]
    fi = FlatIntervals(
        starts=_flat([10, 20, 30], offsets, np.int32),
        ends=_flat([15, 25, 35], offsets, np.int32),
        values=_flat([1.0, 2.0, 3.0], offsets, np.float32),
    )
    assert fi.shape == (2, None)
    ri = fi.to_ragged()
    assert isinstance(ri, RaggedIntervals)
    assert ak.to_list(ri.starts) == [[10], [20, 30]]
    assert ak.to_list(ri.ends) == [[15], [25, 35]]
    assert ak.to_list(ri.values) == [[1.0], [2.0, 3.0]]


def test_flat_intervals_public_export():
    import genvarloader as gvl

    assert gvl.FlatIntervals is FlatIntervals
    assert "FlatIntervals" in gvl.__all__
