import numpy as np
from seqpro.rag import Ragged
from genvarloader._ragged import RaggedIntervals


def test_prepend_pad_itv_prepends_one_per_group():
    def mk(vals, off):
        return Ragged.from_offsets(
            np.array(vals, np.int32), (1, 1, None), np.array(off, np.int64)
        )

    ri = RaggedIntervals(
        mk([0, 5], [0, 2]),
        mk([5, 9], [0, 2]),
        Ragged.from_offsets(
            np.array([1.0, 2.0], np.float32), (1, 1, None), np.array([0, 2], np.int64)
        ),
    )
    out = ri.prepend_pad_itv(start=-1, end=-1, value=0.0)
    assert out.starts.to_ak().to_list() == [[[-1, 0, 5]]]
    assert out.values.to_ak().to_list() == [[[0.0, 1.0, 2.0]]]
