import numpy as np
from seqpro.rag import lengths_to_offsets

from genvarloader._dataset._reference import (
    _get_reference_ser,
    _get_reference_par,
)


def test_get_reference_kernels_agree():
    rng = np.random.default_rng(1)
    reference = rng.integers(65, 85, size=500, dtype=np.uint8)
    ref_offsets = np.array([0, 200, 500], dtype=np.int64)
    # regions: (c_idx, start, end, strand)
    regions = np.array(
        [[0, -5, 10, 1], [1, 10, 30, 1], [0, 190, 205, 1], [1, 0, 300, 1]],
        dtype=np.int64,
    )
    out_offsets = lengths_to_offsets(regions[:, 2] - regions[:, 1])
    pad = ord("N")
    ser = np.empty(int(out_offsets[-1]), np.uint8)
    par = np.empty(int(out_offsets[-1]), np.uint8)
    _get_reference_ser(regions, out_offsets, reference, ref_offsets, pad, ser)
    _get_reference_par(regions, out_offsets, reference, ref_offsets, pad, par)
    np.testing.assert_array_equal(ser, par)
