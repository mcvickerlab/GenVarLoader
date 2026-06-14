import numpy as np
from seqpro.rag import lengths_to_offsets

from genvarloader._dataset._reference import _fetch_impl_ser, _fetch_impl_par


def _run(kernel, c_idxs, starts, ends, reference, ref_offsets, pad_char):
    out_offsets = lengths_to_offsets(ends - starts)
    out = np.empty(int(out_offsets[-1]), np.uint8)
    kernel(c_idxs, starts, ends, reference, ref_offsets, pad_char, out, out_offsets)
    return out


def test_serial_and_parallel_kernels_agree():
    rng = np.random.default_rng(0)
    reference = rng.integers(65, 85, size=500, dtype=np.uint8)  # ascii A..T
    ref_offsets = np.array([0, 200, 500], dtype=np.int64)  # 2 contigs
    c_idxs = np.array([0, 1, 0, 1], dtype=np.int64)
    starts = np.array([-5, 10, 190, 0], dtype=np.int64)  # includes OOB left
    ends = np.array([10, 30, 205, 300], dtype=np.int64)  # includes OOB right
    pad = ord("N")
    ser = _run(_fetch_impl_ser, c_idxs, starts, ends, reference, ref_offsets, pad)
    par = _run(_fetch_impl_par, c_idxs, starts, ends, reference, ref_offsets, pad)
    np.testing.assert_array_equal(ser, par)
