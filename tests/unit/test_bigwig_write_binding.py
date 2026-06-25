# tests/unit/test_bigwig_write_binding.py
from pathlib import Path

import numpy as np

from genvarloader.genvarloader import bigwig_write_track


def test_bigwig_write_binding_roundtrip(tmp_path):
    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    paths = [str(data_dir / "sample_0.bw"), str(data_dir / "sample_1.bw")]
    contigs = ["chr1", "chr1"]
    starts = np.array([0, 50], dtype=np.int32)
    ends = np.array([200, 110], dtype=np.int32)
    out = tmp_path
    bigwig_write_track(paths, contigs, starts, ends, 1 << 30, str(out), False)

    starts_arr = np.memmap(out / "starts.npy", dtype=np.int32, mode="r")
    ends_arr = np.memmap(out / "ends.npy", dtype=np.int32, mode="r")
    values_arr = np.memmap(out / "values.npy", dtype=np.float32, mode="r")
    offsets = np.memmap(out / "offsets.npy", dtype=np.int64, mode="r")
    # 2 regions x 2 samples -> offsets length 5
    assert len(offsets) == 2 * 2 + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(starts_arr)
    assert len(starts_arr) == len(ends_arr) == len(values_arr)
    assert starts_arr.dtype == np.int32
    assert ends_arr.dtype == np.int32
    assert values_arr.dtype == np.float32
