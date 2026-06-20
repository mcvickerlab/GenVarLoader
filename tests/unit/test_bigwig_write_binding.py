# tests/unit/test_bigwig_write_binding.py
from pathlib import Path

import numpy as np

from genvarloader._ragged import INTERVAL_DTYPE
from genvarloader.genvarloader import bigwig_write_track


def test_bigwig_write_binding_roundtrip(tmp_path):
    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    paths = [str(data_dir / "sample_0.bw"), str(data_dir / "sample_1.bw")]
    contigs = ["chr1", "chr1"]
    starts = np.array([0, 50], dtype=np.int32)
    ends = np.array([200, 110], dtype=np.int32)
    out = tmp_path
    bigwig_write_track(paths, contigs, starts, ends, 1 << 30, str(out), False)

    itvs = np.memmap(out / "intervals.npy", dtype=INTERVAL_DTYPE, mode="r")
    offsets = np.memmap(out / "offsets.npy", dtype=np.int64, mode="r")
    # 2 regions x 2 samples -> offsets length 5
    assert len(offsets) == 2 * 2 + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(itvs)
    assert itvs.dtype == INTERVAL_DTYPE
