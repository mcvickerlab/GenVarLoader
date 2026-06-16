from pathlib import Path

import numpy as np

from genvarloader._dataset._write import _annot_intervals


def test_annot_intervals_from_bigwig(tmp_path):
    import polars as pl

    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    bw = data_dir / "sample_0.bw"
    # a region known to overlap intervals in the fixture bigwig
    regions = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [1000]}
    )
    itvs = _annot_intervals(regions, bw, max_mem=2**30)
    # shape (regions, None), one region
    assert itvs.values.offsets.shape == (2,)
    assert itvs.starts.data.dtype == np.int32
