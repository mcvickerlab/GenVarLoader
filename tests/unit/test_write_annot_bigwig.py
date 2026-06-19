from pathlib import Path

import numpy as np
import polars as pl

from genvarloader._dataset import _write
from genvarloader._dataset._write import _annot_intervals


def test_annot_intervals_from_bigwig(tmp_path):
    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    bw = data_dir / "sample_0.bw"
    # a region known to overlap intervals in the fixture bigwig
    regions = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [1000]})
    itvs = _annot_intervals(regions, bw, max_mem=2**30)
    # shape (regions, None), one region
    assert itvs.values.offsets.shape == (2,)
    assert itvs.starts.data.dtype == np.int32


def test_write_annot_track_rust_byte_matches_legacy(tmp_path):
    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    bw = data_dir / "sample_0.bw"
    regions = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 50], "chromEnd": [200, 110]}
    )

    legacy_dir = tmp_path / "legacy"
    rust_dir = tmp_path / "rust"
    legacy_dir.mkdir()
    rust_dir.mkdir()

    # legacy
    itvs = _write._annot_intervals(regions, bw, max_mem=2**30)
    _write._write_ragged_intervals(legacy_dir, itvs)
    # rust
    _write._write_annot_track_rust(rust_dir, regions, bw, max_mem=2**30)

    assert (legacy_dir / "intervals.npy").read_bytes() == (
        rust_dir / "intervals.npy"
    ).read_bytes()
    assert (legacy_dir / "offsets.npy").read_bytes() == (
        rust_dir / "offsets.npy"
    ).read_bytes()
