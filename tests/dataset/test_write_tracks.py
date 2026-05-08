from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
from genvarloader._table import Table

ddir = Path(__file__).parents[1] / "data"


def _make_bed(tmp_path: Path) -> pl.DataFrame:
    bed = pl.DataFrame({
        "chrom":      ["chr1", "chr1"],
        "chromStart": [0, 100],
        "chromEnd":   [50, 200],
    })
    return bed


def _make_table_df() -> pl.DataFrame:
    return pl.DataFrame({
        "sample_id": ["s0", "s0", "s1", "s1"],
        "chrom":     ["chr1", "chr1", "chr1", "chr1"],
        "start":     [10, 110, 5, 150],
        "end":       [20, 130, 15, 160],
        "value":     [1.0, 2.0, 3.0, 4.0],
    })


def test_write_with_table_only_roundtrip(tmp_path):
    bed = _make_bed(tmp_path)
    table = Table("signal", _make_table_df())

    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed, tracks=table)

    # Sanity: the dataset directory has the expected per-track folder.
    assert (out / "intervals" / "signal" / "intervals.npy").exists()
    assert (out / "intervals" / "signal" / "offsets.npy").exists()

    # Read intervals back and confirm values round-trip.
    INTERVAL_DTYPE = np.dtype(
        [("start", np.int32), ("end", np.int32), ("value", np.float32)],
        align=True,
    )
    arr = np.memmap(out / "intervals" / "signal" / "intervals.npy", dtype=INTERVAL_DTYPE, mode="r")
    # Both samples + both regions should produce 4 intervals total.
    assert arr.shape[0] == 4
    values = sorted(float(v) for v in arr["value"])
    assert values == [1.0, 2.0, 3.0, 4.0]
