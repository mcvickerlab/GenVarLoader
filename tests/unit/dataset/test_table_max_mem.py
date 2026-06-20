"""max_mem is respected by the Rust Table writer (the polars-bio blow-up regression)."""

import numpy as np
import polars as pl
import pytest
from genvarloader import Table
from genvarloader._dataset._write import _write_track_table


def _dense_table(n_intervals: int) -> Table:
    starts = np.arange(0, n_intervals * 10, 10, dtype=np.int64)
    return Table(
        "signal",
        pl.DataFrame(
            {
                "sample_id": ["s0"] * n_intervals,
                "chrom": ["chr1"] * n_intervals,
                "start": starts,
                "end": starts + 5,
                "value": np.ones(n_intervals, np.float32),
            }
        ),
    )


def test_write_track_table_raises_when_region_exceeds_max_mem(tmp_path):
    t = _dense_table(1000)
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [10_000]})
    # One region overlaps ~1000 intervals = ~12 KB; cap at 12 bytes -> must raise.
    with pytest.raises(RuntimeError, match="max_mem"):
        _write_track_table(tmp_path, bed, t, ["s0"], max_mem=12)


def test_write_track_table_succeeds_within_budget(tmp_path):
    t = _dense_table(1000)
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [10_000]})
    _write_track_table(tmp_path, bed, t, ["s0"], max_mem=1 << 20)
    assert (tmp_path / "intervals.npy").exists()
    assert (tmp_path / "offsets.npy").exists()
