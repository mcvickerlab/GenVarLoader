import numpy as np
import polars as pl
import pytest

from genvarloader._table import Table


def make_long_df():
    return pl.DataFrame({
        "sample_id": ["s0", "s0", "s1", "s1"],
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "start": [10, 100, 20, 0],
        "end":   [20, 110, 30, 5],
        "value": [1.0, 2.0, 3.0, 4.0],
    })


def test_table_init_from_long_df():
    t = Table("signal", make_long_df())
    assert t.name == "signal"
    assert t.samples == ["s0", "s1"]
    assert set(t.contigs) == {"chr1", "chr2"}
    assert t.contigs["chr1"] >= 110
    assert t.contigs["chr2"] >= 5


def test_table_init_missing_canonical_column_raises():
    bad = make_long_df().drop("value")
    with pytest.raises(ValueError, match="value"):
        Table("signal", bad)
