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


def test_table_init_from_dict_of_dfs():
    per_sample = {
        "s0": pl.DataFrame({"chrom": ["chr1"], "start": [10], "end": [20], "value": [1.0]}),
        "s1": pl.DataFrame({"chrom": ["chr2"], "start": [0],  "end": [5],  "value": [2.0]}),
    }
    t = Table("signal", per_sample)
    assert t.samples == ["s0", "s1"]
    assert set(t.contigs) == {"chr1", "chr2"}


def test_table_column_map_renames_long_form():
    df = pl.DataFrame({
        "donor":      ["s0"],
        "chrom":      ["chr1"],
        "chromStart": [10],
        "chromEnd":   [20],
        "signal":     [1.5],
    })
    t = Table(
        "signal",
        df,
        column_map={"sample_id": "donor", "start": "chromStart",
                    "end": "chromEnd", "value": "signal"},
    )
    assert t.samples == ["s0"]
    assert t.contigs["chr1"] == 20


def test_table_column_map_per_sample_dict():
    per_sample = {
        "s0": pl.DataFrame({
            "chrom": ["chr1"], "chromStart": [10], "chromEnd": [20], "signal": [1.5],
        }),
    }
    t = Table(
        "signal",
        per_sample,
        column_map={"start": "chromStart", "end": "chromEnd", "value": "signal"},
    )
    assert t.samples == ["s0"]
