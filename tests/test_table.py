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


@pytest.fixture
def long_df():
    return make_long_df()


@pytest.mark.parametrize("ext,reader_attr", [
    (".csv", "csv"),
    (".tsv", "tsv"),
    (".parquet", "parquet"),
    (".arrow", "arrow"),
])
def test_table_from_path_long_form(long_df, tmp_path, ext, reader_attr):
    p = tmp_path / f"data{ext}"
    if ext == ".csv":
        long_df.write_csv(p)
    elif ext == ".tsv":
        long_df.write_csv(p, separator="\t")
    elif ext == ".parquet":
        long_df.write_parquet(p)
    elif ext == ".arrow":
        long_df.write_ipc(p)
    t = Table.from_path("signal", p)
    assert t.samples == ["s0", "s1"]


def test_table_from_path_per_sample_dict(long_df, tmp_path):
    s0 = long_df.filter(pl.col("sample_id") == "s0").drop("sample_id")
    s1 = long_df.filter(pl.col("sample_id") == "s1").drop("sample_id")
    p0 = tmp_path / "s0.parquet"
    p1 = tmp_path / "s1.parquet"
    s0.write_parquet(p0)
    s1.write_parquet(p1)
    t = Table.from_path("signal", {"s0": p0, "s1": p1})
    assert t.samples == ["s0", "s1"]


def test_table_from_path_unknown_extension(tmp_path):
    p = tmp_path / "data.bogus"
    p.write_text("nope")
    with pytest.raises(ValueError, match="extension"):
        Table.from_path("signal", p)
