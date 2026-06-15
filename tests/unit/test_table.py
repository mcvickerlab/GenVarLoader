import os

import numpy as np
import polars as pl
import pytest
from genvarloader._utils import lengths_to_offsets
from genvarloader.experimental import Table

# gvl.Table is experimental and deliberately NOT exercised in CI: its polars-bio
# overlap backend has intermittently segfaulted the interpreter during overlap
# queries (observed on CPython 3.12 and 3.13), which can crash the whole test
# run. polars-bio is a transitive dependency, so an importorskip would not keep
# these out of CI; instead they are opt-in via an env var. Set
# GVL_TEST_EXPERIMENTAL=1 to run them locally (requires the `table` extra:
# `pip install genvarloader[table]`).
# Upstream: https://github.com/biodatageeks/polars-bio/issues/395
if not os.environ.get("GVL_TEST_EXPERIMENTAL"):
    pytest.skip(
        "gvl.Table is experimental and not tested in CI; set "
        "GVL_TEST_EXPERIMENTAL=1 to run these tests.",
        allow_module_level=True,
    )

# Constructing a Table emits an ExperimentalWarning by design; silence it here.
pytestmark = pytest.mark.filterwarnings(
    "ignore::genvarloader._table.ExperimentalWarning"
)


def make_long_df():
    return pl.DataFrame(
        {
            "sample_id": ["s0", "s0", "s1", "s1"],
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "start": [10, 100, 20, 0],
            "end": [20, 110, 30, 5],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


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
        "s0": pl.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [10],
                "end": [20],
                "value": [1.0],
            }
        ),
        "s1": pl.DataFrame(
            {
                "chrom": ["chr2"],
                "start": [0],
                "end": [5],
                "value": [2.0],
            }
        ),
    }
    t = Table("signal", per_sample)
    assert t.samples == ["s0", "s1"]
    assert set(t.contigs) == {"chr1", "chr2"}


def test_table_column_map_renames_long_form():
    df = pl.DataFrame(
        {
            "donor": ["s0"],
            "chrom": ["chr1"],
            "chromStart": [10],
            "chromEnd": [20],
            "signal": [1.5],
        }
    )
    t = Table(
        "signal",
        df,
        column_map={
            "sample_id": "donor",
            "start": "chromStart",
            "end": "chromEnd",
            "value": "signal",
        },
    )
    assert t.samples == ["s0"]
    assert t.contigs["chr1"] == 20


def test_table_column_map_per_sample_dict():
    per_sample = {
        "s0": pl.DataFrame(
            {
                "chrom": ["chr1"],
                "chromStart": [10],
                "chromEnd": [20],
                "signal": [1.5],
            }
        ),
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


@pytest.mark.parametrize(
    "ext,reader_attr",
    [
        (".csv", "csv"),
        (".tsv", "tsv"),
        (".parquet", "parquet"),
        (".arrow", "arrow"),
    ],
)
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


def _brute_count(df: pl.DataFrame, contig: str, starts, ends, samples):
    """Reference implementation: O(n*m) overlap count."""
    out = np.zeros((len(starts), len(samples)), dtype=np.int32)
    for si, s in enumerate(samples):
        sub = df.filter((pl.col("sample_id") == s) & (pl.col("chrom") == contig))
        ts = sub["start"].to_numpy()
        te = sub["end"].to_numpy()
        for ri, (rs, re_) in enumerate(zip(starts, ends)):
            out[ri, si] = int(((ts < re_) & (te > rs)).sum())
    return out


def test_table_count_intervals_matches_brute_force():
    df = pl.DataFrame(
        {
            "sample_id": ["s0", "s0", "s0", "s1", "s1"],
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1"],
            "start": [0, 50, 200, 10, 60],
            "end": [10, 60, 210, 20, 70],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    t = Table("signal", df)
    starts = np.array([0, 55, 100, 200], dtype=np.int32)
    ends = np.array([15, 65, 150, 205], dtype=np.int32)
    counts = t.count_intervals("chr1", starts, ends, sample=["s0", "s1"])
    expected = _brute_count(df, "chr1", starts, ends, ["s0", "s1"])
    assert counts.dtype == np.int32
    assert counts.shape == (4, 2)
    np.testing.assert_array_equal(counts, expected)


def test_table_count_intervals_unknown_contig_returns_zeros():
    t = Table("signal", make_long_df())
    counts = t.count_intervals("chrX", np.array([0]), np.array([10]), sample=["s0"])
    np.testing.assert_array_equal(counts, np.zeros((1, 1), dtype=np.int32))


def test_table_intervals_from_offsets_roundtrip():
    df = pl.DataFrame(
        {
            "sample_id": ["s0", "s0", "s1"],
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 50, 10],
            "end": [10, 60, 20],
            "value": [1.5, 2.5, 3.5],
        }
    )
    t = Table("signal", df)
    starts = np.array([0, 40], dtype=np.int32)
    ends = np.array([15, 70], dtype=np.int32)
    samples = ["s0", "s1"]

    counts = t.count_intervals("chr1", starts, ends, sample=samples)
    offsets = lengths_to_offsets(counts.ravel())
    intervals = t._intervals_from_offsets("chr1", starts, ends, offsets, sample=samples)

    # shape: (regions=2, samples=2, ragged)
    assert intervals.starts.data.dtype == np.int32
    assert intervals.values.data.dtype == np.float32
    # cell (region=0, sample=s0): one interval [0, 10) value 1.5
    flat_start = intervals.starts.data
    flat_end = intervals.ends.data
    flat_val = intervals.values.data
    assert flat_start[0] == 0 and flat_end[0] == 10 and flat_val[0] == np.float32(1.5)
    # total interval count == sum of counts
    assert len(flat_start) == int(counts.sum())


def test_table_count_intervals_normalizes_contig_names():
    """Table should accept either `chr1` or `1` styled contig names, like BigWigs."""
    df = pl.DataFrame(
        {
            "sample_id": ["s0"],
            "chrom": ["1"],  # no chr prefix
            "start": [0],
            "end": [10],
            "value": [1.0],
        }
    )
    t = Table("signal", df)
    # Query using "chr1" should match the "1"-keyed table.
    counts = t.count_intervals("chr1", np.array([0]), np.array([20]), sample=["s0"])
    np.testing.assert_array_equal(counts, np.array([[1]], dtype=np.int32))
