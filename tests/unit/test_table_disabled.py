"""gvl.Table is temporarily disabled because its polars-bio backend
intermittently segfaults during overlap queries (observed on CPython 3.12 and
3.13). polars-bio has been removed as a direct dependency (it remains transitive
via genoray). This asserts the disabled state so the functional Table tests
(skipped) can be re-enabled deliberately once the upstream issue is resolved.

Upstream: https://github.com/biodatageeks/polars-bio/issues/395
"""

import polars as pl
import pytest

import genvarloader as gvl


def test_table_construction_raises_not_implemented():
    df = pl.DataFrame(
        {
            "sample_id": ["s0"],
            "chrom": ["chr1"],
            "start": [0],
            "end": [10],
            "value": [1.0],
        }
    )
    with pytest.raises(NotImplementedError, match="temporarily disabled"):
        gvl.Table("signal", df)


def test_table_from_path_raises_not_implemented(tmp_path):
    p = tmp_path / "track.csv"
    p.write_text("sample_id,chrom,start,end,value\ns0,chr1,0,10,1.0\n")
    with pytest.raises(NotImplementedError, match="temporarily disabled"):
        gvl.Table.from_path("signal", p)
