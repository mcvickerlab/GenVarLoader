import genvarloader as gvl
import polars as pl
import pytest

# gvl.Table is temporarily disabled (polars-bio backend intermittently segfaults
# on CPython 3.12 and 3.13; polars-bio removed as a direct dependency, still
# transitive via genoray). These Table-driven write tests are skipped until it
# is re-enabled.
# Upstream: https://github.com/biodatageeks/polars-bio/issues/395
pytestmark = pytest.mark.skip(
    reason="gvl.Table temporarily disabled pending polars-bio segfault fix; see "
    "https://github.com/biodatageeks/polars-bio/issues/395",
)


def test_write_duplicate_track_names_rejected(tmp_path):
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    t1 = gvl.Table(
        "dup",
        pl.DataFrame(
            {
                "sample_id": ["s0"],
                "chrom": ["chr1"],
                "start": [0],
                "end": [10],
                "value": [1.0],
            }
        ),
    )
    t2 = gvl.Table(
        "dup",
        pl.DataFrame(
            {
                "sample_id": ["s0"],
                "chrom": ["chr1"],
                "start": [50],
                "end": [60],
                "value": [2.0],
            }
        ),
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        gvl.write(path=tmp_path / "x.gvl", bed=bed, tracks=[t1, t2])
