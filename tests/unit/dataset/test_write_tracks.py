import genvarloader as gvl
import polars as pl
import pytest


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
