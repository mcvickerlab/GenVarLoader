import sys

import genvarloader as gvl
import polars as pl
import pytest

# polars-bio's overlap kernel segfaults on CPython 3.12 (passes on 3.11 and 3.13);
# Table-driven gvl.write poisons the interpreter for later variant writes.
# Upstream: https://github.com/biodatageeks/polars-bio/issues/395
pytestmark = pytest.mark.skipif(
    sys.version_info[:2] == (3, 12),
    reason="polars-bio overlap segfaults on py3.12; see "
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
