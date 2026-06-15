import os

import genvarloader as gvl
import polars as pl
import pytest
from genvarloader.experimental import Table

# gvl.Table is experimental and deliberately NOT exercised in CI (its polars-bio
# overlap backend has intermittently segfaulted the interpreter on CPython 3.12
# and 3.13). polars-bio is transitive, so these Table-driven write tests are
# opt-in via an env var rather than gated on the dependency. Set
# GVL_TEST_EXPERIMENTAL=1 to run them locally.
# Upstream: https://github.com/biodatageeks/polars-bio/issues/395
if not os.environ.get("GVL_TEST_EXPERIMENTAL"):
    pytest.skip(
        "gvl.Table is experimental and not tested in CI; set "
        "GVL_TEST_EXPERIMENTAL=1 to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.filterwarnings(
    "ignore::genvarloader._table.ExperimentalWarning"
)


def test_write_duplicate_track_names_rejected(tmp_path):
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    t1 = Table(
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
    t2 = Table(
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
