import os

import numpy as np
import polars as pl
import pytest

# polars-bio is gated exactly like gvl.Table (see tests/unit/test_table.py).
if not os.environ.get("GVL_TEST_EXPERIMENTAL"):
    pytest.skip(
        "annot polars-bio path is experimental; set GVL_TEST_EXPERIMENTAL=1 to run.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.filterwarnings(
    "ignore::genvarloader._table.ExperimentalWarning"
)


def _regions():
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "chromStart": [0, 50, 0],
            "chromEnd": [100, 150, 100],
        }
    )


def _annot():
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "chromStart": [10, 60, 90, 5],
            "chromEnd": [20, 70, 95, 15],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_annot_overlap_explicit():
    from genvarloader._table import annot_overlap

    regions, annot = _regions(), _annot()
    got = annot_overlap(regions, annot)
    # region 0 [chr1:0-100] overlaps the 3 chr1 annots; region 1 [chr1:50-150] overlaps
    # the chr1 annots at 60-70 and 90-95; region 2 [chr2:0-100] overlaps the chr2 annot.
    np.testing.assert_array_equal(
        np.diff(got.values.offsets), np.array([3, 2, 1])
    )
    # region 2's single interval is the chr2 annot 5-15 with score 4.0
    np.testing.assert_array_equal(np.asarray(got.starts[2]), [5])
    np.testing.assert_array_equal(np.asarray(got.ends[2]), [15])
    np.testing.assert_allclose(np.asarray(got.values[2]), [4.0])
