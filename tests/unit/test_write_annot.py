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


def test_annot_overlap_matches_pyranges_oracle():
    from genvarloader._dataset._impl import _annot_to_intervals  # pyranges oracle
    from genvarloader._table import annot_overlap

    regions, annot = _regions(), _annot()
    got = annot_overlap(regions, annot)
    want = _annot_to_intervals(regions, annot)

    # same per-region counts and the same multiset of intervals per region
    np.testing.assert_array_equal(got.values.offsets, want.values.offsets)
    for r in range(regions.height):
        gs, ge, gv = got.starts[r], got.ends[r], got.values[r]
        ws, we, wv = want.starts[r], want.ends[r], want.values[r]
        go = np.lexsort((gv, ge, gs))
        wo = np.lexsort((wv, we, ws))
        np.testing.assert_array_equal(np.asarray(gs)[go], np.asarray(ws)[wo])
        np.testing.assert_array_equal(np.asarray(ge)[go], np.asarray(we)[wo])
        np.testing.assert_allclose(np.asarray(gv)[go], np.asarray(wv)[wo])
