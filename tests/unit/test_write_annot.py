import numpy as np
import polars as pl


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
    np.testing.assert_array_equal(np.diff(got.values.offsets), np.array([3, 2, 1]))
    # region 2's single interval is the chr2 annot 5-15 with score 4.0
    np.testing.assert_array_equal(np.asarray(got.starts[2]), [5])
    np.testing.assert_array_equal(np.asarray(got.ends[2]), [15])
    np.testing.assert_allclose(np.asarray(got.values[2]), [4.0])


def test_annot_overlap_empty_annot():
    """Regression: empty annot DataFrame (0 rows) must return all-empty RaggedIntervals."""
    from genvarloader._table import annot_overlap

    regions = _regions()
    empty_annot = pl.DataFrame(
        {
            "chrom": pl.Series([], dtype=pl.Utf8),
            "chromStart": pl.Series([], dtype=pl.Int32),
            "chromEnd": pl.Series([], dtype=pl.Int32),
            "score": pl.Series([], dtype=pl.Float32),
        }
    )
    got = annot_overlap(regions, empty_annot)
    n_regions = regions.height
    # Shape is (n_regions, None) — all offsets equal 0 → every region has 0 intervals.
    assert len(got.starts) == n_regions
    assert len(got.ends) == n_regions
    assert len(got.values) == n_regions
    np.testing.assert_array_equal(
        got.values.offsets, np.zeros(n_regions + 1, dtype=np.int64)
    )
    for i in range(n_regions):
        assert len(np.asarray(got.starts[i])) == 0
        assert len(np.asarray(got.ends[i])) == 0
        assert len(np.asarray(got.values[i])) == 0
