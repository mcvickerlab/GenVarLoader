import numpy as np
import polars as pl
from genoray._utils import ContigNormalizer
from genvarloader._dataset._utils import bed_to_regions, splits_sum_le_value
from genvarloader._utils import normalize_contig_name
from pytest_cases import parametrize_with_cases


def test_bed_to_regions_categorical_strand_returns_int32() -> None:
    """Regression: BEDs whose strand column is Categorical (e.g. from
    `polars.bed.sort` or any pipeline that round-trips a strand category)
    must produce an int32 regions array, not dtype=object.

    Without the Categorical-aware branch in `bed_to_regions`, the strand
    column survived the `select(...)` call as Categorical, polars' mixed
    Int32 + Categorical `to_numpy()` collapsed to dtype=object, and
    downstream numba kernels (`get_diffs_sparse`) refused to compile with
    `non-precise type array(pyobject, 1d, A)`. See PR for the chr19 ADNI
    cohort reproducer.
    """
    bed = pl.DataFrame(
        {
            "chrom": ["chr19", "chr19"],
            "chromStart": [44906624, 44907759],
            "chromEnd": [44906667, 44907952],
            "strand": ["+", "+"],
        }
    ).with_columns(pl.col("strand").cast(pl.Categorical))
    assert bed.schema["strand"] == pl.Categorical
    regions = bed_to_regions(bed, ContigNormalizer(["chr19"]))
    assert regions.dtype == np.int32, f"want int32, got {regions.dtype}"
    assert regions.shape == (2, 4)
    np.testing.assert_array_equal(
        regions,
        np.array([[0, 44906624, 44906667, 1], [0, 44907759, 44907952, 1]], np.int32),
    )


def test_bed_to_regions_utf8_strand_still_works() -> None:
    """Sanity: the existing Utf8-strand path still produces int32."""
    bed = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [100],
            "chromEnd": [200],
            "strand": ["-"],
        }
    )
    assert bed.schema["strand"] == pl.Utf8
    regions = bed_to_regions(bed, ContigNormalizer(["chr1"]))
    assert regions.dtype == np.int32
    np.testing.assert_array_equal(regions, np.array([[0, 100, 200, -1]], np.int32))


def test_bed_to_regions_no_strand_defaults_to_plus() -> None:
    """BEDs without a strand column get strand=1 (existing behaviour)."""
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})
    regions = bed_to_regions(bed, ContigNormalizer(["chr1"]))
    assert regions.dtype == np.int32
    np.testing.assert_array_equal(regions, np.array([[0, 100, 200, 1]], np.int32))


def test_splits_sum_le_value():
    max_size = 10
    sizes = np.array([3, 5, 2, 4, 7, 5, 2], np.int32)
    splits = splits_sum_le_value(sizes, max_size)
    np.testing.assert_equal(splits, np.array([0, 3, 4, 5, 7], np.intp))
    np.testing.assert_array_less(np.add.reduceat(sizes, splits[:-1]), max_size + 1)


def contig_match():
    unnormed = "chr1"
    source = ["chr1", "chr2"]
    desired = "chr1"
    return unnormed, source, desired


def contig_add_match():
    unnormed = "1"
    source = ["chr1", "chr2"]
    desired = "chr1"
    return unnormed, source, desired


def contig_strip_match():
    unnormed = "chr1"
    source = ["1", "2"]
    desired = "1"
    return unnormed, source, desired


def contig_no_match():
    unnormed = "chr3"
    source = ["chr1", "chr2"]
    desired = None
    return unnormed, source, desired


def contig_list():
    unnormed = ["chr1", "1", "chr3"]
    source = ["chr1", "chr2"]
    desired = ["chr1", "chr1", None]
    return unnormed, source, desired


@parametrize_with_cases("unnormed, source, desired", cases=".", prefix="contig_")
def test_normalize_contig_name(unnormed, source, desired):
    assert normalize_contig_name(unnormed, source) == desired
