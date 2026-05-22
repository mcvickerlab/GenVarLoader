from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
import pytest

DDIR = Path(__file__).parent / "data"
REF = DDIR / "fasta" / "hg38.fa.bgz"


@pytest.fixture
def reference() -> gvl.Reference:
    return gvl.Reference.from_path(REF, in_memory=False)


@pytest.fixture
def two_transcript_bed() -> pl.DataFrame:
    # Two transcripts, both '+' strand. T1 has 2 exons; T2 has 1 exon.
    return pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "chromStart": [1000, 2000, 5000],
        "chromEnd": [1010, 2010, 5010],
        "strand": [1, 1, 1],
        "transcript_id": ["T1", "T1", "T2"],
        "exon_number": [1, 2, 1],
    })


def _as_s1(x) -> np.ndarray:
    """Convert bytes or S1 array to a 1-D S1 numpy array."""
    if isinstance(x, (bytes, bytearray)):
        return np.frombuffer(x, dtype="S1")
    return np.asarray(x, dtype="S1").ravel()


def test_spliced_single_col(reference: gvl.Reference, two_transcript_bed: pl.DataFrame):
    ds = gvl.RefDataset(
        reference, two_transcript_bed, splice_info="transcript_id"
    )
    assert ds.is_spliced is True
    assert len(ds) == 2  # two transcripts

    spliced = ds[:]  # ragged: (2, ~l)

    unsp = gvl.RefDataset(reference, two_transcript_bed)[:]
    expected_t1 = np.concatenate([_as_s1(unsp[0]), _as_s1(unsp[1])])
    expected_t2 = _as_s1(unsp[2])

    np.testing.assert_equal(_as_s1(spliced[0]), expected_t1)
    np.testing.assert_equal(_as_s1(spliced[1]), expected_t2)


def test_spliced_two_col_reorders_exons(reference: gvl.Reference):
    # Exons stored out-of-order; exon_number column dictates splice order.
    bed = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "chromStart": [2000, 1000],
        "chromEnd": [2010, 1010],
        "strand": [1, 1],
        "transcript_id": ["T1", "T1"],
        "exon_number": [2, 1],
    })

    ds = gvl.RefDataset(reference, bed, splice_info=("transcript_id", "exon_number"))
    spliced = ds[0]

    unsp = gvl.RefDataset(reference, bed)[:]
    expected = np.concatenate([_as_s1(unsp[1]), _as_s1(unsp[0])])  # exon 1 then exon 2
    np.testing.assert_equal(_as_s1(spliced), expected)


def test_spliced_mixed_strand(reference: gvl.Reference):
    # Negative-strand exons: per-exon RC, then concat.
    bed = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "chromStart": [1000, 2000],
        "chromEnd": [1010, 2010],
        "strand": [-1, -1],
        "transcript_id": ["T1", "T1"],
        "exon_number": [1, 2],
    })

    ds = gvl.RefDataset(reference, bed, splice_info="transcript_id")
    spliced = ds[0]

    unsp = gvl.RefDataset(reference, bed, rc_neg=True)[:]
    expected = np.concatenate([_as_s1(unsp[0]), _as_s1(unsp[1])])
    np.testing.assert_equal(_as_s1(spliced), expected)
