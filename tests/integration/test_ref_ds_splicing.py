import genvarloader as gvl
import numpy as np
import polars as pl
import pytest


@pytest.fixture
def reference(ref_fasta) -> gvl.Reference:
    return gvl.Reference.from_path(ref_fasta, in_memory=False)


@pytest.fixture
def two_transcript_bed() -> pl.DataFrame:
    # Two transcripts, both '+' strand. T1 has 2 exons; T2 has 1 exon.
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1000, 2000, 5000],
            "chromEnd": [1010, 2010, 5010],
            "strand": [1, 1, 1],
            "transcript_id": ["T1", "T1", "T2"],
            "exon_number": [1, 2, 1],
        }
    )


def _as_s1(x) -> np.ndarray:
    """Convert bytes or S1 array to a 1-D S1 numpy array."""
    if isinstance(x, (bytes, bytearray)):
        return np.frombuffer(x, dtype="S1")
    return np.asarray(x, dtype="S1").ravel()


def test_spliced_single_col(reference: gvl.Reference, two_transcript_bed: pl.DataFrame):
    ds = gvl.RefDataset(reference, two_transcript_bed, splice_info="transcript_id")
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
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [2000, 1000],
            "chromEnd": [2010, 1010],
            "strand": [1, 1],
            "transcript_id": ["T1", "T1"],
            "exon_number": [2, 1],
        }
    )

    ds = gvl.RefDataset(reference, bed, splice_info=("transcript_id", "exon_number"))
    spliced = ds[0]

    unsp = gvl.RefDataset(reference, bed)[:]
    expected = np.concatenate([_as_s1(unsp[1]), _as_s1(unsp[0])])  # exon 1 then exon 2
    np.testing.assert_equal(_as_s1(spliced), expected)


def test_spliced_mixed_strand(reference: gvl.Reference):
    # Negative-strand exons: per-exon RC, then concat.
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [1000, 2000],
            "chromEnd": [1010, 2010],
            "strand": [-1, -1],
            "transcript_id": ["T1", "T1"],
            "exon_number": [1, 2],
        }
    )

    ds = gvl.RefDataset(reference, bed, splice_info="transcript_id")
    spliced = ds[0]

    unsp = gvl.RefDataset(reference, bed, rc_neg=True)[:]
    expected = np.concatenate([_as_s1(unsp[0]), _as_s1(unsp[1])])
    np.testing.assert_equal(_as_s1(spliced), expected)


def test_with_settings_disable_splice(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed, splice_info="transcript_id")
    assert ds.is_spliced
    plain = ds.with_settings(splice_info=False)
    assert plain.is_spliced is False
    assert len(plain) == 3  # back to per-exon row count


def test_with_settings_enable_splice(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed)
    assert not ds.is_spliced
    sp = ds.with_settings(splice_info="transcript_id")
    assert sp.is_spliced
    assert len(sp) == 2


def test_with_settings_validation(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed, jitter=0)
    with pytest.raises(RuntimeError, match="Jitter is not supported"):
        ds.with_settings(splice_info="transcript_id", jitter=1)

    with pytest.raises(RuntimeError, match="Non-deterministic"):
        ds.with_settings(splice_info="transcript_id", deterministic=False)


def test_subset_to_transcripts(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed, splice_info="transcript_id")
    sub = ds.subset_to(["T2"])
    assert len(sub) == 1
    spliced = sub[0]
    unsp = gvl.RefDataset(reference, two_transcript_bed)[:]
    # The single exon of T2 should match unsp[2].
    np.testing.assert_equal(_as_s1(spliced), _as_s1(unsp[2]))


def test_spliced_output_length_variable(reference, two_transcript_bed):
    ds = gvl.RefDataset(
        reference, two_transcript_bed, splice_info="transcript_id"
    ).with_len("variable")
    out = ds[:]
    # variable-length pads to the longest transcript in the batch.
    assert out.shape[0] == 2
    assert out.shape[1] == 20  # T1 has 2 × 10 = 20; T2 padded to 20.


def test_spliced_rejects_fixed_length(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed)
    with pytest.raises(RuntimeError, match="Splicing requires output_length"):
        ds.with_settings(splice_info="transcript_id").with_len(5)
