"""Unit tests for ``RefDataset`` splice-related settings and validation.

Originally lived in tests/integration/test_ref_ds_splicing.py;
extracted to the unit tier because every test here exercises only
in-memory state transitions or validation errors — no byte-level output
comparison against the reference genome. The 4 byte-comparison tests
(``test_spliced_single_col``, ``test_spliced_two_col_reorders_exons``,
``test_spliced_mixed_strand``, ``test_subset_to_transcripts``) remain in
the original file as integration regression tests.
"""

import genvarloader as gvl
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
