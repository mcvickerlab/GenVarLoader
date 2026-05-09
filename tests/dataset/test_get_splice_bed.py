from pathlib import Path

import polars as pl
import pytest

import genvarloader as gvl


GTF_TEXT = (
    '1\ttest\texon\t100\t200\t.\t+\t.\tgene_id "G1"; gene_name "GENEA"; transcript_id "T1"; exon_number "1"; transcript_support_level "1";\n'
    '1\ttest\tCDS\t300\t308\t.\t+\t0\tgene_id "G1"; gene_name "GENEA"; transcript_id "T1"; exon_number "2"; transcript_support_level "1";\n'
    '1\ttest\tCDS\t100\t108\t.\t+\t0\tgene_id "G1"; gene_name "GENEA"; transcript_id "T1"; exon_number "1"; transcript_support_level "1";\n'
    '2\ttest\tCDS\t500\t506\t.\t-\t0\tgene_id "G2"; gene_name "GENEB"; transcript_id "T2"; exon_number "1"; transcript_support_level "1";\n'
    '2\ttest\tCDS\t600\t606\t.\t-\t0\tgene_id "G2"; gene_name "GENEB"; transcript_id "T2"; exon_number "2"; transcript_support_level "1";\n'
    '3\ttest\tCDS\t700\t705\t.\t+\t0\tgene_id "G3"; gene_name "GENEC"; transcript_id "T3"; exon_number "1"; transcript_support_level "2";\n'
    '4\ttest\tCDS\t800\t804\t.\t+\t0\tgene_id "G4"; transcript_id "T4"; exon_number "1"; transcript_support_level "1";\n'
    '1\ttest\tfive_prime_utr\t50\t99\t.\t+\t.\tgene_id "G1"; gene_name "GENEA"; transcript_id "T1";\n'
)


@pytest.fixture
def gtf_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.gtf"
    p.write_text(GTF_TEXT)
    return p


def test_default_keeps_only_t1(gtf_path: Path):
    """Defaults: TSL=='1', require_multiple_of_3=True. Only T1 (chr 1) survives."""
    bed = gvl.get_splice_bed(gtf_path)
    assert set(bed.columns) == {
        "chrom",
        "chromStart",
        "chromEnd",
        "strand",
        "gene_name",
        "transcript_id",
        "exon_number",
    }
    assert bed["transcript_id"].unique().to_list() == ["T1"]
    assert bed.height == 2


def test_zero_based_start(gtf_path: Path):
    """GTF starts (1-based) become BED chromStart (0-based) by subtracting 1."""
    bed = gvl.get_splice_bed(gtf_path)
    starts = bed.sort("chromStart")["chromStart"].to_list()
    # T1 had GTF starts 100 and 300 -> 99 and 299
    assert starts == [99, 299]


def test_chrom_end_unchanged(gtf_path: Path):
    """GTF end (1-based inclusive) numerically equals BED chromEnd (0-based exclusive)."""
    bed = gvl.get_splice_bed(gtf_path)
    ends = bed.sort("chromStart")["chromEnd"].to_list()
    assert ends == [108, 308]


def test_dropped_non_cds_rows(gtf_path: Path):
    """exon and five_prime_utr rows are removed."""
    bed = gvl.get_splice_bed(
        gtf_path, transcript_support_level=None, require_multiple_of_3=False
    )
    # Every surviving row corresponds to a CDS feature; we have 6 CDS rows in fixture.
    assert bed.height == 6


def test_sorted_output(gtf_path: Path):
    """Output is sorted by chrom (natural), then chromStart."""
    bed = gvl.get_splice_bed(
        gtf_path, transcript_support_level=None, require_multiple_of_3=False
    )
    chroms = bed["chrom"].to_list()
    starts = bed["chromStart"].to_list()
    assert chroms == sorted(chroms, key=lambda c: (len(c), c))  # natural order
    # Within each chrom, starts are non-decreasing
    for c in set(chroms):
        sub = [s for ch, s in zip(chroms, starts) if ch == c]
        assert sub == sorted(sub)


def test_multiple_of_3_filter_off_keeps_t2(gtf_path: Path):
    """T2 (length 14, not multiple of 3) is kept when require_multiple_of_3=False."""
    bed = gvl.get_splice_bed(gtf_path, require_multiple_of_3=False)
    assert "T2" in bed["transcript_id"].unique().to_list()


def test_tsl_none_keeps_t3(gtf_path: Path):
    """T3 (TSL=='2') is kept when transcript_support_level=None."""
    bed = gvl.get_splice_bed(gtf_path, transcript_support_level=None)
    # T3 length is 6 (multiple of 3), so default require_multiple_of_3 still keeps it.
    assert "T3" in bed["transcript_id"].unique().to_list()


def test_tsl_explicit_value(gtf_path: Path):
    """transcript_support_level='2' selects only T3 among multiple-of-3 transcripts."""
    bed = gvl.get_splice_bed(gtf_path, transcript_support_level="2")
    assert bed["transcript_id"].unique().to_list() == ["T3"]


def test_contigs_filter(gtf_path: Path):
    """contigs=['1'] restricts to chr 1 rows."""
    bed = gvl.get_splice_bed(
        gtf_path,
        contigs=["1"],
        transcript_support_level=None,
        require_multiple_of_3=False,
    )
    assert bed["chrom"].unique().to_list() == ["1"]


def test_gene_name_nulls_preserved(gtf_path: Path):
    """T4 has no gene_name attribute -> gene_name is null and the row is retained."""
    bed = gvl.get_splice_bed(
        gtf_path, transcript_support_level=None, require_multiple_of_3=False
    )
    t4 = bed.filter(pl.col("transcript_id") == "T4")
    assert t4.height == 1
    assert t4["gene_name"].to_list() == [None]


def test_dtypes(gtf_path: Path):
    """exon_number is Int32; chromStart/chromEnd are integers."""
    bed = gvl.get_splice_bed(
        gtf_path, transcript_support_level=None, require_multiple_of_3=False
    )
    assert bed.schema["exon_number"] == pl.Int32
    assert bed.schema["chromStart"].is_integer()
    assert bed.schema["chromEnd"].is_integer()
    assert bed.schema["chrom"] == pl.Utf8
    assert bed.schema["strand"] == pl.Utf8
    assert bed.schema["gene_name"] == pl.Utf8
    assert bed.schema["transcript_id"] == pl.Utf8
