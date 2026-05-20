from pathlib import Path

import awkward as ak
import genvarloader as gvl
import numpy as np
import polars as pl
import seqpro as sp
from genoray import PGEN, VCF, Reader
from seqpro.rag import Ragged
from genvarloader._utils import lengths_to_offsets
from polars.testing.asserts import assert_frame_equal
from pytest import fixture, mark
from pytest_cases import parametrize_with_cases

ddir = Path(__file__).parents[1] / "data"


def reader_vcf():
    vcf = VCF(ddir / "vcf" / "filtered_sample.vcf.gz")
    vcf._write_gvi_index()
    vcf._load_index()
    return vcf


def reader_pgen():
    index_path = ddir / "pgen" / "filtered_sample.pvar.gvi"
    index_path.unlink()
    pgen = PGEN(ddir / "pgen" / "filtered_sample.pgen")
    return pgen


@fixture
def bed():
    return sp.bed.read(ddir / "vcf" / "sample.bed")


@fixture
def ref():
    return ddir / "fasta" / "hg38.fa.bgz"


@mark.skip
@parametrize_with_cases("reader", cases=".", prefix="reader_")
def test_write(reader: Reader, bed: pl.DataFrame, ref: Path, tmp_path):
    out = tmp_path / "test.gvl"
    gvl.write(out, bed, reader)

    ds = gvl.Dataset.open(out, ref)
    assert ds.shape == (bed.height, reader.n_samples)
    assert_frame_equal(ds.regions, bed, categorical_as_str=True)

    var_idxs = np.memmap(out / "genotypes" / "variant_idxs.npy", dtype=np.int32)
    offsets = np.memmap(out / "genotypes" / "offsets.npy", dtype=np.int64)
    shape = (*ds.shape, reader.ploidy)
    actual = Ragged.from_offsets(var_idxs, (*shape, None), offsets).to_ak()

    # *                 0,
    # * 2,3,   3,3,     1,
    # * 4,5,   4,5,4,5, 6,
    # * 7,8,   9,10,7,  7,7,9,10,
    # *        11,      11,11,
    # *        12,
    # * 13,14, 14,13,   14,14,
    # * 15,    16,
    var_idxs = (
        ak.flatten(
            ak.Array(
                [
                    [0],
                    [2, 3, 3, 3, 1],
                    [4, 5, 4, 5, 4, 5, 6],
                    [7, 8, 9, 10, 7, 7, 7, 9, 10],
                    [11, 11, 11],
                    [12],
                    [13, 14, 14, 13, 14, 14],
                    [15, 16],
                ]
            ),
            -1,
        )
        .to_numpy()
        .astype(np.int32)
    )
    # (r s p)
    lengths = np.array(
        [
            [[0, 0], [0, 0], [0, 1]],
            [[1, 1], [1, 1], [0, 1]],
            [[0, 2], [2, 2], [0, 1]],
            [[1, 1], [2, 1], [1, 3]],
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [0, 1], [0, 0]],
            [[1, 1], [1, 1], [1, 1]],
            [[0, 1], [0, 1], [0, 0]],
        ]
    )
    offsets = lengths_to_offsets(lengths)
    shape = (8, 3, 2)
    desired = Ragged.from_offsets(var_idxs, (*shape, None), offsets).to_ak()

    max_len = lengths.max()
    for len_ in range(1, max_len + 1):
        mask = ak.num(desired, -1) == len_
        assert ak.all(actual[mask][:, :len_] == desired[mask][:, :len_])  # type: ignore


def test_write_warns_when_index_dominates_max_mem(tmp_path, monkeypatch):
    """If variants.nbytes exceeds 50% of max_mem, gvl.write emits a UserWarning."""
    import pytest

    vcf = VCF(ddir / "vcf" / "filtered_source.vcf.gz")
    vcf._write_gvi_index()
    vcf._load_index()

    bed = sp.bed.read(ddir / "source.bed")

    # Force nbytes to a large value relative to max_mem.
    # max_mem = 4 MiB; nbytes = 3 MiB → 75% of budget, should warn.
    monkeypatch.setattr(type(vcf), "nbytes", property(lambda self: 3 * 1024 * 1024))

    out = tmp_path / "test.gvl"
    with pytest.warns(UserWarning, match="exceeds 50% of max_mem"):
        gvl.write(out, bed, vcf, max_mem=4 * 1024 * 1024)

    # Sanity: dataset directory was actually created
    assert (out / "metadata.json").exists()


def _write_two_transcripts_gtf(path: Path) -> None:
    """Write a minimal GTF with one gene (``GENE1``) and two TSL=1 transcripts
    (``T1`` and ``T2``) whose CDS exons sit at identical genomic coordinates.

    The shared exons are at ``chr1:1-99`` and ``chr1:201-290``; each transcript's
    summed CDS length is ``99 + 90 = 189`` bytes (a multiple of 3, so the
    ``require_multiple_of_3`` filter keeps both).
    """
    lines = [
        '1\ttest\tCDS\t1\t99\t.\t+\t0\t'
        'gene_id "G1"; gene_name "GENE1"; transcript_id "T1"; '
        'transcript_support_level "1"; exon_number "1";',
        '1\ttest\tCDS\t201\t290\t.\t+\t0\t'
        'gene_id "G1"; gene_name "GENE1"; transcript_id "T1"; '
        'transcript_support_level "1"; exon_number "2";',
        '1\ttest\tCDS\t1\t99\t.\t+\t0\t'
        'gene_id "G1"; gene_name "GENE1"; transcript_id "T2"; '
        'transcript_support_level "1"; exon_number "1";',
        '1\ttest\tCDS\t201\t290\t.\t+\t0\t'
        'gene_id "G1"; gene_name "GENE1"; transcript_id "T2"; '
        'transcript_support_level "1"; exon_number "2";',
    ]
    path.write_text("\n".join(lines) + "\n")


def test_get_splice_bed_dedupe_overlapping_cds(tmp_path):
    """When `deduplicate_overlapping_cds=True`, rows whose (chrom, chromStart,
    chromEnd, strand) was already produced are dropped — useful when a gene
    has multiple TSL=1 transcripts sharing CDS exons.

    Default behaviour (``False``) keeps every matching transcript's row.
    """
    gtf = tmp_path / "two_transcripts.gtf"
    _write_two_transcripts_gtf(gtf)

    # Default: 4 rows (2 per transcript, 2 transcripts)
    bed_full = gvl.get_splice_bed(gtf)
    assert bed_full.height == 4
    # Two distinct (chromStart, chromEnd) positions, each duplicated
    assert (
        bed_full.unique(subset=["chromStart", "chromEnd"]).height == 2
    )

    # With dedupe: one row per unique CDS position
    bed_dedup = gvl.get_splice_bed(gtf, deduplicate_overlapping_cds=True)
    assert bed_dedup.height == 2
    # Output is still sorted by (chrom, chromStart) and preserves transcript_id
    # from the first row encountered at each position.
    assert bed_dedup["chromStart"].to_list() == [0, 200]  # 0-based BED
    assert bed_dedup["chromEnd"].to_list() == [99, 290]
    assert set(bed_dedup["transcript_id"].to_list()) == {"T1"}
