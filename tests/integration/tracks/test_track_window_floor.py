import numpy as np
import polars as pl
import pyBigWig
import pytest
import seqpro as sp
from genoray import PGEN, VCF, SparseVar

import genvarloader as gvl

VARIANT_SOURCES = ["vcf", "pgen", "svar"]


def _open_variants(source, vcf_dir, pgen_dir, filtered_svar):
    if source == "vcf":
        return VCF(vcf_dir / "filtered_source.vcf.gz")
    if source == "pgen":
        return PGEN(pgen_dir / "filtered_source.pgen")
    return SparseVar(filtered_svar)


def _chr1_len(ref_fasta):
    ref = gvl.Reference.from_path(ref_fasta, in_memory=False)
    return int(dict(zip(ref.contigs, np.diff(ref.offsets)))["chr1"])


@pytest.mark.parametrize("source", VARIANT_SOURCES)
def test_stored_window_floored_to_input(
    source, vcf_dir, pgen_dir, filtered_svar, ref_fasta, tmp_path
):
    chr1_len = _chr1_len(ref_fasta)
    # One wide region spanning the chr1 variant cluster out to the contig end.
    # Its tail is variant-free, so a pre-fix writer truncates chromEnd to the
    # rightmost variant (< chr1_len); the fix floors it at the input end.
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [chr1_len]})
    variants = _open_variants(source, vcf_dir, pgen_dir, filtered_svar)
    out = tmp_path / "ds.gvl"
    gvl.write(out, bed, variants=variants, overwrite=True)

    regions = np.load(out / "regions.npy")  # (n, 4) int32, writer-sorted order
    input_end = sp.bed.sort(bed)["chromEnd"].to_numpy()  # same sorted order
    assert (regions[:, 2] >= input_end).all(), (
        f"{source}: stored chromEnd {regions[:, 2].tolist()} truncated below "
        f"input {input_end.tolist()}"
    )


def test_annot_track_tail_not_truncated_by_variants(vcf_dir, ref_fasta, tmp_path):
    chr1_len = _chr1_len(ref_fasta)
    # Constant-signal bigWig over all of chr1: every position reads 0.5.
    bw_path = tmp_path / "sig.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", chr1_len)])
    bw.addEntries(["chr1"], [0], ends=[chr1_len], values=[0.5])
    bw.close()

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [chr1_len]})
    out = tmp_path / "ds.gvl"
    gvl.write(
        out,
        bed,
        variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        annot_tracks={"sig": str(bw_path)},
        overwrite=True,
    )

    fr = (
        gvl.Dataset.open(out, ref_fasta)
        .with_seqs(None)
        .with_output_format("flat")
        .with_settings(realign_tracks=False)
        .with_tracks(["sig"])
    )[0:1, 0]
    data, offs = np.asarray(fr.data), np.asarray(fr.offsets)
    seg = data[offs[0] : offs[1]]

    width = chr1_len - 100
    assert seg.shape[0] == width
    # Pre-fix the tail past the rightmost variant reads back 0; post-fix it is
    # fully covered at the source value.
    assert np.count_nonzero(seg) == width
    assert np.allclose(seg, 0.5)
