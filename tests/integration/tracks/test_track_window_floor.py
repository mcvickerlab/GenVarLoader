import numpy as np
import polars as pl
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
