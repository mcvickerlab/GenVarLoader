import os

import awkward as ak
import genvarloader as gvl
import polars as pl
import pytest
from genoray import VCF


def test_write_with_annot_tracks(vcf_dir, bigwig_dir, ref_fasta, tmp_path):
    out = tmp_path / "ds"
    # chr1:100-200 overlaps the first variant at POS 111
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(out, bed, variants=vcf, annot_tracks={"sig": str(bigwig_dir / "sample_0.bw")})
    ds = gvl.Dataset.open(out, ref_fasta).with_seqs("annotated").with_tracks("sig", "tracks")
    assert "sig" in ds.available_tracks


@pytest.mark.skipif(
    not os.environ.get("GVL_TEST_EXPERIMENTAL"),
    reason="annot DataFrame source uses polars-bio; set GVL_TEST_EXPERIMENTAL=1",
)
def test_annot_tracks(vcf_dir, ref_fasta, tmp_path):
    out = tmp_path / "ds"
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(out, bed, variants=vcf)
    ds = gvl.Dataset.open(out, ref_fasta).with_seqs("annotated")
    annots = ds.regions.with_columns(
        chromEnd=pl.col("chromStart") + 1, score=pl.lit(1.0)
    )
    gvl.update(out, annot_tracks={"5ss": annots})
    annot_ds = gvl.Dataset.open(out, ref_fasta).with_seqs("annotated").with_tracks(
        "5ss", "tracks"
    )
    haps, tracks = annot_ds[:]
    mask = haps.ref_coords == ak.Array(
        annot_ds.regions["chromStart"].to_numpy()[:, None, None]
    )
    assert ak.all(tracks[:, :, 0][mask] == 1)
