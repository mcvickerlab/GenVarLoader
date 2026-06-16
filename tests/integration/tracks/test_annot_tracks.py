import awkward as ak
import genvarloader as gvl
import polars as pl
from genoray import VCF
from genvarloader._ragged import RaggedAnnotatedHaps
from pytest import fixture


@fixture
def dataset(phased_vcf_gvl, ref_fasta):
    return gvl.Dataset.open(phased_vcf_gvl, ref_fasta).with_seqs("annotated")


def test_write_with_annot_tracks(vcf_dir, bigwig_dir, ref_fasta, tmp_path):
    out = tmp_path / "ds"
    # chr1:100-200 overlaps the first variant at POS 111
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    # pyrefly: ignore [unexpected-keyword]
    gvl.write(out, bed, variants=vcf, annot_tracks={"sig": str(bigwig_dir / "sample_0.bw")})
    ds = gvl.Dataset.open(out, ref_fasta).with_seqs("annotated").with_tracks("sig", "tracks")
    assert ds.available_tracks is not None
    assert "sig" in ds.available_tracks


def test_annot_tracks(dataset: gvl.RaggedDataset[RaggedAnnotatedHaps, None]):
    annots = dataset.regions.with_columns(
        chromEnd=pl.col("chromStart") + 1, score=pl.lit(1.0)
    )
    annot_ds = dataset.write_annot_tracks({"5ss": annots}, overwrite=True).with_tracks(
        "5ss", "tracks"
    )
    haps, tracks = annot_ds[:]
    mask = haps.ref_coords == ak.Array(
        annot_ds.regions["chromStart"].to_numpy()[:, None, None]
    )
    assert ak.all(tracks[:, :, 0][mask] == 1)
