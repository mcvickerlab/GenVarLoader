import awkward as ak
import genvarloader as gvl
import polars as pl
from genvarloader._ragged import RaggedAnnotatedHaps
from pytest import fixture


@fixture
def dataset(phased_vcf_gvl, ref_fasta):
    return gvl.Dataset.open(phased_vcf_gvl, ref_fasta).with_seqs("annotated")


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
