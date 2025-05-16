from pathlib import Path

import awkward as ak
import genvarloader as gvl
import polars as pl
from genvarloader._ragged import RaggedAnnotatedHaps
from pytest import fixture

DDIR = Path(__file__).parent.parent / "data"
REF = DDIR / "fasta" / "hg38.fa.bgz"


@fixture
def dataset():
    return gvl.Dataset.open(DDIR / "phased_dataset.vcf.gvl", REF).with_seqs("annotated")


def test_annot_tracks(
    dataset: gvl.RaggedDataset[RaggedAnnotatedHaps, None, None, None],
):
    annots = dataset.regions.with_columns(
        chromEnd=pl.col("chromStart") + 1, score=pl.lit(1.0)
    )
    annot_ds = dataset.write_annot_tracks({"5ss": annots}).with_tracks("5ss")
    haps, tracks = annot_ds[:]
    mask = haps.ref_coords.to_awkward() == ak.Array(
        annot_ds.regions["chromStart"].to_numpy()[:, None, None]
    )
    assert ak.all(tracks.to_awkward()[:, :, 0][mask] == 1)
