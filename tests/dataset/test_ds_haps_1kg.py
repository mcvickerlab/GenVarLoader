from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import pytest
import seqpro as sp
from genvarloader._ragged import RaggedSeqs
from pytest_cases import parametrize_with_cases

data_dir = Path(__file__).resolve().parents[1] / "data"
ref = data_dir / "fasta" / "hg38.fa.bgz"
cons_dir = data_dir / "1kg_consensus"

pytestmark = pytest.mark.slow


def dataset_bcf():
    return (
        gvl.Dataset.open(data_dir / "1kg" / "phased_1kg.bcf.gvl", ref, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )


def dataset_pgen():
    return (
        gvl.Dataset.open(data_dir / "1kg" / "phased_1kg.pgen.gvl", ref, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )


def dataset_svar():
    return (
        gvl.Dataset.open(data_dir / "1kg" / "phased_1kg.svar.gvl", ref, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )


@parametrize_with_cases("dataset", cases=".", prefix="dataset_")
def test_ds_haps_1kg(dataset: gvl.RaggedDataset[RaggedSeqs, None]):
    for region, sample in product(range(dataset.n_regions), dataset.samples):
        c, s, e, _ = dataset.regions.select(
            "chrom", "chromStart", "chromEnd", "strand"
        ).row(region)
        haps = dataset[region, sample]
        for h in range(2):
            actual = sp.cast_seqs(haps[h])
            fpath = f"1kg_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(cons_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]).upper())
            np.testing.assert_equal(
                actual,
                desired,
                f"region: {region}, sample: {sample}, hap: {h}, "
                f"coords: {c}:{s + 1}-{e}",
            )
