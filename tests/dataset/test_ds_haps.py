from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import seqpro as sp
from pytest_cases import parametrize_with_cases

data_dir = Path(__file__).resolve().parents[1] / "data"
ref = data_dir / "fasta" / "hg38.fa.bgz"
cons_dir = data_dir / "consensus"


def dataset_vcf():
    ds = (
        gvl.Dataset.open(data_dir / "phased_dataset.vcf.gvl", ref)
        .with_len("ragged")
        .with_seqs("haplotypes")
    )
    return ds


def dataset_svar():
    ds = (
        gvl.Dataset.open(data_dir / "phased_dataset.svar.gvl", ref)
        .with_len("ragged")
        .with_seqs("haplotypes")
    )
    return ds


@parametrize_with_cases("dataset", cases=".", prefix="dataset_")
def test_ds_haps(dataset: gvl.RaggedDataset[gvl.Ragged[np.bytes_], None, None, None]):
    for region, sample in product(range(dataset.n_regions), dataset.samples):
        c, s, e = dataset.regions.select("chrom", "chromStart", "chromEnd").row(region)
        # ragged (p)
        haps = dataset[region, sample]
        for h in range(2):
            actual = haps[h]
            fpath = f"source_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(cons_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]).upper())
            np.testing.assert_equal(
                actual,
                desired,
                f"region: {region}, sample: {sample}, hap: {h}, coords: {c}:{s + 1}-{e}",
            )
