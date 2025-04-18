from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import pytest
import seqpro as sp

data_dir = Path(__file__).resolve().parents[1] / "data"
ds_path = data_dir / "phased_dataset.gvl"
ref = data_dir / "fasta" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.bgz"
cons_dir = data_dir / "consensus"


@pytest.fixture
def dataset():
    ds = (
        gvl.Dataset.open(ds_path, ref, deterministic=True)
        .with_len("ragged")
        .with_seqs("haplotypes")
    )
    return ds


def test_ds_haps(dataset: gvl.RaggedDataset[gvl.Ragged[np.bytes_], None, None, None]):
    for region, sample in product(range(dataset.n_regions), dataset.samples):
        c, s, e = dataset.regions.select("chrom", "chromStart", "chromEnd").row(region)
        # ragged (p)
        haps = dataset[region, sample]
        for h in range(2):
            actual = haps[h]
            fpath = f"sample_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(cons_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]))
            np.testing.assert_equal(
                actual,
                desired,
                f"region: {region}, sample: {sample}, hap: {h}, coords: {c}:{s + 1}-{e}",
            )