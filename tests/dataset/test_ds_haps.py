import genvarloader as gvl
import pysam
import numpy as np
from pathlib import Path
from itertools import product
import seqpro as sp
import pytest

cons_dir = Path("/cellar/users/dlaub/projects/GenVarLoader/tests/data/consensus")
ds_path = "/carter/users/dlaub/projects/GenVarLoader/tests/data/phased_dataset.gvl"
ref = "/cellar/users/dlaub/projects/GenVarLoader/tests/data/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa.bgz"


@pytest.fixture
def dataset():
    ds = gvl.Dataset.open(ds_path, ref, deterministic=True, return_annotations=True)
    return ds


def test_ds_haps(dataset: gvl.Dataset):
    for i, (region, sample) in enumerate(
        product(range(dataset.n_regions), dataset.samples)
    ):
        contig, start, end = dataset.input_regions.row(region)
        haps, vidx, positions = dataset[i].values()
        h1 = haps[0][(positions[0] >= start) & (positions[0] < end)]
        h2 = haps[1][(positions[1] >= start) & (positions[1] < end)]
        haps = [h1, h2]

        for h in range(1, 3):
            actual = haps[h - 1]
            fpath = f"sample_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(cons_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]))[
                    : dataset.output_length
                ]
            np.testing.assert_equal(
                actual, desired, f"region: {region}, sample: {sample}, hap: {h}"
            )
