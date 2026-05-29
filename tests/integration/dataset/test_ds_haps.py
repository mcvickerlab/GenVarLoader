from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import pytest
import seqpro as sp
from genvarloader._ragged import RaggedSeqs


@pytest.fixture(
    scope="session",
    params=["vcf", "pgen", "svar"],
)
def dataset(request, phased_vcf_gvl, phased_pgen_gvl, phased_svar_gvl, ref_fasta):
    gvl_path = {
        "vcf": phased_vcf_gvl,
        "pgen": phased_pgen_gvl,
        "svar": phased_svar_gvl,
    }[request.param]
    return (
        gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )


def test_ds_haps(dataset: gvl.RaggedDataset[RaggedSeqs, None], consensus_dir: Path):
    for region, sample in product(range(dataset.n_regions), dataset.samples):
        c, s, e, rc = dataset.regions.select(
            "chrom", "chromStart", "chromEnd", "strand"
        ).row(region)
        # ragged (p)
        haps = dataset[region, sample]
        for h in range(2):
            actual = sp.cast_seqs(haps[h])
            fpath = f"source_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(consensus_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]).upper())
            np.testing.assert_equal(
                actual,
                desired,
                f"region: {region}, sample: {sample}, hap: {h}, coords: {c}:{s + 1}-{e}",
            )
