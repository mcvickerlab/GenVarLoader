from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import pytest
import seqpro as sp
from genvarloader._ragged import RaggedSeqs

pytestmark = pytest.mark.slow


@pytest.fixture(
    scope="session",
    params=["bcf", "pgen", "svar"],
)
def dataset(request, kg_bcf_gvl, kg_pgen_gvl, kg_svar_gvl, ref_fasta):
    gvl_path = {
        "bcf": kg_bcf_gvl,
        "pgen": kg_pgen_gvl,
        "svar": kg_svar_gvl,
    }[request.param]
    return (
        gvl.Dataset
        .open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )


def test_ds_haps_1kg(dataset: gvl.RaggedDataset[RaggedSeqs, None], data_dir: Path):
    cons_dir = data_dir / "1kg_consensus"
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
