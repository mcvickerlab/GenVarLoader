from copy import deepcopy
from pathlib import Path

import genvarloader as gvl
import numpy as np
from einops import repeat
from genvarloader._dataset._reconstruct import Haps
from pytest_cases import parametrize_with_cases

DDIR = Path(__file__).parent.parent
REF = DDIR / "data" / "fasta" / "hg38.fa.bgz"


def ds_vcf():
    return gvl.Dataset.open(DDIR / "data" / "phased_dataset.vcf.gvl", REF)


def ds_pgen():
    return gvl.Dataset.open(DDIR / "data" / "phased_dataset.pgen.gvl", REF)


def ds_svar():
    return gvl.Dataset.open(DDIR / "data" / "phased_dataset.svar.gvl", REF)


@parametrize_with_cases("dataset", cases=".", prefix="ds_")
def test_jitter(dataset: gvl.RaggedDataset):
    ds = (
        dataset.with_settings(jitter=dataset.max_jitter, rc_neg=False)
        .with_seqs("annotated")
        .with_tracks(False)
    )
    assert isinstance(ds._seqs, Haps)
    no_var_regions = ds._seqs.genotypes.lengths.sum((1, 2)) == 0
    rng = deepcopy(ds._rng)
    jitter = rng.integers(
        -ds.jitter, ds.jitter + 1, size=(no_var_regions.sum(), ds.n_samples)
    )

    desired_starts = ds._full_regions[no_var_regions, 1, None] + jitter
    desired_starts = repeat(
        desired_starts, "b s -> b s p", p=ds._seqs.genotypes.shape[-2]
    )

    no_var_regions = (ds.regions["chrom"] == "chr1").to_numpy()
    annhaps = ds[no_var_regions]
    # (b s p ~l) -> (b s p)
    actual_starts = annhaps.ref_coords[..., 0].to_numpy()  # type: ignore
    np.testing.assert_equal(actual_starts, desired_starts)
