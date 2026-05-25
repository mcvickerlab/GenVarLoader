from copy import deepcopy

import genvarloader as gvl
import numpy as np
import pytest
from einops import repeat
from genvarloader._dataset._reconstruct import Haps


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
    return gvl.Dataset.open(gvl_path, ref_fasta)


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
