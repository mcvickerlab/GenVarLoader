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

    # Convert the sorted-order genotype mask to display order before indexing ds.
    # ds._seqs.genotypes is indexed by the internal sorted region order; ds[mask]
    # expects a mask over the display order (ds.regions row order).
    # full_region_idxs[display_idx] = sorted_idx, so indexing by it remaps the mask.
    # In the new chr1/chr2 geography, chr1 carries variants too, so the old
    # `chrom == "chr1"` proxy no longer matches the no-variant set.
    no_var_display = no_var_regions[ds._idxer.full_region_idxs]
    annhaps = ds[no_var_display]
    # (b s p ~l) -> (b s p): take element 0 of each haplotype sequence
    import awkward as ak

    ref_coords_ak = annhaps.ref_coords.to_ak()
    actual_starts = ak.to_numpy(ref_coords_ak[..., 0])
    np.testing.assert_equal(actual_starts, desired_starts)
