"""Edge-case dataset shapes: empty selection, single-sample, single-region."""

import numpy as np
import pytest

import genvarloader as gvl


@pytest.fixture
def ds(phased_vcf_gvl, ref_fasta):
    return gvl.Dataset.open(phased_vcf_gvl, ref_fasta)


def test_subset_to_single_region(ds: gvl.Dataset):
    sub = ds.subset_to(regions=[0])
    assert sub.n_regions == 1
    assert len(sub) == sub.n_samples
    out = sub[0, ds.samples[0]]
    assert out is not None


def test_subset_to_single_sample(ds: gvl.Dataset):
    sub = ds.subset_to(samples=[ds.samples[0]])
    assert sub.n_samples == 1
    assert len(sub) == sub.n_regions
    out = sub[0, ds.samples[0]]
    assert out is not None


def test_subset_to_empty_regions_raises_or_yields_empty(ds: gvl.Dataset):
    try:
        sub = ds.subset_to(regions=np.array([], dtype=int))
    except (ValueError, IndexError):
        return
    assert sub.n_regions == 0
    assert len(sub) == 0
