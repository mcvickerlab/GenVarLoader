import genvarloader as gvl
import numpy as np
import polars as pl
from pytest import fixture
from pytest_cases import parametrize_with_cases


@fixture(scope="session")
def dataset(phased_vcf_gvl, ref_fasta):
    return gvl.Dataset.open(phased_vcf_gvl, ref_fasta)


def idx_none():
    return None


def idx_scalar():
    return 0


def idx_neg_scalar():
    return -1


def idx_slice_none():
    return slice(None)


def idx_slice_start_none():
    return slice(1, None)


def idx_slice_none_stop():
    return slice(None, 2)


def idx_list():
    return [0, 1, 2]


def idx_array():
    return np.arange(3)


def idx_bool():
    # Returned as a sentinel; test body will create a bool array of the right size.
    return "bool"


def smp_none():
    return None


def smp_scalar():
    return 0


def smp_neg_scalar():
    return -1


def smp_slice_none():
    return slice(None)


def smp_slice_start_none():
    return slice(1, None)


def smp_slice_none_stop():
    return slice(None, 2)


def smp_list():
    return [2, 0, 1]


def smp_array():
    return np.arange(3)


def smp_bool():
    # Returned as a sentinel; test body will create a bool array of the right size.
    return "bool"


def smp_str():
    # Returned as a sentinel; test body will use dataset.samples[0].
    return "str"


def smp_strs():
    # Returned as a sentinel; test body will use dataset._idxer.full_samples[[2, 0, 1]].tolist().
    return "strs"


def smp_series():
    # Returned as a sentinel; test body will use pl.Series(dataset.samples[:2]).
    return "series"


@parametrize_with_cases("regions", cases=".", prefix="idx_")
@parametrize_with_cases("samples", cases=".", prefix="smp_")
def test_subset(
    dataset: gvl.Dataset,
    regions,
    samples,
):
    full_regions = dataset._idxer.full_region_idxs
    full_samples = dataset._idxer.full_samples

    # Resolve sentinel values that depend on dataset shape
    if isinstance(regions, str) and regions == "bool":
        regions = np.random.random(dataset.n_regions) > 0.5

    if isinstance(samples, str) and samples == "bool":
        samples = np.random.random(len(full_samples)) > 0.5
    elif isinstance(samples, str) and samples == "str":
        samples = dataset.samples[0]
    elif isinstance(samples, str) and samples == "strs":
        samples = full_samples[[2, 0, 1]].tolist()
    elif isinstance(samples, str) and samples == "series":
        samples = pl.Series(dataset.samples[:2])

    # Compute desired values
    if regions is None:
        desired_regions = full_regions
    else:
        desired_regions = full_regions[regions]

    if samples is None:
        desired_samples = full_samples
    elif isinstance(samples, str):
        # scalar string sample name
        desired_samples = np.atleast_1d(np.array(samples))
    elif isinstance(samples, list):
        if len(samples) > 0 and isinstance(samples[0], str):
            desired_samples = np.array(samples)
        else:
            desired_samples = full_samples[samples]
    elif isinstance(samples, pl.Series):
        desired_samples = full_samples[:2]
    else:
        desired_samples = full_samples[samples]

    sub = dataset.subset_to(regions, samples)
    actual_regions = sub._idxer._r_idx
    actual_samples = np.atleast_1d(sub.samples)

    np.testing.assert_equal(actual_regions, desired_regions)
    np.testing.assert_equal(actual_samples, desired_samples)
