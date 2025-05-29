from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
from genvarloader._types import Idx, StrIdx
from numpy.typing import NDArray
from pytest import fixture
from pytest_cases import parametrize_with_cases

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REF = DATA_DIR / "fasta" / "hg38.fa.bgz"
DATASET = gvl.Dataset.open(DATA_DIR / "phased_dataset.vcf.gvl", REF)
REGIONS = DATASET._idxer.full_region_idxs
SAMPLES = DATASET._idxer.full_samples


@fixture(scope="session")
def dataset():
    ds = gvl.Dataset.open(DATA_DIR / "phased_dataset.vcf.gvl", REF)
    return ds


def idx_none():
    regions = None
    desired = REGIONS
    return regions, desired


def idx_scalar():
    regions = 0
    desired = REGIONS[regions]
    return regions, desired


def idx_neg_scalar():
    regions = -1
    desired = REGIONS[regions]
    return regions, desired


def idx_slice_none():
    regions = slice(None)
    desired = REGIONS[regions]
    return regions, desired


def idx_slice_start_none():
    regions = slice(1, None)
    desired = REGIONS[regions]
    return regions, desired


def idx_slice_none_stop():
    regions = slice(None, 2)
    desired = REGIONS[regions]
    return regions, desired


def idx_list():
    regions = [0, 1, 2]
    desired = REGIONS[regions]
    return regions, desired


def idx_array():
    regions = np.arange(3)
    desired = REGIONS[regions]
    return regions, desired


def idx_bool():
    regions = np.random.random(DATASET.n_regions) > 0.5
    desired = REGIONS[regions]
    return regions, desired


def smp_none():
    samples = None
    desired = SAMPLES
    return samples, desired


def smp_scalar():
    samples = 0
    desired = SAMPLES[samples]
    return samples, desired


def smp_neg_scalar():
    samples = -1
    desired = SAMPLES[samples]
    return samples, desired


def smp_slice_none():
    samples = slice(None)
    desired = SAMPLES[samples]
    return samples, desired


def smp_slice_start_none():
    samples = slice(1, None)
    desired = SAMPLES[samples]
    return samples, desired


def smp_slice_none_stop():
    samples = slice(None, 2)
    desired = SAMPLES[samples]
    return samples, desired


def smp_list():
    samples = [2, 0, 1]
    desired = SAMPLES[samples]
    return samples, desired


def smp_array():
    samples = np.arange(3)
    desired = SAMPLES[samples]
    return samples, desired


def smp_bool():
    samples = np.random.random(3) > 0.5
    desired = SAMPLES[samples]
    return samples, desired


def smp_str():
    samples = DATASET.samples[0]
    desired = np.atleast_1d(samples)
    return samples, desired


def smp_strs():
    samples = DATASET._idxer.full_samples[[2, 0, 1]]
    desired = samples
    return samples.tolist(), desired


def smp_series():
    samples = pl.Series(DATASET.samples[:2])
    desired = SAMPLES[:2]
    return samples, desired


@parametrize_with_cases("regions, desired_regions", cases=".", prefix="idx_")
@parametrize_with_cases("samples, desired_samples", cases=".", prefix="smp_")
def test_subset(
    dataset: gvl.Dataset,
    regions: Idx,
    samples: StrIdx,
    desired_regions: NDArray[np.integer],
    desired_samples: NDArray[np.str_],
):
    sub = dataset.subset_to(regions, samples)
    actual_regions = sub._idxer._r_idx
    actual_samples = np.atleast_1d(sub.samples)

    np.testing.assert_equal(actual_regions, desired_regions)
    np.testing.assert_equal(actual_samples, desired_samples)
