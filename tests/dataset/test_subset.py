from pathlib import Path

import genvarloader as gvl
import numpy as np
from genvarloader._utils import idx_like_to_array
from pytest import fixture
from pytest_cases import parametrize_with_cases

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REF = DATA_DIR / "fasta" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.bgz"
DATASET = gvl.Dataset.open(DATA_DIR / "phased_dataset.gvl", REF)


@fixture(scope="session")
def dataset():
    ds = gvl.Dataset.open(DATA_DIR / "phased_dataset.gvl", REF)
    return ds


def idx_none():
    return


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
    return np.random.random(8) > 0.5


def smp_none():
    return


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
    return [0, 1, 2]


def smp_array():
    return np.arange(3)


def smp_bool():
    return np.random.random(3) > 0.5


def smp_str():
    return DATASET.samples[0]


def smp_strs():
    return DATASET.samples[:2]


@parametrize_with_cases("regions", cases=".", prefix="idx_")
@parametrize_with_cases("samples", cases=".", prefix="smp_")
def test_subset(dataset: gvl.Dataset, regions, samples):
    actual = dataset.subset_to(regions, samples).shape

    if isinstance(regions, int):
        regions = [regions]
    elif regions is None:
        regions = slice(None)

    regions = idx_like_to_array(regions, dataset.n_regions)

    if isinstance(samples, int):
        samples = [samples]
    elif samples is None:
        samples = slice(None)
    elif isinstance(samples, str):
        samples = [dataset.samples.index(samples)]
    elif isinstance(samples, list) and isinstance(samples[0], str):
        samples = [dataset.samples.index(s) for s in samples]

    samples = idx_like_to_array(samples, dataset.n_samples)

    desired = (len(regions), len(samples))
    assert actual == desired
