from pathlib import Path

import genvarloader as gvl
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import basic_indices
from pytest import fixture

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET = gvl.Dataset.open(DATA_DIR / "phased_dataset.gvl")


@fixture
def dataset():
    ds = gvl.Dataset.open(DATA_DIR / "phased_dataset.gvl")
    return ds


@given(basic_indices(DATASET.shape, min_dims=2, allow_ellipsis=False))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_subset(dataset: gvl.Dataset, idx):
    regions, samples = idx
    actual = dataset.subset_to(regions, samples).shape

    dummy_array = np.arange(len(dataset)).reshape(dataset.shape)

    if isinstance(regions, int):
        regions = (regions,)
    elif regions is None:
        regions = slice(None)

    if isinstance(samples, int):
        samples = (samples,)
    elif samples is None:
        samples = slice(None)

    desired = dummy_array[regions, samples].shape
    assert actual == desired
