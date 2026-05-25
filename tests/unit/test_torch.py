"""Smoke tests for the PyTorch integration."""
import numpy as np  # noqa: F401  # import before torch to avoid numpy/torch ABI abort in py310 env
import pytest

torch = pytest.importorskip("torch")  # module-level skip if torch missing

import genvarloader as gvl  # noqa: E402  # must follow importorskip
from genvarloader._torch import get_dataloader, get_sampler  # noqa: E402


@pytest.fixture(scope="module")
def small_torch_ds(phased_vcf_gvl, reference):
    """A small Dataset configured for torch usage (as a TorchDataset)."""
    ds = gvl.Dataset.open(phased_vcf_gvl, reference=reference)
    # Dataset must be converted to a torch-compatible Dataset via to_torch_dataset.
    return ds.to_torch_dataset(return_indices=False, transform=None)


@pytest.fixture(scope="module")
def small_gvl_ds(phased_vcf_gvl, reference):
    """Raw gvl Dataset (not yet wrapped as a TorchDataset)."""
    return gvl.Dataset.open(phased_vcf_gvl, reference=reference)


def test_get_dataloader_smoke(small_torch_ds):
    """get_dataloader yields at least one batch with non-zero length."""
    dl = get_dataloader(small_torch_ds, batch_size=2, shuffle=False, num_workers=0)
    batches = list(dl)
    assert len(batches) >= 1


def test_get_dataloader_via_dataset_method(small_gvl_ds):
    """Dataset.to_dataloader() is the high-level convenience wrapper."""
    dl = small_gvl_ds.to_dataloader(batch_size=2, shuffle=False, num_workers=0)
    batches = list(dl)
    assert len(batches) >= 1


def test_get_sampler_no_shuffle_is_sequential(small_torch_ds):
    """Without shuffle, sampler yields 0..N-1 in order."""
    n = len(small_torch_ds)
    sampler = get_sampler(n, batch_size=1, shuffle=False)
    indices = list(sampler)
    # get_sampler returns a BatchSampler → each element is a list[int]
    if indices and isinstance(indices[0], list):
        indices = [i for batch in indices for i in batch]
    assert indices == list(range(n))


def test_get_sampler_shuffle_changes_order(small_torch_ds):
    """With shuffle=True, sampler ordering should NOT always be sequential."""
    n = len(small_torch_ds)
    if n < 2:
        pytest.skip("Dataset too small to detect shuffle")

    sequential = list(range(n))
    matches = 0
    for _ in range(5):
        s = get_sampler(n, batch_size=1, shuffle=True)
        idxs = list(s)
        if idxs and isinstance(idxs[0], list):
            idxs = [i for b in idxs for i in b]
        if idxs == sequential:
            matches += 1
    # At least one of 5 shuffles should be non-identity.
    assert matches < 5


def test_get_sampler_batch_size_groups_indices(small_torch_ds):
    """BatchSampler groups indices into batches of the requested size."""
    n = len(small_torch_ds)
    batch_size = max(2, n // 2)
    sampler = get_sampler(n, batch_size=batch_size, shuffle=False)
    batches = list(sampler)
    # Every batch (except possibly the last) should have exactly batch_size elements.
    assert all(len(b) == batch_size for b in batches[:-1])


def test_torch_dataset_len_matches_gvl(small_gvl_ds, small_torch_ds):
    """TorchDataset.__len__ matches gvl Dataset n_regions * n_samples."""
    expected = small_gvl_ds.n_regions * small_gvl_ds.n_samples
    assert len(small_torch_ds) == expected
