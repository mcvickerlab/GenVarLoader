"""Smoke tests for the PyTorch integration."""

import math

import genvarloader as gvl  # eagerly loads numpy in the order torch expects
import pytest
from genvarloader._torch import get_dataloader, get_sampler


def _n_instances(batch) -> int:
    """Outer (instance) dimension of a dataloader batch, across the gvl output
    types that buffered/double_buffered modes can yield."""
    import numpy as np
    from seqpro.rag import Ragged

    if isinstance(batch, tuple):
        batch = batch[0]
    if isinstance(batch, np.ndarray):
        return batch.shape[0]
    if isinstance(batch, Ragged):
        return batch.shape[0]
    if hasattr(batch, "haps"):  # AnnotatedHaps / RaggedAnnotatedHaps
        return batch.haps.shape[0]
    return len(batch)  # ak.Array (RaggedVariants) and fallbacks


torch = pytest.importorskip("torch")  # module-level skip if torch missing


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


def test_torch_dataset_return_indices_appends_r_and_s(small_gvl_ds):
    """With return_indices=True, __getitem__ output ends with (r_idx, s_idx) arrays
    matching what ravel/unravel would produce for the queried flat index."""
    import numpy as np

    td_ds = small_gvl_ds.to_torch_dataset(return_indices=True, transform=None)
    # Use a batch of flat indices so r_idx/s_idx are arrays (matches __getitem__ contract).
    flat_idx = [0, 1]
    out = td_ds[flat_idx]
    # Should be a tuple ending with two index arrays.
    assert isinstance(out, tuple)
    assert len(out) >= 3, f"expected at least (data, r_idx, s_idx), got len={len(out)}"
    r_idx, s_idx = out[-2], out[-1]
    expected_r, expected_s = np.unravel_index(flat_idx, small_gvl_ds.shape)
    np.testing.assert_array_equal(r_idx, expected_r)
    np.testing.assert_array_equal(s_idx, expected_s)

    # And without return_indices, the same batch has fewer elements (or is not a tuple).
    td_no = small_gvl_ds.to_torch_dataset(return_indices=False, transform=None)
    out_no = td_no[flat_idx]
    if isinstance(out_no, tuple):
        assert len(out_no) == len(out) - 2
    # else: single tensor — definitely no appended indices.


def test_torch_dataset_transform_is_applied(small_gvl_ds):
    """A user-supplied transform receives the unpacked batch and its return value
    is what __getitem__ yields."""
    sentinel = object()
    captured = {}

    def transform(*batch):
        captured["nargs"] = len(batch)
        captured["batch"] = batch
        return sentinel

    td_ds = small_gvl_ds.to_torch_dataset(return_indices=False, transform=transform)
    out = td_ds[[0, 1]]
    assert out is sentinel
    assert captured["nargs"] >= 1

    # With return_indices=True, transform receives the data args + (r_idx, s_idx).
    captured.clear()
    td_with = small_gvl_ds.to_torch_dataset(return_indices=True, transform=transform)
    out2 = td_with[[0, 1]]
    assert out2 is sentinel
    # transform should now have been called with >=2 more args than the no-indices case
    # (last two are the unravelled r/s indices).
    assert captured["nargs"] >= 3


def test_default_mode_drop_last_true_does_not_crash(small_gvl_ds):
    """mode=None with drop_last=True must not raise. The BatchSampler applies
    drop_last; forwarding it to the DataLoader (which also gets batch_size=None)
    is what PyTorch rejected."""
    ds = small_gvl_ds.with_seqs("reference").with_tracks(False)
    N = len(ds)
    bs = next((c for c in range(2, N) if N % c), 1)
    assert N % bs != 0, "need an indivisible batch_size to exercise drop_last"
    dl = ds.to_dataloader(batch_size=bs, shuffle=False, drop_last=True)  # mode=None
    n_batches = sum(1 for _ in dl)
    assert n_batches == N // bs


@pytest.mark.parametrize("mode", ["buffered", "double_buffered"])
@pytest.mark.parametrize("drop_last", [False, True])
def test_buffered_modes_respect_drop_last(small_gvl_ds, mode, drop_last):
    ds = small_gvl_ds.with_seqs("reference").with_tracks(False)
    N = len(ds)
    bs = next((c for c in range(2, N) if N % c), 1)
    assert N % bs != 0, "need an indivisible batch_size to exercise drop_last"

    dl = ds.to_dataloader(batch_size=bs, shuffle=False, drop_last=drop_last, mode=mode)
    batches = list(dl)
    expected = N // bs if drop_last else math.ceil(N / bs)
    assert len(batches) == expected
    assert len(dl) == expected  # __len__ must match what iteration yields
    if not drop_last:
        # The final batch is the smaller, partial one.
        assert _n_instances(batches[-1]) == N % bs


def test_buffered_drop_last_false_with_custom_batch_sampler(small_gvl_ds):
    """DDP-shaped case: when the (r,s) indices come from a user-supplied sampler
    whose count is not a multiple of batch_size, the partial batch must survive."""
    import torch.utils.data as tud

    ds = small_gvl_ds.with_seqs("reference").with_tracks(False)
    N = len(ds)
    bs = next((c for c in range(2, N) if N % c), 1)
    sampler = tud.BatchSampler(tud.SequentialSampler(range(N)), bs, drop_last=False)
    dl = ds.to_dataloader(sampler=sampler, drop_last=False, mode="buffered")
    assert len(list(dl)) == math.ceil(N / bs)
