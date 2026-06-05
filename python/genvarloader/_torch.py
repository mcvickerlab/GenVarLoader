from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, overload

import awkward as ak
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from seqpro.rag import Ragged, is_rag_dtype

from ._types import AnnotatedHaps
from ._utils import lengths_to_offsets

try:
    import torch
    import torch.utils.data as td

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TYPE_CHECKING:
    import torch
    import torch.utils.data as td

    from ._dataset._impl import Dataset


def no_torch_error(*args, **kwargs):
    raise ImportError(
        "PyTorch is not available. Please install PyTorch to use this function/class."
    )


def requires_torch(func):
    if TORCH_AVAILABLE:
        return func
    else:
        return no_torch_error


def _resolve_buffered_inputs(
    dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    sampler,
    generator,
    buffer_bytes: int,
    n_slots: int,
    include_offsets: bool = False,
):
    """Compute flat (r_idx, s_idx) epoch order, bytes_per_instance table, and slot_bytes.

    ``dataset`` must be a raw gvl Dataset (not a TorchDataset wrapper) so that
    ``_output_bytes_per_instance`` and ``.shape`` are available.

    ``include_offsets`` adds the serialized offset/lengths overhead to the
    byte table. Required for ``double_buffered``, whose fixed-size shm slots
    must hold the full serialized chunk (payload + offsets); ``buffered`` packs
    by payload alone since it never serializes into a fixed slot.
    """
    # 1) Resolve full epoch order from the BatchSampler.
    if sampler is None:
        sampler = get_sampler(
            len(dataset), batch_size, shuffle, drop_last, generator=generator
        )
    flat = []
    for batch in sampler:
        flat.extend(batch)
    flat = np.asarray(flat, dtype=np.int64)
    # Only drop the trailing partial batch when the caller asked for it. When
    # drop_last=False, keep it -- ChunkPlanner emits it as a partial batch.
    if drop_last:
        n_keep = (len(flat) // batch_size) * batch_size
        flat = flat[:n_keep]
    r_idx, s_idx = np.unravel_index(flat, dataset.shape)

    # 2) Pre-pass: exact bytes per instance for the entire (n_regions, n_samples) grid.
    # Pass None, None so parse_idx uses slice(None), slice(None) → "basic" indexing →
    # out_reshape=(n_regions, n_samples), and _output_bytes_per_instance returns a 2-D array.
    bpi = dataset._output_bytes_per_instance(
        None, None, include_offsets=include_offsets
    )
    # Ensure exactly (n_regions, n_samples) 2-D regardless of squeeze/reshape behavior.
    bpi = np.asarray(bpi, dtype=np.int64).reshape(dataset.shape)

    slot_bytes = buffer_bytes // n_slots
    return r_idx, s_idx, bpi, slot_bytes, sampler


@requires_torch
def get_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: td.Sampler | Iterable | None = None,
    num_workers: int = 0,
    collate_fn: Callable | None = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Callable | None = None,
    multiprocessing_context: Callable | None = None,
    generator: torch.Generator | None = None,
    *,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
    mode: str | None = None,
    buffer_bytes: int = 2 * 1024**3,
    copy: bool = True,
    heartbeat_seconds: float = 60.0,
):
    if mode is None:
        # Existing path unchanged.
        if num_workers > 1:
            logger.warning(
                "It is recommended to use num_workers <= 1 with GenVarLoader since it leverages"
                " multithreading which has lower overhead than multiprocessing."
            )

        if sampler is None:
            sampler = get_sampler(
                len(dataset),
                batch_size,
                shuffle,
                drop_last,
                generator=generator,
            )

        return td.DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    if mode not in {"buffered", "double_buffered"}:
        raise ValueError(
            f"unknown mode={mode!r}; expected None, 'buffered', or 'double_buffered'"
        )
    if num_workers > 0:
        raise ValueError(
            f"mode={mode!r} is incompatible with num_workers>0; "
            "the loader IS the concurrency strategy"
        )

    # When the caller passes a BatchSampler directly, use its batch_size so that
    # the buffered loader re-batches at the same granularity the sampler intended.
    if isinstance(sampler, td.BatchSampler):
        batch_size = sampler.batch_size

    n_slots = 1 if mode == "buffered" else 2
    r_idx, s_idx, bpi, slot_bytes, _sampler = _resolve_buffered_inputs(
        dataset,
        batch_size,
        shuffle,
        drop_last,
        sampler,
        generator,
        buffer_bytes,
        n_slots,
        include_offsets=(mode == "double_buffered"),
    )

    if mode == "buffered":
        from ._buffered_loader import make_buffered_dataset

        inner_ds = make_buffered_dataset(
            dataset, batch_size, slot_bytes, bpi, r_idx, s_idx
        )
    else:
        from ._double_buffered_loader import make_double_buffered_dataset

        inner_ds = make_double_buffered_dataset(
            dataset,
            batch_size,
            slot_bytes,
            bpi,
            r_idx,
            s_idx,
            copy=copy,
            heartbeat_seconds=heartbeat_seconds,
        )

    return td.DataLoader(
        inner_ds,
        batch_size=None,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )


@requires_torch
def get_sampler(
    ds_len: int,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
):
    if shuffle:
        inner_sampler = td.RandomSampler(range(ds_len), generator=generator)
    else:
        inner_sampler = td.SequentialSampler(range(ds_len))

    return td.BatchSampler(inner_sampler, batch_size, drop_last)


@overload
def tensor_from_maybe_bytes(array: NDArray) -> torch.Tensor: ...
@overload
def tensor_from_maybe_bytes(array: AnnotatedHaps) -> dict[str, torch.Tensor]: ...
@requires_torch
def tensor_from_maybe_bytes(
    array: NDArray | AnnotatedHaps,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(array, AnnotatedHaps):
        return {
            "haps": tensor_from_maybe_bytes(array.haps),
            "var_idxs": tensor_from_maybe_bytes(array.var_idxs),
            "ref_coords": tensor_from_maybe_bytes(array.ref_coords),
        }
    else:
        if array.dtype.type == np.bytes_:
            array = array.view(np.uint8)
        return torch.from_numpy(array)


@requires_torch
def to_nested_tensor(rag: Ragged | ak.Array) -> torch.Tensor:
    """Convert a Ragged array to a PyTorch `nested tensor <https://pytorch.org/docs/stable/nested.html>`_. Will cast byte arrays
    (dtype "S1") to uint8.

    Parameters
    ----------
    rag
        Ragged array to convert.
    """
    if isinstance(rag, ak.Array):
        rag = Ragged(rag)

    if is_rag_dtype(rag, np.bytes_):
        rag = rag.view(np.uint8)

    values = torch.from_numpy(rag.data)
    offsets = torch.from_numpy(lengths_to_offsets(rag.lengths, np.int32))
    nt = torch.nested.nested_tensor_from_jagged(
        values, offsets, max_seqlen=rag.lengths.max()
    )
    return nt


if TORCH_AVAILABLE:

    class TorchDataset(td.Dataset):
        dataset: Dataset
        include_indices: bool
        transform: Callable | None

        def __init__(
            self,
            dataset: Dataset,
            include_indices: bool,
            transform: Callable | None,
        ):
            self.dataset = dataset
            self.include_indices = include_indices
            self.transform = transform

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(
            self, idx: int | list[int]
        ) -> torch.Tensor | tuple[torch.Tensor, ...] | Any:
            r_idx, s_idx = np.unravel_index(idx, self.dataset.shape)
            batch = self.dataset[r_idx, s_idx]

            if not isinstance(batch, tuple):
                batch = (batch,)

            if self.include_indices:
                batch = (*batch, r_idx, s_idx)

            if self.transform is not None:
                batch = self.transform(*batch)
            elif len(batch) == 1:
                batch = batch[0]

            return batch

    class StratifiedSampler(td.Sampler[np.intp]):
        """Stratified sampler for GVL datasets. This ensures that each batch has the most diversity of samples possible.

        Parameters
        ----------
        n_regions : int
            Number of regions.
        n_samples : int
            Number of samples.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default False.
        seed : int, optional
            Random seed, by default None.

        Examples
        --------
        >>> n_regions = 10
        >>> n_samples = 100
        >>> batch_size = 7
        >>> sampler = torch.utils.data.BatchSampler(
                gvl.StratifiedSampler(n_regions, n_samples),
                batch_size,
                drop_last=True,
            )
        >>> dl = ds.to_dataloader(sampler=sampler)
        """

        ds_idx: NDArray[np.intp]

        def __init__(
            self,
            n_regions: int,
            n_samples: int,
            shuffle: bool = False,
            seed: int | None = None,
        ):
            rng = np.random.default_rng(seed)
            if shuffle:
                r_idx = rng.permutation(n_regions)
                s_idx = rng.permutation(n_samples)
            else:
                r_idx = np.arange(n_regions)
                s_idx = np.arange(n_samples)
            self.ds_idx = np.ravel_multi_index(
                (r_idx[:, None], s_idx), (n_regions, n_samples)
            ).ravel()

        def __len__(self):
            return len(self.ds_idx)

        def __iter__(self):
            return iter(self.ds_idx)

else:
    TorchDataset = no_torch_error
    StratifiedSampler = no_torch_error
