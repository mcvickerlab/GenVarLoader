from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Union,
    overload,
)

import awkward as ak
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from seqpro._ragged import Ragged

from ._ragged import is_rag_dtype
from ._types import AnnotatedHaps

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


def requires_torch(func_or_class):
    if TORCH_AVAILABLE:
        return func_or_class
    else:
        return no_torch_error


@requires_torch
def get_dataloader(
    dataset: td.Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[Union[td.Sampler, Iterable]] = None,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Optional[Callable] = None,
    multiprocessing_context: Optional[Callable] = None,
    generator: Optional[torch.Generator] = None,
    *,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
):
    if num_workers > 1:
        logger.warning(
            "It is recommended to use num_workers <= 1 with GenVarLoader since it leverages"
            " multithreading which has lower overhead than multiprocessing."
        )

    if sampler is None:
        sampler = get_sampler(
            len(dataset),  # type: ignore
            batch_size,
            shuffle,
            drop_last,
        )

    return td.DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory_device=pin_memory_device,
    )


@requires_torch
def get_sampler(
    ds_len: int, batch_size: int, shuffle: bool = False, drop_last: bool = False
):
    if shuffle:
        inner_sampler = td.RandomSampler(range(ds_len))
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
) -> torch.Tensor | Dict[str, torch.Tensor]:
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
        rag = Ragged.from_awkward(rag)

    if is_rag_dtype(rag, np.bytes_):
        rag = rag.view(np.uint8)

    values = torch.from_numpy(rag.data)
    lengths = torch.from_numpy(rag.lengths)
    nt = torch.nested.nested_tensor_from_jagged(
        values, lengths=lengths, max_seqlen=rag.lengths.max()
    )
    return nt


if TORCH_AVAILABLE:

    class TorchDataset(td.Dataset):
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
        ) -> torch.Tensor | Tuple[torch.Tensor, ...] | Any:
            r_idx, s_idx = np.unravel_index(idx, self.dataset.shape)
            batch = self.dataset[r_idx, s_idx]

            if not isinstance(batch, tuple):
                batch = (batch,)
                single_item = True
            else:
                single_item = False

            if self.include_indices:
                batch = (*batch, r_idx, s_idx)

            if self.transform is not None:
                batch = self.transform(*batch)

            batch = tuple(tensor_from_maybe_bytes(b) for b in batch)
            if single_item:
                batch = batch[0]

            return batch

    class StratifiedSampler(td.Sampler):
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

        def __init__(
            self,
            n_regions: int,
            n_samples: int,
            shuffle: bool = False,
            seed: Optional[int] = None,
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
    TorchDataset = no_torch_error  # type: ignore
    StratifiedSampler = no_torch_error  # type: ignore
