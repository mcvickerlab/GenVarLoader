from textwrap import dedent
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

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ._types import AnnotatedHaps

try:
    import torch
    import torch.utils.data as td

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if TYPE_CHECKING:
    import torch
    import torch.utils.data as td

    from ._dataset._impl import Dataset


def no_torch_error():
    raise ImportError(
        "PyTorch is not available. Please install PyTorch to use this function."
    )


def get_dataloader(
    dataset: "td.Dataset",
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[Union["td.Sampler", Iterable]] = None,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Optional[Callable] = None,
    multiprocessing_context: Optional[Callable] = None,
    generator: Optional["torch.Generator"] = None,
    *,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
):
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not available. Please install PyTorch to use this function."
        )

    if num_workers > 1:
        logger.warning(
            dedent(
                """
                It is recommended to use num_workers <= 1 with GenVarLoader since it leverages
                multithreading which has lower overhead than multiprocessing.
                """
            )
        )

    if sampler is None:
        sampler = get_sampler(len(dataset), batch_size, shuffle, drop_last)  # type: ignore

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


def get_sampler(
    ds_len: int, batch_size: int, shuffle: bool = False, drop_last: bool = False
):
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not available. Please install PyTorch to use this function."
        )

    if shuffle:
        inner_sampler = td.RandomSampler(range(ds_len))
    else:
        inner_sampler = td.SequentialSampler(range(ds_len))

    return td.BatchSampler(inner_sampler, batch_size, drop_last)


@overload
def _tensor_from_maybe_bytes(array: NDArray) -> "torch.Tensor": ...
@overload
def _tensor_from_maybe_bytes(array: AnnotatedHaps) -> dict[str, "torch.Tensor"]: ...
def _tensor_from_maybe_bytes(
    array: NDArray | AnnotatedHaps,
) -> "torch.Tensor" | Dict[str, "torch.Tensor"]:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "Could not import PyTorch. Please install PyTorch to use torch features."
        )
    if isinstance(array, AnnotatedHaps):
        return {
            "haps": _tensor_from_maybe_bytes(array.haps),
            "var_idxs": _tensor_from_maybe_bytes(array.var_idxs),
            "ref_coords": _tensor_from_maybe_bytes(array.ref_coords),
        }
    else:
        if array.dtype.type == np.bytes_:
            array = array.view(np.uint8)
        return torch.from_numpy(array)


if _TORCH_AVAILABLE:

    class TorchDataset(td.Dataset):
        def __init__(self, dataset: "Dataset"):
            if not _TORCH_AVAILABLE:
                raise ImportError(
                    "Could not import PyTorch. Please install PyTorch to use torch features."
                )
            self.dataset = dataset

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(
            self, idx: int | list[int]
        ) -> Union["torch.Tensor", Tuple["torch.Tensor", ...], Any]:
            r_idx, s_idx = np.unravel_index(idx, self.dataset.shape)
            batch = self.dataset[r_idx, s_idx]
            if isinstance(batch, np.ndarray):
                batch = _tensor_from_maybe_bytes(batch)
            elif isinstance(batch, tuple):
                batch = tuple(_tensor_from_maybe_bytes(b) for b in batch)
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
