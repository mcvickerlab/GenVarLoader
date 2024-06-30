from textwrap import dedent
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

import numpy as np
from loguru import logger

try:
    import torch
    import torch.utils.data as td

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if TYPE_CHECKING:
    import torch.utils.data as td

__all__ = []


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
        inner_sampler = td.RandomSampler(np.arange(ds_len))
    else:
        inner_sampler = td.SequentialSampler(np.arange(ds_len))

    return td.BatchSampler(inner_sampler, batch_size, drop_last)
