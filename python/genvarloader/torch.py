from textwrap import dedent
from typing import Callable, Iterable, Optional, Union

import numpy as np
from loguru import logger

try:
    import torch
    import torch.utils.data as td

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_dataloader(
    dataset: td.Dataset,  # type: ignore
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[Union[td.Sampler, Iterable]] = None,  # type: ignore
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Optional[Callable] = None,
    multiprocessing_context: Optional[Callable] = None,
    generator: Optional[torch.Generator] = None,  # type: ignore
    *,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
):
    if num_workers > 1:
        logger.warning(
            dedent(
                """
                It is recommended to use num_workers <= 1 with GenVarLoader since it leverages
                extensive multithreading which has lower overhead than multiprocessing.
                """
            )
        )

    if sampler is None:
        sampler = get_sampler(len(dataset), batch_size, shuffle, drop_last)  # type: ignore

    return td.DataLoader(  # type: ignore
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
    if shuffle:
        inner_sampler = td.RandomSampler(np.arange(ds_len))  # type: ignore
    else:
        inner_sampler = td.SequentialSampler(np.arange(ds_len))  # type: ignore

    return td.BatchSampler(inner_sampler, batch_size, drop_last)  # type: ignore
