from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .types import Reader
from .util import construct_virtual_data, process_bed

try:
    import torch
    import torch.utils.data as td

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GVLDataset(td.Dataset):  # pyright: ignore[reportPossiblyUnboundVariable]
    def __init__(
        self,
        *readers: Reader,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Could not import PyTorch. Please install PyTorch to use torch features."
            )
        self.readers = readers
        self.fixed_length = fixed_length
        self.transform = transform
        self.bed = process_bed(bed, fixed_length)
        self.virtual_data = construct_virtual_data(
            *readers, n_regions=self.bed.height, fixed_length=fixed_length
        )
        self.batch_dims = ["region"]
        if batch_dims is not None:
            self.batch_dims += batch_dims
        self._batch_shape = tuple(self.virtual_data.sizes[d] for d in self.batch_dims)

    def __len__(self):
        return np.prod([self.virtual_data.sizes[d] for d in self.batch_dims])

    def __getitem__(self, idx):
        _indices = np.unravel_index(idx, self._batch_shape)
        indices: Dict[str, int] = {
            d: _indices[i] for i, d in enumerate(self.batch_dims)
        }
        contig, start, end = self.bed.select("chrom", "chromStart", "chromEnd").row(
            indices["region"]
        )
        read_kwargs = {
            d: self.virtual_data.coords[d][[indices[d]]].to_numpy()
            for d in self.batch_dims[1:]
        }
        batch = {
            r.name: r.read(contig, start, end, **read_kwargs) for r in self.readers
        }
        if self.transform is not None:
            batch = self.transform(batch)
        return batch


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
