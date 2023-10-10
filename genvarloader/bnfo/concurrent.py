from typing import Dict, Hashable, Tuple

import numba as nb
import numpy as np
import ray
import xarray as xr
from attrs import define
from numpy.typing import NDArray

from .types import Reader

DataVarsLike = Dict[Hashable, Tuple[Tuple[Hashable, ...], NDArray]]


@nb.njit(nogil=True, cache=True)
def partition_regions(
    starts: NDArray[np.int64], ends: NDArray[np.int64], max_length: int
):
    partitions = np.zeros_like(starts)
    partition = 0
    curr_length = ends[0] - starts[0]
    for i in range(1, len(partitions)):
        curr_length += ends[i] - ends[i - 1]
        if curr_length > max_length:
            partition += 1
            curr_length = ends[i] - starts[i]
        partitions[i] = partition
    return partitions


@define
class Buffer:
    buffer: xr.Dataset
    buffer_idx: NDArray[np.integer]
    dim_slices: Dict[str, slice]
    actor_idx: int
    instances_in_buffer: int
    len_unused_buffer: int
    idx_slice: slice

    def __init__(
        self, buffer, buffer_idx, dim_slices: Dict[str, slice], actor_idx: int
    ) -> None:
        self.dim_slices = dim_slices
        self.buffer = buffer
        self.buffer_idx = buffer_idx
        self.instances_in_buffer = len(self.buffer_idx)
        self.len_unused_buffer = self.instances_in_buffer
        self.idx_slice = slice(0, 0)
        self.actor_idx = actor_idx

    def __len__(self):
        return self.instances_in_buffer


@define
class BufferMeta:
    buffer_idx: NDArray[np.integer]
    dim_slices: Dict[str, slice]
    actor_idx: int

    def to_buffer(self, buffer: DataVarsLike):
        return Buffer(
            xr.Dataset(buffer), self.buffer_idx, self.dim_slices, self.actor_idx
        )


@ray.remote
class ReaderActor:
    def __init__(self, *readers: Reader, actor_idx: int) -> None:
        self.readers = readers
        self.actor_idx = actor_idx

    @ray.method(num_returns=1)
    def read(
        self, contig: str, start: int, end: int, **kwargs
    ) -> Tuple[DataVarsLike, int]:
        buffer = {
            r.virtual_data.name: r.read(contig, start, end, **kwargs)
            for r in self.readers
        }
        buffer = {name: (arr.dims, arr.to_numpy()) for name, arr in buffer.items()}
        return buffer, self.actor_idx
