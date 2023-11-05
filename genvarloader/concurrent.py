from typing import Dict, Hashable, List, Tuple

import numpy as np
import ray
import xarray as xr
from attrs import define
from numpy.typing import NDArray

from .types import Reader

DataVarsLike = Dict[Hashable, Tuple[Tuple[Hashable, ...], NDArray]]


@define
class Buffer:
    buffer: xr.Dataset
    buffer_idx: NDArray[np.integer]
    dim_idxs: Dict[Hashable, List[int]]
    actor_idx: int
    instances_in_buffer: int
    len_unused_buffer: int
    idx_slice: slice

    def __init__(
        self,
        buffer: xr.Dataset,
        buffer_idx: NDArray[np.integer],
        dim_idxs: Dict[Hashable, List[int]],
        actor_idx: int,
    ) -> None:
        self.dim_idxs = dim_idxs
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
    dim_idxs: Dict[Hashable, List[int]]
    actor_idx: int

    def to_buffer(self, buffer: DataVarsLike):
        return Buffer(
            xr.Dataset(buffer), self.buffer_idx, self.dim_idxs, self.actor_idx
        )


@ray.remote
class ReaderActor:
    def __init__(self, *readers: Reader, actor_idx: int) -> None:
        self.readers = readers
        self.actor_idx = actor_idx

    @ray.method(num_returns=1)
    def read(
        self, contig: str, starts: NDArray[np.int64], ends: NDArray[np.int64], **kwargs
    ) -> Tuple[DataVarsLike, int]:
        buffer = {
            r.virtual_data.name: r.read(contig, starts, ends, strands=None, **kwargs)
            for r in self.readers
        }
        buffer = {name: (arr.dims, arr.to_numpy()) for name, arr in buffer.items()}
        return buffer, self.actor_idx
