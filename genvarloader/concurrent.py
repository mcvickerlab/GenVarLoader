from typing import Dict, List, Tuple

import numpy as np
import ray
import xarray as xr
from attrs import define, field
from numpy.typing import NDArray

from .types import Reader

DataVarsLike = Dict[str, Tuple[Tuple[str, ...], NDArray]]


@define
class Buffer:
    buffer: xr.Dataset
    buffer_idx: NDArray[np.integer]
    dim_idxs: Dict[str, List[int]]
    actor_idx: int
    instances_in_buffer: int = field(init=False)
    len_unused_buffer: int = field(init=False)
    idx_slice: slice = field(init=False)

    def __attrs_post_init__(self):
        self.instances_in_buffer = len(self)
        self.len_unused_buffer = len(self)
        self.idx_slice = slice(0, 0)

    def __len__(self):
        return len(self.buffer_idx)


@define
class BufferMeta:
    buffer_idx: NDArray[np.integer]
    dim_idxs: Dict[str, List[int]]
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
            r.name: r.read(contig, starts, ends, strands=None, **kwargs)
            for r in self.readers
        }
        buffer = {name: (arr.dims, arr.to_numpy()) for name, arr in buffer.items()}
        return buffer, self.actor_idx
