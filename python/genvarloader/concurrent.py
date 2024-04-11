from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from attrs import define, field
from numpy.typing import NDArray

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
