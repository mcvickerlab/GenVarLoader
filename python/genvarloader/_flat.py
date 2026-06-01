"""Flat-buffer ragged transport used inside the getitem hot path.

`_Flat` is a pure-numpy `(data, offsets, shape)` container. Unlike seqpro
`Ragged` it never wraps an awkward array, so operating on it runs no awkward
kernels. It converts to seqpro `Ragged` only via `to_ragged()`, called at the
getitem return boundary when the caller requested ragged output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from seqpro.rag import RDTYPE_co as RDTYPE
from seqpro.rag import Ragged
from seqpro.rag import to_padded as _sp_to_padded


@dataclass(slots=True, frozen=True)
class _Flat(Generic[RDTYPE]):
    data: NDArray
    offsets: NDArray[np.int64]
    shape: tuple[int | None, ...]  # outer fixed dims; exactly one None (ragged axis)

    @classmethod
    def from_offsets(
        cls, data: NDArray, shape: tuple[int | None, ...], offsets: NDArray
    ) -> "_Flat":
        return cls(data, np.asarray(offsets, np.int64), tuple(shape))

    @property
    def rag_dim(self) -> int:
        return self.shape.index(None)

    @property
    def n_rows(self) -> int:
        return int(np.prod([d for d in self.shape if d is not None], dtype=np.int64))

    def view(self, dtype: Any) -> "_Flat":
        return _Flat(self.data.view(dtype), self.offsets, self.shape)

    def to_ragged(self) -> Ragged:
        return Ragged.from_offsets(self.data, self.shape, self.offsets)

    def to_fixed(self, length: int) -> NDArray:
        """Densify when every row has exactly `length` elements: pure reshape."""
        outer = tuple(d for d in self.shape if d is not None)
        return self.data.reshape(*outer, length)

    def to_padded(self, pad_value: Any) -> NDArray:
        """Variable-length densify via the flat seqpro kernel."""
        return _sp_to_padded(self.to_ragged(), pad_value)

    def reshape(self, shape: int | tuple[int, ...]) -> "_Flat":
        if isinstance(shape, int):
            shape = (shape,)
        new = tuple(shape) + (None,)
        return _Flat(self.data, self.offsets, new)

    def squeeze(self, axis: int | None = None) -> "_Flat":
        outer = [d for d in self.shape if d is not None]
        if axis is None:
            outer = [d for d in outer if d != 1]
        else:
            if outer[axis] != 1:
                raise ValueError(f"cannot squeeze axis {axis} with size {outer[axis]}")
            del outer[axis]
        return _Flat(self.data, self.offsets, (*outer, None))
