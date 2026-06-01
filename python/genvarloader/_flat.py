"""Flat-buffer ragged transport used inside the getitem hot path.

`_Flat` is a pure-numpy `(data, offsets, shape)` container. Unlike seqpro
`Ragged` it never wraps an awkward array, so operating on it runs no awkward
kernels. It converts to seqpro `Ragged` only via `to_ragged()`, called at the
getitem return boundary when the caller requested ragged output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

import numba as nb
import numpy as np
from numpy.typing import NDArray
from seqpro.rag import RDTYPE_co as RDTYPE
from seqpro.rag import Ragged
from seqpro.rag import to_padded as _sp_to_padded


@nb.njit(parallel=True, cache=True)
def _reverse_rows_masked(data, offsets, mask):  # pragma: no cover - njit
    n = mask.shape[0]
    for i in nb.prange(n):
        if mask[i]:
            lo = offsets[i]
            hi = offsets[i + 1] - 1
            while lo < hi:
                tmp = data[lo]
                data[lo] = data[hi]
                data[hi] = tmp
                lo += 1
                hi -= 1


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

    def reverse_masked(self, mask: NDArray[np.bool_], comp: NDArray | None = None) -> "_Flat":
        """Reverse (or DNA reverse-complement) the `mask`-selected rows.

        ``comp`` is a **mode selector**, not a complement LUT that gets applied:

        * Pass ``comp=None`` (default) for a plain reversal of each selected row.
        * Pass ``comp=_COMP`` (gvl's standard ACGT→TGCA lookup) to enable DNA
          reverse-complement mode.  The LUT that is actually applied is always
          gvl's internal ``_COMP`` (via ``reverse_complement_masked``); passing a
          *different* array is not supported and will raise ``ValueError``.

        ``mask`` is one entry per outer query; it is replicated across any inner
        fixed axes in C order to produce one entry per flattened ragged row,
        matching the ``ak.where`` broadcast it replaces.
        """
        m = np.ascontiguousarray(mask, np.bool_).reshape(-1)
        if m.size != self.n_rows:
            factor, rem = divmod(self.n_rows, m.size)
            if rem != 0:
                raise ValueError(
                    f"mask has {m.size} entries but {self.n_rows} rows "
                    "(not an integer multiple)."
                )
            m = np.repeat(m, factor)
        if comp is not None:
            # DNA reverse-complement via the flat seqpro kernel (reuses gvl's LUT).
            # seqpro requires S1 dtype; view uint8 as S1 for the call, then keep uint8.
            from ._ragged import _COMP, reverse_complement_masked

            if comp is not _COMP and not np.array_equal(comp, _COMP):
                raise ValueError(
                    "reverse_masked only supports gvl's standard complement LUT (_COMP); "
                    "pass comp=_COMP to enable DNA reverse-complement or comp=None for a plain reverse."
                )
            s1_flat = self if self.data.dtype == np.dtype("S1") else self.view("S1")
            rag = reverse_complement_masked(s1_flat.to_ragged(), m)
            result_data = np.asarray(rag.data)
            if self.data.dtype != np.dtype("S1"):
                result_data = result_data.view(self.data.dtype)
            return _Flat(result_data, self.offsets, self.shape)
        _reverse_rows_masked(self.data, self.offsets, m)
        return self
