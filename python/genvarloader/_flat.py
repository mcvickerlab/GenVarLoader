"""Flat-buffer ragged transport used inside the getitem hot path.

`_Flat` is a pure-numpy `(data, offsets, shape)` container that stays in
numpy throughout the hot path. It converts to seqpro `Ragged` only via
`to_ragged()`, called at the getitem return boundary when the caller
requested ragged output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from seqpro.rag import RDTYPE_co as RDTYPE
from seqpro.rag import Ragged
from seqpro.rag import to_padded as _sp_to_padded


def _reverse_rows_masked(data, offsets, mask):
    n = mask.shape[0]
    for i in range(n):
        if mask[i]:
            s, e = int(offsets[i]), int(offsets[i + 1])
            data[s:e] = data[s:e][::-1]


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

    def __getitem__(self, key) -> "_Flat":
        """Slice the leading (instance) axis. Supports a `slice` with step 1.

        Rebases offsets so the result is a self-contained `_Flat`. Inner fixed
        dims (e.g. ploidy) are preserved; `groups_per_inst` accounts for them.
        """
        if not isinstance(key, slice):
            raise TypeError(
                f"_Flat supports only instance-axis slicing (a slice), got {key!r}"
            )
        n_inst = self.shape[0]
        if n_inst is None:
            raise ValueError("_Flat.__getitem__: leading axis is the ragged axis")
        start, stop, step = key.indices(n_inst)
        if step != 1:
            raise ValueError("_Flat slicing supports step=1 only")
        groups_per_inst = self.n_rows // n_inst if n_inst else 0
        g0, g1 = start * groups_per_inst, stop * groups_per_inst
        base = self.offsets[g0]
        new_offsets = np.ascontiguousarray(self.offsets[g0 : g1 + 1] - base)
        new_data = self.data[self.offsets[g0] : self.offsets[g1]]
        new_shape = (stop - start,) + self.shape[1:]
        return _Flat(new_data, new_offsets, new_shape)

    def reverse_masked(
        self, mask: NDArray[np.bool_], comp: NDArray | None = None
    ) -> "_Flat":
        """Reverse (or DNA reverse-complement) the `mask`-selected rows.

        ``comp`` is a **mode selector**, not a complement LUT that gets applied:

        * Pass ``comp=None`` (default) for a plain reversal of each selected row.
        * Pass ``comp=_COMP`` (gvl's standard ACGT→TGCA lookup) to enable DNA
          reverse-complement mode.  The LUT that is actually applied is always
          gvl's internal ``_COMP`` (via ``reverse_complement_masked``); passing a
          *different* array is not supported and will raise ``ValueError``.

        ``mask`` is one entry per outer query; it is replicated across any inner
        fixed axes in C order to produce one entry per flattened ragged row,
        matching seqpro's flat-kernel broadcast convention.
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


@dataclass(slots=True)
class _FlatAnnotatedHaps:
    """Composite flat-buffer analog of :class:`AnnotatedHaps` over three :class:`_Flat` s."""

    haps: _Flat
    var_idxs: _Flat
    ref_coords: _Flat

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.haps.shape

    def reverse_masked(
        self, mask: NDArray[np.bool_], comp: NDArray
    ) -> "_FlatAnnotatedHaps":
        self.haps = self.haps.reverse_masked(mask, comp=comp)
        self.var_idxs = self.var_idxs.reverse_masked(mask)
        self.ref_coords = self.ref_coords.reverse_masked(mask)
        return self

    def reshape(self, shape) -> "_FlatAnnotatedHaps":
        return _FlatAnnotatedHaps(
            self.haps.reshape(shape),
            self.var_idxs.reshape(shape),
            self.ref_coords.reshape(shape),
        )

    def squeeze(self, axis=None) -> "_FlatAnnotatedHaps":
        return _FlatAnnotatedHaps(
            self.haps.squeeze(axis),
            self.var_idxs.squeeze(axis),
            self.ref_coords.squeeze(axis),
        )

    def __getitem__(self, key) -> "_FlatAnnotatedHaps":
        return _FlatAnnotatedHaps(
            self.haps[key], self.var_idxs[key], self.ref_coords[key]
        )

    def to_ragged(self):
        from ._ragged import RaggedAnnotatedHaps  # boundary import

        return RaggedAnnotatedHaps(
            self.haps.view("S1").to_ragged(),
            self.var_idxs.to_ragged(),
            self.ref_coords.to_ragged(),
        )

    def to_fixed(self, length: int):
        from ._types import AnnotatedHaps

        return AnnotatedHaps(
            self.haps.view("S1").to_fixed(length),
            self.var_idxs.to_fixed(length),
            self.ref_coords.to_fixed(length),
        )

    def to_padded(self):
        from ._types import AnnotatedHaps

        return AnnotatedHaps(
            self.haps.view("S1").to_padded(b"N"),
            self.var_idxs.to_padded(-1),
            self.ref_coords.to_padded(np.iinfo(self.ref_coords.data.dtype).max),
        )
