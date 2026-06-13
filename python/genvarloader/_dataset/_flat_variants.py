"""Flat-buffer analog of RaggedVariants: pure-numpy (data, offsets) per field,
no awkward on the hot path. Converts to RaggedVariants only via to_ragged()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray



@dataclass(slots=True)
class _FlatAlleles:
    """Two-level flat bytestring for an alt/ref allele field, shape (b, p, ~v, ~l).

    Layout matches _build_allele_layout (inner-before-outer):
    - byte_data:   uint8 contiguous allele bytes
    - seq_offsets: per-variant byte boundaries (allele_offsets), len n_variants + 1
    - var_offsets: per-(b*p)-row variant boundaries (group_offsets), len b*p + 1
    - shape:       outer fixed dims with exactly one None (the ragged variant axis)
    """

    byte_data: NDArray[np.uint8]
    seq_offsets: NDArray[np.int64]
    var_offsets: NDArray[np.int64]
    shape: tuple[int | None, ...]

    @property
    def ploidy(self) -> int:
        # shape is (b, p, None) for variants; ploidy is the last fixed dim.
        # For a flat (2, None) shape (b*p flattened), ploidy defaults to 1.
        fixed = [d for d in self.shape if d is not None]
        return fixed[-1] if len(fixed) >= 2 else 1

    def to_ragged(self):
        from ._haps import _build_allele_layout

        return _build_allele_layout(
            np.ascontiguousarray(self.byte_data, np.uint8),
            np.asarray(self.seq_offsets, np.int64),
            np.asarray(self.var_offsets, np.int64),
            self.ploidy,
        )

    def reverse_masked(self, mask: NDArray[np.bool_]) -> "_FlatAlleles":
        """DNA reverse-complement the mask-selected (b*p) rows' alleles, in place.
        ``mask`` is one entry per (b*p) row; broadcast across that row's variants."""
        from seqpro.rag import Ragged

        from .._ragged import reverse_complement_masked

        m = np.ascontiguousarray(mask, np.bool_).reshape(-1)
        # per-allele mask: repeat each row's flag across its variant count
        per_allele = np.repeat(m, np.diff(self.var_offsets))
        view = Ragged.from_offsets(
            self.byte_data.view("S1"),
            (per_allele.size, None),
            np.asarray(self.seq_offsets, np.int64),
        )
        reverse_complement_masked(view, per_allele)  # mutates byte_data in place
        return self

    def reshape(self, shape: tuple[int | None, ...]) -> "_FlatAlleles":
        return _FlatAlleles(self.byte_data, self.seq_offsets, self.var_offsets, shape)

    def squeeze(self, axis: int | None = None) -> "_FlatAlleles":
        fixed = [d for d in self.shape if d is not None]
        if axis is None:
            fixed = [d for d in fixed if d != 1]
        else:
            del fixed[axis]
        return _FlatAlleles(self.byte_data, self.seq_offsets, self.var_offsets,
                            (*fixed, None))


@dataclass(slots=True)
class _FlatVariants:
    """Flat analog of RaggedVariants. `fields` maps field name -> _Flat (scalar
    fields: start/ilen/dosage/info) or _FlatAlleles (alt/ref)."""

    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.fields["start"].shape

    def to_ragged(self):
        from ._rag_variants import RaggedVariants

        kw = {}
        for name, f in self.fields.items():
            kw[name] = f.to_ragged()
        return RaggedVariants(**kw)

    def reshape(self, shape) -> "_FlatVariants":
        return _FlatVariants({k: v.reshape(shape) for k, v in self.fields.items()})

    def squeeze(self, axis: int | None = None) -> "_FlatVariants":
        return _FlatVariants(
            {k: v.squeeze(axis) for k, v in self.fields.items()}
        )

    def reverse_masked(self, mask: NDArray[np.bool_]) -> "_FlatVariants":
        # Only alt/ref alleles are reverse-complemented; scalar fields unchanged
        # (matches RaggedVariants.rc_ which only touches alt/ref).
        for name in ("alt", "ref"):
            if name in self.fields:
                self.fields[name] = self.fields[name].reverse_masked(mask)
        return self
