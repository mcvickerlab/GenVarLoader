from __future__ import annotations

from typing import Any, Literal
from collections.abc import Callable

import numpy as np
from genoray._types import POS_TYPE
from numpy.typing import NDArray
from seqpro.rag import Ragged

from .._torch import TORCH_AVAILABLE, requires_torch

if TORCH_AVAILABLE:
    import torch
    from torch.nested._internal.nested_tensor import NestedTensor


_ALLELE_FIELDS = ("alt", "ref")


def _as_opaque(rag: Ragged) -> Ragged:
    """Normalize an allele field to opaque-string (b,p,~v). Accepts an S1 char
    (b,p,~v,~l) Ragged (collapse via to_strings) or an already-opaque Ragged."""
    return rag.to_strings() if not getattr(rag, "is_string", False) else rag


def _share_offsets(rag: Ragged, offsets: NDArray) -> Ragged:
    """Rebuild `rag` onto the given (identical) variant-level offsets object so all
    record fields share it (Ragged.from_fields requires value equality; sharing the
    same object guarantees that and avoids redundant equality checks)."""
    if rag.offsets is offsets:
        return rag
    if getattr(rag, "is_string", False):
        chars = rag.to_chars()
        return Ragged.from_offsets(
            chars.data, rag.shape, offsets, str_offsets=chars._layout.offsets[-1]
        ).to_strings()
    return Ragged.from_offsets(rag.data, rag.shape, offsets)


class RaggedVariants:
    """Variable-length variants as a single record Ragged with shape
    (batch, ploidy, ~variants). ``alt``/``ref`` are opaque-string fields; ``start``
    and optional ``ilen``/``dosage``/extra fields are numeric. Guaranteed: ``alt``,
    ``start``, and one of ``ref``/``ilen``.
    """

    __slots__ = ("_rag",)

    def __init__(
        self,
        alt: Ragged,
        start: Ragged,
        ref: Ragged | None = None,
        ilen: Ragged | None = None,
        dosage: Ragged | None = None,
        **fields: Ragged,
    ):
        if ref is None and ilen is None:
            raise ValueError("Must provide one of ref or ilen.")
        alt = _as_opaque(alt)
        off = alt.offsets
        rec: dict[str, Ragged] = {"alt": alt, "start": _share_offsets(start, off)}
        if ref is not None:
            rec["ref"] = _share_offsets(_as_opaque(ref), off)
        if ilen is not None:
            rec["ilen"] = _share_offsets(ilen, off)
        if dosage is not None:
            rec["dosage"] = _share_offsets(dosage, off)
        for k, v in fields.items():
            rec[k] = _share_offsets(v, off)
        self._rag = Ragged.from_fields(rec)

    @classmethod
    def from_record(cls, rag: Ragged) -> "RaggedVariants":
        """Wrap an existing record Ragged directly (no copy)."""
        obj = cls.__new__(cls)
        obj._rag = rag
        return obj

    @property
    def fields(self) -> list[str]:
        return self._rag.fields

    def _alt_chars(self, field: str = "alt") -> Ragged:
        """Return the S1 char view (b,p,~v,~l) of an allele field."""
        return self._rag[field].to_chars()

    @property
    def alt(self) -> Ragged:
        """Alternative alleles (opaque-string Ragged, shape (b,p,~v))."""
        return self._rag["alt"]

    @property
    def ref(self) -> Ragged:
        """Reference alleles (opaque-string Ragged, shape (b,p,~v))."""
        return self._rag["ref"]

    @property
    def start(self) -> Ragged:
        """0-based start positions (numeric Ragged, shape (b,p,~v))."""
        return self._rag["start"]

    @property
    def dosage(self) -> Ragged:
        """Dosages (numeric Ragged, shape (b,p,~v))."""
        return self._rag["dosage"]

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self._rag.shape

    @property
    def ilen(self) -> Ragged:
        """Indel lengths. Infallible — derived from alt/ref char lengths when absent."""
        if "ilen" in self.fields:
            return self._rag["ilen"]
        # _rl.str_offsets gives per-variant byte boundaries for each opaque-string field.
        # np.diff produces a flat array of per-variant character counts.
        alt_field = self._rag["alt"]
        alt_len = np.diff(alt_field._rl.str_offsets).astype(np.int32)
        if "ref" in self.fields:
            ref_field = self._rag["ref"]
            ref_len = np.diff(ref_field._rl.str_offsets).astype(np.int32)
        else:
            ref_len = np.zeros_like(alt_len)
        start = self._rag["start"]
        return Ragged.from_offsets(
            (alt_len - ref_len).astype(np.int32),
            start.shape,
            start.offsets,
        )

    @property
    def end(self) -> Ragged:
        """0-based exclusive end positions."""
        if "ref" in self.fields:
            ref_field = self._rag["ref"]
            ref_len = np.diff(ref_field._rl.str_offsets).astype(POS_TYPE)
            reflen = Ragged.from_offsets(ref_len, self.start.shape, self.start.offsets)
            return self.start + reflen
        ilen = self.ilen
        return self.start - np.clip(ilen, None, 0) + 1

    def __len__(self) -> int:
        return len(self._rag)

    def __getitem__(self, idx: Any) -> "RaggedVariants":
        rag = self._rag
        # For multi-leading-dim records, an integer idx would hit _getitem_record_rows
        # which returns a dict of raw arrays rather than a Ragged. Convert to a tuple
        # so _getitem_tuple_multidim is used instead, which preserves the RecordLayout.
        if rag._is_record and rag.rag_dim > 1 and isinstance(idx, (int, np.integer)):
            result = rag[(idx,)]
        else:
            result = rag[idx]
        return RaggedVariants.from_record(result)

    def reshape(self, shape: tuple[int | None, ...]) -> "RaggedVariants":
        if isinstance(shape, tuple):
            return RaggedVariants.from_record(self._rag.reshape(*shape))
        return RaggedVariants.from_record(self._rag.reshape(shape))

    def squeeze(self, axis: int | None = None, **kw: Any) -> "RaggedVariants":
        """Squeeze first axis."""
        return self[0]

    def to_packed(self) -> "RaggedVariants":
        return RaggedVariants.from_record(self._rag.to_packed())

    def rc_(self, to_rc: NDArray[np.bool_] | None = None) -> "RaggedVariants":
        from .._ragged import _COMP

        from seqpro.rag import reverse_complement as _sp_reverse_complement

        b = self.shape[0]
        if to_rc is None:
            to_rc = np.ones(b, np.bool_)
        elif not np.asarray(to_rc).any():
            return self

        to_rc = np.asarray(to_rc, dtype=np.bool_)
        p = self.shape[1]

        rec: dict[str, Ragged] = {}
        shared_var_off: NDArray | None = None

        for f in self.fields:
            field = self._rag[f]
            if f in _ALLELE_FIELDS:
                # field: opaque-string, shape (b, p, ~v)
                chars = field.to_chars().to_packed()  # (b, p, ~v, ~l) S1
                # _layout.offsets = [var_off (b*p+1,), char_off (n_alleles+1,)]
                var_off = chars._layout.offsets[0]  # variant-level: (b*p+1,)
                char_off = chars._layout.offsets[-1]  # char-level: (n_alleles+1,)
                n_alleles = len(char_off) - 1

                # Build a flat allele-level R=1 view on a copy of the data buffer.
                data = chars.data.copy()
                view = Ragged.from_offsets(data, (n_alleles, None), char_off)

                # Expand to_rc (per-batch, size b) to per-allele (size n_alleles).
                # Batch element i_b owns alleles var_off[i_b*p] .. var_off[(i_b+1)*p]-1.
                batch_starts = np.arange(b, dtype=np.int64) * p
                alleles_per_batch = var_off[batch_starts + p] - var_off[batch_starts]
                allele_mask = np.repeat(to_rc, alleles_per_batch)

                _sp_reverse_complement(view, _COMP, mask=allele_mask, copy=False)

                # Rebuild as opaque-string field with the same shape and offsets.
                rebuilt = Ragged.from_offsets(
                    data, field.shape, var_off, str_offsets=char_off
                )
                if shared_var_off is None:
                    shared_var_off = var_off
                rec[f] = rebuilt
            else:
                rec[f] = field

        # All fields must share the same outer (variant-level) offsets for from_fields.
        # Non-allele fields from self._rag already share the record's offsets. After
        # to_packed() the packed var_off may be a new object; re-share via _share_offsets.
        if shared_var_off is not None:
            rec = {k: _share_offsets(v, shared_var_off) for k, v in rec.items()}

        return RaggedVariants.from_record(Ragged.from_fields(rec))

    def pad(
        self,
        allele: str | bytes = b"N",
        ilen: int = 0,
        start: int = -1,
        dosage: float = 0.0,
        **pad_values: Any,
    ) -> "RaggedVariants":
        raise NotImplementedError("ported in Task G4")

    @requires_torch
    def to_nested_tensor_batch(
        self,
        device: "str | torch.device" = "cpu",
        tokenizer: "Literal['seqpro'] | Callable[[NDArray[np.bytes_]], NDArray[np.integer]] | None" = None,
    ) -> "dict[str, NestedTensor | int]":
        raise NotImplementedError("ported in Task G5")
