from __future__ import annotations

from typing import Any, Literal
from collections.abc import Callable

import numpy as np
from genoray._types import POS_TYPE
from numpy.typing import NDArray
from seqpro.rag import Ragged
from seqpro.rag import concatenate as _rag_concatenate

from ._flat_variants import _rc_alleles_rust
from .._torch import TORCH_AVAILABLE, requires_torch

if TORCH_AVAILABLE:
    import torch
    from torch.nested._internal.nested_tensor import NestedTensor


_ALLELE_FIELDS = ("alt", "ref")


def _empty_group_pad(
    field_rag: Ragged,
    value: Any,
    empty_mask: NDArray[np.bool_],
    is_allele: bool = False,
) -> Ragged:
    """Return a Ragged with one sentinel element per empty group, zero for non-empty.

    Loop-free: offsets built from empty_mask.astype(int64) via cumsum; data buffer
    filled with `value` repeated empty_mask.sum() times.

    For allele fields (is_allele=True), `value` is bytes; produces an opaque-string
    Ragged with str_offsets matching the sentinel byte length.

    Parameters
    ----------
    field_rag
        The per-field Ragged to pad against.  Used only for shape/dtype.
    value
        Sentinel scalar.  For allele fields: bytes (e.g. b"N").
    empty_mask
        Flat bool array, length = number of groups (b*p).
    is_allele
        If True, treat value as bytes and produce an opaque-string Ragged.
    """
    n_empty = int(empty_mask.sum())
    # Variant-level offsets: group i gets 1 element if empty_mask[i] else 0.
    lengths = empty_mask.astype(np.int64)
    offsets = np.empty(len(empty_mask) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    if is_allele:
        # value is bytes, e.g. b"N" (len L).
        bval = value if isinstance(value, bytes) else value.encode()
        L = len(bval)
        # char data buffer: repeat the sentinel bytes for each empty group.
        char_data = np.frombuffer(bval * n_empty, dtype="S1").copy()
        # str_offsets: byte boundaries per variant — [0, L, 2L, ..., n_empty*L]
        str_offsets = np.arange(n_empty + 1, dtype=np.int64) * L
        # Shape: same as field_rag (opaque-string, has None at the variant ragged dim).
        shape = field_rag.shape
        return Ragged.from_offsets(char_data, shape, offsets, str_offsets=str_offsets)
    else:
        # Numeric sentinel.
        dtype = field_rag.data.dtype
        data = np.full(n_empty, value, dtype=dtype)
        # Shape: same as field_rag (has a None for the ragged dim).
        shape = field_rag.shape
        return Ragged.from_offsets(data, shape, offsets)


def _concat_string_ragged(base: Ragged, pad: Ragged) -> Ragged:
    """Concatenate two opaque-string Rageds at the variant axis (loop-free).

    For each group, appends pad variants after base variants. Merges variant-level
    offsets, reorders char data, and builds new str_offsets. Works with any
    opaque-string Ragged of shape (..., None) where the ragged axis is the variant axis.

    seqpro.rag.concatenate does not support opaque-string fields (the nested
    str_offsets structure requires special handling), so this helper fills that gap.
    """
    assert base.is_string and pad.is_string

    # Pack both to canonical (zero-based, contiguous) layout.
    base = base.to_packed()
    pad = pad.to_packed()

    base_var_off = np.asarray(base.offsets, dtype=np.int64)
    pad_var_off = np.asarray(pad.offsets, dtype=np.int64)
    # `_rl.str_offsets` is a private seqpro `_core.Ragged` attribute that holds the
    # inner (char-level) byte boundaries for an opaque-string Ragged.  No public
    # accessor for inner char offsets exists yet.  NOTE: for an opaque-string field
    # `_rl.str_offsets` is the correct handle — do NOT use `_layout.offsets[-1]`,
    # which on an opaque field is the variant-level offsets, not the char-level ones.
    base_str_off = np.asarray(base._rl.str_offsets, dtype=np.int64)
    pad_str_off = np.asarray(pad._rl.str_offsets, dtype=np.int64)

    n_groups = len(base_var_off) - 1
    n_base_vars = int(base_var_off[-1])
    n_pad_vars = int(pad_var_off[-1])

    # New variant-level offsets: sum base and pad lengths per group.
    base_var_lens = np.diff(base_var_off)
    pad_var_lens = np.diff(pad_var_off)
    new_var_lens = base_var_lens + pad_var_lens
    new_var_off = np.empty(n_groups + 1, dtype=np.int64)
    new_var_off[0] = 0
    np.cumsum(new_var_lens, out=new_var_off[1:])
    n_total_vars = int(new_var_off[-1])

    # Per-variant char lengths from base and pad.
    base_char_lens = np.diff(base_str_off)  # shape (n_base_vars,)
    pad_char_lens = np.diff(pad_str_off)  # shape (n_pad_vars,)

    # New per-variant char lengths: scatter base then pad into new positions.
    # For base variant k (global), it belongs to group g[k]; its new position is
    #   new_var_off[g[k]] + (k - base_var_off[g[k]])
    # = k + (new_var_off[g[k]] - base_var_off[g[k]])
    # Similarly for pad variants.
    new_char_lens = np.empty(n_total_vars, dtype=np.int64)
    base_dst_idx: NDArray[np.int64] | None = None
    pad_dst_idx: NDArray[np.int64] | None = None

    if n_base_vars > 0:
        group_of_base = np.repeat(np.arange(n_groups, dtype=np.int64), base_var_lens)
        shift = (new_var_off[:-1] - base_var_off[:-1])[group_of_base]
        base_dst_idx = np.arange(n_base_vars, dtype=np.int64) + shift
        new_char_lens[base_dst_idx] = base_char_lens

    if n_pad_vars > 0:
        group_of_pad = np.repeat(np.arange(n_groups, dtype=np.int64), pad_var_lens)
        shift = (new_var_off[:-1] + base_var_lens - pad_var_off[:-1])[group_of_pad]
        pad_dst_idx = np.arange(n_pad_vars, dtype=np.int64) + shift
        new_char_lens[pad_dst_idx] = pad_char_lens

    # Build new str_offsets (per-variant byte boundaries).
    new_str_off = np.empty(n_total_vars + 1, dtype=np.int64)
    new_str_off[0] = 0
    if n_total_vars > 0:
        np.cumsum(new_char_lens, out=new_str_off[1:])

    # Build new char data by scattering base then pad chars into their new positions.
    total_chars = int(new_str_off[-1]) if n_total_vars > 0 else 0
    new_data = np.empty(total_chars, dtype="S1")

    if total_chars > 0 and n_base_vars > 0 and int(base_str_off[-1]) > 0:
        assert base_dst_idx is not None
        # For each char in base: which base variant does it belong to?
        variant_of_char = np.repeat(
            np.arange(n_base_vars, dtype=np.int64), base_char_lens
        )
        # Offset within that variant.
        char_off_in_var = (
            np.arange(int(base_str_off[-1]), dtype=np.int64)
            - base_str_off[variant_of_char]
        )
        # Destination in new_data.
        dst = new_str_off[base_dst_idx[variant_of_char]] + char_off_in_var
        new_data[dst] = base.data[np.arange(int(base_str_off[-1]))]

    if total_chars > 0 and n_pad_vars > 0 and int(pad_str_off[-1]) > 0:
        assert pad_dst_idx is not None
        variant_of_char = np.repeat(
            np.arange(n_pad_vars, dtype=np.int64), pad_char_lens
        )
        char_off_in_var = (
            np.arange(int(pad_str_off[-1]), dtype=np.int64)
            - pad_str_off[variant_of_char]
        )
        dst = new_str_off[pad_dst_idx[variant_of_char]] + char_off_in_var
        new_data[dst] = pad.data[np.arange(int(pad_str_off[-1]))]

    return Ragged.from_offsets(
        new_data, base.shape, new_var_off, str_offsets=new_str_off
    )


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


class RaggedVariants(Ragged):
    """Variable-length variants as a single record Ragged with shape
    (batch, ploidy, ~variants). ``alt``/``ref`` are opaque-string fields; ``start``
    and optional ``ilen``/``dosage``/extra fields are numeric. Guaranteed: ``alt``,
    ``start``, and one of ``ref``/``ilen``.
    """

    __slots__ = ()

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
        super().__init__(Ragged.from_fields(rec))

    @classmethod
    def from_record(cls, rag: Ragged) -> "RaggedVariants":
        """Wrap an existing record Ragged directly (no copy), preserving subclass."""
        obj = object.__new__(cls)
        obj._layout = rag._layout
        return obj

    def _alt_chars(self, field: str = "alt") -> Ragged:
        """Return the S1 char view (b,p,~v,~l) of an allele field."""
        return self[field].to_chars()

    @property
    def alt(self) -> Ragged:
        """Alternative alleles (opaque-string Ragged, shape (b,p,~v))."""
        return self["alt"]

    @property
    def ref(self) -> Ragged:
        """Reference alleles (opaque-string Ragged, shape (b,p,~v))."""
        return self["ref"]

    @property
    def start(self) -> Ragged:
        """0-based start positions (numeric Ragged, shape (b,p,~v))."""
        return self["start"]

    @property
    def dosage(self) -> Ragged:
        """Dosages (numeric Ragged, shape (b,p,~v))."""
        return self["dosage"]

    @property
    def ilen(self) -> Ragged:
        """Indel lengths. Infallible — derived from alt/ref char lengths when absent."""
        if "ilen" in self.fields:
            return self["ilen"]
        # _rl.str_offsets gives per-variant byte boundaries for each opaque-string field.
        # np.diff produces a flat array of per-variant character counts.
        alt_field = self["alt"]
        alt_len = np.diff(alt_field._rl.str_offsets).astype(np.int32)
        if "ref" in self.fields:
            ref_field = self["ref"]
            ref_len = np.diff(ref_field._rl.str_offsets).astype(np.int32)
        else:
            ref_len = np.zeros_like(alt_len)
        start = self["start"]
        return Ragged.from_offsets(
            (alt_len - ref_len).astype(np.int32),
            start.shape,
            start.offsets,
        )

    @property
    def end(self) -> Ragged:
        """0-based exclusive end positions."""
        if "ref" in self.fields:
            ref_field = self["ref"]
            ref_len = np.diff(ref_field._rl.str_offsets).astype(POS_TYPE)
            reflen = Ragged.from_offsets(ref_len, self.start.shape, self.start.offsets)
            return self.start + reflen
        ilen = self.ilen
        return self.start - np.clip(ilen, None, 0) + 1

    def rc_(self, to_rc: NDArray[np.bool_] | None = None) -> "RaggedVariants":
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
            field = self[f]
            if f in _ALLELE_FIELDS:
                # field: opaque-string, shape (b, p, ~v)
                chars = field.to_chars().to_packed()  # (b, p, ~v, ~l) S1
                # _layout.offsets = [var_off (b*p+1,), char_off (n_alleles+1,)]
                var_off = chars._layout.offsets[0]  # variant-level: (b*p+1,)
                char_off = chars._layout.offsets[-1]  # char-level: (n_alleles+1,)
                n_alleles = len(char_off) - 1

                # Copy the data buffer; rc_alleles mutates it in place.
                data = chars.data.copy()

                # Expand to_rc (per-batch, size b) to per-allele (size n_alleles).
                # Batch element i_b owns alleles var_off[i_b*p] .. var_off[(i_b+1)*p]-1.
                batch_starts = np.arange(b, dtype=np.int64) * p
                alleles_per_batch = var_off[batch_starts + p] - var_off[batch_starts]
                allele_mask = np.repeat(to_rc, alleles_per_batch)

                _rc_alleles_rust(
                    data.view(np.uint8),
                    np.asarray(char_off, np.int64),
                    np.arange(n_alleles + 1, dtype=np.int64),
                    allele_mask,
                )

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
        # Non-allele fields from self already share the record's offsets. After
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
        if isinstance(allele, str):
            allele = allele.encode()
        all_pads: dict[str, Any] = {
            "alt": allele,
            "ref": allele,
            "ilen": ilen,
            "start": start,
            "dosage": dosage,
            **pad_values,
        }
        missing = set(self.fields) - set(all_pads)
        if missing:
            raise ValueError(f"Missing pad values for fields: {missing}")

        # Flat bool mask: True where a group has zero variants.
        empty = self["start"].lengths.reshape(-1) == 0

        out_fields: dict[str, Ragged] = {}
        shared_offsets: NDArray | None = None

        for f in self.fields:
            base = self[f]
            is_allele = f in _ALLELE_FIELDS
            pad_val = all_pads[f]
            pad_rag = _empty_group_pad(base, pad_val, empty, is_allele=is_allele)

            if is_allele:
                # Opaque-string: use _concat_string_ragged (seqpro.rag.concatenate
                # does not support the nested str_offsets structure of string Rageds).
                merged = _concat_string_ragged(base, pad_rag)
            else:
                var_axis = base.rag_dim
                merged = _rag_concatenate([base, pad_rag], axis=var_axis)

            # Collect shared offsets from first field processed.
            if shared_offsets is None:
                shared_offsets = merged.offsets
            out_fields[f] = merged

        # Re-share offsets across all fields so from_fields value-equality check passes.
        assert shared_offsets is not None
        out_fields = {
            k: _share_offsets(v, shared_offsets) for k, v in out_fields.items()
        }
        return RaggedVariants.from_record(Ragged.from_fields(out_fields))

    @requires_torch
    def to_nested_tensor_batch(
        self,
        device: "str | torch.device" = "cpu",
        tokenizer: "Literal['seqpro'] | Callable[[NDArray[np.bytes_]], NDArray[np.integer]] | None" = None,
    ) -> "dict[str, NestedTensor | int]":
        """Convert a RaggedVariants object to a dictionary of nested tensors.

        Numeric fields (``start``, ``ilen``, ``dosage``, any extra) are flattened
        across the ploidy dimension so their shape is ``(batch * ploidy, ~variants)``.
        Allele fields (``alt``, ``ref``) are flattened across both the ploidy and
        variant dimensions so their shape is
        ``(batch * ploidy * ~variants, ~alt_len)``.

        Parameters
        ----------
        device
            Device to move tensors to.
        tokenizer
            How to encode allele characters.

            - ``"seqpro"`` — use ``seqpro.tokenize`` (ACGTN → 0 1 2 3 4).
            - ``None`` — uint8 ASCII values (ACGTN → 65 67 71 84 78).
            - Callable — called with the flat ``NDArray[np.bytes_]`` data,
              returns an integer array of the same length.

        Returns
        -------
        dict
            - ``"alt"`` — nested tensor ``(batch*ploidy*~vars, ~alt_len)``
            - ``"ref"`` — nested tensor ``(batch*ploidy*~vars, ~ref_len)`` (if present)
            - numeric field keys — nested tensor ``(batch*ploidy, ~vars)``
            - ``"max_n_vars"`` — int
            - ``"max_alt_len"`` — int
            - ``"max_ref_len"`` — int (if ``ref`` present)
        """
        import seqpro as sp
        from torch.nested import nested_tensor_from_jagged as nt_jag

        batch: "dict[str, NestedTensor | int]" = {}
        batch["max_n_vars"] = int(self["start"].lengths.max())

        # Shared variant-level offsets (int32 for torch) — computed once from the
        # first numeric field; all numeric fields share the same offsets object.
        var_offsets_t: "torch.Tensor | None" = None

        for f in self.fields:
            field = self[f]
            if f in _ALLELE_FIELDS:
                # Allele field: opaque-string (b, p, ~v) → char view (b, p, ~v, ~l).
                # After to_chars().to_packed():
                #   _layout.offsets = [var_off, char_off]
                #   _layout.offsets[-1] = char_off: per-allele byte boundaries.
                # NOTE: .offsets returns _layout.offsets[0] (variant-level), so we
                # must use ._layout.offsets[-1] for the inner (char-level) boundaries.
                chars = field.to_chars().to_packed()
                char_off = np.asarray(chars._layout.offsets[-1], dtype=np.int64)
                char_lens = np.diff(char_off)
                max_len = int(char_lens.max()) if char_lens.size > 0 else 0
                batch[f"max_{f}_len"] = max_len

                if tokenizer is None:
                    raw: "NDArray" = chars.data.view(np.uint8)
                elif tokenizer == "seqpro":
                    # ACGTN → 0 1 2 3 4 (unknown token = 4)
                    raw = sp.tokenize(
                        chars.data,
                        dict(zip(sp.DNA.alphabet, range(4))),
                        4,
                    )
                else:
                    raw = tokenizer(chars.data)

                data_t = torch.from_numpy(np.ascontiguousarray(raw)).to(device)
                off_t = torch.from_numpy(char_off.astype(np.int32)).to(device)
                batch[f] = nt_jag(data_t, off_t, max_seqlen=max_len)
            else:
                # Numeric field: shape (b, p, ~v), flattened to (b*p, ~v).
                packed = field.to_packed()
                if var_offsets_t is None:
                    var_offsets_t = torch.from_numpy(
                        np.asarray(packed.offsets, dtype=np.int32)
                    ).to(device)
                data_t = torch.from_numpy(np.ascontiguousarray(packed.data)).to(device)
                batch[f] = nt_jag(data_t, var_offsets_t)

        return batch
