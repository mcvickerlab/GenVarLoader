"""Flat-buffer analog of RaggedVariants: pure-numpy (data, offsets) per field,
all-numpy hot path. Converts to RaggedVariants only via to_ragged()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numba as nb
import numpy as np
from numpy.typing import NDArray

from .._dispatch import get, register
from ..genvarloader import compact_keep_f32 as _compact_keep_f32_rust
from ..genvarloader import compact_keep_i32 as _compact_keep_i32_rust
from ..genvarloader import fill_empty_fixed_f32 as _fill_empty_fixed_f32_rust
from ..genvarloader import fill_empty_fixed_i32 as _fill_empty_fixed_i32_rust
from ..genvarloader import fill_empty_scalar_f32 as _fill_empty_scalar_f32_rust
from ..genvarloader import fill_empty_scalar_i32 as _fill_empty_scalar_i32_rust
from ..genvarloader import assemble_variant_buffers_i32 as _assemble_variant_buffers_i32_rust
from ..genvarloader import assemble_variant_buffers_u8 as _assemble_variant_buffers_u8_rust
from ..genvarloader import fill_empty_seq_i32 as _fill_empty_seq_i32_rust
from ..genvarloader import fill_empty_seq_u8 as _fill_empty_seq_u8_rust
from ..genvarloader import gather_alleles as _gather_alleles_rust
from ..genvarloader import gather_rows_f32 as _gather_rows_f32_rust
from ..genvarloader import gather_rows_i32 as _gather_rows_i32_rust
from ._genotypes import _as_starts_stops

if TYPE_CHECKING:
    from ._haps import Haps


@dataclass(frozen=True)
class DummyVariant:
    """Per-field values for the dummy variant inserted into empty
    (region, sample, ploid) groups. Unspecified info fields default to ``0``
    for integer columns and ``NaN`` for float columns."""

    start: int = -1
    ilen: int = 0
    dosage: float = 0.0
    ref: bytes = b"N"
    alt: bytes = b"N"
    info: dict[str, Any] = field(default_factory=dict)

    def scalar_for(self, name: str, dtype: np.dtype):
        """Return the dummy fill value for a scalar field, as a numpy scalar of ``dtype``."""
        dt = np.dtype(dtype)
        if name == "start":
            return dt.type(self.start)
        if name == "ilen":
            return dt.type(self.ilen)
        if name == "dosage":
            return dt.type(self.dosage)
        if name in self.info:
            return dt.type(self.info[name])
        if np.issubdtype(dt, np.floating):
            return dt.type(np.nan)
        return dt.type(0)


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
        from seqpro.rag import Ragged

        # Build an opaque-string Ragged from flat buffers:
        #   byte_data  — S1 char bytes
        #   seq_offsets  — per-variant char boundaries (inner, len=n_variants+1)
        #   var_offsets  — per-(b*p)-group variant boundaries (outer, len=b*p+1)
        # Two ragged axes: (b*p, ~variants, ~chars) → collapse chars via
        # to_strings() → (b*p, ~variants) opaque string → reshape to (b, p, ~v).
        char_data = np.ascontiguousarray(self.byte_data).view(dtype="S1")
        var_off = np.asarray(self.var_offsets, dtype=np.int64)
        seq_off = np.asarray(self.seq_offsets, dtype=np.int64)
        b_times_p = len(var_off) - 1

        # Extract fixed dims: shape is (b, p, None) or (b*p, None).
        fixed = [d for d in self.shape if d is not None]
        if len(fixed) >= 2:
            # Re-derive b from b*p and ploidy (last fixed dim).
            p = fixed[-1]
            b = b_times_p // p
        else:
            b = b_times_p
            p = 1

        return (
            Ragged.from_offsets(char_data, (b_times_p, None, None), [var_off, seq_off])
            .to_strings()
            .reshape(b, p, None)
        )

    def reverse_masked(self, mask: NDArray[np.bool_]) -> "_FlatAlleles":
        """DNA reverse-complement the mask-selected rows' alleles, in place.

        ``mask`` is one entry per region (length ``b``); it is broadcast across
        ploidy then across each (b*p) row's variant count, exactly matching
        ``RaggedVariants.rc_`` (``np.repeat(to_rc, ploidy)`` then
        ``np.repeat(per_bp, np.diff(group_off))``).
        """
        from seqpro.rag import Ragged

        from .._ragged import reverse_complement_masked

        m = np.ascontiguousarray(mask, np.bool_).reshape(-1)
        # per-(b*p) mask: broadcast each region's flag across ploidy
        per_bp = np.repeat(m, self.ploidy)
        # per-allele mask: repeat each row's flag across its variant count
        per_allele = np.repeat(per_bp, np.diff(self.var_offsets))
        view = Ragged.from_offsets(
            self.byte_data.view("S1"),
            (per_allele.size, None),
            np.asarray(self.seq_offsets, np.int64),
        )
        reverse_complement_masked(view, per_allele)  # mutates byte_data in place
        return self

    def reshape(self, shape: int | tuple[int, ...]) -> "_FlatAlleles":
        # Mirror _Flat.reshape: accept outer dims and APPEND our own ragged None.
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)
        if shape and shape[-1] is None:  # be defensive: strip a trailing None
            shape = shape[:-1]
        new = shape + (None,)
        return _FlatAlleles(self.byte_data, self.seq_offsets, self.var_offsets, new)

    def squeeze(self, axis: int | None = None) -> "_FlatAlleles":
        fixed = [d for d in self.shape if d is not None]
        if axis is None:
            fixed = [d for d in fixed if d != 1]
        else:
            del fixed[axis]
        return _FlatAlleles(
            self.byte_data, self.seq_offsets, self.var_offsets, (*fixed, None)
        )

    def __getitem__(self, key) -> "_FlatAlleles":
        """Slice the leading (instance) axis, rebasing both offset levels."""
        if not isinstance(key, slice):
            raise TypeError(
                f"_FlatAlleles supports only instance-axis slicing, got {key!r}"
            )
        n_inst = self.shape[0]
        if n_inst is None:
            raise ValueError(
                "_FlatAlleles.__getitem__: leading axis is the ragged axis"
            )
        start, stop, step = key.indices(n_inst)
        if step != 1:
            raise ValueError("_FlatAlleles slicing supports step=1 only")
        rows_per_inst = (len(self.var_offsets) - 1) // n_inst if n_inst else 0
        r0, r1 = start * rows_per_inst, stop * rows_per_inst
        v0, v1 = int(self.var_offsets[r0]), int(self.var_offsets[r1])
        new_var = np.ascontiguousarray(
            self.var_offsets[r0 : r1 + 1] - self.var_offsets[r0]
        )
        new_seq = np.ascontiguousarray(
            self.seq_offsets[v0 : v1 + 1] - self.seq_offsets[v0]
        )
        new_bytes = self.byte_data[
            int(self.seq_offsets[v0]) : int(self.seq_offsets[v1])
        ]
        new_shape = (stop - start,) + self.shape[1:]
        return _FlatAlleles(new_bytes, new_seq, new_var, new_shape)


@dataclass(slots=True)
class _FlatWindow:
    """Two-level flat token buffer for ref/alt windows, shape (b, p, ~v, ~win).

    Mirrors _FlatAlleles but `data` holds tokens (configured int dtype), not bytes,
    so to_ragged() drops the byte/bytestring string parameters. Both inner axes
    (variant count and window length) are ragged, so to_ragged() returns a numeric
    two-ragged-axis _core.Ragged with shape (b, p, ~v, ~w).
    """

    data: NDArray  # tokens (uint8 or int32), flat
    seq_offsets: NDArray[np.int64]  # per-variant window offsets, n_variants + 1
    var_offsets: NDArray[np.int64]  # per (instance, ploid) offsets, b*p + 1
    shape: tuple[int | None, ...]

    def to_ragged(self):
        from seqpro.rag import Ragged

        # Build a numeric Ragged with shape (b, p, ~v, ~w): two ragged axes.
        # var_offsets: per-(b*p)-group variant boundaries (len b*p + 1)
        # seq_offsets: per-variant window token boundaries (len n_variants + 1)
        fixed = [d for d in self.shape if d is not None]
        if len(fixed) >= 2:
            p = fixed[-1]
            b = (len(self.var_offsets) - 1) // p
        else:
            b = len(self.var_offsets) - 1
            p = 1
        data = np.ascontiguousarray(self.data)
        return Ragged.from_offsets(
            data,
            (b, p, None, None),
            [
                np.asarray(self.var_offsets, np.int64),
                np.asarray(self.seq_offsets, np.int64),
            ],
        )

    def reshape(self, shape) -> "_FlatWindow":
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)
        # strip any trailing None defensively, then append our two ragged axes
        while shape and shape[-1] is None:
            shape = shape[:-1]
        return _FlatWindow(
            self.data, self.seq_offsets, self.var_offsets, (*shape, None, None)
        )

    def squeeze(self, axis: int | None = None) -> "_FlatWindow":
        fixed = [d for d in self.shape if d is not None]
        if axis is None:
            fixed = [d for d in fixed if d != 1]
        else:
            del fixed[axis]
        return _FlatWindow(
            self.data, self.seq_offsets, self.var_offsets, (*fixed, None, None)
        )


@dataclass(frozen=True)
class VarWindowOpt:
    """Options for ``with_seqs('variant-windows')``.

    Bundles every variant-window setting in one place so they are explicit
    rather than inherited from ``with_settings``. ``ref`` and ``alt`` are chosen
    independently: ``"window"`` emits the flanked, tokenized window (ref =
    ``[start-L, end+L)`` reference read; alt = ``flank5 . alt . flank3``), while
    ``"allele"`` emits the bare tokenized allele with no flanks.
    """

    flank_length: int
    token_alphabet: bytes
    unknown_token: int
    ref: Literal["window", "allele"] = "window"
    alt: Literal["window", "allele"] = "window"


_WINDOW_FIELD_NAMES = ("ref_window", "alt_window", "ref", "alt")


@dataclass(slots=True)
class _FlatVariantWindows:
    """Window-mode variants output: scalar fields + per-allele token buffers.

    Each allele is emitted either as a flanked window (``ref_window`` /
    ``alt_window``) or a bare tokenized allele (``ref`` / ``alt``); the unused
    slot of each pair is ``None``. Raw (byte) alleles are intentionally absent.
    Returned directly in flat output mode (the query boundary never converts it).
    Reverse-complement is intentionally NOT supported (reference-oriented).
    """

    fields: dict[str, Any]  # start / ilen / dosage / info -> _Flat
    ref_window: _FlatWindow | None = None
    alt_window: _FlatWindow | None = None
    ref: _FlatWindow | None = None  # bare tokenized ref allele (no flanks)
    alt: _FlatWindow | None = None  # bare tokenized alt allele (no flanks)

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.fields["start"].shape

    def _present(self) -> dict[str, "_FlatWindow"]:
        return {
            n: getattr(self, n)
            for n in _WINDOW_FIELD_NAMES
            if getattr(self, n) is not None
        }

    def to_ragged(self):
        out = {k: v.to_ragged() for k, v in self.fields.items()}
        for n, w in self._present().items():
            out[n] = w.to_ragged()
        return out

    def reshape(self, shape) -> "_FlatVariantWindows":
        present = {n: w.reshape(shape) for n, w in self._present().items()}
        return _FlatVariantWindows(
            {k: v.reshape(shape) for k, v in self.fields.items()}, **present
        )

    def squeeze(self, axis: int | None = None) -> "_FlatVariantWindows":
        present = {n: w.squeeze(axis) for n, w in self._present().items()}
        return _FlatVariantWindows(
            {k: v.squeeze(axis) for k, v in self.fields.items()}, **present
        )

    def fill_empty_groups(
        self, dummy: "DummyVariant", unk: int, flank_length: int
    ) -> "_FlatVariantWindows":
        """Insert one all-``unk`` dummy entry into each empty (b*p) group.

        Scalar fields take ``DummyVariant`` values; window fields take ``unk``.
        Window length: ``2*flank_length + len(dummy allele)`` for ref/alt
        windows, ``len(dummy allele)`` for bare ref/alt alleles."""
        from .._flat import _Flat

        new_fields: dict[str, Any] = {}
        for name, f in self.fields.items():
            fill = dummy.scalar_for(name, f.data.dtype)
            nd, noff = _fill_empty_scalar(f.data, f.offsets, fill)
            new_fields[name] = _Flat.from_offsets(nd, f.shape, noff)

        present: dict[str, _FlatWindow] = {}
        for name, w in self._present().items():
            allele = dummy.alt if name in ("alt", "alt_window") else dummy.ref
            base = len(allele)
            win_len = (2 * flank_length + base) if name.endswith("_window") else base
            dwin = np.full(win_len, unk, dtype=w.data.dtype)
            nd, nvar, nseq = _fill_empty_seq(w.data, w.var_offsets, w.seq_offsets, dwin)
            present[name] = _FlatWindow(nd, nseq, nvar, w.shape)

        return _FlatVariantWindows(new_fields, **present)


@dataclass(slots=True)
class _FlatVariants:
    """Flat analog of RaggedVariants. `fields` maps field name -> _Flat (scalar
    fields: start/ilen/dosage/info) or _FlatAlleles (alt/ref)."""

    fields: dict[str, Any] = field(default_factory=dict)
    flank_tokens: Any = (
        None  # _Flat | None — ride-along, shape (b, p, ~v, 2L); flat-mode only
    )

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
        new = _FlatVariants({k: v.reshape(shape) for k, v in self.fields.items()})
        if self.flank_tokens is not None:
            from .._flat import _Flat

            ft = self.flank_tokens
            inner = ft.shape[-1]  # 2L, fixed
            # Normalize like _Flat.reshape (accept int or any sequence of dims).
            outer = (shape,) if isinstance(shape, int) else tuple(shape)
            new.flank_tokens = _Flat(ft.data, ft.offsets, (*outer, None, inner))
        return new

    def squeeze(self, axis: int | None = None) -> "_FlatVariants":
        new = _FlatVariants({k: v.squeeze(axis) for k, v in self.fields.items()})
        if self.flank_tokens is not None:
            from .._flat import _Flat

            ft = self.flank_tokens
            inner = ft.shape[-1]
            outer = [d for d in ft.shape[:-1] if d is not None]
            if axis is None:
                outer = [d for d in outer if d != 1]
            else:
                del outer[axis]
            new.flank_tokens = _Flat(ft.data, ft.offsets, (*outer, None, inner))
        return new

    def __getitem__(self, key) -> "_FlatVariants":
        # flank_tokens (shape (b, ploidy, None, 2L), ragged axis in the middle)
        # cannot be sliced by the instance-axis _Flat.__getitem__, and the
        # buffered transport path does not carry it. Slicing a _FlatVariants that
        # has flank_tokens is unsupported rather than silently lossy.
        if self.flank_tokens is not None:
            raise NotImplementedError(
                "Instance-axis slicing of _FlatVariants with flank_tokens is not "
                "supported; flank tokens are not carried on the buffered transport path."
            )
        return _FlatVariants({k: v[key] for k, v in self.fields.items()})

    def reverse_masked(self, mask: NDArray[np.bool_]) -> "_FlatVariants":
        # Only alt/ref alleles are reverse-complemented; scalar fields unchanged
        # (matches RaggedVariants.rc_ which only touches alt/ref).
        for name in ("alt", "ref"):
            if name in self.fields:
                self.fields[name] = self.fields[name].reverse_masked(mask)
        return self

    def fill_empty_groups(
        self, dummy: "DummyVariant", unk: int | None = None
    ) -> "_FlatVariants":
        """Insert one dummy variant into each empty (b*p) group; non-empty
        groups are unchanged. Every field shares the same empty-row pattern, so
        the rebuilt offsets stay consistent across fields. When ``flank_tokens``
        is present, its empty rows are filled with ``2L`` ``unk`` tokens."""
        from .._flat import _Flat

        new_fields: dict[str, Any] = {}
        for name, f in self.fields.items():
            if isinstance(f, _FlatAlleles):
                db = np.frombuffer(
                    dummy.alt if name == "alt" else dummy.ref, np.uint8
                ).copy()
                nd, nvar, nseq = _fill_empty_seq(
                    f.byte_data, f.var_offsets, f.seq_offsets, db
                )
                new_fields[name] = _FlatAlleles(nd, nseq, nvar, f.shape)
            else:
                fill = dummy.scalar_for(name, f.data.dtype)
                nd, noff = _fill_empty_scalar(f.data, f.offsets, fill)
                new_fields[name] = _Flat.from_offsets(nd, f.shape, noff)
        out = _FlatVariants(new_fields)
        if self.flank_tokens is not None:
            # flank_tokens is only set on the token-enabled ride-along path, where
            # unknown_token (-> unk) is always provided; so unk is non-None here.
            ft = self.flank_tokens
            inner = ft.shape[-1]  # 2L, fixed
            nd, noff = _fill_empty_fixed(ft.data, ft.offsets, inner, unk)
            out.flank_tokens = _Flat(nd, noff, ft.shape)
        return out


@nb.njit(nogil=True, cache=True)
def _gather_v_idxs_numba(
    geno_offset_idx, geno_offsets, geno_v_idxs
):  # pragma: no cover - njit
    """Gather per-row variant indices: for each row's offset slice into the
    sparse arrays, copy its values out into flat ``(data, offsets)``.

    ``geno_offsets`` must be 1-D contiguous (length n_rows + 1).  For the
    non-contiguous (2, n_rows) starts/stops form use
    :func:`_gather_v_idxs_ss_numba`.
    """
    n_rows = geno_offset_idx.shape[0]
    out_offsets = np.empty(n_rows + 1, np.int64)
    out_offsets[0] = 0
    for i in range(n_rows):
        goi = geno_offset_idx[i]
        out_offsets[i + 1] = out_offsets[i] + (
            geno_offsets[goi + 1] - geno_offsets[goi]
        )
    total = out_offsets[n_rows]
    v_idxs = np.empty(total, geno_v_idxs.dtype)
    dst = 0
    for i in range(n_rows):
        goi = geno_offset_idx[i]
        s = geno_offsets[goi]
        e = geno_offsets[goi + 1]
        for k in range(s, e):
            v_idxs[dst] = geno_v_idxs[k]
            dst += 1
    return v_idxs, out_offsets


@nb.njit(nogil=True, cache=True)
def _gather_v_idxs_ss_numba(
    geno_offset_idx, geno_starts, geno_stops, geno_v_idxs
):  # pragma: no cover - njit
    """Like :func:`_gather_v_idxs_numba` but for non-contiguous (starts, stops) offsets.

    ``geno_starts`` and ``geno_stops`` are the two rows of a ``(2, n)`` offset
    array (``geno_starts = geno_offsets[0]``, ``geno_stops = geno_offsets[1]``).
    """
    n_rows = geno_offset_idx.shape[0]
    out_offsets = np.empty(n_rows + 1, np.int64)
    out_offsets[0] = 0
    for i in range(n_rows):
        goi = geno_offset_idx[i]
        out_offsets[i + 1] = out_offsets[i] + (geno_stops[goi] - geno_starts[goi])
    total = out_offsets[n_rows]
    v_idxs = np.empty(total, geno_v_idxs.dtype)
    dst = 0
    for i in range(n_rows):
        goi = geno_offset_idx[i]
        s = geno_starts[goi]
        e = geno_stops[goi]
        for k in range(s, e):
            v_idxs[dst] = geno_v_idxs[k]
            dst += 1
    return v_idxs, out_offsets


@nb.njit(nogil=True, cache=True)
def _gather_alleles_numba(
    v_idxs, allele_bytes, allele_offsets
):  # pragma: no cover - njit
    """Gather variable-length allele bytestrings for ``v_idxs`` from the global
    allele byte buffer into flat ``(data, seq_offsets)``."""
    n = v_idxs.shape[0]
    seq_offsets = np.empty(n + 1, np.int64)
    seq_offsets[0] = 0
    for i in range(n):
        v = v_idxs[i]
        seq_offsets[i + 1] = seq_offsets[i] + (
            allele_offsets[v + 1] - allele_offsets[v]
        )
    data = np.empty(seq_offsets[n], np.uint8)
    dst = 0
    for i in range(n):
        v = v_idxs[i]
        s = allele_offsets[v]
        e = allele_offsets[v + 1]
        for k in range(s, e):
            data[dst] = allele_bytes[k]
            dst += 1
    return data, seq_offsets


register(
    "gather_alleles",
    numba=_gather_alleles_numba,
    rust=_gather_alleles_rust,
    default="rust",
)


def _gather_alleles(v_idxs, allele_bytes, allele_offsets):
    return get("gather_alleles")(
        np.ascontiguousarray(v_idxs, np.int32),
        np.ascontiguousarray(allele_bytes, np.uint8),
        np.ascontiguousarray(allele_offsets, np.int64),
    )


@nb.njit(nogil=True, cache=True)
def _compact_keep_numba(v_idxs, row_offsets, keep):  # pragma: no cover - njit
    """Drop variants where ``keep`` is False, rebuilding row offsets. The first
    param is per-variant values to compact -- either ``v_idxs`` itself or a
    parallel array (e.g. gathered dosage values) sharing the same row layout.
    Preserves the input dtype exactly (no down-cast)."""
    n_rows = row_offsets.shape[0] - 1
    new_offsets = np.empty(n_rows + 1, np.int64)
    new_offsets[0] = 0
    n_keep = 0
    for i in range(n_rows):
        for j in range(row_offsets[i], row_offsets[i + 1]):
            if keep[j]:
                n_keep += 1
        new_offsets[i + 1] = n_keep
    new_v = np.empty(n_keep, v_idxs.dtype)
    dst = 0
    for j in range(v_idxs.shape[0]):
        if keep[j]:
            new_v[dst] = v_idxs[j]
            dst += 1
    return new_v, new_offsets


register(
    "compact_keep_i32",
    numba=_compact_keep_numba,
    rust=_compact_keep_i32_rust,
    default="rust",
)
register(
    "compact_keep_f32",
    numba=_compact_keep_numba,
    rust=_compact_keep_f32_rust,
    default="rust",
)


def _compact_keep(v_idxs, row_offsets, keep):
    """Dispatch compact-keep by dtype, preserving the input dtype without down-cast.

    Routes int32 → compact_keep_i32 (Rust), float32 → compact_keep_f32 (Rust).
    All other dtypes (e.g. int16, int64 custom FORMAT fields, issue #231) fall
    back to the dtype-preserving numba kernel so values are never silently
    coerced.
    """
    values = np.ascontiguousarray(v_idxs)
    row_offsets = np.ascontiguousarray(row_offsets, np.int64)
    keep = np.ascontiguousarray(keep, np.bool_)
    if values.dtype == np.int32:
        return get("compact_keep_i32")(values, row_offsets, keep)
    if values.dtype == np.float32:
        return get("compact_keep_f32")(values, row_offsets, keep)
    # Arbitrary dtypes (custom FORMAT fields, e.g. int16, int64): dtype-preserving
    # numba fallback — never down-cast.
    return _compact_keep_numba(values, row_offsets, keep)


def _gather_rows_numba(geno_offset_idx, geno_offsets, geno_v_idxs):
    # geno_offsets is the normalized (2, n) form.
    return _gather_v_idxs_ss_numba(
        geno_offset_idx, geno_offsets[0], geno_offsets[1], geno_v_idxs
    )


register(
    "gather_rows_i32",
    numba=_gather_rows_numba,
    rust=_gather_rows_i32_rust,
    default="rust",
)
register(
    "gather_rows_f32",
    numba=_gather_rows_numba,
    rust=_gather_rows_f32_rust,
    default="rust",
)


def _gather_rows(
    geno_offset_idx: NDArray[np.intp],
    offsets: NDArray[np.int64],
    data: NDArray,
) -> tuple[NDArray, NDArray[np.int64]]:
    """Dispatch per-row gather (numba/rust), preserving data dtype.

    Routes int32 and float32 to typed Rust cores; all other dtypes fall back to
    the dtype-preserving numba kernel so values are never silently down-cast
    (e.g. custom per-call FORMAT fields, issue #231).
    """
    goi = np.ascontiguousarray(geno_offset_idx, np.int64)
    off2d = _as_starts_stops(offsets)
    data = np.ascontiguousarray(data)
    if data.dtype == np.int32:
        return get("gather_rows_i32")(goi, off2d, data)
    if data.dtype == np.float32:
        return get("gather_rows_f32")(goi, off2d, data)
    # Arbitrary custom-FORMAT-field dtypes (#231): no typed Rust core — use the
    # dtype-preserving numba kernel directly so values are never down-cast.
    return _gather_rows_numba(goi, off2d, data)


@nb.njit(nogil=True, cache=True)
def _fill_empty_scalar_numba(data, offsets, fill):  # pragma: no cover - njit
    """Insert one ``fill`` element into each empty row; copy non-empty rows
    through. Returns ``(new_data, new_offsets)``. Preserves ``data.dtype``."""
    n_rows = offsets.shape[0] - 1
    new_offsets = np.empty(n_rows + 1, np.int64)
    new_offsets[0] = 0
    for i in range(n_rows):
        ln = offsets[i + 1] - offsets[i]
        new_offsets[i + 1] = new_offsets[i] + (ln if ln > 0 else 1)
    new_data = np.empty(new_offsets[n_rows], data.dtype)
    for i in range(n_rows):
        s = offsets[i]
        e = offsets[i + 1]
        d = new_offsets[i]
        if e == s:
            new_data[d] = fill
        else:
            for k in range(s, e):
                new_data[d] = data[k]
                d += 1
    return new_data, new_offsets


register(
    "fill_empty_scalar_i32",
    numba=_fill_empty_scalar_numba,
    rust=_fill_empty_scalar_i32_rust,
    default="rust",
)
register(
    "fill_empty_scalar_f32",
    numba=_fill_empty_scalar_numba,
    rust=_fill_empty_scalar_f32_rust,
    default="rust",
)


def _fill_empty_scalar(data, offsets, fill):
    """Dtype-preserving dispatch for fill-empty-scalar.

    Routes int32 and float32 to typed Rust cores; all other dtypes (e.g.
    custom FORMAT fields, issue #231) fall back to the dtype-preserving numba
    kernel so values are never silently down-cast.
    """
    data = np.ascontiguousarray(data)
    offsets = np.ascontiguousarray(offsets, np.int64)
    if data.dtype == np.int32:
        return get("fill_empty_scalar_i32")(data, offsets, int(fill))
    if data.dtype == np.float32:
        return get("fill_empty_scalar_f32")(data, offsets, float(fill))
    # Arbitrary dtype (custom FORMAT fields): preserve dtype via numba fallback.
    return _fill_empty_scalar_numba(data, offsets, fill)


@nb.njit(nogil=True, cache=True)
def _fill_empty_seq_numba(
    data, var_offsets, seq_offsets, dummy
):  # pragma: no cover - njit
    """Two-level analogue of ``_fill_empty_scalar`` for allele bytestrings.
    Empty variant-rows receive one dummy allele of ``dummy`` bytes. Returns
    ``(new_data, new_var_offsets, new_seq_offsets)``. Preserves ``data.dtype``."""
    n_rows = var_offsets.shape[0] - 1
    L = dummy.shape[0]
    new_var = np.empty(n_rows + 1, np.int64)
    new_var[0] = 0
    for i in range(n_rows):
        nv = var_offsets[i + 1] - var_offsets[i]
        new_var[i + 1] = new_var[i] + (nv if nv > 0 else 1)
    total_vars = new_var[n_rows]
    new_seq = np.empty(total_vars + 1, np.int64)
    new_seq[0] = 0
    vptr = 0
    for i in range(n_rows):
        vs = var_offsets[i]
        ve = var_offsets[i + 1]
        if ve == vs:
            new_seq[vptr + 1] = new_seq[vptr] + L
            vptr += 1
        else:
            for v in range(vs, ve):
                vlen = seq_offsets[v + 1] - seq_offsets[v]
                new_seq[vptr + 1] = new_seq[vptr] + vlen
                vptr += 1
    new_data = np.empty(new_seq[total_vars], data.dtype)
    vptr = 0
    dptr = 0
    for i in range(n_rows):
        vs = var_offsets[i]
        ve = var_offsets[i + 1]
        if ve == vs:
            for k in range(L):
                new_data[dptr] = dummy[k]
                dptr += 1
            vptr += 1
        else:
            for v in range(vs, ve):
                bs = seq_offsets[v]
                be = seq_offsets[v + 1]
                for k in range(bs, be):
                    new_data[dptr] = data[k]
                    dptr += 1
                vptr += 1
    return new_data, new_var, new_seq


register(
    "fill_empty_seq_u8",
    numba=_fill_empty_seq_numba,
    rust=_fill_empty_seq_u8_rust,
    default="rust",
)
register(
    "fill_empty_seq_i32",
    numba=_fill_empty_seq_numba,
    rust=_fill_empty_seq_i32_rust,
    default="rust",
)


def _fill_empty_seq(data, var_offsets, seq_offsets, dummy):
    """Dtype-preserving dispatch for fill-empty-seq (two-level dummy-fill).

    Routes uint8 (allele bytes) and int32 (token windows) to typed Rust cores.
    All other dtypes fall back to the dtype-preserving numba kernel so values
    are never silently down-cast.
    """
    data = np.ascontiguousarray(data)
    var_offsets = np.ascontiguousarray(var_offsets, np.int64)
    seq_offsets = np.ascontiguousarray(seq_offsets, np.int64)
    dummy = np.ascontiguousarray(dummy, data.dtype)
    if data.dtype == np.uint8:
        return get("fill_empty_seq_u8")(data, var_offsets, seq_offsets, dummy)
    if data.dtype == np.int32:
        return get("fill_empty_seq_i32")(data, var_offsets, seq_offsets, dummy)
    # Arbitrary dtype: preserve via numba fallback.
    return _fill_empty_seq_numba(data, var_offsets, seq_offsets, dummy)


@nb.njit(nogil=True, cache=True)
def _fill_empty_fixed_numba(data, offsets, inner, fill):  # pragma: no cover - njit
    """Fixed-inner-stride analogue of ``_fill_empty_scalar`` for ``flank_tokens``.

    ``data`` holds ``n_var * inner`` tokens (variant-major); ``offsets`` are
    *variant-level* (``b*p + 1``). Each empty row receives one dummy variant of
    ``inner`` tokens all equal to ``fill``; non-empty rows pass through.
    Returns ``(new_data, new_offsets)``. Preserves ``data.dtype``."""
    n_rows = offsets.shape[0] - 1
    new_offsets = np.empty(n_rows + 1, np.int64)
    new_offsets[0] = 0
    for i in range(n_rows):
        nv = offsets[i + 1] - offsets[i]
        new_offsets[i + 1] = new_offsets[i] + (nv if nv > 0 else 1)
    total_vars = new_offsets[n_rows]
    new_data = np.empty(total_vars * inner, data.dtype)
    dptr = 0
    for i in range(n_rows):
        vs = offsets[i]
        ve = offsets[i + 1]
        if ve == vs:
            for _ in range(inner):
                new_data[dptr] = fill
                dptr += 1
        else:
            for k in range(vs * inner, ve * inner):
                new_data[dptr] = data[k]
                dptr += 1
    return new_data, new_offsets


register(
    "fill_empty_fixed_i32",
    numba=_fill_empty_fixed_numba,
    rust=_fill_empty_fixed_i32_rust,
    default="rust",
)
register(
    "fill_empty_fixed_f32",
    numba=_fill_empty_fixed_numba,
    rust=_fill_empty_fixed_f32_rust,
    default="rust",
)


def _fill_empty_fixed(data, offsets, inner, fill):
    """Dtype-preserving dispatch for fill-empty-fixed.

    Routes int32 and float32 to typed Rust cores; all other dtypes (e.g.
    custom FORMAT fields, issue #231) fall back to the dtype-preserving numba
    kernel so values are never silently down-cast.
    """
    data = np.ascontiguousarray(data)
    offsets = np.ascontiguousarray(offsets, np.int64)
    if data.dtype == np.int32:
        return get("fill_empty_fixed_i32")(data, offsets, int(inner), int(fill))
    if data.dtype == np.float32:
        return get("fill_empty_fixed_f32")(data, offsets, int(inner), float(fill))
    # Arbitrary dtype (custom FORMAT fields): preserve dtype via numba fallback.
    return _fill_empty_fixed_numba(data, offsets, inner, fill)


def _assemble_variant_buffers_numba_entry(*args, **kwargs):
    """Lazy wrapper for _assemble_variant_buffers_numba to avoid circular import.

    ``_flat_flanks`` imports ``_FlatWindow`` from ``_flat_variants`` at module
    level, so ``_flat_variants`` cannot import from ``_flat_flanks`` at module
    level. This thin wrapper defers the import to call time.
    """
    from ._flat_flanks import _assemble_variant_buffers_numba

    return _assemble_variant_buffers_numba(*args, **kwargs)


def _assemble_variant_buffers_rust(
    mode,
    v_idxs,
    row_offsets,
    alt_global,
    alt_off_global,
    ref_global,
    ref_off_global,
    want_ref_bytes,
    want_flank,
    ref_mode,
    alt_mode,
    flank_len,
    lut,
    v_contigs,
    v_starts,
    ilens,
    reference,
    ref_offsets,
    pad_char,
):
    """Dtype-selecting shim: routes to assemble_variant_buffers_u8/i32 by lut dtype.

    If ``lut`` is None (variants mode with no flank tokens), defaults to the u8
    monomorphization (token buffers are empty so dtype is irrelevant).
    """
    if lut is None:
        fn = _assemble_variant_buffers_u8_rust
        lut_arr = None
    else:
        lut_arr = np.asarray(lut)
        if lut_arr.dtype == np.uint8:
            fn = _assemble_variant_buffers_u8_rust
            lut_arr = np.ascontiguousarray(lut_arr, np.uint8)
        else:
            fn = _assemble_variant_buffers_i32_rust
            lut_arr = np.ascontiguousarray(lut_arr, np.int32)
    return fn(
        int(mode),
        np.ascontiguousarray(v_idxs, np.int32),
        np.ascontiguousarray(row_offsets, np.int64),
        np.ascontiguousarray(alt_global, np.uint8),
        np.ascontiguousarray(alt_off_global, np.int64),
        None if ref_global is None else np.ascontiguousarray(ref_global, np.uint8),
        None
        if ref_off_global is None
        else np.ascontiguousarray(ref_off_global, np.int64),
        bool(want_ref_bytes),
        bool(want_flank),
        int(ref_mode),
        int(alt_mode),
        int(flank_len),
        lut_arr,
        np.ascontiguousarray(v_contigs, np.int32),
        np.ascontiguousarray(v_starts, np.int32),
        np.ascontiguousarray(ilens, np.int32),
        np.ascontiguousarray(reference, np.uint8),
        np.ascontiguousarray(ref_offsets, np.int64),
        int(pad_char),
    )


register(
    "assemble_variant_buffers",
    numba=_assemble_variant_buffers_numba_entry,
    rust=_assemble_variant_buffers_rust,
    default="rust",
)


def get_variants_flat(
    haps: "Haps", idx: NDArray[np.integer], regions=None
) -> "_FlatVariants | _FlatVariantWindows":
    """Flat-buffer analog of :meth:`Haps._get_variants`: builds a
    :class:`_FlatVariants` on the pure-numpy hot path. Re-wrapping the
    result via :meth:`_FlatVariants.to_ragged` is byte-identical to the
    :class:`RaggedVariants` produced by ``_get_variants``.

    Replicates ONLY AF filtering (min_af/max_af); exonic filtering is not
    threaded into the variants output (its ``keep``/``keep_offsets`` params are
    dead in ``_get_variants``).
    """
    from .._flat import _Flat

    genotypes = haps.genotypes
    ploidy = genotypes.shape[-2]
    b = len(idx)

    # (b, ploidy) indices into the sparse-genotype offsets. Flatten C-order to
    # (b*ploidy,) so per-row slicing reproduces genotypes[r,s].to_packed() order.
    geno_offset_idx = haps._get_geno_offset_idx(idx, genotypes).reshape(-1)
    geno_offset_idx = np.ascontiguousarray(geno_offset_idx, np.intp)

    geno_offsets = np.asarray(genotypes.offsets, np.int64)
    geno_v_idxs = np.asarray(genotypes.data)

    # v_idxs: gathered per (b*ploidy) row; row_offsets length b*ploidy + 1.
    # Dispatch on offsets shape: 1-D contiguous vs 2-D starts/stops.
    v_idxs, row_offsets = _gather_rows(geno_offset_idx, geno_offsets, geno_v_idxs)

    # Unfiltered offsets needed for dosage parallel-gather + compaction.
    unfiltered_row_offsets = row_offsets

    # AF filtering (mirrors _get_variants). Computed before gathering dosage so we
    # can compact dosage with the SAME keep mask + UNFILTERED offsets.
    keep = None
    if haps.min_af is not None or haps.max_af is not None:
        geno_afs = np.asarray(haps.variants.info["AF"])[v_idxs]
        keep = np.full(len(v_idxs), True, np.bool_)
        if haps.min_af is not None:
            keep &= geno_afs >= haps.min_af
        if haps.max_af is not None:
            keep &= geno_afs <= haps.max_af

    # Dosage: parallel to genotypes (one value per variant, gathered by the SAME
    # genotype offset ranges). Gather against UNFILTERED offsets first.
    dosage_data = None
    if haps.dosages is not None and "dosage" in haps.var_fields:
        dos_offsets = np.asarray(haps.dosages.offsets, np.int64)
        dos_all = np.asarray(haps.dosages.data)
        # The returned row offsets == unfiltered_row_offsets by construction
        # (genotypes and dosages share offset structure), so discard them.
        dosage_data, _ = _gather_rows(geno_offset_idx, dos_offsets, dos_all)

    # Apply AF compaction to v_idxs / row_offsets / dosage.
    if keep is not None:
        v_idxs, row_offsets = _compact_keep(v_idxs, unfiltered_row_offsets, keep)
        if dosage_data is not None:
            dosage_data, _ = _compact_keep(dosage_data, unfiltered_row_offsets, keep)

    # Unphased ploidy-1 union: fold the C-order (b, ploidy) rows onto b rows by
    # keeping every ploidy-th offset. row_offsets has length b*ploidy + 1, so the
    # slice yields b + 1 offsets that span each region/sample's variants across all
    # stored haplotypes. v_idxs is untouched: hap-0's calls then hap-1's, concatenated
    # (no sort, no dedup; a hom call appears once per haplotype). Safe because the
    # downstream consumer is permutation-invariant (issue #222). eff_ploidy drives the
    # output shape and per-variant contig broadcasting below.
    eff_ploidy = ploidy
    if haps.unphased_union:
        row_offsets = np.ascontiguousarray(row_offsets[::ploidy])
        eff_ploidy = 1

    shape: tuple[int | None, ...] = (b, eff_ploidy, None)

    opt = haps.window_opt

    # --- Build scalar (non-allele) fields shared between both return paths ---
    fields: dict[str, Any] = {}

    # start: ALWAYS
    start_data = np.asarray(haps.variants.start)[v_idxs]
    fields["start"] = _Flat.from_offsets(start_data, shape, row_offsets)

    # ilen: if "ilen" in var_fields
    if "ilen" in haps.var_fields:
        ilen_data = np.asarray(haps.variants.ilen)[v_idxs]
        fields["ilen"] = _Flat.from_offsets(ilen_data, shape, row_offsets)

    # dosage: if dosages present and requested
    if dosage_data is not None:
        fields["dosage"] = _Flat.from_offsets(dosage_data, shape, row_offsets)

    # Custom per-call FORMAT fields (issue #231): same gather/compaction as dosage.
    for name, rag in haps.var_field_data.items():
        if name not in haps.var_fields:
            continue
        cf_off = np.asarray(rag.offsets, np.int64)
        cf_all = np.asarray(rag.data)
        cf_data, _ = _gather_rows(geno_offset_idx, cf_off, cf_all)
        if keep is not None:
            cf_data, _ = _compact_keep(cf_data, unfiltered_row_offsets, keep)
        fields[name] = _Flat.from_offsets(cf_data, shape, row_offsets)

    # other info fields
    for k in haps.var_fields:
        if k in {"alt", "start", "ref", "ilen", "dosage"} or k in haps.var_field_data:
            continue
        info_data = np.asarray(haps.variants.info[k])[v_idxs]
        fields[k] = _Flat.from_offsets(info_data, shape, row_offsets)

    # --- Step 1: Compute shared kernel inputs ---
    stat = haps.ffi_static
    needs_fetch = (
        regions is not None
        and haps.token_lut is not None
        and (
            (issubclass(haps.kind, _FlatVariantWindows) and opt is not None)
            or bool(haps.flank_length)
        )
    )
    if needs_fetch:
        regions_arr = np.asarray(regions)
        group_contigs = np.repeat(regions_arr[:, 0], eff_ploidy)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets)).astype(np.int32)
    else:
        v_contigs = np.zeros(len(v_idxs), np.int32)

    ref_present = "ref" in haps.var_fields and haps.variants.ref is not None
    ref_global = ref_off_global = None
    if ref_present or (
        issubclass(haps.kind, _FlatVariantWindows)
        and opt is not None
        and (opt.ref == "allele")
    ):
        ref_global = np.asarray(haps.variants.ref.data).view(np.uint8)
        ref_off_global = np.asarray(haps.variants.ref.offsets, np.int64)

    # --- Step 2: variant-windows kind: emit per-allele token buffers (early return) ---
    if (
        regions is not None
        and issubclass(haps.kind, _FlatVariantWindows)
        and opt is not None
    ):
        L = opt.flank_length
        ref_mode = 1 if opt.ref == "window" else 2
        alt_mode = 1 if opt.alt == "window" else 2
        bufs = get("assemble_variant_buffers")(
            1,  # windows mode
            v_idxs,
            row_offsets,
            stat.alt_alleles,
            stat.alt_offsets,
            ref_global,
            ref_off_global,
            False,  # want_ref_bytes (windows mode emits tokens, not raw bytes)
            False,  # want_flank
            ref_mode,
            alt_mode,
            L,
            haps.token_lut,
            v_contigs,
            stat.v_starts,
            stat.ilens,
            stat.ref,
            stat.ref_offsets,
            haps.reference.pad_char,
        )
        wshape = (b, eff_ploidy, None, None)
        wfields = {k: v for k, v in fields.items() if k not in ("alt", "ref")}
        win = _FlatVariantWindows(wfields)
        for name, (data, seq_off) in bufs.items():
            fw = _FlatWindow(data, np.asarray(seq_off, np.int64), row_offsets, wshape)
            setattr(win, name, fw)
        if haps.dummy_variant is not None:
            win = win.fill_empty_groups(
                haps.dummy_variant, unk=haps.unknown_token, flank_length=L
            )
        return win

    # --- Step 3: plain-variants path: route allele bytes + flank tokens through kernel ---
    want_flank = bool(
        haps.flank_length and haps.token_lut is not None and regions is not None
    )
    L = haps.flank_length or 0
    bufs = get("assemble_variant_buffers")(
        0,  # variants mode
        v_idxs,
        row_offsets,
        stat.alt_alleles,
        stat.alt_offsets,
        ref_global,
        ref_off_global,
        ref_present,  # want_ref_bytes
        want_flank,
        0,  # ref_mode (unused in variants mode)
        0,  # alt_mode (unused)
        L,
        haps.token_lut,
        v_contigs,
        stat.v_starts,
        stat.ilens,
        stat.ref if stat.ref is not None else np.zeros(0, np.uint8),
        stat.ref_offsets if stat.ref_offsets is not None else np.zeros(1, np.int64),
        haps.reference.pad_char if haps.reference is not None else 0,
    )

    # Build fields in ORIGINAL insertion order (alt FIRST, then start, ref, rest).
    # Prepend alt; reconstruct from scalar fields inserting ref after start.
    final_fields: dict[str, Any] = {}
    alt_data, alt_seq_off = bufs["alt"]
    final_fields["alt"] = _FlatAlleles(
        np.asarray(alt_data, np.uint8),
        np.asarray(alt_seq_off, np.int64),
        row_offsets,
        shape,
    )
    for k, v in fields.items():
        if k == "start":
            final_fields["start"] = v
            # Insert ref immediately after start (original order: alt, start, ref, ilen, ...)
            if "ref" in bufs:
                ref_data, ref_seq_off = bufs["ref"]
                final_fields["ref"] = _FlatAlleles(
                    np.asarray(ref_data, np.uint8),
                    np.asarray(ref_seq_off, np.int64),
                    row_offsets,
                    shape,
                )
        else:
            final_fields[k] = v

    flat = _FlatVariants(final_fields)

    if "flank_tokens" in bufs:
        tok, off = bufs["flank_tokens"]
        flat.flank_tokens = _Flat.from_offsets(
            tok, (b, eff_ploidy, None, 2 * L), np.asarray(off, np.int64)
        )

    # dummy-variant empty-group fill (scalars, alleles, and flank_tokens).
    if haps.dummy_variant is not None:
        flat = flat.fill_empty_groups(haps.dummy_variant, unk=haps.unknown_token)

    return flat
