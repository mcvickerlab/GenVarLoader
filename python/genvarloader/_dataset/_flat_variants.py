"""Flat-buffer analog of RaggedVariants: pure-numpy (data, offsets) per field,
no awkward on the hot path. Converts to RaggedVariants only via to_ragged()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numba as nb
import numpy as np
from numpy.typing import NDArray

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
        import awkward as ak
        from awkward.contents import ListOffsetArray, NumpyArray, RegularArray
        from awkward.index import Index

        # Build the ragged variant-row node (leaf -> bytestring -> variant-row
        # ListOffsetArray) giving (n_groups, ~v, ~l) where n_groups =
        # len(var_offsets) - 1. This mirrors the inner part of
        # _build_allele_layout but WITHOUT the ploidy RegularArray wrap.
        leaf = NumpyArray(
            np.ascontiguousarray(self.byte_data, np.uint8),
            parameters={"__array__": "byte"},
        )
        l_content = ListOffsetArray(
            Index(np.asarray(self.seq_offsets, np.int64)),
            leaf,
            parameters={"__array__": "bytestring"},
        )
        vl_content = ListOffsetArray(
            Index(np.asarray(self.var_offsets, np.int64)), l_content
        )
        # Wrap with RegularArrays for the INNER fixed dims (everything except the
        # outermost, which is implied by the remaining group count). Shape-driven:
        # this makes to_ragged() agnostic to ploidy after a scalar-scalar squeeze.
        #   fixed=[b,p]   -> reversed([p])   -> RegularArray(vl,p) -> (b,p,~v,~l)
        #   fixed=[p]     -> reversed([])    -> no wrap            -> (p,~v,~l)
        #   fixed=[b,s,p] -> reversed([s,p]) -> nested             -> (b,s,p,~v,~l)
        node = vl_content
        fixed = [d for d in self.shape if d is not None]
        for size in reversed(fixed[1:]):
            node = RegularArray(node, size)
        return ak.Array(node)

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
            raise ValueError("_FlatAlleles.__getitem__: leading axis is the ragged axis")
        start, stop, step = key.indices(n_inst)
        if step != 1:
            raise ValueError("_FlatAlleles slicing supports step=1 only")
        rows_per_inst = (len(self.var_offsets) - 1) // n_inst if n_inst else 0
        r0, r1 = start * rows_per_inst, stop * rows_per_inst
        v0, v1 = int(self.var_offsets[r0]), int(self.var_offsets[r1])
        new_var = np.ascontiguousarray(self.var_offsets[r0 : r1 + 1] - self.var_offsets[r0])
        new_seq = np.ascontiguousarray(self.seq_offsets[v0 : v1 + 1] - self.seq_offsets[v0])
        new_bytes = self.byte_data[int(self.seq_offsets[v0]) : int(self.seq_offsets[v1])]
        new_shape = (stop - start,) + self.shape[1:]
        return _FlatAlleles(new_bytes, new_seq, new_var, new_shape)


@dataclass(slots=True)
class _FlatWindow:
    """Two-level flat token buffer for ref/alt windows, shape (b, p, ~v, ~win).

    Mirrors _FlatAlleles but `data` holds tokens (configured int dtype), not bytes,
    so to_ragged() drops the byte/bytestring awkward parameters. Both inner axes
    (variant count and window length) are ragged, so to_ragged() returns a raw
    awkward Array (seqpro Ragged supports only one ragged axis).
    """

    data: NDArray  # tokens (uint8 or int32), flat
    seq_offsets: NDArray[np.int64]  # per-variant window offsets, n_variants + 1
    var_offsets: NDArray[np.int64]  # per (instance, ploid) offsets, b*p + 1
    shape: tuple[int | None, ...]

    def to_ragged(self):
        import awkward as ak
        from awkward.contents import ListOffsetArray, NumpyArray, RegularArray
        from awkward.index import Index

        leaf = NumpyArray(np.ascontiguousarray(self.data))
        l_content = ListOffsetArray(Index(np.asarray(self.seq_offsets, np.int64)), leaf)
        vl_content = ListOffsetArray(
            Index(np.asarray(self.var_offsets, np.int64)), l_content
        )
        node = vl_content
        fixed = [d for d in self.shape if d is not None]
        for size in reversed(fixed[1:]):
            node = RegularArray(node, size)
        return ak.Array(node)

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

    def fill_empty_groups(self, dummy: "DummyVariant") -> "_FlatVariants":
        """Insert one dummy variant into each empty (b*p) group; non-empty
        groups are unchanged. Every field shares the same empty-row pattern, so
        the rebuilt offsets stay consistent across fields."""
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
        return _FlatVariants(new_fields)


@nb.njit(nogil=True, cache=True)
def _gather_v_idxs(
    geno_offset_idx, geno_offsets, geno_v_idxs
):  # pragma: no cover - njit
    """Gather per-row variant indices: for each row's offset slice into the
    sparse arrays, copy its values out into flat ``(data, offsets)``.

    ``geno_offsets`` must be 1-D contiguous (length n_rows + 1).  For the
    non-contiguous (2, n_rows) starts/stops form use :func:`_gather_v_idxs_ss`.
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
def _gather_v_idxs_ss(
    geno_offset_idx, geno_starts, geno_stops, geno_v_idxs
):  # pragma: no cover - njit
    """Like :func:`_gather_v_idxs` but for non-contiguous (starts, stops) offsets.

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
def _gather_alleles(v_idxs, allele_bytes, allele_offsets):  # pragma: no cover - njit
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


@nb.njit(nogil=True, cache=True)
def _compact_keep(v_idxs, row_offsets, keep):  # pragma: no cover - njit
    """Drop variants where ``keep`` is False, rebuilding row offsets. The first
    param is per-variant values to compact -- either ``v_idxs`` itself or a
    parallel array (e.g. gathered dosage values) sharing the same row layout."""
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


def _gather_rows(
    geno_offset_idx: NDArray[np.intp],
    offsets: NDArray[np.int64],
    data: NDArray,
) -> tuple[NDArray, NDArray[np.int64]]:
    """Dispatch to the correct gather kernel based on offset array shape.

    ``offsets`` may be:
    - 1-D ``(n + 1,)``: contiguous offsets — use :func:`_gather_v_idxs`.
    - 2-D ``(2, n)``: non-contiguous starts/stops — use :func:`_gather_v_idxs_ss`.
    """
    if offsets.ndim == 1:
        return _gather_v_idxs(geno_offset_idx, offsets, data)
    else:
        return _gather_v_idxs_ss(geno_offset_idx, offsets[0], offsets[1], data)


@nb.njit(nogil=True, cache=True)
def _fill_empty_scalar(data, offsets, fill):  # pragma: no cover - njit
    """Insert one ``fill`` element into each empty row; copy non-empty rows
    through. Returns ``(new_data, new_offsets)``."""
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


@nb.njit(nogil=True, cache=True)
def _fill_empty_seq(data, var_offsets, seq_offsets, dummy):  # pragma: no cover - njit
    """Two-level analogue of ``_fill_empty_scalar`` for allele bytestrings.
    Empty variant-rows receive one dummy allele of ``dummy`` bytes. Returns
    ``(new_data, new_var_offsets, new_seq_offsets)``."""
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
    new_data = np.empty(new_seq[total_vars], np.uint8)
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


def get_variants_flat(
    haps: "Haps", idx: NDArray[np.integer], regions=None
) -> "_FlatVariants | _FlatVariantWindows":
    """Flat-buffer analog of :meth:`Haps._get_variants`: builds a
    :class:`_FlatVariants` with no awkward on the hot path. Re-wrapping the
    result via :meth:`_FlatVariants.to_ragged` is byte-identical to the awkward
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

    shape: tuple[int | None, ...] = (b, ploidy, None)

    fields: dict[str, Any] = {}

    # alt: ALWAYS (required)
    alt_bytes = np.asarray(haps.variants.alt.data).view(np.uint8)
    alt_off = np.asarray(haps.variants.alt.offsets, np.int64)
    alt_data, alt_seq_off = _gather_alleles(v_idxs, alt_bytes, alt_off)
    fields["alt"] = _FlatAlleles(alt_data, alt_seq_off, row_offsets, shape)

    # start: ALWAYS (added unconditionally by _get_variants)
    start_data = np.asarray(haps.variants.start)[v_idxs]
    fields["start"] = _Flat.from_offsets(start_data, shape, row_offsets)

    # ref: if "ref" in var_fields
    if "ref" in haps.var_fields:
        ref_bytes = np.asarray(haps.variants.ref.data).view(np.uint8)
        ref_off = np.asarray(haps.variants.ref.offsets, np.int64)
        ref_data, ref_seq_off = _gather_alleles(v_idxs, ref_bytes, ref_off)
        fields["ref"] = _FlatAlleles(ref_data, ref_seq_off, row_offsets, shape)

    # ilen: if "ilen" in var_fields
    if "ilen" in haps.var_fields:
        ilen_data = np.asarray(haps.variants.ilen)[v_idxs]
        fields["ilen"] = _Flat.from_offsets(ilen_data, shape, row_offsets)

    # dosage: if dosages present and requested
    if dosage_data is not None:
        fields["dosage"] = _Flat.from_offsets(dosage_data, shape, row_offsets)

    # other info fields
    for k in haps.var_fields:
        if k in {"alt", "start", "ref", "ilen", "dosage"}:
            continue
        info_data = np.asarray(haps.variants.info[k])[v_idxs]
        fields[k] = _Flat.from_offsets(info_data, shape, row_offsets)

    flat = _FlatVariants(fields)

    # variant-windows kind: emit per-allele window/allele token buffers (a
    # different output type) and return early.
    opt = haps.window_opt
    if (
        regions is not None
        and issubclass(haps.kind, _FlatVariantWindows)
        and opt is not None
    ):
        from ._flat_flanks import (
            compute_alt_window,
            compute_ref_window,
            tokenize_alleles,
        )

        L = opt.flank_length
        lut = haps.token_lut
        starts_v = np.asarray(haps.variants.start)[v_idxs]
        ilens_v = np.asarray(haps.variants.ilen)[v_idxs]
        regions = np.asarray(regions)
        group_contigs = np.repeat(regions[:, 0], ploidy)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets))
        wshape = (b, ploidy, None, None)
        wfields = {k: v for k, v in fields.items() if k not in ("alt", "ref")}
        win = _FlatVariantWindows(wfields)

        if opt.ref == "window":
            rw = compute_ref_window(
                haps.reference, v_contigs, starts_v, ilens_v, L, lut, row_offsets
            )
            rw.shape = wshape
            win.ref_window = rw
        else:  # "allele": bare tokenized ref allele
            ref_bytes = np.asarray(haps.variants.ref.data).view(np.uint8)
            ref_off = np.asarray(haps.variants.ref.offsets, np.int64)
            ref_data, ref_seq_off = _gather_alleles(v_idxs, ref_bytes, ref_off)
            rw = tokenize_alleles(ref_data, ref_seq_off, lut, row_offsets)
            rw.shape = wshape
            win.ref = rw

        if opt.alt == "window":
            aw = compute_alt_window(
                haps.reference,
                v_contigs,
                starts_v,
                ilens_v,
                alt_data,
                alt_seq_off,
                L,
                lut,
                row_offsets,
            )
            aw.shape = wshape
            win.alt_window = aw
        else:  # "allele": bare tokenized alt allele
            aw = tokenize_alleles(alt_data, alt_seq_off, lut, row_offsets)
            aw.shape = wshape
            win.alt = aw

        return win

    # ride-along flank tokens on the plain variants output.
    if haps.flank_length and haps.token_lut is not None and regions is not None:
        if haps.dummy_variant is not None:
            raise ValueError(
                "dummy_variant cannot be combined with flank tokens: the empty-group"
                " fill rebuilds the variant fields and would drop the ride-along"
                " flank_tokens. Disable one of them."
            )
        from ._flat_flanks import compute_flank_tokens

        L = haps.flank_length
        starts_v = np.asarray(haps.variants.start)[v_idxs]
        ilens_v = np.asarray(haps.variants.ilen)[v_idxs]
        regions = np.asarray(regions)
        group_contigs = np.repeat(regions[:, 0], ploidy)  # (b*p,)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets))  # (n_var,)

        tok, off = compute_flank_tokens(
            haps.reference,
            v_contigs,
            starts_v,
            ilens_v,
            L,
            haps.token_lut,
            row_offsets,
        )
        flat.flank_tokens = _Flat.from_offsets(tok, (b, ploidy, None, 2 * L), off)

    # dummy-variant empty-group fill (plain variants output only).
    if haps.dummy_variant is not None:
        flat = flat.fill_empty_groups(haps.dummy_variant)

    return flat
