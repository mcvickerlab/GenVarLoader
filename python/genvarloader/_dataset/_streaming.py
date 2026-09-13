from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Literal,
    NamedTuple,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np
import polars as pl
import seqpro as sp
from genoray._contigs import ContigNormalizer
from numpy.typing import NDArray
from seqpro.rag import Ragged

from .._ragged import RaggedAnnotatedHaps, RaggedSeqs
from ._rag_variants import RaggedVariants
from .._torch import requires_torch
from .._variants._utils import path_is_pgen, path_is_vcf
from ._utils import bed_to_regions

if TYPE_CHECKING:
    import torch.utils.data as td
    import genoray

    from .._ragged import RaggedIntervals, RaggedTracks
    from .._types import IntervalTrack
    from ._flat_variants import VarWindowOpt
    from ._insertion_fill import InsertionFill
    from ._track_stream import _TrackBackend

# Wave B PR-B3a (#304): human-readable names for `with_seqs`' output kinds, used to
# report a `with_seqs(...)` value back in error messages (`_iter_batches`'s
# var_fields/non-variants guard) -- the inverse of `with_seqs`'s own `kind_map`.
_SEQ_KIND_NAMES: dict[type, str] = {
    RaggedSeqs: "haplotypes",
    RaggedAnnotatedHaps: "annotated",
    RaggedVariants: "variants",
    # Wave B PR-B4 (#304): `with_seqs("variant-windows")` output is a plain `dict[str,
    # Ragged]` (there is no dedicated ragged-container class for it, unlike the other
    # three kinds), so `dict` itself is the `_seq_kind` sentinel -- see `with_seqs`.
    dict: "variant-windows",
}

# Wave B PR-B3a (#304) name-collision guard: `next_batch_variants()`'s FFI dict
# (`stream_engine.rs`/`record_stream/engine.rs`) is keyed by field name and
# `PyDict::set_item` silently overwrites -- a live-source INFO/index column that
# happened to be named one of these would clobber the corresponding fixed-schema
# array with no error (wrong data, no exception). Cheapest correct fix: exclude any
# such colliding name from a backend's `available_var_fields` at the source, so it
# can never be selected via `var_fields` in the first place -- `with_settings`'s
# `not in available_var_fields` check then raises the normal, clear "not available"
# `ValueError` for it. This is also semantically the right call independent of the
# FFI concern: "alt"/"start"/"ilen"/"ref" already name the builtin fields, so a
# same-named INFO/index column would be ambiguous to request even if it could be
# threaded through safely.
_RESERVED_VAR_FIELD_NAMES = frozenset(
    {"alt", "alt_offsets", "start", "ilen", "offsets", "ref", "ref_offsets"}
)
# Forward-looking note (final review, PR-B3a/B3b, #304): `variant-windows` mode
# builds a SEPARATE FFI dict with its own fixed keys -- `ref_window`/`alt_window`/
# `ref`/`alt` token buffers plus their `<name>_offsets` (see `with_seqs`'s
# `_variant_windows` branch) -- that aren't in this set. `var_fields` ride-alongs
# are currently guarded OFF in window mode (see `active_var_fields`/`with_seqs`),
# so no user column can collide there YET. If/when window-mode ride-alongs are
# enabled (tracked follow-up), this exclusion set must also cover those
# windows-dict token-buffer keys, or a same-named INFO/index column could clobber
# a token buffer via the same `PyDict::set_item` silent-overwrite hazard this set
# already guards against for the plain "variants" dict.

# Wave B PR-B3a (#304): `with_seqs("variants")`'s pre-var_fields default, reproduced
# byte-for-byte when `var_fields` is never set (see `active_var_fields` and every
# backend's `build_engine`).
_DEFAULT_VAR_FIELDS = ("alt", "ilen", "start")

# Wave B PR-B3b review (#304, Important 1): the streaming FFI (`CallVals`/`InfoVals`
# in `stream_engine.rs`) can carry a per-call FORMAT/dosage column into Rust WITHOUT
# a dtype-breaking cast only for these exact dtypes -- `float32` (dosage's fixed
# `DOSAGE_TYPE`), `int32`, and `int16` (genoray's one real custom FORMAT field,
# `mutcat`, is always `int16`). A field registered with any OTHER dtype (e.g. a
# hypothetical `float64`/`int64`/`uint32`) cannot be forwarded losslessly-AND-
# dtype-preservingly, so `_Svar1Backend.__init__` marks it available but NOT
# servable -- `with_settings` then raises a clear `NotImplementedError` naming it,
# rather than `build_engine` silently coercing (the pre-fix behavior: EVERY
# non-float dtype was force-cast to `int32` via `np.ascontiguousarray`, breaking
# byte-identical parity with the written path for e.g. `mutcat` and unsafely
# truncating/wrapping any wider hypothetical integer dtype).
_SUPPORTED_CALL_FIELD_DTYPES = frozenset(
    {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int16)}
)


def _normalize_var_fields(var_fields: "list[str] | None") -> list[str]:
    """Resolve `var_fields`, falling back to the builtin default when `None`.

    Shared by every backend's `build_engine` and
    `StreamingDataset.active_var_fields` so the default list literal exists in
    exactly one place.
    """
    return list(var_fields) if var_fields is not None else list(_DEFAULT_VAR_FIELDS)


def _win_mode_kwargs(
    var_window: "tuple[NDArray, np.dtype, VarWindowOpt] | None",
) -> dict[str, object]:
    """Build the `win_*` keyword arguments for `with_seqs("variant-windows")`.

    Every record-style backend's `build_engine` (SVAR1/VCF/PGEN) forwards these to
    its Rust engine constructor (Wave B PR-B4, #304). `var_window` is the
    `(lut, lut_dtype, opt)` bundle `StreamingDataset._iter_batches` threads through
    (`None` when variant-windows output was not requested); one shared builder here
    -- rather than duplicating the same dtype dispatch three times -- keeps the
    three backends' `#[new]` calls symmetric, matching
    `WindowModeConfig::from_python` (`src/variants/mod.rs`), the single Rust-side
    decode point for this exact wire format.
    """
    if var_window is None:
        return {}
    lut, lut_dtype, opt = var_window
    kwargs: dict[str, object] = {
        "win_ref_mode": opt.ref,
        "win_alt_mode": opt.alt,
        "win_flank_len": int(opt.flank_length),
    }
    if lut_dtype == np.uint8:
        kwargs["win_token_lut_u8"] = np.ascontiguousarray(lut, np.uint8)
    else:
        kwargs["win_token_lut_i32"] = np.ascontiguousarray(lut, np.int32)
    return kwargs


# SVAR2 reconstruct super-batch: the rayon dispatch grain. Sized to saturate cores
# (n_work = rows*ploidy must be >> num_threads) while the output buffer stays
# max_mem-bounded and cohort-independent (#284). This is the measured knee
# (benchmarking/streaming/svar2_superbatch_sweep.py; recorded in
# docs/roadmaps/streaming-dataset.md); `StreamingDataset.__init__` refines it down
# against `max_mem_bytes` once that's computed (the backend is built before then, so
# it can't see the real budget at construction time -- see `_Svar2Backend.__init__`).
SUPERBATCH_TARGET_ROWS = 4096  # confirmed by the Task-5 sweep


def _super_batch_bounds(n_rows: int, sb_rows: int) -> Iterator[tuple[int, int]]:
    """``(lo, hi)`` super-batch row spans one window of ``n_rows`` is driven in.

    ``sb_rows >= n_rows`` degenerates to a single full-width span, which is exactly
    the shape of the drives that do *not* super-batch -- so every drive and the
    batch counter can share one nesting instead of re-deriving it (issue #379).
    """
    for lo in range(0, n_rows, max(1, sb_rows)):
        yield lo, min(lo + sb_rows, n_rows)


def _batch_bounds(lo: int, hi: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """``(lo, hi)`` batch row spans within one super-batch span.

    Bounded by ``hi``, not by the enclosing window: a super-batch whose width is
    not a multiple of ``batch_size`` ends in a PARTIAL batch, and both the drive
    and :meth:`StreamingDataset.n_batches` must account for it (issue #379).
    """
    for b_lo in range(lo, hi, batch_size):
        yield b_lo, min(b_lo + batch_size, hi)


def _parse_max_mem(max_mem: str | int) -> int:
    """Bytes from an int or a size string like '512MB' / '1g' / '2GiB'."""
    if isinstance(max_mem, int):
        return int(max_mem)
    s = str(max_mem).strip().lower().replace("ib", "b")
    units = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
    }
    # Multi-char suffixes first: "1tb" must match "tb", not fall through to bare "t"
    # (checking "t" first would strip only the trailing "b" and leave "1t" behind).
    for suffix in ("tb", "gb", "mb", "kb", "t", "g", "m", "k", "b"):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * units[suffix])
    return int(float(s))  # bare number = bytes


def _declared_info_numeric_dtypes(vcf: "genoray.VCF") -> dict[str, bool]:
    """Numeric (Integer/Float) INFO fields declared in a VCF/BCF header.

    Wave B PR-B3a (#304): the live-source counterpart of ``_Variants.available_info_fields``
    (`_haps.py`, which scans a WRITTEN ``variants.arrow``'s polars schema) -- here there is no
    on-disk arrow table, so this scans the VCF header directly via ``cyvcf2``'s
    ``header_iter()`` (the same primitive ``genoray.VCF._declared_info_fields`` uses, generalized
    from a fixed candidate tuple to "every declared INFO field"). Flag/String/Character INFO
    types are excluded -- they have no numeric representation, matching the written path's
    numeric-only filter.

    Takes an already-constructed ``genoray.VCF`` (only its header is read here) rather than a
    path, so callers that already have one in scope (`_VcfBackend.__init__`) don't pay for a
    second ``genoray.VCF(...)`` construction -- which, unlike this header scan, eagerly loads
    the on-disk ``.gvi`` index (see `genoray.VCF.__init__`'s ``with_gvi_index`` default).

    Args:
        vcf: An already-opened ``genoray.VCF``.

    Returns:
        Mapping of INFO field name -> ``is_float`` (``True`` for ``Type=Float``, ``False``
        for ``Type=Integer``), in VCF header declaration order.
    """
    out: dict[str, bool] = {}
    for h in vcf._vcf.header_iter():
        info = h.info()
        if info.get("HeaderType") != "INFO":
            continue
        name = info.get("ID")
        htype = info.get("Type")
        if name is None or name in _RESERVED_VAR_FIELD_NAMES:
            continue
        if htype == "Integer":
            out[name] = False
        elif htype == "Float":
            out[name] = True
        # Flag/String/Character: not numeric, excluded (matches
        # `_Variants.available_info_fields`'s `v.is_numeric()` filter).
    return out


class _ItvArrays(NamedTuple):
    """One track's WINDOW intervals, already coerced for the FFI.

    The Rust kernels take C-contiguous arrays of a fixed dtype, so a coercion
    has to happen somewhere. Doing it per BATCH repeats a WINDOW-scale copy
    ``ceil(window_rows / batch_size)`` times per track:
    ``BigWigs._intervals_from_offsets`` slices ``coordinates[:, 0]`` /
    ``coordinates[:, 1]`` out of one interleaved ``(n_intervals, 2)`` buffer,
    so ``starts``/``ends`` are strided views and each coercion is a genuine
    materialization. (``Table``'s happen to be contiguous already, and so does
    the written path's on-disk interval store, but that is a property of those
    sources, not a contract the kernels' callers can assume.) Coercing ONCE per
    window -- at :func:`_coerce_window_itvs`, right after the window read --
    keeps the same bytes for one copy instead of one per batch.

    Deliberately NOT `_ffi_array`: unlike ``geno_v_idxs`` (a genuine
    dataset-global memmap, where ``_ffi_array`` guards against an accidental
    sample-scale copy), these arrays are WINDOW-bounded -- freshly read per
    window by ``_TrackBackend.read_window``, never a memmap over the whole
    dataset -- so a copy here is window-sized by construction.
    """

    starts: "NDArray[np.int32]"
    ends: "NDArray[np.int32]"
    values: "NDArray[np.float32]"
    #: CSR row offsets over the WHOLE window's ``(row,)`` cells, so a batch is
    #: selected with ``offset_idxs`` rather than by re-slicing the intervals.
    offsets: "NDArray[np.int64]"


def _coerce_window_itvs(per_track: "list[RaggedIntervals]") -> "list[_ItvArrays]":
    """Coerce one window's per-track intervals for the FFI, once per window.

    The ``RaggedIntervals`` handed back by ``_TrackBackend.read_window`` are
    shaped ``(n_regions, n_samples, None)``; the kernels only ever read the
    flat data buffers and the CSR offsets, which a reshape to ``(row, None)``
    would leave untouched, so this both flattens and coerces in one step.

    Args:
        per_track: One ``RaggedIntervals`` per track, name-sorted.

    Returns:
        One :class:`_ItvArrays` per track, in the same order.
    """
    return [
        _ItvArrays(
            starts=np.ascontiguousarray(itvs.starts.data, np.int32),
            ends=np.ascontiguousarray(itvs.ends.data, np.int32),
            values=np.ascontiguousarray(itvs.values.data, np.float32),
            offsets=np.ascontiguousarray(itvs.starts.offsets, np.int64),
        )
        for itvs in per_track
    ]


def _tracks_from_intervals(
    per_track: "list[_ItvArrays]",
    offset_idxs: NDArray[np.int64],
    starts: NDArray[np.int32],
    lengths: NDArray[np.int64],
) -> "RaggedTracks":
    """Rasterize one batch's intervals into a `(batch, n_tracks, None)` Ragged.

    Mirrors `Tracks._call_float32`'s rasterization (`_tracks.py:388-419`): each
    track fills its own contiguous `n_per_track` block with a single vectorized
    `intervals_to_tracks` call. The per-cell VALUES are what byte-parity is
    defined over, and those come from the kernel calls -- do not change them.

    It deliberately does NOT copy that function's final assembly step. The
    written path builds a track-major buffer but labels it with
    `lengths_to_offsets(repeat(lengths, "b -> b t"))`, whose flattened cumsum
    is `(b, t)`-ordered. Track-major `(t, b)` and interleaved `(b, t)` agree
    only when `batch == 1`, which is the only way the written path is ever
    reached in the parity oracle (`Dataset[r, s]` indexes one cell). Streaming
    yields real batches, so this reorders the track-major buffer into `(b, t)`
    order before labelling it. Copying the written assembly verbatim here
    produced track `t`'s row `b+1` where row `b`'s track `t+1` belonged.

    Args:
        per_track: One `_ItvArrays` per track, name-sorted, each covering the
            WHOLE window's `(row,)` cells -- not pre-sliced to the batch. The
            batch is selected with `offset_idxs`, which is the mechanism
            `_call_float32` itself uses (its `o_idx` indexes the un-sliced
            interval array), and which avoids a redundant reshape-and-slice per
            batch. Already FFI-coerced once per window by
            `_coerce_window_itvs`; do NOT re-coerce here (see `_ItvArrays`).
        offset_idxs: `(batch,)` int64 row indices into `per_track`, selecting
            this batch's cells out of the window.
        starts: `(batch,)` int32 query starts, one per row.
        lengths: `(batch,)` int64 output length, one per row.

    Returns:
        A `RaggedTracks` of shape `(batch, n_tracks, None)`.
    """
    from .._ragged import RaggedTracks
    from .._utils import lengths_to_offsets
    from ._intervals import intervals_to_tracks
    from ._svar2_haps import _ragged_arange_gather

    n_tracks = len(per_track)
    batch = len(lengths)
    ofsts_per_t = lengths_to_offsets(lengths)
    n_per_track = int(ofsts_per_t[-1])
    out = np.empty(n_tracks * n_per_track, np.float32)

    for t, itvs in enumerate(per_track):
        intervals_to_tracks(
            offset_idxs=offset_idxs,
            starts=starts,
            itv_starts=itvs.starts,
            itv_ends=itvs.ends,
            itv_values=itvs.values,
            itv_offsets=itvs.offsets,
            out=out[t * n_per_track : (t + 1) * n_per_track],
            out_offsets=ofsts_per_t,
        )

    # `out` is now TRACK-major: block `t` holds every row of track `t`, so its
    # flat row order is `(t, b)`. A `(b, t, None)` Ragged indexes element
    # `b * n_tracks + t`, i.e. `(b, t)` order. Those two coincide only when
    # `batch == 1`, which is why the written path (whose oracle is indexed one
    # cell at a time) never surfaces the difference -- see the module note in
    # `_tracks_from_intervals`'s docstring. Reorder once, vectorized.
    tm_offsets = lengths_to_offsets(np.tile(np.asarray(lengths), n_tracks))
    k = np.arange(batch * n_tracks)
    perm = (k % n_tracks) * batch + (k // n_tracks)
    data, out_offsets = _ragged_arange_gather(out, tm_offsets, perm)

    result = Ragged.from_offsets(data, (batch, n_tracks, None), out_offsets)  # type: ignore[bad-argument-type, no-matching-overload]  # shape tuple carries an explicit None for the ragged axis
    return cast(RaggedTracks, result)


class _MixedRealign(Protocol):
    """One window's haplotype-realign state PLUS the kernel that consumes it.

    Mixed variants+tracks needs two things per window: state computed once
    (per-row genotype views, deletion diffs) and a per-batch kernel that turns
    that state plus the window's intervals into `RaggedTracks`. Different
    variant sources need BOTH to differ -- SVAR1 and the record backends share
    the fused `intervals_and_realign_track_fused` kernel over a CSR
    (`_RealignWindow`), while SVAR2 has no fused kernel and must split into
    `intervals_to_tracks` + `shift_and_realign_tracks_from_svar2_readbound`
    (`_Svar2Realign` -- Track A, not yet landed). Bundling state with its
    kernel is what keeps the drive loop free of `isinstance` dispatch: the
    drive holds one optional and calls one method on it.
    """

    def realign_batch(
        self,
        lo: int,
        hi: int,
        itvs: "list[_ItvArrays]",
        names: list[str],
        insertion_fill: "dict[str, InsertionFill] | None",
        regions_batch: NDArray[np.int32],
        row_lengths: NDArray[np.int64],
        out_lengths: NDArray[np.int64],
        base_seed: int,
    ) -> "RaggedTracks":
        """Re-align window rows ``[lo, hi)``'s tracks to haplotype coordinates.

        Args:
            lo: First window row of this batch.
            hi: One past the last window row of this batch.
            itvs: One `_ItvArrays` per track, covering the WHOLE window's rows
                (already FFI-coerced once per window by `_coerce_window_itvs`).
            names: Track names in `itvs` order.
            insertion_fill: Per-track insertion-fill overrides, or `None`.
            regions_batch: `(hi-lo, 3)` int32 `(0, start, end)` rows.
            row_lengths: `(hi-lo,)` int64 reference-coordinate region lengths.
            out_lengths: `(hi-lo, ploidy)` int64 per-hap output length, taken
                from the batch's actual haplotype offsets.
            base_seed: `FlankSample` base seed.

        Returns:
            `RaggedTracks` of shape `(hi-lo, n_tracks, ploidy, None)`.
        """
        ...


@runtime_checkable
class _MixedTracksBackend(Protocol):
    """A stream backend that can serve mixed variants + tracks.

    The backend half of the `_MixedRealign` seam (issue #375): `_MixedRealign`
    is the per-window STATE plus the kernel that consumes it; this is the
    CAPABILITY a backend declares in order to produce that state in the first
    place. Declaring it as a `@runtime_checkable` Protocol -- rather than only
    the loose `supports_mixed_tracks` bool every backend already carries --
    makes the pairing machine-checkable: `to_iter`'s `isinstance(backend,
    _MixedTracksBackend)` narrows the backend union at the one call site that
    matters, so a backend that flips `supports_mixed_tracks = True` without a
    correctly-shaped `mixed_realign_window` (wrong return order, typo'd name,
    wrong parameter types) is caught there -- statically by `pyrefly`, or at
    worst by a clear `AssertionError` at the narrowing site -- instead of an
    `AttributeError` hundreds of lines away in the batch loop.
    """

    supports_mixed_tracks: ClassVar[bool]

    def mixed_realign_window(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        t_starts: NDArray[np.int32],
        t_ends: NDArray[np.int32],
        row_starts: NDArray[np.int32],
        row_ends: NDArray[np.int32],
    ) -> "tuple[_MixedRealign, NDArray[np.int32]]":
        """Per-window realign state + deletion-extended track ends.

        See `_Svar1Backend.mixed_realign_window` for the full contract every
        conforming backend must satisfy (index spaces, row ordering, shapes,
        and the exact 2-tuple return order).
        """
        ...


class _RealignWindow(NamedTuple):
    """CSR-shaped realign state: SVAR1 and the record backends (VCF/PGEN).

    Exists as its own optional rather than separately-nullable fields on
    :class:`_TrackWindow` so that "re-alignment is on" and "the arrays
    re-alignment needs are present" cannot disagree: there is one thing to
    check, and checking it hands you everything already narrowed.

    The last three fields are the per-variant tables the fused kernel reads.
    They are dataset-GLOBAL for `_Svar1Backend` (memmaps over the whole store)
    and window-LOCAL for the record backends (one decoded window's table), but
    the kernel only ever indexes them through `geno_v_idxs`, so the two cases
    are interchangeable here -- which is exactly why they live on the state
    rather than being read off the backend at the call site.
    """

    diffs: "NDArray[np.int32]"
    geno_offsets: "NDArray[np.int64]"
    geno_offset_idx: "NDArray[np.intp]"
    geno_v_idxs: "NDArray"
    v_starts: "NDArray[np.int32]"
    ilens: "NDArray[np.int32]"

    def realign_batch(
        self,
        lo: int,
        hi: int,
        itvs: "list[_ItvArrays]",
        names: list[str],
        insertion_fill: "dict[str, InsertionFill] | None",
        regions_batch: NDArray[np.int32],
        row_lengths: NDArray[np.int64],
        out_lengths: NDArray[np.int64],
        base_seed: int,
    ) -> "RaggedTracks":
        """See :meth:`_MixedRealign.realign_batch`. Fused-kernel implementation."""
        return _realigned_tracks_from_intervals(
            itvs,
            names,
            insertion_fill,
            np.arange(lo, hi, dtype=np.int64),
            regions_batch,
            self.geno_offset_idx[lo:hi],
            self.geno_offsets,
            self.geno_v_idxs,
            self.v_starts,
            self.ilens,
            row_lengths,
            self.diffs[lo:hi],
            out_lengths,
            base_seed,
        )


class _TrackWindow(NamedTuple):
    """One window's track state, computed once and consumed per batch.

    Bundled rather than left as eight loose locals because these values are
    produced under a ``tb is not None`` guard and consumed under a *second*
    one several hundred lines downstream. Two guards that must agree but
    cannot be seen together is exactly the shape a reader mis-reads and a type
    checker rejects; one optional binding narrows the whole set at once.
    """

    itvs: "list[_ItvArrays]"
    names: "list[str]"
    row_starts: "NDArray[np.int32]"
    row_ends: "NDArray[np.int32]"
    row_lengths: "NDArray[np.int64]"
    realign: "_MixedRealign | None"


def _realigned_tracks_from_intervals(
    per_track: "list[_ItvArrays]",
    names: list[str],
    insertion_fill: "dict[str, InsertionFill] | None",
    offset_idxs: NDArray[np.int64],
    regions_batch: NDArray[np.int32],
    geno_offset_idx: NDArray[np.int64],
    geno_offsets: NDArray[np.int64],
    geno_v_idxs: NDArray,
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    row_lengths: NDArray[np.int64],
    diffs: NDArray[np.int32],
    out_lengths: NDArray[np.int64],
    base_seed: int,
) -> "RaggedTracks":
    """Realign one batch's tracks to haplotype coordinates via the fused kernel.

    Mirrors `HapsTracks.__call__`'s per-track loop (`_reconstruct.py:168-260`)
    almost verbatim -- same fused kernel, same per-track buffer -- but does
    NOT copy its final assembly. That call site builds a TRACK-major buffer
    (`out[track_ofst*n_per_track:(track_ofst+1)*n_per_track]`) and labels it
    with offsets computed from `(b, t, p)`-shaped `out_lengths`
    (`repeat(out_lengths, "b p -> b t p")`), so the claimed layout and the
    physical layout only agree when `batch == 1` -- the only way the written
    path's `Dataset[r, s]` oracle ever reaches it (see `_tracks_from_intervals`,
    which fixes the same defect for the un-realigned case). That written-path
    defect is runtime-verified and filed as issue #371 -- batched
    `Dataset[r_arr, s_arr]` attaches tracks to the wrong rows whenever
    `n_tracks > 1`; do not "fix" streaming to match it. Streaming yields
    real batches, so this reorders the track-major buffer into `(b, t, p)`
    order before labelling it, generalizing `_tracks_from_intervals`'s 2-D
    `(t, b) -> (b, t)` transpose to the 3-D `(t, b, p) -> (b, t, p)` case.

    Args:
        per_track: One `_ItvArrays` per track (name-sorted; `names` gives the
            matching order), each covering the WHOLE window's `(row,)` cells.
            `offset_idxs` selects this batch's rows out of the window. Already
            FFI-coerced once per window by `_coerce_window_itvs`; do NOT
            re-coerce here (see `_ItvArrays`).
        names: Track names, same order as `per_track` -- used to resolve each
            track's `insertion_fill` strategy.
        insertion_fill: Per-track override of the default `Repeat5p()`
            insertion-fill strategy, or `None` (every track uses the default).
        offset_idxs: `(batch,)` int64 row indices into `per_track`, selecting
            this batch's cells out of the window (always per-cell for
            `tracks=`-constructed datasets -- the `TrackType.ANNOT` per-region
            branch is unreachable from this constructor).
        regions_batch: `(batch, 3)` int32 `(contig_idx, start, end)`, one row
            per (region, sample) pair in this batch. Only column 1 (start) is
            read by the Rust kernel.
        geno_offset_idx: `(batch, ploidy)` int64, WINDOW-LOCAL indices into
            `geno_offsets` -- unlike the written path's dataset-global
            `ravel_multi_index` (`_haps.py:776`), the window read backend hands
            back window-local CSR offsets, so no re-basing is needed beyond
            slicing to this batch's rows.
        geno_offsets: `(2, n_window_rows*ploidy)` int64 window CSR
            starts/stops (`_Svar1Backend.read_window`'s `(o_starts, o_stops)`,
            stacked) -- the WHOLE window's, indexed by `geno_offset_idx`.
        geno_v_idxs: The store's global `variant_idxs` memmap
            (`_Svar1Backend.geno_v_idxs`).
        v_starts: The store's global per-variant start positions.
        ilens: The store's global per-variant ILEN differences.
        row_lengths: `(batch,)` int64 region lengths (possibly
            jitter-translated), one per (region, sample) row -- the raw
            reference-coordinate track length before deletion extension.
        diffs: `(batch, ploidy)` int32 per-(row, hap) reference-length diffs
            from `get_diffs_sparse`, already sliced to this batch's rows.
        out_lengths: `(batch, ploidy)` int64 output length per (row, hap) --
            ragged (the real haplotype length) or a fixed `with_len(L)` value.
        base_seed: Base seed for the `FlankSample` insertion-fill strategy;
            unused by every other strategy.

    Returns:
        A `RaggedTracks` of shape `(batch, n_tracks, ploidy, None)`.
    """
    from .._ragged import RaggedTracks
    from .._threads import should_parallelize
    from .._utils import lengths_to_offsets
    from ..genvarloader import intervals_and_realign_track_fused
    from ._insertion_fill import Repeat5p
    from ._insertion_fill import lower as _lower_insertion_fills
    from ._svar2_haps import _ragged_arange_gather
    from ._utils import _ffi_array

    n_tracks = len(per_track)
    batch, ploidy = geno_offset_idx.shape
    fill_map = insertion_fill or {}

    track_lengths = row_lengths - diffs.clip(max=0).min(1)
    out_ofsts_per_t = np.ascontiguousarray(lengths_to_offsets(out_lengths), np.int64)
    track_ofsts_per_t = np.ascontiguousarray(
        lengths_to_offsets(track_lengths), np.int64
    )
    n_per_track = int(out_ofsts_per_t[-1])
    out = np.empty(n_tracks * n_per_track, np.float32)

    strat_list = [fill_map.get(name, Repeat5p()) for name in names]
    strat_ids, strat_params = _lower_insertion_fills(strat_list)

    geno_offset_idx = np.ascontiguousarray(geno_offset_idx, np.int64)
    geno_offsets = np.ascontiguousarray(geno_offsets, np.int64)
    # All-zero shifts is correct ONLY because streaming never sets
    # `deterministic=False`: the written path draws nonzero shifts exclusively
    # when a fixed output length is combined with non-deterministic sampling
    # (`_haps.py:740-742`), and `to_iter`'s docstring records `deterministic` as
    # having no observable effect here. Wiring `deterministic=False` without
    # also wiring these shifts would silently produce wrong bytes, so change
    # both together or neither.
    shifts = np.zeros((batch, ploidy), np.int32)
    offset_idxs = np.ascontiguousarray(offset_idxs, np.int64)

    for t, (name, itvs) in enumerate(zip(names, per_track)):
        _out = out[t * n_per_track : (t + 1) * n_per_track]
        intervals_and_realign_track_fused(
            out=_out,
            out_offsets=out_ofsts_per_t,
            regions=regions_batch,
            shifts=shifts,
            geno_offset_idx=geno_offset_idx,
            geno_v_idxs=_ffi_array(geno_v_idxs, np.int32, "geno_v_idxs"),
            geno_offsets=geno_offsets,
            v_starts=v_starts,
            ilens=ilens,
            offset_idxs=offset_idxs,
            # Already coerced ONCE per window by `_coerce_window_itvs` -- these
            # are window-scale buffers, and re-coercing here would repeat that
            # copy for every batch and every track. See `_ItvArrays` for why
            # the coercion is necessary and why it is not `_ffi_array`.
            itv_starts=itvs.starts,
            itv_ends=itvs.ends,
            itv_values=itvs.values,
            itv_offsets=itvs.offsets,
            track_offsets=track_ofsts_per_t,
            params=np.ascontiguousarray(strat_params[t], np.float64),
            strategy_id=int(strat_ids[t]),
            base_seed=int(base_seed),
            keep=None,
            keep_offsets=None,
            to_rc=None,
            parallel=should_parallelize(n_per_track * 4),
        )

    # `out` is TRACK-major: block `t` holds every (row, hap) cell of track
    # `t` in `(row, hap)` C-order, so its flat row order is `(t, row, hap)`.
    # A `(batch, n_tracks, ploidy, None)` Ragged indexes element
    # `row*n_tracks*ploidy + t*ploidy + hap`, i.e. `(row, t, hap)` order.
    # Reorder once, vectorized (see the docstring above).
    n_bp = batch * ploidy
    tm_offsets = lengths_to_offsets(np.tile(out_lengths.reshape(-1), n_tracks))
    perm = (
        np.arange(n_tracks * n_bp)
        .reshape(n_tracks, batch, ploidy)
        .transpose(1, 0, 2)
        .reshape(-1)
    )
    data, out_offsets = _ragged_arange_gather(out, tm_offsets, perm)

    result = Ragged.from_offsets(
        data,
        (batch, n_tracks, ploidy, None),  # type: ignore[bad-argument-type, no-matching-overload]
        out_offsets,
    )
    return cast(RaggedTracks, result)


@dataclass(frozen=True, slots=True)
class StreamingDataset:
    """Write-free, iterable-only dataset. Region-major iteration; no random access.

    Two ways to construct:

    - Public API: ``StreamingDataset(regions, reference=..., variants=<path>,
      tracks=...)``. At least one of ``variants`` or ``tracks`` is required.
      ``variants`` is classified by path suffix, mirroring :func:`gvl.write`'s
      classification (``_write.py``): a ``.svar`` directory (a genoray
      ``SparseVar``/SVAR1 store), a ``.svar2`` directory (a genoray
      ``SparseVar2``/SVAR2 store), a VCF/BCF, and a PGEN file-set are all
      supported. ``jitter>0`` (a per-region reproducible read-window
      translation) is supported with the default ``"engine"`` prefetch
      strategy -- see :meth:`to_iter`'s docstring for the rng contract.
      ``tracks`` (an :class:`~genvarloader._types.IntervalTrack`, e.g.
      :class:`~genvarloader.BigWigs` or :class:`~genvarloader.Table`, or a
      sequence of them) is optional and composes with ``variants``: supplied
      alone, the dataset is tracks-only (no ``reference``/``variants`` needed,
      no ploidy axis, and :meth:`with_seqs` raises); supplied alongside
      ``variants``, both sources are attached to the same dataset. When
      ``variants`` is omitted, ``contigs``/``samples`` are derived from the
      tracks' own intersection instead of a variant source (mirroring
      :func:`gvl.write`'s sample-intersection rule).
    - Internal/test-oriented: ``StreamingDataset(regions, contigs=..., n_samples=...,
      ploidy=..., _reconstruct_window=...)`` injects a reconstruction callback
      directly, bypassing variant-source classification. Used by
      ``test_streaming_scheduler.py`` and ``test_svar1_window.py``. ``samples`` may
      be supplied too; when omitted, placeholder names ``"0", "1", ...`` are used
      (this path never touches a real sample list).

    ``sample_idx`` means "index into :attr:`samples`" (lexicographically-sorted
    sample names, matching :func:`gvl.write`'s convention) -- NOT a variant store's
    native column order. See :attr:`samples`.

    Args:
        max_mem: Approximate byte budget for the read-window's offsets buffer, i.e.
            the ``o_start``/``o_stop`` CSR-index pair read per
            ``(region, sample, ploid)`` cell
            (``window_regions * window_samples * ploidy * 16`` bytes). Accepts an
            ``int`` (bytes) or a size string like ``"512MB"``, ``"1g"``, ``"2GiB"``
            (default ``"512MB"``). This budget only bounds the READ window -- the
            separate GENERATION granularity is ``batch_size`` (a :meth:`to_iter`
            argument), which bounds per-batch haplotype OUTPUT independently. Neither
            term scales with cohort size, so peak memory is bounded by
            ``max_mem`` (offsets) + ``batch_size`` (output), independent of the number
            of samples in the dataset.
        iteration_order: ``"auto"`` (default), ``"regions"``, or ``"samples"`` --
            see :meth:`to_iter` for the cartesian sweep this controls. It is a
            no-op whenever the sample axis fits in one read-window chunk. At the
            default ``max_mem="512MB"`` the threshold depends on how many tracks
            are attached, since each track adds 768 B per cell: about 8.4M
            samples with no tracks, ~340k with one track, ~170k with two (see
            ``TRACK_BYTES_PER_CELL``). Below the threshold the sample loop
            runs exactly once and both orders emit the identical plan. Check
            :attr:`iteration_order_is_active` to see whether this setting is
            actually doing anything for a given dataset/``max_mem`` combination.
    """

    # (n_regions, 4) sorted: (contig_idx, start, end, strand). Only cols 0-2 are
    # read here; `bed_to_regions` returns the 4th (strand) column.
    _regions: NDArray[np.int32]
    _sort_order: NDArray[np.intp]  # maps sorted position -> original bed row
    contigs: list[str]
    n_samples: int
    ploidy: int
    _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
    # Sample names in `sample_idx` order (lexicographically sorted -- see `samples`).
    _samples: list[str]
    # Read-window sizing, DERIVED from `max_mem` in __init__ (not user-set directly).
    # The window (regions x sample-chunk x ploidy) is the READ granularity; its offsets
    # buffer is what `max_mem` bounds. Per-batch generation (Task 3) bounds OUTPUT
    # separately by batch_size, so neither term scales with cohort size.
    #
    # 64 is a pragmatic REGION_TARGET, NOT a measured knee: a sweep (window_regions in
    # {1, 4, 16, 64, 256, 1024}) showed wall-clock improving monotonically with fewer
    # windows and flattening past ~64, with everything beyond that inside this shared
    # node's run-to-run noise. entries_touched was exactly flat across every setting,
    # confirming I/O is windowing-invariant, as designed. See
    # docs/roadmaps/streaming-dataset.md (Plan 2 Task 4) for the full sweep narrative.
    _window_regions: int = 64
    _window_samples: int = 1
    _max_mem_bytes: int = 512 * 1024 * 1024
    # Resolved iteration order; NEVER "auto" after __init__. See `_plan`.
    _iteration_order: str = "regions"
    # The split read/generate backend (real SVAR1 path). When set, _iter_batches
    # generates per batch (output bounded by batch_size). The injected
    # `_reconstruct_window` remains a whole-window TEST seam used when `_backend` is None.
    _backend: "_Svar1Backend | _Svar2Backend | _VcfBackend | _PgenBackend | None" = None
    # Interval-track read backend (issue #279 Task 5), set whenever `tracks=`
    # was supplied (variants-only, tracks-only, or mixed). `None` means no
    # tracks were requested.
    _track_backend: "_TrackBackend | None" = None
    # INTERNAL/EXPERIMENTAL (issue #283) -- not a public `__init__` kwarg, set only via
    # `object.__setattr__`. Selects which prefetch drive `_iter_batches` uses when
    # `_backend is not None`:
    #   - "engine" (default): the landed producer-thread `Svar1StreamEngine` (Design A).
    #   - "readahead": a single-thread read-ahead-one-window drive (Design C) that reuses
    #     the SAME `_Svar1Backend.read_window`/`generate_batch` calls the engine's
    #     consumer makes, just prefetching the next window's pages inline before
    #     generating the current one (no background thread). Output-identical to
    #     "engine" -- prefetch only warms pages, never changes what is generated.
    # This toggle exists ONLY for the cold-cache A-vs-C measurement
    # (`benchmarking/streaming/cold_cache_overlap.py`); it will be removed once a winner
    # is chosen (see docs/roadmaps/streaming-dataset.md).
    _prefetch_strategy: str = "engine"
    # --- Wave A (#277) output-mode config; defaults preserve pre-Wave-A behavior ---
    # Sequence output kind: RaggedSeqs (haplotypes) | RaggedAnnotatedHaps (annotated).
    _seq_kind: type = RaggedSeqs
    # Output length: "ragged" (per-hap actual length) or a fixed int >= 1.
    _output_length: "int | str" = "ragged"
    # Read-time jitter (0 = deterministic, byte-parity gate).
    _jitter: int = 0
    # rng seed/Generator for jitter + fixed-length shifts (matches Dataset.with_settings).
    _rng: "int | np.random.Generator | None" = None
    # Deterministic: disables random within-window shifts for fixed-length output.
    _deterministic: bool = True
    # Inclusive allele-frequency bounds for `with_seqs("variants")` (PR-B2, #317).
    _min_af: "float | None" = None
    _max_af: "float | None" = None
    # Requested variant fields for `with_seqs("variants")` output (PR-B3a, #304).
    # `None` means the default `["alt", "ilen", "start"]` -- see `active_var_fields`.
    _var_fields: "list[str] | None" = None
    # `with_seqs("variant-windows")` configuration (Wave B PR-B4, #304): `_var_window_opt`
    # is the caller's `VarWindowOpt` (only meaningful when `_seq_kind is dict`); the LUT
    # is derived from it ONCE in `with_seqs` (via `build_token_lut`, exactly like
    # `_impl.py`'s written-path `with_seqs`) and cached alongside it here so
    # `_iter_batches` never rebuilds it per `to_iter` call.
    _var_window_opt: "VarWindowOpt | None" = None
    _var_window_lut: "NDArray | None" = None
    _var_window_lut_dtype: "np.dtype | None" = None
    # Issue #279 Task 7: mixed SVAR1 variants + re-aligned tracks. `_realign_tracks`
    # mirrors `Dataset.with_settings(realign_tracks=...)` -- `True` (default)
    # re-aligns float tracks to haplotype coordinates via the fused kernel;
    # `False` returns reference-coordinate (as-is) tracks with no ploidy axis.
    # `_insertion_fill` mirrors `Dataset.with_insertion_fill`: `None` means
    # every track defaults to `Repeat5p()` (resolved per-track at read time,
    # matching the written path's `tracks.insertion_fill.get(name, Repeat5p())`).
    _realign_tracks: bool = True
    _insertion_fill: "dict[str, InsertionFill] | None" = None

    def __init__(
        self,
        regions,
        reference: str | Path | None = None,
        variants: str | Path | None = None,
        tracks: "IntervalTrack | Sequence[IntervalTrack] | None" = None,
        *,
        jitter: int = 0,
        max_mem: str | int = "512MB",
        contigs: list[str] | None = None,
        iteration_order: Literal["auto", "regions", "samples"] = "auto",
        n_samples: int | None = None,
        ploidy: int | None = None,
        samples: list[str] | None = None,
        _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
        | None = None,
    ):
        # Normalize `tracks` into a flat list ONCE, before the variants/tracks
        # classification below -- both the tracks-only branch and the
        # post-regions `_TrackBackend` construction (further down) consume this
        # list. A bare `IntervalTrack` (e.g. a single `BigWigs`) is wrapped;
        # anything else Sequence-like (list/tuple, NOT str/bytes) is flattened.
        if tracks is None:
            _track_list: "list[IntervalTrack]" = []
        elif isinstance(tracks, Sequence) and not isinstance(tracks, (str, bytes)):
            _track_list = list(tracks)
        else:
            _track_list = [tracks]

        # Every construction path must define this: the injected-callback (test) path
        # leaves it None; the real `.svar` branch below sets it to the `_Svar1Backend`
        # instance.
        _backend_obj = None
        if _reconstruct_window is not None:
            # Internal/test-oriented path: caller injects the reconstruction
            # callback directly and must supply everything it would otherwise
            # be derived from.
            if contigs is None or n_samples is None or ploidy is None:
                raise ValueError(
                    "StreamingDataset(_reconstruct_window=...) requires "
                    "`contigs`, `n_samples`, and `ploidy` to be supplied "
                    "explicitly."
                )
            # `samples` is optional here: this path is for scheduling/window-plan
            # tests that don't exercise real sample identity. Placeholder names
            # keep `.samples` well-defined without forcing every such test to
            # supply a real sample list.
            if samples is None:
                samples = [str(i) for i in range(n_samples)]
        elif variants is not None:
            # Public API path: classify `variants` and build the backend.
            if reference is None:
                raise ValueError(
                    "StreamingDataset(...) requires `reference` to reconstruct "
                    "haplotypes."
                )
            p = Path(variants)
            if p.is_dir() and p.suffix == ".svar":
                from genoray import SparseVar

                contigs = SparseVar(str(p)).contigs
                backend = _Svar1Backend(p, reference, contigs, regions)
                n_samples = backend.n_samples
                ploidy = backend.ploidy
                samples = backend._sample_names
                _reconstruct_window = None
                _backend_obj = backend
            elif p.is_dir() and p.suffix == ".svar2":
                from genoray import SparseVar2

                contigs = SparseVar2(str(p)).contigs
                backend = _Svar2Backend(p, reference, contigs, regions)
                n_samples = backend.n_samples
                ploidy = backend.ploidy
                samples = backend._sample_names
                _reconstruct_window = None
                _backend_obj = backend
            elif path_is_pgen(p):
                backend = _PgenBackend(p, reference, contigs, regions)
                n_samples = backend.n_samples
                ploidy = backend.ploidy
                samples = backend._sample_names
                contigs = backend._contigs
                _reconstruct_window = None
                _backend_obj = backend
                # No `_prefetch_strategy` override needed here: `__init__`
                # unconditionally sets it to "engine" below (same reasoning as
                # the VCF branch just below) -- PGEN, like VCF, has no
                # `read_window`/`generate_batch` split (SVAR1-only seam), so
                # only "engine" is supported.
            elif path_is_vcf(p):
                backend = _VcfBackend(p, reference, contigs, regions)
                n_samples = backend.n_samples
                ploidy = backend.ploidy
                samples = backend._sample_names
                contigs = backend._contigs
                _reconstruct_window = None
                _backend_obj = backend
                # No `_prefetch_strategy` override needed here: `__init__`
                # unconditionally sets it to "engine" below (the "readahead"
                # value is only ever flipped in post-construction by the
                # cold-cache A-vs-C harness, never chosen per-branch here).
                # VCF supports only "engine" -- `_VcfBackend` deliberately has
                # no `read_window`/`generate_batch` split (SVAR1-only seam).
            else:
                raise ValueError(
                    f"variants={p} has an unrecognized file type; expected a "
                    "VCF, PGEN, or SparseVar (.svar) store."
                )
        elif _track_list:
            # Tracks-only: no variant backend. `contigs`/`samples` are derived
            # from the tracks' intersection instead, mirroring `gvl.write`'s
            # sample-intersection rule (`_write.py:346-364`). There is no
            # ploidy axis for track-only output -- `ploidy` is set to a
            # placeholder that is never used to shape output (no haplotype
            # reconstruction happens on this path).
            if contigs is None:
                contigs = sorted(
                    set.intersection(*(set(t.contigs) for t in _track_list))
                )
                if not contigs:
                    raise ValueError(
                        "Tracks share no contigs; a tracks-only StreamingDataset"
                        " needs at least one contig common to every track."
                    )
            if samples is None:
                samples = sorted(
                    set.intersection(*(set(t.samples) for t in _track_list))
                )
                if not samples:
                    raise ValueError(
                        "Tracks share no samples; a tracks-only StreamingDataset"
                        " needs at least one sample common to every track."
                    )
            n_samples = len(samples)
            ploidy = 1  # placeholder; never used for output shaping
        else:
            raise ValueError(
                "StreamingDataset requires at least one source: `variants=`,"
                " `tracks=`, or the internal `_reconstruct_window=`."
            )

        bed = regions if isinstance(regions, pl.DataFrame) else sp.bed.read(regions)
        # record original-row order so emitted indices refer to the user's input order.
        # Positional (row-index carried through the sort), not value-based: a join on
        # BED columns would fan out on duplicate rows and corrupt `_sort_order`.
        sorted_bed = sp.bed.sort(bed.with_row_index("_r"))
        order = sorted_bed["_r"].to_numpy().astype(np.intp)
        regs = bed_to_regions(sorted_bed.drop("_r"), ContigNormalizer(contigs))
        object.__setattr__(self, "_regions", regs)
        object.__setattr__(self, "_sort_order", order)
        object.__setattr__(self, "contigs", list(contigs))
        object.__setattr__(self, "n_samples", int(n_samples))
        object.__setattr__(self, "ploidy", int(ploidy))
        object.__setattr__(self, "_reconstruct_window", _reconstruct_window)
        object.__setattr__(self, "_samples", list(samples))
        object.__setattr__(self, "_backend", _backend_obj)
        _track_backend_obj = None
        if _track_list:
            from ._track_stream import _TrackBackend

            _track_backend_obj = _TrackBackend(
                _track_list, self._regions, list(self.contigs), list(self._samples)
            )
        object.__setattr__(self, "_track_backend", _track_backend_obj)
        # See the field's comment: internal/experimental, flipped only by the
        # cold-cache A-vs-C harness via `object.__setattr__`. Backend-derived so a
        # non-SVAR1 backend (e.g. `_Svar2Backend`, whose producer-thread engine ships
        # gated off -- default `"sync"`) can select its own drive without a hardcoded
        # "engine" default.
        _strat = getattr(_backend_obj, "_default_strategy", "engine")
        object.__setattr__(self, "_prefetch_strategy", _strat)
        # Wave A (#277) output-mode fields: `slots=True` + a custom `__init__` means
        # the dataclass-generated defaults are never auto-applied (there's no class
        # attribute to fall back on at runtime), so every construction path -- public
        # and injected -- must set them explicitly here. Task 1 defaults exactly
        # preserve pre-Wave-A output.
        for _name in (
            "_seq_kind",
            "_output_length",
            "_jitter",
            "_rng",
            "_deterministic",
            "_min_af",
            "_max_af",
            "_var_fields",
            "_var_window_opt",
            "_var_window_lut",
            "_var_window_lut_dtype",
            "_realign_tracks",
            "_insertion_fill",
        ):
            object.__setattr__(
                self, _name, type(self).__dataclass_fields__[_name].default
            )
        # Wire the public `jitter=` kwarg AFTER the default-init loop above (else the
        # loop's dataclass-default (0) would overwrite it). Default `jitter=0` exactly
        # preserves pre-Task-3 behavior (no translation, byte-parity gate unaffected).
        if jitter < 0:
            raise ValueError(f"jitter must be non-negative, got {jitter}.")
        object.__setattr__(self, "_jitter", int(jitter))
        # Derive the read-window sizing from `max_mem`, NOT from the field defaults
        # above (those are just fallback literals for `__dataclass_fields__`; `slots=True`
        # means there's no class attribute to fall back on at runtime, and this class
        # defines its own `__init__` so the dataclass-generated one never runs). The
        # offsets buffer is `window_regions * window_samples * ploidy * 16 B`
        # (o_start + o_stop, i64 each); bound it by `max_mem` so peak memory stays
        # independent of cohort size, keeping whole sample sets when they fit and
        # holding regions at the measured amortization knee (REGION_TARGET=64).
        max_mem_bytes = _parse_max_mem(max_mem)
        # The engine (#283) double-buffers windows -- the producer reads the NEXT
        # window's offsets while the consumer generates batches from the CURRENT one --
        # so two windows' offsets can be resident at once; size the budget for that.
        n_slots = 2
        # o_start + o_stop, i64 each, per (region,sample,ploid) for SVAR1; a backend
        # may declare a different per-cell resident-window cost (e.g. `_Svar2Backend`
        # caches vk_snp_range + vk_indel_range pairs instead), so defer to it when set.
        cell_bytes = getattr(_backend_obj, "_cell_bytes", int(ploidy) * 16)
        if _track_backend_obj is not None:
            # Each (region, sample) cell also materializes intervals for every
            # track: start+end (i32) + value (f32) = 12 B per interval. Budget a
            # conservative constant per cell per track until a measured
            # estimate is available (follow-up) -- this is what lets
            # `_window_samples` actually shrink under a tight `max_mem` when
            # tracks are present (spec Sections 2.5, 4.4).
            TRACK_BYTES_PER_CELL = 12 * 64
            cell_bytes += TRACK_BYTES_PER_CELL * len(_track_backend_obj.names)
        max_cells = max(1, max_mem_bytes // (cell_bytes * n_slots))
        window_samples = max(1, min(int(n_samples), max_cells))
        region_target = 64  # measured read-amortization knee; see roadmap Plan 2.
        window_regions = max(1, min(region_target, max_cells // window_samples))
        object.__setattr__(self, "_max_mem_bytes", max_mem_bytes)
        object.__setattr__(self, "_window_samples", int(window_samples))
        object.__setattr__(self, "_window_regions", int(window_regions))
        # Refine the SVAR2 backend's super-batch sizing now that `max_mem_bytes` is
        # known -- the backend was built above (before this point) with a
        # self-contained default (`SUPERBATCH_TARGET_ROWS`), since it can't see
        # `max_mem` at construction time. `_Svar2Backend` is a plain (non-frozen)
        # class, so a direct attribute assignment is fine.
        if isinstance(_backend_obj, _Svar2Backend):
            widths = _backend_obj._regions[:, 2] - _backend_obj._regions[:, 1]
            mean_region_width = int(max(1, widths.mean())) if len(widths) else 1
            bytes_per_row = _backend_obj.ploidy * mean_region_width
            _backend_obj._super_batch_rows = max(
                1, min(SUPERBATCH_TARGET_ROWS, max_mem_bytes // max(1, bytes_per_row))
            )

        if iteration_order not in ("auto", "regions", "samples"):
            raise ValueError(
                "iteration_order must be one of 'auto', 'regions', 'samples';"
                f" got {iteration_order!r}."
            )
        if iteration_order == "auto":
            # Keyed off the SOURCE MIX, never per-class micro-properties.
            # Tracks-only -> samples, because BigWigs holds one file per sample.
            # Mixed -> regions, the roadmap's decision (non-optimal for the
            # track axis; `iteration_order="samples"` is the escape hatch).
            has_tracks = _track_backend_obj is not None
            has_variants = _backend_obj is not None
            resolved = "samples" if (has_tracks and not has_variants) else "regions"
        else:
            resolved = iteration_order
        object.__setattr__(self, "_iteration_order", resolved)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._regions), self.n_samples)

    @property
    def iteration_order_is_active(self) -> bool:
        """Whether `iteration_order` actually changes the visit order.

        `iteration_order` only matters when the sample axis is chunked. The
        chunk size is derived from `max_mem`; at the default `max_mem="512MB"`
        the cohort size below which it holds the entire sample axis depends on
        track count (768 B/cell/track) -- about 8.4M with no tracks, ~340k with
        one, ~170k with two -- below which the sample loop runs once and both
        orders emit the identical plan.

        Returns:
            `True` when `_window_samples < n_samples`, so the two orders differ.
        """
        return self._window_samples < self.shape[1]

    @property
    def samples(self) -> list[str]:
        """The samples in the dataset, in ``sample_idx`` order.

        Lexicographically sorted (matching :func:`gvl.write`'s convention and
        :attr:`Dataset.samples <genvarloader.Dataset.samples>`) -- **not** the
        variant store's native (e.g. VCF column) order. ``to_iter``'s
        ``sample_idxs`` index into this list: ``samples[i]`` is the sample whose
        data arrives at ``sample_idx == i``.
        """
        return list(self._samples)

    @property
    def available_var_fields(self) -> list[str]:
        """Variant fields this source can serve, in a stable order.

        Derived from the LIVE source (SVAR1's on-disk index schema, or the VCF/BCF
        header) rather than a written artifact, so it can differ per backend --
        unlike the written :attr:`Dataset.available_var_fields
        <genvarloader.Dataset.available_var_fields>`, which reads an already-shipped
        ``variants.arrow``. The injected-callback (test) construction path has no
        backend and always returns the builtin default.

        Returns:
            Field names requestable via :meth:`with_settings`'s ``var_fields``.
        """
        if self._backend is None:
            return list(_DEFAULT_VAR_FIELDS)
        return list(self._backend.available_var_fields)

    @property
    def servable_var_fields(self) -> list[str]:
        """Variant fields this source's streaming engine can actually gather today.

        A subset of :attr:`available_var_fields` (Wave B PR-B3a review, Important 1):
        advertising a field and being able to serve it are separate, static,
        per-backend facts -- e.g. an AF-cached SVAR1 store advertises ``"AF"`` via
        :attr:`available_var_fields` (it's a real on-disk numeric index column) but
        the SVAR1 Rust engine has no general INFO/index-column gather yet (only
        ``ref`` is wired beyond the three defaults), so ``"AF"`` is available but not
        servable. :meth:`with_settings` checks this eagerly so an unservable request
        fails at configuration time, not after a full ``build_engine`` + first
        window read.

        Returns:
            Field names :meth:`with_settings`'s ``var_fields`` can request without
            raising :class:`NotImplementedError` at iterate time.
        """
        if self._backend is None:
            return list(_DEFAULT_VAR_FIELDS)
        return list(self._backend.servable_var_fields)

    @property
    def active_var_fields(self) -> list[str]:
        """The variant fields currently selected for ``with_seqs("variants")`` output.

        Returns:
            The configured ``var_fields``, or the default ``["alt", "ilen", "start"]``
            when none was set via :meth:`with_settings`.
        """
        return _normalize_var_fields(self._var_fields)

    def __len__(self) -> int:
        return len(self._regions) * self.n_samples

    def _plan(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield one WINDOW per step: `(region_idxs, sample_chunk)`.

        Cartesian and single-contig. The contig-run loop is outermost in BOTH
        iteration orders: it is required by the single-contig Rust invariant,
        and it is what lets `RustTable`'s per-contig tree cache hold.

        `iteration_order` only changes the VISIT ORDER, never the window set.
        It is also a no-op whenever `_window_samples == n_samples` (the
        default at `max_mem="512MB"` for any cohort under ~8.4M with no
        tracks, ~340k with one track, or ~170k with two -- each track adds
        768 B/cell, see `TRACK_BYTES_PER_CELL`), because the sample loop
        then runs exactly once.
        """
        n_regions, n_samples = self.shape
        if n_regions == 0:
            return
        contig_idxs = self._regions[:, 0]
        run_bounds = np.flatnonzero(np.diff(contig_idxs)) + 1
        run_starts = np.concatenate(([0], run_bounds))
        run_ends = np.concatenate((run_bounds, [n_regions]))
        sample_major = self._iteration_order == "samples"
        for r_lo, r_hi in zip(run_starts, run_ends):
            # Materialize each region window's index array ONCE and re-yield the
            # same object per sample chunk. `_iter_batches` stores these in
            # `plan_jobs`, so allocating inside the inner loop would multiply
            # job residency by n_sample_chunks.
            windows = [
                np.arange(
                    w_lo, min(w_lo + self._window_regions, int(r_hi)), dtype=np.intp
                )
                for w_lo in range(int(r_lo), int(r_hi), self._window_regions)
            ]
            chunks = [
                np.arange(
                    s_lo, min(s_lo + self._window_samples, n_samples), dtype=np.intp
                )
                for s_lo in range(0, n_samples, self._window_samples)
            ]
            if sample_major:
                for s in chunks:
                    for w in windows:
                        yield w, s
            else:
                for w in windows:
                    for s in chunks:
                        yield w, s

    def _rng_gen(self) -> np.random.Generator:
        """Build the one `Generator` for this `to_iter` call's jitter draws.

        `_iter_batches` creates exactly one of these per call (never per-window) so
        the per-region draw sequence is deterministic in sweep order -- see
        `to_iter`'s docstring for the full rng contract.
        """
        return np.random.default_rng(self._rng)

    def _region_jitter_offsets(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Draw ONE jitter offset per region, ONCE per `to_iter` call.

        Offsets are indexed by absolute region index into `self._regions`, which is
        pre-sorted by `(contig, start)`. They are drawn before `_plan()`'s
        per-window/per-sample-chunk loop, not once per plan job. `_plan()` re-yields
        the same region-window `r_idx`
        once per sample chunk whenever `n_samples > _window_samples` (cohort
        scale); drawing offsets here and looking them up by region index (see
        `_jitter_region_bounds`) means every sample chunk of the same region gets
        the SAME offset, independent of `_window_samples`/`_window_regions`
        chunking. Drawing all `len(self._regions)` offsets in one vectorized call
        also visits them in ascending region-index order, which IS sweep order
        (regions are pre-sorted and `_plan` sweeps ascending indices) -- see
        `to_iter`'s docstring for the full rng contract.
        """
        return rng.integers(-self._jitter, self._jitter + 1, size=len(self._regions))

    def _jitter_region_bounds(
        self,
        region_offsets: NDArray[np.int64],
        r_idx: NDArray[np.intp],
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
        """Translate `r_idx`'s region bounds by its precomputed jitter offset.

        Uses the per-region `region_offsets` (see `_region_jitter_offsets`; indexed
        by absolute region
        index, so the same region always gets the same offset regardless of which
        window/sample-chunk it's visited from). The window SIZE is preserved
        (translate, don't resize) -- a fixed `output_length` that fit the base
        region still fits the jittered one. This is a pure Python-side
        region-bounds transform; no Rust change. See `to_iter`'s docstring for the
        rng contract (translation-only; NOT byte-parity).

        Only the LOWER bound is clamped (`start + offset >= 0`): region bounds
        cross into Rust as `u32`, so a negative start would silently wrap to a
        huge unsigned value -- that must never happen. An upper bound (`end <=
        contig_len`) is deliberately NOT enforced: the shared reconstruction
        kernel (`reconstruct_haplotype_core`, `src/reconstruct/mod.rs`) already
        tail-pads with `pad_char` whenever a region's end runs past the contig
        (the same N-padding behavior `gvl.Dataset` gives any region that
        overhangs a chromosome edge), so an end beyond `contig_len` is safe, not
        a bug. Clamping the upper bound too would make jitter a deterministic
        no-op for any region that already spans its full contig (exactly the
        single-contig VCF/PGEN streaming test fixtures) -- mathematically, a
        window that already covers `[0, contig_len)` cannot be translated at all
        under a strict two-sided in-bounds clamp, since ANY nonzero offset would
        push one side out of range.
        """
        starts = self._regions[r_idx, 1].astype(np.int64)
        ends = self._regions[r_idx, 2].astype(np.int64)
        jitter_off = region_offsets[r_idx]
        jitter_off = np.maximum(jitter_off, -starts)
        return (
            np.ascontiguousarray(starts + jitter_off, np.uint32),
            np.ascontiguousarray(ends + jitter_off, np.uint32),
        )

    def _iter_batches(self, batch_size: int) -> Iterator[tuple]:
        """Drive the plan; generate each window PER BATCH so output is batch-bounded.

        The window is the READ granularity; a batch is the GENERATION granularity. For
        the real SVAR1 backend this drives a `Svar1StreamEngine` (#283) that overlaps
        producer I/O (reading the next window) with consumer generation (reconstructing
        the current one) -- output is still batch_size-bounded (issue #284). The
        injected `_reconstruct_window` path (tests) reconstructs the whole window and
        slices -- memory-unbounded, but only ever used with tiny fixtures.
        """
        if self._backend is None and self._track_backend is not None:
            # Tracks-only: no variant engine, so drive the plan directly. The
            # window is the read granularity; batches slice it.
            if self._jitter > 0:
                # `read_window`'s contract is that jittered reads must be given
                # the SAME translated bounds the variant engine got. There is no
                # variant engine here to derive them from, so fail fast rather
                # than silently reading unjittered bounds -- the same choice the
                # SVAR2 and non-"engine"-prefetch guards make in this file.
                raise NotImplementedError(
                    "Read-time jitter (jitter>0) is not yet supported for"
                    " tracks-only StreamingDatasets: there is no variant engine"
                    " to derive translated region bounds from. Use jitter=0, or"
                    " supply `variants=` as well."
                )
            tb = self._track_backend
            for r_idx, s_idx in self._plan():
                per_track = tb.read_window(r_idx, s_idx)
                n_s = len(s_idx)
                flat_r = np.repeat(self._sort_order[r_idx], n_s)
                flat_s = np.tile(np.asarray(s_idx, np.intp), len(r_idx))
                n_rows = len(flat_r)
                starts = self._regions[r_idx, 1]
                ends = self._regions[r_idx, 2]
                flat_starts = np.repeat(starts, n_s).astype(np.int32)
                lengths = np.repeat(ends - starts, n_s).astype(np.int64)
                # Flatten (region, sample) -> row ONCE per window, so each
                # batch is a contiguous row slice and offset_idxs is a plain
                # arange. `output_length` is applied here when it is an int.
                if isinstance(self._output_length, int):
                    lengths = np.full(len(lengths), self._output_length, np.int64)
                # Flatten AND FFI-coerce ONCE per window, then select each
                # batch's rows with `offset_idxs` rather than re-slicing the
                # intervals. This is the mechanism the written path uses
                # (`_tracks.py`'s `o_idx` indexes the un-sliced array) and it
                # keeps the per-batch work to one kernel call per track over
                # buffers that are already contiguous (see `_ItvArrays`).
                flat_itvs = _coerce_window_itvs(per_track)
                for lo, hi in _batch_bounds(0, n_rows, batch_size):
                    out = _tracks_from_intervals(
                        flat_itvs,
                        np.arange(lo, hi, dtype=np.int64),
                        flat_starts[lo:hi],
                        lengths[lo:hi],
                    )
                    yield out, flat_r[lo:hi], flat_s[lo:hi]
            return

        if self._backend is not None:
            # Resolve the effective fixed/ragged length ONCE: -1 = ragged (per-hap
            # actual length, the pre-Wave-A default), >=0 = fixed (issue #277
            # `with_len(L)`). Threaded through both prefetch-strategy branches below;
            # the Ragged output shaping further down is UNCHANGED -- offsets from the
            # engine/generate_batch already encode the fixed length when >=0.
            _out_len = (
                -1 if self._output_length == "ragged" else int(self._output_length)
            )
            # Wave A (#277) Task 4: whether to request AnnotatedHaps (var_idxs +
            # ref_coords) alongside haplotypes. Resolved ONCE here (not per window) and
            # threaded through `build_engine`/the engine constructor -- the engine only
            # allocates/computes the two annotation arrays when this is `True`.
            _annotated = self._seq_kind is RaggedAnnotatedHaps
            # Wave B PR-B1 (#304): whether to request flat variant buffers instead of
            # reconstructed haplotype bytes. Resolved ONCE here, mirroring `_annotated`
            # -- threaded through `build_engine`/the engine constructor and selects the
            # `next_batch_variants` puller below.
            _variants = self._seq_kind is RaggedVariants
            # Wave B PR-B4 (#304): whether to request tokenized variant-window
            # buffers instead of reconstructed haplotype bytes. Resolved ONCE here,
            # mirroring `_variants` -- threaded through `build_engine`/the engine
            # constructor and selects the `next_batch_variant_windows` puller below.
            _variant_windows = self._seq_kind is dict
            # Issue #375: mixed variants + tracks is a per-backend CAPABILITY,
            # not an isinstance test. A backend opts in by setting
            # `supports_mixed_tracks = True` and defining
            # `mixed_realign_window`; the two must be added together. Checked
            # BEFORE any engine/plan work so a doomed combination fails fast,
            # matching the SVAR2 jitter/out_len guard just below.
            if (
                self._track_backend is not None
                and not self._backend.supports_mixed_tracks
            ):
                raise NotImplementedError(
                    "StreamingDataset tracks= combined with a variant source is "
                    f"not supported for {type(self._backend).__name__} yet; only "
                    "the SVAR1 (.svar) backend supports mixed variants+tracks "
                    "today (issue #375)."
                )
            # Task 8 (spec §3.3/§8): `with_seqs("variant-windows")` + tracks +
            # `realign_tracks=True` mirrors the WRITTEN path's own `ValueError`
            # verbatim (`_reconstruct.py:537-543`) -- windows are
            # reference-oriented and the written path itself refuses to
            # re-align them, so this is a real semantic rejection, not a
            # streaming gap. Exception type matters here: `NotImplementedError`
            # is not a `ValueError` subclass, so a caller mirroring the written
            # path's error handling would not catch it.
            if (
                self._track_backend is not None
                and _variant_windows
                and self._realign_tracks
            ):
                raise ValueError(
                    "with_seqs('variant-windows') with tracks requires"
                    " with_settings(realign_tracks=False) (windows are"
                    " reference-oriented; re-alignment is not supported)."
                )
            # Deviation from the spec §3.3 table's row 2 -- recorded, not
            # silently dropped: the table also claims `with_seqs("variants")` +
            # tracks + `realign_tracks=True` raises the written path's
            # `ValueError`, "same message shape" as variant-windows. Verified
            # against the actual written path (`_build_reconstructor`,
            # `_reconstruct.py:514-547`, and confirmed empirically): NEITHER
            # `with_seqs("variants")` nor `with_seqs("annotated")` has any
            # guard there for ANY `realign_tracks` value -- both dispatch to
            # `HapsTracks`/`SeqsTracks` and are fully SUPPORTED by
            # `Dataset[r, s]` today (`with_insertion_fill`'s own allow-list at
            # `_impl.py:872` includes "variants" for the same reason). So there
            # is no written-path `ValueError` to mirror for either kind; both
            # are purely a streaming wiring gap (the fused kernel below only
            # assembles bare haplotype bytes) and get `NotImplementedError`
            # uniformly, independent of `realign_tracks`.
            if self._track_backend is not None and (_variants or _variant_windows):
                raise NotImplementedError(
                    "StreamingDataset tracks= combined with "
                    f"with_seqs({_SEQ_KIND_NAMES.get(self._seq_kind, self._seq_kind)!r}) "
                    "is not yet wired for streaming (the written Dataset DOES "
                    "support this combination); use with_seqs('haplotypes') "
                    "(the default) with tracks=."
                )
            if self._track_backend is not None and _annotated:
                raise NotImplementedError(
                    "StreamingDataset tracks= combined with "
                    "with_seqs('annotated') is not yet wired for streaming (the "
                    "written Dataset DOES support this combination); use "
                    "with_seqs('haplotypes') (the default) with tracks=."
                )
            # Deliberate seam, NOT covered by any required parity test (the
            # fixture is built with `max_jitter=None`): `_Svar1Backend.read_window`
            # (used by `mixed_realign_window`, per-window, to size the
            # deletion-extended track query) always queries the RAW unjittered
            # `_regions` bounds, while the
            # haplotype engine and the track query itself use the jitter-translated
            # bounds (`_jitter_region_bounds`). Under jitter>0 this can miss a
            # boundary deletion and under-extend the track query. Fail fast rather
            # than silently under-size the track buffer, matching the tracks-only
            # branch's identical jitter guard just above in this file.
            if (
                self._track_backend is not None
                and self._jitter > 0
                and self._realign_tracks
            ):
                raise NotImplementedError(
                    "StreamingDataset read-time jitter (jitter>0) is not yet "
                    "supported together with tracks= re-alignment (realign_tracks="
                    "True, the default): the deletion-extension query would need "
                    "jitter-translated region bounds too. Use jitter=0, or "
                    "with_settings(realign_tracks=False) (which skips the "
                    "deletion-extension query entirely)."
                )
            # Minor (Wave B PR-B3a review): resolve ONCE per `_iter_batches` call, not
            # once per batch inside the packing loop below -- `active_var_fields`
            # allocates a new list every call.
            _active_var_fields = self.active_var_fields if _variants else []
            # Wave B PR-B2 (#317): min_af/max_af filtering is only implemented for
            # `with_seqs("variants")` output -- mirroring the written `Dataset`, which
            # raises for haplotype/annotated output when AF bounds are requested (AF
            # filtering drops whole variants, which haplotype/annotated reconstruction
            # has no way to represent). Fail fast rather than silently ignore the bounds.
            _af_filter = self._min_af is not None or self._max_af is not None
            if _af_filter and not _variants:
                raise NotImplementedError(
                    'min_af/max_af filtering is only supported for with_seqs("variants") '
                    "output (matching the written Dataset, which raises for "
                    "haplotype/annotated output)."
                )
            # Wave B PR-B3a (#304): `var_fields` only shapes `with_seqs("variants")`
            # output. The written path silently ignores it elsewhere; streaming fails
            # fast instead, matching how it treats every other ignorable setting (the
            # AF guard just above, and the jitter/out_len/annotated SVAR2 guard below).
            if self._var_fields is not None and not _variants:
                raise NotImplementedError(
                    'var_fields only applies to with_seqs("variants") output; got '
                    f"with_seqs({_SEQ_KIND_NAMES.get(self._seq_kind, self._seq_kind)!r})."
                )
            # Wave A output-mode knobs (issue #277) are wired only through the
            # SVAR1/VCF/PGEN engines. The SVAR2 drives ("sync"/"svar2_engine") read
            # unjittered region bounds and emit ragged haplotypes only, so combining
            # them with jitter, `with_len`, `with_seqs("annotated")`, or
            # `with_seqs("variants")` would silently ignore the request. Fail fast
            # rather than return wrong output; SVAR2 Wave A/B support is a follow-up.
            if isinstance(self._backend, _Svar2Backend) and (
                self._jitter > 0 or _out_len != -1 or _annotated or _variants
            ):
                raise NotImplementedError(
                    "StreamingDataset read-time jitter (jitter>0), with_len (a fixed "
                    'output length), with_seqs("annotated"), and with_seqs("variants") '
                    "are not yet supported for the SVAR2 (.svar2) backend; they are "
                    "wired only through the SVAR1/VCF/PGEN engines (issues #277, #304). "
                    "Use ragged haplotype output with jitter=0 for .svar2 sources."
                )
            # `with_seqs("annotated")` emits per-position dataset-GLOBAL variant ids
            # (`AnnotatedHaps.var_idxs`). SVAR1 carries these for free and PGEN derives
            # them from the `.pvar` absolute row index, but a VCF source has no cheap
            # per-record global id: genoray leaves `RawRecord.global_idx = -1` for VCF
            # (`vcf_reader.rs`), so the local->global gather is skipped and the emitted
            # ids are silently WRONG for any window that is multi-contig, mid-contig
            # (narrowed), or drops an interior variant behind a spanning deletion --
            # i.e. essentially every real training window (issues #305, #311). Rather
            # than return silently-wrong ids, fail fast: annotated output requires a
            # PGEN or SVAR source. Populating VCF global ids needs a one-time full-source
            # record scan and is deferred until there is a concrete use case (#305).
            if isinstance(self._backend, _VcfBackend) and _annotated:
                raise NotImplementedError(
                    'with_seqs("annotated") is not supported for the VCF backend: '
                    "AnnotatedHaps.var_idxs are dataset-global variant ids, which a VCF "
                    "source cannot produce without an expensive full-file scan (genoray "
                    "leaves them unset, so they would be silently wrong for multi-contig, "
                    "narrowed, or interior-exclusion windows; issues #305, #311). Use a "
                    "PGEN (.pgen) or SVAR (.svar/.svar2) source for annotated output, or "
                    'plain with_seqs("haplotypes") for the VCF backend.'
                )
            if self._prefetch_strategy == "engine":
                # "engine" drives the record-style backends (SVAR1, VCF, PGEN), which
                # share the same `build_engine(jobs, batch_size, out_len, annotated,
                # variants)` producer/consumer interface. `_Svar2Backend` is NOT one of
                # them --
                # its `_default_strategy` is "sync" and its `build_engine` is
                # differently shaped -- so narrow the union to exclude it (and `None`),
                # letting the calls below resolve against the record backends'
                # signatures. See the `_backend` field's comment.
                assert isinstance(
                    self._backend, (_Svar1Backend, _VcfBackend, _PgenBackend)
                ), (
                    '"engine" prefetch strategy requires a record-style backend (SVAR1/VCF/PGEN)'
                )
                backend = self._backend
                # Wave B PR-B2 (#317/#319): AF filtering needs cached per-variant AF --
                # `SparseVar.cache_afs()` for SVAR1, an INFO/AF header field for VCF, never
                # for PGEN (no INFO path). Fail fast with the SAME message the written
                # `Dataset` path raises (`_haps.py`), so streaming and written agree.
                if _af_filter and not backend.has_cached_af:
                    raise RuntimeError(
                        "Either this dataset is not backed by an SVAR file, or the SVAR file has not had AFs cached yet."
                        + "Doing this automatically is not yet supported."
                    )
                # Build a COMPACT, region-scale plan ONCE and drive off THAT (never a
                # second `list(self._plan())`). `_plan` always yields a CONTIGUOUS sample
                # chunk `arange(s_lo, s_hi)`, so each window is captured losslessly by
                # `(contig_idx, r_idx, s_lo, s_hi)` -- r_idx is `window_regions`-scale and
                # (s_lo, s_hi) is two ints. Total residency is O(n_windows x window_regions)
                # (region-scale), NEVER O(n_windows x n_samples): the engine holds the
                # public->physical map ONCE and its producer slices it per window (issue
                # #284 / final-review Finding 1).
                plan_jobs: list[tuple[int, NDArray[np.intp], int, int]] = []
                for r_idx, s_idx in self._plan():
                    contig_idx = int(self._regions[r_idx[0], 0])
                    plan_jobs.append(
                        (contig_idx, r_idx, int(s_idx[0]), int(s_idx[-1]) + 1)
                    )
                # Region bounds (u32) per window for the engine constructor; transient
                # (dropped after `build_engine`), also region-scale. When `_jitter>0`,
                # ONE `Generator` is created here (outside this loop) and the
                # per-region offsets are drawn ONCE, indexed by absolute region index
                # (see `_region_jitter_offsets`) -- NOT once per `plan_jobs` entry.
                # `_plan()` re-yields the same region-window `r_idx` once per sample
                # chunk whenever `n_samples > _window_samples`, so drawing per
                # `plan_jobs` entry would give the same region a DIFFERENT offset per
                # sample chunk; indexing a precomputed per-region array instead means
                # every sample chunk of a region shares its offset, independent of
                # `_window_samples`/`_window_regions` chunking (issue #277 review
                # finding). `_jitter==0` (the default) takes the untranslated path
                # unchanged, preserving the jitter=0 byte-parity gate exactly.
                # `region_offsets` is None exactly when jitter is off. Narrowing
                # on the array rather than re-testing `self._jitter > 0` at each use
                # keeps the flag and the data it implies from disagreeing, and lets
                # the producer here and the track consumer below share one guard.
                region_offsets = (
                    self._region_jitter_offsets(self._rng_gen())
                    if self._jitter > 0
                    else None
                )
                if region_offsets is not None:
                    engine_jobs = [
                        (
                            contig_idx,
                            *self._jitter_region_bounds(region_offsets, r_idx),
                            s_lo,
                            s_hi,
                        )
                        for (contig_idx, r_idx, s_lo, s_hi) in plan_jobs
                    ]
                else:
                    engine_jobs = [
                        (
                            contig_idx,
                            np.ascontiguousarray(self._regions[r_idx, 1], np.uint32),
                            np.ascontiguousarray(self._regions[r_idx, 2], np.uint32),
                            s_lo,
                            s_hi,
                        )
                        for (contig_idx, r_idx, s_lo, s_hi) in plan_jobs
                    ]
                engine = backend.build_engine(
                    engine_jobs,
                    batch_size,
                    _out_len,
                    _annotated,
                    _variants,
                    min_af=self._min_af,
                    max_af=self._max_af,
                    var_fields=self._var_fields,
                    # Wave B PR-B4 (#304): `(lut, lut_dtype, opt)` cached by `with_seqs`
                    # -- `None` unless `_variant_windows`, matching how `min_af`/
                    # `max_af`/`var_fields` stay at their no-op defaults for every
                    # other output kind.
                    var_window=(
                        (
                            self._var_window_lut,
                            self._var_window_lut_dtype,
                            self._var_window_opt,
                        )
                        if _variant_windows
                        else None
                    ),
                )
                del engine_jobs
                # Issue #277 Wave A Task 4: annotated output pulls the 4-tuple
                # `(data, annot_v_idxs, annot_ref_pos, offsets)` from the engine and
                # packs a `RaggedAnnotatedHaps`; haplotype output is unchanged (2-tuple,
                # `RaggedSeqs`-shaped). Both share the SAME `offsets` layout (every
                # array is one entry per output position), so all three `Ragged`s below
                # are built from one `Ragged.from_offsets` call each over that offsets
                # array -- mirrors the written path's `_FlatAnnotatedHaps.to_padded()`
                # packing (`_flat.py`). Wave B PR-B1 (#304): variants output pulls a
                # `dict` (keys `alt`/`alt_offsets`/`start`/`ilen`/`offsets`) instead of
                # a tuple and packs a `RaggedVariants` -- mirrors `_FlatAlleles.to_ragged`
                # (`_flat_variants.py`) for the ragged `alt` field and `_Flat.to_ragged`
                # for the scalar `start`/`ilen` fields. Wave B PR-B4 (#304):
                # variant-windows output also pulls a `dict` (keys `start`/`ilen`/
                # `offsets` plus `<name>`/`<name>_offsets` per emitted token buffer)
                # and packs a plain `dict[str, Ragged]` -- mirrors
                # `_FlatWindow.to_ragged` (two ragged axes) and `_Flat.to_ragged` (one),
                # `_flat_variants.py`.
                if _variants:
                    next_batch = engine.next_batch_variants
                elif _variant_windows:
                    next_batch = engine.next_batch_variant_windows
                elif _annotated:
                    next_batch = engine.next_batch_annotated
                else:
                    next_batch = engine.next_batch
                for _contig_idx, r_idx, s_lo, s_hi in plan_jobs:
                    n_s = s_hi - s_lo
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(np.arange(s_lo, s_hi, dtype=np.intp), len(r_idx))
                    n_rows = len(flat_r)
                    # Issue #279 Task 7: read this window's tracks once, here --
                    # the composition point the per-window loop already has, no
                    # separate cursor needed. Guarded on `tb is not None` so a
                    # haplotype-only stream pays nothing extra.
                    tb = self._track_backend
                    if tb is not None:
                        s_idx_w = np.arange(s_lo, s_hi, dtype=np.intp)
                        if region_offsets is not None:
                            # Tracks MUST use the SAME translated bounds the
                            # engine got (`region_offsets`, drawn once above,
                            # before `plan_jobs` was built), or tracks and
                            # haplotypes silently disagree by up to `jitter`
                            # bases.
                            t_starts, t_ends = self._jitter_region_bounds(
                                region_offsets, r_idx
                            )
                        else:
                            t_starts = np.ascontiguousarray(
                                self._regions[r_idx, 1], np.int32
                            )
                            t_ends = np.ascontiguousarray(
                                self._regions[r_idx, 2], np.int32
                            )
                        row_starts_w = np.repeat(t_starts, n_s).astype(np.int32)
                        row_ends_w = np.repeat(t_ends, n_s).astype(np.int32)
                        row_lengths_w = (row_ends_w - row_starts_w).astype(np.int64)

                        if self._realign_tracks:
                            # M2 (#375 review): narrow to the checked
                            # `_MixedTracksBackend` protocol here, at the one
                            # call site that matters, rather than trusting the
                            # `supports_mixed_tracks` guard above alone. The
                            # guard already makes this assert unreachable in
                            # practice; its job is to turn a backend that sets
                            # the flag without (or with a wrongly-shaped)
                            # `mixed_realign_window` into a `pyrefly` error or
                            # a clear `AssertionError` here, instead of an
                            # `AttributeError` deep in the batch loop.
                            assert isinstance(backend, _MixedTracksBackend), (
                                f"{type(backend).__name__} sets "
                                "supports_mixed_tracks without a conforming "
                                "mixed_realign_window."
                            )
                            realign_w, t_ends_ext = backend.mixed_realign_window(
                                r_idx,
                                s_idx_w,
                                np.ascontiguousarray(t_starts, np.int32),
                                np.ascontiguousarray(t_ends, np.int32),
                                row_starts_w,
                                row_ends_w,
                            )
                        else:
                            # Un-realigned tracks stay in reference coordinates
                            # -- no deletion-extension read-ahead needed.
                            realign_w = None
                            t_ends_ext = t_ends

                        per_track = tb.read_window(
                            r_idx,
                            s_idx_w,
                            np.ascontiguousarray(t_starts, np.int32),
                            np.ascontiguousarray(t_ends_ext, np.int32),
                        )
                        track_w = _TrackWindow(
                            # Flattened AND FFI-coerced ONCE per window, not
                            # once per batch per track (see `_ItvArrays`).
                            itvs=_coerce_window_itvs(per_track),
                            names=tb.names,
                            row_starts=row_starts_w,
                            row_ends=row_ends_w,
                            row_lengths=row_lengths_w,
                            realign=realign_w,
                        )
                    else:
                        track_w = None
                    for lo, hi in _batch_bounds(0, n_rows, batch_size):
                        nxt = next_batch()
                        if nxt is None:
                            raise RuntimeError(
                                "streaming engine exhausted before the plan did"
                            )
                        if _variants:
                            b_times_p = (hi - lo) * backend.ploidy
                            row_off = np.asarray(nxt["offsets"], np.int64)
                            seq_off = np.asarray(nxt["alt_offsets"], np.int64)
                            char = np.asarray(nxt["alt"], np.uint8).view("S1")
                            alt = (
                                Ragged.from_offsets(
                                    char, (b_times_p, None, None), [row_off, seq_off]
                                )
                                .to_strings()
                                .reshape(hi - lo, backend.ploidy, None)
                            )
                            start = Ragged.from_offsets(
                                np.asarray(nxt["start"], np.int32),
                                (b_times_p, None),
                                row_off,
                            ).reshape(hi - lo, backend.ploidy, None)
                            # Minor (Wave B PR-B3a review): `nxt` always carries `ilen`
                            # regardless of what was requested, but building the `Ragged`
                            # is wasted work when the caller didn't ask for it -- guard on
                            # `_active_var_fields`, same as the `RaggedVariants(...)`
                            # construction below already does.
                            ilen = (
                                Ragged.from_offsets(
                                    np.asarray(nxt["ilen"], np.int32),
                                    (b_times_p, None),
                                    row_off,
                                ).reshape(hi - lo, backend.ploidy, None)
                                if "ilen" in _active_var_fields
                                else None
                            )
                            # Wave B PR-B3a (#304): `ref` (an allele field, opaque-string
                            # like `alt`) and any other requested var_field (a numeric
                            # INFO/index column, plain numeric like `start`/`ilen`) are
                            # packed the same way `alt`/`start`/`ilen` are above, keyed
                            # by the field's own name from `next_batch_variants`'s dict.
                            # `active_var_fields` (not raw dict keys) drives which of
                            # `ilen`/`ref` are actually attached to the output record --
                            # `nxt` always carries `ilen` (and `ref`/`ref_offsets` iff
                            # `want_ref` was set at `build_engine` time), independent of
                            # what the caller asked for, so the record-shape decision
                            # must be var_fields-driven, not presence-driven.
                            ref_rag = None
                            if "ref" in nxt:
                                ref_char = np.asarray(nxt["ref"], np.uint8).view("S1")
                                ref_rag = (
                                    Ragged.from_offsets(
                                        ref_char,
                                        (b_times_p, None, None),
                                        [
                                            row_off,
                                            np.asarray(nxt["ref_offsets"], np.int64),
                                        ],
                                    )
                                    .to_strings()
                                    .reshape(hi - lo, backend.ploidy, None)
                                )
                            extra: dict[str, Ragged] = {}
                            for _name in _active_var_fields:
                                if _name in ("alt", "start", "ilen", "ref"):
                                    continue
                                if _name not in nxt:
                                    # Defensive assert, not reachable via the public API:
                                    # `with_settings`'s `servable_var_fields` check
                                    # (Important 1, Wave B PR-B3a review) now rejects any
                                    # available-but-not-servable field before a
                                    # `build_engine` even happens. Kept here in case a
                                    # backend's `available_var_fields`/`servable_var_fields`
                                    # ever drift out of sync with what its engine actually
                                    # forwards -- fail loudly instead of a bare `KeyError`.
                                    raise NotImplementedError(
                                        f"var_fields={_name!r} is not yet forwarded by "
                                        "the streaming engine for this backend "
                                        "(deferred follow-up work)."
                                    )
                                extra[_name] = Ragged.from_offsets(
                                    np.asarray(nxt[_name]),
                                    (b_times_p, None),
                                    row_off,
                                ).reshape(hi - lo, backend.ploidy, None)
                            out = RaggedVariants(
                                alt=alt,
                                start=start,
                                ilen=ilen,
                                ref=ref_rag,
                                **extra,
                            )
                        elif _variant_windows:
                            # Wave B PR-B4 (#304): `nxt` carries `start`/`ilen`/
                            # `offsets` (one entry per kept variant, same
                            # `(row, ploid)`-boundary `offsets` every other output
                            # kind uses) plus `<name>`/`<name>_offsets` for whichever
                            # of `ref_window`/`alt_window`/`ref`/`alt` the engine's
                            # `WindowModeConfig` emitted (`ref`/`alt` mode "allele";
                            # `ref_window`/`alt_window` mode "window" -- the unused
                            # pair member is simply absent from `nxt`, never `None`).
                            # Token buffers have TWO ragged axes (variant count AND
                            # window length: `(b, p, ~v, ~w)`) -- mirrors
                            # `_FlatWindow.to_ragged` (`_flat_variants.py`); the
                            # scalars have one (`(b, p, ~v)`) -- mirrors
                            # `_Flat.to_ragged`.
                            b_times_p = (hi - lo) * backend.ploidy
                            row_off = np.asarray(nxt["offsets"], np.int64)
                            # No type annotation here (Minor, Wave B PR-B4 review):
                            # `out` is reused across this if/elif chain's mutually
                            # exclusive branches (`RaggedVariants` / this dict /
                            # `RaggedAnnotatedHaps` / plain `Ragged`), and annotating
                            # just this branch as `dict[str, Ragged]` reads as if that
                            # were `out`'s type everywhere in the function, not only
                            # here.
                            out = {}
                            for _name in ("ref_window", "alt_window", "ref", "alt"):
                                if _name not in nxt:
                                    continue
                                out[_name] = Ragged.from_offsets(
                                    np.asarray(nxt[_name]),
                                    (hi - lo, backend.ploidy, None, None),
                                    [
                                        row_off,
                                        np.asarray(nxt[f"{_name}_offsets"], np.int64),
                                    ],
                                )
                            out["start"] = Ragged.from_offsets(
                                np.asarray(nxt["start"], np.int32),
                                (b_times_p, None),
                                row_off,
                            ).reshape(hi - lo, backend.ploidy, None)
                            out["ilen"] = Ragged.from_offsets(
                                np.asarray(nxt["ilen"], np.int32),
                                (b_times_p, None),
                                row_off,
                            ).reshape(hi - lo, backend.ploidy, None)
                        elif _annotated:
                            data, annot_v, annot_pos, offsets = nxt
                            shape = (hi - lo, backend.ploidy, None)
                            offsets = np.asarray(offsets, np.int64)
                            out = RaggedAnnotatedHaps(
                                Ragged.from_offsets(
                                    np.asarray(data).view("S1"), shape, offsets
                                ),
                                Ragged.from_offsets(
                                    np.asarray(annot_v), shape, offsets
                                ),
                                Ragged.from_offsets(
                                    np.asarray(annot_pos), shape, offsets
                                ),
                            )
                        else:
                            data, offsets = nxt
                            if track_w is not None:
                                # Issue #380: the mixed drive yields BOTH halves,
                                # `(haplotypes, tracks)`, matching the written
                                # path's `HapsTracks.__call__` return order
                                # (`_reconstruct.py:121-144`). It previously
                                # reconstructed the haplotype bytes and then
                                # DISCARDED them, keeping `offsets` alone: that was
                                # both wasted work on every batch and an
                                # un-checkable parity gap, since the mixed fixtures
                                # could only compare the track half against
                                # `Dataset[r, s]`. `offsets` is still needed below
                                # for the real per-(row, hap) output length -- it
                                # already reflects `with_len(L)` when set, since
                                # `svar1_generate_batch` bakes the fixed length
                                # into `offsets` itself.
                                offsets = np.asarray(offsets, np.int64)
                                haps = Ragged.from_offsets(
                                    np.asarray(data).view("S1"),
                                    (hi - lo, backend.ploidy, None),
                                    offsets,
                                )
                                if track_w.realign is not None:
                                    out_lengths = np.diff(offsets).reshape(
                                        hi - lo, backend.ploidy
                                    )
                                    regions_batch = np.stack(
                                        [
                                            np.zeros(hi - lo, np.int32),
                                            track_w.row_starts[lo:hi],
                                            track_w.row_ends[lo:hi],
                                        ],
                                        axis=1,
                                    ).astype(np.int32)
                                    # The written path's own formula, verbatim:
                                    # xor-reduce of the dataset-global ravel
                                    # index (`_reconstruct.py:216-218`). Only
                                    # `FlankSample` reads this seed -- every
                                    # strategy the parity tests exercise
                                    # (Repeat5p, Repeat5pNormalized) ignores it
                                    # -- but matching the formula costs nothing
                                    # and removes a gratuitous divergence. Exact
                                    # `FlankSample` parity remains unattainable
                                    # in ANY batched path, written or streaming,
                                    # because the kernel also mixes in the
                                    # batch-local query index
                                    # (`src/tracks/mod.rs:583-590`) and the
                                    # `Dataset[r, s]` oracle always has query 0.
                                    _idx = flat_r[lo:hi].astype(np.uint64) * np.uint64(
                                        self.n_samples
                                    ) + flat_s[lo:hi].astype(np.uint64)
                                    base_seed = int(np.bitwise_xor.reduce(_idx))
                                    tracks = track_w.realign.realign_batch(
                                        lo,
                                        hi,
                                        track_w.itvs,
                                        track_w.names,
                                        self._insertion_fill,
                                        regions_batch,
                                        track_w.row_lengths[lo:hi],
                                        out_lengths,
                                        base_seed,
                                    )
                                else:
                                    if isinstance(self._output_length, int):
                                        _lengths = np.full(
                                            hi - lo, self._output_length, np.int64
                                        )
                                    else:
                                        _lengths = track_w.row_lengths[lo:hi]
                                    tracks = _tracks_from_intervals(
                                        track_w.itvs,
                                        np.arange(lo, hi, dtype=np.int64),
                                        track_w.row_starts[lo:hi],
                                        _lengths,
                                    )
                                out = (haps, tracks)
                            else:
                                out = Ragged.from_offsets(
                                    np.asarray(data).view("S1"),
                                    (hi - lo, backend.ploidy, None),
                                    np.asarray(offsets, np.int64),
                                )
                        yield (out, flat_r[lo:hi], flat_s[lo:hi])
                # `-O`-safe (Minor 3): a bare `assert` is stripped under `python -O`,
                # silently dropping this end-of-plan invariant.
                if next_batch() is not None:
                    raise RuntimeError(
                        "streaming engine had extra batches beyond the plan"
                    )
            elif self._prefetch_strategy == "readahead":
                # Design C (issue #283): single-thread read-ahead-one-window drive.
                # Reuses the SAME `_Svar1Backend.read_window`/`generate_batch` calls
                # the engine's consumer makes (Task 3, parity-green) -- prefetch is a
                # pure page-warming no-op on output, so this is byte-identical to the
                # "engine" branch above. See `test_streaming_matches_written_all_cells`'s
                # "readahead" parity variant.
                #
                # Phase 1: "readahead" is SVAR1-only (see the "engine" branch's
                # comment on the union narrowing).
                assert isinstance(self._backend, _Svar1Backend), (
                    '"readahead" prefetch strategy requires the SVAR1 backend'
                )
                backend = self._backend
                #
                # Jitter (issue #277 Task 3) is NOT supported here: `read_window`/
                # `generate_batch` re-derive region bounds internally from
                # `self._regions` (unjittered) -- there is no seam to pass translated
                # bounds through without a backend signature change. This experimental,
                # non-default toggle (issue #283 A-vs-C measurement) is out of scope
                # for that change; fail fast rather than silently ignoring jitter.
                # Tracks are wired into the "engine" drive only; this branch
                # yields haplotypes and would drop them entirely -- silently
                # wrong output, the failure mode every other guard here exists
                # to prevent. Unreachable through the public API
                # (`_prefetch_strategy` is backend-derived), but the A-vs-C
                # bench harness sets it via `object.__setattr__`.
                if self._track_backend is not None:
                    raise NotImplementedError(
                        "StreamingDataset tracks= is only supported with the "
                        'default "engine" prefetch strategy; the experimental '
                        '"readahead" toggle has no track path and would yield '
                        "haplotypes alone. Do not set tracks= with "
                        '_prefetch_strategy="readahead".'
                    )
                if self._jitter > 0:
                    raise NotImplementedError(
                        "StreamingDataset read-time jitter (jitter>0) is only "
                        'supported with the default "engine" prefetch strategy; the '
                        'experimental "readahead" toggle re-derives region bounds '
                        "internally and cannot accept translated bounds. Use the "
                        'default `_prefetch_strategy="engine"` (do not set jitter '
                        'with "readahead").'
                    )
                # Annotated / variants / variant-windows output are likewise only
                # wired through the default "engine" path: `_Svar1Backend.generate_batch`
                # (the readahead seam) unconditionally produces haplotypes -- it has no
                # annotated/variants/variant-windows variant, and this experimental
                # toggle is out of scope for any of those additions (issue #277 Wave A
                # Task 4 for "annotated"; Wave B PR-B1/PR-B4 for "variants"/
                # "variant-windows", #304). Without this guard the branch would
                # silently yield haplotypes instead of the requested output kind. Fail
                # fast rather than silently ignoring `with_seqs(...)`.
                if _annotated or _variants or _variant_windows:
                    kind = (
                        "annotated"
                        if _annotated
                        else "variants"
                        if _variants
                        else "variant-windows"
                    )
                    raise NotImplementedError(
                        f"StreamingDataset with_seqs({kind!r}) is only supported "
                        'with the default "engine" prefetch strategy; the '
                        'experimental "readahead" toggle only produces haplotypes '
                        "(`_Svar1Backend.generate_batch` has no annotated/variants/"
                        "variant-windows variant). Use the default "
                        '`_prefetch_strategy="engine"` (do not combine '
                        f'with_seqs({kind!r}) with "readahead").'
                    )

                from ..genvarloader import svar1_prefetch_runs

                # Compact, region-scale plan (same treatment as the engine branch): hold
                # only `(r_idx, s_lo, s_hi)` per window, NOT the per-window `s_idx` arrays.
                # `_plan` yields contiguous `arange(s_lo, s_hi)` chunks, so `s_idx` is
                # reconstructed per window on demand (only ~2 windows live at once under
                # the 1-ahead readahead) -- never O(n_windows x n_samples) resident.
                ra_jobs: list[tuple[NDArray[np.intp], int, int]] = [
                    (r_idx, int(s_idx[0]), int(s_idx[-1]) + 1)
                    for r_idx, s_idx in self._plan()
                ]
                if not ra_jobs:
                    return

                def _read(job: tuple[NDArray[np.intp], int, int]):
                    r_idx, s_lo, s_hi = job
                    return backend.read_window(
                        r_idx, np.arange(s_lo, s_hi, dtype=np.intp)
                    )

                # Read window 0's offsets up front; then for each window, prefetch the
                # NEXT window's runs before generating the CURRENT one so kernel
                # readahead of the next window's pages overlaps this window's
                # generation.
                cur = _read(ra_jobs[0])
                for i, (r_idx, s_lo, s_hi) in enumerate(ra_jobs):
                    if i + 1 < len(ra_jobs):
                        nxt = _read(ra_jobs[i + 1])
                        svar1_prefetch_runs(backend._store, nxt[0], nxt[1])
                    else:
                        nxt = None
                    s_idx = np.arange(s_lo, s_hi, dtype=np.intp)
                    n_s = len(s_idx)
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(s_idx, len(r_idx))
                    n_rows = len(flat_r)
                    for lo, hi in _batch_bounds(0, n_rows, batch_size):
                        data = backend.generate_batch(
                            r_idx, s_idx, cur[0], cur[1], lo, hi, _out_len
                        )
                        yield data, flat_r[lo:hi], flat_s[lo:hi]
                    cur = nxt
            elif self._prefetch_strategy == "sync":
                # SVAR2 super-batch drive: read each window's ranges once, reconstruct
                # a coarse super-batch into ONE recycled `Svar2ReconBuf` (the rayon
                # dispatch grain -- `should_parallelize` gates it), then drain
                # batch_size slices out of it. Output stays (hi-lo)-bounded per
                # drained batch (#284); iteration order is deterministic (relaxed
                # order is PR 3). This is `_Svar2Backend`'s Phase 2 drive -- its
                # window shape (`SparseVar2._find_ranges` bundle) doesn't fit the
                # engine/readahead branches above, which are SVAR1-shaped (offsets
                # CSR + Svar1Store). Narrow the union so calls below resolve against
                # `_Svar2Backend`'s signatures (see the "engine" branch's comment).
                from .._threads import should_parallelize
                from ..genvarloader import Svar2ReconBuf

                assert isinstance(self._backend, _Svar2Backend), (
                    '"sync" prefetch strategy requires the SVAR2 backend'
                )
                backend = self._backend
                buf = Svar2ReconBuf(backend.ploidy)  # one recycled buffer per iterator
                sb_rows = backend._super_batch_rows
                for r_idx, s_idx in self._plan():
                    window = backend.read_window(r_idx, s_idx)
                    n_s = len(s_idx)
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(s_idx, len(r_idx))
                    n_rows = len(flat_r)
                    for sb_lo, sb_hi in _super_batch_bounds(n_rows, sb_rows):
                        # Estimated output bytes gate `parallel` *before* the fill
                        # (the fill IS the reconstruct, so exact total_bytes is only
                        # known after): a tiny tail stays serial (PR-1a), a
                        # core-saturating super-batch parallelizes (PR-2). An
                        # overestimate only flips the decision, never correctness.
                        backend._fill_super_batch(
                            r_idx,
                            s_idx,
                            window,
                            sb_lo,
                            sb_hi,
                            buf,
                            parallel=should_parallelize(
                                backend._est_out_bytes(r_idx, sb_hi - sb_lo)
                            ),
                        )
                        for lo, hi in _batch_bounds(sb_lo, sb_hi, batch_size):
                            data = backend._drain(buf, lo - sb_lo, hi - sb_lo)
                            yield data, flat_r[lo:hi], flat_s[lo:hi]
            elif self._prefetch_strategy == "svar2_engine":
                # PR-3 Task 3: drive a `Svar2StreamEngine` (Task 2) in lockstep with
                # `_plan()`. The engine RESETS its batch boundaries at every
                # super-batch (it splits each window into `ceil(n_rows /
                # super_batch_rows)` jobs and drains each independently in
                # `batch_size` chunks from row 0 -- see
                # `src/ffi/svar2_stream_engine.rs`'s `slice_current`/
                # `CurrentWindow`), so the Python drive must NEST over super-batches
                # the same way the "sync" branch above does, not step `batch_size`
                # continuously over a window's full `n_rows` (that desyncs whenever
                # a window spans >1 super-batch and `super_batch_rows` doesn't
                # divide `batch_size` -- whole-branch-review Critical, see
                # `test_svar2_engine_nested_super_batch_matches_written`).
                # `sb_rows` must equal what `build_engine` passed the engine
                # (`int(self._super_batch_rows)`, unmodified) for the two to agree.
                # Only the isinstance check + backend type + `build_engine` call
                # differ from the SVAR1 "engine" branch above (SVAR2-shaped job
                # tuples, no offsets CSR / Svar1Store). Not yet the default
                # (`_Svar2Backend._default_strategy` stays "sync"; Task 4 decides
                # any flip) -- this is a test-only seam (`_with_strategy`) until then.
                assert isinstance(self._backend, _Svar2Backend), (
                    '"svar2_engine" prefetch strategy requires the SVAR2 backend'
                )
                backend = self._backend
                sb_rows = backend._super_batch_rows
                plan_jobs: list[tuple[int, NDArray[np.intp], int, int]] = []
                for r_idx, s_idx in self._plan():
                    contig_idx, _contig = backend._contig_of(r_idx)
                    plan_jobs.append(
                        (contig_idx, r_idx, int(s_idx[0]), int(s_idx[-1]) + 1)
                    )
                engine_jobs = [
                    (
                        c_idx,
                        np.ascontiguousarray(self._regions[r_idx, 1], np.uint32),
                        np.ascontiguousarray(self._regions[r_idx, 2], np.uint32),
                        s_lo,
                        s_hi,
                    )
                    for (c_idx, r_idx, s_lo, s_hi) in plan_jobs
                ]
                engine = backend.build_engine(engine_jobs, batch_size)
                del engine_jobs
                for _c_idx, r_idx, s_lo, s_hi in plan_jobs:
                    n_s = s_hi - s_lo
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(np.arange(s_lo, s_hi, dtype=np.intp), len(r_idx))
                    n_rows = len(flat_r)
                    for sb_lo, sb_hi in _super_batch_bounds(n_rows, sb_rows):
                        for lo, hi in _batch_bounds(sb_lo, sb_hi, batch_size):
                            nxt = engine.next_batch()
                            if nxt is None:
                                raise RuntimeError(
                                    "Svar2StreamEngine exhausted before the plan did"
                                )
                            data, offsets = nxt
                            yield (
                                Ragged.from_offsets(
                                    np.asarray(data).view("S1"),
                                    (hi - lo, backend.ploidy, None),
                                    np.asarray(offsets, np.int64),
                                ),
                                flat_r[lo:hi],
                                flat_s[lo:hi],
                            )
                if engine.next_batch() is not None:
                    raise RuntimeError(
                        "Svar2StreamEngine had extra batches beyond the plan"
                    )
            else:
                raise ValueError(
                    f"StreamingDataset._prefetch_strategy={self._prefetch_strategy!r} "
                    'is not recognized; expected "engine", "readahead", "sync", or '
                    '"svar2_engine".'
                )
        else:
            for r_idx, s_idx in self._plan():
                n_s = len(s_idx)
                flat_r = np.repeat(self._sort_order[r_idx], n_s)
                flat_s = np.tile(s_idx, len(r_idx))
                n_rows = len(flat_r)
                data = self._reconstruct_window(r_idx, s_idx)
                for lo, hi in _batch_bounds(0, n_rows, batch_size):
                    yield data[lo:hi], flat_r[lo:hi], flat_s[lo:hi]

    def to_iter(
        self, batch_size: int = 1, return_indices: bool = True
    ) -> Iterator[tuple]:
        """Iterate haplotype batches.

        **This is the one iteration entry point** -- :meth:`to_torch_dataset` and
        :meth:`to_dataloader` are thin wrappers over it, and there is no
        ``__iter__`` (one and only one obvious way).

        Iteration is a fixed cartesian sweep of BED regions x samples in a
        data-layout-optimal order (region-major for variants). There is no random
        access and no ad-hoc query: ``sds[r, s]`` raises :class:`TypeError`.
        ``iteration_order`` (set at construction) picks between region-major and
        sample-major sweeps, but it only changes anything when the sample axis
        is chunked across more than one read window. At the default
        ``max_mem="512MB"`` the sample chunk holds the entire axis unless the
        cohort is very large -- about 8.4M samples with no tracks, but only
        ~340k with one track and ~170k with two, since each track adds 768 B
        per cell (see ``TRACK_BYTES_PER_CELL``). Below that threshold the
        sample loop runs once and both orders emit the identical plan. Check
        :attr:`iteration_order_is_active` before relying on it to change
        observed behavior.

        Args:
            batch_size: Number of ``(region, sample)`` cells per yielded batch.
                Batches are slices of a much larger read *window*; ``batch_size``
                does not affect I/O granularity.
            return_indices: If ``True`` (the default), yield
                ``(data, region_idxs, sample_idxs)``; if ``False``, yield ``data``
                alone. Indices are in the caller's **original BED-row order** (not
                sorted-storage order), matching ``gvl.Dataset[r, s]``.

        Yields:
            ``(data, region_idxs, sample_idxs)`` when ``return_indices`` is ``True``,
            otherwise ``data`` alone.

        **Tracks** (``tracks=``): with a variant source AND tracks, ``data`` is a
        ``(haplotypes, tracks)`` pair -- the same 2-tuple, in the same order, that
        written ``gvl.Dataset[r, s]`` returns when both are configured
        (``HapsTracks``, ``_reconstruct.py``). Tracks-only datasets (no
        ``variants=``) yield the tracks alone, again matching the written path.
        With variants and the default ``realign_tracks=True`` the tracks are
        re-aligned to haplotype coordinates and carry a ploidy axis, shape
        ``(batch, n_tracks, ploidy, None)``; with ``realign_tracks=False`` they
        stay in reference coordinates and drop it, ``(batch, n_tracks, None)``;
        without variants the shape is ``(batch, n_tracks, None)``. The
        haplotype half is always ``(batch, ploidy, None)`` -- ``realign_tracks``
        only ever affects the track half. The track axis is ordered by track NAME
        (never ``tracks=`` argument order) and is never squeezed -- a single track
        still yields a length-1 axis.

        **Read-time jitter** (``jitter>0``, set via ``with_settings``):
        When ``jitter>0``, each region's read window is translated by an integer
        offset drawn from ``Uniform[-jitter, jitter]`` (inclusive), clamped only so
        the translated start stays ``>= 0`` -- the window SIZE never changes
        (translate, don't resize), so a fixed ``with_len(L)`` output is unaffected.
        A translated end may run past the contig; that's safe, not a bug -- the
        reconstruction kernel already N-pads a region that overhangs a chromosome
        edge (the same behavior any ``gvl.Dataset`` region gets there), and
        clamping the end as well would make jitter a no-op for any region that
        already spans its whole contig. Exactly one ``numpy.random.Generator``
        (seeded from ``rng``, see ``with_settings``) is created per ``to_iter``
        call; one offset per region is drawn from it in a single vectorized draw,
        **once per** ``to_iter`` **call** (before the internal window/sample-chunk
        loop), indexed by each region's absolute index in **sweep order** (regions
        are pre-sorted by ``(contig, start)`` and ``_plan()`` sweeps them in that
        index order). Because the draw is keyed by region index rather than by
        plan step, the same region gets the same offset every time it recurs --
        including when a large cohort makes ``_plan()`` revisit the same region
        once per sample chunk -- so jitter output is independent of
        ``max_mem``/cohort-size-driven chunking. The
        same ``rng`` always reproduces the same translated windows and a different
        ``rng`` almost certainly does not. This is a **reproducible augmentation**,
        **NOT** byte-parity with a written ``gvl.Dataset`` -- ``jitter=0`` (the
        default) is the only byte-parity-gated setting. Only the default
        ``"engine"`` prefetch strategy supports jitter; the experimental
        ``"readahead"`` toggle raises :class:`NotImplementedError` if combined with
        ``jitter>0`` (it re-derives region bounds internally and has no seam to
        accept translated ones).

        Per-hap within-window sub-shifts (``deterministic=False`` further jittering
        each haplotype independently within its already-translated window, for
        fixed-length output) are a **documented Wave A deferral** -- not
        implemented yet; that needs a Rust engine ``shifts`` API addition and is
        out of scope for this plan. ``deterministic`` currently has no effect
        beyond gating that unimplemented path.
        """
        for data, r_idx, s_idx in self._iter_batches(batch_size):
            if return_indices:
                yield data, r_idx, s_idx
            else:
                yield data

    def n_batches(self, batch_size: int) -> int:
        """Number of batches :meth:`to_iter` will yield at ``batch_size``.

        NOT ``ceil(len(self) / batch_size)``: the plan batches *within* each window,
        so every window's last batch may be partial -- and on the super-batched SVAR2
        drives, within each *super-batch*, so every super-batch's last batch may be
        partial too (issue #379). Counting the plan is cheap (it only materializes
        small index arrays).
        """
        return sum(1 for _ in self._iter_batch_spans(batch_size))

    def _drive_super_batch_rows(self) -> int | None:
        """Super-batch width the drive will use, or ``None`` if it does not nest.

        Only the SVAR2 drives (``"sync"``/``"svar2_engine"``) batch *inside* a
        super-batch. Read off the backend rather than recomputed, for the same
        reason the drives read it -- see ``_Svar2Backend.__init__``.
        """
        if self._prefetch_strategy in ("sync", "svar2_engine"):
            return getattr(self._backend, "_super_batch_rows", None)
        return None

    def _iter_batch_spans(self, batch_size: int) -> Iterator[int]:
        """Batch sizes the plan will yield, without reconstructing anything.

        Walks the SAME nesting the drives walk (``_super_batch_bounds`` ->
        ``_batch_bounds``) rather than re-deriving a flat ``range`` that silently
        drops each super-batch's partial tail batch (issue #379).
        """
        sb_rows = self._drive_super_batch_rows()
        for r_idx, s_idx in self._plan():
            n_rows = len(r_idx) * len(s_idx)
            for sb_lo, sb_hi in _super_batch_bounds(n_rows, sb_rows or n_rows):
                for b_lo, b_hi in _batch_bounds(sb_lo, sb_hi, batch_size):
                    yield b_hi - b_lo

    def with_seqs(
        self,
        kind: Literal["haplotypes", "annotated", "variants", "variant-windows"],
        opt: "VarWindowOpt | None" = None,
    ) -> "StreamingDataset":
        """Select the sequence output kind.

        ``"haplotypes"`` (default), ``"annotated"``
        (:class:`AnnotatedHaps` -- haplotypes plus per-position
        variant indices and reference coordinates), ``"variants"`` (no
        sequences, just variants as :class:`RaggedVariants`), or
        ``"variant-windows"`` (no reconstructed sequences; instead, per-variant
        tokenized ref/alt windows -- or bare tokenized alleles -- as a
        ``dict[str, Ragged]``, matching the written path's
        ``Dataset.with_seqs("variant-windows")`` output shape). All four are
        supported for the SVAR1, VCF, and PGEN backends; none is yet wired for
        ``.svar2``. ``"reference"`` is a later follow-up.

        ``"annotated"`` is **not supported for the VCF backend** (its
        ``var_idxs`` are dataset-global variant ids a VCF source cannot produce
        cheaply; issues #305, #311) -- materializing such a dataset raises
        :class:`NotImplementedError`. Use a PGEN or SVAR source for annotated
        output.

        Args:
            kind: The sequence output kind.
            opt: Required for, and only accepted with, ``kind="variant-windows"``
                (Wave B PR-B4, #304): a
                :class:`~genvarloader._dataset._flat_variants.VarWindowOpt`
                configuring the flank length, token alphabet, unknown token, and
                per-side (``ref``/``alt``) window-vs-allele mode. Passing ``opt``
                with any other ``kind``, or omitting it for ``"variant-windows"``,
                raises :class:`ValueError`.

        Raises:
            NotImplementedError: ``kind`` is unrecognized, or
                ``kind="variant-windows"`` was requested against the SVAR2
                (``.svar2``) backend (not yet wired; see the SVAR1/VCF/PGEN
                engines).
            ValueError: ``opt`` was omitted for ``kind="variant-windows"``, or
                supplied for any other ``kind``. Also raised if this
                ``StreamingDataset`` has no variant source at all (tracks-only
                construction, i.e. ``variants=`` was never supplied) --
                ``with_seqs`` has nothing to reconstruct from.
        """
        if self._backend is None and self._reconstruct_window is None:
            # Tracks-only: no variant source to reconstruct from (never true of
            # the injected-callback test path, which sets `_reconstruct_window`
            # even though it also has no `_backend`).
            raise ValueError(
                "with_seqs() requires a variant source; this StreamingDataset"
                " has no variant source (tracks only), so it yields tracks"
                " alone from to_iter()."
            )
        kind_map = {
            "haplotypes": RaggedSeqs,
            "annotated": RaggedAnnotatedHaps,
            "variants": RaggedVariants,
            "variant-windows": dict,
        }
        if kind not in kind_map:
            raise NotImplementedError(
                f"StreamingDataset.with_seqs({kind!r}) is not implemented; "
                'supported: "haplotypes", "annotated", "variants", '
                '"variant-windows". "reference" is a later follow-up.'
            )
        if kind != "variant-windows" and opt is not None:
            raise ValueError(
                f"StreamingDataset.with_seqs({kind!r}, opt=...) is not supported; "
                '`opt` is only accepted for with_seqs("variant-windows", opt).'
            )
        out = copy.copy(self)
        if kind == "variant-windows":
            if opt is None:
                raise ValueError(
                    'StreamingDataset.with_seqs("variant-windows") requires a '
                    'VarWindowOpt, e.g. with_seqs("variant-windows", '
                    "VarWindowOpt(flank_length=..., token_alphabet=..., "
                    "unknown_token=...))."
                )
            if isinstance(self._backend, _Svar2Backend):
                raise NotImplementedError(
                    'StreamingDataset.with_seqs("variant-windows") is not supported '
                    "for the SVAR2 (.svar2) backend; it is wired only through the "
                    "SVAR1/VCF/PGEN engines (Wave B PR-B4, #304)."
                )
            from ._flat_flanks import build_token_lut

            lut, lut_dtype = build_token_lut(opt.token_alphabet, opt.unknown_token)
            object.__setattr__(out, "_var_window_opt", opt)
            object.__setattr__(out, "_var_window_lut", lut)
            object.__setattr__(out, "_var_window_lut_dtype", lut_dtype)
        else:
            # Hygiene: clear a previously-set variant-windows configuration when
            # switching away from it, so a stale `_var_window_opt` never lingers on
            # a dataset that no longer requests that output kind.
            object.__setattr__(out, "_var_window_opt", None)
            object.__setattr__(out, "_var_window_lut", None)
            object.__setattr__(out, "_var_window_lut_dtype", None)
        object.__setattr__(out, "_seq_kind", kind_map[kind])
        return out

    def with_len(self, length: "int | Literal['ragged']") -> "StreamingDataset":
        """Set haplotype/annotated output length.

        ``"ragged"`` (default) yields per-hap actual length; a fixed ``int >= 1``
        yields exactly that many bases
        per hap. Unlike :meth:`Dataset.with_len`, ``"variable"`` is not accepted:
        :meth:`to_iter` always yields ``Ragged`` (there is no ArrayDataset analog),
        so pad the ragged output yourself for a dense array.

        Args:
            length: ``"ragged"``, or a positive int no larger than the smallest
                region in the BED.

        Raises:
            NotImplementedError: If ``length`` is ``"variable"``.
            ValueError: If ``length`` is not a positive int, or exceeds the
                smallest region length.
        """
        if length == "variable":
            raise NotImplementedError(
                'StreamingDataset.with_len("variable") is not supported; to_iter '
                'always yields Ragged. Use with_len(int) or with_len("ragged").'
            )
        if length != "ragged":
            if not isinstance(length, (int, np.integer)) or int(length) < 1:
                raise ValueError(
                    f"with_len(length) must be a positive int or 'ragged', got {length!r}."
                )
            length = int(length)
            # Mirror `Dataset.with_len`'s upper bound (`_impl.py:551-559`):
            # the written oracle refuses `output_length + 2*jitter >
            # min_region_len + 2*max_jitter`, and without the same guard
            # streaming silently ZERO-PADS past the region on exactly the call
            # the oracle rejects -- bytes no parity test can check. Streaming
            # has no `max_jitter` (there is no written store to have been
            # widened at write time) and its jitter TRANSLATES the window
            # rather than consuming slack inside it (see `with_settings`), so
            # both `2 * jitter` terms drop out and the bound is just the
            # smallest region length.
            min_r_len = int((self._regions[:, 2] - self._regions[:, 1]).min())
            if length > min_r_len:
                raise ValueError(
                    f"with_len({length}) exceeds the minimum region length"
                    f" ({min_r_len}), which is the maximum output length of this"
                    " StreamingDataset. `Dataset.with_len` raises for the same"
                    " call, so there would be no parity oracle for the result;"
                    " streaming would silently zero-pad past the region."
                )
        out = copy.copy(self)
        object.__setattr__(out, "_output_length", length)
        return out

    def with_settings(
        self,
        *,
        jitter: "int | None" = None,
        rng: "int | np.random.Generator | None" = None,
        deterministic: "bool | None" = None,
        min_af: "float | None" = None,
        max_af: "float | None" = None,
        var_fields: "list[str] | None" = None,
        realign_tracks: "bool | None" = None,
    ) -> "StreamingDataset":
        """Modify jitter / rng / determinism, returning a new dataset.

        Mirrors the relevant subset of :meth:`Dataset.with_settings` (same
        parameter names).

        Args:
            jitter: Non-negative int; each region's read window is translated by an
                integer offset drawn from ``Uniform[-jitter, jitter]`` (window SIZE
                unchanged), clamped so the translated start stays ``>= 0`` (a
                translated end may safely run past the contig -- see :meth:`to_iter`'s
                docstring). ``0`` (the default) disables jitter and is the only
                byte-parity-gated setting.
            rng: Seed (int) or :class:`numpy.random.Generator` for the jitter draws.
                One ``Generator`` (via ``numpy.random.default_rng(rng)``) is created
                per :meth:`to_iter` call and drawn from once per region, in sweep order
                -- so the same ``rng`` reproduces the same translated windows across
                calls/runs, and a different ``rng`` yields different ones.
            deterministic: Reserved for per-hap within-window sub-shifts on
                fixed-length output; not yet implemented (documented Wave A deferral --
                needs a Rust engine API addition). Currently has no observable effect
                on ``to_iter``'s output.
            min_af: Inclusive lower allele-frequency bound for
                ``with_seqs("variants")``. Requires an available AF (SVAR
                ``cache_afs()``, or a VCF ``INFO/AF`` field); otherwise raises at
                iterate time. Matches :meth:`Dataset.with_settings`.
            max_af: Inclusive upper allele-frequency bound for
                ``with_seqs("variants")``. Requires an available AF (SVAR
                ``cache_afs()``, or a VCF ``INFO/AF`` field); otherwise raises at
                iterate time. Matches :meth:`Dataset.with_settings`.
            var_fields: Variant fields to emit for ``with_seqs("variants")`` output
                (Wave B PR-B3a/PR-B3b, #304). Must be a subset of
                :attr:`available_var_fields`; an unknown field raises
                :class:`ValueError` immediately (not at iterate time). A field that is
                available but not yet servable by the streaming engine (see
                :attr:`servable_var_fields`) raises :class:`NotImplementedError`
                immediately instead -- as of PR-B3b this gap is a SVAR1 numeric INDEX
                column (e.g. a cached ``AF``), not a per-call FORMAT field
                (``dosage``/custom FORMAT columns ARE servable since PR-B3b). Defaults
                to ``["alt", "ilen", "start"]`` (see :attr:`active_var_fields`) when
                never set -- this default reproduces today's ``with_seqs("variants")``
                output byte-for-byte. Only meaningful for ``with_seqs("variants")``;
                combining it with any other output kind raises
                :class:`NotImplementedError` at iterate time (matches how
                ``min_af``/``max_af`` are guarded).
            realign_tracks: Whether tracks are re-aligned to haplotype coordinates
                (default ``True``). ``False`` skips re-alignment: tracks keep
                reference coordinates and DROP the ploidy axis (shape
                ``(b, t, ~l)`` instead of ``(b, t, p, ~l)``), matching
                :meth:`Dataset.with_settings`. Only meaningful when ``tracks=`` was
                given; ignored otherwise.

        ``jitter>0`` is a documented, reproducible augmentation, NOT byte-parity
        with a written ``Dataset`` (see :meth:`to_iter`'s docstring for the full
        rng contract).
        """
        out = copy.copy(self)
        if jitter is not None:
            if jitter < 0:
                raise ValueError(f"jitter must be non-negative, got {jitter}.")
            object.__setattr__(out, "_jitter", int(jitter))
        if rng is not None:
            object.__setattr__(out, "_rng", rng)
        if deterministic is not None:
            object.__setattr__(out, "_deterministic", bool(deterministic))
        if min_af is not None:
            object.__setattr__(out, "_min_af", float(min_af))
        if max_af is not None:
            object.__setattr__(out, "_max_af", float(max_af))
        if var_fields is not None:
            missing = [f for f in var_fields if f not in self.available_var_fields]
            if missing:
                raise ValueError(
                    f"var_fields {missing} are not available for this source. "
                    f"Available: {self.available_var_fields}."
                )
            # Important 1 (Wave B PR-B3a review): available-but-not-servable is a
            # static per-backend fact (e.g. an AF-cached SVAR1 store's numeric index
            # columns beyond `ref`) -- fail here, at configuration time, rather than
            # from inside the per-batch packing loop after a full build_engine +
            # producer thread + first window read.
            unservable = [f for f in var_fields if f not in self.servable_var_fields]
            if unservable:
                raise NotImplementedError(
                    f"var_fields {unservable} are available but not yet servable by "
                    "the streaming engine for this backend (a SVAR1 numeric INDEX "
                    "column, e.g. a cached AF, forwarding it through the streaming "
                    "engine is still deferred follow-up work -- see "
                    "docs/roadmaps/streaming-dataset.md). "
                    f"Servable: {self.servable_var_fields}."
                )
            object.__setattr__(out, "_var_fields", list(var_fields))
        if realign_tracks is not None:
            object.__setattr__(out, "_realign_tracks", bool(realign_tracks))
        return out

    def with_insertion_fill(
        self, fill: "InsertionFill | dict[str, InsertionFill]"
    ) -> "StreamingDataset":
        """Set the insertion-fill strategy for re-aligned tracks.

        Mirrors :meth:`Dataset.with_insertion_fill`. Only meaningful when
        ``tracks=`` AND ``variants=`` were given and ``realign_tracks`` is
        ``True`` (the default). Insertion fill is read only while re-aligning
        tracks to haplotype coordinates, so each of those three conditions is
        required and each raises when it does not hold.

        Args:
            fill: Either a single :class:`InsertionFill` applied to every track,
                or a mapping from track name to a per-track strategy. A mapping
                must cover every track name exactly (see
                :attr:`StreamingDataset` construction docs for track naming);
                an unknown or missing name raises :class:`ValueError`.

        Returns:
            A new :class:`StreamingDataset` with the strategy set.

        Raises:
            ValueError: If the dataset has no tracks, if it has no variants, if
                ``realign_tracks`` is ``False`` (in either case the strategy
                would have no effect), or if ``fill`` is a mapping whose keys
                don't exactly match the track names.
        """
        from ._insertion_fill import InsertionFill

        tb = self._track_backend
        # Spec section 3.1: raise the written path's `ValueError`s
        # (`_impl.py:872-889`) rather than silently accepting a setting that
        # cannot change any byte -- the same fail-fast contract the
        # `var_fields` and `min_af`/`max_af` guards in this file enforce.
        if tb is None:
            raise ValueError(
                "with_insertion_fill requires tracks; there are none on this"
                " StreamingDataset. Pass tracks= at construction first."
            )
        if self._backend is None:
            # The third case in which the setting cannot change a byte, and the
            # one the two checks around it used to miss: on a tracks-only
            # dataset `_realign_tracks` is still `True` by default, but
            # `_insertion_fill` is read at exactly one site -- the MIXED
            # realigned branch of `_iter_batches` -- which the tracks-only drive
            # never reaches. Accepting the call there is precisely the silent
            # no-op the comment above refuses.
            raise ValueError(
                "with_insertion_fill has no effect on a tracks-only"
                " StreamingDataset: insertion fill only applies while"
                " re-aligning tracks to haplotype coordinates, which requires a"
                " variant source. Pass variants= at construction first, or drop"
                " the call."
            )
        if not self._realign_tracks:
            raise ValueError(
                "with_insertion_fill has no effect when realign_tracks=False"
                " (insertion fill only applies while re-aligning tracks to"
                " haplotype coordinates). Use"
                " with_settings(realign_tracks=True) first, or drop the call."
            )
        names = list(tb.names)
        if isinstance(fill, InsertionFill):
            resolved = {name: fill for name in names}
        else:
            fill_names = set(fill)
            if fill_names != set(names):
                missing = set(names) - fill_names
                extra = fill_names - set(names)
                raise ValueError(
                    "with_insertion_fill(dict) must have exactly one entry per "
                    f"track name. Missing: {sorted(missing)}. Unknown: "
                    f"{sorted(extra)}."
                )
            resolved = dict(fill)
        out = copy.copy(self)
        object.__setattr__(out, "_insertion_fill", resolved)
        return out

    def __getitem__(self, idx) -> None:
        raise TypeError(
            "StreamingDataset is iterable-only; use to_iter() instead of map-style "
            "indexing. Iteration order is fixed by the data layout, so there is no "
            "random access."
        )

    @requires_torch
    def to_torch_dataset(
        self, batch_size: int = 1, return_indices: bool = True
    ) -> "td.IterableDataset":
        """Wrap :meth:`to_iter` in a torch :class:`IterableDataset`.

        Thin wrapper -- all the work is in ``to_iter``. Named to match
        :meth:`Dataset.to_torch_dataset` (same concept, same name).
        """
        import torch.utils.data as td

        sds = self

        class _StreamingTorchDataset(td.IterableDataset):
            def __iter__(self):
                return sds.to_iter(batch_size, return_indices)

            def __len__(self) -> int:
                return sds.n_batches(batch_size)

        return _StreamingTorchDataset()

    @requires_torch
    def to_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        return_indices: bool = True,
        *,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Callable | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> "td.DataLoader":
        """Wrap :meth:`to_torch_dataset` in a torch ``DataLoader``.

        Thin wrapper over :class:`DataLoader <torch.utils.data.DataLoader>`.

        Args:
            batch_size: Rows per batch, forwarded to :meth:`to_torch_dataset`. The
                loader itself is constructed with ``batch_size=None`` because the
                dataset already yields assembled batches.
            num_workers: Must be 0. ``StreamingDataset``'s own engine IS the
                concurrency strategy (mirrors :meth:`Dataset.to_dataloader`'s
                ``buffered``/``double_buffered`` restriction); worker-process sharding
                of the window plan is a later plan.
            return_indices: Forwarded to :meth:`to_torch_dataset`; whether each batch
                carries its ``(region_idx, sample_idx)`` arrays.
            collate_fn: Passed through to ``DataLoader``.
            pin_memory: Passed through to ``DataLoader``.
            timeout: Passed through to ``DataLoader``.
            worker_init_fn: Passed through to ``DataLoader``.
            multiprocessing_context: Passed through to ``DataLoader``.
            prefetch_factor: Passed through to ``DataLoader``.
            persistent_workers: Passed through to ``DataLoader``.
            pin_memory_device: Passed through to ``DataLoader``.

        Returns:
            A :class:`DataLoader <torch.utils.data.DataLoader>` over this dataset's
            pre-assembled batches (``batch_size=None`` on the loader itself).

        Raises:
            ValueError: If ``num_workers > 0``.
        """
        if num_workers > 0:
            raise ValueError(
                "StreamingDataset.to_dataloader: num_workers>0 is not implemented "
                "yet; the streaming engine IS the concurrency strategy for "
                "StreamingDataset (mirrors gvl.Dataset.to_dataloader's "
                "buffered/double_buffered modes, which impose the same restriction). "
                "Use num_workers=0."
            )

        import torch.utils.data as td

        return td.DataLoader(
            self.to_torch_dataset(batch_size, return_indices),
            batch_size=None,  # the dataset yields pre-assembled batches
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )


class _Svar1Backend:
    """Streaming SVAR1 read backend, reading a live ``.svar`` store directly.

    Reconstructs haplotypes for a batch of ``(r_idx, s_idx)`` with no on-disk
    gvl dataset. Wraps `Svar1Store`/`svar1_read_window`/`svar1_generate_batch`
    (Rust) -- an instance is assigned to `StreamingDataset._backend`
    internally by the `.svar` construction branch (not a public `__init__`
    kwarg) so `_iter_batches` can read a window's offsets once
    (`read_window`) and generate each batch slice separately
    (`generate_batch`), bounding peak output by `batch_size` (issue #284).

    The static variant table (positions/ILEN/ALT alleles, GLOBAL across
    contigs) is read once at construction from ``SparseVar(path).index``; only
    per-region live genotype reads hit the store during iteration.
    """

    #: Mixed variants+tracks is wired for this backend (issue #375).
    supports_mixed_tracks: ClassVar[bool] = True

    def __init__(
        self,
        svar_path: str | Path,
        reference_path: str | Path,
        contigs: list[str],
        bed: pl.DataFrame | str | Path,
    ) -> None:
        from genoray import SparseVar
        from genoray._types import DOSAGE_TYPE

        from ..genvarloader import Svar1Store
        from ._haps import (
            _canonicalize_variant_table,
            _svar_format_fields,
            _variant_arrays_from_table,
        )
        from ._write import _reject_unsupported_variants
        from ._reference import Reference

        self._contigs = list(contigs)

        # Wave B PR-B2 (#317): `SparseVar._scan_index` only selects a fixed column
        # set (CHROM/POS/REF/ALT/ILEN/`index`) plus whatever `attrs` the caller
        # requests -- a plain `SparseVar(path)` silently omits a cached "AF" column
        # even when `cache_afs()` has already written it to `index.arrow` on disk.
        # Peek the on-disk schema (same `index.arrow` layout `_svar_link.py` reads
        # directly) and request "AF" explicitly when present, so `idx["AF"]` below
        # is actually populated for stores with cached AF. Read ONCE here (Minor,
        # Wave B PR-B3a review) -- the `available_var_fields` numeric-column scan
        # further down reuses `_raw_schema` instead of re-scanning the same file.
        _raw_schema = pl.scan_ipc(Path(svar_path) / "index.arrow").collect_schema()
        _has_cached_af = "AF" in _raw_schema
        sv = SparseVar(str(svar_path), attrs=["AF"] if _has_cached_af else None)
        # `gvl.write()` always lexicographically sorts sample names
        # (`_write.py`'s unconditional `samples.sort()`), so `gvl.Dataset`'s
        # sample index `s` means "the s-th name in sorted order" -- NOT the
        # store's native (VCF column) order. `sample_idx` must mean the same
        # thing here for parity with `gvl.Dataset.open(...)[r, s]` to hold
        # (see `to_iter`'s docstring). Toy fixtures with <=3 single-digit
        # sample names never exposed this because sort order and native
        # order coincide there; a 20-sample "S0".."S19" fixture does not
        # (lexicographically "S10" < "S2"). `_phys_sample_idx[i]` is the
        # store's native column for the i-th name in sorted order; every
        # sample index that crosses into Rust must go through it first.
        native_samples = sv.available_samples
        self._sample_names = sorted(native_samples)
        _name_to_phys = {name: i for i, name in enumerate(native_samples)}
        self._phys_sample_idx = np.array(
            [_name_to_phys[s] for s in self._sample_names], dtype=np.int64
        )
        self.n_samples = len(self._sample_names)
        self.ploidy = sv.ploidy
        # Additive (Phase 1 union type, see `StreamingDataset._backend`'s comment):
        # `__init__` reads these via `getattr(..., default)`, so pre-existing
        # instantiations of this class are unaffected either way.
        self._cell_bytes = int(self.ploidy) * 16
        self._default_strategy = "engine"

        self._ref = Reference.from_path(reference_path, self._contigs)

        # Same region-bounds derivation StreamingDataset itself uses (D1): a
        # batch's `r_idx` indexes into this same sorted regions table, so the
        # two stay aligned when both are built from the same `bed`.
        bed_df = bed if isinstance(bed, pl.DataFrame) else sp.bed.read(bed)
        self._regions = bed_to_regions(
            sp.bed.sort(bed_df), ContigNormalizer(self._contigs)
        )

        idx = sv.index.sort("index")
        # Wave B PR-B2 (#317): cached per-variant AF, row-aligned with `idx` at THIS
        # point -- read before `_canonicalize_variant_table` below, which is a pure
        # per-row column transform (no reorder/filter/row-count change), so alignment
        # with the Rust global `v_starts`/`ilens` tables built from the canonicalized
        # `idx` further down holds. `None` when the store has no cached AF (streaming
        # `min_af`/`max_af` is then a no-op via `has_cached_af`/`build_engine`).
        self._afs = (
            idx["AF"].to_numpy().astype(np.float32, copy=False)
            if "AF" in idx.columns
            else None
        )
        # Same "valid inputs only" contract `gvl.write` enforces (validated, not
        # fixed up). This is load-bearing here, not just parity-cosmetic: `ilens`
        # (used both to derive `v_ends` for the range search and by the
        # reconstruct kernel itself) and `alt` are only meaningful for
        # bi-allelic, non-symbolic records -- a `<DEL>` ALT would corrupt both the
        # window's overlap bound and the reconstructed sequence. Must run BEFORE
        # canonicalization, which collapses the list-typed ALT this check inspects.
        _reject_unsupported_variants(idx, "SparseVar (.svar)")
        idx = _canonicalize_variant_table(idx)
        v_starts, ilens, ref, alt = _variant_arrays_from_table(idx, one_based=True)
        if ref is None:
            raise ValueError(f"SVAR1 store at {svar_path} has no REF allele column.")
        self._v_starts = np.ascontiguousarray(v_starts, np.int32)
        self._ilens = np.ascontiguousarray(ilens, np.int32)
        self._alt_alleles = np.ascontiguousarray(alt.data.view(np.uint8), np.uint8)
        self._alt_offsets = np.ascontiguousarray(alt.offsets, np.int64)
        # Wave B PR-B3a (#304): REF was read and discarded here. It is a `var_field`
        # (and the `ref="allele"` input for variant-windows), so keep the global
        # per-variant byte table alongside the ALT one -- same layout, same gather.
        self._ref_alleles = np.ascontiguousarray(ref.data.view(np.uint8), np.uint8)
        self._ref_offsets = np.ascontiguousarray(ref.offsets, np.int64)

        # Wave B PR-B3a (#304): numeric index columns are the requestable INFO-like
        # fields, filtered exactly as `_Variants.available_info_fields` does on the
        # written path (`_haps.py`): numeric, minus the positional POS/ILEN columns
        # (and minus any name reserved for a builtin/FFI-fixed key -- see
        # `_RESERVED_VAR_FIELD_NAMES`). Scans the RAW on-disk `index.arrow` schema
        # (not `SparseVar.index`/`_scan_index`, which only flattens columns the
        # caller explicitly names via `attrs=`) -- in practice this is `[]` unless
        # `cache_afs()` has written a top-level `AF` column (genoray nests any other
        # declared INFO under a single `INFO` Struct column, which is not
        # `is_numeric()`). `ref` is always available -- the store is rejected above
        # if it has no REF column. NOTE: forwarding an arbitrary numeric INDEX column
        # like a cached `AF` through `next_batch_variants` is still deferred follow-up
        # work (unlike the per-call FORMAT/dosage fields just below, PR-B3b, which
        # travel the SAME `info_out` channel but ARE wired); the packing loop in
        # `_iter_batches` raises a clear `NotImplementedError` rather than a bare
        # `KeyError` if such an INDEX column is ever actually requested. Reuses
        # `_raw_schema` (read once above, for `_has_cached_af`) instead of re-scanning
        # the file.
        _info = [
            k
            for k, v in _raw_schema.items()
            if v.is_numeric()
            and k not in {"POS", "ILEN"}
            and k not in _RESERVED_VAR_FIELD_NAMES
        ]
        self.available_var_fields = [*_DEFAULT_VAR_FIELDS, *_info, "ref"]
        # Important 1 (Wave B PR-B3a review): servability is a static per-backend fact,
        # separate from `available_var_fields` above -- any numeric INDEX column in
        # `_info` (e.g. a cached `AF`) is advertised but not yet servable (see the NOTE
        # above); `with_settings` fails fast on the gap instead of the packing loop.
        self.servable_var_fields = [*_DEFAULT_VAR_FIELDS, "ref"]

        # Wave B PR-B3b (#304): per-call FORMAT fields (dosage + any genoray custom
        # Number=G FORMAT column) live PARALLEL to `variant_idxs.npy` on the SAME
        # hap-major CSR offsets -- i.e. indexed by CSR POSITION, not by variant id (the
        # opposite of the `_info` numeric INDEX columns above, which are indexed by
        # variant id). Each is therefore a plain memmap the Rust walk can index with
        # the SAME `o` it already uses to read `geno_v_idxs()[o]` -- no new search, no
        # new offsets. Field discovery reuses the written path's helper verbatim
        # (`_svar_format_fields`); reserved names are excluded for the same FFI-dict-
        # collision reason `_info` excludes them above (a custom FORMAT field named
        # e.g. "start" must not be requestable -- it would shadow the builtin).
        # `dosage`/custom-fmt fields ARE wired through `next_batch_variants`'s
        # `info_out` (unlike the `_info` INDEX columns above), so both
        # `available_var_fields` and `servable_var_fields` gain them here.
        #
        # Minor 3 (PR-B3b review, #304): discovery here is name+dtype ONLY -- no
        # `np.memmap` open, no CSR-length check. A .svar with a stale/truncated
        # per-call field file (or one simply too small for its registered dtype,
        # which `np.memmap` itself would raise on) must still OPEN for
        # haplotype-only streaming, or a `var_fields` request that never touches
        # this field -- mirroring the written path (`_haps.py`'s `Haps` only
        # memmaps a var_field file when `name in var_fields`). The actual memmap +
        # length guard moves to `build_engine`, which only ever sees fields the
        # caller actually requested.
        self._call_field_dtypes: dict[str, np.dtype] = {}
        if (Path(svar_path) / "dosages.npy").exists():
            self._call_field_dtypes["dosage"] = np.dtype(DOSAGE_TYPE)
        for _name, _dt in _svar_format_fields(Path(svar_path)).items():
            if _name in _RESERVED_VAR_FIELD_NAMES or _name in self._call_field_dtypes:
                continue
            self._call_field_dtypes[_name] = np.dtype(_dt)

        # Important 1 (PR-B3b review, #304): a field is only SERVABLE if its dtype
        # can cross the FFI without a dtype-breaking cast (see
        # `_SUPPORTED_CALL_FIELD_DTYPES`'s doc comment) -- e.g. a custom `mutcat`
        # FORMAT field (`int16`) is available (schema-visible) but would need this
        # gate to keep it out of `servable_var_fields` if it were ever registered
        # with an unsupported dtype; every dtype `_svar_format_fields`/`DOSAGE_TYPE`
        # actually produce today (`int16`/`float32`) IS supported, so this is a
        # forward-looking guard, not a currently-exercised gap.
        self.available_var_fields = [
            *self.available_var_fields,
            *[f for f in self._call_field_dtypes if f not in self.available_var_fields],
        ]
        self.servable_var_fields = [
            *self.servable_var_fields,
            *[
                f
                for f, dt in self._call_field_dtypes.items()
                if dt in _SUPPORTED_CALL_FIELD_DTYPES
                and f not in self.servable_var_fields
            ],
        ]

        self._svar_path = str(svar_path)
        self._store = Svar1Store(str(svar_path), self.n_samples, self.ploidy)
        # Issue #279 Task 7: lazily-opened memmap of `variant_idxs.npy`, exposed
        # via the `geno_v_idxs` property below. `None` until first accessed --
        # a haplotype-only stream never touches it.
        self._geno_v_idxs: NDArray | None = None

        # Per contig: register three scalars and cache the contig-local u32 arrays the
        # range search borrows. The arrays stay HERE (numpy) and cross per call as
        # zero-copy PyReadonlyArray1 -- nothing variant-scale is duplicated into Rust.
        # (The old skeleton pushed the whole POS/REF/ALT table across as Python lists
        # via .tolist() -- ~10M int objects for a human chr1 -- purely to feed
        # Svar1RecordSource's constructor. No record source, no table.)
        chrom = idx["CHROM"].cast(pl.Utf8).to_numpy()
        # v_end = POS_1based - min(ILEN, 0); genoray's `_var_end_expr()` convention
        # (genoray/_var_ranges.py) -- and what `_write.py`'s `v_ends` uses too, via
        # the SAME raw `idx["POS"]` column (1-based; `_canonicalize_variant_table`
        # never touches POS, so pre/post-canonicalization values are identical).
        # MUST be the raw 1-based POS, NOT `v_starts` (already `-1`'d to 0-based by
        # `_variant_arrays_from_table(one_based=True)`) -- subtracting the already-
        # decremented start silently produces a 0-length exclusive end for every SNP
        # (`v_end == v_start` instead of `v_start + 1`), which drops the variant
        # whenever a query's lower bound lands exactly on it. NOT the kernel's
        # `v_start - min(ilen,0) + 1` either -- that `+1` lives inside
        # get_diffs_sparse and is a different convention.
        v_ends_all = (idx["POS"].to_numpy() - np.minimum(ilens, 0)).astype(np.uint32)
        self._contig_arrays: dict[
            str, tuple[NDArray[np.uint32], NDArray[np.uint32]]
        ] = {}
        # Parallel cache of the three scalars also passed to `set_contig_meta`, so
        # `build_engine` (Step 2) can assemble the engine's per-contig arrays later
        # without re-deriving them from the index table.
        self._contig_meta: dict[str, tuple[int, int, int]] = {}

        for c in self._contigs:
            mask = chrom == c
            n_local = int(mask.sum())
            if n_local == 0:
                self._store.set_contig_meta(c, 0, 0, 0)
                self._contig_arrays[c] = (
                    np.empty(0, np.uint32),
                    np.empty(0, np.uint32),
                )
                self._contig_meta[c] = (0, 0, 0)
                continue

            first = int(np.argmax(mask))
            # The per-contig slices below assume this contig's rows are one CONTIGUOUS
            # block starting at `first`. True for a SparseVar built from a
            # position-sorted VCF; if violated the failure mode is a silently WRONG
            # per-contig table -- parity breaks with no error. Fail fast instead.
            if not mask[first : first + n_local].all():
                raise ValueError(
                    f"SVAR index rows for contig {c!r} are not contiguous; "
                    "the streaming SVAR1 backend requires a position-sorted store."
                )

            vs_c = np.ascontiguousarray(v_starts[first : first + n_local], np.uint32)
            ve_c = np.ascontiguousarray(v_ends_all[first : first + n_local], np.uint32)
            # genoray's `var_ranges` binary-searches a `SearchTree` built over `vs_c`
            # and documents its input as ascending -- but enforces nothing beyond a
            # length `debug_assert`. A non-ascending POS within this contig (e.g. a
            # VCF sorted by contig but not by position) passes the contiguity check
            # above and then yields silently WRONG variant ranges with no error --
            # truncated haplotypes, no exception. Fail fast instead, same as above.
            if n_local > 1 and not (np.diff(vs_c.astype(np.int64)) >= 0).all():
                raise ValueError(
                    f"SVAR index POS for contig {c!r} is not ascending; "
                    "the streaming SVAR1 backend requires a position-sorted store."
                )
            # Python's var_ranges convention: max(v_ends - v_starts). Exactly 1 larger
            # than search::overlap_range's `>=` bound -- an OVER-estimate, which only
            # widens the candidate window and is provably overshoot-safe. Do not
            # subtract 1; UNDER-estimating would be a correctness bug.
            max_v_len = int((ve_c.astype(np.int64) - vs_c.astype(np.int64)).max())
            contig_start = int(idx["index"][first])

            self._store.set_contig_meta(c, contig_start, n_local, max_v_len)
            self._contig_arrays[c] = (vs_c, ve_c)
            self._contig_meta[c] = (contig_start, n_local, max_v_len)

    @property
    def has_cached_af(self) -> bool:
        """Whether this store has cached per-variant AF (Wave B PR-B2, #317).

        True once `SparseVar.cache_afs()` has been run, which is when
        `min_af`/`max_af` filtering becomes available on this backend.
        """
        return self._afs is not None

    def build_engine(
        self,
        jobs: list[tuple[int, NDArray[np.uint32], NDArray[np.uint32], int, int]],
        batch_size: int,
        output_length: int,
        annotated: bool = False,
        variants: bool = False,
        min_af: "float | None" = None,
        max_af: "float | None" = None,
        var_fields: "list[str] | None" = None,
        var_window: "tuple[NDArray, np.dtype, VarWindowOpt] | None" = None,
    ) -> object:
        """Construct a `Svar1StreamEngine` (Rust producer/consumer engine, #283).

        The engine overlaps window I/O with batch generation. `jobs` is one entry
        per WINDOW,
        `(contig_idx, region_starts, region_ends, s_lo, s_hi)`, in the SAME order
        `_iter_batches` will drive `.next_batch()`. `output_length` is `-1` for ragged
        (per-hap actual length, pre-Wave-A behavior) or a fixed length >= 1 (issue #277
        Wave A `with_len`); forwarded straight to the engine constructor's trailing
        `output_length` parameter. `annotated` (issue #277 Wave A Task 4) selects
        whether the engine computes `annot_v_idxs`/`annot_ref_pos` -- SVAR1's
        `geno_v_idxs` is already dataset-global, so the Rust side passes `None` for
        the per-variant global-id gather (see `stream_engine.rs`'s
        `Svar1Backend.annotated` doc comment).

        Cohort-independent job residency (issue #284 / final-review Finding 1): the
        full public->physical sample map `self._phys_sample_idx` crosses ONCE (length
        `n_samples`); each job carries only its contiguous physical-sample sub-range
        `[s_lo, s_hi)` (two ints), NOT a per-window copy of that window's physical
        samples. `_plan` always yields a contiguous `arange(s_lo, s_hi)` sample chunk,
        so the engine's producer reconstructs the window's physical samples on the fly
        as `phys_sample_idx[s_lo..s_hi]`. Total job metadata is region-scale
        (`O(n_windows * window_regions)`), never `O(n_windows * n_samples)`.

        `variants` (Wave B PR-B1, #304) selects whether the engine's `next_batch_variants`
        is usable: it forwards straight to the `Svar1StreamEngine` constructor's trailing
        `variants` parameter, which gates `Svar1Backend::generate_variants` (flat
        `alt`/`start`/`ilen` variant buffers via the shared `assemble_variants_window`
        helper) the same way `annotated` gates `next_batch_annotated`.

        `min_af`/`max_af` (Wave B PR-B2, #317) forward straight to the engine
        constructor's trailing `afs`/`min_af`/`max_af` parameters. `afs` (this
        backend's cached per-variant AF, `self._afs`) is only passed when filtering
        is actually requested (`min_af is not None or max_af is not None`) -- a
        no-filter job stays zero-cost, matching the `afs=None` no-op fast path
        documented on the Rust side (`stream_engine.rs`).

        `var_fields` (Wave B PR-B3a, #304) selects whether `ref` is requested:
        `want_ref = "ref" in var_fields` (default `["alt", "ilen", "start"]` when
        `None`) forwards to the engine constructor's trailing `want_ref` parameter,
        which gates whether `generate_variants` gathers `ref_data`/`ref_seq_offsets`
        (`stream_engine.rs`). A numeric INDEX column (e.g. a cached `AF`) is NOT
        forwarded here -- `Svar1Backend::generate_variants` has no general
        INFO/index-column gather yet; such a field is deferred follow-up work, and
        the caller (`_iter_batches`) raises a clear error rather than silently
        omitting it.

        `var_fields` also selects which per-call FORMAT fields (`dosage` and any
        genoray custom Number=G FORMAT column, `self._call_field_dtypes`) are
        forwarded (Wave B PR-B3b, #304): only the REQUESTED ones are memmapped and
        cross into Rust, split by EXACT dtype into the constructor's
        `call_field_{f32,i32,i16}[_names]` parameters -- an unrequested field costs
        no extra allocation/work, matching the `want_ref`/`afs=None` no-op-fast-path
        convention above. Each is a full copy into Rust residency
        (`PyReadonlyArray1::to_owned()`, the same pattern already used for
        `v_starts`/`ilens`/`alt_alleles`), unlike `variant_idxs.npy` itself (which
        stays a zero-copy mmap inside `Svar1Store`) -- per-call fields are CSR-scale
        (one entry per genotype call, not per variant), so this is a real cost at
        whole-cohort scale; a follow-up could open them as their own Rust-side mmap
        instead, mirroring `Svar1Store`'s `variant_idxs.npy` handling.

        Minor 3 (PR-B3b review, #304): the per-field memmap open AND the CSR-length
        guard both happen HERE, not at `__init__` construction time -- so a store
        with a stale/truncated per-call field file still opens fine for
        haplotype-only streaming (or any `var_fields` that never names the bad
        field); the failure only surfaces when the field is actually requested,
        matching the written path's lazy-load contract (`_haps.py`'s `Haps` only
        memmaps a var_field file `if name in var_fields`).

        `var_window` (Wave B PR-B4, #304) is the `(lut, lut_dtype, opt)` bundle
        `StreamingDataset.with_seqs("variant-windows", opt)` cached -- `None` (the
        default) disables `next_batch_variant_windows` exactly like every other
        trailing no-op default above. When set, `_win_mode_kwargs` translates it into
        the engine constructor's `win_ref_mode`/`win_alt_mode`/`win_flank_len`/
        `win_token_lut_{u8,i32}` keyword arguments.
        """
        from ..genvarloader import Svar1StreamEngine

        _active_fields = _normalize_var_fields(var_fields)
        want_ref = "ref" in _active_fields
        # Wave B PR-B3b (#304): only cross the per-call fields actually requested --
        # `self._call_field_dtypes` may hold every discovered field's name/dtype
        # (dosage + every custom FORMAT column), but forwarding one the caller never
        # asked for would be wasted allocation/work for no benefit. Split by EXACT
        # dtype (Important 1, PR-B3b review) -- NOT dtype kind -- since a dtype that
        # can't be losslessly represented in one of the engine's three buckets is
        # already excluded from `servable_var_fields` (see
        # `_SUPPORTED_CALL_FIELD_DTYPES`), so `with_settings` rejects it long before
        # `build_engine` ever runs; reaching the `else` branch below is therefore an
        # internal-consistency bug, not a data condition.
        call_field_f32_names: list[str] = []
        call_field_f32: list[NDArray[np.float32]] = []
        call_field_i32_names: list[str] = []
        call_field_i32: list[NDArray[np.int32]] = []
        call_field_i16_names: list[str] = []
        call_field_i16: list[NDArray[np.int16]] = []
        _n_calls: int | None = None
        for _name in _active_fields:
            _dt = self._call_field_dtypes.get(_name)
            if _dt is None:
                continue
            if _n_calls is None:
                from genoray._types import V_IDX_TYPE

                _n_calls = (
                    Path(self._svar_path) / "variant_idxs.npy"
                ).stat().st_size // np.dtype(V_IDX_TYPE).itemsize
            _path = Path(self._svar_path) / (
                "dosages.npy" if _name == "dosage" else f"{_name}.npy"
            )
            _arr = np.memmap(_path, dtype=_dt, mode="r")
            # Guard against a stale/corrupt field file silently misaligning with the
            # store's CSR (issue #304 review): the field must have exactly one entry
            # per `variant_idxs.npy` entry, or indexing it by CSR position `o` in the
            # Rust walk would silently read garbage/out-of-range values instead of
            # raising.
            if _arr.shape[0] != _n_calls:
                raise ValueError(
                    f"SVAR1 per-call field {_name!r} at {self._svar_path} has "
                    f"{_arr.shape[0]} entries but the store's variant_idxs CSR has "
                    f"{_n_calls}; the field file is stale or corrupt."
                )
            if _dt == np.dtype(np.float32):
                call_field_f32_names.append(_name)
                call_field_f32.append(np.ascontiguousarray(_arr, np.float32))
            elif _dt == np.dtype(np.int32):
                call_field_i32_names.append(_name)
                call_field_i32.append(np.ascontiguousarray(_arr, np.int32))
            elif _dt == np.dtype(np.int16):
                call_field_i16_names.append(_name)
                call_field_i16.append(np.ascontiguousarray(_arr, np.int16))
            else:  # pragma: no cover -- unreachable, see docstring/comment above
                raise AssertionError(
                    f"internal error: per-call field {_name!r} with unsupported "
                    f"dtype {_dt} reached build_engine despite not being servable"
                )

        contig_names = list(self._contigs)
        contig_starts: list[int] = []
        n_locals: list[int] = []
        max_v_lens: list[int] = []
        v_starts_c: list[NDArray[np.uint32]] = []
        v_ends_c: list[NDArray[np.uint32]] = []
        contig_ref_bytes: list[NDArray[np.uint8]] = []
        # Materialize reference bytes ONLY for contigs some job touches (#307): the
        # engine indexes `contig_refs[job.contig_idx]`, so untouched contigs are never
        # read. Materializing every store contig would pull the whole reference into
        # Python (and again into the Rust engine), breaking the cohort-independent
        # bounded-memory story on the reference axis for whole-genome references.
        # Untouched contigs get an empty placeholder to keep the per-contig arrays
        # index-aligned (the engine requires equal per-contig lengths).
        touched_contigs = {int(j[0]) for j in jobs}
        for i, c in enumerate(contig_names):
            cs, nl, mv = self._contig_meta[c]
            vs_c, ve_c = self._contig_arrays[c]
            contig_starts.append(cs)
            n_locals.append(nl)
            max_v_lens.append(mv)
            v_starts_c.append(vs_c)
            v_ends_c.append(ve_c)
            if i in touched_contigs:
                ref_bytes_i, _ref_off = self._ref._contig_slice(i)
            else:
                ref_bytes_i = np.empty(0, np.uint8)
            contig_ref_bytes.append(ref_bytes_i)

        job_contig_idx = [int(j[0]) for j in jobs]
        job_region_starts = [np.ascontiguousarray(j[1], np.uint32) for j in jobs]
        job_region_ends = [np.ascontiguousarray(j[2], np.uint32) for j in jobs]
        job_s_lo = [int(j[3]) for j in jobs]
        job_s_hi = [int(j[4]) for j in jobs]
        # The full public->physical sample map, crossed ONCE (n_samples-scale). Each
        # job's (s_lo, s_hi) slices into this on the producer thread -- no per-window copy.
        phys_sample_idx = self._phys_sample_idx.astype(np.int64, copy=False).tolist()

        return Svar1StreamEngine(
            self._svar_path,
            self.n_samples,
            self.ploidy,
            contig_names,
            contig_starts,
            n_locals,
            max_v_lens,
            v_starts_c,
            v_ends_c,
            contig_ref_bytes,
            phys_sample_idx,
            job_contig_idx,
            job_region_starts,
            job_region_ends,
            job_s_lo,
            job_s_hi,
            self._v_starts,
            self._ilens,
            self._alt_alleles,
            self._alt_offsets,
            self._ref_alleles,
            self._ref_offsets,
            self._ref.pad_char,
            True,
            batch_size,
            output_length,
            annotated,
            variants,
            afs=(self._afs if (min_af is not None or max_af is not None) else None),
            min_af=min_af,
            max_af=max_af,
            want_ref=want_ref,
            call_field_f32_names=call_field_f32_names,
            call_field_f32=call_field_f32,
            call_field_i32_names=call_field_i32_names,
            call_field_i32=call_field_i32,
            call_field_i16_names=call_field_i16_names,
            call_field_i16=call_field_i16,
            **_win_mode_kwargs(var_window),
        )

    @property
    def geno_v_idxs(self) -> NDArray:
        """The store's `variant_idxs` as a read-only memmap.

        The Rust `Svar1Store` holds this as a zero-copy mmap for its own reads;
        the fused track kernel needs it on the Python side, so open a second
        read-only view rather than copying. Opened lazily: a haplotype-only
        stream never touches it.

        Returns:
            A 1-D read-only `np.memmap` of the store's variant indices.
        """
        if self._geno_v_idxs is None:
            from genoray._types import V_IDX_TYPE

            self._geno_v_idxs = np.memmap(
                Path(self._svar_path) / "variant_idxs.npy",
                dtype=V_IDX_TYPE,
                mode="r",
            )
        return self._geno_v_idxs

    def read_window(
        self, r_idx: NDArray[np.intp], s_idx: NDArray[np.intp]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Read one window's CSR offsets, single-contig.

        Covers every region in `r_idx` x every sample in `s_idx`. Returns
        (o_starts, o_stops), each
        `len(r_idx) * len(s_idx) * ploidy`, C-order (region, sample, ploid) -- absolute
        indices into the store's variant_idxs mmap. No haplotypes are generated here.
        """
        from ..genvarloader import svar1_read_window

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)

        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError(
                "_Svar1Backend.read_window: window spans multiple contigs; "
                "every Rust call must be single-contig."
            )
        contig_name = self._contigs[contig_idx]
        vs_c, ve_c = self._contig_arrays[contig_name]
        region_bounds = np.ascontiguousarray(self._regions[r_idx, 1:3], np.int32)

        # `s_idx` is a PUBLIC index (sorted-name order, matching `gvl.Dataset`);
        # the store's genotype CSR is laid out in native (VCF column) order, so
        # translate before crossing into Rust. See `__init__`'s comment on
        # `_phys_sample_idx`. Output row order is unaffected -- only which
        # physical column each row reads from changes.
        phys_s_idx = self._phys_sample_idx[s_idx]

        o_starts, o_stops = svar1_read_window(
            self._store,
            contig_name,
            vs_c,
            ve_c,
            region_bounds,
            np.ascontiguousarray(phys_s_idx, np.int64),
        )
        return np.asarray(o_starts, np.int64), np.asarray(o_stops, np.int64)

    def mixed_realign_window(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        t_starts: NDArray[np.int32],
        t_ends: NDArray[np.int32],
        row_starts: NDArray[np.int32],
        row_ends: NDArray[np.int32],
    ) -> tuple["_MixedRealign", NDArray[np.int32]]:
        """One window's fused-kernel realign state + deletion-extended track ends.

        Moved verbatim out of the `to_iter` drive (issue #375): the drive used
        to reach into `_Svar1Backend`-only attributes directly, which is what
        made the mixed path SVAR1-only. This is the backend HALF of the
        `_MixedRealign` seam: a conforming backend must produce a
        correctly-shaped state object plus extended ends from exactly this
        signature -- see `_MixedTracksBackend` for the checked contract. All
        coordinate arrays below are 0-based, half-open reference coordinates
        unless noted.

        `t_ends` is extended per REGION by that region's max deletion length
        across the window's samples, because `_regions` is the raw BED while
        the written path's intervals were extracted over the post-widening
        `gvl_bed`. The formula (`row_lengths - diffs.clip(max=0).min(1)`) is
        the read path's own, verbatim, from `_realigned_tracks_from_intervals`
        and the WRITTEN reader (`_reconstruct.py:191`, `:382`); taking the
        per-region max over the window's samples makes the query a superset of
        every row's buffer, so it can never under-read.

        It deliberately does NOT match the written WRITER's `chromEnd`
        extension (`_write.py:1084`), which stores less than its own reader
        asks for. That asymmetry can only leave the written path with zeros in
        a tail streaming fills with real values, and is unobservable today
        because ragged re-alignment never reads past reference index
        `region_len - 1` and `with_len(L)` is bounded by the minimum region
        length on both sides. An `extend_to_length` follow-up must revisit both.

        Args:
            r_idx: `(n_regions,)` indices into `self._regions` -- region SORT
                order, the same space the drive iterates in. NOT the public,
                user-facing region index; the drive separately maps to that
                via `self._sort_order[r_idx]` when it needs it.
            s_idx: `(n_samples,)` PUBLIC (sorted-name) sample indices, the
                same space `gvl.Dataset`'s `s` means. A backend whose native
                storage uses a different sample order (e.g. VCF column order)
                MUST translate internally -- the way this implementation's
                `read_window`/`_phys_sample_idx` do -- the caller never
                performs that translation itself.
            t_starts: `(n_regions,)` int32, one un-extended track query start
                per region. Present for contract symmetry with `t_ends` and
                for a future backend that may need it; this implementation
                does not read it (the extension below is end-only).
            t_ends: `(n_regions,)` int32, one un-extended track query end per
                region -- the value this method extends.
            row_starts: `(n_regions * n_samples,)` int32, one track-row start
                per (region, sample) pair, REGION-MAJOR SAMPLE-MINOR: row
                `i * n_samples + j` is `(region r_idx[i], sample s_idx[j])`.
                This is `np.repeat(t_starts, n_samples)` and MUST stay in
                that order -- it is the same layout `geno_offset_idx`/
                `flat_r`/`flat_s` use elsewhere in the drive, and this
                method's own `region_max_del` reshape (`len(r_idx),
                len(s_idx)`) depends on it.
            row_ends: `(n_regions * n_samples,)` int32, same
                region-major/sample-minor layout as `row_starts`, one row end
                per (region, sample) pair.

        Returns:
            A 2-tuple `(realign_window, t_ends_ext)`, in that exact order:

            - `realign_window`: a `_MixedRealign` (here, a `_RealignWindow`)
              bundling this window's re-align state. `diffs` and
              `geno_offset_idx` are `(n_regions * n_samples, ploidy)`,
              region-major/sample-minor like `row_starts` with ploidy as the
              FAST axis (haplotype `h` of row `i` is `geno_offset_idx[i,
              h]`, flat index `i * ploidy + h`). `geno_offsets` is `(2,
              n_regions * n_samples * ploidy)` and WINDOW-LOCAL (indices
              into this window's own CSR, not the dataset-global store).
              `geno_v_idxs`/`v_starts`/`ilens` are the per-variant tables
              the kernel indexes through `geno_offset_idx`; for this backend
              they are the store's dataset-GLOBAL memmaps, but a record
              backend (VCF/PGEN) would hand back a window-LOCAL decoded
              table instead (see `_RealignWindow`'s docstring) -- either
              way the caller only ever reaches them through
              `geno_offset_idx`, never as absolute dataset-global variant
              indices.
            - `t_ends_ext`: `(n_regions,)` int32, `t_ends` extended by each
              region's max deletion length over `s_idx` -- one value PER
              REGION, not per row. Feed this (not `t_ends`) to the track
              backend's `read_window` so the interval query covers every
              sample's realigned tail.

            This computation assumes jitter is OFF for `_realign_tracks=True`
            streams (the caller's `jitter>0` guard just above raises before
            this method would ever be called with jittered bounds): an
            un-translated extension over jittered `t_starts`/`t_ends` would
            under-size the query relative to what the haplotype engine used.
        """
        from ._genotypes import get_diffs_sparse

        n_rows = len(row_starts)
        P = self.ploidy
        n_s = len(s_idx)
        o_starts, o_stops = self.read_window(r_idx, s_idx)
        geno_offsets_w = np.stack([o_starts, o_stops])
        geno_offset_idx_w = np.arange(n_rows * P, dtype=np.intp).reshape(n_rows, P)
        diffs_w = get_diffs_sparse(
            geno_offset_idx_w,
            self.geno_v_idxs,
            geno_offsets_w,
            self._ilens,
            q_starts=row_starts,
            q_ends=row_ends,
            v_starts=self._v_starts,
        )
        max_del_row = -diffs_w.clip(max=0).min(1)
        region_max_del = max_del_row.reshape(len(r_idx), n_s).max(1)
        t_ends_ext = np.ascontiguousarray(
            t_ends.astype(np.int64) + region_max_del, np.int32
        )
        return (
            _RealignWindow(
                diffs=diffs_w,
                geno_offsets=geno_offsets_w,
                geno_offset_idx=geno_offset_idx_w,
                geno_v_idxs=self.geno_v_idxs,
                v_starts=self._v_starts,
                ilens=self._ilens,
            ),
            t_ends_ext,
        )

    def generate_batch(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        o_starts: NDArray[np.int64],
        o_stops: NDArray[np.int64],
        lo: int,
        hi: int,
        output_length: int,
    ) -> Ragged:
        """Generate haplotypes for window rows [lo:hi] (C-order (region, sample)).

        Output is (hi-lo)-bounded -- NEVER the whole window (issue #284). `o_starts`/
        `o_stops` are the whole window's offsets (from `read_window`); this slices the
        CSR rows [lo*ploidy : hi*ploidy] and the matching per-row region bounds.
        `output_length` is `-1` for ragged or a fixed length >= 1 (issue #277 Wave A),
        forwarded straight to `svar1_generate_batch`.
        """
        from ..genvarloader import svar1_generate_batch

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)
        n_s = len(s_idx)
        # `contig_idx` already indexes `self._contigs`, and `Reference.from_path`
        # (called with `self._contigs` in `__init__`) builds `offsets` in that same
        # order -- so `contig_idx` indexes `self._ref` directly, no name lookup
        # needed. Previously this did `self._ref.c_map.contigs.index(contig_name)`,
        # which is both redundant (same answer as `contig_idx`) AND a bug:
        # `Reference.from_path` normalizes contig names to the FASTA's naming style
        # (UCSC "chr1" vs Ensembl "1"), so a store using one style paired with a
        # FASTA in the other style made `contig_name` absent from
        # `c_map.contigs`, raising `ValueError`. See `Reference._contig_slice`'s
        # docstring and `Svar2Haps._ref_for_contig` for the shared convention.
        contig_idx = int(self._regions[r_idx[0], 0])
        ref_bytes, ref_offsets = self._ref._contig_slice(contig_idx)

        # Per (region, sample) row bounds for rows [lo:hi], C-order (region, sample):
        # window row bi = ri*n_s + si -> region r_idx[bi // n_s].
        rows = np.arange(lo, hi)
        region_bounds_b = np.ascontiguousarray(
            self._regions[r_idx[rows // n_s], 1:3], np.int32
        )
        o_lo, o_hi = lo * self.ploidy, hi * self.ploidy

        data, offsets = svar1_generate_batch(
            self._store,
            np.ascontiguousarray(o_starts[o_lo:o_hi], np.int64),
            np.ascontiguousarray(o_stops[o_lo:o_hi], np.int64),
            region_bounds_b,
            self._v_starts,
            self._ilens,
            self._alt_alleles,
            self._alt_offsets,
            ref_bytes,
            ref_offsets,
            self._ref.pad_char,
            output_length,
            True,
        )
        n_rows = hi - lo
        return Ragged.from_offsets(
            data.view("S1"), (n_rows, self.ploidy, None), np.asarray(offsets, np.int64)
        )


class _Svar2Backend:
    """StreamingDataset backend for .svar2 stores.

    Mirrors _Svar1Backend, but per-window variant ranges are computed LIVE via the
    GIL-free Rust `svar2_read_window`
    (genoray_core::query::find_ranges, the same query gvl.write uses at write time,
    `_write.py:_write_from_svar2`) instead of slicing an on-disk `_Svar2Cache`; the
    ranges feed a recycled `Svar2ReconBuf` reconstructed in coarse super-batches
    (`svar2_reconstruct_super_batch`, `_fill_super_batch`/`_drain`) so the fill is
    multi-core (Phase 2). Still "sync" `_iter_batches` strategy -- no engine, no
    readahead.
    """

    #: Mixed variants+tracks not wired for this backend yet (issue #375).
    supports_mixed_tracks: ClassVar[bool] = False

    _default_strategy = "sync"

    def __init__(
        self,
        svar2_path: str | Path,
        reference_path: str | Path,
        contigs: list[str],
        bed: pl.DataFrame | str | Path,
    ) -> None:
        from genoray import SparseVar2

        from ..genvarloader import Svar2Store
        from ._reference import Reference

        self._sv = SparseVar2(str(svar2_path))
        # Same sorted-name convention `_Svar1Backend` uses: `gvl.write` always
        # lexicographically sorts sample names, so `sample_idx` means "the i-th
        # name in sorted order" -- not the store's native (VCF column) order.
        native = list(self._sv.available_samples)
        self._sample_names = sorted(native)
        # Public sorted-name position -> physical store (VCF) column. genoray's Rust
        # find_ranges takes physical usize indices (the Python _find_ranges resolved
        # names internally); we translate here so the Rust read_window gets columns.
        _col_of = {name: i for i, name in enumerate(native)}
        self._phys_sample_idx = np.array(
            [_col_of[n] for n in self._sample_names], dtype=np.int64
        )
        self.n_samples = len(self._sample_names)
        self.ploidy = int(self._sv.ploidy)
        # Resident window buffer holds vk_snp_range + vk_indel_range (i64 pairs) per
        # (region, sample, ploid): 2 arrays x 2 i64 = 32 B/cell. STARTING ESTIMATE
        # (validate in Task 4); gathered flat channels add bounded window-scale extra.
        self._cell_bytes = self.ploidy * 32
        self._contigs = list(contigs)
        # `Svar2Store` opens one query-only reader per contig; sized by the STORE's
        # full sample count (mirrors `Svar2Haps`'s construction at `_svar2_haps.py:271`),
        # not this backend's (identical, since streaming never subsets samples).
        self._store = Svar2Store(
            str(svar2_path), self._sv.contigs, self._sv.n_samples, self.ploidy
        )
        self._ref = Reference.from_path(reference_path, self._contigs)
        # Identical region-bounds derivation to _Svar1Backend (and StreamingDataset):
        # r_idx from the plan indexes THIS sorted regions table. Reuse the same
        # imports _Svar1Backend uses (bed_to_regions, ContigNormalizer, sp.bed).
        bed_df = bed if isinstance(bed, pl.DataFrame) else sp.bed.read(bed)
        self._regions = bed_to_regions(
            sp.bed.sort(bed_df), ContigNormalizer(self._contigs)
        )
        # Self-contained default (see `SUPERBATCH_TARGET_ROWS`'s module comment): the
        # backend is constructed before `StreamingDataset.__init__` computes
        # `max_mem_bytes`, so it can't size against the real budget yet.
        # `StreamingDataset.__init__` refines this down once `max_mem_bytes` is known.
        # `_iter_batches`' super-batch drive must READ this attribute, never
        # recompute it, so later overrides (tests, Task 5's sweep) stick.
        self._super_batch_rows = SUPERBATCH_TARGET_ROWS
        # Wave B PR-B3a (#304): `with_seqs("variants")` (and thus `var_fields`) is not
        # yet wired for `.svar2` (see `_iter_batches`'s SVAR2 guard, which raises
        # `NotImplementedError` before any of this would matter) -- this is just the
        # builtin-default placeholder so `StreamingDataset.available_var_fields` has
        # something to read from every backend type.
        self.available_var_fields = list(_DEFAULT_VAR_FIELDS)
        # Important 1 (Wave B PR-B3a review): placeholder, same as `available_var_fields`
        # above -- `with_seqs("variants")` is guarded to raise before this would matter.
        self.servable_var_fields = list(self.available_var_fields)

    def _contig_of(self, r_idx: NDArray[np.intp]) -> tuple[int, str]:
        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError("_Svar2Backend: window spans multiple contigs")
        return contig_idx, self._contigs[contig_idx]

    def build_engine(
        self,
        jobs: list[tuple[int, NDArray[np.uint32], NDArray[np.uint32], int, int]],
        batch_size: int,
    ) -> object:
        """Construct a `Svar2StreamEngine` (Rust producer/consumer engine, PR-3).

        The engine (Task 2) overlaps window read (`find_ranges`) with super-batch
        reconstruct. `jobs` is one entry per WINDOW, `(contig_idx, region_starts, region_ends,
        s_lo, s_hi)`, in the SAME order `_iter_batches` will drive `.next_batch()` --
        mirrors `_Svar1Backend.build_engine`'s cohort-independent job residency
        contract (issue #284): the full public->physical sample map crosses ONCE,
        each job carries only its contiguous `[s_lo, s_hi)` sub-range.
        """
        from ..genvarloader import Svar2StreamEngine

        contig_names = list(self._contigs)
        # Materialize reference bytes ONLY for contigs some job touches (#307): the
        # engine indexes `contig_refs[job.contig_idx]`, so untouched contigs are never
        # read. Empty placeholder keeps the list index-aligned with `contig_names`
        # (the engine requires equal lengths) without pulling the whole reference.
        touched_contigs = {int(j[0]) for j in jobs}
        contig_ref_bytes = [
            np.asarray(self._ref._contig_slice(i)[0], np.uint8).tobytes()
            if i in touched_contigs
            else b""
            for i in range(len(contig_names))
        ]
        job_contig_idx = [int(j[0]) for j in jobs]
        job_region_starts = [np.ascontiguousarray(j[1], np.uint32) for j in jobs]
        job_region_ends = [np.ascontiguousarray(j[2], np.uint32) for j in jobs]
        job_s_lo = [int(j[3]) for j in jobs]
        job_s_hi = [int(j[4]) for j in jobs]
        phys = self._phys_sample_idx.astype(np.int64, copy=False).tolist()
        return Svar2StreamEngine(
            str(self._sv.path),
            list(self._sv.contigs),
            int(self._sv.n_samples),
            self.ploidy,
            contig_names,
            contig_ref_bytes,
            phys,
            job_contig_idx,
            job_region_starts,
            job_region_ends,
            job_s_lo,
            job_s_hi,
            int(self._ref.pad_char),
            int(self._super_batch_rows),
            batch_size,
        )

    def read_window(
        self, r_idx: NDArray[np.intp], s_idx: NDArray[np.intp]
    ) -> dict[str, object]:
        """Compute the window's live ranges via GIL-free Rust `svar2_read_window`.

        Uses genoray_core::query::find_ranges, replacing the Python
        SparseVar2._find_ranges call + numpy glue. `s_idx` (public sorted-name
        order) is translated to physical store columns via `_phys_sample_idx`
        before crossing into Rust.
        """
        from ..genvarloader import svar2_read_window

        r_idx = np.asarray(r_idx, np.intp)
        s_idx = np.asarray(s_idx, np.intp)
        contig_idx, contig = self._contig_of(r_idx)
        rb = self._regions[r_idx, 1:3]  # (n_reg, 2) int
        starts = np.ascontiguousarray(rb[:, 0], np.uint32)
        ends = np.ascontiguousarray(rb[:, 1], np.uint32)
        phys = np.ascontiguousarray(self._phys_sample_idx[s_idx], np.int64)
        n_reg, n_s, P = len(r_idx), len(s_idx), self.ploidy
        vk_snp, vk_indel, dense_snp, dense_indel, sample_cols = svar2_read_window(
            self._store, contig, starts, ends, phys
        )
        return {
            "contig_idx": contig_idx,
            "region_bounds": np.ascontiguousarray(rb, np.int32),  # (n_reg, 2)
            "orig_samples": np.asarray(sample_cols, np.int64),  # (n_s,)
            "vk_snp": np.asarray(vk_snp, np.int64).reshape(n_reg, n_s, P, 2),
            "vk_indel": np.asarray(vk_indel, np.int64).reshape(n_reg, n_s, P, 2),
            "dense_snp": np.asarray(dense_snp, np.int64).reshape(n_reg, 2),
            "dense_indel": np.asarray(dense_indel, np.int64).reshape(n_reg, 2),
        }

    def _gather_rows(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        window: dict[str, object],
        lo: int,
        hi: int,
    ) -> tuple[
        NDArray[np.uint32],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.uint8],
        NDArray[np.int64],
    ]:
        """Gather the per-row FFI inputs for window rows [lo, hi).

        Rows are C-order (region, sample), mirroring
        `_svar2_haps.py:_gather_inputs`. Shared by the super-batch fill
        (`_fill_super_batch`, production) and the per-batch parity reference
        (`tests/dataset/test_streaming_phase2_pr2.py:_per_batch_reference`).
        """
        r_idx = np.asarray(r_idx, np.intp)
        n_s = len(np.asarray(s_idx))
        P = self.ploidy
        contig_idx = cast(int, window["contig_idx"])
        ref_, ref_offsets = self._ref._contig_slice(contig_idx)

        rows = np.arange(lo, hi)
        ri = rows // n_s  # region-in-window per row
        si = rows % n_s  # sample-in-window per row
        # Per-row FFI inputs (C-order (region, sample)), mirroring _gather_inputs:
        region_bounds = np.ascontiguousarray(
            cast("NDArray[np.int32]", window["region_bounds"])[ri], np.int32
        )  # (m,2)
        region_starts = np.ascontiguousarray(region_bounds[:, 0], np.uint32)  # (m,)
        orig_samples = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["orig_samples"])[si], np.int64
        )  # (m,)
        vk_snp = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["vk_snp"])[ri, si].reshape(-1, 2),
            np.int64,
        )  # (m*P,2)
        vk_indel = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["vk_indel"])[ri, si].reshape(-1, 2),
            np.int64,
        )
        dense_snp = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["dense_snp"])[ri], np.int64
        )  # (m,2)
        dense_indel = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["dense_indel"])[ri], np.int64
        )
        m = hi - lo
        shifts = np.zeros((m, P), np.int32)  # jitter out of scope (jitter=0)
        return (
            region_starts,
            orig_samples,
            vk_snp,
            vk_indel,
            dense_snp,
            dense_indel,
            region_bounds,
            shifts,
            ref_,
            ref_offsets,
        )

    def _fill_super_batch(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        window: dict[str, object],
        sb_lo: int,
        sb_hi: int,
        buf: object,
        parallel: bool,
    ) -> None:
        """Reconstruct C-order window rows [sb_lo, sb_hi) into the recycled buffer.

        Fills `Svar2ReconBuf` rather than returning it: the multi-core super-batch
        path, drained afterwards via `_drain`.
        """
        from ..genvarloader import svar2_reconstruct_super_batch

        (
            region_starts,
            orig_samples,
            vk_snp,
            vk_indel,
            dense_snp,
            dense_indel,
            region_bounds,
            shifts,
            ref_,
            ref_offsets,
        ) = self._gather_rows(r_idx, s_idx, window, sb_lo, sb_hi)
        contig_idx = cast(int, window["contig_idx"])
        contig = self._contigs[contig_idx]
        svar2_reconstruct_super_batch(
            self._store,
            contig,
            region_starts,
            orig_samples,
            vk_snp,
            vk_indel,
            dense_snp,
            dense_indel,
            region_bounds,
            shifts,
            ref_,
            ref_offsets,
            np.uint8(self._ref.pad_char),
            bool(parallel),
            buf,
        )

    def _drain(self, buf: object, lo: int, hi: int) -> Ragged:
        """Copy out rows [lo, hi) of the current `Svar2ReconBuf` fill as a `Ragged`."""
        data, offsets = buf.batch(int(lo), int(hi))  # pyright: ignore[reportAttributeAccessIssue]
        m = hi - lo
        return Ragged.from_offsets(
            np.asarray(data).view("S1"),
            (m, self.ploidy, None),
            np.asarray(offsets, np.int64),
        )

    def _est_out_bytes(self, r_idx: NDArray[np.intp], n_rows: int) -> int:
        """Estimate reconstructed super-batch bytes, ~ rows*ploidy*mean_region_width.

        This gates `should_parallelize` *before* the fill -- the fill IS the
        reconstruct, so the buffer's exact `total_bytes` is only known after. An
        overestimate is harmless (it only flips the parallel decision, never
        correctness).
        """
        r_idx = np.asarray(r_idx, np.intp)
        widths = self._regions[r_idx, 2] - self._regions[r_idx, 1]
        mean_width = int(max(1, widths.mean())) if len(widths) else 1
        return int(n_rows) * self.ploidy * mean_width


class _VcfBackend:
    """Streaming VCF read backend, driving a `RecordStreamEngine` over a live VCF.

    Reads a VCF/BCF directly (issue #276 tasks 3b/5), with no on-disk `.svar`
    store and no on-disk gvl dataset. Unlike `_Svar1Backend` there is no split
    read/generate seam (`read_window`/`generate_batch`) -- a VCF/BCF has no
    equivalent of SVAR1's precomputed CSR offsets to read ahead of generation,
    so this backend supports ONLY the "engine" prefetch strategy
    (`StreamingDataset._iter_batches`'s `"engine"` branch calls nothing but
    `build_engine` on the backend).

    Header metadata (sample names, ploidy, contigs) is read once at
    construction from `genoray.VCF(path)`; per-region variant records are
    decoded window-by-window by the Rust `VcfWindowFiller` inside the engine,
    not read/cached here.
    """

    #: Mixed variants+tracks not wired for this backend yet (issue #375).
    supports_mixed_tracks: ClassVar[bool] = False

    def __init__(
        self,
        vcf_path: str | Path,
        reference_path: str | Path,
        contigs: list[str] | None,
        bed: pl.DataFrame | str | Path,
    ) -> None:
        from genoray import VCF

        from ._reference import Reference

        self._vcf_path = str(vcf_path)

        vcf = VCF(self._vcf_path)
        # `gvl.write()` always lexicographically sorts sample names
        # (`_write.py`'s unconditional `samples.sort()`), so `gvl.Dataset`'s
        # sample index `s` means "the s-th name in sorted order" -- the same
        # convention `_Svar1Backend` follows (see its `__init__` comment). A
        # VCF/BCF has no separate "native order" concern for the streaming
        # engine the way SVAR1's on-disk genotype CSR does -- `RecordStreamEngine`
        # takes `sample_names` directly and looks samples up by name -- but the
        # PUBLIC sample_idx contract must still be sorted-name order to match
        # `gvl.Dataset[r, s]`.
        self._sample_names = sorted(vcf.available_samples)
        self.n_samples = len(self._sample_names)
        self.ploidy = vcf.ploidy

        # `contigs` is `None` unless the caller passed an explicit `contigs=`
        # to `StreamingDataset` -- unlike the `.svar` branch, which always
        # derives `contigs` from the store before constructing its backend,
        # the VCF branch defers to the VCF header (naturally sorted, via
        # `genoray.VCF.contigs`) when the caller didn't supply one.
        self._contigs = list(contigs) if contigs is not None else list(vcf.contigs)

        self._ref = Reference.from_path(reference_path, self._contigs)

        # Whether the source VCF header declares an INFO/AF field (Wave B
        # PR-B2, #319) -- the SAME condition `gvl.write` uses (Task 5) to cache
        # AF into the written `.gvi`, so streaming and written agree on AF
        # availability.
        self._has_cached_af = bool(vcf._declared_info_fields(("AF",)))

        # Wave B PR-B3a (#304): declared numeric INFO fields are requestable
        # `var_fields` -- non-numeric types (Flag/String/Character) are excluded,
        # matching the written path's numeric-only `available_info_fields` filter
        # (`_haps.py`). Recorded as name -> is_float so `build_engine` can pass the
        # `(name, is_float)` pairs the Rust `VcfWindowFiller` needs without
        # re-scanning the header per call.
        self._info_dtypes = _declared_info_numeric_dtypes(vcf)
        self.available_var_fields = [*_DEFAULT_VAR_FIELDS, *self._info_dtypes, "ref"]
        # Important 1 (Wave B PR-B3a review): VCF's Rust engine actually wires every
        # advertised field through (`info_fields`/`want_ref` in `build_engine` below),
        # so servable == available here -- unlike SVAR1 (see `_Svar1Backend.__init__`).
        self.servable_var_fields = list(self.available_var_fields)

        # `bed` is accepted for interface symmetry with `_Svar1Backend.__init__`
        # (the public ladder branch constructs both the same way) but unused
        # here: `build_engine`'s `jobs` already carry each window's
        # (contig_idx, region_starts, region_ends) directly from
        # `StreamingDataset._plan`/`_regions`, so this backend never needs its
        # own region table the way `_Svar1Backend` does for its readahead path.
        del bed

    @property
    def has_cached_af(self) -> bool:
        """Whether the source VCF header declares an INFO/AF field.

        Wave B PR-B2, #319 -- the SAME condition gvl.write uses to cache AF into
        the written .gvi, so streaming <-> written agree on AF availability.
        """
        return self._has_cached_af

    def build_engine(
        self,
        jobs: list[tuple[int, NDArray[np.uint32], NDArray[np.uint32], int, int]],
        batch_size: int,
        output_length: int,
        annotated: bool = False,
        variants: bool = False,
        min_af: "float | None" = None,  # forwarded to the engine -- see below (Wave B PR-B2, #317/#319)
        max_af: "float | None" = None,  # forwarded to the engine -- see below (Wave B PR-B2, #317/#319)
        var_fields: "list[str] | None" = None,
        var_window: "tuple[NDArray, np.dtype, VarWindowOpt] | None" = None,
    ) -> object:
        """Construct a `RecordStreamEngine("vcf", ...)` over the live VCF/BCF.

        The Rust producer/consumer engine (issue #276 tasks 3b/5) decodes each
        window's variant records straight from the VCF/BCF. `jobs` is one entry
        per WINDOW,
        `(contig_idx, region_starts, region_ends, s_lo, s_hi)`, in the SAME
        order `_iter_batches` will drive `.next_batch()` -- mirrors
        `_Svar1Backend.build_engine`'s job-array unpacking exactly, minus the
        SVAR1-only store/physical-sample-map arguments (a VCF job's
        `[s_lo, s_hi)` indexes straight into `sample_names`, no public->
        physical indirection). `output_length` is `-1` for ragged or a fixed
        length >= 1 (issue #277 Wave A `with_len`), forwarded straight to the
        engine constructor's trailing `output_length` parameter. `annotated`
        (issue #277 Wave A Task 4) selects whether the engine computes
        `annot_v_idxs`/`annot_ref_pos`, remapped window-local -> dataset-global
        per-variant via `slot.global_v_idxs` (genoray `DenseChunk.global_idx`,
        gathered by `generate_batch_core`). VCF's genoray ids are hard-coded
        `-1` until Phase 3, so the gather is a no-op and emitted ids stay
        window-local -- the known VCF gap (see `vcf.rs`'s `WindowFiller::fill`
        doc comment and GitHub issue #305). `min_af`/`max_af` are forwarded
        straight to the engine constructor's trailing bounds params; the Rust
        `#[new]` derives `want_af` from them (`min_af.is_some() ||
        max_af.is_some()`) so the `VcfWindowFiller` requests the AF
        `FieldSpec` exactly when filtering is active (Wave B PR-B2, #317/#319)
        -- callers are expected to have already checked `has_cached_af` (see
        `StreamingDataset._iter_batches`'s AF-missing guard).

        `var_fields` (Wave B PR-B3a, #304) selects `want_ref` (`"ref" in var_fields`,
        default `["alt", "ilen", "start"]` when `None`) and the non-builtin numeric
        INFO fields to gather, forwarded as `(name, is_float)` pairs (looked up in
        `self._info_dtypes`, populated at construction from the VCF header) to the
        engine constructor's trailing `info_fields`/`want_ref` parameters, which the
        Rust `VcfWindowFiller` stages as additional `FieldSpec`s alongside `AF`
        (`vcf.rs`).

        `var_window` (Wave B PR-B4, #304): see `_Svar1Backend.build_engine`'s
        matching parameter -- same `(lut, lut_dtype, opt)` bundle, same
        `_win_mode_kwargs` translation into `win_*` keyword arguments.
        """
        from ..genvarloader import RecordStreamEngine

        _active_fields = _normalize_var_fields(var_fields)
        want_ref = "ref" in _active_fields
        info_fields = [
            (name, self._info_dtypes[name])
            for name in _active_fields
            if name not in ("alt", "ilen", "start", "ref")
        ]

        contig_names = list(self._contigs)
        contig_ref_bytes = [
            self._ref._contig_slice(i)[0] for i in range(len(contig_names))
        ]

        job_contig_idx = [int(j[0]) for j in jobs]
        job_region_starts = [np.ascontiguousarray(j[1], np.uint32) for j in jobs]
        job_region_ends = [np.ascontiguousarray(j[2], np.uint32) for j in jobs]
        job_s_lo = [int(j[3]) for j in jobs]
        job_s_hi = [int(j[4]) for j in jobs]

        return RecordStreamEngine(
            "vcf",
            self._vcf_path,
            self._sample_names,
            self.ploidy,
            contig_names,
            contig_ref_bytes,
            job_contig_idx,
            job_region_starts,
            job_region_ends,
            job_s_lo,
            job_s_hi,
            # `fasta_path=None` -- PARITY-CRITICAL, not a placeholder. Task 4
            # established that `gvl.write` does NO read-time reference/left-
            # alignment for VCF input, so the streaming decoder must not
            # either, to stay byte-identical (see `src/record_stream/vcf.rs`'s
            # module doc on the `fasta_path: None` parity default). `self._ref`
            # above is used ONLY to derive `contig_ref_bytes` for haplotype
            # reconstruction padding -- it is NOT the decode-time FASTA, and
            # passing its path here would enable left-alignment in the Rust
            # decoder and silently diverge from the write path.
            None,
            self._ref.pad_char,
            True,
            batch_size,
            output_length,
            annotated,
            variants,
            min_af,
            max_af,
            info_fields=info_fields,
            want_ref=want_ref,
            **_win_mode_kwargs(var_window),
        )


class _PgenBackend:
    """Streaming PGEN read backend, driving a `RecordStreamEngine` over a file-set.

    Reads a live `.pgen`/`.pvar`/`.psam` file-set (issue #276 tasks 3b/11), with no
    on-disk gvl dataset. Mirrors `_VcfBackend` exactly (same duck interface,
    same "engine"-only prefetch restriction -- PGEN has no `read_window`/
    `generate_batch` split either); the only differences are (a) header
    metadata comes from `genoray.PGEN` instead of `genoray.VCF`, (b) ploidy is
    always 2 (PGEN is diploid-only by format -- `genoray.PGEN.ploidy` is a
    class attribute, not read from the file), and (c) `build_engine` passes
    `source_kind="pgen"` and the resolved `.pgen` path.

    Header metadata (sample names, contigs) is read once at construction from
    `genoray.PGEN(path)`; per-region variant records are decoded
    window-by-window by the Rust `PgenWindowFiller` inside the engine, not
    read/cached here.
    """

    #: Mixed variants+tracks not wired for this backend yet (issue #375).
    supports_mixed_tracks: ClassVar[bool] = False

    def __init__(
        self,
        pgen_path: str | Path,
        reference_path: str | Path,
        contigs: list[str] | None,
        bed: pl.DataFrame | str | Path,
    ) -> None:
        from genoray import PGEN

        from ._reference import Reference

        pgen = PGEN(pgen_path)
        # The resolved `.pgen` file path (`genoray.PGEN.__init__` appends the
        # `.pgen` suffix if the caller passed a bare plink2 prefix) -- the Rust
        # `PgenWindowFiller` derives its sibling `.pvar` via `with_extension`,
        # which requires the literal `.pgen` path, not a suffix-less prefix.
        self._pgen_path = str(pgen.geno_path)

        # `gvl.write()` always lexicographically sorts sample names
        # (`_write.py`'s unconditional `samples.sort()`), so `gvl.Dataset`'s
        # sample index `s` means "the s-th name in sorted order" -- the same
        # convention `_VcfBackend`/`_Svar1Backend` follow (see their `__init__`
        # comments). `RecordStreamEngine` takes these sorted names directly and
        # passes them to `PgenWindowFiller`, which reads the `.psam` and maps the
        # sorted-name order onto the physical `.psam` column order (the `.psam`
        # order plink2 preserves from the source VCF is arbitrary and need NOT be
        # sorted) -- the same public->physical concern SVAR1's on-disk CSR has,
        # handled Rust-side here (see `src/record_stream/pgen.rs`'s "Sample
        # subsetting" section) rather than via a Python `_phys_sample_idx`.
        self._sample_names = sorted(pgen.available_samples)
        self.n_samples = len(self._sample_names)
        # PGEN is diploid-only by format (no ploidy field on disk) -- matches
        # the Rust `PgenWindowFiller`'s hardwired `PGEN_PLOIDY = 2` (it takes
        # no ploidy parameter at all; `genoray.PGEN.ploidy` is a class
        # attribute, always 2, not read from the file either).
        self.ploidy = pgen.ploidy

        # `contigs` is `None` unless the caller passed an explicit `contigs=`
        # to `StreamingDataset` -- unlike the `.svar` branch, which always
        # derives `contigs` from the store before constructing its backend,
        # the PGEN branch defers to `genoray.PGEN.contigs` (naturally sorted)
        # when the caller didn't supply one, same as `_VcfBackend`.
        self._contigs = list(contigs) if contigs is not None else list(pgen.contigs)

        self._ref = Reference.from_path(reference_path, self._contigs)

        # Wave B PR-B3a (#304): PGEN has no INFO path (see the PR-B2 AF guard --
        # `has_cached_af` is always `False`); `ref` is the only non-builtin
        # `var_field` it can serve.
        self.available_var_fields = [*_DEFAULT_VAR_FIELDS, "ref"]
        # Important 1 (Wave B PR-B3a review): PGEN's only non-builtin available field
        # is `ref`, and it IS wired through `build_engine` below -- servable == available.
        self.servable_var_fields = list(self.available_var_fields)

        # `bed` is accepted for interface symmetry with `_Svar1Backend.__init__`
        # (the public ladder branch constructs every backend the same way) but
        # unused here, same as `_VcfBackend`: `build_engine`'s `jobs` already
        # carry each window's (contig_idx, region_starts, region_ends) directly
        # from `StreamingDataset._plan`/`_regions`.
        del bed

    @property
    def has_cached_af(self) -> bool:
        """PGEN record streams carry no INFO -> no AF (Wave B PR-B2, #319).

        AF filtering on PGEN is guarded upstream; always False.
        """
        return False

    def build_engine(
        self,
        jobs: list[tuple[int, NDArray[np.uint32], NDArray[np.uint32], int, int]],
        batch_size: int,
        output_length: int,
        annotated: bool = False,
        variants: bool = False,
        min_af: "float | None" = None,  # accepted for the shared call site; PGEN has no INFO/AF so this is a no-op (guarded upstream, see `has_cached_af`)
        max_af: "float | None" = None,  # accepted for the shared call site; PGEN has no INFO/AF so this is a no-op (guarded upstream, see `has_cached_af`)
        var_fields: "list[str] | None" = None,
        var_window: "tuple[NDArray, np.dtype, VarWindowOpt] | None" = None,
    ) -> object:
        """Construct a `RecordStreamEngine("pgen", ...)` over the live file-set.

        The Rust producer/consumer engine (issue #276 tasks 3b/11) decodes each
        window's variant records straight from the `.pgen`/`.pvar`/`.psam`
        file-set. `jobs` is
        one entry per WINDOW, `(contig_idx, region_starts, region_ends, s_lo,
        s_hi)`, in the SAME order `_iter_batches` will drive `.next_batch()` --
        mirrors `_VcfBackend.build_engine` exactly, minus the VCF-only
        `vcf_path` naming (here it's the resolved `.pgen` path). `output_length`
        is `-1` for ragged or a fixed length >= 1 (issue #277 Wave A `with_len`).
        `annotated` (issue #277 Wave A Task 4) selects whether the engine
        computes `annot_v_idxs`/`annot_ref_pos`, remapped window-local ->
        dataset-global per-variant via `slot.global_v_idxs`. Unlike VCF, PGEN's
        genoray ids are sourced from the `.pvar` row index for each kept
        variant -- always known/correct, including across the region-overlap
        gap that made the earlier scalar `var_base` (derived from the padded
        `var_start` search bound, not the first KEPT variant) silently
        undercount `annot_v_idxs` for narrowed windows. PGEN no longer shares
        VCF's Phase-3 gap (see `pgen.rs`'s `PgenWindowFiller::fill` doc comment
        and GitHub issue #305).

        `var_fields` (Wave B PR-B3a, #304) selects `want_ref` (`"ref" in
        var_fields`, default `["alt", "ilen", "start"]` when `None`) -- PGEN has no
        INFO path, so `available_var_fields` never offers anything beyond `ref` and
        no `info_fields` are forwarded here.

        `var_window` (Wave B PR-B4, #304): see `_Svar1Backend.build_engine`'s
        matching parameter -- same `(lut, lut_dtype, opt)` bundle, same
        `_win_mode_kwargs` translation into `win_*` keyword arguments.
        """
        from ..genvarloader import RecordStreamEngine

        _active_fields = _normalize_var_fields(var_fields)
        want_ref = "ref" in _active_fields

        contig_names = list(self._contigs)
        contig_ref_bytes = [
            self._ref._contig_slice(i)[0] for i in range(len(contig_names))
        ]

        job_contig_idx = [int(j[0]) for j in jobs]
        job_region_starts = [np.ascontiguousarray(j[1], np.uint32) for j in jobs]
        job_region_ends = [np.ascontiguousarray(j[2], np.uint32) for j in jobs]
        job_s_lo = [int(j[3]) for j in jobs]
        job_s_hi = [int(j[4]) for j in jobs]

        return RecordStreamEngine(
            "pgen",
            self._pgen_path,
            self._sample_names,
            self.ploidy,
            contig_names,
            contig_ref_bytes,
            job_contig_idx,
            job_region_starts,
            job_region_ends,
            job_s_lo,
            job_s_hi,
            # `fasta_path=None` -- PARITY-CRITICAL, not a placeholder, for the
            # same reason as `_VcfBackend.build_engine`: `gvl.write`'s PGEN
            # path does no read-time reference/left-alignment, so the
            # streaming decoder must not either (see
            # `src/record_stream/pgen.rs`'s module doc, "Config parity with
            # VCF" section). `self._ref` above is used ONLY to derive
            # `contig_ref_bytes` for haplotype reconstruction padding.
            None,
            self._ref.pad_char,
            True,
            batch_size,
            output_length,
            annotated,
            variants,
            # `min_af`/`max_af`: PGEN stages no INFO, so `PgenWindowFiller` never
            # populates AFs and the Rust AF fold is a no-op regardless of these
            # values -- forwarded only to keep the two backends' `RecordStreamEngine`
            # call sites symmetric; AF filtering on PGEN is guarded out upstream
            # (see `has_cached_af` and `StreamingDataset._iter_batches`'s guard).
            min_af,
            max_af,
            want_ref=want_ref,
            **_win_mode_kwargs(var_window),
        )
