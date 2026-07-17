"""Eager-indexing query path for :class:`Dataset`.

:class:`Dataset.__getitem__` packages its state into a :class:`QueryView` and
calls :func:`getitem` here. ``QueryView`` is a typed contract; the free
functions in this module take it explicitly and don't depend on a full
``Dataset`` instance, so each stage is unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, overload

import numpy as np
from numpy.typing import NDArray
from seqpro.rag import Ragged
from typing_extensions import assert_never

from .._flat import _Flat, _FlatAnnotatedHaps
from ._flat_variants import _FlatVariants, _FlatVariantWindows
from .._ragged import (
    FlatIntervals,
    RaggedAnnotatedHaps,
    RaggedIntervals,
    _COMP,
)
from .._types import AnnotatedHaps, StrIdx
from ._haps import Haps
from ._indexing import DatasetIndexer, SpliceIndexer, is_str_arr
from ._protocol import Reconstructor
from ._rag_variants import RaggedVariants
from ._ref import Ref
from ._splice import SplicePlan, build_splice_plan
from ._tracks import Tracks


@dataclass(frozen=True, slots=True)
class QueryView:
    """Typed view over the Dataset state needed to answer a query.

    Constructed by :meth:`Dataset.__getitem__` and passed to the free
    functions in this module. Holding this in its own dataclass makes the
    contract between :class:`Dataset` and the query path explicit and lets
    each stage be tested with a synthetic view.
    """

    idxer: DatasetIndexer
    sp_idxer: SpliceIndexer | None
    full_regions: NDArray[np.int32]
    rng: np.random.Generator
    recon: Reconstructor
    output_length: Literal["ragged", "variable"] | int
    jitter: int
    deterministic: bool
    rc_neg: bool
    flat_output: bool = False

    @property
    def full_shape(self) -> tuple[int, int]:
        return self.idxer.full_shape


_QueryIdx = StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx]


def getitem(
    view: QueryView, idx: _QueryIdx
) -> (
    Ragged[np.bytes_ | np.float32]
    | RaggedAnnotatedHaps
    | RaggedVariants
    | RaggedIntervals
    | NDArray[np.bytes_ | np.float32]
    | AnnotatedHaps
    | tuple[
        Ragged[np.bytes_ | np.float32]
        | RaggedAnnotatedHaps
        | RaggedVariants
        | RaggedIntervals
        | NDArray[np.bytes_ | np.float32]
        | AnnotatedHaps,
        ...,
    ]
):
    """Top-level eager query. Dispatches to the (un)spliced helpers and then
    applies the shape-massaging steps (pad / to_numpy, reshape, squeeze)."""
    if is_str_arr(idx) and view.idxer.r2i_map is None:
        raise ValueError("Cannot query regions by name because no region name was set.")

    if view.sp_idxer is not None:
        recon, squeeze, out_reshape = _getitem_spliced(view, idx, view.sp_idxer)
    else:
        recon, squeeze, out_reshape = _getitem_unspliced(view, idx)

    if not view.flat_output:
        if view.output_length == "variable":
            recon = tuple(
                r if isinstance(r, (RaggedVariants, RaggedIntervals)) else pad(r)
                for r in recon
            )
        elif isinstance(view.output_length, int):
            recon = tuple(
                r
                if isinstance(r, (RaggedVariants, RaggedIntervals))
                else r.to_fixed(view.output_length)
                for r in recon
            )

        # Convert any still-flat elements (ragged output_length path) to their
        # public Ragged types before reshape/squeeze apply the existing logic.
        recon = tuple(
            o.to_ragged()
            if isinstance(o, (_Flat, _FlatAnnotatedHaps, _FlatVariants))
            else o
            for o in recon
        )

    if out_reshape is not None:
        recon = tuple(_reshape_outer(o, out_reshape) for o in recon)

    if squeeze:
        # (1 [p] l) -> ([p] l); axis 0 is a singleton here, so squeeze(0) drops it.
        recon = tuple(o.squeeze(0) for o in recon)

    if len(recon) == 1:
        recon = recon[0]

    return recon


def _reshape_outer(o, out_reshape: tuple[int, ...]):
    """Reshape the outer (leading) dims of a query output to ``out_reshape``.

    Reshape conventions differ by type. A ``Ragged`` (or
    ``RaggedAnnotatedHaps``) ``.reshape()`` takes the FULL new shape, including
    the trailing ragged ``None`` axis, so we pass ``out_reshape + o.shape[1:]``.
    By contrast ``_Flat``/``_FlatAnnotatedHaps`` ``.reshape()`` takes only the
    OUTER fixed dims and re-appends its own trailing ``None``; passing the full
    shape (which already ends in ``None``) would yield a double ragged axis.
    For those we drop the trailing ``None`` and pass only the outer dims.
    """
    if isinstance(
        o,
        (_Flat, _FlatAnnotatedHaps, _FlatVariants, _FlatVariantWindows, FlatIntervals),
    ):
        # _FlatVariantWindows.shape mirrors _FlatVariants ((b, p, None) — one None,
        # taken from its scalar `start` field), so it shares this branch: drop the
        # single trailing None and pass only the outer fixed dims; reshape() re-appends
        # its own ragged axis/axes.
        return o.reshape(out_reshape + o.shape[1:-1])
    return o.reshape(out_reshape + o.shape[1:])  # type: ignore[bad-argument-type, no-matching-overload]  # heterogeneous reshape() across array kinds; shape tuple may contain None for ragged dims


def _getitem_unspliced(
    view: QueryView, idx: _QueryIdx
) -> tuple[
    tuple[Ragged[np.bytes_ | np.float32] | RaggedAnnotatedHaps | RaggedVariants, ...],
    bool,
    tuple[int, ...] | None,
]:
    # (b)
    ds_idx, squeeze, out_reshape = view.idxer.parse_idx(idx)
    r_idx, _ = np.unravel_index(ds_idx, view.full_shape)

    # makes a copy because r_idx is at least 1D & triggers advanced indexing
    regions = view.full_regions[r_idx]
    lengths = regions[:, 2] - regions[:, 1]
    jitter_off = view.rng.integers(
        -view.jitter, view.jitter + 1, size=len(regions), dtype=np.int32
    )
    regions[:, 1] += jitter_off
    regions[:, 2] = regions[:, 1] + lengths

    to_rc: NDArray[np.bool_] | None = (
        view.full_regions[r_idx, 3] == -1 if view.rc_neg else None
    )

    recon = view.recon(
        idx=ds_idx,
        r_idx=r_idx,
        regions=regions,
        output_length=view.output_length,
        jitter=view.jitter,
        rng=view.rng,
        deterministic=view.deterministic,
        flat=view.flat_output,
        to_rc=to_rc,
    )

    if not isinstance(recon, tuple):
        recon = (recon,)

    if view.rc_neg and to_rc is not None:
        # Rust: flat-seq kinds (bytes, tracks, annotated-haps) have RC
        # folded into the kernel or handled Python-side inside the
        # reconstructor.  Variant types have no in-kernel RC and are
        # deferred here.  (_FlatVariantWindows RC is a no-op in
        # reverse_complement_ragged; RaggedVariants is Target 7.)
        _VARIANT_TYPES = (RaggedVariants, _FlatVariants, _FlatVariantWindows)
        recon = tuple(
            reverse_complement_ragged(r, to_rc) if isinstance(r, _VARIANT_TYPES) else r
            for r in recon
        )

    return recon, squeeze, out_reshape


def _getitem_spliced(
    view: QueryView, idx: _QueryIdx, splice_idxer: SpliceIndexer
) -> tuple[
    tuple[
        Ragged[np.bytes_ | np.float32] | RaggedAnnotatedHaps | RaggedVariants,
        ...,
    ],
    bool,
    tuple[int, ...] | None,
]:
    if isinstance(view.output_length, int):
        raise RuntimeError(
            "In general, splicing cannot be done with fixed length data because even if the length of each region's data"
            " is fixed/constant, the number of elements in each spliced element is not. Thus, the final length of the"
            " spliced elements will be variable."
        )

    assert not isinstance(view.output_length, int)
    assert view.jitter == 0
    assert view.deterministic

    # Internally the spliced kernel runs against ragged output, regardless of
    # the user's outward output_length (which has already been asserted not to
    # be an int).

    (
        ds_idx,
        squeeze,
        out_reshape,
        offsets,
        n_rows_sel,
        n_samples_sel,
    ) = splice_idxer.parse_idx(idx)
    r_idx, _ = np.unravel_index(ds_idx, view.idxer.full_shape)
    regions = view.full_regions[r_idx]

    if isinstance(view.recon, Haps) and issubclass(view.recon.kind, RaggedVariants):
        if view.flat_output:
            raise ValueError("Spliced variants require output_format='ragged'.")
        variants = _fetch_spliced_variants(
            view=view,
            ds_idx=ds_idx,
            r_idx=r_idx,
            regions=regions,
            splice_row_offsets=offsets,
            n_rows=n_rows_sel,
            n_samples=n_samples_sel,
        )
        return (variants,), squeeze, out_reshape

    # Build the splice plan from per-query lengths produced by the active
    # reconstructor. The plan drives the kernel into writing pre-spliced
    # bytes in (splice_row, sample, *inner_fixed, splice_element) C-order.
    plan = build_recon_splice_plan(
        recon=view.recon,
        ds_idx=ds_idx,
        regions=regions,
        splice_row_offsets=offsets,
        n_rows=n_rows_sel,
        n_samples=n_samples_sel,
    )

    # Compute the permuted per-element to_rc mask (used for both the in-kernel
    # pass and the post-pass guard below).
    to_rc_per_elem: NDArray[np.bool_] | None = None
    if view.rc_neg:
        B = regions.shape[0]
        n_k = int(plan.permutation.shape[0])
        inner_factor, rem = divmod(n_k, B)
        if rem != 0:
            raise AssertionError(
                "plan.permutation length is not a multiple of len(regions); "
                "inner-fixed flatten factor inconsistent."
            )
        to_rc_unperm = regions[:, 3] == -1
        if inner_factor == 1:
            to_rc_flat = to_rc_unperm
        else:
            # (B, E) C-order: same value across the inner axis for a given
            # query. np.repeat gives (B*E,) in (query, inner) C-order.
            to_rc_flat = np.repeat(to_rc_unperm, inner_factor)
        to_rc_per_elem = to_rc_flat[plan.permutation]

    recon = view.recon(
        idx=ds_idx,
        r_idx=r_idx,
        regions=regions,
        output_length="ragged",
        jitter=view.jitter,
        rng=view.rng,
        deterministic=view.deterministic,
        splice_plan=plan,
        flat=view.flat_output,
        to_rc=to_rc_per_elem,
    )

    if not isinstance(recon, tuple):
        recon = (recon,)

    recon = cast(
        tuple[Ragged[np.bytes_ | np.float32] | RaggedAnnotatedHaps, ...], recon
    )

    # Rewrap each per-element Ragged with the plan's group_offsets to expose
    # one contiguous spliced element per (row, sample[, inner]) cell. Collapse
    # (n_rows, n_samples) into a single leading "pair" axis so the downstream
    # out_reshape step in getitem can reshape it back to whatever shape the
    # user's index requested.
    recon = tuple(_regroup(r, plan.group_offsets, plan.flat_out_shape) for r in recon)

    return recon, squeeze, out_reshape


def _fetch_spliced_variants(
    *,
    view: QueryView,
    ds_idx: NDArray[np.intp],
    r_idx: NDArray[np.intp],
    regions: NDArray[np.int32],
    splice_row_offsets: NDArray[np.int64],
    n_rows: int,
    n_samples: int,
) -> RaggedVariants:
    """Decode exon records once, then regroup them into complete transcripts.

    Variant counts are data-dependent, so unlike sequence output the splice plan
    cannot be built before reconstruction.  The active ``Haps`` reconstructor
    first decodes the whole query block (including its native Rust batching),
    then this helper builds the plan from the returned per-exon/per-phase counts.
    The final record buffers are rewrapped without copying scalar values.
    """
    raw = view.recon(
        idx=ds_idx,
        r_idx=r_idx,
        regions=regions,
        output_length="ragged",
        jitter=0,
        rng=view.rng,
        deterministic=True,
        splice_plan=None,
        flat=False,
        to_rc=None,
    )
    if isinstance(raw, tuple):
        if len(raw) != 1:
            raise NotImplementedError(
                "Spliced variants cannot currently be combined with track output."
            )
        raw = raw[0]
    if isinstance(raw, _FlatVariants):
        raw = raw.to_ragged()
    if not isinstance(raw, RaggedVariants):
        raise TypeError(
            "The variants reconstructor returned an unexpected output type: "
            f"{type(raw).__name__}."
        )

    batch = len(ds_idx)
    ploidy = raw.shape[-2]
    assert isinstance(ploidy, int)
    lengths = np.asarray(raw.start.lengths, np.int32).reshape(batch, ploidy)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=n_samples,
        n_rows=n_rows,
    )

    flat = raw.reshape(batch * ploidy, None)
    permuted = flat[plan.permutation].to_packed()
    fields: dict[str, Ragged] = {}
    for name in permuted.fields:
        field = permuted[name].to_packed()
        str_offsets = field._rl.str_offsets
        fields[name] = Ragged.from_offsets(
            field.data,
            plan.flat_out_shape,
            plan.group_offsets,
            str_offsets=(
                None if str_offsets is None else np.asarray(str_offsets, np.int64)
            ),
        )

    alt = fields.pop("alt")
    start = fields.pop("start")
    ref = fields.pop("ref", None)
    ilen = fields.pop("ilen", None)
    dosage = fields.pop("dosage", None)
    grouped = RaggedVariants(
        alt=alt,
        start=start,
        ref=ref,
        ilen=ilen,
        dosage=dosage,
        **fields,
    )

    if view.rc_neg and len(splice_row_offsets) > 1:
        pair_lengths = np.diff(splice_row_offsets)
        if (pair_lengths <= 0).any():
            raise ValueError("Splice rows must contain at least one element.")
        per_element_rc = regions[:, 3] == -1
        pair_rc = per_element_rc[splice_row_offsets[:-1]]
        if not np.array_equal(np.repeat(pair_rc, pair_lengths), per_element_rc):
            raise ValueError("All elements of a splice row must use the same strand.")
        grouped = grouped.rc_(pair_rc)

    return grouped


def build_recon_splice_plan(
    recon: Reconstructor,
    ds_idx: NDArray[np.intp],
    regions: NDArray[np.int32],
    splice_row_offsets: NDArray[np.int64],
    n_rows: int,
    n_samples: int,
) -> SplicePlan:
    """Build a ``SplicePlan`` from the active reconstructor.

    Dispatches on ``type(recon)``: ``Haps`` uses (B, P) haplotype lengths;
    ``Ref`` uses 1-D per-region lengths; track-bearing reconstructors are
    not yet supported in the spliced path (except solo ``Tracks``).
    """
    # Local import to avoid the dispatcher module being imported on every
    # query path.
    from ._reconstruct import HapsTracks, SeqsTracks

    if isinstance(recon, HapsTracks):
        raise NotImplementedError(
            "Splicing of haplotypes + tracks (shape (b, t, p, ~l)) is not supported."
        )
    if isinstance(recon, SeqsTracks):
        raise NotImplementedError(
            "Splicing of sequences + un-realigned tracks is not supported."
        )
    if isinstance(recon, Tracks):
        # Tracks have deterministic per-region lengths (no haplotype indels).
        # Replicate the (B,) length array across the n_tracks inner axis so
        # the plan's inner_fixed = (n_tracks,).
        n_tracks = len(recon.active_tracks)
        base = (regions[:, 2] - regions[:, 1]).astype(np.int32, copy=False)
        lengths_2d = np.broadcast_to(base[:, None], (base.shape[0], n_tracks))
        return build_splice_plan(
            lengths=np.ascontiguousarray(lengths_2d),
            splice_row_offsets=splice_row_offsets,
            n_samples=n_samples,
            n_rows=n_rows,
        )
    if isinstance(recon, Haps):
        lengths_2d = recon.haplotype_lengths_for_plan(idx=ds_idx, regions=regions)
        return build_splice_plan(
            lengths=lengths_2d.astype(np.int32, copy=False),
            splice_row_offsets=splice_row_offsets,
            n_samples=n_samples,
            n_rows=n_rows,
        )
    if isinstance(recon, Ref):
        lengths_1d = (regions[:, 2] - regions[:, 1]).astype(np.int32, copy=False)
        return build_splice_plan(
            lengths=lengths_1d,
            splice_row_offsets=splice_row_offsets,
            n_samples=n_samples,
            n_rows=n_rows,
        )
    raise NotImplementedError(
        f"Splicing not supported for reconstructor {type(recon).__name__}."
    )


@overload
def reverse_complement_ragged(rag: _Flat, to_rc: NDArray[np.bool_]) -> _Flat: ...
@overload
def reverse_complement_ragged(
    rag: _FlatAnnotatedHaps, to_rc: NDArray[np.bool_]
) -> _FlatAnnotatedHaps: ...
@overload
def reverse_complement_ragged(
    rag: _FlatVariants, to_rc: NDArray[np.bool_]
) -> _FlatVariants: ...
@overload
def reverse_complement_ragged(
    rag: _FlatVariantWindows, to_rc: NDArray[np.bool_]
) -> _FlatVariantWindows: ...
@overload
def reverse_complement_ragged(
    rag: RaggedVariants, to_rc: NDArray[np.bool_]
) -> RaggedVariants: ...
@overload
def reverse_complement_ragged(
    rag: RaggedIntervals, to_rc: NDArray[np.bool_]
) -> RaggedIntervals: ...
def reverse_complement_ragged(
    rag: _Flat
    | _FlatAnnotatedHaps
    | _FlatVariants
    | _FlatVariantWindows
    | FlatIntervals
    | RaggedVariants
    | RaggedIntervals,
    to_rc: NDArray[np.bool_],
) -> (
    _Flat
    | _FlatAnnotatedHaps
    | _FlatVariants
    | _FlatVariantWindows
    | FlatIntervals
    | RaggedVariants
    | RaggedIntervals
):
    """Reverse-complement (or reverse) ragged outputs according to a per-row mask."""
    if isinstance(rag, FlatIntervals):
        # Intervals are not reverse-complemented (same as RaggedIntervals).
        return rag
    if isinstance(rag, _FlatVariantWindows):
        # Windows are reference-oriented; reverse-complement is not applied.
        return rag
    if isinstance(rag, _Flat):
        comp = _COMP if rag.data.dtype.kind == "S" else None
        return rag.reverse_masked(to_rc, comp=comp)
    if isinstance(rag, _FlatAnnotatedHaps):
        return rag.reverse_masked(to_rc, _COMP)
    if isinstance(rag, _FlatVariants):
        return rag.reverse_masked(to_rc)
    if isinstance(rag, RaggedVariants):
        return rag.rc_(to_rc)
    if isinstance(rag, RaggedIntervals):
        return rag
    assert_never(rag)  # type: ignore[arg-type]


@overload
def pad(rag: _Flat) -> NDArray: ...
@overload
def pad(rag: _FlatAnnotatedHaps) -> AnnotatedHaps: ...
def pad(rag: _Flat | _FlatAnnotatedHaps) -> NDArray | AnnotatedHaps:
    """Materialize a _Flat (or _FlatAnnotatedHaps) into a dense padded array."""
    if isinstance(rag, _Flat):
        if rag.data.dtype.kind == "S":
            return rag.view("S1").to_padded(b"N")
        else:
            return rag.to_padded(0)
    if isinstance(rag, _FlatAnnotatedHaps):
        return rag.to_padded()
    assert_never(rag)  # type: ignore[arg-type]


def _regroup(
    rag: _Flat | _FlatAnnotatedHaps | Ragged | RaggedAnnotatedHaps,
    group_offsets: NDArray[np.int64],
    out_shape: tuple[int | None, ...],
) -> _Flat | _FlatAnnotatedHaps | Ragged | RaggedAnnotatedHaps:
    """Rewrap a per-element flat ragged (or Ragged) with grouped offsets so
    each cell holds one contiguous spliced element.

    Both branches share the same data buffer; only the outer offsets / shape
    change.
    """
    if isinstance(rag, _FlatAnnotatedHaps):
        return _FlatAnnotatedHaps(
            _regroup(rag.haps, group_offsets, out_shape),  # type: ignore[arg-type]
            _regroup(rag.var_idxs, group_offsets, out_shape),  # type: ignore[arg-type]
            _regroup(rag.ref_coords, group_offsets, out_shape),  # type: ignore[arg-type]
        )
    if isinstance(rag, _Flat):
        return _Flat.from_offsets(rag.data, out_shape, group_offsets)
    if isinstance(rag, RaggedAnnotatedHaps):
        return RaggedAnnotatedHaps(
            haps=cast(
                Ragged[np.bytes_],
                _regroup(rag.haps, group_offsets, out_shape),
            ),
            var_idxs=cast(Ragged, _regroup(rag.var_idxs, group_offsets, out_shape)),
            ref_coords=cast(Ragged, _regroup(rag.ref_coords, group_offsets, out_shape)),
        )
    return Ragged.from_offsets(rag.data, out_shape, group_offsets)
