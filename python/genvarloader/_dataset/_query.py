"""Eager-indexing query path for :class:`Dataset`.

:class:`Dataset.__getitem__` packages its state into a :class:`QueryView` and
calls :func:`getitem` here. ``QueryView`` is a typed contract; the free
functions in this module take it explicitly and don't depend on a full
``Dataset`` instance, so each stage is unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, overload

import awkward as ak
import numpy as np
from numpy.typing import NDArray
from seqpro.rag import DTYPE_co as DTYPE
from seqpro.rag import Ragged, is_rag_dtype
from typing_extensions import assert_never

from .._ragged import (
    RaggedAnnotatedHaps,
    RaggedIntervals,
    reverse_complement,
    to_padded,
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

    if view.output_length == "variable":
        recon = tuple(
            r if isinstance(r, (RaggedVariants, RaggedIntervals)) else pad(r)
            for r in recon
        )
    elif isinstance(view.output_length, int):
        recon = tuple(
            r if isinstance(r, (RaggedVariants, RaggedIntervals)) else r.to_numpy()
            for r in recon
        )

    if out_reshape is not None:
        recon = tuple(o.reshape(out_reshape + o.shape[1:]) for o in recon)  # type: ignore

    if squeeze:
        # (1 [p] l) -> ([p] l)
        recon = tuple(o.squeeze(0) for o in recon)  # type: ignore

    if len(recon) == 1:
        recon = recon[0]

    return recon


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

    recon = view.recon(
        idx=ds_idx,
        r_idx=r_idx,
        regions=regions,
        output_length=view.output_length,
        jitter=view.jitter,
        rng=view.rng,
        deterministic=view.deterministic,
    )

    if not isinstance(recon, tuple):
        recon = (recon,)

    if view.rc_neg:
        to_rc: NDArray[np.bool_] = view.full_regions[r_idx, 3] == -1
        recon = tuple(reverse_complement_ragged(r, to_rc) for r in recon)

    return recon, squeeze, out_reshape  # type: ignore


def _getitem_spliced(
    view: QueryView, idx: _QueryIdx, splice_idxer: SpliceIndexer
) -> tuple[
    tuple[Ragged[np.bytes_ | np.float32] | RaggedAnnotatedHaps, ...],
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

    recon = view.recon(
        idx=ds_idx,
        r_idx=r_idx,
        regions=regions,
        output_length="ragged",
        jitter=view.jitter,
        rng=view.rng,
        deterministic=view.deterministic,
        splice_plan=plan,
    )

    if not isinstance(recon, tuple):
        recon = (recon,)

    recon = cast(
        tuple[Ragged[np.bytes_ | np.float32] | RaggedAnnotatedHaps, ...], recon
    )

    if view.rc_neg:
        # Permute the per-region to_rc mask the same way the plan permuted
        # the kernel queries. The plan acts on a flattened (B, *inner_fixed)
        # k-index, so first replicate to_rc across the inner axes, then
        # gather via plan.permutation.
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
        to_rc_per_elem: NDArray[np.bool_] = to_rc_flat[plan.permutation]
        recon = tuple(reverse_complement_ragged(r, to_rc_per_elem) for r in recon)

    # Rewrap each per-element Ragged with the plan's group_offsets to expose
    # one contiguous spliced element per (row, sample[, inner]) cell. Collapse
    # (n_rows, n_samples) into a single leading "pair" axis so the downstream
    # out_reshape step in getitem can reshape it back to whatever shape the
    # user's index requested.
    recon = tuple(_regroup(r, plan.group_offsets, plan.flat_out_shape) for r in recon)

    return recon, squeeze, out_reshape


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
    from ._reconstruct import HapsTracks, RefTracks

    if isinstance(recon, HapsTracks):
        raise NotImplementedError(
            "Splicing of haplotypes + tracks (shape (b, t, p, ~l)) is not supported."
        )
    if isinstance(recon, RefTracks):
        raise NotImplementedError(
            "Splicing of reference + tracks is not yet supported."
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
def reverse_complement_ragged(
    rag: Ragged[DTYPE], to_rc: NDArray[np.bool_]
) -> Ragged[DTYPE]: ...
@overload
def reverse_complement_ragged(
    rag: RaggedAnnotatedHaps, to_rc: NDArray[np.bool_]
) -> RaggedAnnotatedHaps: ...
@overload
def reverse_complement_ragged(
    rag: RaggedVariants, to_rc: NDArray[np.bool_]
) -> RaggedVariants: ...
@overload
def reverse_complement_ragged(
    rag: RaggedIntervals, to_rc: NDArray[np.bool_]
) -> RaggedIntervals: ...
def reverse_complement_ragged(
    rag: Ragged | RaggedAnnotatedHaps | RaggedVariants | RaggedIntervals,
    to_rc: NDArray[np.bool_],
) -> Ragged | RaggedAnnotatedHaps | RaggedVariants | RaggedIntervals:
    """Reverse-complement (or reverse) ragged outputs according to a per-row mask."""
    if isinstance(rag, Ragged):
        if is_rag_dtype(rag, np.bytes_):
            rag = Ragged(ak.to_packed(ak.where(to_rc, reverse_complement(rag), rag)))  # type: ignore
        else:
            rag = Ragged(ak.to_packed(ak.where(to_rc, rag[..., ::-1], rag)))  # type: ignore
    elif isinstance(rag, RaggedAnnotatedHaps):
        rag.haps = reverse_complement_ragged(rag.haps, to_rc)
        rag.var_idxs = reverse_complement_ragged(rag.var_idxs, to_rc)
        rag.ref_coords = reverse_complement_ragged(rag.ref_coords, to_rc)
    elif isinstance(rag, RaggedVariants):
        rag = rag.rc_(to_rc)
    elif isinstance(rag, RaggedIntervals):
        rag = rag
    else:
        assert_never(rag)
    return rag


@overload
def pad(rag: Ragged[DTYPE]) -> NDArray[DTYPE]: ...
@overload
def pad(rag: RaggedAnnotatedHaps) -> AnnotatedHaps: ...
def pad(rag: Ragged | RaggedAnnotatedHaps) -> NDArray | AnnotatedHaps:
    """Materialize a Ragged (or RaggedAnnotatedHaps) into a dense padded array."""
    if isinstance(rag, Ragged):
        if is_rag_dtype(rag, np.bytes_):
            return to_padded(rag, b"N")
        elif is_rag_dtype(rag, np.float32):
            return to_padded(rag, 0)
        else:
            raise ValueError(f"Unsupported pad dtype: {rag.data.dtype}")
    elif isinstance(rag, RaggedAnnotatedHaps):
        return rag.to_padded()
    else:
        assert_never(rag)


def _regroup(
    rag: Ragged | RaggedAnnotatedHaps,
    group_offsets: NDArray[np.int64],
    out_shape: tuple[int | None, ...],
) -> Ragged | RaggedAnnotatedHaps:
    """Rewrap a per-element Ragged (or RaggedAnnotatedHaps) with grouped
    offsets so each cell holds one contiguous spliced element.

    Both branches share the same data buffer; only the outer offsets / shape
    change.
    """
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
