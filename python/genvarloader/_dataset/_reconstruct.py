"""Reconstructor dispatcher.

Houses the *compound* reconstructors (:class:`SeqsTracks`, :class:`HapsTracks`)
that combine a sequence source with tracks, plus the
:func:`_build_reconstructor` factory.

Re-exports the leaf reconstructors (:class:`Ref`, :class:`Haps`,
:class:`Tracks`) and supporting types (``Reconstructor``,
``ReconstructionRequest``, ``_Variants``, ``TrackType``) from their split
modules for backward-compatible import paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from einops import repeat
from numpy.typing import NDArray
from typing_extensions import assert_never

from .._flat import _Flat
from .._ragged import RaggedAnnotatedHaps, RaggedIntervals, RaggedSeqs, RaggedTracks
from .._utils import lengths_to_offsets
from ._genotypes import _as_starts_stops
from ._haps import _H, Haps, ReconstructionRequest, _NewH, _Variants
from ._insertion_fill import Repeat5p
from ._insertion_fill import lower as _lower_insertion_fills
from ._flat_variants import _FlatVariantWindows
from ._protocol import Reconstructor
from ._rag_variants import RaggedVariants
from ._ref import Ref
from ._splice import SplicePlan
from ._tracks import (
    _T,
    Tracks,
    TrackType,
    _NewT,
)  # noqa: F401
from ._utils import _ffi_array

# Fused tracks entry (Task 14): intervals → scratch → realign, one FFI crossing.
# Imported at module level so the spy in test_fused_tracks_parity can monkeypatch it.
from ..genvarloader import (
    intervals_and_realign_track_fused as intervals_and_realign_track_fused,
)


# Re-exports for back-compat (callers historically imported these from
# ``_reconstruct``):
__all__ = [
    "Haps",
    "HapsTracks",
    "ReconstructionRequest",
    "Reconstructor",
    "Ref",
    "SeqsTracks",
    "TrackType",
    "Tracks",
    "_Variants",
    "_build_reconstructor",
]


@dataclass(slots=True)
class SeqsTracks(Reconstructor[tuple[Any, _T]]):
    """Any seq reconstructor (`Ref` or `Haps` in any kind) paired with
    un-realigned `Tracks`. Seqs and tracks are computed independently; tracks
    stay in reference coordinates (no haplotype re-alignment)."""

    seqs: Ref | Haps
    tracks: Tracks[_T]

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        flat: bool = False,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> tuple[Any, _T]:
        if splice_plan is not None:
            raise NotImplementedError(
                "Splicing of sequences + un-realigned tracks is not supported."
            )
        seqs = self.seqs(
            idx=idx,
            r_idx=r_idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
            deterministic=deterministic,
            flat=flat,
            to_rc=to_rc,
        )
        tracks = self.tracks(
            idx=idx,
            r_idx=r_idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
            deterministic=deterministic,
            flat=flat,
            to_rc=to_rc,
        )
        return seqs, tracks


@dataclass(slots=True)
class HapsTracks(Reconstructor[tuple[_H, _T]]):
    haps: Haps[_H]
    tracks: Tracks[_T]

    def to_kind(
        self, kind: tuple[type[_NewH], type[_NewT]]
    ) -> HapsTracks[_NewH, _NewT]:
        haps = self.haps.to_kind(kind[0])
        tracks = self.tracks.to_kind(kind[1])
        return cast(HapsTracks[_NewH, _NewT], replace(self, haps=haps, tracks=tracks))

    def __call__(
        self,
        idx: NDArray[np.integer],  # (b)
        r_idx: NDArray[np.integer],  # (b)
        regions: NDArray[np.int32],  # (b 3)
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        flat: bool = False,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> tuple[_H, _T]:
        if splice_plan is not None:
            raise NotImplementedError(
                "Splicing of haplotypes + tracks (shape (b, t, p, ~l)) is not "
                "supported."
            )
        lengths = regions[:, 2] - regions[:, 1]

        # ragged (b p l), (b p), (b p), (b*p*v), (b*p+1), (b p)
        haps, geno_idx, shifts, diffs, hap_lengths, keep, keep_offsets = (
            self.haps.get_haps_and_shifts(
                idx=idx,
                regions=regions,
                output_length=output_length,
                rng=rng,
                deterministic=deterministic,
                to_rc=to_rc,
            )
        )

        if issubclass(self.tracks.kind, RaggedTracks):
            if isinstance(output_length, int):
                # (b p)
                out_lengths = np.full_like(hap_lengths, output_length)
            else:
                # (b p)
                out_lengths = hap_lengths

            # (b) = lengths (b) + max deletion length across ploidy (b p) -> (b)
            track_lengths = lengths - diffs.clip(max=0).min(1)

            # (b*p+1)
            out_ofsts_per_t = lengths_to_offsets(out_lengths)
            # (b+1)
            track_ofsts_per_t = lengths_to_offsets(track_lengths)
            n_per_track: int = out_ofsts_per_t[-1]
            # ragged (b t p l)
            out = np.empty(len(self.tracks.active_tracks) * n_per_track, np.float32)
            out_lens = repeat(
                out_lengths, "b p -> b t p", t=len(self.tracks.active_tracks)
            )
            out_offsets = lengths_to_offsets(out_lens)

            # Lower per-track strategies into numba-friendly arrays.
            strat_list = [
                self.tracks.insertion_fill.get(name, Repeat5p())
                for name in self.tracks.active_tracks
            ]
            strat_ids, strat_params = _lower_insertion_fills(strat_list)
            # Base seed for FlankSample determinism. When deterministic, derive
            # from the full idx array so different batches produce different
            # fills; same input always produces the same fill. Uses the full
            # uint64 range.
            if deterministic:
                base_seed = np.uint64(
                    np.bitwise_xor.reduce(idx.astype(np.uint64, copy=False))
                )
            else:
                base_seed = np.uint64(
                    rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
                )

            # Pre-compute (2, n) geno_offsets once for the fused Rust path
            # (avoids re-computing _as_starts_stops n_tracks times).
            _geno_offsets_2d = _as_starts_stops(self.haps.genotypes.offsets)

            for track_ofst, (name, tracktype) in enumerate(
                self.tracks.active_tracks.items()
            ):
                intervals = self.tracks.intervals[name]

                if tracktype is TrackType.SAMPLE:
                    o_idx = idx
                else:
                    o_idx = r_idx

                _out = out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track]

                # Fused path (Rust): one FFI crossing, no Python-side
                # intermediate buffer.  Replaces:
                #   _tracks = np.empty(...)                (audit T2)
                #   intervals_to_tracks(...)               (FFI crossing #3)
                #   shift_and_realign_tracks_sparse(...)   (FFI crossing #4)
                #
                # _out is a contiguous f32 slice of the pre-allocated `out`
                # buffer (np.empty, step=1).  No ascontiguousarray needed for
                # `out`; the fused entry writes in-place into its buffer.
                # Expand per-query to_rc to per-(query, hap) for the track kernel.
                # out_ofsts_per_t is (b*p+1); ploidy = geno_idx.shape[-1].
                _ploidy = geno_idx.shape[-1]
                _to_rc_hap = (
                    None
                    if to_rc is None
                    else np.ascontiguousarray(np.repeat(to_rc, _ploidy), np.bool_)
                )
                intervals_and_realign_track_fused(
                    out=_out,
                    out_offsets=np.ascontiguousarray(out_ofsts_per_t, np.int64),
                    regions=np.ascontiguousarray(regions, np.int32),
                    shifts=np.ascontiguousarray(shifts, np.int32),
                    geno_offset_idx=np.ascontiguousarray(geno_idx, np.int64),
                    geno_v_idxs=_ffi_array(
                        self.haps.genotypes.data, np.int32, "geno_v_idxs"
                    ),
                    geno_offsets=_geno_offsets_2d,
                    v_starts=self.haps.ffi_static.v_starts,
                    ilens=self.haps.ffi_static.ilens,
                    offset_idxs=np.ascontiguousarray(o_idx, np.int64),
                    itv_starts=_ffi_array(
                        intervals.starts.data, np.int32, "itv_starts"
                    ),
                    itv_ends=_ffi_array(intervals.ends.data, np.int32, "itv_ends"),
                    itv_values=_ffi_array(
                        intervals.values.data, np.float32, "itv_values"
                    ),
                    itv_offsets=_ffi_array(
                        intervals.starts.offsets, np.int64, "itv_offsets"
                    ),
                    track_offsets=np.ascontiguousarray(track_ofsts_per_t, np.int64),
                    params=np.ascontiguousarray(strat_params[track_ofst], np.float64),
                    strategy_id=int(strat_ids[track_ofst]),
                    base_seed=int(base_seed),
                    keep=None if keep is None else np.ascontiguousarray(keep, np.bool_),
                    keep_offsets=None
                    if keep_offsets is None
                    else np.ascontiguousarray(keep_offsets, np.int64),
                    to_rc=_to_rc_hap,
                )

            out_shape = (
                len(idx),
                len(self.tracks.active_tracks),
                self.haps.genotypes.shape[-2],
                None,
            )

            # flat (b t p l)
            tracks = _Flat.from_offsets(out, out_shape, out_offsets)

        else:
            tracks = self.tracks._call_intervals(idx)

        tracks = cast(_T, tracks)

        return haps, tracks


def _build_reconstructor(
    seqs: Haps | Ref | None,
    tracks: Tracks | None,
    seqs_kind: Literal[
        "haplotypes", "reference", "annotated", "variants", "variant-windows"
    ]
    | None,
    realign_tracks: bool = True,
) -> Reconstructor:
    """Construct the reconstructor for the given (storage + view) state.

    The user's view choice is carried in ``seqs_kind`` (``None`` means "user does
    not want sequences"). Track activation is read from ``tracks.active_tracks``
    (``None`` means "user does not want tracks"). This function maps that
    explicit state to one of the 5 reconstructor classes.

    Invariant: after resolving view state, at least one of (active_seqs,
    active_tracks) must be non-None.
    """
    # Resolve active seqs from storage + view kind.
    active_seqs: Haps | Ref | None
    if seqs_kind is None or seqs is None:
        active_seqs = None
    elif seqs_kind == "reference":
        if isinstance(seqs, Ref):
            active_seqs = seqs
        elif isinstance(seqs, Haps):
            if seqs.reference is None:
                raise ValueError(
                    "Cannot view as 'reference': storage has no reference genome."
                )
            active_seqs = Ref(reference=seqs.reference)
        else:
            assert_never(seqs)
    elif seqs_kind in ("haplotypes", "annotated", "variants", "variant-windows"):
        if not isinstance(seqs, Haps):
            raise ValueError(
                f"Cannot view as {seqs_kind!r}: storage has no haplotypes."
            )
        kind_map = {
            "haplotypes": RaggedSeqs,
            "annotated": RaggedAnnotatedHaps,
            "variants": RaggedVariants,
            "variant-windows": _FlatVariantWindows,
        }
        active_seqs = seqs.to_kind(kind_map[seqs_kind])
    else:
        assert_never(seqs_kind)

    # Resolve active tracks from storage + active_tracks subset.
    active_tracks = (
        tracks if (tracks is not None and bool(tracks.active_tracks)) else None
    )

    # Dispatch.
    match active_seqs, active_tracks:
        case None, None:
            raise ValueError(
                "_build_reconstructor requires at least one of (seqs, tracks) "
                "to be active. Got seqs_kind=None and tracks inactive."
            )
        case (Haps() | Ref()) as s, None:
            return s
        case None, Tracks() as t:
            return t
        case Ref() as s, Tracks() as t:
            return SeqsTracks(seqs=s, tracks=t)
        case Haps() as s, Tracks() as t:
            is_intervals = issubclass(t.kind, RaggedIntervals)
            if is_intervals:
                if realign_tracks:
                    raise ValueError(
                        "Track intervals cannot be re-aligned to haplotype"
                        " coordinates. Set with_settings(realign_tracks=False) to"
                        " return reference-coordinate (as-is) intervals."
                    )
                return SeqsTracks(seqs=s, tracks=t)
            # Float tracks (RaggedTracks).
            if seqs_kind == "variant-windows":
                if realign_tracks:
                    raise ValueError(
                        "with_seqs('variant-windows') with tracks requires"
                        " with_settings(realign_tracks=False) (windows are"
                        " reference-oriented; re-alignment is not supported)."
                    )
                return SeqsTracks(seqs=s, tracks=t)
            if realign_tracks:
                return HapsTracks(haps=s, tracks=t)
            return SeqsTracks(seqs=s, tracks=t)
        case _:
            raise AssertionError(
                f"unreachable: active_seqs={type(active_seqs).__name__}, "
                f"active_tracks={type(active_tracks).__name__}"
            )
