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
    """Any seq reconstructor (`Ref` or `Haps` in any kind) paired with un-realigned `Tracks`.

    Seqs and tracks are computed independently; tracks
    stay in reference coordinates (no haplotype re-alignment).
    """

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
        """Haplotypes plus their haplotype-realigned tracks, ``(b, t, p, ~l)``.

        One body for every haplotype backend. The two things that differ --
        which request shapes a backend can serve, and how one track's realigned
        block is produced -- are asked of the reconstructor via
        :meth:`Haps.check_track_realign_support` and
        :meth:`Haps.track_realigner`.
        """
        ragged_tracks = issubclass(self.tracks.kind, RaggedTracks)
        self.haps.check_track_realign_support(
            output_length=output_length,
            splice_plan=splice_plan,
            to_rc=to_rc,
            ragged_tracks=ragged_tracks,
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

        if ragged_tracks:
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
            #
            # Seed-dependent fills stay byte-identical across backends even
            # though the readbound path realigns per contig group where
            # `k / ploidy` is contig-LOCAL: that path passes the group's global
            # row indices into the FFI and the kernel seeds with those. Fixed in
            # issue #267.
            if deterministic:
                base_seed = np.uint64(
                    np.bitwise_xor.reduce(idx.astype(np.uint64, copy=False))
                )
            else:
                base_seed = np.uint64(
                    rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
                )

            # Everything the realign kernels need that does not vary per track,
            # prepared once (see TrackRealigner).
            realigner = self.haps.track_realigner(
                idx=idx,
                regions=regions,
                shifts=shifts,
                geno_idx=geno_idx,
                track_lengths=track_lengths,
                out_offsets=out_ofsts_per_t,
                keep=keep,
                keep_offsets=keep_offsets,
                to_rc=to_rc,
                base_seed=int(base_seed),
            )

            for track_ofst, (name, tracktype) in enumerate(
                self.tracks.active_tracks.items()
            ):
                if tracktype is TrackType.SAMPLE:
                    o_idx = idx
                else:
                    o_idx = r_idx

                realigner.fill(
                    out=out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track],
                    o_idx=o_idx,
                    intervals=self.tracks.intervals[name],
                    params=strat_params[track_ofst],
                    strategy_id=int(strat_ids[track_ofst]),
                )

            out_shape = (
                len(idx),
                len(self.tracks.active_tracks),
                self.haps.stored_ploidy,
                None,
            )

            # flat (b t p l)
            tracks = _Flat.from_offsets(out, out_shape, out_offsets)

        else:
            tracks = self.tracks._call_intervals(idx)

        tracks = cast(_T, tracks)

        return cast(_H, haps), tracks


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
