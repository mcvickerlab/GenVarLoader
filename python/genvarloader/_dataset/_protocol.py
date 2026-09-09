"""Shared :class:`Reconstructor` Protocol.

Lives in its own module so the leaf reconstructor classes (``Ref``, ``Haps``,
``Tracks``) and the compound classes / factory in ``_reconstruct.py`` can all
depend on it without circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .._ragged import RaggedIntervals
    from ._splice import SplicePlan

T = TypeVar("T", covariant=True)


class Reconstructor(Protocol[T]):
    """Reconstructs data on-the-fly. e.g. personalized sequences, tracks, etc."""

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
    ) -> T:
        """``flat`` only changes behavior for :class:`Haps` producing ``RaggedVariants`` (it returns a flat ``_FlatVariants`` instead).

        All other reconstructors are already flat-native and accept-and-ignore it.

        ``to_rc`` is a per-row boolean mask (True = reverse-complement this row).
        On the Rust backend, flat-seq kinds fold RC in-kernel; on numba the
        caller's post-pass handles it and this param is ignored by each method.
        """
        ...


class TrackRealigner(Protocol):
    """Per-batch state for filling haplotype-realigned track blocks.

    ``HapsTracks.__call__`` allocates one output buffer for the whole batch and
    then fills it one track at a time. Everything the realign kernel needs that
    does *not* vary across tracks -- the batch's regions, shifts, haplotype
    offsets and whatever backend-specific arrays those imply -- is prepared once
    by :meth:`Haps.track_realigner` and carried here, so the per-track loop stays
    backend-agnostic and per-batch work is not repeated per track.
    """

    def fill(
        self,
        out: NDArray[np.float32],
        o_idx: NDArray[np.integer],
        intervals: "RaggedIntervals",
        params: NDArray[np.float64],
        strategy_id: int,
    ) -> None:
        """Write one track's realigned haplotype block into ``out``.

        Args:
            out: The destination slice for this track, laid out on the batch's
                per-haplotype output offsets. Written in place.
            o_idx: Per-query row into this track's intervals -- the sample-major
                index for :attr:`TrackType.SAMPLE` tracks, the region index
                otherwise.
            intervals: This track's stored intervals.
            params: The lowered insertion-fill parameters for this track.
            strategy_id: The lowered insertion-fill strategy for this track.
        """
        ...
