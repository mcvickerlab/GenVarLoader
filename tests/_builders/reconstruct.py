"""In-memory builders for reconstruction-component types.

Currently exports ``make_tracks`` — a minimal-shape ``Tracks`` reconstructor
suitable for testing track-method plumbing (``with_insertion_fill``,
``with_tracks``, etc.) without opening a Dataset.

The builder produces a 1-region, 1-sample, 1-interval ``Tracks`` per
named track, with default ``Repeat5p`` insertion-fill unless overridden.
Callers that need more elaborate shapes should add a new builder rather
than parametrize this one.
"""

from __future__ import annotations

import numpy as np
from genvarloader._dataset._insertion_fill import InsertionFill, Repeat5p
from genvarloader._dataset._reconstruct import Tracks, TrackType
from genvarloader._ragged import RaggedIntervals, RaggedTracks
from seqpro.rag import Ragged


def make_tracks(
    names: list[str],
    insertion_fill: dict[str, InsertionFill] | None = None,
) -> Tracks:
    """Build a minimal ``Tracks`` instance for plumbing tests.

    Each name produces a 1-region, 1-sample track with a single dummy
    interval at position 0. The default ``insertion_fill`` is
    ``{name: Repeat5p()}`` for every name; pass an explicit mapping to
    override.

    Example::

        tracks = make_tracks(["a", "b"])
        tracks.insertion_fill["a"]  # Repeat5p()

        tracks = make_tracks(["a"], insertion_fill={"a": Constant(0.0)})
    """
    starts = ends = values = np.array([0], dtype=np.int32)
    offsets = np.array([0, 1], dtype=np.int64)
    intervals = {
        n: RaggedIntervals(
            Ragged.from_offsets(starts, (1, None), offsets),
            Ragged.from_offsets(ends, (1, None), offsets),
            Ragged.from_offsets(values, (1, None), offsets),
        )
        for n in names
    }
    active = {n: TrackType.SAMPLE for n in names}
    if insertion_fill is None:
        insertion_fill = {n: Repeat5p() for n in names}
    return Tracks(
        intervals=intervals,
        active_tracks=active,
        available_tracks=active,
        kind=RaggedTracks,
        n_regions=1,
        n_samples=1,
        insertion_fill=insertion_fill,
    )
