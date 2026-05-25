"""In-memory ragged-array builders for tests.

Two helpers cover the two shapes that ragged-component tests need:

- ``make_ragged_seqs`` — wrap a list of byte rows as a ``Ragged[S1]``.
- ``make_ragged_intervals`` — wrap a list-of-lists of (start, end, value)
  triples as a ``RaggedIntervals`` (the three-Ragged-array dataclass used
  by BigWig output and track plumbing).

Neither helper opens a Dataset or touches disk. They exist so unit tests
can construct synthetic ragged inputs without re-implementing the
``Ragged.from_lengths`` boilerplate inline.
"""

from __future__ import annotations

import numpy as np
from seqpro.rag import Ragged

from genvarloader._ragged import RaggedIntervals


def make_ragged_seqs(rows: list[bytes]) -> Ragged:
    """Build a ``Ragged[S1]`` from a list of byte-string rows.

    Each row becomes one ragged element; lengths are derived from each
    row's byte length.

    Example::

        rag = make_ragged_seqs([b"ACGT", b"NN", b"GGGG"])
        rag.lengths  # array([4, 2, 4], dtype=int32)
        rag.data     # ascii bytes for "ACGTNNGGGG" as S1
    """
    if not rows:
        data = np.empty(0, dtype="S1")
        lengths = np.empty(0, dtype=np.int32)
        return Ragged.from_lengths(data, lengths)

    joined = b"".join(rows)
    data = np.frombuffer(joined, dtype="S1")
    lengths = np.array([len(r) for r in rows], dtype=np.int32)
    return Ragged.from_lengths(data, lengths)


def make_ragged_intervals(
    per_region: list[list[tuple[int, int, float]]],
) -> RaggedIntervals:
    """Build a ``RaggedIntervals`` from a per-region list of intervals.

    ``per_region[i]`` is the list of ``(start, end, value)`` triples for
    region ``i``. Empty inner lists are allowed (region with no
    intervals).

    Example::

        rag = make_ragged_intervals([
            [(0, 10, 1.0), (10, 20, 2.0)],
            [],
            [(5, 15, 0.5)],
        ])
        rag.starts.lengths  # array([2, 0, 1], dtype=int32)
    """
    lengths = np.array([len(r) for r in per_region], dtype=np.int32)
    flat_starts: list[int] = []
    flat_ends: list[int] = []
    flat_values: list[float] = []
    for region in per_region:
        for start, end, value in region:
            flat_starts.append(start)
            flat_ends.append(end)
            flat_values.append(value)

    starts = Ragged.from_lengths(np.asarray(flat_starts, dtype=np.int32), lengths)
    ends = Ragged.from_lengths(np.asarray(flat_ends, dtype=np.int32), lengths)
    values = Ragged.from_lengths(np.asarray(flat_values, dtype=np.float32), lengths)
    return RaggedIntervals(starts, ends, values)
