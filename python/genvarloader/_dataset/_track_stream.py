"""Streaming read backend for interval-valued tracks (issue #279).

One backend serves every :class:`~genvarloader._types.IntervalTrack` -- both
:class:`~genvarloader.BigWigs` and :class:`~genvarloader.Table` -- because the
Protocol is the only surface this module touches. Nothing here branches on the
concrete class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._ragged import RaggedIntervals
from .._utils import lengths_to_offsets, normalize_contig_name

if TYPE_CHECKING:
    from .._types import IntervalTrack


class _TrackBackend:
    """Window-granular interval reads over the ``IntervalTrack`` Protocol.

    The window (regions x sample-chunk) is the READ granularity, mirroring the
    variant backends' ``read_window``/``generate_batch`` split. Every call is
    single-contig, which ``_plan()``'s contig-run outer loop guarantees.

    Args:
        tracks: One or more interval tracks. Stored sorted by ``name`` -- the
            written path's track axis comes from a sorted directory listing
            (``_tracks.py:283``), NOT from argument order, so sorting here is
            what makes byte-parity hold.
        regions: ``(n_regions, 4)`` sorted regions, as held by
            ``StreamingDataset._regions``. Only columns 0-2 are read.
        contigs: Contig names indexed by ``regions[:, 0]``.
        samples: Dataset sample names in public (sorted) order.

    Raises:
        ValueError: If two tracks share a name, if a track does not cover a
            contig the BED references, or if a track is missing a dataset
            sample.
    """

    def __init__(
        self,
        tracks: Sequence["IntervalTrack"],
        regions: NDArray[np.int32],
        contigs: list[str],
        samples: list[str],
    ) -> None:
        _tracks = list(tracks)
        names = [t.name for t in _tracks]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Duplicate track name(s) {dupes}. Track names must be unique:"
                " the written layout stores each track under intervals/<name>/,"
                " so duplicates are indistinguishable."
            )
        # Name-sorted, matching the written path's `available_tracks.sort()`.
        order = sorted(range(len(_tracks)), key=lambda i: names[i])
        self._tracks = [_tracks[i] for i in order]
        self.names = [names[i] for i in order]

        self._regions = regions
        self._contigs = contigs
        self._samples = samples

        # Contig coverage, validated ONCE. The two implementations fail
        # differently at read time -- BigWigs raises, Table silently returns
        # zeros (src/tables.rs:93-95) -- so neither is an acceptable guard.
        used = {contigs[i] for i in np.unique(regions[:, 0])}
        for track in self._tracks:
            missing = sorted(
                c for c in used if normalize_contig_name(c, track.contigs) is None
            )
            if missing:
                raise ValueError(
                    f"Track {track.name!r} does not cover contig(s) {missing},"
                    " which the BED references."
                )
            absent = sorted(set(samples) - set(track.samples))
            if absent:
                raise ValueError(
                    f"Sample(s) {absent} are not present in track"
                    f" {track.name!r}. Tracks may have extra samples, but must"
                    " cover every sample in the dataset."
                )

    def read_window(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        starts: NDArray[np.int32] | None = None,
        ends: NDArray[np.int32] | None = None,
    ) -> list[RaggedIntervals]:
        """Read one window's intervals for every track, single-contig.

        Args:
            r_idx: Region indices, all on one contig.
            s_idx: Public sample indices (into the dataset's sorted samples).
            starts: Query starts overriding the stored region starts. Under
                `jitter > 0` the caller MUST pass the same translated bounds
                the variant engine got, or tracks and haplotypes silently
                disagree by up to `jitter` bases. `None` uses the regions.
            ends: Query ends, same contract as `starts`.

        Returns:
            One ``RaggedIntervals`` per track in ``self.names`` order, each of
            shape ``(len(r_idx), len(s_idx), None)``.

        Raises:
            ValueError: If the window spans more than one contig.
        """
        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)

        if r_idx.size == 0:
            raise ValueError(
                "_TrackBackend.read_window: r_idx is empty; there is no region to"
                " determine the contig from."
            )

        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError(
                "_TrackBackend.read_window: window spans multiple contigs;"
                " every track query must be single-contig."
            )
        contig = self._contigs[contig_idx]
        if starts is None:
            starts = self._regions[r_idx, 1]
        if ends is None:
            ends = self._regions[r_idx, 2]
        starts = np.ascontiguousarray(starts, np.int32)
        ends = np.ascontiguousarray(ends, np.int32)

        # Query by NAME, never by index: BigWigs.samples is dict-insertion
        # order (_bigwig.py:41) while Table.samples is sorted (_table.py:58),
        # so a positional index means different samples in the two classes.
        # This is the track-side analogue of _Svar1Backend._phys_sample_idx.
        names = [self._samples[int(i)] for i in s_idx]

        out: list[RaggedIntervals] = []
        for track in self._tracks:
            # Clamp to the track's OWN contig length. `ends` may have been
            # extended past the region end to read ahead for deletions (issue
            # #279 spec section 3.2), and that extension can run off the contig.
            # The clamp cannot lose data: a track's header DEFINES its contig
            # lengths, so a query past the contig end has nothing to return --
            # only an out-of-bounds error (BigWigs raises; Table silently
            # returns zeros). NOTE: `gvl.write` does NOT clamp its bed --
            # `_prep_bed` (`_write.py:649-679`) only sorts, adds a strand
            # column, and EXPANDS by `max_jitter` -- so do not justify this by
            # appeal to the written path; the written path simply has no
            # intervals out there either, for the same header reason. Done per
            # track, not once per window, because two tracks may disagree about
            # a contig's length and each must be queried within its own bounds.
            t_contig = normalize_contig_name(contig, track.contigs)
            if t_contig is None:  # pragma: no cover - validated in __init__
                raise AssertionError(
                    f"track {track.name!r} lost contig {contig!r} after"
                    " construction-time validation"
                )
            t_ends = np.minimum(ends, np.int32(track.contigs[t_contig]))
            # A region starting at or past the contig end would otherwise
            # produce an inverted query; keep the interval empty instead.
            t_ends = np.ascontiguousarray(np.maximum(t_ends, starts), np.int32)
            counts = track.count_intervals(contig, starts, t_ends, sample=names)
            offsets = lengths_to_offsets(np.asarray(counts).ravel())
            out.append(
                track._intervals_from_offsets(
                    contig, starts, t_ends, offsets, sample=names
                )
            )
        return out
