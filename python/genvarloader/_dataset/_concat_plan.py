"""Pure planning for :func:`genvarloader.concat` — no IO.

A GVL dataset stores parallel ragged arrays over an ``(R, S[, P])`` C-order grid;
the flat slot for ``(r, s, p)`` is ``((r * S) + s) * P + p``. Merging two or more
datasets means deciding, for each *merged* flat slot, which input dataset and which
*source* flat slot it comes from. That mapping is the provenance map, and
:func:`coalesce` compresses it into maximal contiguous runs so the IO layer can move
large byte ranges instead of individual slots.
"""

from __future__ import annotations

from typing import Iterable, Iterator, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CONCAT_CHUNK_BYTES",
    "ExplicitRunPlan",
    "Run",
    "RunPlan",
    "as_plan",
    "coalesce",
    "provenance",
]

CONCAT_CHUNK_BYTES = 16 << 20
"""Buffered-IO chunk size. 16 MiB is the measured knee on NFSv3; 64 MiB is no better."""

_SLOT_BATCH_SLOTS = 1 << 20
"""Slots per `slot_batches` chunk on the regions axis: 8 MiB of int64 indices.

A regions-axis run can cover the whole merged grid (4.0e9 slots at chr22), so
emitting one batch per run would rebuild exactly the array this module exists to
avoid. The samples axis is naturally bounded at `n_merged * ploidy` instead.
"""


def _default_order(n_ds: int, counts: list[int]) -> NDArray[np.int64]:
    """Block-concatenation order: dataset 0's whole block, then dataset 1's, etc.

    Args:
        n_ds: Number of input datasets.
        counts: Positions along the merged axis contributed by each dataset.

    Returns:
        An ``(sum(counts), 2)`` int64 array of ``(dataset_idx, within_idx)``.
    """
    if n_ds == 0:
        return np.zeros((0, 2), np.int64)
    ds_col = np.repeat(np.arange(n_ds, dtype=np.int64), counts)
    w_col = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])
    return np.stack([ds_col, w_col], axis=1)


class Run(NamedTuple):
    """A maximal contiguous span of merged slots drawn from one source dataset.

    Attributes:
        src: Index into the input dataset list.
        src_start: First source flat slot, inclusive.
        src_stop: Last source flat slot, exclusive.
        dst_start: First merged flat slot, inclusive.
    """

    src: int
    src_start: int
    src_stop: int
    dst_start: int


def provenance(
    axis: str,
    shape_per_ds: list[tuple[int, int]],
    ploidy: int,
    *,
    order: "NDArray[np.int64] | None" = None,
) -> NDArray[np.int64]:
    """Map each merged flat slot to its ``(dataset, source flat slot)`` origin.

    Without ``order``, merged positions along ``axis`` are laid out as dataset 0's
    whole block, then dataset 1's, etc. (block-concatenation) — the default is
    correct only when the inputs' key ranges do not interleave along ``axis``.

    ``order`` overrides that layout: it gives, for each *merged* position along
    ``axis`` (in destination order), which input dataset and which position
    *within that dataset's own axis* it came from. This supports merges where
    the true sorted order interleaves inputs (e.g. samples ``[s0, s2]`` +
    ``[s1]``, or regions spanning natsort-interleaving contigs) — every on-disk
    GVL store is sorted along both axes, so a block layout is wrong whenever the
    shards' key ranges interleave.

    Args:
        axis: Either ``"regions"`` or ``"samples"``.
        shape_per_ds: ``(n_regions, n_samples)`` per input dataset, in input order.
        ploidy: Slots per ``(region, sample)`` cell. Pass ``1`` for interval stores,
            which have no ploidy axis.
        order: ``(n_merged_along_axis, 2)`` int array of ``(dataset_idx,
            within_dataset_idx)`` per merged position along ``axis``, in
            destination order. ``None`` reproduces block-concatenation.

    Returns:
        An ``(n_slots, 2)`` int64 array; column 0 is the dataset index and column 1
        is the flat slot within that dataset.

    Raises:
        ValueError: If ``axis`` is not ``"regions"`` or ``"samples"``.
    """
    if axis not in ("regions", "samples"):
        raise ValueError(f'axis must be "regions" or "samples", got {axis!r}')

    n_ds = len(shape_per_ds)

    if axis == "regions":
        # Merged region i's S*P slots are contiguous in both source and
        # destination: they're `order[i, 1] * S*P .. +S*P` in dataset
        # `order[i, 0]`'s flat layout, and `i * S*P .. +S*P` in the merged one.
        n_samples = shape_per_ds[0][1]
        cell = n_samples * ploidy
        if order is None:
            order = _default_order(n_ds, [shape_per_ds[d][0] for d in range(n_ds)])
        else:
            order = np.asarray(order, dtype=np.int64)

        n_merged = len(order)
        out = np.empty((n_merged * cell, 2), np.int64)
        out[:, 0] = np.repeat(order[:, 0], cell)
        base = order[:, 1] * cell
        out[:, 1] = np.repeat(base, cell) + np.tile(
            np.arange(cell, dtype=np.int64), n_merged
        )
        return out

    # axis == "samples": regions are shared; per region, lay out each merged
    # sample's ploidy slots. Each dataset has its own sample count `S_d`, so a
    # merged sample's source slot depends on which dataset it came from.
    n_regions = shape_per_ds[0][0]
    per_ds_samples = [s for _, s in shape_per_ds]
    if order is None:
        order = _default_order(n_ds, per_ds_samples)
    else:
        order = np.asarray(order, dtype=np.int64)

    ds_of = order[:, 0]
    w_of = order[:, 1]
    s_d = np.asarray(per_ds_samples, dtype=np.int64)[ds_of]

    r_idx = np.arange(n_regions, dtype=np.int64)
    # starts[r, j] = flat slot (before ploidy) of merged sample j's cell in
    # region r, within its own source dataset.
    starts = (r_idx[:, None] * s_d[None, :] + w_of[None, :]) * ploidy
    p_idx = np.arange(ploidy, dtype=np.int64)
    slot = starts[:, :, None] + p_idx[None, None, :]
    ds_expand = np.broadcast_to(ds_of[None, :, None], slot.shape)

    out = np.empty((slot.size, 2), np.int64)
    out[:, 0] = ds_expand.reshape(-1)
    out[:, 1] = slot.reshape(-1)
    return out


def coalesce(prov: NDArray[np.int64]) -> list[Run]:
    """Compress a provenance map into maximal contiguous runs.

    A run is a maximal span of consecutive merged slots over which the source
    dataset is constant and the source slot increases by exactly 1. Iterating the
    returned runs in order walks the destination sequentially *and* each source
    monotonically, so both sides of the copy stay sequential.

    Args:
        prov: An ``(n_slots, 2)`` provenance map from :func:`provenance`.

    Returns:
        Runs in destination order. Empty if ``prov`` has no rows.
    """
    if len(prov) == 0:
        return []

    src = prov[:, 0]
    slot = prov[:, 1]
    # A new run starts wherever the dataset changes or the source slot jumps.
    breaks = (src[1:] != src[:-1]) | (slot[1:] != slot[:-1] + 1)
    starts = np.concatenate([[0], np.flatnonzero(breaks) + 1])
    stops = np.concatenate([starts[1:], [len(prov)]])

    return [
        Run(
            src=int(src[a]),
            src_start=int(slot[a]),
            src_stop=int(slot[a]) + int(b - a),
            dst_start=int(a),
        )
        for a, b in zip(starts, stops)
    ]


class RunPlan:
    """Destination-ordered runs, derived from ``order`` without materializing slots.

    Equivalent to ``coalesce(provenance(axis, shape_per_ds, ploidy, order=order))``
    and pinned against it by
    ``test_run_plan_matches_coalesce_provenance_exhaustively``, but it never
    builds the ``(n_slots, 2)`` map: at the All of Us chr22 grid that map is
    32-64 GB and the run list it compresses to is 384-768 GB, because on an
    interleaved sample merge every run is one slot long.

    The whole thing rests on one property: the run-break condition is "source
    dataset changes, or source slot does not increment", and on both axes that
    reduces to a predicate over ``order`` alone -- the region index ``r`` cancels
    out. So the break pattern is computed once, in ``O(R + S)``, and the runs
    stream in ``O(1)``.

    **Re-iterable on purpose, not a generator.** ``copy_runs`` iterates runs
    twice (once for offsets, once to stream bytes) and so does
    ``_gather_svar_offsets``; a one-shot iterator would yield an empty second
    pass and silently truncate the output rather than raise.

    Args:
        axis: Either ``"regions"`` or ``"samples"``.
        shape_per_ds: ``(n_regions, n_samples)`` per input dataset, in input order.
        ploidy: Slots per ``(region, sample)`` cell. Pass ``1`` for interval
            stores, which have no ploidy axis.
        order: ``(n_merged_along_axis, 2)`` int array of ``(dataset_idx,
            within_dataset_idx)`` per merged position along ``axis``, in
            destination order. ``None`` reproduces block-concatenation.

    Raises:
        ValueError: If ``axis`` is not ``"regions"`` or ``"samples"``.
    """

    def __init__(
        self,
        axis: str,
        shape_per_ds: list[tuple[int, int]],
        ploidy: int,
        *,
        order: "NDArray[np.int64] | None" = None,
    ) -> None:
        if axis not in ("regions", "samples"):
            raise ValueError(f'axis must be "regions" or "samples", got {axis!r}')
        self.axis = axis
        self.ploidy = int(ploidy)
        self.shape_per_ds = [(int(r), int(s)) for r, s in shape_per_ds]
        n_ds = len(self.shape_per_ds)
        if order is None:
            counts = [r if axis == "regions" else s for r, s in self.shape_per_ds]
            self.order = _default_order(n_ds, counts)
        else:
            self.order = np.asarray(order, dtype=np.int64).reshape(-1, 2)

    @property
    def n_slots(self) -> int:
        """Total merged flat slots this plan covers, computed arithmetically."""
        if not self.shape_per_ds:
            return 0
        if self.axis == "regions":
            return len(self.order) * self.shape_per_ds[0][1] * self.ploidy
        return self.shape_per_ds[0][0] * len(self.order) * self.ploidy

    def slot_batches(
        self,
    ) -> "Iterator[tuple[int, NDArray[np.int64], NDArray[np.int64]]]":
        """Yield ``(dst_start, src_ds, src_slots)`` batches in destination order.

        Each batch describes a destination-contiguous span: ``src_ds[i]`` and
        ``src_slots[i]`` are the origin of merged slot ``dst_start + i``.
        Concatenating every batch in order rebuilds :func:`provenance`'s output
        exactly, which is what pins this method.

        Yields:
            ``(dst_start, src_ds, src_slots)``, where the two arrays are int64
            and equal in length.
        """
        if self.axis == "regions":
            for run in self:
                pos, dst = run.src_start, run.dst_start
                while pos < run.src_stop:
                    n = min(_SLOT_BATCH_SLOTS, run.src_stop - pos)
                    yield (
                        dst,
                        np.full(n, run.src, np.int64),
                        np.arange(pos, pos + n, dtype=np.int64),
                    )
                    pos += n
                    dst += n
            return

        n_regions = self.shape_per_ds[0][0] if self.shape_per_ds else 0
        n_merged = len(self.order)
        if n_regions == 0 or n_merged == 0 or self.ploidy == 0:
            return
        per_ds_samples = np.asarray([s for _, s in self.shape_per_ds], np.int64)
        ds, w = self.order[:, 0], self.order[:, 1]
        s_d = per_ds_samples[ds]
        p = np.arange(self.ploidy, dtype=np.int64)
        # `order` is per merged SAMPLE; each contributes `ploidy` adjacent slots.
        ds_vec = np.repeat(ds, self.ploidy)
        for r in range(n_regions):
            base = (r * s_d + w) * self.ploidy
            slots = (base[:, None] + p[None, :]).reshape(-1)
            yield (r * n_merged * self.ploidy, ds_vec, slots)

    def _segments(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Half-open ``[start, stop)`` spans of ``order`` with no break inside."""
        ds, w = self.order[:, 0], self.order[:, 1]
        brk = (ds[1:] != ds[:-1]) | (w[1:] != w[:-1] + 1)
        starts = np.concatenate([np.zeros(1, np.int64), np.flatnonzero(brk) + 1])
        stops = np.concatenate([starts[1:], np.array([len(self.order)], np.int64)])
        return starts, stops

    def __iter__(self) -> "Iterator[Run]":
        if self.axis == "regions":
            yield from self._iter_regions()
        else:
            yield from self._iter_samples()

    def _iter_regions(self) -> "Iterator[Run]":
        # Merged region i owns one contiguous S*P block in BOTH source and
        # destination, so a segment of `order` maps to exactly one run and no
        # carry is needed: adjacent segments are non-contiguous by construction.
        cell = self.shape_per_ds[0][1] * self.ploidy if self.shape_per_ds else 0
        if len(self.order) == 0 or cell == 0:
            return
        ds, w = self.order[:, 0], self.order[:, 1]
        for a, b in zip(*self._segments()):
            yield Run(
                src=int(ds[a]),
                src_start=int(w[a]) * cell,
                src_stop=(int(w[a]) + int(b - a)) * cell,
                dst_start=int(a) * cell,
            )

    def _iter_samples(self) -> "Iterator[Run]":
        n_regions = self.shape_per_ds[0][0] if self.shape_per_ds else 0
        n_merged = len(self.order)
        if n_regions == 0 or n_merged == 0 or self.ploidy == 0:
            return
        per_ds_samples = np.asarray([s for _, s in self.shape_per_ds], np.int64)
        ds, w = self.order[:, 0], self.order[:, 1]
        seg_starts, seg_stops = self._segments()

        # One pending run, extended whenever the next segment continues it in
        # both source and destination. This is what makes a run that spans a
        # region boundary come out merged, with no special case for that
        # boundary -- verified against the oracle at 2,880 configurations.
        pending: Run | None = None
        for r in range(n_regions):
            for a, b in zip(seg_starts, seg_stops):
                d = int(ds[a])
                s_d = int(per_ds_samples[d])
                src_start = (r * s_d + int(w[a])) * self.ploidy
                src_stop = (r * s_d + int(w[b - 1]) + 1) * self.ploidy
                dst_start = (r * n_merged + int(a)) * self.ploidy
                if (
                    pending is not None
                    and pending.src == d
                    and pending.src_stop == src_start
                    and pending.dst_start + (pending.src_stop - pending.src_start)
                    == dst_start
                ):
                    pending = Run(d, pending.src_start, src_stop, pending.dst_start)
                else:
                    if pending is not None:
                        yield pending
                    pending = Run(d, src_start, src_stop, dst_start)
        if pending is not None:
            yield pending


class ExplicitRunPlan:
    """A hand-built run list presented through :class:`RunPlan`'s interface.

    The IO layer takes either this or a :class:`RunPlan`, so its unit tests can
    exercise streaming with a two-run list without constructing a merge.

    Args:
        runs: Destination-ordered runs. Materialized, so it may be any iterable.
    """

    def __init__(self, runs: "Iterable[Run]") -> None:
        self._runs = list(runs)

    def __iter__(self) -> "Iterator[Run]":
        return iter(self._runs)

    @property
    def n_slots(self) -> int:
        """Total merged flat slots covered by the run list."""
        return sum(r.src_stop - r.src_start for r in self._runs)

    def slot_batches(
        self,
    ) -> "Iterator[tuple[int, NDArray[np.int64], NDArray[np.int64]]]":
        """Yield one ``(dst_start, src_ds, src_slots)`` batch per run."""
        for r in self._runs:
            n = r.src_stop - r.src_start
            yield (
                r.dst_start,
                np.full(n, r.src, np.int64),
                np.arange(r.src_start, r.src_stop, dtype=np.int64),
            )


def as_plan(
    runs: "RunPlan | ExplicitRunPlan | Sequence[Run]",
) -> "RunPlan | ExplicitRunPlan":
    """Normalize a run source so the IO layer has one consuming path.

    Args:
        runs: A plan, or a re-iterable sequence of runs. A one-shot generator is
            deliberately not accepted: ``copy_runs`` iterates its runs twice.

    Returns:
        ``runs`` itself when it is already a plan, else an
        :class:`ExplicitRunPlan` wrapping it.
    """
    if isinstance(runs, (RunPlan, ExplicitRunPlan)):
        return runs
    return ExplicitRunPlan(runs)
