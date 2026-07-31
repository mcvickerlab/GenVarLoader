"""Pure planning for :func:`genvarloader.concat` — no IO.

A GVL dataset stores parallel ragged arrays over an ``(R, S[, P])`` C-order grid;
the flat slot for ``(r, s, p)`` is ``((r * S) + s) * P + p``. Merging two or more
datasets means deciding, for each *merged* flat slot, which input dataset and which
*source* flat slot it comes from. That mapping is the provenance map, and
:func:`coalesce` compresses it into maximal contiguous runs so the IO layer can move
large byte ranges instead of individual slots.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["CONCAT_CHUNK_BYTES", "Run", "coalesce", "provenance"]

CONCAT_CHUNK_BYTES = 16 << 20
"""Buffered-IO chunk size. 16 MiB is the measured knee on NFSv3; 64 MiB is no better."""


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

    def _default_order(counts: list[int]) -> NDArray[np.int64]:
        ds_col = np.repeat(np.arange(n_ds, dtype=np.int64), counts)
        w_col = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])
        return np.stack([ds_col, w_col], axis=1)

    if axis == "regions":
        # Merged region i's S*P slots are contiguous in both source and
        # destination: they're `order[i, 1] * S*P .. +S*P` in dataset
        # `order[i, 0]`'s flat layout, and `i * S*P .. +S*P` in the merged one.
        n_samples = shape_per_ds[0][1]
        cell = n_samples * ploidy
        if order is None:
            order = _default_order([shape_per_ds[d][0] for d in range(n_ds)])
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
        order = _default_order(per_ds_samples)
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
