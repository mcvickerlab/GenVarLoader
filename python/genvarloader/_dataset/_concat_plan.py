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
) -> NDArray[np.int64]:
    """Map each merged flat slot to its ``(dataset, source flat slot)`` origin.

    Args:
        axis: Either ``"regions"`` or ``"samples"``.
        shape_per_ds: ``(n_regions, n_samples)`` per input dataset, in input order.
        ploidy: Slots per ``(region, sample)`` cell. Pass ``1`` for interval stores,
            which have no ploidy axis.

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
        n_samples = shape_per_ds[0][1]
        blocks = []
        for d in range(n_ds):
            n_slots = shape_per_ds[d][0] * n_samples * ploidy
            block = np.empty((n_slots, 2), np.int64)
            block[:, 0] = d
            block[:, 1] = np.arange(n_slots, dtype=np.int64)
            blocks.append(block)
        return np.concatenate(blocks, axis=0)

    # axis == "samples": regions are shared; per region, lay out each dataset's
    # samples in input-dataset order.
    n_regions = shape_per_ds[0][0]
    per_ds_samples = [s for _, s in shape_per_ds]
    cell_blocks = []
    for d, n_s in enumerate(per_ds_samples):
        block = np.empty((n_s * ploidy, 2), np.int64)
        block[:, 0] = d
        cell_blocks.append(block)

    out = []
    for r in range(n_regions):
        for d, n_s in enumerate(per_ds_samples):
            block = cell_blocks[d].copy()
            base = r * n_s * ploidy
            block[:, 1] = np.arange(base, base + n_s * ploidy, dtype=np.int64)
            out.append(block)
    return np.concatenate(out, axis=0)


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
