from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl
import seqpro as sp
from genoray._contigs import ContigNormalizer
from numpy.typing import NDArray

from ._utils import bed_to_regions


@dataclass(frozen=True, slots=True)
class StreamingDataset:
    """Write-free, iterable-only dataset. Region-major iteration; no random access."""

    _bed: pl.DataFrame
    _regions: NDArray[np.int32]  # (n_regions, 3) sorted (contig_idx, start, end)
    _sort_order: NDArray[np.intp]  # maps sorted position -> original bed row
    contigs: list[str]
    n_samples: int
    ploidy: int
    _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
    _batch_size: int = 1

    def __init__(self, regions, *, contigs, n_samples, ploidy, _reconstruct_window):
        bed = regions if isinstance(regions, pl.DataFrame) else sp.bed.read(regions)
        sorted_bed = sp.bed.sort(bed)
        # record original-row order so emitted indices refer to the user's input order
        order = (
            bed.with_row_index("_r")
            .join(
                sorted_bed.with_row_index("_sorted"), on=list(bed.columns), how="right"
            )
            .sort("_sorted")["_r"]
            .to_numpy()
            .astype(np.intp)
        )
        regs = bed_to_regions(sorted_bed, ContigNormalizer(contigs))
        object.__setattr__(self, "_bed", bed)
        object.__setattr__(self, "_regions", regs)
        object.__setattr__(self, "_sort_order", order)
        object.__setattr__(self, "contigs", list(contigs))
        object.__setattr__(self, "n_samples", int(n_samples))
        object.__setattr__(self, "ploidy", int(ploidy))
        object.__setattr__(self, "_reconstruct_window", _reconstruct_window)
        object.__setattr__(self, "_batch_size", 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._regions), self.n_samples)

    def __len__(self) -> int:
        return len(self._regions) * self.n_samples

    def _with_batch_size(self, batch_size: int) -> "StreamingDataset":
        # dataclasses.replace() would re-invoke __init__ (which doesn't accept
        # every field as a kwarg), so shallow-copy and mutate the frozen instance.
        new = copy.copy(self)
        object.__setattr__(new, "_batch_size", int(batch_size))
        return new

    def _plan(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        # region-major flat index over (n_regions, n_samples): sample varies fastest.
        n_regions, n_samples = self.shape
        flat = np.arange(n_regions * n_samples, dtype=np.intp)
        for start in range(0, flat.size, self._batch_size):
            chunk = flat[start : start + self._batch_size]
            r_idx, s_idx = np.unravel_index(chunk, (n_regions, n_samples))
            yield r_idx.astype(np.intp), s_idx.astype(np.intp)

    def __iter__(self) -> Iterator[tuple]:
        for r_idx, s_idx in self._plan():
            data = self._reconstruct_window(r_idx, s_idx)
            # map sorted region positions back to the user's original bed rows
            yield (data, self._sort_order[r_idx], s_idx)
