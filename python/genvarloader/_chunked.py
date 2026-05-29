"""Chunk planner: groups (r, s) pairs into per-slot chunks aligned to
mini-batch boundaries."""
from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray


class ChunkPlanner:
    """Plan chunks for the prefetching dataloader.

    Walks the (r_idx, s_idx) sequence in order; accumulates
    bytes_per_instance[r, s] in mini-batch increments. When the next mini-batch
    would push the running sum above slot_bytes, closes the chunk on the
    nearest mini-batch boundary.

    Iteration yields (chunk_r_idx, chunk_s_idx, n_batches_in_chunk).

    After construction, .peak_chunk_bytes holds the maximum chunk byte size
    (computed eagerly in __init__ via a non-consuming pre-pass, so it is
    available before any iteration).
    """

    def __init__(
        self,
        r_idx: NDArray[np.integer],
        s_idx: NDArray[np.integer],
        batch_size: int,
        bytes_per_instance: NDArray[np.int64],
        slot_bytes: int,
    ) -> None:
        if len(r_idx) != len(s_idx):
            raise ValueError("r_idx and s_idx must have the same length")
        n = len(r_idx)
        if n % batch_size != 0:
            raise ValueError(
                f"len(r_idx)={n} is not a multiple of batch_size={batch_size}. "
                "Use drop_last or pad the sampler before passing to ChunkPlanner."
            )
        self.r_idx = np.asarray(r_idx)
        self.s_idx = np.asarray(s_idx)
        self.batch_size = batch_size
        self.bytes_per_instance = bytes_per_instance
        self.slot_bytes = int(slot_bytes)

        per_inst = bytes_per_instance[self.r_idx, self.s_idx].astype(np.int64)
        batch_totals = per_inst.reshape(-1, batch_size).sum(-1)
        too_big = batch_totals > self.slot_bytes
        if too_big.any():
            offender = int(np.argmax(too_big))
            raise ValueError(
                f"Mini-batch {offender} totals {int(batch_totals[offender])} bytes "
                f"which exceeds slot_bytes={self.slot_bytes}. "
                f"Either lower batch_size or raise buffer_bytes."
            )
        self._batch_totals = batch_totals

        # Compute peak_chunk_bytes before __iter__ so callers can size shm slots
        # at construction time without consuming the iterator.
        self.peak_chunk_bytes: int = self._compute_peak_chunk_bytes()

    def _compute_peak_chunk_bytes(self) -> int:
        n_batches = len(self._batch_totals)
        peak = 0
        i = 0
        while i < n_batches:
            running = 0
            j = i
            while j < n_batches and running + int(self._batch_totals[j]) <= self.slot_bytes:
                running += int(self._batch_totals[j])
                j += 1
            assert j > i  # guaranteed by per-batch validation
            if running > peak:
                peak = running
            i = j
        return peak

    def __iter__(self) -> Iterator[tuple[NDArray[np.integer], NDArray[np.integer], int]]:
        n_batches = len(self._batch_totals)
        i = 0
        while i < n_batches:
            running = 0
            j = i
            while j < n_batches and running + int(self._batch_totals[j]) <= self.slot_bytes:
                running += int(self._batch_totals[j])
                j += 1
            assert j > i  # at least one batch per chunk, guaranteed by per-batch validation
            start = i * self.batch_size
            end = j * self.batch_size
            yield self.r_idx[start:end], self.s_idx[start:end], j - i
            i = j


def slice_chunk(chunk_output, batch_size: int):
    """Yield mini-batches of size ``batch_size`` from a chunk-shaped output.

    Supports ndarray, seqpro.rag.Ragged, RaggedAnnotatedHaps, AnnotatedHaps,
    RaggedVariants (ak.Array subclass), and tuples thereof.
    """
    from seqpro.rag import Ragged
    from ._types import AnnotatedHaps
    from ._ragged import RaggedAnnotatedHaps
    from ._dataset._rag_variants import RaggedVariants
    import awkward as ak

    def _len(arr) -> int:
        """Return the outer (instance) dimension length."""
        if isinstance(arr, (np.ndarray, Ragged, RaggedAnnotatedHaps)):
            return arr.shape[0]
        if isinstance(arr, ak.Array):
            # RaggedVariants is an ak.Array subclass; len() works.
            return len(arr)
        if isinstance(arr, AnnotatedHaps):
            return arr.haps.shape[0]
        raise TypeError(f"slice_chunk: cannot determine length of {type(arr)}")

    def _slice_one(arr, start: int, stop: int):
        if isinstance(arr, np.ndarray):
            return arr[start:stop]
        if isinstance(arr, Ragged):
            return arr[start:stop]
        if isinstance(arr, RaggedAnnotatedHaps):
            return RaggedAnnotatedHaps(
                haps=arr.haps[start:stop],
                var_idxs=arr.var_idxs[start:stop],
                ref_coords=arr.ref_coords[start:stop],
            )
        if isinstance(arr, AnnotatedHaps):
            return AnnotatedHaps(
                haps=_slice_one(arr.haps, start, stop),
                var_idxs=_slice_one(arr.var_idxs, start, stop),
                ref_coords=_slice_one(arr.ref_coords, start, stop),
            )
        if isinstance(arr, ak.Array):
            # Covers RaggedVariants (ak.Array subclass) — slicing preserves the type.
            return arr[start:stop]
        raise TypeError(f"slice_chunk: unsupported array type {type(arr)}")

    is_tuple = isinstance(chunk_output, tuple)
    arrs = chunk_output if is_tuple else (chunk_output,)
    n = _len(arrs[0])
    for start in range(0, n, batch_size):
        stop = start + batch_size
        sliced = tuple(_slice_one(a, start, stop) for a in arrs)
        yield sliced if is_tuple else sliced[0]
