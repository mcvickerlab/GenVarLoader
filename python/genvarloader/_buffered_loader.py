"""mode='buffered' dataloader path: synchronous chunked fetch in main process."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._chunked import ChunkPlanner, slice_chunk

if TYPE_CHECKING:
    import torch.utils.data as td

    from ._dataset._impl import Dataset


def make_buffered_dataset(
    dataset: "Dataset",
    batch_size: int,
    slot_bytes: int,
    bytes_per_instance: np.ndarray,
    flat_r: np.ndarray,
    flat_s: np.ndarray,
) -> "td.IterableDataset":
    import torch.utils.data as td

    class BufferedTorchDataset(td.IterableDataset):
        def __init__(self) -> None:
            self._dataset = dataset
            self._batch_size = batch_size
            self._planner = ChunkPlanner(
                r_idx=flat_r,
                s_idx=flat_s,
                batch_size=batch_size,
                bytes_per_instance=bytes_per_instance,
                slot_bytes=slot_bytes,
            )

        def __iter__(self):
            for chunk_r, chunk_s, _n in self._planner:
                chunk = self._dataset[chunk_r, chunk_s]
                yield from slice_chunk(chunk, self._batch_size)

        def __len__(self) -> int:
            return len(flat_r) // batch_size

    return BufferedTorchDataset()
