from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray

from .types import Reader


class Zarr(Reader):
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.root = zarr.open_group(path)
        arr = self.root["data"]
        self.bytes_per_length = arr.itemsize * np.prod(arr.shape[:-1])
        offsets = self.root["_offsets"][:]
        contigs = self.root["_contigs"][:]
        self.contig_to_offset = dict(zip(contigs, offsets))

    def read(self, contig: str, start: int, end: int) -> NDArray:
        offset = self.contig_to_offset[contig]
        c_start, c_end = start + offset, end + offset
        return self.root["data"][c_start:c_end]
