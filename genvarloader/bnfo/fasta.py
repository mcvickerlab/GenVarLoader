from pathlib import Path
from typing import Union

import numpy as np
import pysam
import xarray as xr

from .types import Reader


class Fasta(Reader):
    def __init__(self, name: str, path: Union[str, Path]) -> None:
        self.name = name
        self.path = path
        self.dtype = np.dtype("S1")
        self.sizes = {}
        self.indexes = {}

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        with pysam.FastaFile(str(self.path)) as f:
            seq = f.fetch(contig, start, end)
            return xr.DataArray(np.frombuffer(seq.encode("ascii"), "S1"), dims="length")
