from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
import pysam
import xarray as xr

from .types import Reader


class Fasta(Reader):
    def __init__(self, name: str, path: Union[str, Path]) -> None:
        self.virtual_data = xr.DataArray(da.empty(0, dtype="S1"), name=name, dims="")
        self.path = path

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        with pysam.FastaFile(str(self.path)) as f:
            # TODO handle start < 0 and end > len(contig)
            # option to make this an error or to pad the sequence
            seq = f.fetch(contig, start, end)
            return xr.DataArray(np.frombuffer(seq.encode("ascii"), "S1"), dims="length")
