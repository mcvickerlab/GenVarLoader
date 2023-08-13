from pathlib import Path
from typing import Union

import numpy as np
import pysam
from numpy.typing import NDArray

from .types import Reader


class Fasta(Reader):
    def __init__(self, name: str, path: Union[str, Path]) -> None:
        self.name = name
        self.path = path
        self.dtype = np.dtype("S1")
        self.sizes = {}

    def read(self, contig: str, start: int, end: int, **kwargs) -> NDArray[np.bytes_]:
        with pysam.FastaFile(str(self.path)) as f:
            seq = f.fetch(contig, start, end)
            return np.frombuffer(seq.encode("ascii"), "S1")
