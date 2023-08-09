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
        self.bytes_per_length = 1

    def read(self, contig: str, start: int, end: int) -> NDArray[np.bytes_]:
        with pysam.FastaFile(str(self.path)) as f:
            seq = f.fetch(contig, start, end)
            return np.frombuffer(seq.encode("ascii"), "S1")
