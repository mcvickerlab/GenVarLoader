from pathlib import Path
from typing import List, Union

import numpy as np
import pyBigWig
from numpy.typing import NDArray

from .types import Reader


class BigWig(Reader):
    def __init__(
        self,
        name: str,
        paths: List[Path],
        samples: List[str],
        dtype: Union[str, np.dtype],
        null_value: int = 0,
    ) -> None:
        self.name = name
        self.paths = paths
        self.samples = samples
        self.dtype = np.dtype(dtype)
        self.null_value = null_value
        self.bytes_per_length = self.dtype.itemsize * len(samples)

    def read(self, contig: str, start: int, end: int) -> NDArray:
        length = end - start
        out = np.empty((len(self.samples), length), dtype=self.dtype)
        for i, path in enumerate(self.paths):
            with pyBigWig.open(str(path)) as bw:
                vals = bw.values(contig, start, end, numpy=True)
                vals[np.isnan(vals)] = self.null_value
                out[i] = vals
        return out
