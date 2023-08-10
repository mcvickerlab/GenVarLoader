from pathlib import Path
from typing import List, Union

import joblib
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
        n_jobs: int = 1,
    ) -> None:
        self.name = name
        self.paths = paths
        self.samples = samples
        self.dtype = np.dtype(dtype)
        self.null_value = null_value
        self.bytes_per_length = self.dtype.itemsize * len(samples)
        self.n_jobs = n_jobs

    def read(self, contig: str, start: int, end: int) -> NDArray:
        length = end - start
        out = np.empty((len(self.samples), length), dtype=self.dtype)
        tasks = [
            joblib.delayed(self.read_one_bigwig)(
                self.null_value, i, path, contig, start, end, out
            )
            for i, path in enumerate(self.paths)
        ]
        with joblib.Parallel(n_jobs=self.n_jobs) as parallel:
            parallel(tasks)
        return out

    @staticmethod
    def read_one_bigwig(null_value, sample_idx, path, contig, start, end, out):
        with pyBigWig.open(str(path)) as bw:
            in_bounds_start = max(0, start)
            in_bounds_end = min(bw.chroms()[contig], end)
            vals = bw.values(contig, in_bounds_start, in_bounds_end, numpy=True)
            vals[np.isnan(vals)] = null_value
            relative_start = in_bounds_start - start
            relative_end = in_bounds_end - start
            out[sample_idx, relative_start:relative_end] = vals
