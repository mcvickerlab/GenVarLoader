from pathlib import Path
from typing import List, Union

import numpy as np
import pyBigWig
import ray
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
        self.sizes = {"sample": len(samples)}
        self.null_value = null_value
        ray.init()

    def read(self, contig: str, start: int, end: int, **kwargs) -> NDArray:
        length = end - start
        out = np.empty((len(self.samples), length), dtype=self.dtype)
        shared_out = ray.put(out)
        futures = [
            read_one_bigwig.remote(
                self.null_value, i, path, contig, start, end, shared_out
            )
            for i, path in enumerate(self.paths)
        ]
        ray.get(futures)
        return out


@ray.remote
def read_one_bigwig(null_value, sample_idx, path, contig, start, end, out):
    with pyBigWig.open(str(path)) as bw:
        in_bounds_start = max(0, start)
        in_bounds_end = min(bw.chroms()[contig], end)
        vals = bw.values(contig, in_bounds_start, in_bounds_end, numpy=True)
        vals[np.isnan(vals)] = null_value
        relative_start = in_bounds_start - start
        relative_end = in_bounds_end - start
        out[sample_idx, relative_start:relative_end] = vals
