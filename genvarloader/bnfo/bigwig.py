from pathlib import Path
from typing import List, Sequence, Union

import dask.array as da
import numpy as np
import pyBigWig
import ray
import xarray as xr
from numpy.typing import NDArray

from .types import Reader


class BigWig(Reader):
    def __init__(
        self,
        name: str,
        paths: Union[Path, List[Path]],
        samples: Union[str, List[str]],
        dtype: Union[str, np.dtype],
        null_value: float = 0.0,
    ) -> None:
        """Read values from bigWig, bigBed, or Wig files. Enable parallel processing
        by initializing a Ray cluster with the appropriate resources. For example, to
        process up to 4 files in parallel, initialize a Ray cluster with 4 cpus by
        calling `ray.init(num_cpus=4)` before using `read()`.


        Parameters
        ----------
        name : str
            Name of the reader, for example `'coverage'`.
        paths : List[Path]
            Paths to the bigWig files.
        samples : List[str]
            Sample names for each bigWig file.
        dtype : Union[str, np.dtype]
            Data type for the output values. BigWig stores all values as float and this
            will be converted to `dtype`, truncating values if necessary.
        null_value : float, optional
            Value for data that is not represented by any interval in the BigWig, by
            default 0.0.
        """
        if isinstance(paths, Path):
            paths = [paths]
        if isinstance(samples, str):
            samples = [samples]

        self.virtual_data = xr.DataArray(
            da.empty(len(samples), dtype=dtype),
            name=name,
            coords={"sample": np.asarray(samples)},
        )
        self.paths = paths
        self.samples = samples
        self.null_value = null_value
        ray.init()

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        samples: Sequence[str]
        samples = kwargs.get("samples", self.samples)

        _, _, sample_idxs = np.intersect1d(
            samples, self.samples, return_indices=True, assume_unique=True
        )
        paths = np.asarray(self.paths)[sample_idxs]

        length = end - start
        out = np.empty((len(samples), length), dtype=self.virtual_data.dtype)
        shared_out = ray.put(out)
        futures = [
            read_one_bigwig.remote(
                self.null_value, i, path, contig, start, end, shared_out
            )
            for i, path in enumerate(paths)
        ]
        ray.get(futures)
        return xr.DataArray(out, dims=["sample", "length"])


@ray.remote(num_cpus=1)
def read_one_bigwig(
    null_value: float,
    sample_idx: int,
    path: Path,
    contig: str,
    start: int,
    end: int,
    out: NDArray,
):
    with pyBigWig.open(str(path)) as bw:
        in_bounds_start = max(0, start)
        in_bounds_end = min(bw.chroms(contig), end)
        vals = bw.values(contig, in_bounds_start, in_bounds_end, numpy=True)
        vals[np.isnan(vals)] = null_value
        relative_start = in_bounds_start - start
        relative_end = in_bounds_end - start
        out[sample_idx, relative_start:relative_end] = vals.astype(out.dtype)
