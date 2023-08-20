import tempfile
from pathlib import Path
from typing import List, Union

import dask.array as da
import joblib
import numpy as np
import pyBigWig
import xarray as xr
from numpy.typing import NDArray

from .types import Reader


class BigWig(Reader):
    def __init__(
        self,
        name: str,
        paths: Union[str, Path, List[str], List[Path], List[Union[str, Path]]],
        samples: Union[str, List[str]],
        dtype: Union[str, np.dtype],
        null_value: Union[int, float] = 0,
        n_jobs: int = 1,
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
        null_value : int, float, optional
            Value for data that is not represented by any interval in the BigWig, by
            default 0.
        n_jobs : int, optional
            How many files to process in parallel
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if isinstance(samples, str):
            samples = [samples]

        self.virtual_data = xr.DataArray(
            da.empty(len(samples), dtype=dtype),
            name=name,
            coords={"sample": np.asarray(samples)},
        )
        self.paths = np.asarray(list(map(str, paths)))
        self.samples = np.asarray(samples)
        self.null_value = null_value
        self.n_jobs = n_jobs

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        samples = kwargs.get("samples", None)

        if samples is not None:
            _, _, sample_idxs = np.intersect1d(
                samples, self.samples, return_indices=True, assume_unique=True
            )
            paths = self.paths[sample_idxs]
        else:
            samples = self.samples
            paths = self.paths

        # chunk samples into n_cpus chunks, have each process read them sequentially and
        # send the resulting array to the main process so it can concatenate them all
        # together and return
        out_handle = tempfile.NamedTemporaryFile()
        length = end - start
        out = np.memmap(
            out_handle.name,
            shape=(len(samples), length),
            dtype=self.virtual_data.dtype,
            mode="w+",
        )
        out[:] = self.null_value
        with joblib.Parallel(
            n_jobs=self.n_jobs,
            batch_size=-(-len(samples) // self.n_jobs),  # ceil
        ) as parallel:
            parallel(
                joblib.delayed(read_one_bigwig)(
                    self.null_value, i, path, contig, start, end, out
                )
                for i, path in enumerate(paths)
            )
        out_handle.close()
        return xr.DataArray(
            out, dims=["sample", "length"], coords={"sample": np.asarray(samples)}
        )


def read_one_bigwig(
    null_value: float,
    sample_idx: int,
    path: str,
    contig: str,
    start: int,
    end: int,
    out: NDArray,
):
    with pyBigWig.open(path) as bw:
        in_bounds_start = max(0, start)
        in_bounds_end = min(bw.chroms(contig), end)
        vals = bw.values(contig, in_bounds_start, in_bounds_end, numpy=True)
        vals[np.isnan(vals)] = null_value
        relative_start = in_bounds_start - start
        relative_end = in_bounds_end - start
        out[sample_idx, relative_start:relative_end] = vals.astype(out.dtype)
