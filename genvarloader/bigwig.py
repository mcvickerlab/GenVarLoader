from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import joblib
import numpy as np
import polars as pl
import pyBigWig
from attrs import define
from numpy.typing import ArrayLike, NDArray

from .types import Reader
from .util import get_rel_starts, normalize_contig_name


class BigWigs(Reader):
    dtype: np.float32  # pyBigWig always returns f32
    chunked = False

    def __init__(self, name: str, paths: Dict[str, str]) -> None:
        """Read data from bigWig files.

        Parameters
        ----------
        name : str
            Name of the reader, for example `'signal'`.
        paths : Dict[str, str]
            Dictionary of sample names and paths to bigWig files for those samples.
        """
        self.name = name
        self.paths = paths
        self.readers = None
        self.samples = list(self.paths)
        self.coords = {"sample": np.asarray(self.samples)}
        self.sizes = {"sample": len(self.samples)}
        contigs: Optional[Dict[str, int]] = None
        for path in self.paths.values():
            with pyBigWig.open(path, "r") as f:
                if contigs is None:
                    contigs = {
                        contig: int(length) for contig, length in f.chroms().items()
                    }
                else:
                    common_contigs = contigs.keys() & f.chroms()
                    contigs = {k: v for k, v in contigs.items() if k in common_contigs}
        if contigs is None:
            raise ValueError("No bigWig files provided.")
        self.contigs: Dict[str, int] = contigs

    @classmethod
    def from_table(cls, name: str, table: Union[str, Path, pl.DataFrame]):
        """Read data from bigWig files.

        Parameters
        ----------
        name : str
            Name of the reader, for example `'signal'`.
        table : Union[str, Path, pl.DataFrame]
            Path to a table or a DataFrame containing sample names and paths to bigWig files for those samples.
        """
        if isinstance(table, (str, Path)):
            table = Path(table)
            if table.suffix == ".csv":
                table = pl.read_csv(table)
            elif table.suffix == ".tsv" or table.suffix == ".txt":
                table = pl.read_tsv(table)
            else:
                raise ValueError("Table should be a csv or tsv file.")
        paths = dict(zip(table["sample"], table["path"]))
        return cls(name, paths)

    def rev_strand_fn(self, x):
        return x[..., ::-1]

    def read(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        **kwargs,
    ) -> NDArray[np.float32]:
        """Read data corresponding to given genomic coordinates. The output shape will
        have length as the final dimension/axis i.e. (..., length).

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : ArrayLike
            Start coordinates, 0-based.
        ends : ArrayLike
            End coordinates, 0-based, exclusive.
        sample : ArrayLike
            Name of the samples to read data from.

        Returns
        -------
        NDArray
            Shape: (samples length). Data corresponding to the given genomic coordinates and samples.
        """
        _contig = normalize_contig_name(contig, self.contigs)
        if _contig is None:
            raise ValueError(f"Contig {contig} not found.")
        else:
            contig = _contig

        if self.readers is None:
            self.readers = {s: pyBigWig.open(p, "r") for s, p in self.paths.items()}

        samples = kwargs.get("sample", self.samples)
        if isinstance(samples, str):
            samples = [samples]
        if not set(samples).issubset(self.samples):
            raise ValueError(f"Sample {samples} not found in bigWig paths.")

        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))
        rel_starts = get_rel_starts(starts, ends)
        rel_ends = rel_starts + (ends - starts)

        out = np.empty((len(samples), (ends - starts).sum()), dtype=np.float32)
        for s, e, r_s, r_e in zip(starts, ends, rel_starts, rel_ends):
            for i, sample in enumerate(samples):
                out[i, r_s:r_e] = self.readers[sample].values(contig, s, e, numpy=True)

        out[np.isnan(out)] = 0

        return out

    def intervals(self, contig: str, starts: ArrayLike, ends: ArrayLike, **kwargs):
        _contig = normalize_contig_name(contig, self.contigs)
        if _contig is None:
            raise ValueError(f"Contig {contig} not found.")
        else:
            contig = _contig

        samples = kwargs.get("sample", self.samples)
        if isinstance(samples, str):
            samples = [samples]
        if not set(samples).issubset(self.samples):
            raise ValueError(f"Sample {samples} not found in bigWig paths.")

        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))
        starts = np.maximum(0, starts)
        ends = np.minimum(ends, self.contigs[contig])

        def task(contig: str, starts: NDArray, ends: NDArray, path: str):
            intervals_ls: List[Tuple[int, int, float]] = []
            # (n_queries)
            n_per_query = np.empty(len(starts), np.uint32)
            with pyBigWig.open(path, "r") as f:
                for i, (s, e) in enumerate(zip(starts, ends)):
                    _intervals = cast(
                        Optional[Tuple[Tuple[int, int, float], ...]],
                        f.intervals(contig, s, e),
                    )
                    if _intervals is not None:
                        intervals_ls.extend(_intervals)
                        n_per_query[i] = len(_intervals)
                    else:
                        n_per_query[i] = 0
            # (n_intervals, 2)
            intervals = np.array([i[:2] for i in intervals_ls], np.uint32)
            # (n_intervals)
            values = np.array([i[2] for i in intervals_ls], np.float32)
            return Intervals(intervals, values, n_per_query)

        with joblib.Parallel(n_jobs=-1) as parallel:
            result = cast(
                List[Intervals],
                parallel(
                    joblib.delayed(task)(contig, starts, ends, self.paths[sample])
                    for sample in samples
                ),
            )

        intervals = np.concatenate([i.intervals for i in result])
        values = np.concatenate([i.values for i in result])
        n_per_query = np.concatenate([i.n_per_query for i in result])
        return Intervals(intervals, values, n_per_query)


@define
class Intervals:
    intervals: NDArray[np.uint32]  # (n_intervals, 2)
    values: NDArray[np.float32]  # (n_intervals)
    n_per_query: NDArray[np.uint32]  # (n_queries)
