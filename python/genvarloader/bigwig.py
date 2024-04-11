from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numba as nb
import numpy as np
import polars as pl
import pyBigWig
from numpy.typing import ArrayLike, NDArray

from .genvarloader import intervals as bw_intervals
from .types import INTERVAL_DTYPE, RaggedIntervals, Reader
from .utils import get_rel_starts, normalize_contig_name


class BigWigs(Reader):
    dtype = np.float32  # pyBigWig always returns f32
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
                table = pl.read_csv(table, separator="\t")
            else:
                raise ValueError("Table should be a csv or tsv file.")
        paths = dict(zip(table["sample"], table["path"]))
        return cls(name, paths)

    def rev_strand_fn(self, x):
        return x[..., ::-1]

    def close(self):
        if self.readers is not None:
            for reader in self.readers.values():
                reader.close()
            self.readers = None

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

        out = np.nan_to_num(out, copy=False)

        return out

    def intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: Optional[Union[str, List[str]]] = None,
        progress: bool = False,
        **kwargs,
    ) -> RaggedIntervals:
        _contig = normalize_contig_name(contig, self.contigs)
        if _contig is None:
            raise ValueError(f"Contig {contig} not found.")
        else:
            contig = _contig

        if sample is None:
            samples = self.samples
        elif isinstance(sample, str):
            samples = [sample]
        else:
            samples = sample

        if not set(samples).issubset(self.samples):
            raise ValueError(f"Sample {samples} not found in bigWig paths.")

        starts = np.atleast_1d(np.asarray(starts, dtype=np.int32))
        ends = np.atleast_1d(np.asarray(ends, dtype=np.int32))
        paths = [self.paths[s] for s in samples]

        coordinates, values, n_per_query = bw_intervals(paths, contig, starts, ends)

        coordinates = coordinates.astype(np.int32)

        intervals = np.empty(len(coordinates), dtype=INTERVAL_DTYPE)
        intervals["start"] = coordinates[:, 0]
        intervals["end"] = coordinates[:, 1]
        intervals["value"] = values

        intervals = RaggedIntervals.from_lengths(intervals, n_per_query)

        return intervals


def task(contig: str, starts: NDArray, ends: NDArray, path: str):
    intervals_ls: List[Tuple[int, int, float]] = []
    # (n_queries)
    n_per_query = np.empty(len(starts), np.int32)
    with pyBigWig.open(path, "r") as f:
        print("opened bw")
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
            if i % 10000 == 0:
                print(i)
        print("read itvs")
    if len(intervals_ls) == 0:
        return RaggedIntervals.empty(len(starts), INTERVAL_DTYPE)  # type: ignore
    # (n_intervals)
    intervals = np.array(intervals_ls, INTERVAL_DTYPE)
    return RaggedIntervals.from_lengths(intervals, n_per_query)


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_tracks(
    start: int,
    end: int,
    intervals: NDArray[np.void],  # structured dtype (start, end, value)
    offsets: NDArray[np.intp],
    n_samples: int,
) -> NDArray[np.float32]:
    """Convert intervals for a single query to tracks at base-pair resolution.

    Parameters
    ----------
    start : int
        Start position of query.
    end : int
        End position of query.
    intervals : NDArray[np.void]
        Shape = (n_intervals) Sorted intervals, each is (start, end, value) as a structured dtype.
    offsets : NDArray[np.uint32]
        Shape = (n_samples + 1) Offsets into intervals.
    shape : int
        Number of samples.

    Returns
    -------
    NDArray[np.float32]
        Shape = (n_samples, length) Tracks.
    """
    length = end - start
    out = np.empty((n_samples, length), np.float32)

    for sample in nb.prange(n_samples):
        o_s, o_e = offsets[sample], offsets[sample + 1]
        if o_e - o_s == 0:
            out[sample] = 0
            continue

        for interval in nb.prange(o_s, o_e):
            out_s = intervals[interval].start - start
            out_e = intervals[interval].end - start
            if out_s > length:
                break
            out[sample, out_s:out_e] = intervals[interval].value
    return out
