from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
import pyBigWig
from numpy.typing import ArrayLike, NDArray

from ._ragged import INTERVAL_DTYPE, RaggedIntervals
from ._types import Reader
from ._utils import _get_rel_starts, _lengths_to_offsets, _normalize_contig_name
from .genvarloader import count_intervals as bw_count_intervals
from .genvarloader import intervals as bw_intervals

__all__ = []


class BigWigs(Reader):
    dtype = np.float32  # pyBigWig always returns f32
    chunked = False

    def __init__(self, name: str, paths: Dict[str, str]) -> None:
        """Read data from bigWig files.

        Parameters
        ----------
        name
            Name of the reader, for example `'signal'`.
        paths
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
        name
            Name of the reader, for example `'signal'`.
        table
            Path to a table or a DataFrame containing sample names and paths to bigWig files for those samples.
            It must have columns "sample" and "path".
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
        contig
            Name of the contig/chromosome.
        starts : ArrayLike
            Start coordinates, 0-based.
        ends : ArrayLike
            End coordinates, 0-based, exclusive.
        sample
            Name of the samples to read data from.

        Returns
        -------
            Shape: (samples length). Data corresponding to the given genomic coordinates and samples.
        """
        _contig = _normalize_contig_name(contig, self.contigs)
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
        rel_starts = _get_rel_starts(starts, ends)
        rel_ends = rel_starts + (ends - starts)

        out = np.empty((len(samples), (ends - starts).sum()), dtype=np.float32)
        for s, e, r_s, r_e in zip(starts, ends, rel_starts, rel_ends):
            for i, sample in enumerate(samples):
                out[i, r_s:r_e] = self.readers[sample].values(contig, s, e, numpy=True)

        out = np.nan_to_num(out, copy=False)

        return out

    def count_intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> NDArray[np.int32]:
        """Count the number of intervals corresponding to given genomic coordinates.

        Parameters
        ----------
        contig
            Name of the contig/chromosome.
        starts : ArrayLike
            Start coordinates, 0-based.
        ends : ArrayLike
            End coordinates, 0-based, exclusive.
        sample
            Name of the samples to read data from.

        Returns
        -------
            Shape: (regions, samples). Number of intervals that overlap with each region and sample.
        """
        _contig = _normalize_contig_name(contig, self.contigs)
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

        # (regions, samples)
        n_per_query = bw_count_intervals(paths, contig, starts, ends)

        return n_per_query

    def intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> RaggedIntervals:
        """Read intervals corresponding to given genomic coordinates. The output data
        will be a 2D Ragged array of :code:`struct{start, end, value}` with shape (regions, samples).

        Parameters
        ----------
        contig
            Name of the contig/chromosome.
        starts : ArrayLike
            Start coordinates, 0-based.
        ends : ArrayLike
            End coordinates, 0-based, exclusive.
        sample
            Name of the samples to read data from.
        """
        _contig = _normalize_contig_name(contig, self.contigs)
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

        # (regions, samples)
        n_per_query = bw_count_intervals(paths, contig, starts, ends)
        offsets = _lengths_to_offsets(n_per_query.ravel())

        intervals = self._intervals_from_offsets(contig, starts, ends, offsets, sample)

        return intervals

    def _intervals_from_offsets(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> RaggedIntervals:
        """This function is unsafe! Reads intervals corresponding to given genomic coordinates
        using provided offsets. If the offsets are incorrect this function is undefined behavior.
        To ensure offsets are correct, use the count_intervals function (see :meth:`intervals()`).
        The output data will be a 2D Ragged array of :code:`struct{start, end, value}` with shape (regions, samples).

        Parameters
        ----------
        contig
            Name of the contig/chromosome.
        starts : ArrayLike
            Start coordinates, 0-based.
        ends : ArrayLike
            End coordinates, 0-based, exclusive.
        offsets
            Offsets corresponding to the returned interval data of shape (regions, samples). Can be
            computed from the number of intervals per query, e.g. with the count_intervals function.
        sample
            Name of the samples to read data from.
        """
        _contig = _normalize_contig_name(contig, self.contigs)
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

        # (intervals, 2) (intervals)
        coordinates, values = bw_intervals(paths, contig, starts, ends, offsets)

        coordinates = coordinates.astype(np.int32)

        intervals = np.empty(len(coordinates), dtype=INTERVAL_DTYPE)
        intervals["start"] = coordinates[:, 0]
        intervals["end"] = coordinates[:, 1]
        intervals["value"] = values

        n_regions = len(starts)
        n_samples = len(samples)
        intervals = RaggedIntervals.from_offsets(
            intervals, (n_regions, n_samples), offsets
        )

        return intervals
