from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pysam import FastaFile

from genome_loader.loaders import Queries
from genome_loader.utils import (
    ALPHABETS,
    DNA_COMPLEMENT,
    PathType,
    bytes_to_ohe,
    rev_comp_byte,
)


class Sequence(ABC):
    class Encoding(Enum):
        BYTES = "bytes"
        ONEHOT = "onehot"

    path: Path

    @abstractmethod
    def sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Load sequences matching query intervals.

        Query intervals can go beyond the range of contigs and will be padded with 'N' where
        there is no underlying sequence to fetch. For example, if an interval has a
        negative start, there will be 'N' added to the start of the returned sequence.

        Parameters
        ----------
        queries: Queries
            Query intervals.
        length : int
            Length of all intervals.
        **kwargs : dict, optional
            sorted : bool, whether the queries are sorted or not
            encoding : 'bytes' or 'onehot', how to encode the sequences

        Returns
        -------
        seqs : ndarray[bytes or uint8]
            Sequences for each interval. Has shape (intervals length [alphabet])
            where the final dimension is only present if one hot encoding.
        """
        raise NotImplementedError


class FastaSequence(Sequence):
    def __init__(self, fasta_path: PathType) -> None:
        """Load sequences from a fasta as NumPy arrays.

        Parameters
        ----------
        fasta_path : str, Path
        """
        self.path = Path(fasta_path)
        self.fasta = FastaFile(str(fasta_path))

    def sel(self, queries: Queries, length: int, **kwargs) -> NDArray:
        sorted = kwargs.get("sorted", False)
        encoding = Sequence.Encoding(kwargs.get("encoding"))
        dtype = np.uint8 if encoding == "onehot" else "|S1"
        seqs = np.empty((len(queries), length), dtype=dtype)  # type: ignore

        # go in sorted order to minimize file seeking
        if not sorted:
            _queries = queries.sort_values(["contig", "start"])
        else:
            _queries = queries
        for tup in _queries.itertuples():
            i = tup.Index
            contig = tup.contig
            start = tup.start
            end = start + length
            prepend = min(start, 0)
            if prepend > 0:
                start = 0
            append = max(end - self.fasta.get_reference_length(contig), 0)
            if append > 0:
                end = self.fasta.get_reference_length(contig)
            seq = np.full_like(seqs[0], b"N")
            seq[prepend : length - append] = np.frombuffer(
                self.fasta.fetch(contig, start, end).encode(), "S1"
            )
            seqs[i] = seq

        rev_comp_idx = np.flatnonzero(queries.strand == "-")
        if len(rev_comp_idx) > 0:
            seqs[rev_comp_idx] = rev_comp_byte(
                seqs[rev_comp_idx], complement_map=DNA_COMPLEMENT
            )
        if encoding is Sequence.Encoding.ONEHOT:
            seqs = bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"])

        return seqs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fasta.close()


class ZarrSequence(Sequence):
    def __init__(self, zarr_path: PathType) -> None:
        self.path = Path(zarr_path)
        self.zarr = xr.open_dataset()

    def sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        raise NotImplementedError


def ref_loader_factory(ref_path: PathType) -> Sequence:
    _ref_path = Path(ref_path)
    if ".fa" in _ref_path.name:
        return FastaSequence(_ref_path)
    elif ".zarr" in _ref_path.name:
        return ZarrSequence(_ref_path)
    else:
        raise ValueError("File extension for reference is neither FASTA nor HDF5.")
