from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union

import h5py
import numpy as np
from natsort import index_natsorted
from numpy.typing import NDArray
from pysam import FastaFile

from genome_loader.utils import (
    ALPHABETS,
    DNA_COMPLEMENT,
    PathType,
    bytes_to_ohe,
    rev_comp_byte,
)


class Sequence(ABC):
    ENCODING = Literal["bytes", "onehot"]

    path: Path
    encoding: ENCODING

    @abstractmethod
    def sel(
        self,
        contigs: NDArray[np.str_],
        starts: NDArray[np.integer],
        length: int,
        strands: NDArray[np.str_],
        sorted: bool = False,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Load sequences matching query intervals.

        Query intervals can go beyond the range of contigs and will be padded with 'N' where
        there is no underlying sequence to fetch. In other words, if an interval has a
        negative start, there will be 'N' added to the start of the returned sequence.

        Parameters
        ----------
        contigs : NDArray[np.str_]
            Contig of each interval.
        starts : NDArray[np.integer]
            Start of each interval.
        length : int
            Length of all intervals.
        strands : NDArray[np.str_]
            Strand of each interval, '+' or '-'. Minus strands are reverse complemented.

        Returns
        -------
        seqs : ndarray[bytes or uint8]
            Sequences for each interval. Has shape (intervals length [alphabet])
            where the final dimension is only present if one hot encoding.
        """
        raise NotImplementedError


class FastaSequence(Sequence):
    def __init__(self, fasta: PathType, encoding: Sequence.ENCODING) -> None:
        """Load sequences from a fasta as NumPy arrays.

        Parameters
        ----------
        fasta : str, Path
        encoding : 'bytes' or 'onehot'
            Return sequence arrays as bytes or one hot encoded (uint8).
        """
        self.path = Path(fasta)
        self.encoding = encoding
        self.fasta = FastaFile(str(fasta))
        self._dtype = np.uint8 if encoding == "onehot" else "|S1"

    def sel(self, contigs, starts, length, strands, sorted=False) -> NDArray:
        seqs = np.empty((len(contigs), length), dtype=self._dtype)  # type: ignore

        # go in sorted order to minimize file seeking
        regions = np.char.add(contigs, starts.astype("U"))
        sorter = index_natsorted(regions)
        for i in sorter:
            contig = contigs[i]
            start = starts[i]
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

        rev_comp_idx = np.nonzero(strands == "-")[0]
        if len(rev_comp_idx) > 0:
            seqs[rev_comp_idx] = rev_comp_byte(
                seqs[rev_comp_idx], complement_map=DNA_COMPLEMENT
            )
        if self.encoding == "onehot":
            seqs = bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"])

        return seqs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fasta.close()


class H5Sequence(Sequence):
    def __init__(self, h5: PathType) -> None:
        self.path = Path(h5)

    def _open(self):
        """Only use this as a context manager."""
        return h5py.File(self.path)

    def sel(self, contigs, starts, length, strands, sorted=False) -> NDArray:
        raise NotImplementedError


def ref_loader_factory(ref_path: PathType) -> Sequence:
    _ref_path = Path(ref_path)
    if ".fa" in _ref_path.name:
        return FastaSequence(_ref_path, encoding="bytes")
    elif ".h5" in _ref_path.name:
        return H5Sequence(_ref_path)
    else:
        raise ValueError("File extension for reference is neither FASTA nor HDF5.")
