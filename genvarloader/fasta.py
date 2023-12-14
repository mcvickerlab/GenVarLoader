from functools import partial
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
import pysam
import seqpro as sp
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from typing_extensions import assert_never

from .types import Reader
from .util import splice_subarrays


class NoPadError(Exception):
    pass


class Fasta(Reader):
    dtype = np.dtype("S1")
    sizes = {}
    coords = {}
    chunked = False

    def __init__(
        self,
        name: str,
        path: Union[str, Path],
        pad: Optional[str] = None,
        alphabet: Optional[Union[str, sp.NucleotideAlphabet, sp.AminoAlphabet]] = None,
        in_memory=False,
    ) -> None:
        """Read sequences from a FASTA file.

        Parameters
        ----------
        name : str
            Name of the reader, for example `'seq'`.
        path : Union[str, Path]
            Path to the FASTA file.
        pad : Optional[str], optional
            A single character which, if passed, will pad out-of-bound ranges with this
            value. By default no padding is done and out-of-bound ranges raise an error.
        alphabet : str, sp.NucleotideAlphabet, sp.AminoAlphabet, optional
            Alphabet to use for the sequences. If not passed, defaults to DNA.

        Raises
        ------
        ValueError
            If pad value is not a single character.
        """
        self.name = name
        self.path = path
        if pad is not None:
            if len(pad) > 1:
                raise ValueError("Pad value must be a single character.")
            self.pad = pad.encode("ascii")
        else:
            self.pad = pad

        with self._open() as f:
            self.contigs = {c: f.get_reference_length(c) for c in f.references}

        if alphabet is None:
            self.alphabet = sp.alphabets.DNA
        elif isinstance(alphabet, str):
            alphabet = alphabet.upper()
            try:
                self.alphabet = cast(
                    Union[sp.NucleotideAlphabet, sp.AminoAlphabet],
                    getattr(sp.alphabets, alphabet),
                )
            except AttributeError:
                raise ValueError(f"Alphabet {alphabet} not found.")
        else:
            self.alphabet = alphabet

        if isinstance(self.alphabet, sp.NucleotideAlphabet):
            self.rev_strand_fn = partial(
                self.alphabet.reverse_complement, length_axis=-1
            )
        elif isinstance(self.alphabet, sp.AminoAlphabet):

            def rev_strand_fn(a: NDArray[np.bytes_]):
                return a[::-1]

            self.rev_strand_fn = rev_strand_fn
        else:
            assert_never(self.alphabet)

        self.contig_starts_with_chr = self.infer_contig_prefix(self.contigs)

        self.handle: Optional[pysam.FastaFile] = None

        if in_memory:
            self.sequences = self._load_all_contigs()
        else:
            self.sequences = None

    def _load_all_contigs(self) -> dict[str, NDArray[np.bytes_]]:
        """Load all contigs into memory."""
        with self._open() as f:
            return {
                c: np.frombuffer(f.fetch(c).encode("ascii"), "S1") for c in f.references
            }

    def _open(self):
        return pysam.FastaFile(str(self.path))

    def close(self):
        if self.handle is not None:
            self.handle.close()
        self.handle = None

    def read(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        out: Optional[NDArray] = None,
        **kwargs,
    ) -> xr.DataArray:
        """Read a sequence from a FASTA file.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[np.int64]
            Start coordinates, 0-based.
        ends : NDArray[np.int64]
            End coordinates, 0-based exclusive.
        out : NDArray, optional
            Array to put the result into. Otherwise allocates one.
        **kwargs
            Not used.

        Returns
        -------
        xr.DataArray
            Sequence from FASTA file.

        Raises
        ------
        ValueError
            Coordinates are out-of-bounds and pad value is not set.
        """
        contig = self.normalize_contig_name(contig)

        starts = np.atleast_1d(np.asarray(starts, dtype=np.int64))
        ends = np.atleast_1d(np.asarray(ends, dtype=np.int64))

        if starts.ndim > 1 or ends.ndim > 1:
            raise ValueError("Starts and ends must be coercible to 1D arrays.")

        if len(starts) != len(ends):
            raise ValueError("Starts and ends must be the same length.")

        start = starts.min()
        end = ends.max()

        pad_left = -min(0, start)
        if pad_left > 0 and self.pad is None:
            raise NoPadError("Padding is disabled and a start is < 0.")

        if self.sequences is None:
            if self.handle is None:
                self.handle = self._open()

            pad_right = max(0, end - self.handle.get_reference_length(contig))
            if pad_right > 0 and self.pad is None:
                raise NoPadError("Padding is disabled and end is > contig length.")

            # pysam behavior
            # start < 0 => error
            # end > contig length => truncate
            seq = self.handle.fetch(contig, max(0, start), end)

            seq = np.frombuffer(seq.encode("ascii"), "S1")
        else:
            # numpy slice behavior
            # start < 0 => wrap around
            # end > contig length => truncate
            pad_right = max(0, end - len(self.sequences[contig]))
            if pad_right > 0 and self.pad is None:
                raise NoPadError("Padding is disabled and end is > contig length.")
            seq = self.sequences[contig][max(0, start) : end]

        padded_seq = []
        if pad_left > 0:
            pad_left = np.full(pad_left, self.pad)
            padded_seq.append(pad_left)
        padded_seq.append(seq)
        if pad_right > 0:
            pad_right = np.full(pad_right, self.pad)
            padded_seq.append(pad_right)
        seq = cast(NDArray[np.bytes_], np.concatenate(padded_seq))

        out = splice_subarrays(seq, starts, ends)

        return xr.DataArray(out, dims="length")
